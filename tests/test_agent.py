import asyncio
import json

import httpx
import pytest

from app.agent import runtime
from app.db import get_db, init_db
from app.llm.types import StreamChunk, ToolCall
from app.main import app
from app.memory import writer as memory_writer
from app.retrieval.bm25_search import invalidate
from app.retrieval.types import RetrievedChunk

DIM = 8
SESSION = "s1"


def chunk(text: str) -> StreamChunk:
    return StreamChunk(text_delta=text)


def final(calls: list[ToolCall] | None = None) -> StreamChunk:
    return StreamChunk(finish=True, tool_calls=calls or [])


class FakeLLM:
    """按脚本逐轮返回 chunk；记录每次收到的 messages 供断言。"""

    def __init__(self, rounds: list[list[StreamChunk]]):
        self.rounds = rounds
        self.calls: list[list] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        for c in self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]:
            yield c


class BoomLLM:
    async def chat_stream(self, messages, tools=None):
        raise RuntimeError("llm 挂了")
        yield  # pragma: no cover —— 让它成为异步生成器


class _SilentWriterLLM:
    """记忆抽取桩：T5 的用例只管 Agent 循环，不产生真实的抽取调用。"""

    async def chat(self, messages, tools=None):
        from app.llm.types import ChatResult

        return ChatResult(text="[]")


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """tmp 数据库 + 隔离的 BM25 索引缓存 + 默认 db_path 指向 tmp。

    run_agent 不建表（生产由 main.py 的 lifespan 负责），所以这里显式 init_db。
    记忆写入是 fire-and-forget，这里换成空抽取桩并在收尾 drain，避免测试
    真的去调外部 LLM（配置了 API key 时会发真实请求）。
    """
    monkeypatch.setattr(runtime.settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    invalidate()
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()
    invalidate()


def use_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)


def fake_chunks(monkeypatch, chunks: list[RetrievedChunk]) -> list[str]:
    """替换 hybrid_search，记录收到的 query 并返回固定结果。"""
    queries: list[str] = []

    async def fake_hybrid_search(query, k=8, mode="hybrid", db_path=None):
        queries.append(query)
        return chunks

    monkeypatch.setattr(runtime, "hybrid_search", fake_hybrid_search)
    return queries


def tool_round(query: str = "RAG") -> list[StreamChunk]:
    return [
        final([ToolCall(id="c1", name="search_knowledge", arguments={"query": query})])
    ]


async def collect(session_id: str = SESSION) -> list[runtime.AgentEvent]:
    return [e async for e in runtime.run_agent(session_id, "什么是 RAG？")]


async def messages_of(session_id: str, db) -> list[dict]:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    return [dict(r) for r in rows]


async def test_tool_call_then_answer(db, monkeypatch):
    hit = RetrievedChunk(
        chunk_id=7, doc_id=1, content="RAG 结合检索与生成", title="检索笔记", score=0.5
    )
    queries = fake_chunks(monkeypatch, [hit])
    llm = FakeLLM([tool_round("RAG"), [chunk("根据笔记"), chunk("……"), final()]])
    use_llm(monkeypatch, llm)

    events = await collect()
    types = [e.type for e in events]

    assert types == [
        "tool_start",
        "tool_end",
        "text_delta",
        "text_delta",
        "done",
    ]
    assert queries == ["RAG"]
    assert events[0].data["name"] == "search_knowledge"
    assert events[0].data["arguments"] == {"query": "RAG"}

    # 工具结果（含标题与 chunk_id）作为 tool 消息进入了第二轮的 messages
    second = llm.calls[1]
    tool_msgs = [m for m in second if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "检索笔记" in tool_msgs[0].content and "chunk 7" in tool_msgs[0].content

    # assistant 消息带 tool_calls，且排在 tool 结果之前
    assistant = [m for m in second if m.role == "assistant"][-1]
    assert assistant.tool_calls[0].name == "search_knowledge"
    assert second.index(assistant) < second.index(tool_msgs[0])
    assert second[0].role == "system" and second[1].role == "user"

    assert events[-1].data["text"] == "根据笔记……"
    assert events[-1].data["session_id"] == SESSION

    saved = await messages_of(SESSION, db)
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[0]["content"] == "什么是 RAG？"
    # 落库的助手消息带上工具调用标记，前端历史里能看到「查过什么」
    assert saved[1]["content"].startswith("根据笔记……")
    assert "search_knowledge" in saved[1]["content"]


async def test_direct_answer_without_tool(db, monkeypatch):
    llm = FakeLLM([[chunk("直接回答"), final()]])
    use_llm(monkeypatch, llm)
    queries = fake_chunks(monkeypatch, [])

    events = await collect()

    assert [e.type for e in events] == ["text_delta", "done"]
    assert queries == []
    assert len(llm.calls) == 1
    assert [m["role"] for m in await messages_of(SESSION, db)] == ["user", "assistant"]


async def test_tool_result_is_truncated(db, monkeypatch):
    hit = RetrievedChunk(
        chunk_id=1, doc_id=1, content="甲" * 5000, title="长文", score=1.0
    )
    fake_chunks(monkeypatch, [hit])
    llm = FakeLLM([tool_round(), [chunk("好"), final()]])
    use_llm(monkeypatch, llm)

    await collect()

    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert len(tool_msg.content) < 5000
    assert "截断" in tool_msg.content


async def test_tool_call_round_limit_stops_with_error(db, monkeypatch):
    llm = FakeLLM([tool_round()])  # 每一轮都请求工具
    use_llm(monkeypatch, llm)
    fake_chunks(monkeypatch, [])

    events = await collect()
    types = [e.type for e in events]

    assert types.count("tool_start") == runtime.MAX_TOOL_ROUNDS
    assert types.count("tool_end") == runtime.MAX_TOOL_ROUNDS
    assert len(llm.calls) == runtime.MAX_TOOL_ROUNDS
    assert types[-1] == "error"
    assert "上限" in events[-1].data["message"]
    # 已发生的对话仍要落库，用户下一轮能看到上文的提问
    assert [m["role"] for m in await messages_of(SESSION, db)] == ["user", "assistant"]


async def test_llm_exception_yields_error_and_persists(db, monkeypatch):
    use_llm(monkeypatch, BoomLLM())
    fake_chunks(monkeypatch, [])

    events = await collect()

    assert [e.type for e in events] == ["error"]
    assert "llm 挂了" in events[-1].data["message"]
    assert [m["role"] for m in await messages_of(SESSION, db)] == ["user", "assistant"]


async def test_tool_failure_degrades_to_message(db, monkeypatch):
    async def boom(query, k=8, mode="hybrid", db_path=None):
        raise RuntimeError("检索崩了")

    monkeypatch.setattr(runtime, "hybrid_search", boom)
    llm = FakeLLM([tool_round(), [chunk("没查到"), final()]])
    use_llm(monkeypatch, llm)

    events = await collect()

    assert [e.type for e in events] == ["tool_start", "tool_end", "text_delta", "done"]
    assert "失败" in events[1].data["summary"]
    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert "检索崩了" in tool_msg.content


async def test_history_is_loaded_into_prompt(db, monkeypatch):
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (SESSION, "2026-09-24T10:00:00"),
        )
        for i in range(25):
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (SESSION, "user" if i % 2 == 0 else "assistant", f"旧消息{i}", "2026-09-24T10:00:00"),
            )
        await conn.commit()

    llm = FakeLLM([[chunk("好"), final()]])
    use_llm(monkeypatch, llm)

    await collect()

    sent = llm.calls[0]
    assert sent[0].role == "system" and "知识库" in sent[0].content
    # 只取最近 20 条，且丢掉更早的
    assert len(sent) == 1 + runtime.HISTORY_LIMIT + 1
    assert "旧消息0" not in [m.content for m in sent]
    assert "旧消息24" in [m.content for m in sent]


async def test_unknown_session_is_created(db, monkeypatch):
    llm = FakeLLM([[chunk("好"), final()]])
    use_llm(monkeypatch, llm)

    await collect("brand-new")

    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id FROM sessions WHERE id = ?", ("brand-new",)
        )
    assert len(rows) == 1


async def test_assemble_messages_is_the_assembly_hook(db, monkeypatch):
    # T7/T8/T11 的治理接入点：替换 assemble_messages 应当影响发给模型的消息
    seen: list[list] = []

    def spy(history, user_message, memory=None, skill_prompt=None):
        # 该用例的 memories 表是空的，召回无结果；记忆注入在 tests/test_memory.py 覆盖。
        # 这条提问不含任何 skill 触发词，所以 skill_prompt 也是 None（触发注入见
        # tests/test_skills.py）
        assert memory is None and skill_prompt is None
        msgs = runtime.Message(
            role="system", content="被治理过的 system"
        )
        seen.append([msgs, *history, runtime.Message(role="user", content=user_message)])
        return seen[-1]

    monkeypatch.setattr(runtime, "assemble_messages", spy)
    llm = FakeLLM([[chunk("好"), final()]])
    use_llm(monkeypatch, llm)
    fake_chunks(monkeypatch, [])

    await collect()

    assert seen and seen[0][0].content == "被治理过的 system"
    assert llm.calls[0][0].content == "被治理过的 system"


class HangingLLM:
    """按脚本返回：第一轮请求工具，第二轮吐一个 text_delta 后挂住不返回。

    `hang_on_round` 指定挂住发生在第几轮（1 起算），迟到调用不再挂。
    """

    def __init__(self, hang_on_round: int = 2) -> None:
        self.hang_on_round = hang_on_round
        self.released = asyncio.Event()
        self.round = 0

    async def chat_stream(self, messages, tools=None):
        self.round += 1
        if self.round < self.hang_on_round:
            yield final(
                [ToolCall(id="c1", name="search_knowledge", arguments={"query": "RAG"})]
            )
            return
        if self.round == self.hang_on_round:
            yield chunk("前半段回答")
            await self.released.wait()
            return
        yield chunk("根据笔记……")
        yield final()


async def test_client_disconnect_persists_user_message(db, monkeypatch):
    use_llm(monkeypatch, HangingLLM())
    fake_chunks(monkeypatch, [])

    stream = runtime.run_agent(SESSION, "什么是 RAG？")
    seen = [await anext(stream)]  # tool_start
    while seen[-1].type != "text_delta":  # 走到第二轮生成的第一个增量
        seen.append(await anext(stream))
    assert [e.type for e in seen] == ["tool_start", "tool_end", "text_delta"]

    # 客户端关页面 = 生成器被 aclose()，内部抛 GeneratorExit
    await stream.aclose()

    saved = await messages_of(SESSION, db)
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[0]["content"] == "什么是 RAG？"
    # 已生成的部分回答保留，并标记未完成
    assert saved[1]["content"].startswith("前半段回答")
    assert "未完成" in saved[1]["content"]


async def test_cancelled_task_still_persists_user_message(db, monkeypatch):
    """真实 HTTP 断开走的是取消任务（CancelledError），不是 aclose：
    清理阶段的 await 会被打断，但用户提问必须已经落库。"""
    use_llm(monkeypatch, HangingLLM())
    fake_chunks(monkeypatch, [])

    async def consume():
        async for _ in runtime.run_agent(SESSION, "被取消的提问"):
            pass  # 流会挂住，直到任务被取消

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)  # 跑到 stub 的挂起点
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    saved = await messages_of(SESSION, db)
    assert saved[0]["role"] == "user"
    assert saved[0]["content"] == "被取消的提问"


async def test_disconnect_keeps_partial_answer_in_next_history(db, monkeypatch):
    """断开的半截回答进历史（带未完成标记），且只落一次、不影响下一轮提问。"""
    use_llm(monkeypatch, HangingLLM())
    fake_chunks(monkeypatch, [])

    stream = runtime.run_agent(SESSION, "第一问")
    while (await anext(stream)).type != "text_delta":
        pass
    await stream.aclose()

    # 同一 session 再问一轮：断开那轮只留 2 行，历史里带着未完成标记
    use_llm(monkeypatch, FakeLLM([[chunk("第二次回答"), final()]]))
    await collect(SESSION)

    saved = await messages_of(SESSION, db)
    assert [m["role"] for m in saved] == ["user", "assistant", "user", "assistant"]
    assert saved[0]["content"] == "第一问"
    assert "前半段回答" in saved[1]["content"] and "未完成" in saved[1]["content"]
    assert saved[3]["content"] == "第二次回答"


async def test_answer_is_persisted_before_done_event(db, monkeypatch):
    """done 一到客户端就会关掉 SSE、任务随即被取消。助手消息必须在 done 之前
    落库，否则正常成功路径下 assistant 行会永久丢失（复审实测 0/8）。"""
    use_llm(monkeypatch, FakeLLM([[chunk("答案"), final()]]))
    fake_chunks(monkeypatch, [])

    stream = runtime.run_agent(SESSION, "问题")
    async for event in stream:
        if event.type == "done":
            # 还没关流：done 之前就应该已经落库，且是完整回答（无未完成标记）
            saved = await messages_of(SESSION, db)
            assert [m["role"] for m in saved] == ["user", "assistant"]
            assert saved[1]["content"] == "答案"
            break
    await stream.aclose()

    # 关流后不会重复落库
    assert [m["role"] for m in await messages_of(SESSION, db)] == ["user", "assistant"]


async def test_get_llm_failure_yields_error_event(db, monkeypatch):
    """get_llm() 自己抛（比如没配 API key）也必须发 error 事件，不能逃出生成器。"""

    def boom():
        raise RuntimeError("没有配置 API key")

    monkeypatch.setattr(runtime, "get_llm", boom)
    fake_chunks(monkeypatch, [])

    events = await collect()

    assert [e.type for e in events] == ["error"]
    assert "没有配置 API key" in events[0].data["message"]
    # 提问仍然落库
    assert (await messages_of(SESSION, db))[0]["content"] == "什么是 RAG？"


class PartialThenBoomLLM:
    """吐半个回答后抛异常。"""

    async def chat_stream(self, messages, tools=None):
        yield chunk("半截")
        raise RuntimeError("llm 挂了")


async def test_no_incomplete_marker_on_round_limit(db, monkeypatch):
    """「未完成」标记只属于中断路径，不能误加到工具轮数上限。"""
    use_llm(monkeypatch, FakeLLM([[chunk("半截回答"), *tool_round()]]))
    fake_chunks(monkeypatch, [])

    events = await collect()
    assert events[-1].type == "error"

    saved = await messages_of(SESSION, db)
    assert "半截回答" in saved[1]["content"]
    assert "未完成" not in saved[1]["content"]


async def test_no_incomplete_marker_on_llm_exception(db, monkeypatch):
    """LLM 异常路径同理：回答由 error 事件交代，历史里不打「未完成」标记。"""
    use_llm(monkeypatch, PartialThenBoomLLM())
    fake_chunks(monkeypatch, [])

    events = await collect()
    assert [e.type for e in events] == ["text_delta", "error"]

    saved = await messages_of(SESSION, db)
    assert "半截" in saved[1]["content"]
    assert "未完成" not in saved[1]["content"]


# ---------- API 层 ----------


@pytest.fixture
async def client(db, monkeypatch):
    """db fixture 已把 runtime.settings.db_path 指向 tmp；main.py 各接口均走默认路径。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


async def test_chat_streams_sse_with_new_session(client, monkeypatch):
    llm = FakeLLM([tool_round(), [chunk("根据笔记"), final()]])
    use_llm(monkeypatch, llm)
    fake_chunks(
        monkeypatch,
        [RetrievedChunk(chunk_id=3, doc_id=1, content="RAG 内容", title="笔记", score=1.0)],
    )

    resp = await client.post("/api/chat", json={"session_id": None, "message": "什么是 RAG？"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "session"
    session_id = events[0]["data"]["session_id"]
    assert session_id
    assert "tool_start" in types and "tool_end" in types and "text_delta" in types
    assert types[-1] == "done"
    assert events[-1]["data"]["session_id"] == session_id

    # 同一 session 续聊：历史被带进第二轮
    resp2 = await client.post(
        "/api/chat", json={"session_id": session_id, "message": "再展开说说"}
    )
    events2 = parse_sse(resp2.text)
    assert "session" not in [e["type"] for e in events2]
    assert llm.calls[-1][-1].content == "再展开说说"
    assert any(m.content == "什么是 RAG？" for m in llm.calls[-1])

    history = await client.get(f"/api/sessions/{session_id}/messages")
    assert history.status_code == 200
    rows = history.json()
    assert [r["role"] for r in rows] == ["user", "assistant", "user", "assistant"]
    assert rows[0]["content"] == "什么是 RAG？"


async def test_chat_reports_llm_error_as_event(client, monkeypatch):
    use_llm(monkeypatch, BoomLLM())
    fake_chunks(monkeypatch, [])

    resp = await client.post("/api/chat", json={"session_id": "s9", "message": "hi"})
    events = parse_sse(resp.text)

    assert [e["type"] for e in events] == ["error"]
    assert resp.status_code == 200


async def test_chat_survives_run_agent_exception(client, monkeypatch):
    """run_agent 自己抛（比如装配阶段就崩、写库失败），HTTP 码已经发出去了，
    必须补一个 error 事件，否则前端拿到 200 之后就永远等不到 done。"""

    async def boom(session_id, message, db_path=None):
        yield runtime.AgentEvent("text_delta", {"text": "开头"})
        raise RuntimeError("装配阶段崩了")

    monkeypatch.setattr("app.main.run_agent", boom)

    resp = await client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
    events = parse_sse(resp.text)

    assert resp.status_code == 200
    assert [e["type"] for e in events] == ["text_delta", "error"]
    assert "装配阶段崩了" in events[-1]["data"]["message"]


async def test_chat_session_event_survives_ensure_session_failure(client, monkeypatch):
    """发 session 事件时就失败：仍然要给一个 error 事件，而不是空响应体。"""

    async def boom(session_id, db_path=None):
        raise RuntimeError("建会话失败")

    monkeypatch.setattr("app.main.ensure_session", boom)

    resp = await client.post("/api/chat", json={"session_id": None, "message": "hi"})
    events = parse_sse(resp.text)

    assert resp.status_code == 200
    assert [e["type"] for e in events] == ["error"]
    assert "建会话失败" in events[-1]["data"]["message"]


async def test_ingest_and_documents_endpoints(client, monkeypatch):
    calls: list[str] = []

    async def fake_ingest(source):
        calls.append(source)
        return 3

    async def fake_list():
        return [{"id": 1, "source": "notes/a.md", "title": "笔记", "chunk_count": 3}]

    async def fake_delete(doc_id):
        calls.append(f"delete:{doc_id}")

    monkeypatch.setattr("app.main.ingest", fake_ingest)
    monkeypatch.setattr("app.main.list_documents", fake_list)
    monkeypatch.setattr("app.main.delete_document", fake_delete)

    resp = await client.post("/api/ingest", json={"source": "notes/a.md"})
    assert resp.status_code == 200 and resp.json() == {"chunks": 3}

    docs = await client.get("/api/documents")
    assert docs.json()[0]["title"] == "笔记"

    deleted = await client.delete("/api/documents/1")
    assert deleted.json() == {"deleted": 1}
    assert calls == ["notes/a.md", "delete:1"]


async def test_ingest_surfaces_loader_error(client, monkeypatch):
    async def failing_ingest(source):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr("app.main.ingest", failing_ingest)

    resp = await client.post("/api/ingest", json={"source": "nope.md"})
    assert resp.status_code == 400
    assert "FileNotFoundError" in resp.json()["detail"]