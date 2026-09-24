import asyncio
import json

import pytest

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk
from app.memory import recall as recall_module
from app.memory import writer
from app.memory.recall import recall_memories
from app.memory.writer import (
    CONFLICT_PROMPT,
    EXTRACT_PROMPT,
    estimate_importance,
    extract_and_store,
)

DIM = 8
SESSION = "s-mem"


def extraction(*items: tuple[str, str, int | None]) -> str:
    """构造 LLM 的抽取输出（importance 为 None 时省略该字段，走规则兜底）。"""
    payload = []
    for kind, content, importance in items:
        entry: dict = {"kind": kind, "content": content}
        if importance is not None:
            entry["importance"] = importance
        payload.append(entry)
    return json.dumps(payload, ensure_ascii=False)


class MemoryLLM:
    """抽取请求返回脚本里的记忆条目，冲突判定请求按脚本返回矛盾 id 列表。

    两类请求靠 system prompt 区分，因此同一实例可同时服务两条链路。
    """

    def __init__(
        self,
        items: list[tuple[str, str, int | None]] | None = None,
        conflicts: list[list[int]] | None = None,
        raw_extraction: str | None = None,
    ):
        self.items = items or []
        self.raw_extraction = raw_extraction
        self.conflict_script = list(conflicts or [])
        self.chats: list[list] = []
        self.conflict_prompts: list[str] = []

    async def chat(self, messages, tools=None):
        self.chats.append(list(messages))
        if messages[0].content == EXTRACT_PROMPT:
            text = self.raw_extraction
            return ChatResult(text=text if text is not None else extraction(*self.items))
        self.conflict_prompts.append(messages[1].content)
        ids = self.conflict_script.pop(0) if self.conflict_script else []
        return ChatResult(text=json.dumps({"conflicts": ids}))


class StreamLLM:
    """run_agent 用的流式桩：直接吐一段回答，不发工具调用。"""

    def __init__(self, text: str = "好的，记下了"):
        self.text = text
        self.calls: list[list] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        yield StreamChunk(text_delta=self.text)
        yield StreamChunk(finish=True, tool_calls=[])


class GatedMemoryLLM(MemoryLLM):
    """抽取请求卡在 gate 上，用来验证记忆写入不阻塞 SSE 流。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()

    async def chat(self, messages, tools=None):
        if messages[0].content == EXTRACT_PROMPT:
            await self.gate.wait()
        return await super().chat(messages, tools)


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "memory_enabled", True)
    monkeypatch.setattr(settings, "memory_recall_top_k", 5)
    monkeypatch.setattr(settings, "memory_dedup_threshold", 0.85)
    monkeypatch.setattr(settings, "memory_decay", 0.9)
    path = tmp_path / "app.db"
    await init_db(path, DIM)
    yield path
    await runtime.drain_memory_writes()


def use_writer_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(writer, "get_llm", lambda: llm)


async def memories(db, session_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM memories"
    params: tuple = ()
    if session_id is not None:
        sql += " WHERE source = ?"
        params = (session_id,)
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(sql + " ORDER BY id", params)
    return [dict(r) for r in rows]


async def insert_memory(
    db,
    kind: str,
    content: str,
    confidence: float,
    status: str = "active",
    source: str = SESSION,
) -> int:
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO memories "
            "(kind, content, confidence, source, created_at, updated_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, content, confidence, source, "2026-09-24T10:00:00", "2026-09-24T10:00:00", status),
        )
        await conn.commit()
    return cursor.lastrowid


# ---------- 写入侧 ----------


async def test_extract_and_store_writes_structured_memories(db, monkeypatch):
    llm = MemoryLLM(
        [
            ("preference", "用户偏好用 Markdown 记笔记", 4),
            ("goal", "用户想在三个月内拿到大模型实习", 5),
        ]
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记，三个月内想找实习", "好的", db)

    rows = await memories(db, SESSION)
    assert [(r["kind"], r["content"]) for r in rows] == [
        ("preference", "用户偏好用 Markdown 记笔记"),
        ("goal", "用户想在三个月内拿到大模型实习"),
    ]
    assert [r["confidence"] for r in rows] == [0.8, 1.0]
    assert {r["status"] for r in rows} == {"active"}
    assert {r["supersedes"] for r in rows} == {None}
    assert all(r["created_at"] and r["updated_at"] for r in rows)

    # 抽取 prompt 里带上了这轮的用户消息与助手回答
    prompt = llm.chats[0][1].content
    assert "我喜欢用 Markdown 记笔记" in prompt and "好的" in prompt


async def test_extract_and_store_skips_all_work_when_disabled(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_enabled", False)
    llm = MemoryLLM([("fact", "用户在用 Python 3.12", 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我用 Python 3.12", "好的", db)

    assert await memories(db) == []
    assert llm.chats == []


async def test_extract_and_store_without_candidates_writes_nothing(db, monkeypatch):
    llm = MemoryLLM([])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "今天天气不错", "是的", db)

    assert llm.chats  # 抽取发生过，只是没有候选
    assert await memories(db) == []


async def test_invalid_or_fenced_extraction_output(db, monkeypatch):
    llm = MemoryLLM(raw_extraction="（抱歉，我不确定该怎么抽取）")
    use_writer_llm(monkeypatch, llm)
    await extract_and_store(SESSION, "随便说说", "嗯", db)
    assert await memories(db) == []

    fenced = MemoryLLM(
        raw_extraction="```json\n"
        + extraction(("preference", "用户偏好清晨写代码", 4))
        + "\n```"
    )
    use_writer_llm(monkeypatch, fenced)
    await extract_and_store(SESSION, "我喜欢早起写代码", "好的", db)
    assert [r["content"] for r in await memories(db)] == ["用户偏好清晨写代码"]


async def test_unknown_kind_falls_back_to_fact_and_rule_scores_importance(db, monkeypatch):
    llm = MemoryLLM(
        raw_extraction=json.dumps(
            [{"kind": "hobby", "content": "用户养了一只猫叫咪咪"}], ensure_ascii=False
        )
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我养了只猫", "哦", db)

    rows = await memories(db)
    assert rows[0]["kind"] == "fact"
    # 规则兜底：基准 2，无数字/偏好词且短于 50 字
    assert rows[0]["confidence"] == 2 / 5


def test_estimate_importance_rule():
    assert estimate_importance("嗯") == 2
    assert estimate_importance("我喜欢用 vim") == 4  # 偏好 +1、拉丁词 +1
    assert estimate_importance("我需要在 2026 年 3 月前投 20 份简历") == 4  # 数字 +1、偏好 +1
    long_plain = "这是一段只用来说明篇幅的中文记忆内容" * 4  # 80 字，无数字与偏好词
    assert len(long_plain) > 50 and estimate_importance(long_plain) == 3
    assert estimate_importance("我讨厌在 2026 年 3 月前用 vim 写 " + "很长" * 30) == 5


async def test_dedup_keeps_single_memory_and_prefers_higher_confidence(db, monkeypatch):
    llm = MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记", "好的", db)
    # 几乎相同、置信度不更高的候选：直接丢弃，不新增也不改动旧记忆
    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记。", "好的", db)

    rows = await memories(db)
    assert len(rows) == 1
    assert rows[0]["status"] == "active" and rows[0]["confidence"] == 0.6

    # 置信度更高时走追加式版本链：旧记忆 supersede，历史行仍可查
    llm.items = [("preference", "用户偏好用 Markdown 做笔记", 5)]
    await extract_and_store(SESSION, "我强烈偏好用 Markdown 记笔记", "好的", db)

    rows = await memories(db)
    assert len(rows) == 2
    old, new = rows
    assert old["status"] == "superseded" and old["content"] == "用户偏好用 Markdown 记笔记"
    assert new["status"] == "active" and new["supersedes"] == old["id"]
    assert new["confidence"] == 1.0


async def test_conflict_marks_new_memory_and_decays_old(db, monkeypatch):
    old_id = await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM([("preference", "用户讨厌用 Markdown 记笔记", 4)], conflicts=[[old_id]])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我现在很讨厌 Markdown", "好的", db)

    rows = await memories(db)
    assert len(rows) == 2
    assert rows[0]["status"] == "active"
    assert rows[0]["confidence"] == pytest.approx(0.72)  # 0.8 × 0.9
    # 新记忆挂冲突态，不覆盖旧记忆
    assert rows[1]["status"] == "conflict"
    assert rows[1]["content"] == "用户讨厌用 Markdown 记笔记"

    # 冲突判定交给 LLM：prompt 里带上了新记忆与候选旧记忆
    prompt = llm.conflict_prompts[0]
    assert "用户讨厌用 Markdown 记笔记" in prompt
    assert f"{old_id}. 用户喜欢用 Markdown 记笔记" in prompt


async def test_conflict_ignores_unknown_ids_and_bad_payload(db, monkeypatch):
    await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM(
        [("preference", "用户讨厌 Markdown 里的表格语法", 4)], conflicts=[[999]]
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我讨厌 Markdown 的表格", "好的", db)

    rows = await memories(db)
    # 幻觉 id 被忽略，于是按普通新记忆入库，旧记忆不衰减
    assert [r["status"] for r in rows] == ["active", "active"]
    assert rows[0]["confidence"] == 0.8


async def test_conflict_detection_only_compares_same_kind(db, monkeypatch):
    await insert_memory(db, "goal", "用户想在三个月内拿到大模型实习", 0.8)
    llm = MemoryLLM([("preference", "用户讨厌用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我讨厌 Markdown 的表格", "好的", db)

    # 不同 kind 不进冲突判定窗口：没有旧记忆可比时不做 LLM 冲突判定
    assert llm.conflict_prompts == []
    assert [r["status"] for r in await memories(db)] == ["active", "active"]


async def test_write_failure_is_swallowed(db, monkeypatch):
    def boom():
        raise RuntimeError("没有配置 API key")

    monkeypatch.setattr(writer, "get_llm", boom)

    await extract_and_store(SESSION, "我喜欢用 Markdown", "好的", db)

    assert await memories(db) == []


# ---------- 召回侧 ----------


async def test_recall_formats_top_k_by_confidence(db):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.6)
    await insert_memory(db, "fact", "用户在准备大模型实习面试", 0.9)
    await insert_memory(db, "goal", "用户想在三个月内写完项目", 0.4)
    await insert_memory(db, "fact", "用户养了一只猫", 0.3, status="conflict")
    await insert_memory(db, "fact", "用户在北京", 0.2, status="superseded")

    text = await recall_memories("我该怎么记笔记？", db)

    assert text == (
        "以下是关于用户的一些长期记忆，供参考：\n"
        "- [fact] 用户在准备大模型实习面试\n"
        "- [preference] 用户偏好用 Markdown 记笔记\n"
        "- [goal] 用户想在三个月内写完项目"
    )


async def test_recall_respects_top_k(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_recall_top_k", 2)
    await insert_memory(db, "fact", "用户在准备面试", 0.9)
    await insert_memory(db, "fact", "用户在北京", 0.8)
    await insert_memory(db, "fact", "用户养了一只猫", 0.7)

    text = await recall_memories("hi", db)

    assert "用户在准备面试" in text and "用户在北京" in text
    assert "猫" not in text


async def test_recall_returns_none_without_memories(db):
    assert await recall_memories("hi", db) is None


async def test_recall_returns_none_when_disabled(db, monkeypatch):
    await insert_memory(db, "fact", "用户在准备面试", 0.9)
    monkeypatch.setattr(settings, "memory_enabled", False)

    assert await recall_memories("hi", db) is None


async def test_recall_degrades_on_db_error(db, monkeypatch):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def boom(db_path=None):
        raise RuntimeError("库挂了")
        yield  # pragma: no cover

    monkeypatch.setattr(recall_module, "get_db", boom)

    assert await recall_memories("hi", db) is None


# ---------- 接入点 ----------


def test_assemble_messages_inserts_memory_after_system_prompt():
    history = [runtime.Message(role="user", content="旧消息")]
    memory = "以下是关于用户的一些长期记忆，供参考：\n- [fact] 用户在北京"

    messages = runtime.assemble_messages(history, "新问题", memory)

    assert [m.role for m in messages] == ["system", "system", "user", "user"]
    assert messages[0].content == runtime.SYSTEM_PROMPT
    assert messages[1].content == memory
    assert messages[-1].content == "新问题"


def test_assemble_messages_without_memory_is_unchanged():
    messages = runtime.assemble_messages([], "新问题", None)
    assert [m.role for m in messages] == ["system", "user"]


async def test_run_agent_injects_recalled_memory_into_prompt(db, monkeypatch):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.9)
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([]))

    await collect_events(SESSION)

    sent = stream.calls[0]
    assert [m.role for m in sent] == ["system", "system", "user"]
    assert sent[1].content.startswith("以下是关于用户的一些长期记忆")
    assert "用户偏好用 Markdown 记笔记" in sent[1].content


async def test_run_agent_skips_memory_message_when_disabled(db, monkeypatch):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.9)
    monkeypatch.setattr(settings, "memory_enabled", False)
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([]))

    await collect_events(SESSION)

    assert [m.role for m in stream.calls[0]] == ["system", "user"]


def collect(session_id: str = SESSION, message: str = "我喜欢用 Markdown 记笔记"):
    return runtime.run_agent(session_id, message, None)


async def collect_events(
    session_id: str = SESSION, message: str = "我喜欢用 Markdown 记笔记"
) -> list[runtime.AgentEvent]:
    return [e async for e in collect(session_id, message)]


async def test_memory_write_is_fire_and_forget(db, monkeypatch):
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    gated = GatedMemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, gated)

    events = [e async for e in collect(SESSION)]

    # done 已经到达，而抽取还卡在 gate 上：SSE 流没有被记忆写入阻塞
    assert events[-1].type == "done"
    assert await memories(db) == []

    gated.gate.set()
    await runtime.drain_memory_writes()

    rows = await memories(db, SESSION)
    assert [r["content"] for r in rows] == ["用户偏好用 Markdown 记笔记"]
    assert rows[0]["source"] == SESSION


async def test_memory_write_failure_does_not_break_stream(db, monkeypatch):
    def boom():
        raise RuntimeError("LLM 不可用")

    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    monkeypatch.setattr(writer, "get_llm", boom)

    events = [e async for e in collect("s-boom")]
    await runtime.drain_memory_writes()

    assert events[-1].type == "done"
    assert await memories(db) == []


async def test_interrupted_answer_does_not_spawn_memory_write(db, monkeypatch):
    calls: list[str] = []

    def recorder():
        calls.append("get_llm")
        return MemoryLLM([])

    monkeypatch.setattr(runtime, "get_llm", lambda: HangingLLM())
    monkeypatch.setattr(writer, "get_llm", recorder)

    stream = collect(SESSION)
    while (await anext(stream)).type != "text_delta":
        pass
    await stream.aclose()
    await runtime.drain_memory_writes()

    # 半截回答不抽取记忆：写入侧根本没被唤起
    assert calls == []
    assert await memories(db) == []


class HangingLLM:
    """吐一个 text_delta 后挂住，供中断路径使用。"""

    def __init__(self) -> None:
        self.released = asyncio.Event()

    async def chat_stream(self, messages, tools=None):
        yield StreamChunk(text_delta="半截回答")
        await self.released.wait()


# ---------- 端到端（HTTP 层） ----------


async def test_chat_end_to_end_learns_and_recalls_memory(db, monkeypatch):
    """走真实 HTTP 链路：一轮对话后记忆落库，下一轮该记忆被注入提示词。"""
    import httpx

    from app.main import app

    stream = StreamLLM("好的，我会用 Markdown 给你记笔记")
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"session_id": SESSION, "message": "我喜欢用 Markdown"})
        assert resp.status_code == 200
        await runtime.drain_memory_writes()

        rows = await memories(db, SESSION)
        assert [r["content"] for r in rows] == ["用户偏好用 Markdown 记笔记"]

        # 第二轮：召回的记忆作为 system 消息进入提示词
        await client.post("/api/chat", json={"session_id": SESSION, "message": "继续"})

    sent = stream.calls[-1]
    assert sent[1].content.startswith("以下是关于用户的一些长期记忆")
    assert "用户偏好用 Markdown 记笔记" in sent[1].content
