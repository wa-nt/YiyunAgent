"""T1 模式框架的用例：会话模式持久化、API 校验、模式 prompt 与工具白名单。

隔离方式与 tests/test_api.py 一致：db 夹具把 settings.db_path 指到 tmp 并 init_db
（ASGITransport 不跑 lifespan，建表的责任在夹具），LLM 一律换脚本桩，所以不碰
data/app.db、也不发真实请求。记忆抽取走 memory_writer 自己的桩（同 test_agent.py）。
"""

from __future__ import annotations

import json
import logging

import aiosqlite
import httpx
import pytest

from app.agent import runtime
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk, ToolCall, ToolDef
from app.main import app
from app.memory import writer as memory_writer

DIM = 8
SESSION = "s1"
WORK_SESSION = "w1"


class FakeLLM:
    """记录每轮收到的 messages / tools，返回一段固定回答。"""

    def __init__(self, text: str = "好的"):
        self.text = text
        self.calls: list[list] = []
        self.tools: list[list | None] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        self.tools.append(list(tools) if tools is not None else None)
        yield StreamChunk(text_delta=self.text)
        yield StreamChunk(finish=True, tool_calls=[])

    async def chat(self, messages, tools=None):  # _write_title 用得到
        return ChatResult(text="标题")


class ToolCallingLLM:
    """第一轮要求调用指定工具，第二轮正常收尾。"""

    def __init__(self, call: ToolCall):
        self.call = call
        self.rounds = 0
        self.tools: list[list | None] = []

    async def chat_stream(self, messages, tools=None):
        self.tools.append(list(tools) if tools is not None else None)
        self.rounds += 1
        if self.rounds == 1:
            yield StreamChunk(finish=True, tool_calls=[self.call])
            return
        yield StreamChunk(text_delta="收到")
        yield StreamChunk(finish=True, tool_calls=[])

    async def chat(self, messages, tools=None):
        return ChatResult(text="标题")


class _SilentWriterLLM:
    """记忆抽取桩：这些用例只管模式框架，不做真实抽取调用。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    # 默认给个不发请求的桩：用例要断言 prompt/tools 时再用 use_llm 换成自己的实例
    monkeypatch.setattr(runtime, "get_llm", lambda: FakeLLM())
    # 这些用例断言的是 system 消息里的**模式** prompt：人格另由 tests/test_persona.py
    # 覆盖，这里钉成「用户明确不要人格」，免得默认人格原文混进模式断言
    monkeypatch.setattr(runtime, "load_persona", _no_persona)
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()


async def _no_persona(db_path=None):
    return ""


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def use_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


async def seed_session(db, session_id: str, mode: str | None, messages=()) -> None:
    """直接种一条会话（mode=None 模拟迁移前的存量行），消息串成一条线性链。"""
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO sessions (id, created_at, mode) VALUES (?, ?, ?)",
            (session_id, "2026-10-04T09:00:00", mode),
        )
        leaf = None
        for role, content in messages:
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, parent_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, "2026-10-04T09:00:00", leaf),
            )
            leaf = cursor.lastrowid
        if leaf is not None:
            await conn.execute(
                "UPDATE sessions SET active_leaf = ? WHERE id = ?", (leaf, session_id)
            )
        await conn.commit()


async def session_row(db, session_id: str) -> dict:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, mode, source FROM sessions WHERE id = ?", (session_id,)
        )
    assert rows, f"会话 {session_id} 不存在"
    return dict(rows[0])


# ---------- 新建会话：mode 持久化 + SSE ----------


async def test_new_work_session_persists_mode(client, db, monkeypatch):
    llm = FakeLLM()
    use_llm(monkeypatch, llm)

    resp = await client.post(
        "/api/chat",
        json={"session_id": None, "message": "帮我拆一下下周的学习计划", "mode": "work"},
    )

    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert events[0]["type"] == "session"
    assert events[0]["data"]["mode"] == "work"
    session_id = events[0]["data"]["session_id"]

    row = await session_row(db, session_id)
    assert row["mode"] == "work"
    assert row["source"] == "manual"

    # 这一轮真的按 work 组装了 prompt 与工具
    system = llm.calls[0][0].content
    assert system == runtime.SYSTEM_PROMPT + runtime.MODE_PROMPTS["work"]
    assert "search_knowledge" in [t.name for t in llm.tools[0]]


async def test_new_session_mode_defaults_to_chat(client, db):
    resp = await client.post("/api/chat", json={"session_id": None, "message": "你好"})

    events = parse_sse(resp.text)
    assert events[0]["data"]["mode"] == "chat"
    assert (await session_row(db, events[0]["data"]["session_id"]))["mode"] == "chat"


# ---------- 已有会话：以库里的 mode 为准 ----------


async def test_existing_work_session_uses_work_without_request_mode(client, db, monkeypatch):
    await seed_session(db, WORK_SESSION, "work")
    llm = FakeLLM()
    use_llm(monkeypatch, llm)

    resp = await client.post(
        "/api/chat", json={"session_id": WORK_SESSION, "message": "继续"}
    )

    assert resp.status_code == 200
    assert llm.calls[0][0].content == runtime.SYSTEM_PROMPT + runtime.MODE_PROMPTS["work"]
    assert (await session_row(db, WORK_SESSION))["mode"] == "work"


async def test_existing_work_session_accepts_matching_request_mode(client, db, monkeypatch):
    await seed_session(db, WORK_SESSION, "work")

    resp = await client.post(
        "/api/chat",
        json={"session_id": WORK_SESSION, "message": "继续", "mode": "work"},
    )

    assert resp.status_code == 200


async def test_existing_work_session_rejects_chat_mode(client, db):
    await seed_session(db, WORK_SESSION, "work")

    resp = await client.post(
        "/api/chat",
        json={"session_id": WORK_SESSION, "message": "换个模式", "mode": "chat"},
    )

    assert resp.status_code == 409
    assert "work" in resp.json()["detail"]


async def test_existing_chat_session_rejects_work_mode(client, db):
    await seed_session(db, SESSION, "chat")

    resp = await client.post(
        "/api/chat", json={"session_id": SESSION, "message": "升级", "mode": "work"}
    )

    assert resp.status_code == 409


async def test_unknown_session_is_404(client, db):
    """带 session_id 的请求 = 「续这个会话」，未知 id 一律 404（补建只走新建会话路径）。

    不论请求有没有带 mode：认不出来的会话都该让前端知道，而不是偷偷建一条新的。
    """
    assert (
        await client.post("/api/chat", json={"session_id": "nope", "message": "hi"})
    ).status_code == 404
    resp = await client.post(
        "/api/chat", json={"session_id": "nope", "message": "hi", "mode": "work"}
    )
    assert resp.status_code == 404
    # 404 是真的没建会话：库里不该多出一行
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall("SELECT id FROM sessions")
    assert rows == []


# ---------- respond：模式同样从会话读 ----------


async def test_respond_uses_session_mode(client, db, monkeypatch):
    await seed_session(db, WORK_SESSION, "work", messages=[("user", "什么是间隔重复？")])
    llm = FakeLLM()
    use_llm(monkeypatch, llm)

    resp = await client.post(f"/api/sessions/{WORK_SESSION}/respond")

    assert resp.status_code == 200
    assert llm.calls[0][0].content == runtime.SYSTEM_PROMPT + runtime.MODE_PROMPTS["work"]


# ---------- 非法 mode ----------


async def test_unknown_mode_is_rejected_with_400(client, db):
    resp = await client.post(
        "/api/chat", json={"session_id": None, "message": "hi", "mode": "chatty"}
    )

    assert resp.status_code == 400
    assert "chatty" in resp.json()["detail"]


async def test_code_mode_is_rejected_with_400(client, db):
    """code 模式本期不开放：API 也必须挡住，前端 disabled 只是 UX。"""
    resp = await client.post(
        "/api/chat", json={"session_id": None, "message": "hi", "mode": "code"}
    )

    assert resp.status_code == 400
    assert "code" in resp.json()["detail"]
    assert "未开放" in resp.json()["detail"]


async def test_session_with_unknown_stored_mode_is_conflict(client, db):
    """库里存了未知 mode（迁移后被人工改脏）：不猜、不降级，直接说清楚。"""
    await seed_session(db, SESSION, "bogus")

    resp = await client.post("/api/chat", json={"session_id": SESSION, "message": "hi"})

    assert resp.status_code == 409
    assert "bogus" in resp.json()["detail"]


# ---------- 迁移与运行时读取 ----------


async def test_legacy_db_gets_mode_and_source_defaults(tmp_path):
    """迁移前的库：sessions 连 mode 列都没有，init_db 要补列并归一化。"""
    path = tmp_path / "legacy.db"
    conn = await aiosqlite.connect(path)
    try:
        await conn.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT);"
            "INSERT INTO sessions (id, created_at) VALUES ('old1', '2026-01-01T00:00:00');"
        )
        await conn.commit()
    finally:
        await conn.close()

    await init_db(path, DIM)

    row = await session_row(path, "old1")
    assert row["mode"] == "chat"
    assert row["source"] == "manual"


async def test_reinit_normalizes_stored_unknown_mode(tmp_path, caplog):
    path = tmp_path / "app.db"
    await init_db(path, DIM)
    async with get_db(path) as conn:
        await conn.execute(
            "INSERT INTO sessions (id, created_at, mode) VALUES (?, ?, ?)",
            ("dirty1", "2026-10-04T09:00:00", "bogus"),
        )
        await conn.commit()

    with caplog.at_level(logging.WARNING, logger="app.db"):
        await init_db(path, DIM)

    assert (await session_row(path, "dirty1"))["mode"] == "chat"
    assert "dirty1" in caplog.text


async def test_ensure_session_does_not_overwrite_existing_mode(db):
    await seed_session(db, WORK_SESSION, "work")

    returned = await runtime.ensure_session(WORK_SESSION, str(db), mode="chat")

    assert returned == WORK_SESSION
    assert (await session_row(db, WORK_SESSION))["mode"] == "work"


async def test_get_session_mode_defaults_and_rejects_unknown(db):
    await seed_session(db, "legacy", None)  # 存量 NULL：迁移口径之外再兜一层
    await seed_session(db, "dirty", "bogus")

    assert await runtime.get_session_mode("legacy", str(db)) == "chat"
    assert await runtime.get_session_mode("missing", str(db)) == "chat"
    with pytest.raises(runtime.UnknownModeError):
        await runtime.get_session_mode("dirty", str(db))


async def test_run_agent_reports_unknown_mode(db):
    await seed_session(db, "dirty", "bogus")

    with pytest.raises(runtime.UnknownModeError):
        [event async for event in runtime.run_agent("dirty", "hi", str(db))]


# ---------- 模式 prompt 与工具白名单 ----------


def test_mode_prompts_cover_supported_modes_only():
    assert set(runtime.SUPPORTED_MODES) == {"chat", "work"}
    assert set(runtime.MODE_PROMPTS) == set(runtime.SUPPORTED_MODES)
    assert all(prompt.strip() for prompt in runtime.MODE_PROMPTS.values())


def test_mode_tools_whitelist():
    assert runtime.MODE_TOOLS == {
        "chat": ["search_knowledge"],
        "work": ["search_knowledge", "record_knowledge_gap", "review_knowledge_gap"],
    }


FAKE_GAP_TOOL = ToolDef(
    name="record_knowledge_gap",
    description="记录知识漏洞（测试替身，T3 才实现）",
    parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
)


def test_tools_for_mode_filters_by_whitelist(monkeypatch):
    """白名单真的在过滤，而不是「BUILTIN_TOOLS 里恰好没有 gap 工具」。

    注入一个 gap 名字的工具定义，让 chat / work 的差别真正来自 MODE_TOOLS：T3 接上
    真实现后这条断言仍然成立（chat 永远只有 search_knowledge）。
    """
    monkeypatch.setattr(runtime, "BUILTIN_TOOLS", [*runtime.BUILTIN_TOOLS, FAKE_GAP_TOOL])

    chat_names = [tool.name for tool in runtime.tools_for_mode("chat")]
    work_names = [tool.name for tool in runtime.tools_for_mode("work")]

    assert chat_names == ["search_knowledge"]
    assert chat_names.count(FAKE_GAP_TOOL.name) == 0
    assert FAKE_GAP_TOOL.name in work_names
    assert "search_knowledge" in work_names


async def test_run_agent_offers_only_the_current_modes_tools(db, monkeypatch):
    """端到端同一件事：注入 gap 工具后，chat 会话看不到它，work 会话拿得到。"""
    monkeypatch.setattr(runtime, "BUILTIN_TOOLS", [*runtime.BUILTIN_TOOLS, FAKE_GAP_TOOL])
    await seed_session(db, SESSION, "chat")
    await seed_session(db, WORK_SESSION, "work")
    llm = FakeLLM()
    use_llm(monkeypatch, llm)

    [event async for event in runtime.run_agent(SESSION, "什么是 RAG？", str(db))]
    [event async for event in runtime.run_agent(WORK_SESSION, "继续", str(db))]

    chat_tools = [tool.name for tool in llm.tools[0]]
    work_tools = [tool.name for tool in llm.tools[1]]
    assert "search_knowledge" in chat_tools
    assert FAKE_GAP_TOOL.name not in chat_tools
    assert FAKE_GAP_TOOL.name in work_tools


async def test_chat_dispatch_refuses_gap_tool_call(db):
    """白名单之外的工具即使模型硬调也不能执行（dispatch 二次检查）。"""
    result, label = await runtime.execute_tool(
        ToolCall(id="c1", name="record_knowledge_gap", arguments={"topic": "x", "detail": "y"}),
        str(db),
        mode="chat",
    )

    assert "不可用" in result
    assert "不可用" in label


async def test_chat_run_agent_survives_hallucinated_gap_tool(db, monkeypatch):
    await seed_session(db, SESSION, "chat")
    llm = ToolCallingLLM(
        ToolCall(id="c1", name="record_knowledge_gap", arguments={"topic": "x", "detail": "y"})
    )
    use_llm(monkeypatch, llm)

    events = [event async for event in runtime.run_agent(SESSION, "记一下", str(db))]

    ends = [e.data["summary"] for e in events if e.type == "tool_end"]
    assert ends and all("不可用" in summary for summary in ends)
    assert events[-1].type == "done"


async def test_assemble_messages_prepends_mode_prompt_and_persona(db):
    messages = runtime.assemble_messages([], "hi", mode="work")
    assert messages[0].role == "system"
    assert messages[0].content == runtime.SYSTEM_PROMPT + runtime.MODE_PROMPTS["work"]

    with_persona = runtime.assemble_messages([], "hi", mode="chat", persona="你是小助手。")
    assert with_persona[0].content.startswith("你是小助手。")
    assert runtime.MODE_PROMPTS["chat"] in with_persona[0].content


# ---------- 会话列表 ----------


async def test_sessions_list_exposes_mode_and_source(client, db):
    await seed_session(db, WORK_SESSION, "work")

    rows = (await client.get("/api/sessions")).json()

    row = next(r for r in rows if r["id"] == WORK_SESSION)
    assert row["mode"] == "work"
    assert row["source"] == "manual"
