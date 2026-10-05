"""会话/消息操作、分支、导入导出、记忆编辑的 API 用例。

隔离方式与 tests/test_api.py 一致：db 夹具把 settings.db_path 指到 tmp 并 init_db
（ASGITransport 不跑 lifespan，建表的责任在夹具），完全不碰 data/app.db。
"""

import json

import httpx
import pytest

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk
from app.main import app
from app.memory import writer as memory_writer
from app.retrieval.bm25_search import invalidate

DIM = 8


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    # 记忆抽取换成空桩，标题生成由各用例自己的 fake LLM 决定（见 TitleLLM）
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    invalidate()
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()
    invalidate()


class _SilentWriterLLM:
    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def seed_session(db, session_id="s1", messages=()):
    """种线性会话：parent 链 + active_leaf（load_history/list_messages 都沿链回溯）。"""
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, "2026-09-28T10:00:00"),
        )
        prev_id = None
        for role, content in messages:
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, parent_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, "2026-09-28T10:01:00", prev_id),
            )
            prev_id = cursor.lastrowid
        if prev_id is not None:
            await conn.execute(
                "UPDATE sessions SET active_leaf = ? WHERE id = ?",
                (prev_id, session_id),
            )
        await conn.commit()


async def messages_of(db, session_id="s1"):
    """会话里的全部消息（含所有分支），按插入顺序。"""
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    return [(r["role"], r["content"]) for r in rows]


async def active_path(client, session_id="s1"):
    """前端看到的当前分支。"""
    return (await client.get(f"/api/sessions/{session_id}/messages")).json()


# ---------- 消息编辑（PUT /api/messages/{id} = 开分支） ----------


async def test_edit_user_message_creates_sibling_branch(client, db):
    await seed_session(db, messages=[("user", "旧提问"), ("assistant", "旧回答"), ("user", "后续")])

    resp = await client.put("/api/messages/1", json={"content": "改过的提问"})

    assert resp.status_code == 200
    assert resp.json()["new_id"] == 4
    # 旧分支原样保留，新分支是同 parent（根）的兄弟
    assert await messages_of(db) == [
        ("user", "旧提问"), ("assistant", "旧回答"), ("user", "后续"), ("user", "改过的提问"),
    ]
    path = await active_path(client)
    assert [m["content"] for m in path] == ["改过的提问"]
    assert path[0]["branch_count"] == 2 and path[0]["branch_index"] == 2


async def test_edit_rejects_assistant_message(client, db):
    """assistant 内容不允许改——改它等于伪造模型说过的历史。"""
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])

    resp = await client.put("/api/messages/2", json={"content": "篡改"})

    assert resp.status_code == 422
    assert await messages_of(db) == [("user", "问"), ("assistant", "答")]


async def test_edit_missing_message_is_404(client, db):
    assert (await client.put("/api/messages/99", json={"content": "x"})).status_code == 404


async def test_edit_rejects_empty_content(client, db):
    await seed_session(db, messages=[("user", "问")])
    assert (await client.put("/api/messages/1", json={"content": "  "})).status_code == 422


# ---------- 分支切换（POST /api/messages/{id}/branch） ----------


async def test_branch_switch_walks_between_roots(client, db):
    await seed_session(db, messages=[("user", "旧提问"), ("assistant", "旧回答")])
    await client.put("/api/messages/1", json={"content": "改过的提问"})  # 根分支 1 → 2 个

    # 从新分支（消息 3）切回旧分支：active_leaf 落到旧分支的叶子（消息 2）
    resp = await client.post("/api/messages/3/branch", json={"direction": -1})
    assert resp.status_code == 200 and resp.json() == {"leaf": 2}
    assert [m["content"] for m in await active_path(client)] == ["旧提问", "旧回答"]

    # 再切回来
    resp = await client.post("/api/messages/1/branch", json={"direction": 1})
    assert resp.status_code == 200 and resp.json() == {"leaf": 3}
    assert [m["content"] for m in await active_path(client)] == ["改过的提问"]


async def test_branch_switch_out_of_range_is_404(client, db):
    await seed_session(db, messages=[("user", "唯一分支")])
    resp = await client.post("/api/messages/1/branch", json={"direction": 1})
    assert resp.status_code == 404


# ---------- 消息删除（DELETE /api/messages/{id} = 删子树） ----------


async def test_delete_message_removes_subtree_and_fixes_leaf(client, db):
    await seed_session(db, messages=[("user", "问1"), ("assistant", "答1"), ("user", "问2")])

    resp = await client.delete("/api/messages/2")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": 2}
    assert await messages_of(db) == [("user", "问1")]
    assert [m["content"] for m in await active_path(client)] == ["问1"]


async def test_delete_branch_keeps_sibling_alive(client, db):
    """删掉一个分支，兄弟分支不受影响，active_leaf 落到兄弟子树的叶子。"""
    await seed_session(db, messages=[("user", "旧提问"), ("assistant", "旧回答")])
    await client.put("/api/messages/1", json={"content": "新提问"})  # 叶子切到消息 3

    resp = await client.delete("/api/messages/3")

    assert resp.json() == {"deleted": 1}
    assert [m["content"] for m in await active_path(client)] == ["旧提问", "旧回答"]


async def test_delete_missing_message_is_404(client, db):
    assert (await client.delete("/api/messages/99")).status_code == 404


# ---------- 会话重命名 / 删除 / 搜索 / 供应商覆盖 ----------


async def test_session_rename_overrides_derived_title(client, db):
    await seed_session(db, messages=[("user", "知识库是什么")])

    resp = await client.patch("/api/sessions/s1", json={"title": "改名了"})

    assert resp.status_code == 200
    rows = (await client.get("/api/sessions")).json()
    assert rows[0]["title"] == "改名了"


async def test_session_provider_override_roundtrip(client, db):
    await seed_session(db, messages=[("user", "问")])

    resp = await client.patch("/api/sessions/s1", json={"provider": "anthropic"})
    assert resp.status_code == 200
    rows = (await client.get("/api/sessions")).json()
    assert rows[0]["provider"] == "anthropic"

    # 空串 = 清掉覆盖，回到跟随全局
    resp = await client.patch("/api/sessions/s1", json={"provider": ""})
    assert resp.status_code == 200
    rows = (await client.get("/api/sessions")).json()
    assert rows[0]["provider"] is None


async def test_session_patch_validation(client, db):
    await seed_session(db)
    assert (await client.patch("/api/sessions/nope", json={"title": "x"})).status_code == 404
    assert (await client.patch("/api/sessions/s1", json={"title": " "})).status_code == 422
    assert (await client.patch("/api/sessions/s1", json={})).status_code == 422


async def test_session_delete_is_soft_and_restorable(client, db):
    """删除只打 deleted_at 标记：列表里消失、消息原样留着，撤销后完整回来。"""
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])
    await seed_session(db, "s2", [("user", "别删我")])

    resp = await client.delete("/api/sessions/s1")

    assert resp.status_code == 200
    assert [c for _, c in await messages_of(db)] == ["问", "答"]  # 消息没被物理删除
    rows = (await client.get("/api/sessions")).json()
    assert [r["id"] for r in rows] == ["s2"]
    assert (await client.get("/api/sessions/s1")).status_code == 404
    assert (await client.get("/api/sessions/s1/messages")).status_code == 404

    restored = await client.post("/api/sessions/s1/restore")
    assert restored.status_code == 200
    assert sorted(r["id"] for r in (await client.get("/api/sessions")).json()) == ["s1", "s2"]
    assert len((await client.get("/api/sessions/s1/messages")).json()) == 2


async def test_session_delete_missing_is_404(client, db):
    assert (await client.delete("/api/sessions/nope")).status_code == 404


async def test_session_search_matches_title_and_message_content(client, db):
    await seed_session(db, "s1", [("user", "讲一讲向量检索")])
    await seed_session(db, "s2", [("user", "晚饭吃什么")])
    await client.patch("/api/sessions/s2", json={"title": "生活琐事"})

    by_content = (await client.get("/api/sessions", params={"q": "向量"})).json()
    assert [r["id"] for r in by_content] == ["s1"]
    by_title = (await client.get("/api/sessions", params={"q": "生活"})).json()
    assert [r["id"] for r in by_title] == ["s2"]
    assert (await client.get("/api/sessions", params={"q": "不存在"})).json() == []


async def test_session_search_treats_percent_as_literal(client, db):
    """LIKE 通配符必须转义：搜 100% 不该匹配所有会话。"""
    await seed_session(db, "s1", [("user", "进度 100% 完成")])
    await seed_session(db, "s2", [("user", "普通的 100 字提问")])

    rows = (await client.get("/api/sessions", params={"q": "100%"})).json()

    assert [r["id"] for r in rows] == ["s1"]


# ---------- 导出 ----------


async def test_export_contains_user_data(client, db):
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO memories (kind, content, confidence, created_at, status) "
            "VALUES ('fact', '喜欢 Python', 0.8, '2026-09-28T10:00:00', 'active')"
        )
        await conn.commit()

    resp = await client.get("/api/export")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    data = resp.json()
    assert [m["content"] for m in data["messages"]] == ["问", "答"]
    assert data["memories"][0]["content"] == "喜欢 Python"
    assert data["sessions"][0]["id"] == "s1"


# ---------- 导入（ChatGPT / Claude 自动嗅探） ----------


def chatgpt_export() -> bytes:
    """最小的 conversations.json：一问一答，主分支 linearize 后两条消息。"""
    convo = {
        "title": "讨论 RAG",
        "create_time": 1714000000,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["n1"], "message": None},
            "n1": {
                "id": "n1", "parent": "root", "children": ["n2"],
                "message": {
                    "author": {"role": "user"}, "create_time": 1714000001,
                    "content": {"content_type": "text", "parts": ["什么是 RAG？"]},
                },
            },
            "n2": {
                "id": "n2", "parent": "n1", "children": [],
                "message": {
                    "author": {"role": "assistant"}, "create_time": 1714000002,
                    "content": {"content_type": "text", "parts": ["检索增强生成。"]},
                },
            },
        },
    }
    return json.dumps([convo]).encode()


def claude_export() -> bytes:
    """Claude 导出：chat_messages 线性列表，sender 用 human/assistant。"""
    convo = {
        "uuid": "c1",
        "name": "Claude 聊天",
        "created_at": "2026-01-01T00:00:00.000000Z",
        "chat_messages": [
            {"uuid": "m1", "sender": "human", "text": "你好",
             "created_at": "2026-01-01T00:00:01Z"},
            {"uuid": "m2", "sender": "assistant", "text": "",
             "content": [{"type": "text", "text": "你好！有什么可以帮你？"}],
             "created_at": "2026-01-01T00:00:02Z"},
        ],
    }
    return json.dumps([convo]).encode()


async def test_import_chatgpt_creates_session_with_title(client, db):
    resp = await client.post(
        "/api/import/chatgpt",
        files={"file": ("conversations.json", chatgpt_export(), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"sessions": 1, "messages": 2}
    rows = (await client.get("/api/sessions")).json()
    assert rows[0]["title"] == "讨论 RAG"
    msgs = await active_path(client, rows[0]["id"])
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "什么是 RAG？"),
        ("assistant", "检索增强生成。"),
    ]


async def test_import_claude_creates_session_with_name(client, db):
    resp = await client.post(
        "/api/import/chatgpt",
        files={"file": ("conversations.json", claude_export(), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"sessions": 1, "messages": 2}
    rows = (await client.get("/api/sessions")).json()
    assert rows[0]["title"] == "Claude 聊天"
    msgs = await active_path(client, rows[0]["id"])
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "你好"),
        ("assistant", "你好！有什么可以帮你？"),
    ]


async def test_import_rejects_bad_json(client, db):
    resp = await client.post(
        "/api/import/chatgpt",
        files={"file": ("conversations.json", b"not json", "application/json")},
    )
    assert resp.status_code == 422


async def test_import_rejects_non_list(client, db):
    resp = await client.post(
        "/api/import/chatgpt",
        files={"file": ("conversations.json", b'{"foo": 1}', "application/json")},
    )
    assert resp.status_code == 422


# ---------- 记忆编辑 / 删除 ----------


async def seed_memory(db) -> int:
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO memories (kind, content, confidence, created_at, status) "
            "VALUES ('fact', '旧记忆', 0.8, '2026-09-28T10:00:00', 'active')"
        )
        await conn.commit()
        return cursor.lastrowid


async def test_memory_edit_and_delete(client, db):
    mid = await seed_memory(db)

    resp = await client.patch(f"/api/memories/{mid}", json={"content": "改过的记忆"})
    assert resp.status_code == 200
    rows = (await client.get("/api/memories")).json()
    assert rows[0]["content"] == "改过的记忆"

    resp = await client.delete(f"/api/memories/{mid}")
    assert resp.status_code == 200
    assert (await client.get("/api/memories")).json() == []


async def test_memory_missing_is_404(client, db):
    assert (await client.patch("/api/memories/99", json={"content": "x"})).status_code == 404
    assert (await client.delete("/api/memories/99")).status_code == 404


# ---------- respond（编辑/重生成的后端入口） ----------


async def test_respond_requires_user_message_at_tail(client, db):
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])

    resp = await client.post("/api/sessions/s1/respond")

    assert resp.status_code == 409


async def test_respond_streams_answer_without_duplicating_question(client, db, monkeypatch):
    """respond 复用 run_agent(user_saved=True)：提问已在库里，不得再插一条。"""
    await seed_session(db, messages=[("user", "什么是 RAG？")])

    async def fake_events(session_id, message, **kwargs):
        assert kwargs.get("user_saved") is True
        yield runtime.AgentEvent("text_delta", {"text": "检索增强生成"})
        yield runtime.AgentEvent("done", {"session_id": session_id, "text": "检索增强生成"})

    monkeypatch.setattr("app.main.run_agent", fake_events)

    resp = await client.post("/api/sessions/s1/respond")

    assert resp.status_code == 200
    assert '"text_delta"' in resp.text and '"done"' in resp.text
    # run_agent 被换成桩，库里仍只有种进去的那一条提问
    assert await messages_of(db) == [("user", "什么是 RAG？")]


async def test_respond_mid_rewinds_to_that_question(client, db, monkeypatch):
    """重新生成 = 把 active_leaf 切回那条提问；旧回答的分支原样保留。"""
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])

    async def fake_events(session_id, message, **kwargs):
        assert message == "问"
        yield runtime.AgentEvent("done", {"session_id": session_id, "text": ""})

    monkeypatch.setattr("app.main.run_agent", fake_events)

    resp = await client.post("/api/sessions/s1/respond", params={"mid": 1})

    assert resp.status_code == 200
    async with get_db(db) as conn:
        leaf = (await conn.execute_fetchall(
            "SELECT active_leaf FROM sessions WHERE id = 's1'"
        ))[0]["active_leaf"]
    assert leaf == 1
    assert await messages_of(db) == [("user", "问"), ("assistant", "答")]


async def test_respond_mid_rejects_assistant_message(client, db):
    await seed_session(db, messages=[("user", "问"), ("assistant", "答")])
    resp = await client.post("/api/sessions/s1/respond", params={"mid": 2})
    assert resp.status_code == 422


# ---------- runtime：user_saved / 自动标题 / 供应商覆盖 ----------


class TitleLLM:
    """chat_stream 给正文、chat 给标题——两个入口都是 runtime 自己的 get_llm。"""

    def __init__(self):
        self.calls: list[list] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        yield StreamChunk(text_delta="回答正文")
        yield StreamChunk(finish=True, tool_calls=[])

    async def chat(self, messages, tools=None):
        return ChatResult(text="RAG 简介")


async def test_run_agent_user_saved_skips_duplicate_insert(db, monkeypatch):
    await seed_session(db, messages=[("user", "什么是 RAG？")])
    llm = TitleLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [
        e async for e in runtime.run_agent("s1", "什么是 RAG？", user_saved=True)
    ]

    assert events[-1].type == "done"
    rows = await messages_of(db)
    assert rows == [("user", "什么是 RAG？"), ("assistant", "回答正文")]
    # 历史里已含的提问不得再进一遍 prompt：发给模型的消息里 user 只出现一次
    prompt_users = [m for m in llm.calls[0] if m.role == "user"]
    assert len(prompt_users) == 1


async def test_regenerate_keeps_old_answer_on_sibling_branch(db, monkeypatch):
    """真实跑一轮 re-answer：旧回答留在原分支，新回答是它的兄弟，当前分支切换到新回答。"""
    await seed_session(db, messages=[("user", "问"), ("assistant", "旧答")])
    monkeypatch.setattr(runtime, "get_llm", lambda: TitleLLM())
    async with get_db(db) as conn:
        await conn.execute("UPDATE sessions SET active_leaf = 1 WHERE id = 's1'")
        await conn.commit()

    events = [e async for e in runtime.run_agent("s1", "问", user_saved=True)]

    assert events[-1].type == "done"
    assert await messages_of(db) == [("user", "问"), ("assistant", "旧答"), ("assistant", "回答正文")]


async def test_session_provider_override_is_used(db, monkeypatch):
    """会话级 provider 覆盖透传给 get_llm；无覆盖时保持零参调用（兼容既有桩）。"""
    await seed_session(db, messages=[("user", "问")])
    async with get_db(db) as conn:
        await conn.execute("UPDATE sessions SET provider = 'anthropic' WHERE id = 's1'")
        await conn.commit()
    captured: dict = {}

    def fake_get_llm(**kwargs):
        captured.update(kwargs)
        return TitleLLM()

    monkeypatch.setattr(runtime, "get_llm", fake_get_llm)

    events = [e async for e in runtime.run_agent("s1", "问", user_saved=True)]

    assert events[-1].type == "done"
    assert captured == {"provider": "anthropic", "model": None, "effort": None}


async def test_first_turn_generates_title(db, monkeypatch):
    monkeypatch.setattr(runtime, "get_llm", lambda: TitleLLM())

    events = [e async for e in runtime.run_agent("s1", "什么是 RAG？")]
    assert events[-1].type == "done"
    await runtime.drain_memory_writes()

    rows = await client_get_sessions(db)
    assert rows[0]["title"] == "RAG 简介"


async def test_title_write_never_overwrites_existing_title(db, monkeypatch):
    """用户改过名 / 导入带来的标题（title 非 NULL）不被自动标题覆盖。"""
    await seed_session(db, messages=[("user", "什么是 RAG？")])
    async with get_db(db) as conn:
        await conn.execute("UPDATE sessions SET title = '我的标题' WHERE id = 's1'")
        await conn.execute("DELETE FROM messages")  # 清空消息，让下一轮仍是 first_turn
        await conn.execute("UPDATE sessions SET active_leaf = NULL WHERE id = 's1'")
        await conn.commit()
    monkeypatch.setattr(runtime, "get_llm", lambda: TitleLLM())

    events = [e async for e in runtime.run_agent("s1", "换个提问")]
    assert events[-1].type == "done"
    await runtime.drain_memory_writes()

    async with get_db(db) as conn:
        row = (await conn.execute_fetchall("SELECT title FROM sessions WHERE id = 's1'"))[0]
    assert row["title"] == "我的标题"


async def client_get_sessions(db):
    async with get_db(db) as conn:
        return await conn.execute_fetchall("SELECT id, title FROM sessions")


async def test_import_own_export_roundtrip(client, db):
    """自家 /api/export 的回灌：按原 id 落库（幂等），分支链 parent_id/active_leaf 原样保住。"""
    export = {
        "version": 1,
        "exported_at": "2026-10-05T00:00:00",
        "sessions": [
            {"id": "s-own", "created_at": "2026-10-01T00:00:00", "title": "旧会话", "active_leaf": 2}
        ],
        "messages": [
            {"id": 1, "session_id": "s-own", "role": "user", "content": "Q",
             "created_at": "2026-10-01T00:00:01", "parent_id": None},
            {"id": 2, "session_id": "s-own", "role": "assistant", "content": "A",
             "created_at": "2026-10-01T00:00:02", "parent_id": 1},
        ],
        "memories": [
            {"id": 1, "kind": "fact", "content": "备份的记忆", "confidence": 0.9,
             "status": "active", "created_at": "2026-10-01T00:00:00"}
        ],
    }
    payload = json.dumps(export).encode()
    resp = await client.post(
        "/api/import", files={"file": ("export.json", payload, "application/json")}
    )
    assert resp.status_code == 200
    assert resp.json()["imported"]["sessions"] == 1
    assert resp.json()["imported"]["messages"] == 2
    msgs = await active_path(client, "s-own")
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "Q"), ("assistant", "A")]
    assert (await client.get("/api/memories")).json()[0]["content"] == "备份的记忆"

    # 幂等：同一份备份再灌一次，零新增
    resp2 = await client.post(
        "/api/import", files={"file": ("export.json", payload, "application/json")}
    )
    assert all(n == 0 for n in resp2.json()["imported"].values())


async def test_import_unified_still_accepts_chatgpt(client, db):
    resp = await client.post(
        "/api/import",
        files={"file": ("conversations.json", chatgpt_export(), "application/json")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"sessions": 1, "messages": 2}


async def test_import_rejects_unknown_shape(client, db):
    resp = await client.post(
        "/api/import", files={"file": ("x.json", b'{"foo": 1}', "application/json")}
    )
    assert resp.status_code == 422
