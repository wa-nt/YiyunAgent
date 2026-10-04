"""T2 可配置人格：多行原文存 app_settings，未设置回退仓库内默认人格，空串 = 明确禁用。

隔离方式与 tests/test_api.py 一致：db 夹具把 settings.db_path 指到 tmp 并 init_db
（ASGITransport 不跑 lifespan，建表的责任在夹具），LLM 一律换脚本桩，不碰 data/app.db。

三条语义边界是这个任务的重点，也各有一条用例：无记录 → 默认人格文件；空字符串 →
「用户明确不要人格」并按原样 round-trip；非空 → 原文注入 system 消息。
"""

from __future__ import annotations

import httpx
import pytest

from app.agent import runtime
from app.config import EDITABLE_FIELDS, settings
from app.db import init_db
from app.llm.types import ChatResult, StreamChunk
from app.main import app
from app.memory import writer as memory_writer
from app.resources import resource_path
from app.settings_store import get_setting, set_setting

DIM = 8
SESSION = "p1"

# 中文 + 换行 + 半角/全角引号 + 结尾空行：写进 .env 一定会被写坏的那类原文
MULTILINE = '你是一位可靠的思考伙伴。\n第二行：「引号」与 \'单引号\'、"双引号" 原样保留。\n\n'


class FakeLLM:
    """记录每轮收到的 messages，返回一段固定回答（同 tests/test_modes.py 的桩）。"""

    def __init__(self, text: str = "好的"):
        self.text = text
        self.calls: list[list] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        yield StreamChunk(text_delta=self.text)
        yield StreamChunk(finish=True, tool_calls=[])

    async def chat(self, messages, tools=None):  # _write_title 用得到
        return ChatResult(text="标题")


class SilentWriterLLM:
    """记忆抽取桩：这些用例不关心抽取结果。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(memory_writer, "get_llm", lambda: SilentWriterLLM())
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def default_text() -> str:
    return runtime.persona_default_path().read_text(encoding="utf-8")


# ---------- 存储层：多行原文 round-trip ----------


async def test_multiline_persona_round_trips(db):
    await set_setting("persona", MULTILINE, db)

    assert await get_setting("persona", db) == MULTILINE
    assert await runtime.load_persona(db) == MULTILINE  # 换行/引号/空行都不动


async def test_set_setting_overwrites_previous_value(db):
    await set_setting("persona", "旧人格", db)
    await set_setting("persona", "新人格", db)

    assert await get_setting("persona", db) == "新人格"


# ---------- 加载语义：无记录 → 默认文件；空串 → 保持为空 ----------


async def test_no_record_loads_repository_default(db):
    assert await get_setting("persona", db) is None  # 没设置过 ≠ 设置成空

    text = await runtime.load_persona(db)

    assert text == default_text()
    assert text.strip(), "默认人格不能是空文件"


async def test_empty_string_stays_empty(db):
    await set_setting("persona", "", db)

    assert await get_setting("persona", db) == ""
    assert await runtime.load_persona(db) == ""  # 不回退默认文件


async def test_missing_default_file_raises_clear_error(db, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "persona_default_path", lambda: tmp_path / "persona_default.md")

    with pytest.raises(runtime.PersonaUnavailableError) as exc:
        await runtime.load_persona(db)

    assert "persona_default.md" in str(exc.value)  # 报错要能看出缺的是哪个文件


# ---------- 资源路径：源码环境可读 ----------


def test_default_persona_resource_is_readable_in_source_tree():
    path = runtime.persona_default_path()

    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()
    # 同一套解析口径也覆盖 schema 与 web（打包时资源布局与仓库一致）
    assert resource_path("app/schema.sql").is_file()
    assert resource_path("web/index.html").is_file()


# ---------- 注入：人格真的进了发给模型的 system 消息 ----------


async def test_persona_is_injected_before_system_prompt(db, monkeypatch):
    await set_setting("persona", MULTILINE, db)
    llm = FakeLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    async for _ in runtime.run_agent(SESSION, "你好", db):
        pass

    system = llm.calls[0][0].content
    assert system.startswith(MULTILINE)
    assert runtime.SYSTEM_PROMPT in system
    assert runtime.MODE_PROMPTS[runtime.DEFAULT_MODE] in system


async def test_default_persona_is_injected_when_unset(db, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    async for _ in runtime.run_agent(SESSION, "你好", db):
        pass

    assert llm.calls[0][0].content.startswith(default_text())


async def test_empty_persona_is_not_injected(db, monkeypatch):
    await set_setting("persona", "", db)
    llm = FakeLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    async for _ in runtime.run_agent(SESSION, "你好", db):
        pass

    assert llm.calls[0][0].content == (
        runtime.SYSTEM_PROMPT + runtime.MODE_PROMPTS[runtime.DEFAULT_MODE]
    )


# ---------- 设置 API ----------


async def test_settings_api_returns_default_persona(client):
    data = (await client.get("/api/settings")).json()

    assert data["persona"] == default_text()


async def test_settings_api_round_trips_multiline_persona(client, db, tmp_path, monkeypatch):
    # 供应商配置是接口既有的必填校验项，这里钉成完整配置，让 persona 成为唯一变量
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "deepseek-chat")
    env = tmp_path / "settings.env"
    monkeypatch.setattr("app.main.env_path", lambda: env)

    resp = await client.post("/api/settings", json={"persona": MULTILINE})

    assert resp.status_code == 200, resp.text
    assert "persona" in resp.json()["updated"]
    assert (await client.get("/api/settings")).json()["persona"] == MULTILINE
    # 多行原文不进 .env：.env 是单行键值口径，写进去会被写坏
    env_text = env.read_text(encoding="utf-8") if env.exists() else ""
    assert "PERSONA" not in env_text.upper()
    assert "persona" not in EDITABLE_FIELDS


async def test_settings_api_empty_persona_stays_empty(client, db, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "deepseek-chat")
    await set_setting("persona", "先存一版", db)

    resp = await client.post("/api/settings", json={"persona": ""})

    assert resp.status_code == 200, resp.text
    assert (await client.get("/api/settings")).json()["persona"] == ""


async def test_settings_api_without_persona_field_keeps_stored_value(client, db, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "deepseek-chat")
    await set_setting("persona", MULTILINE, db)

    resp = await client.post("/api/settings", json={"openai_model": "deepseek-chat"})

    assert resp.status_code == 200, resp.text
    assert "persona" not in resp.json()["updated"]
    assert await get_setting("persona", db) == MULTILINE
