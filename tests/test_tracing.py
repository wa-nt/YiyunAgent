"""T9 可观测（trace + 成本看板）测试。

埋点总开关由 tests/conftest.py 的 autouse 夹具钉住（默认关，避免真客户端用例往默认库
写 trace），本文件的 db 夹具显式打开它。所有 LLM 交互都是假传输（monkeypatch 掉
SDK 的 create/stream），不发真实请求。
"""

import logging
from types import SimpleNamespace

import httpx
import pytest

from app import tracing
from app.agent import context as ctx
from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.anthropic import AnthropicClient
from app.llm.openai_compat import OpenAICompatClient, detect_provider
from app.llm.types import ChatResult, Message, StreamChunk, ToolCall
from app.main import app
from app.memory import writer as memory_writer
from app.retrieval.bm25_search import invalidate
from app.retrieval.types import RetrievedChunk

DIM = 8
SESSION = "s-trace"
MODEL = "gpt-4o-mini"


# ---------- 桩与夹具 ----------


class SilentWriterLLM:
    """记忆抽取桩：既不调外部服务，也不写 trace（它不是真客户端）。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


class ToolLoopLLM:
    """前 tool_rounds 轮请求 search_knowledge，之后给一段纯文本答案。"""

    def __init__(self, tool_rounds: int):
        self.tool_rounds = tool_rounds
        self.calls = 0

    async def chat_stream(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self.tool_rounds:
            yield StreamChunk(
                finish=True,
                tool_calls=[
                    ToolCall(
                        id=f"c{self.calls}",
                        name="search_knowledge",
                        arguments={"query": f"第{self.calls}问"},
                    )
                ],
            )
            return
        yield StreamChunk(text_delta="答")
        yield StreamChunk(finish=True, tool_calls=[])


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """tmp 数据库 + 显式打开埋点开关 + 隔离的 BM25 缓存。"""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "memory_enabled", True)
    monkeypatch.setattr(memory_writer, "get_llm", lambda: SilentWriterLLM())
    invalidate()
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()
    await tracing.drain_traces()
    invalidate()


@pytest.fixture
async def client(db):
    """API 层用例：db 夹具已把 settings.db_path 指向 tmp，各接口都走默认路径。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def openai_client(model: str = MODEL, base_url: str | None = None) -> OpenAICompatClient:
    return OpenAICompatClient(api_key="k", model=model, base_url=base_url)


def stub_openai_chat(client, text: str = "好的", tokens_in: int = 10, tokens_out: int = 5):
    """把 SDK 的 create 换成假传输，返回一条带 usage 的完成响应。"""

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))
            ],
            usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )


def stub_openai_stream(
    client, tokens_in: int = 3, tokens_out: int = 2, with_usage: bool = True
):
    """流式假传输：正文两个分片，收尾 chunk 带（或不带）usage。"""

    async def fake_create(**kwargs):
        async def gen():
            for text in ("你", "好"):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text, tool_calls=None)
                        )
                    ],
                    usage=None,
                )
            yield SimpleNamespace(
                choices=[],
                usage=(
                    SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out)
                    if with_usage
                    else None
                ),
            )

        return gen()

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )


def anthropic_response(tokens_in: int = 10, tokens_out: int = 5):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="好的")],
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def stub_anthropic_chat(client, tokens_in: int = 10, tokens_out: int = 5) -> None:
    async def fake_create(**kwargs):
        return anthropic_response(tokens_in, tokens_out)

    client.client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))


class FakeAnthropicStream:
    def __init__(self, final):
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def text_stream(self):
        async def gen():
            yield "你"
            yield "好"

        return gen()

    async def get_final_message(self):
        return self.final


async def read(db, **filters) -> list[dict]:
    """按查询接口读 traces（最新在前），供断言埋点结果。"""
    return (await tracing.list_traces(db_path=db, **filters))["items"]


async def seed(db, rows: list[tuple]) -> None:
    """直接写库造数据：时间戳可控，便于断言时间范围过滤与分组数学。

    行格式：(kind, name, detail, tokens_in, tokens_out, cost, ts)
    """
    async with get_db(db) as conn:
        for kind, name, detail, tokens_in, tokens_out, cost, ts in rows:
            await conn.execute(
                "INSERT INTO traces (ts, kind, name, detail, tokens_in, tokens_out, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, kind, name, detail, tokens_in, tokens_out, cost),
            )
        await conn.commit()


def fake_search(monkeypatch, chunks: int = 1) -> list[str]:
    """替换 hybrid_search（默认 hybrid 会真的调 embedding 接口），记录 query。"""
    queries: list[str] = []

    async def fake(query, k=8, mode="hybrid", db_path=None):
        queries.append(query)
        return [
            RetrievedChunk(chunk_id=i, doc_id=1, content="检索到的内容", title="笔记", score=1.0)
            for i in range(chunks)
        ]

    monkeypatch.setattr(runtime, "hybrid_search", fake)
    return queries


def history(n: int) -> list[Message]:
    return [
        Message(role="user", content=f"问题{i}")
        if i % 2 == 0
        else Message(role="assistant", content=f"回答{i}")
        for i in range(n)
    ]


# ---------- 成本计算 ----------


def test_estimate_cost_per_provider():
    assert tracing.estimate_cost("openai", 1000, 1000) == pytest.approx(0.75)
    assert tracing.estimate_cost("deepseek", 1000, 1000) == pytest.approx(0.42)
    # 1M in / 0 out：3.0 美元每 1K → 3000
    assert tracing.estimate_cost("anthropic", 1_000_000, 0) == pytest.approx(3000.0)
    # 不满 1K 的按比例算，不取整
    assert tracing.estimate_cost("openai", 500, 0) == pytest.approx(0.075)
    assert tracing.estimate_cost("openai", 0, 0) == 0.0


def test_estimate_cost_uses_settings_prices(monkeypatch):
    """单价从 settings 读（不是写死的常量），改配置立即生效。"""
    monkeypatch.setattr(settings, "price_deepseek_input", 1.0)
    monkeypatch.setattr(settings, "price_deepseek_output", 2.0)

    assert tracing.estimate_cost("deepseek", 1000, 1000) == pytest.approx(3.0)


def test_estimate_cost_falls_back_to_openai_prices():
    """定价表只备 OpenAI / DeepSeek / Claude 三档，认不出的后端按 openai 档估算。"""
    assert tracing.estimate_cost("qwen", 1000, 1000) == tracing.estimate_cost(
        "openai", 1000, 1000
    )


def test_detect_provider_from_base_url():
    assert detect_provider(None) == "openai"
    assert detect_provider("https://api.openai.com/v1") == "openai"
    assert detect_provider("https://api.deepseek.com/v1") == "deepseek"
    assert detect_provider("https://dashscope.aliyuncs.com/compatible-mode/v1") == "openai"


# ---------- LLM 埋点 ----------


async def test_openai_chat_records_llm_trace(db):
    client = openai_client()
    stub_openai_chat(client, tokens_in=1000, tokens_out=1000)

    await client.chat([Message(role="user", content="hi")])
    await tracing.drain_traces()

    rows = await read(db)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "llm"
    assert row["name"] == "openai/gpt-4o-mini"
    assert (row["tokens_in"], row["tokens_out"]) == (1000, 1000)
    assert row["cost"] == pytest.approx(0.75)  # 0.15 + 0.60
    assert row["ts"]  # 时间戳落在库里


async def test_openai_stream_records_trace_with_final_usage(db):
    client = openai_client()
    stub_openai_stream(client, tokens_in=300, tokens_out=100)

    text = "".join([c.text_delta async for c in client.chat_stream([])])
    await tracing.drain_traces()

    assert text == "你好"
    rows = await read(db)
    assert len(rows) == 1
    assert rows[0]["name"] == "openai/gpt-4o-mini"
    assert (rows[0]["tokens_in"], rows[0]["tokens_out"]) == (300, 100)
    assert rows[0]["cost"] == pytest.approx(300 / 1000 * 0.15 + 100 / 1000 * 0.60)


async def test_stream_without_usage_still_counts_the_call(db):
    """兼容端点不返回 usage 时记 0 token、0 成本：调用次数仍要看得到，不编造 token。"""
    client = openai_client()
    stub_openai_stream(client, with_usage=False)

    [c async for c in client.chat_stream([])]
    await tracing.drain_traces()

    rows = await read(db)
    assert len(rows) == 1
    assert (rows[0]["tokens_in"], rows[0]["tokens_out"], rows[0]["cost"]) == (0, 0, 0.0)


async def test_anthropic_chat_records_llm_trace(db):
    client = AnthropicClient(api_key="k", model="claude-sonnet-4-5")
    stub_anthropic_chat(client, tokens_in=1000, tokens_out=1000)

    await client.chat([Message(role="user", content="hi")])
    await tracing.drain_traces()

    rows = await read(db)
    assert len(rows) == 1
    assert rows[0]["name"] == "anthropic/claude-sonnet-4-5"
    assert rows[0]["cost"] == pytest.approx(18.0)  # 3.0 + 15.0


async def test_anthropic_stream_records_llm_trace(db):
    client = AnthropicClient(api_key="k", model="claude-sonnet-4-5")
    final = anthropic_response(tokens_in=200, tokens_out=50)
    client.client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: FakeAnthropicStream(final))
    )

    text = "".join([c.text_delta async for c in client.chat_stream([])])
    await tracing.drain_traces()

    assert text == "你好"
    rows = await read(db)
    assert len(rows) == 1
    assert (rows[0]["tokens_in"], rows[0]["tokens_out"]) == (200, 50)


async def test_deepseek_base_url_uses_deepseek_prices(db):
    client = openai_client(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
    stub_openai_chat(client, tokens_in=1000, tokens_out=1000)

    await client.chat([Message(role="user", content="hi")])
    await tracing.drain_traces()

    rows = await read(db)
    assert rows[0]["name"] == "deepseek/deepseek-chat"
    assert rows[0]["cost"] == pytest.approx(0.42)  # 0.14 + 0.28


async def test_summary_llm_call_is_traced(db):
    """T8 的历史压缩走同一个 get_llm() 客户端，客户端层埋点自动覆盖它。"""
    client = openai_client()
    stub_openai_chat(client, text="用户之前问过 X", tokens_in=100, tokens_out=20)
    view = runtime.assemble_messages(history(12), "新问题")  # 12 条 > 默认阈值 10

    governed = await ctx.govern_context(view, llm=client)
    await tracing.drain_traces()

    assert any(m.content.startswith(ctx.SUMMARY_PREFIX) for m in governed)
    rows = await read(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "llm"
    assert (rows[0]["tokens_in"], rows[0]["tokens_out"]) == (100, 20)


async def test_trace_write_failure_is_logged_not_raised(db, tmp_path, monkeypatch, caplog):
    """写不进去只记 warning，不上抛：观测坏了不能影响对话。"""
    monkeypatch.setattr(settings, "db_path", str(tmp_path))  # 目录路径 → sqlite 打不开

    with caplog.at_level(logging.WARNING, logger="app.tracing"):
        tracing.record_trace("llm", "openai/gpt-4o-mini", tokens_in=1)
        await tracing.drain_traces()

    assert "trace 写入失败" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


# ---------- 工具埋点 ----------


async def test_execute_tool_records_tool_trace(db, monkeypatch):
    fake_search(monkeypatch, chunks=2)

    result, label = await runtime.execute_tool(
        ToolCall(id="c1", name="search_knowledge", arguments={"query": "RAG"}), db
    )
    await tracing.drain_traces()

    rows = await read(db)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "tool"
    assert row["name"] == "search_knowledge"
    assert row["detail"] == "search_knowledge(RAG) → 2 条"
    # 工具自己不烧 LLM 的钱
    assert (row["tokens_in"], row["tokens_out"], row["cost"]) == (0, 0, 0.0)
    assert "chunk 0" in result and label == row["detail"]


async def test_failed_tool_calls_are_traced(db, monkeypatch):
    """未知工具与缺 query 都是工具层的失败，成本看板上要看得见。"""
    queries = fake_search(monkeypatch)

    await runtime.execute_tool(ToolCall(id="c1", name="不存在的工具", arguments={}), db)
    await runtime.execute_tool(
        ToolCall(id="c2", name="search_knowledge", arguments={}), db
    )
    await tracing.drain_traces()

    assert queries == []  # 两条失败路径都没走到检索
    rows = sorted(r["detail"] for r in await read(db))
    assert rows == ["search_knowledge（缺少 query 参数）", "未知工具 不存在的工具"]


async def test_run_agent_traces_each_tool_call(db, monkeypatch):
    """接入点：run_agent 每执行一次工具调用就落一条 trace（假 LLM 不产生 llm 记录）。"""
    fake_search(monkeypatch)
    monkeypatch.setattr(runtime, "get_llm", lambda: ToolLoopLLM(tool_rounds=2))

    [e async for e in runtime.run_agent(SESSION, "什么是 RAG？", db)]
    await runtime.drain_memory_writes()
    await tracing.drain_traces()

    rows = await read(db)
    assert {r["kind"] for r in rows} == {"tool"}
    assert sorted(r["detail"] for r in rows) == [
        "search_knowledge(第1问) → 1 条",
        "search_knowledge(第2问) → 1 条",
    ]


# ---------- 开关 ----------


async def test_tracing_disabled_records_nothing(db, monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", False)
    client = openai_client()
    stub_openai_chat(client)

    await client.chat([Message(role="user", content="hi")])
    await runtime.execute_tool(ToolCall(id="c1", name="未知工具", arguments={}), db)
    await tracing.drain_traces()

    assert await read(db) == []


# ---------- 看板 API ----------


SEED = [
    # (kind, name, detail, tokens_in, tokens_out, cost, ts)
    ("llm", "openai/gpt-4o-mini", "", 100, 20, 0.027, "2026-09-25T10:00:00.000+00:00"),
    ("llm", "deepseek/deepseek-chat", "", 200, 40, 0.0392, "2026-09-25T11:00:00.000+00:00"),
    (
        "tool",
        "search_knowledge",
        "search_knowledge(RAG) → 1 条",
        0,
        0,
        0.0,
        "2026-09-26T09:00:00.000+00:00",
    ),
]


async def test_traces_endpoint_lists_newest_first_with_total(client, db):
    await seed(db, SEED)

    resp = await client.get("/api/traces")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["limit"] == 100 and body["offset"] == 0
    # 最新（id 最大）在前
    assert [r["name"] for r in body["items"]] == [
        "search_knowledge",
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
    ]
    assert body["items"][0]["detail"] == "search_knowledge(RAG) → 1 条"


async def test_traces_endpoint_filters_by_kind_name_and_time(client, db):
    await seed(db, SEED)

    by_kind = (await client.get("/api/traces", params={"kind": "llm"})).json()
    assert by_kind["total"] == 2
    assert {r["kind"] for r in by_kind["items"]} == {"llm"}

    by_name = (await client.get("/api/traces", params={"name": "search_knowledge"})).json()
    assert by_name["total"] == 1
    assert by_name["items"][0]["kind"] == "tool"

    window = (
        await client.get(
            "/api/traces",
            params={
                "start": "2026-09-25T10:30:00.000+00:00",
                "end": "2026-09-26T00:00:00.000+00:00",
            },
        )
    ).json()
    assert window["total"] == 1
    assert window["items"][0]["name"] == "deepseek/deepseek-chat"

    combined = (
        await client.get(
            "/api/traces", params={"kind": "llm", "name": "openai/gpt-4o-mini"}
        )
    ).json()
    assert combined["total"] == 1

    missing = (await client.get("/api/traces", params={"kind": "skill"})).json()
    assert missing["total"] == 0 and missing["items"] == []


async def test_traces_endpoint_paginates(client, db):
    await seed(db, SEED)

    first = (await client.get("/api/traces", params={"limit": 2})).json()
    second = (await client.get("/api/traces", params={"limit": 2, "offset": 2})).json()

    assert [r["id"] for r in first["items"]] == [3, 2]
    assert [r["id"] for r in second["items"]] == [1]
    assert first["limit"] == 2 and second["offset"] == 2
    # total 是过滤后的总条数，不随分页变
    assert first["total"] == second["total"] == 3


async def test_traces_endpoint_rejects_out_of_range_paging(client, db):
    assert (await client.get("/api/traces", params={"limit": 0})).status_code == 422
    assert (await client.get("/api/traces", params={"limit": 1001})).status_code == 422
    assert (await client.get("/api/traces", params={"offset": -1})).status_code == 422


async def test_traces_summary_groups_by_kind_and_name(client, db):
    await seed(db, SEED)

    resp = await client.get("/api/traces/summary")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_calls"] == 3
    assert (body["tokens_in"], body["tokens_out"], body["tokens_total"]) == (300, 60, 360)
    assert body["cost"] == pytest.approx(0.0662)

    # 分组按成本降序：llm（0.0662）> tool（0）；同组内 tokens 累加
    assert [g["kind"] for g in body["by_kind"]] == ["llm", "tool"]
    assert body["by_kind"][0]["calls"] == 2
    assert body["by_kind"][0]["tokens_in"] == 300
    assert body["by_kind"][0]["cost"] == pytest.approx(0.0662)
    assert body["by_kind"][1]["calls"] == 1 and body["by_kind"][1]["cost"] == 0.0

    assert [g["name"] for g in body["by_name"]] == [
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
        "search_knowledge",
    ]
    assert body["by_name"][0]["cost"] == pytest.approx(0.0392)


async def test_traces_summary_honours_filters(client, db):
    await seed(db, SEED)

    body = (await client.get("/api/traces/summary", params={"kind": "llm"})).json()

    assert body["total_calls"] == 2
    assert [g["kind"] for g in body["by_kind"]] == ["llm"]
    assert [g["name"] for g in body["by_name"]] == [
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
    ]


async def test_traces_summary_on_empty_table_returns_zeros(client, db):
    body = (await client.get("/api/traces/summary")).json()

    assert body["total_calls"] == 0
    assert (body["tokens_in"], body["tokens_out"], body["tokens_total"], body["cost"]) == (
        0,
        0,
        0,
        0,
    )
    assert body["by_kind"] == [] and body["by_name"] == []


async def test_llm_call_reaches_the_dashboard_end_to_end(client, db):
    """真客户端（假传输）→ 埋点 → 看板 API：成本在接口上能对上定价表。"""
    llm = openai_client()
    stub_openai_chat(llm, tokens_in=1000, tokens_out=1000)

    await llm.chat([Message(role="user", content="hi")])
    await tracing.drain_traces()

    listing = (await client.get("/api/traces", params={"kind": "llm"})).json()
    assert listing["total"] == 1
    assert listing["items"][0]["name"] == "openai/gpt-4o-mini"
    assert listing["items"][0]["cost"] == pytest.approx(0.75)

    summary = (await client.get("/api/traces/summary")).json()
    assert summary["total_calls"] == 1
    assert summary["cost"] == pytest.approx(0.75)
    assert summary["by_kind"] == [
        {
            "kind": "llm",
            "calls": 1,
            "tokens_in": 1000,
            "tokens_out": 1000,
            "cost": 0.75,
        }
    ]