"""T9 可观测（trace + 成本看板）测试。

埋点总开关由 tests/conftest.py 的 autouse 夹具钉住（默认关，避免真客户端用例往默认库
写 trace），本文件的 db 夹具显式打开它。所有 LLM 交互都是假传输（monkeypatch 掉
SDK 的 create/stream），不发真实请求。
"""

import asyncio
import logging
import sqlite3
import time
from types import SimpleNamespace

import httpx
import pytest

from app import main as main_module
from app import tracing
from app.agent import context as ctx
from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.anthropic import AnthropicClient
from app.llm.openai_compat import OpenAICompatClient, detect_provider
from app.llm.types import ChatResult, Message, StreamChunk, ToolCall
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
    transport = httpx.ASGITransport(app=main_module.app)
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


def count_sync(db) -> int:
    """同步数一串 traces 行。

    用 stdlib sqlite3 而不是 aiosqlite：读它的过程中一行 await 都没有，事件循环拿不到
    执行权，因此「刚 record_trace 完就读」看到的必然是写库任务还没跑过的状态——
    aiosqlite 的读会 await，后台任务有机会抢跑，断言就不确定了。
    """
    with sqlite3.connect(str(db)) as conn:
        return conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]


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
    """定价表是**每 1M tokens** 的美元单价（与官方报价同刻度），别按每 1K 算。"""
    # 1M in / 0 out 的 claude-sonnet 正好是牌价本身：3.0 美元
    assert tracing.estimate_cost("anthropic", 1_000_000, 0) == pytest.approx(3.0)
    assert tracing.estimate_cost("anthropic", 0, 1_000_000) == pytest.approx(15.0)
    # gpt-4o-mini：0.15 + 0.60 每 1M
    assert tracing.estimate_cost("openai", 1_000_000, 1_000_000) == pytest.approx(0.75)
    # deepseek-chat：0.14 + 0.28 每 1M
    assert tracing.estimate_cost("deepseek", 1_000_000, 1_000_000) == pytest.approx(0.42)
    # 一次普通调用（千 token 量级）的成本在 1e-4 量级，按比例算、不取整
    assert tracing.estimate_cost("openai", 1000, 1000) == pytest.approx(0.00075)
    assert tracing.estimate_cost("openai", 500, 0) == pytest.approx(0.000075)
    assert tracing.estimate_cost("openai", 0, 0) == 0.0


def test_estimate_cost_uses_settings_prices(monkeypatch):
    """单价从 settings 读（不是写死的常量），改配置立即生效。"""
    monkeypatch.setattr(settings, "price_deepseek_input", 1.0)
    monkeypatch.setattr(settings, "price_deepseek_output", 2.0)

    # 1M in / 1M out：1.0 + 2.0（单位仍是每 1M tokens）
    assert tracing.estimate_cost("deepseek", 1_000_000, 1_000_000) == pytest.approx(3.0)


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
    # 大小写不敏感（m7）：配置里写 API.DeepSeek.com 也要认成 deepseek 档
    assert detect_provider("https://API.DeepSeek.COM/v1") == "deepseek"


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
    assert row["cost"] == pytest.approx(0.00075)  # (1000×0.15 + 1000×0.60) / 1M
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
    assert rows[0]["cost"] == pytest.approx((300 * 0.15 + 100 * 0.60) / 1_000_000)


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
    assert rows[0]["cost"] == pytest.approx(0.018)  # (1000×3.0 + 1000×15.0) / 1M


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
    assert rows[0]["cost"] == pytest.approx(0.00042)  # (1000×0.14 + 1000×0.28) / 1M


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


# ---------- 不阻塞主流程（fire-and-forget 的判别点） ----------


async def test_record_trace_returns_before_the_write_happens(db):
    """M2：record_trace 只是调度任务——返回时行还没落库，落库要等 drain。

    断言分两半，各自都有判别力：
    1. 返回后立刻用同步 sqlite3 数行（中间没有任何 await，后台任务拿不到执行权）
       必须是 0——若 record_trace 改成内部 await 写库，这里就是 1；
    2. 调用自身的耗时必须是微秒级——若里面插了 time.sleep(0.5) 这类阻塞，这里变红。
    """
    started = time.perf_counter()
    tracing.record_trace("llm", "openai/gpt-4o-mini", tokens_in=10, tokens_out=5)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"record_trace 阻塞了 {elapsed:.3f}s"
    assert count_sync(db) == 0  # 主流程已经走远，写库还没发生
    assert len(tracing._pending) == 1  # 任务挂着，等收尾

    await tracing.drain_traces()

    assert count_sync(db) == 1
    assert tracing._pending == set()


async def test_tracing_off_does_not_even_schedule_a_task(db, monkeypatch):
    """关掉开关时不建任务、不开连接：连 _pending 都不该动。"""
    monkeypatch.setattr(settings, "tracing_enabled", False)
    before = set(tracing._pending)

    tracing.record_trace("llm", "openai/gpt-4o-mini", tokens_in=10)

    assert tracing._pending == before
    assert count_sync(db) == 0


def test_record_trace_without_running_loop_only_warns(monkeypatch, caplog):
    """m1：同步上下文（无事件循环）误调时不能让 RuntimeError 穿透到业务调用。

    同步用例拿不到 db 夹具（那是 async 的），开关自己打开。
    """
    monkeypatch.setattr(settings, "tracing_enabled", True)
    before = set(tracing._pending)

    with caplog.at_level(logging.WARNING, logger="app.tracing"):
        tracing.record_trace("tool", "search_knowledge", detail="同步上下文误调")

    assert "无运行中的事件循环" in caplog.text
    assert tracing._pending == before


def test_drain_discards_tasks_left_by_other_event_loops(caplog):
    """m3：别的循环留下的任务在本循环里 wait/cancel 都会抛 RuntimeError。

    它属于已经关掉的循环，永远等不到结果——drain 应丢弃并告警，否则 _pending 里
    的这几条会污染之后每一次 drain。
    """
    other = asyncio.new_event_loop()
    leftover = other.create_task(asyncio.sleep(30))
    tracing._pending.add(leftover)
    try:
        with caplog.at_level(logging.WARNING, logger="app.tracing"):
            asyncio.run(tracing.drain_traces())

        assert "来自其他事件循环" in caplog.text
        assert leftover not in tracing._pending
    finally:
        # 兜底清理：断言失败（或实现回归）时别把这条外来任务留在 _pending 里，
        # 否则它会让后续每个 drain 都崩，把一个失败放大成一片
        tracing._pending.discard(leftover)
        leftover.cancel()
        try:
            other.run_until_complete(leftover)
        except asyncio.CancelledError:
            pass
        other.close()


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


async def test_run_agent_traces_tool_exception(db, monkeypatch):
    """m5：检索抛异常时 execute_tool 上抛、记不上，由 run_agent 的降级分支补记。

    否则检索故障只出现在 SSE 事件里，成本看板看不到这类失败。
    """

    async def boom(query, k=8, mode="hybrid", db_path=None):
        raise RuntimeError("向量库挂了")

    monkeypatch.setattr(runtime, "hybrid_search", boom)
    monkeypatch.setattr(runtime, "get_llm", lambda: ToolLoopLLM(tool_rounds=1))

    events = [e async for e in runtime.run_agent(SESSION, "什么是 RAG？", db)]
    await runtime.drain_memory_writes()
    await tracing.drain_traces()

    # 事件侧照旧降级，回答没有中断
    assert [e.type for e in events] == ["tool_start", "tool_end", "text_delta", "done"]
    rows = await read(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "tool"
    assert rows[0]["name"] == "search_knowledge"
    assert rows[0]["detail"] == events[1].data["summary"]  # 与 SSE 上的失败说明一致
    assert "向量库挂了" in rows[0]["detail"]


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
    # cost 按 per-Mtok 定价表估值：openai 100/20 → (100×0.15 + 20×0.60)/1M = 0.000027
    ("llm", "openai/gpt-4o-mini", "", 100, 20, 0.000027, "2026-09-25T10:00:00.000+00:00"),
    ("llm", "deepseek/deepseek-chat", "", 200, 40, 0.0000392, "2026-09-25T11:00:00.000+00:00"),
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


async def test_traces_endpoint_orders_by_id_not_ts(client, db):
    """排序键是 id（写入序），不是 ts：让两个键给出**相反**的顺序才区分得开。

    第 1 行先写入但 ts 更晚，第 2 行后写入但 ts 更早：
    ORDER BY id DESC → [2, 1]；ORDER BY ts DESC → [1, 2]。
    """
    await seed(
        db,
        [
            ("llm", "先写入·时间晚", "", 0, 0, 0.0, "2026-12-31T00:00:00.000+00:00"),
            ("llm", "后写入·时间早", "", 0, 0, 0.0, "2026-01-01T00:00:00.000+00:00"),
        ],
    )

    body = (await client.get("/api/traces")).json()

    assert [r["id"] for r in body["items"]] == [2, 1]
    assert [r["name"] for r in body["items"]] == ["后写入·时间早", "先写入·时间晚"]


async def test_traces_endpoint_accepts_any_timezone_offset(client, db):
    """M1：任意偏移与 Z 后缀都要归一化到同一瞬时，不能按原串比字典序。

    同一时刻的三种写法（UTC 毫秒串 / Z / +08:00）必须给出同一个结果：直接字符串比较
    时 '…T11:00:00Z' 既不 ≥ 也不 ≤ 行里的 '…T11:00:00.000+00:00'，会静默漏掉那条。
    """
    await seed(db, SEED)

    for start in (
        "2026-09-25T11:00:00.000+00:00",
        "2026-09-25T11:00:00Z",
        "2026-09-25T19:00:00+08:00",
        "2026-09-25T10:59:59.999+00:00",
    ):
        body = (await client.get("/api/traces", params={"start": start})).json()
        assert body["total"] == 2, f"{start} 漏掉了 11:00 那条"
        assert body["items"][-1]["name"] == "deepseek/deepseek-chat"

    # 边界闭区间：start / end 恰好等于某行的 ts 时该行必须在结果里
    window = (
        await client.get(
            "/api/traces",
            params={
                "start": "2026-09-25T19:00:00+08:00",  # = 11:00Z，该行本身
                "end": "2026-09-26T17:00:00+08:00",  # = 09-26 09:00Z，该行本身
            },
        )
    ).json()
    assert [r["name"] for r in window["items"]] == [
        "search_knowledge",
        "deepseek/deepseek-chat",
    ]

    # 恰好比 11:00 早 1 毫秒的 end：那条被排除，闭区间的另一半也确认了
    just_before = (
        await client.get(
            "/api/traces", params={"end": "2026-09-25T10:59:59.999+00:00"}
        )
    ).json()
    assert [r["name"] for r in just_before["items"]] == ["openai/gpt-4o-mini"]


async def test_traces_endpoint_rejects_bad_timestamps(client, db):
    """解析不了或没带时区的时间戳回 422，而不是静默当成「空结果」。"""
    await seed(db, SEED)

    for params in (
        {"start": "2026-09-25"},  # 只给日期：没有时间点，也没有时区
        {"start": "2026-09-25T10:00:00"},  # 有时间点但没时区，瞬时不确定
        {"end": "昨天"},
        {"end": "2026-13-45T00:00:00Z"},
    ):
        resp = await client.get("/api/traces", params=params)
        assert resp.status_code == 422, f"{params} 应回 422，实际 {resp.status_code}"
        assert "时间戳" in resp.json()["detail"]

    # 汇总接口同义
    summary = await client.get("/api/traces/summary", params={"start": "2026-09-25"})
    assert summary.status_code == 422

    # 合法值不受影响
    ok = await client.get("/api/traces", params={"start": "2026-09-25T00:00:00Z"})
    assert ok.status_code == 200


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
    assert body["cost"] == pytest.approx(0.0000662)

    # 分组按成本降序：llm（0.0000662）> tool（0）；同组内 tokens 累加
    assert [g["kind"] for g in body["by_kind"]] == ["llm", "tool"]
    assert body["by_kind"][0]["calls"] == 2
    assert body["by_kind"][0]["tokens_in"] == 300
    assert body["by_kind"][0]["cost"] == pytest.approx(0.0000662)
    assert body["by_kind"][1]["calls"] == 1 and body["by_kind"][1]["cost"] == 0.0

    assert [g["name"] for g in body["by_name"]] == [
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
        "search_knowledge",
    ]
    assert body["by_name"][0]["cost"] == pytest.approx(0.0000392)


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
    assert listing["items"][0]["cost"] == pytest.approx(0.00075)

    summary = (await client.get("/api/traces/summary")).json()
    assert summary["total_calls"] == 1
    assert summary["cost"] == pytest.approx(0.00075)
    assert summary["by_kind"] == [
        {
            "kind": "llm",
            "calls": 1,
            "tokens_in": 1000,
            "tokens_out": 1000,
            "cost": 0.00075,
        }
    ]


# ---------- 关停接线 ----------


async def test_lifespan_drains_memory_before_traces(monkeypatch):
    """关停顺序：先收记忆写入，再收 trace。

    反过来会丢掉记忆抽取那次 LLM 调用产生的 trace——它是记忆写入任务的后半段。
    """
    calls: list[str] = []

    async def fake_init_db() -> None:
        calls.append("init")

    async def fake_memory_drain() -> None:
        calls.append("memory")

    async def fake_trace_drain() -> None:
        calls.append("traces")

    monkeypatch.setattr(main_module, "init_db", fake_init_db)
    monkeypatch.setattr(main_module, "drain_memory_writes", fake_memory_drain)
    monkeypatch.setattr(main_module, "drain_traces", fake_trace_drain)

    async with main_module.lifespan(main_module.app):
        calls.append("serving")

    assert calls == ["init", "serving", "memory", "traces"]
