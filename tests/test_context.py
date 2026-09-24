"""T8 上下文治理（Context Governor）测试。

三个策略各自独立可开关，用例按「token 估算 / 压缩 / 工具结果清理 / token 预算 / 组合
顺序 / 接入点」分组。每个用例都用 monkeypatch 显式设定开关，不依赖 .env，也不把状态
漏给下一个用例（裸 settings.x = y 会污染，消融开关必须逐例隔离）。
"""

import logging

import pytest

from app.agent import context as ctx
from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, Message, StreamChunk, ToolCall
from app.memory import writer as memory_writer
from app.memory.recall import RECALL_HEADER
from app.retrieval.bm25_search import invalidate
from app.retrieval.types import RetrievedChunk

DIM = 8
SESSION = "s-ctx"
SUMMARY = "用户之前问过关于 X、Y、Z 的问题，助手回答了相关结论。"


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str) -> Message:
    return Message(role="assistant", content=text)


def called(call_id: str, query: str) -> Message:
    """带工具调用的助手消息（query 是占位符要保留的信息来源）。"""
    return Message(
        role="assistant",
        content=f"[调用工具 search_knowledge {query}]",
        tool_calls=[
            ToolCall(id=call_id, name="search_knowledge", arguments={"query": query})
        ],
    )


def result(call_id: str, content: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, content=content)


def tool_round(index: int) -> list[Message]:
    """第 index 轮的「调用 + 结果」两条消息。"""
    return [
        called(f"c{index}", f"第{index}问"),
        result(f"c{index}", f"第{index}轮的检索结果" * 10),
    ]


def history(n: int) -> list[Message]:
    """交替的 user/assistant 历史：第 i 条按 i 的奇偶决定角色。"""
    return [
        user(f"问题{i}") if i % 2 == 0 else assistant(f"回答{i}") for i in range(n)
    ]


class SummaryLLM:
    """压缩用的 LLM 桩：记录收到的 prompt，返回固定摘要。"""

    def __init__(self, text: str = SUMMARY):
        self.text = text
        self.calls: list[list[Message]] = []

    async def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        return ChatResult(text=self.text)


class BoomSummaryLLM:
    async def chat(self, messages, tools=None):
        raise RuntimeError("摘要服务挂了")


class _SilentWriterLLM:
    """记忆抽取桩：治理用例不产生真实的抽取调用。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """tmp 数据库 + 隔离的 BM25 缓存；治理参数按 brief 默认值显式固定。"""
    monkeypatch.setattr(runtime.settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "context_compaction_enabled", True)
    monkeypatch.setattr(settings, "context_tool_clean_enabled", True)
    monkeypatch.setattr(settings, "context_token_budget_enabled", True)
    monkeypatch.setattr(settings, "context_max_tokens", 8000)
    monkeypatch.setattr(settings, "context_compaction_threshold", 10)
    monkeypatch.setattr(settings, "memory_enabled", True)
    invalidate()
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()
    invalidate()


@pytest.fixture
async def seeded_db(db, monkeypatch):
    """13 条历史（超过默认阈值 10）落在库里，记忆写入换成空抽取桩。"""
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    await seed_history(db, 13, "已落库的历史")
    return db


async def seed_history(db, count: int, prefix: str) -> None:
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (SESSION, "2026-09-24T10:00:00"),
        )
        for i in range(count):
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    SESSION,
                    "user" if i % 2 == 0 else "assistant",
                    f"{prefix}{i}",
                    "2026-09-24T10:00:00",
                ),
            )
        await conn.commit()


def fake_search(monkeypatch, content: str = "检索到的内容" * 50) -> list[str]:
    """替换 hybrid_search，记录收到的 query 并返回固定结果。"""
    queries: list[str] = []

    async def fake(query, k=8, mode="hybrid", db_path=None):
        queries.append(query)
        return [
            RetrievedChunk(
                chunk_id=1, doc_id=1, content=content, title="笔记", score=1.0
            )
        ]

    monkeypatch.setattr(runtime, "hybrid_search", fake)
    return queries


# ---------- token 估算 ----------


def test_estimate_tokens_is_char_based():
    assert ctx.estimate_tokens("") == 0
    assert ctx.estimate_tokens("abcd") == 1
    assert ctx.estimate_tokens("a" * 9) == 3  # 向上取整，不低估预算
    assert ctx.estimate_tokens("中文" * 2) == 1


def test_prompt_tokens_counts_tool_calls():
    plain = [user("你好")]
    with_call = [called("c1", "RAG 检索")]

    assert ctx.prompt_tokens(plain) < ctx.prompt_tokens(with_call)
    assert ctx.prompt_tokens(with_call) > ctx.estimate_tokens("RAG 检索")


# ---------- 历史压缩 ----------


async def test_compaction_replaces_early_history_with_summary():
    llm = SummaryLLM()
    view = runtime.assemble_messages(history(12), "新问题")

    governed = await ctx.govern_context(view, llm=llm)

    # 12 条历史 > 阈值 10 → 触发压缩；摘要以 system 消息插入，带 [历史摘要] 前缀
    summaries = [m for m in governed if m.content.startswith(ctx.SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert SUMMARY in summaries[0].content
    assert len(llm.calls) == 1
    assert llm.calls[0][0].content == ctx.SUMMARY_PROMPT
    # 送进摘要的只有被压掉的那一段，最近保留的完整对话不在里面
    assert "问题0" in llm.calls[0][1].content
    assert "问题10" not in llm.calls[0][1].content
    # 被压掉的消息已从 prompt view 里消失，最近的消息原样保留
    contents = [m.content for m in governed]
    assert "问题0" not in contents
    assert "问题10" in contents


async def test_compaction_keeps_recent_threshold_half():
    llm = SummaryLLM()
    # 12 条 = 6 轮问答，阈值 10 → 保留最近 5 条，边界前贴到轮次起点，不劈开一轮
    view = runtime.assemble_messages(history(12), "新问题")

    governed = await ctx.govern_context(view, llm=llm)

    kept = [m.content for m in governed]
    assert "问题4" not in kept and "回答5" not in kept  # 第 3 轮被压掉
    assert "问题6" in kept and "回答7" in kept  # 第 4 轮起完整保留
    # 保留 6 条历史 + 末尾的当前提问（用户侧与助手侧都不缺）
    assert len([m for m in governed if m.role in ("user", "assistant")]) == 7


async def test_compaction_inserts_summary_after_memory():
    llm = SummaryLLM()
    memory = f"{RECALL_HEADER}\n- [fact] 用户在北京"
    view = runtime.assemble_messages(history(12), "新问题", memory)

    governed = await ctx.govern_context(view, llm=llm)

    assert [m.role for m in governed[:4]] == ["system", "system", "system", "user"]
    assert governed[0].content == runtime.SYSTEM_PROMPT
    assert governed[1].content == memory
    assert governed[2].content.startswith(ctx.SUMMARY_PREFIX)
    assert governed[-1].content == "新问题"


async def test_compaction_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "context_compaction_enabled", False)
    llm = SummaryLLM()
    view = runtime.assemble_messages(history(12), "新问题")

    governed = await ctx.govern_context(view, llm=llm)

    assert governed == view
    assert llm.calls == []  # 关掉压缩就不该调摘要 LLM


async def test_compaction_triggers_only_above_threshold(monkeypatch):
    monkeypatch.setattr(settings, "context_compaction_threshold", 10)
    llm = SummaryLLM()

    at_threshold = runtime.assemble_messages(history(10), "新问题")
    assert await ctx.govern_context(at_threshold, llm=llm) == at_threshold

    above = runtime.assemble_messages(history(11), "新问题")
    assert await ctx.govern_context(above, llm=llm) != above
    assert len(llm.calls) == 1  # 只有超阈值那一次真的调了 LLM


async def test_compaction_degrades_when_summary_llm_fails(caplog):
    view = runtime.assemble_messages(history(12), "新问题")

    with caplog.at_level(logging.WARNING, logger="app.agent.context"):
        governed = await ctx.govern_context(view, llm=BoomSummaryLLM())

    assert governed == view  # 摘要失败退化为保留完整历史，不拖垮本轮
    assert "历史压缩失败" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


async def test_compaction_does_not_mutate_input():
    llm = SummaryLLM()
    built = history(12)
    view = runtime.assemble_messages(built, "新问题")
    before = [m.model_copy(deep=True) for m in view]

    governed = await ctx.govern_context(view, llm=llm)

    assert governed is not view
    assert view == before  # 原始视图（及其引用的历史）一个字段都没改
    assert built == history(12)


async def test_existing_summary_is_not_compacted_again():
    llm = SummaryLLM()
    view = runtime.assemble_messages(history(12), "新问题")

    once = await ctx.govern_context(view, llm=llm)
    twice = await ctx.govern_context(once, llm=llm)

    assert twice == once
    assert len(llm.calls) == 1  # 已有摘要，第二轮不再重复调 LLM


# ---------- 工具结果清理 ----------


async def test_tool_cleanup_replaces_old_tool_results(monkeypatch):
    _governance_off(monkeypatch, budget=False, compaction=False)
    built = [*tool_round(1), *tool_round(2), *tool_round(3)]
    view = runtime.assemble_messages(built, "新问题")
    before = [m.model_copy(deep=True) for m in view]

    governed = await ctx.govern_context(view)

    tools = [m for m in governed if m.role == "tool"]
    # 3 轮工具调用，保留最近 2 轮 → 只有第 1 轮被换成占位符（带原始 query）
    assert tools[0].content == '[之前检索过 "第1问"，结果已省略]'
    assert "第2轮的检索结果" in tools[1].content
    assert "第3轮的检索结果" in tools[2].content
    # tool_call_id 不变：Anthropic 协议靠它配对 tool_use
    assert [t.tool_call_id for t in tools] == ["c1", "c2", "c3"]
    # 不原地改：入参视图与它引用的历史都保持原文（换内容要产出新对象）
    assert view == before
    assert built == [*tool_round(1), *tool_round(2), *tool_round(3)]


async def test_tool_cleanup_keeps_last_two_rounds(monkeypatch):
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = runtime.assemble_messages(
        [*tool_round(1), *tool_round(2), *tool_round(3), *tool_round(4)], "新问题"
    )

    governed = await ctx.govern_context(view)

    contents = [m.content for m in governed if m.role == "tool"]
    assert "第1轮的检索结果" not in contents[0] and "第2轮的检索结果" not in contents[1]
    assert "第3轮的检索结果" in contents[2]
    assert "第4轮的检索结果" in contents[3]


async def test_tool_cleanup_skipped_when_disabled(monkeypatch):
    _governance_off(monkeypatch)
    monkeypatch.setattr(settings, "context_tool_clean_enabled", False)
    view = runtime.assemble_messages([*tool_round(1), *tool_round(2), *tool_round(3)], "新问题")

    governed = await ctx.govern_context(view)

    assert governed == view
    tools = [m for m in governed if m.role == "tool"]
    assert "第1轮的检索结果" in tools[0].content


async def test_tool_cleanup_falls_back_to_generic_placeholder(monkeypatch):
    """工具结果找不到对应的调用（历史被裁剪过）时，用不带 query 的通用占位符。"""
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = runtime.assemble_messages(
        [
            called("c1", "第1问"),
            result("missing", "第1轮的检索结果"),
            *tool_round(2),
            *tool_round(3),
        ],
        "新问题",
    )

    governed = await ctx.govern_context(view)

    assert [m.content for m in governed if m.role == "tool"][0] == ctx.TOOL_OMITTED


async def test_tool_cleanup_ignores_current_round(monkeypatch):
    """当前轮刚产生的工具结果模型还要看，不能被清理（只有一轮时纯 no-op）。"""
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = runtime.assemble_messages(tool_round(1), "新问题")

    assert await ctx.govern_context(view) == view


def _governance_off(monkeypatch, budget: bool = True, compaction: bool = True) -> None:
    """关掉与本组断言无关的策略，避免别的策略顺手改掉被测行为。"""
    monkeypatch.setattr(settings, "context_token_budget_enabled", budget)
    monkeypatch.setattr(settings, "context_compaction_enabled", compaction)


# ---------- token 预算 ----------


def long_view(rounds: int = 6, chars: int = 800) -> list[Message]:
    """每轮问答都带可区分的编号，便于断言「删的是最早的轮次」。"""
    messages: list[Message] = []
    for i in range(rounds):
        messages.append(user(f"问题{i}：" + "长" * chars))
        messages.append(assistant(f"回答{i}：" + "长" * chars))
    return runtime.assemble_messages(messages, "新问题")


async def test_token_budget_truncates_within_budget(monkeypatch):
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = long_view()
    assert ctx.prompt_tokens(view) > 200

    governed = await ctx.govern_context(view, max_tokens=200)

    assert ctx.prompt_tokens(governed) <= 200
    # 丢过内容就注明，免得模型把残缺的历史当成全部事实
    assert any(m.content == ctx.TRUNCATION_NOTE for m in governed)
    assert governed[-1].content == "新问题"
    assert not any("长" in m.content for m in governed if m.role != "system")


async def test_token_budget_skipped_when_disabled(monkeypatch):
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = long_view()

    governed = await ctx.govern_context(view, max_tokens=200)

    assert governed == view


async def test_token_budget_noop_when_within_budget(monkeypatch):
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = long_view(rounds=1, chars=40)

    assert await ctx.govern_context(view, max_tokens=8000) == view


async def test_token_budget_drops_earliest_rounds_first(monkeypatch):
    """优先级 1：历史整轮从最早开始删，且不把一轮问答劈开（用户侧与助手侧成对消失）。"""
    _governance_off(monkeypatch, budget=True, compaction=False)
    monkeypatch.setattr(settings, "context_tool_clean_enabled", False)
    view = long_view()
    budget = ctx.prompt_tokens(view) - 900

    governed = await ctx.govern_context(view, max_tokens=budget)

    assert ctx.prompt_tokens(governed) <= budget
    heads = [m.content[:3] for m in governed if m.role in ("user", "assistant")]
    assert "问题0" not in heads and "回答0" not in heads  # 最早一轮成对消失
    assert "问题5" in heads  # 最近一轮留着
    # 剩下的历史是原历史的连续后缀（删的是开头，没掏空中间）
    original = [m.content[:3] for m in view if m.role in ("user", "assistant")]
    assert heads == original[len(original) - len(heads) :]


async def test_token_budget_replaces_tool_results_before_dropping_memory(monkeypatch):
    """优先级 2：历史删完仍超预算时，把工具结果换成占位符；记忆（优先级 4）保住。

    只用 2 轮工具调用（工具清理策略会完整保留它们），所以出现的占位符只能是
    token 预算删出来的。
    """
    _governance_off(monkeypatch, budget=True, compaction=False)
    memory = f"{RECALL_HEADER}\n- [fact] 用户偏好用 Markdown 记笔记"
    view = runtime.assemble_messages([user("旧问题"), assistant("旧回答")], "新问题", memory)
    # 本轮刚检索到的结果按真实链路追加在当前提问之后
    for i in (1, 2):
        view += [called(f"c{i}", f"第{i}问"), result(f"c{i}", f"第{i}轮" * 2000)]
    budget = 600

    governed = await ctx.govern_context(view, max_tokens=budget)

    tools = [m for m in governed if m.role == "tool"]
    assert [t.content for t in tools] == [
        '[之前检索过 "第1问"，结果已省略]',
        '[之前检索过 "第2问"，结果已省略]',
    ]
    assert memory in [m.content for m in governed]  # 记忆是最后手段，不该被动
    assert ctx.prompt_tokens(governed) <= budget


async def test_token_budget_drops_summary_as_last_resort(monkeypatch):
    """优先级 3：摘要比记忆先丢（记忆是跨会话画像，摘要只是被压过的历史）。"""
    _governance_off(monkeypatch, budget=True, compaction=False)
    memory = f"{RECALL_HEADER}\n- [fact] 用户在北京"
    summary = f"{ctx.SUMMARY_PREFIX} " + "被压过的历史" * 60
    view = [
        Message(role="system", content=runtime.SYSTEM_PROMPT),
        Message(role="system", content=memory),
        Message(role="system", content=summary),
        user("新问题"),
    ]
    budget = ctx.prompt_tokens(view) - 50

    governed = await ctx.govern_context(view, max_tokens=budget)

    contents = [m.content for m in governed]
    assert summary not in contents
    assert memory in contents
    assert ctx.prompt_tokens(governed) <= budget


async def test_token_budget_drops_memory_only_as_last_resort(monkeypatch):
    _governance_off(monkeypatch, budget=True, compaction=False)
    memory = f"{RECALL_HEADER}\n- [fact] " + "用户在北京" * 40
    view = runtime.assemble_messages([], "新问题", memory)

    governed = await ctx.govern_context(view, max_tokens=40)

    assert memory not in [m.content for m in governed]
    assert governed[0].content == runtime.SYSTEM_PROMPT
    assert governed[1].content == ctx.TRUNCATION_NOTE


async def test_token_budget_note_not_added_when_nothing_dropped(monkeypatch):
    """视图本身完整（没有可丢的内容）时不谎报截断。"""
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = runtime.assemble_messages([], "新问题" + "长" * 400)

    governed = await ctx.govern_context(view, max_tokens=10)

    assert governed == view
    assert ctx.TRUNCATION_NOTE not in [m.content for m in governed]


# ---------- 策略开关独立性与执行顺序 ----------


async def test_strategies_are_independently_toggleable(monkeypatch):
    """只关压缩时，工具清理与 token 预算照常生效（T10 单因素对照的前提）。"""
    monkeypatch.setattr(settings, "context_compaction_enabled", False)
    monkeypatch.setattr(settings, "context_tool_clean_enabled", True)
    monkeypatch.setattr(settings, "context_token_budget_enabled", True)
    llm = SummaryLLM()
    view = runtime.assemble_messages(
        [*tool_round(1), *tool_round(2), *tool_round(3), *history(12)], "新问题"
    )

    governed = await ctx.govern_context(view, max_tokens=400, llm=llm)

    assert llm.calls == []  # 压缩关着，没调摘要
    assert "第1轮的检索结果" not in [m.content for m in governed if m.role == "tool"]
    assert ctx.prompt_tokens(governed) <= 400  # 预算仍然生效


async def test_governance_runs_strategies_in_order(monkeypatch):
    """清理 → 压缩 → 预算：每一步都拿到上一步的产物，顺序即优先级。"""
    seen: list[tuple[str, list[Message]]] = []
    view = runtime.assemble_messages([user("旧问题")], "新问题")

    def clean(messages):
        seen.append(("clean", messages))
        return [*messages, Message(role="system", content="clean")]

    async def compact(messages, llm):
        seen.append(("compact", messages))
        return [*messages, Message(role="system", content="compact")]

    def budget(messages, max_tokens):
        seen.append(("budget", messages))
        assert max_tokens == 123
        return messages

    monkeypatch.setattr(ctx, "_clean_tool_results", clean)
    monkeypatch.setattr(ctx, "_compact_history", compact)
    monkeypatch.setattr(ctx, "_apply_token_budget", budget)

    await ctx.govern_context(view, max_tokens=123)

    assert [name for name, _ in seen] == ["clean", "compact", "budget"]
    assert seen[0][1] == view
    assert seen[1][1][-1].content == "clean"
    assert seen[2][1][-1].content == "compact"


async def test_governance_order_is_clean_then_compact(monkeypatch):
    """清理先于压缩：摘要 prompt 里看到的已经是占位符，而不是旧工具结果的全文。"""
    monkeypatch.setattr(settings, "context_compaction_threshold", 4)
    llm = SummaryLLM()
    view = runtime.assemble_messages(
        [*tool_round(1), *tool_round(2), *tool_round(3), user("问题4")], "新问题"
    )

    governed = await ctx.govern_context(view, llm=llm)

    transcript = llm.calls[0][1].content
    assert '[之前检索过 "第1问"，结果已省略]' in transcript
    assert "第1轮的检索结果第1轮的检索结果" not in transcript
    assert "第2轮的检索结果" in transcript  # 最近 2 轮不清理
    assert any(m.content.startswith(ctx.SUMMARY_PREFIX) for m in governed)


async def test_governance_edge_cases(monkeypatch):
    assert await ctx.govern_context([]) == []  # 空视图
    single = runtime.assemble_messages([], "只有一问")
    assert await ctx.govern_context(single, llm=SummaryLLM()) == single  # 无历史无工具
    with_one = runtime.assemble_messages([user("旧问题")], "新问题")
    assert await ctx.govern_context(with_one, llm=SummaryLLM()) == with_one
    no_tools = runtime.assemble_messages(history(12), "新问题")
    assert not any(m.role == "tool" for m in await ctx.govern_context(no_tools))


# ---------- 接入点（app/agent/runtime.py） ----------


class StreamLLM:
    """run_agent 桩：chat 返回摘要，chat_stream 直接给答案。"""

    def __init__(self, summary: str = SUMMARY):
        self.summary = summary
        self.chats: list[list[Message]] = []
        self.streams: list[list[Message]] = []

    async def chat(self, messages, tools=None):
        self.chats.append(list(messages))
        return ChatResult(text=self.summary)

    async def chat_stream(self, messages, tools=None):
        self.streams.append(list(messages))
        yield StreamChunk(text_delta="好的")
        yield StreamChunk(finish=True, tool_calls=[])


class ToolLoopLLM(StreamLLM):
    """前 tool_rounds 轮都请求工具，之后给一个纯文本答案。"""

    def __init__(self, tool_rounds: int, summary: str = SUMMARY):
        super().__init__(summary)
        self.tool_rounds = tool_rounds

    async def chat_stream(self, messages, tools=None):
        self.streams.append(list(messages))
        if len(self.streams) <= self.tool_rounds:
            index = len(self.streams)
            yield StreamChunk(
                finish=True,
                tool_calls=[
                    ToolCall(
                        id=f"c{index}",
                        name="search_knowledge",
                        arguments={"query": f"第{index}问"},
                    )
                ],
            )
            return
        yield StreamChunk(text_delta="答")
        yield StreamChunk(finish=True, tool_calls=[])


async def test_run_agent_governs_prompt_but_leaves_db_intact(seeded_db, monkeypatch):
    llm = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [event async for event in runtime.run_agent(SESSION, "新问题", seeded_db)]
    await runtime.drain_memory_writes()

    sent = llm.streams[0]
    assert any(m.content.startswith(ctx.SUMMARY_PREFIX) for m in sent)
    assert "已落库的历史0" not in [m.content for m in sent]
    assert sent[-1].content == "新问题"

    # 数据库里的原始历史一条不少：治理只改 prompt view，不做持久化压缩
    async with get_db(seeded_db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY id", (SESSION,)
        )
    contents = [r["content"] for r in rows]
    assert contents[:13] == [f"已落库的历史{i}" for i in range(13)]
    assert contents[-2:] == ["新问题", "好的"]


async def test_run_agent_cleans_old_tool_results_across_rounds(seeded_db, monkeypatch):
    """同一轮里第 3 次生成时，第 1 轮的检索结果已换成占位符，最近 2 轮保留。"""
    fake_search(monkeypatch)
    llm = ToolLoopLLM(tool_rounds=3)
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [event async for event in runtime.run_agent(SESSION, "新问题", seeded_db)]
    await runtime.drain_memory_writes()

    sent = llm.streams[-1]  # 第 4 次调用：3 轮工具之后生成答案
    tools = [m for m in sent if m.role == "tool"]
    assert len(tools) == 3
    assert tools[0].content == '[之前检索过 "第1问"，结果已省略]'
    assert "检索到的内容" in tools[1].content and "检索到的内容" in tools[2].content

    # 治理不写库：库里仍然只有本轮的 user/assistant 两行（检索结果从不落库）
    async with get_db(seeded_db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT role FROM messages WHERE session_id = ? ORDER BY id", (SESSION,)
        )
    assert [r["role"] for r in rows][-2:] == ["user", "assistant"]


async def test_run_agent_summarizes_history_once_per_turn(seeded_db, monkeypatch):
    """一轮里多次调模型（多轮工具）只生成一次摘要，不重复烧 LLM 调用。"""
    fake_search(monkeypatch)
    llm = ToolLoopLLM(tool_rounds=3)
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [event async for event in runtime.run_agent(SESSION, "新问题", seeded_db)]
    await runtime.drain_memory_writes()

    assert len(llm.streams) == 4  # 3 轮工具 + 1 轮收尾生成
    assert len(llm.chats) == 1  # 摘要只调一次
    assert llm.chats[0][0].content == ctx.SUMMARY_PROMPT
    # 后续每次生成都带上摘要
    assert all(
        any(m.content.startswith(ctx.SUMMARY_PREFIX) for m in sent)
        for sent in llm.streams
    )


async def test_run_agent_without_governance_sends_history_untouched(db, monkeypatch):
    """三个策略全关 = T5 的行为：prompt 里就是原始历史，也不额外调 LLM。"""
    await seed_history(db, 13, "已落库的历史")
    monkeypatch.setattr(settings, "context_compaction_enabled", False)
    monkeypatch.setattr(settings, "context_tool_clean_enabled", False)
    monkeypatch.setattr(settings, "context_token_budget_enabled", False)
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    llm = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [event async for event in runtime.run_agent(SESSION, "新问题", db)]
    await runtime.drain_memory_writes()

    sent = llm.streams[0]
    assert "已落库的历史0" in [m.content for m in sent]
    assert not any(m.content.startswith(ctx.SUMMARY_PREFIX) for m in sent)
    assert llm.chats == []
