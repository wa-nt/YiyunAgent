"""T8 上下文治理（Context Governor）测试。

三个策略各自独立可开关，用例按「token 估算 / 压缩 / 工具结果清理 / token 预算 / 组合
顺序 / 接入点」分组。开关默认值由 tests/conftest.py 的 autouse 夹具统一钉住（不受 .env
与环境变量影响），用例要改哪个策略必须自己显式 monkeypatch，保证断言的是用例的意图而不是
环境残留。
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


class LeakySummaryLLM:
    """异常信息里带一大段内容（模拟 provider 把响应体/对话回显进异常）。"""

    SECRET_BODY = "响应体回显：" + "敏感内容" * 200

    async def chat(self, messages, tools=None):
        raise ValueError(f"400 Bad Request: {self.SECRET_BODY}")


class _SilentWriterLLM:
    """记忆抽取桩：治理用例不产生真实的抽取调用。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """tmp 数据库 + 隔离的 BM25 缓存。

    治理开关由 tests/conftest.py 的 autouse 夹具统一钉住（本文件不再重复）。
    """
    monkeypatch.setattr(runtime.settings, "db_path", str(tmp_path / "app.db"))
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
    assert ctx.estimate_tokens("中文" * 2) == 4  # CJK 按字符计，见下一条用例


def test_estimate_tokens_counts_cjk_per_character():
    """len/4 会把中文低估 3~4 倍：CJK 区间按 1 字符 ≈ 1 token。"""
    assert ctx.estimate_tokens("a" * 8 + "中文") == 4  # 非 CJK 部分仍按 4 字符/token
    assert ctx.estimate_tokens("中文abcd") == 3
    assert ctx.estimate_tokens("中文，。！") == 5  # CJK 标点同样按字符计


def test_prompt_tokens_counts_tool_calls():
    """两条消息的 content 完全相同，只差 tool_calls——忽略 tool_calls 时断言必红。"""
    plain = [assistant("同样的一段回答")]
    with_call = [
        Message(
            role="assistant",
            content="同样的一段回答",
            tool_calls=[
                ToolCall(id="c1", name="search_knowledge", arguments={"query": "RAG 检索"})
            ],
        )
    ]

    assert ctx.prompt_tokens(with_call) > ctx.prompt_tokens(plain)
    # 差额至少是工具名 + 参数的估算值，说明算的是 tool_calls 而不是正文
    assert ctx.prompt_tokens(with_call) - ctx.prompt_tokens(plain) >= (
        ctx.estimate_tokens("search_knowledge")
        + ctx.estimate_tokens('{"query": "RAG 检索"}')
    )


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


async def test_summary_input_truncation_keeps_the_tail():
    """m2：被压区间超长时送进摘要 prompt 的是尾部——紧邻保留窗口的条目更该留下。

    这也说明摘要是有损的：被截掉的开头不会以任何形式进入摘要。
    """
    llm = SummaryLLM()
    big = "长" * 300
    history_items = [
        user(f"问题{i}：{big}") if i % 2 == 0 else assistant(f"回答{i}：{big}")
        for i in range(12)
    ]
    # 被压区间的最后一条（紧邻保留窗口）以独一无二的标记结尾
    history_items[5] = assistant(big * 10 + "尾部标记TAIL")
    history_items[0] = user("开头标记HEAD：" + big * 10)
    view = runtime.assemble_messages(history_items, "新问题")

    await ctx.govern_context(view, llm=llm)

    transcript = llm.calls[0][1].content
    assert len(transcript) == ctx.SUMMARY_INPUT_LIMIT
    assert "尾部标记TAIL" in transcript  # 尾部保留：紧邻保留窗口的条目还在
    assert "开头标记HEAD" not in transcript  # 开头被丢掉（有损）
    assert transcript.endswith("尾部标记TAIL")  # 切的正是尾部窗口


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


async def test_summary_failure_log_truncates_exception_body(caplog):
    """m5：provider 异常的 str() 常带着响应体，日志只留类型 + 一小段。"""
    view = runtime.assemble_messages(history(12), "新问题")

    with caplog.at_level(logging.WARNING, logger="app.agent.context"):
        await ctx.govern_context(view, llm=LeakySummaryLLM())

    assert "ValueError" in caplog.text
    assert "响应体回显" in caplog.text  # 开头还在，够定位
    assert "敏感内容" * 50 not in caplog.text  # 整段响应体不落日志
    assert len(caplog.records[-1].getMessage()) < len(LeakySummaryLLM.SECRET_BODY)


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


async def test_existing_summary_guard_stops_repeated_compaction():
    """已有摘要 + 历史仍超阈值时直接返回：命中守卫，不再调摘要 LLM。

    这是守卫唯一有判别力的形态——若只复用上一轮的治理产物（历史已被压到阈值以下），
    拦住重复压缩的是阈值而不是守卫（见上一个用例，去掉守卫它照样绿）。
    """
    llm = SummaryLLM()
    memory = f"{RECALL_HEADER}\n- [fact] 用户在北京"
    summary = f"{ctx.SUMMARY_PREFIX} 早先的对话已压缩"
    view = [
        Message(role="system", content=runtime.SYSTEM_PROMPT),
        Message(role="system", content=memory),
        Message(role="system", content=summary),
        *history(12),  # 仍超过阈值 10，没有守卫就会再压一次
        user("新问题"),
    ]

    governed = await ctx.govern_context(view, llm=llm)

    assert governed == view
    assert llm.calls == []
    # 原摘要原样留着，没有被二次压缩掉的痕迹
    assert [m.content for m in governed if m.content == summary] == [summary]


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
    """工具结果找不到对应的调用（历史被裁剪过）时，用不带 query 的通用占位符。

    内容要长于通用占位符，否则按「不放大」的硬约束会保留原文（另有专门用例覆盖）。
    """
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = runtime.assemble_messages(
        [
            called("c1", "第1问"),
            result("missing", "第1轮的检索结果" * 5),
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


def _tool_view(*queries: str, content: str) -> list[Message]:
    """若干轮「调用 + 结果」的视图，工具结果用同一段 content 便于对比。

    末尾再补两轮「最近」的工具调用（= KEEP_TOOL_ROUNDS），让给定的这些轮次都落在
    清理范围内。
    """
    messages: list[Message] = [Message(role="system", content=runtime.SYSTEM_PROMPT)]
    for i, query in enumerate(queries):
        messages.append(called(f"c{i}", query))
        messages.append(result(f"c{i}", content))
    for i in range(ctx.KEEP_TOOL_ROUNDS):
        messages.append(called(f"recent{i}", f"最近的问题{i}"))
        messages.append(result(f"recent{i}", "最近的检索结果"))
    messages.append(user("新问题"))
    return messages


async def test_tool_cleanup_never_grows_the_prompt(monkeypatch):
    """C1：query 很长而结果很短时，占位符不得比原内容长（否则「清理」是放大）。

    无命中轮次的结果只有十几个字，query 却由模型自由生成、没有长度上限。
    """
    _governance_off(monkeypatch, budget=False, compaction=False)
    short = "未检索到相关内容"  # 8 字符，比通用占位符还短
    view = _tool_view("q" * 5000, short, "q" * 28, content=short)

    governed = await ctx.govern_context(view)

    assert ctx.prompt_tokens(governed) <= ctx.prompt_tokens(view)  # 治理不放大
    tools = [m.content for m in governed if m.role == "tool"]
    # 结果比任何占位符都短时保留原文：宁可不清，也不放大
    assert tools[0] == short and tools[1] == short and tools[2] == short


async def test_tool_cleanup_truncates_long_query_in_placeholder(monkeypatch):
    """query 超上限时截断内联，占位符仍然比原文短（结果够长时才换）。"""
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = _tool_view("q" * 5000, content="检索结果" * 200)

    tools = [m.content for m in await ctx.govern_context(view) if m.role == "tool"]

    assert tools[0] == f'[之前检索过 "{"q" * 40}…"，结果已省略]'
    assert len(tools[0]) < len("检索结果" * 200)


async def test_tool_cleanup_keeps_short_query_intact(monkeypatch):
    """query 在上限内时不截断，占位符保持可读的原始 query。"""
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = _tool_view(
        "RAG 是什么", "装饰器怎么写", "过拟合", content="检索结果" * 200
    )

    tools = [m.content for m in await ctx.govern_context(view) if m.role == "tool"]

    assert tools[0] == '[之前检索过 "RAG 是什么"，结果已省略]'
    assert tools[1] == '[之前检索过 "装饰器怎么写"，结果已省略]'
    assert tools[2] == '[之前检索过 "过拟合"，结果已省略]'


async def test_token_budget_does_not_replace_results_with_longer_queries(monkeypatch):
    """C1 连带后果：占位符更长时不换，预算不得因此去删摘要和记忆。"""
    _governance_off(monkeypatch, budget=True, compaction=False)
    memory = f"{RECALL_HEADER}\n- [fact] 用户在北京"
    view = _tool_view("长" * 100, "长" * 100, content="短结果") + [
        Message(role="system", content=memory)
    ]
    # 让记忆落在前缀里（真实链路的顺序是 system prompt → 记忆 → 历史 → 提问）
    view.insert(1, view.pop())

    governed = await ctx.govern_context(view, max_tokens=8000)

    assert memory in [m.content for m in governed]
    assert ctx.prompt_tokens(governed) <= 8000


async def test_token_budget_judges_tool_replacement_by_token_benefit(monkeypatch):
    """预算循环按 token 收益判定替换，而不是「内容变了没有」。

    _omitted 自己已保证占位符更短，这条判据是第二道闸：万一占位符被改成更长的版本
    （换措辞、加字段），只看内容变化就会越换越大，把预算越耗越紧（进而更早去删记忆）。

    用「变长的占位符」模拟：正确实现拒绝换（工具结果保持原样），旧实现照换。
    """
    _governance_off(monkeypatch, budget=True, compaction=False)
    memory = f"{RECALL_HEADER}\n- [fact] 用户在北京"
    view = runtime.assemble_messages([user("旧问题"), assistant("旧回答")], "新问题", memory)
    for i in (1, 2):
        view += [called(f"c{i}", f"第{i}问"), result(f"c{i}", "结" * 200)]
    budget = ctx.prompt_tokens(view) - 50  # 略超预算，必须靠预算策略动手
    assert ctx.prompt_tokens(view) > budget

    def bloated(message: Message, queries: dict[str, str]) -> Message:
        return message.model_copy(update={"content": "更长的占位符" * 200})

    monkeypatch.setattr(ctx, "_omitted", bloated)

    governed = await ctx.govern_context(view, max_tokens=budget)

    # 变长的占位符没被塞进去：工具结果保持原文（旧行为会把它们换成 653 token 的巨型占位符）
    assert not any(m.content.startswith("更长的占位符") for m in governed)
    assert "结" * 200 in [m.content for m in governed]
    # 治理没往相反方向加 token：结果不超过「超预算的入参」
    assert ctx.prompt_tokens(governed) <= ctx.prompt_tokens(view)


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


async def test_token_budget_never_exceeds_limit_including_note(monkeypatch):
    """M1：截断提示本身也占预算，最终结果（含提示）不得超 max_tokens。

    构造每轮约 81 token 的历史（CJK 按 1 字符 ≈ 1 token）：旧实现按 max_tokens 判定、
    提示最后才追加，会卡在「刚好不超预算」的档位上再被提示顶出去。预留提示的预算后，
    判定目标低一截，会多删一轮。下界取「system prompt + 提示 + 当前提问」的估算值，
    比它更小的档位本来就放不下必需消息（另有不可再压下限的专门用例）。
    """
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = [Message(role="system", content="SYS")]
    for i in range(5):
        view.append(user(f"问题{i}：" + "长" * 77))  # 每轮 81 token
    view.append(user("新问题"))

    floor = ctx.prompt_tokens(
        [view[0], Message(role="system", content=ctx.TRUNCATION_NOTE), view[-1]]
    )
    for budget in range(floor + 1, floor + 80):
        governed = await ctx.govern_context(list(view), max_tokens=budget)
        assert ctx.prompt_tokens(governed) <= budget, f"budget={budget} 超预算"

    # 提示仍在：这条路径上确实删过内容，不能因为预留预算就把提示弄丢
    tight = await ctx.govern_context(list(view), max_tokens=floor + 66)
    assert any(m.content == ctx.TRUNCATION_NOTE for m in tight)


async def test_truncation_note_is_not_accumulated(monkeypatch):
    """m1：治理反复调用（run_agent 每轮都治理）不得把提示越堆越多。"""
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = [Message(role="system", content="SYS"), user("长" * 400), user("新问题")]

    governed = await ctx.govern_context(view, max_tokens=60)
    for _ in range(3):
        # 模拟 Agent 循环：每轮追加一条工具结果后重新治理
        governed = [
            *governed,
            called("c1", "问"),
            result("c1", "检索结果" * 100),
        ]
        governed = await ctx.govern_context(governed, max_tokens=60)
        notes = [m for m in governed if m.content == ctx.TRUNCATION_NOTE]
        assert len(notes) <= 1, f"提示累积到 {len(notes)} 条"


async def test_token_budget_has_an_irreducible_floor(monkeypatch):
    """预算小到放不下「system prompt + 提示 + 当前提问」时，保留必需消息而不是删光。

    这是诚实的做不到：删掉当前提问就没得回答了。用例锁住这一档的行为，免得
    后续改动把它悄悄变成「删空视图」或「无限循环」。
    """
    _governance_off(monkeypatch, budget=True, compaction=False)
    view = [Message(role="system", content="SYS"), user("长" * 400), user("新问题")]

    governed = await ctx.govern_context(list(view), max_tokens=1)

    assert governed[0].content == "SYS"  # system prompt 留着（工具契约）
    assert governed[-1].content == "新问题"  # 当前提问留着（否则无可答）
    assert "长" not in "".join(m.content for m in governed)  # 超长的历史删干净了
    assert any(m.content == ctx.TRUNCATION_NOTE for m in governed)  # 有截断就如实注明


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

    # 逐轮检查：每一轮要么问答两条都在、要么都不在——逐条删（劈开轮次）会在这里变红
    for i in range(6):
        assert (f"问题{i}" in heads) == (f"回答{i}" in heads), f"第 {i} 轮被劈开了"
    kept_rounds = [i for i in range(6) if f"问题{i}" in heads]
    assert kept_rounds == list(range(kept_rounds[0], 6))  # 保留的是连续的最后几轮
    assert kept_rounds and kept_rounds[0] > 0  # 确实删掉了最早的一批


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


async def test_history_span_is_empty_without_user_message(caplog):
    """S2：没有 user 消息（调用方自拼视图）时历史区间为空，绝不动这些消息。

    正常链路不会出现——assemble_messages 总把当前提问放末尾。没有这个锚点就分不清
    哪段是历史、哪段是本回合，此时宁可不删。
    """
    view = [
        Message(role="system", content=runtime.SYSTEM_PROMPT),
        assistant("回答：" + "长" * 300),
        result("c1", "检索结果" * 200),
    ]

    assert ctx._history_span(view) == (1, 1)

    governed = await ctx.govern_context(view, max_tokens=5)

    # 预算超了但没有可删的历史区间，且这些消息都不是摘要/记忆——原样返回
    assert governed == view


async def test_governed_result_is_always_a_new_list(monkeypatch):
    """S3：即使内容没变也返回新 list，调用方不该拿到与入参同一对象。"""
    _governance_off(monkeypatch, budget=False, compaction=False)
    view = runtime.assemble_messages([user("旧问题")], "新问题")

    governed = await ctx.govern_context(view)

    assert governed == view and governed is not view


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
