"""T8 上下文治理（Context Governor）。

三种策略各自独立可开关（T10 消融实验用），按固定顺序作用在「发给模型的 prompt view」上：

    工具结果清理 → 历史压缩 → token 预算截断

治理只读原始历史、只产出新的 Message 对象：数据库里的 messages 表不动，调用方传进来的
Message 也不被修改，所以治理后重新加载历史就能拿到完整原文（不做持久化压缩）。

压缩要调 LLM 生成摘要，因此本模块的入口是异步的；`assemble_messages` 保持同步（T7 的
记忆注入在那里），由 run_agent 在每次调用模型前对组装好的视图走一遍治理。
"""

from __future__ import annotations

import json
import logging

from app.config import settings
from app.llm import get_llm
from app.llm.types import LLMClient, Message
from app.memory.recall import RECALL_HEADER

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "[历史摘要]"
TOOL_OMITTED = "[之前的工具结果已省略]"
TRUNCATION_NOTE = "[上下文已截断，部分历史可能丢失]"

# 简单字符估算：1 token ≈ 4 字符。这是偏乐观的口径，会**低估**实际占用：中文约 1~1.5
# 字符/token，也就是同样一段中文的真实 token 数约为本估算的 3~4 倍。用它把关够挡住上下文
# 爆炸，但不等于真实计费口径；不引 tiktoken 是为了不新增依赖，见 T8 brief 的边界说明
CHARS_PER_TOKEN = 4
KEEP_TOOL_ROUNDS = 2  # 保留最近 2 轮的工具结果，更早的换成占位符
SUMMARY_INPUT_LIMIT = 4000  # 送进摘要 prompt 的历史文本上限（字符）
# 占位符内联 query 时的上限：query 是模型自由生成的、没有长度上限，原样塞进占位符会让
# 「清理」反而放大 prompt（见 _omitted）
MAX_QUERY_IN_PLACEHOLDER = 40
# 异常日志里保留的信息长度：provider 异常的 str() 常带着响应体（可能回显对话内容），
# 只留类型 + 一小段，够定位问题即可（与 T7 敏感信息日志的口径一致）
ERROR_TEXT_LIMIT = 200
MIN_COMPACTION_THRESHOLD = 2  # 阈值下限：低于 2 条时压缩会每轮空转

SUMMARY_PROMPT = (
    "你是对话历史的压缩器。把给定的历史对话压缩成一段简洁的中文摘要："
    "保留用户问过的话题、已经确认的结论与事实，丢掉寒暄和过程性内容。"
    "参考格式：用户之前问过关于 X、Y、Z 的问题，助手回答了……。"
    "只输出摘要正文，不要输出 JSON、代码块标记或解释。"
)


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：1 token ≈ 4 字符（中文按字符数近似，够用于预算把关）。"""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def prompt_tokens(messages: list[Message]) -> int:
    """估算整个 prompt view 的 token 数（正文 + 工具调用名与参数）。"""
    total = 0
    for m in messages:
        total += estimate_tokens(m.content)
        for call in m.tool_calls or []:
            total += estimate_tokens(call.name)
            total += estimate_tokens(json.dumps(call.arguments, ensure_ascii=False))
    return total


async def govern_context(
    messages: list[Message],
    max_tokens: int | None = None,
    llm: LLMClient | None = None,
) -> list[Message]:
    """治理 prompt view：工具结果清理 → 历史压缩 → token 预算截断。

    返回的总是新的 list；内容没变时 Message 对象与入参共享（本模块从不原地改 Message）。
    max_tokens 缺省取 settings.context_max_tokens；三个策略各自读自己的开关，关掉的
    策略在这一步被完全跳过，互不影响（消融实验依赖这一点）。llm 只在压缩时用得上，
    缺省取 app.llm.get_llm()；run_agent 传入本轮实例，测试可注入桩。
    """
    if not messages:
        return list(messages)
    budget = settings.context_max_tokens if max_tokens is None else max_tokens
    governed = _clean_tool_results(messages)
    governed = await _compact_history(governed, llm)
    # 最外层再包一次：各策略无变化时会把入参原样返回，调用方拿到的应是新 list
    return list(_apply_token_budget(governed, budget))


# ---------- 历史压缩 ----------


async def _compact_history(
    messages: list[Message], llm: LLMClient | None
) -> list[Message]:
    """历史超过阈值时，把更早的部分压成一条摘要 system 消息，保留最近若干条完整对话。

    摘要插在记忆消息之后、完整历史之前。已有摘要说明本轮压过了（Agent 循环每轮都会
    治理一次），不再重复调 LLM。
    """
    if not settings.context_compaction_enabled:
        return messages
    if any(_is_summary(m) for m in messages):
        # 兜底，正常链路不可达：run_agent 每轮从 load_history 重新组装，已有历史的摘要
        # 一定是本函数刚生成的。命中说明调用方自己拼了带摘要的视图（如直接复用上一轮的
        # 治理产物），此时不重复压缩也不再调一次摘要 LLM
        return messages

    threshold = max(settings.context_compaction_threshold, MIN_COMPACTION_THRESHOLD)
    start, end = _history_span(messages)
    history = messages[start:end]
    if len(history) <= threshold:
        return messages

    boundary = _keep_boundary(history, threshold // 2)
    if boundary <= 0:
        return messages

    summary = await _summarize(history[:boundary], llm)
    if summary is None:
        return messages  # 摘要失败退化为不压缩：治理是增强项，不能拖垮这一轮
    return [
        *messages[:start],
        Message(role="system", content=f"{SUMMARY_PREFIX} {summary}"),
        *history[boundary:],
        *messages[end:],
    ]


def _keep_boundary(history: list[Message], keep: int) -> int:
    """最近 keep 条历史的起点，往前贴到轮次边界，不把一轮问答从中间劈开。"""
    raw = max(len(history) - max(keep, 1), 0)
    for i in range(raw, -1, -1):
        if history[i].role == "user":
            return i
    return raw


async def _summarize(messages: list[Message], llm: LLMClient | None) -> str | None:
    """用 LLM 把一段历史压成摘要文本；失败（含未配 API key）返回 None。

    送进 prompt 的文本有 SUMMARY_INPUT_LIMIT 上限，超出时保留**尾部**：被压区间里越
    靠近保留窗口的条目越可能与后续对话关联，丢它们比丢最开头更亏。这仍是有损的——
    被截掉的部分不会以任何形式进入摘要。
    """
    lines = [f"{m.role}：{m.content.strip()}" for m in messages if m.content.strip()]
    transcript = "\n".join(lines)[-SUMMARY_INPUT_LIMIT:]
    if not transcript:
        return None
    try:
        result = await (llm or get_llm()).chat(
            [
                Message(role="system", content=SUMMARY_PROMPT),
                Message(role="user", content=transcript),
            ]
        )
    except Exception as exc:  # 摘要失败退化为保留完整历史，不影响本轮回答
        # 只记类型与截断后的信息：provider 异常的 str() 常带着响应体，可能回显对话内容
        detail = str(exc)[:ERROR_TEXT_LIMIT]
        logger.warning("历史压缩失败，本轮不压缩：%s: %s", type(exc).__name__, detail)
        return None
    summary = result.text.strip()
    return summary or None


# ---------- 工具结果清理 ----------


def _clean_tool_results(messages: list[Message]) -> list[Message]:
    """保留最近 KEEP_TOOL_ROUNDS 轮的工具结果，更早的换成带原始 query 的占位符。

    最近几轮的检索证据模型还在用，清理掉会让这轮回答变成瞎猜；更早的结果在后续轮次
    里已经不重要了，占位符保留「检索发生过 + 查了什么」，不丢可追溯性。
    """
    if not settings.context_tool_clean_enabled:
        return messages
    rounds = _tool_rounds(messages)
    stale = [i for rnd in rounds[:-KEEP_TOOL_ROUNDS] for i in rnd]
    if not stale:
        return messages
    queries = _query_by_call_id(messages)
    out = list(messages)
    for i in stale:
        out[i] = _omitted(out[i], queries)
    return out


def _omitted(message: Message, queries: dict[str, str]) -> Message:
    """工具结果的占位符；能查到原始 query 时截断后写进占位符，方便模型与人工回溯。

    硬约束：占位符不得比原内容长。query 是模型自由生成的、没有长度上限，而工具结果
    可能很短（无命中时只有十几个字），原样内联会让「清理」反过来放大 prompt，进而让
    token 预算去删本该保留的摘要和记忆。拿不到更短的占位符时就返回原文——宁可不清，
    也不放大。
    """
    query = queries.get(message.tool_call_id or "")
    candidates = [TOOL_OMITTED]
    if query:
        trimmed = (
            query
            if len(query) <= MAX_QUERY_IN_PLACEHOLDER
            else query[:MAX_QUERY_IN_PLACEHOLDER] + "…"
        )
        candidates.insert(0, f'[之前检索过 "{trimmed}"，结果已省略]')
    # 优先带 query 的版本（信息更多），它不比原文短才退回通用占位符，再不行就保留原文
    content = next((c for c in candidates if len(c) < len(message.content)), None)
    if content is None:
        return message
    return message.model_copy(update={"content": content})


def _tool_rounds(messages: list[Message]) -> list[list[int]]:
    """每个工具轮次里 tool 结果消息的下标（按时间正序）。

    一轮 = 一条含 tool_calls 的 assistant 消息 + 紧随其后的 tool 结果。没有工具结果的
    轮次（正在生成中）不计入，否则「保留最近 2 轮」会被空轮次挤掉。
    """
    rounds: list[list[int]] = []
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            rounds.append([])
        elif m.role == "tool" and rounds:
            rounds[-1].append(i)
    return [rnd for rnd in rounds if rnd]


def _query_by_call_id(messages: list[Message]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in messages:
        for call in m.tool_calls or []:
            query = call.arguments.get("query")
            if isinstance(query, str) and query:
                out[call.id] = query
    return out


# ---------- token 预算 ----------


def _apply_token_budget(messages: list[Message], max_tokens: int) -> list[Message]:
    """超预算时按「最早历史 → 工具结果 → 摘要 → 记忆」的优先级丢内容。

    第一项删的是完整轮次（不把一轮问答劈开）；工具结果按轮次从最早开始换占位符，且只在
    真的更省 token 时才换（见 _omitted）。丢不掉（没东西可丢）说明视图本身是完整的，
    此时不注明截断；只要动过内容就在 system 消息里注明「上下文已截断」，免得模型把残缺
    的历史当成全部事实。

    截断提示本身也占预算，所以内部目标先扣掉它的长度，保证「加完提示」的最终结果仍不超
    预算。若视图里已有提示（run_agent 每轮都治理一次），不再追加，也不会重复扣。

    有个不可再压的下限：system prompt + 截断提示 + 当前提问。预算给到比这个下限还小
    （如 max_tokens=5）时无法满足，此时保留这些必需消息而不是删光——删掉当前提问就没有
    可回答的东西了，删 system prompt 会丢工具契约。这一档是诚实的「做不到」，不是漏算。
    """
    if not settings.context_token_budget_enabled:
        return messages
    if prompt_tokens(messages) <= max_tokens:
        return messages

    has_note = any(m.content == TRUNCATION_NOTE for m in messages)
    target = max_tokens if has_note else max_tokens - estimate_tokens(TRUNCATION_NOTE)

    out = list(messages)
    truncated = False

    start, end = _history_span(out)
    while start < end and prompt_tokens(out) > target:
        stop = _round_end(out, start, end)
        del out[start:stop]
        end -= stop - start
        truncated = True

    if prompt_tokens(out) > target:
        queries = _query_by_call_id(out)
        for i in [i for rnd in _tool_rounds(out) for i in rnd]:
            if prompt_tokens(out) <= target:
                break
            replaced = _omitted(out[i], queries)
            # 按 token 收益判定，而不是「内容变了没有」：占位符更长时会越换越大
            if prompt_tokens([replaced]) < prompt_tokens([out[i]]):
                out[i] = replaced
                truncated = True

    # 摘要是「已确认结论」的浓缩，记忆是跨会话的用户画像，都比工具结果更值得留
    for drop in (_is_summary, _is_memory):
        if prompt_tokens(out) <= target:
            break
        kept = [m for m in out if not drop(m)]
        if len(kept) != len(out):
            out = kept
            truncated = True

    if not truncated:
        return messages
    return out if has_note else _with_truncation_note(out)


def _with_truncation_note(messages: list[Message]) -> list[Message]:
    """把截断提示作为 system 消息插在 system prompt 之后（不原地改 prompt 文本）。"""
    note = Message(role="system", content=TRUNCATION_NOTE)
    if messages and messages[0].role == "system":
        return [messages[0], note, *messages[1:]]
    return [note, *messages]


# ---------- 视图结构 ----------


def _prefix_end(messages: list[Message]) -> int:
    """前导 system 消息（system prompt / 记忆 / 摘要 / 截断提示）之后的下标。"""
    i = 0
    while i < len(messages) and messages[i].role == "system":
        i += 1
    return i


def _history_span(messages: list[Message]) -> tuple[int, int]:
    """历史区间的 [start, end)：前缀之后、最后一条用户消息之前。

    组装好的视图是「system 前缀 + 历史 + 当前提问」。Agent 循环里追加的 assistant/tool
    消息都排在当前提问之后，所以循环各轮的历史区间仍是已有历史——压缩针对的是已有历史，
    不会把本轮刚产生的检索证据压掉。

    找不到用户消息时（调用方自己拼的视图，正常链路不会出现：assemble_messages 总会把
    当前提问放在末尾）返回空区间。没有「当前提问」这个锚点就分不清哪段是历史、哪段是
    正在进行的回合，此时宁可不删——乱删会把模型这一轮要看的消息删掉。
    """
    start = _prefix_end(messages)
    end = len(messages)
    for i in range(len(messages) - 1, start - 1, -1):
        if messages[i].role == "user":
            end = i
            break
    else:
        return start, start
    return start, end


def _round_end(messages: list[Message], start: int, end: int) -> int:
    """历史区间里第一轮的结束下标（到下一个 user 消息之前）。"""
    for i in range(start + 1, end):
        if messages[i].role == "user":
            return i
    return end


def _is_summary(message: Message) -> bool:
    return message.role == "system" and message.content.startswith(SUMMARY_PREFIX)


def _is_memory(message: Message) -> bool:
    return message.role == "system" and message.content.startswith(RECALL_HEADER)
