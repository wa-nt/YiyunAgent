from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from app.agent.context import compact_history_persistent, govern_context
from app.config import settings
from app.db import get_db
from app.llm import get_llm
from app.llm.types import Message, ToolCall, ToolDef
from app.memory.recall import recall_memories
from app.memory.writer import extract_and_store
from app.resources import resource_path
from app.retrieval.hybrid import hybrid_search
from app.retrieval.types import RetrievedChunk
from app.settings_store import PERSONA_KEY, get_setting
from app.skills import (
    Skill,
    ToolFn,
    get_skill,
    load_skills,
    match_skill,
    render_skill_prompt,
)
from app.study.tools import TOOL_FNS as STUDY_TOOL_FNS
from app.study.tools import TOOLS as STUDY_TOOLS
from app.tracing import record_skill, record_tool

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 20
MAX_TOOL_ROUNDS = 6
TOOL_RESULT_LIMIT = 3000

SYSTEM_PROMPT = (
    "你是个人知识库助手。回答必须基于检索结果，并使用 search_knowledge 工具"
    "查证后再作答；检索结果不足以回答时明确说明，不要编造。\n"
    "给出结论时标注来源：在引用处写上对应的 chunk_id，格式如 [chunk 12]。"
)

SEARCH_TOOL = ToolDef(
    name="search_knowledge",
    description="在个人知识库中检索相关笔记内容",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "检索查询"}},
        "required": ["query"],
    },
)

# ---------- 模式框架（T1） ----------
#
# chat / work 是本期的两个可用模式；code 的运行时（sidecar、沙箱）不在本期，API 也不
# 接受它。SUPPORTED_MODES 是**唯一**的模式校验来源：API 校验、迁移归一化、运行时读取
# 都从它取，MODE_PROMPTS / MODE_TOOLS 也以它为准，避免几处各写一份允许清单。

SUPPORTED_MODES = ("chat", "work")
DEFAULT_MODE = "chat"

MODE_PROMPTS = {
    "chat": (
        "\n\n当前模式：聊天。\n"
        "像一位有温度但不谄媚的对话伙伴：有不同看法就说出来，不为顺着对方而附和。\n"
        "保持边界和自主判断，不替用户做决定，也不假装拥有自己并不具备的权威。\n"
        "优先从事实、原理和证据出发分析问题；抽象概念用类比或具体例子讲清楚。\n"
        "不确定的信息不要编造，直接说明哪些地方需要用户确认。\n"
        "少用破折号和模板化收尾，用自然的节奏收束回答。"
    ),
    "work": (
        "\n\n当前模式：工作。\n"
        "把用户的问题当成要交付的任务：先明确目标与约束，再拆成可执行的步骤。\n"
        "需要事实依据时先用 search_knowledge 查证，结论要落到能直接使用的产出或动作上。\n"
        "信息不足时直接指出还缺什么，不用套话填充。\n"
        "同时你是用户的复习教练，全程留意他掌握到什么程度：\n"
        "- 发现某个知识点没掌握（没听懂、答错、只有模糊印象）就调 record_knowledge_gap "
        "记一张复习卡片：topic 写能一眼认出的主题，detail 写清卡在哪一步。\n"
        "- 讲完一个知识点后追问一句检查理解的问题，让用户自己复述或举例。"
        "别用「明白了吗」这类只能换回一声「嗯」的问法。\n"
        "- 复习时先出题让用户答（对着卡片的 topic 出题），等他真的回答了再调 "
        "review_knowledge_gap：答对传 passed=true，答错或答不上来传 false，"
        "不要替用户判断对错。\n"
        "记卡片是帮用户攒复习材料，不是扣分：措辞保持中性，不评判用户。"
    ),
}

# 每个模式开放的**内置**工具名。skill 自带的工具不受这张表约束：它们只在触发的那一轮
# 生效，由 run_agent 单独追加（见 _apply_skill）。
MODE_TOOLS: dict[str, list[str]] = {
    "chat": ["search_knowledge"],
    "work": ["search_knowledge", "record_knowledge_gap", "review_knowledge_gap"],
}

# 所有模式白名单的并集：dispatch 用它区分「本模式没开放」和「根本没这个工具」
_MODE_TOOL_NAMES = frozenset(name for names in MODE_TOOLS.values() for name in names)

# 内置工具定义。漏洞工具（T3）来自 app/study/tools.py，与白名单里的名字一一对应；
# tools_for_mode 只按 MODE_TOOLS 过滤，不关心实现放在哪个模块。
BUILTIN_TOOLS: list[ToolDef] = [SEARCH_TOOL, *STUDY_TOOLS]


class UnknownModeError(ValueError):
    """读到的 mode 不在 SUPPORTED_MODES 内。

    不静默降级到默认模式：脏数据的会话如果被当成 chat 继续跑，用户看到的行为与
    记录的模式对不上，问题会被藏起来；这里的调用方（HTTP 层）要能看见并说清楚。
    """


def _check_mode(mode: str) -> str:
    if mode not in SUPPORTED_MODES:
        raise UnknownModeError(f"未知模式：{mode}（支持：{'、'.join(SUPPORTED_MODES)}）")
    return mode


def tools_for_mode(mode: str) -> list[ToolDef]:
    """本模式可用的内置工具定义（按 MODE_TOOLS 的白名单过滤）。"""
    allowed = set(MODE_TOOLS[_check_mode(mode)])
    return [tool for tool in BUILTIN_TOOLS if tool.name in allowed]


@dataclass
class AgentEvent:
    type: str  # text_delta | tool_start | tool_end | done | error
    data: dict[str, Any] = field(default_factory=dict)


# 未完成的记忆写入任务。fire-and-forget 不能裸 create_task：任务只被事件循环弱引用，
# 随时可能被 GC 掉；同时测试需要在同一事件循环里 await 到写入结束。
_pending_writes: set[asyncio.Task] = set()
# 收尾等待记忆写入的上限：写库卡住时不让进程退出被无限拖住
DRAIN_TIMEOUT = 5.0


def spawn_memory_write(
    session_id: str, user_message: str, answer: str, db_path: str
) -> asyncio.Task:
    """把一轮对话的记忆抽取丢到后台，不阻塞 SSE 流；失败在 extract_and_store 内消化。"""
    task = asyncio.create_task(
        extract_and_store(session_id, user_message, answer, db_path),
        name=f"memory-write:{session_id}",
    )
    _pending_writes.add(task)
    task.add_done_callback(_pending_writes.discard)
    return task


async def _write_title(session_id: str, user_message: str, db_path: str) -> None:
    """首轮对话后用 LLM 起个短标题。只在 title 还是 NULL 时写入：用户改名或
    ChatGPT 导入带来的标题不被覆盖。失败只告警——标题是锦上添花，不值得打断对话。"""
    try:
        llm = get_llm()
        result = await llm.chat(
            [
                Message(
                    role="user",
                    content="为下面这段提问起一个不超过 15 字的会话标题，"
                    "只输出标题本身，不要引号、不要标点结尾：\n\n" + user_message[:200],
                )
            ]
        )
        title = (result.text or "").strip().strip('"“”').splitlines()[0][:30].strip()
        if not title:
            return
        async with get_db(db_path) as conn:
            await conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
                (title, session_id),
            )
            await conn.commit()
    except Exception as exc:
        logger.warning("生成会话标题失败：%s: %s", type(exc).__name__, exc)


def spawn_title_write(
    session_id: str, user_message: str, db_path: str
) -> asyncio.Task:
    """与记忆写入同一口径：fire-and-forget，进 _pending_writes 保证退出/测试时能 drain。"""
    task = asyncio.create_task(
        _write_title(session_id, user_message, db_path),
        name=f"title-write:{session_id}",
    )
    _pending_writes.add(task)
    task.add_done_callback(_pending_writes.discard)
    return task


async def drain_memory_writes(timeout: float = DRAIN_TIMEOUT) -> None:
    """等所有在途的记忆写入结束（测试与服务退出时用，不影响 HTTP 流）。

    有超时上限：写入卡住时不能让进程退出或测试收尾无限等下去，超时后放弃并告警。
    """
    while _pending_writes:
        done, pending = await asyncio.wait(list(_pending_writes), timeout=timeout)
        # 显式摘掉本轮看到的任务，不依赖 done_callback 的调度时机，保证循环必然收敛
        for task in done:
            _pending_writes.discard(task)
        if not pending:
            continue
        logger.warning("记忆写入在 %.1fs 内未完成，放弃等待：%d 个", timeout, len(pending))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in pending:
            _pending_writes.discard(task)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def ensure_session(
    session_id: str | None,
    db_path: str | None = None,
    *,
    mode: str = DEFAULT_MODE,
    source: str = "manual",
    scheduled_task_id: str | None = None,
    scheduled_occurrence_at: str | None = None,
) -> str:
    """session_id 为空时新建会话；传入未知 id 时补建对应行（外键约束要求）。

    mode / source / 调度归属只在**真正建行**时写入：已存在的会话一律不改（模式在创建
    那一刻固定，续聊以库里的值为准），这条约定由 INSERT OR IGNORE 保证。未知 mode 直接
    抛 UnknownModeError，不落一条跑不起来的会话。
    """
    mode = _check_mode(mode)
    if session_id is None:
        session_id = uuid.uuid4().hex
    async with get_db(db_path) as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at, mode, source, "
            "scheduled_task_id, scheduled_occurrence_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, _now(), mode, source, scheduled_task_id, scheduled_occurrence_at),
        )
        await conn.commit()
    return session_id


async def get_session_mode(session_id: str, db_path: str | None = None) -> str:
    """会话模式。行不存在或 mode 为 NULL/空（迁移前的存量数据）时按 chat 处理；
    非空的未知值抛 UnknownModeError——不静默进入一个没定义的运行时。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT mode FROM sessions WHERE id = ?", (session_id,)
        )
    if not rows or not rows[0]["mode"]:
        return DEFAULT_MODE
    return _check_mode(rows[0]["mode"])


async def _save_message(session_id: str, role: str, content: str, db_path: str | None) -> None:
    """落库一条消息并把它挂到当前分支末尾：parent = 会话的 active_leaf
    （旧数据没有 leaf 时回退最后一条），随后 active_leaf 指向新消息。"""
    async with get_db(db_path) as conn:
        leaf = (
            await conn.execute_fetchall(
                "SELECT COALESCE(s.active_leaf, "
                "(SELECT MAX(id) FROM messages WHERE session_id = ?)) AS leaf "
                "FROM sessions s WHERE s.id = ?",
                (session_id, session_id),
            )
        )[0]["leaf"]
        cursor = await conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at, parent_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, _now(), leaf),
        )
        await conn.execute(
            "UPDATE sessions SET active_leaf = ? WHERE id = ?",
            (cursor.lastrowid, session_id),
        )
        await conn.commit()


# 沿当前分支向上回溯的递归 CTE：起点是 active_leaf（无则回退最后一条），
# 逐步走 parent_id。分支模型下「历史」= 从叶子到根的一条链，而不是全表按 id 排
_LEAF_CHAIN_SQL = """
WITH RECURSIVE chain AS (
    SELECT m.id, m.role, m.content, m.parent_id FROM messages m
    WHERE m.session_id = :sid
      AND m.id = COALESCE((SELECT active_leaf FROM sessions WHERE id = :sid),
                          (SELECT MAX(id) FROM messages WHERE session_id = :sid))
    UNION ALL
    SELECT m.id, m.role, m.content, m.parent_id FROM messages m
    JOIN chain c ON m.id = c.parent_id
)
"""


async def load_history_rows(
    session_id: str, limit: int = HISTORY_LIMIT, db_path: str | None = None
) -> list[dict]:
    """按时间正序返回当前分支最近 limit 条历史，带消息 id（持久化压缩要按 id 记覆盖位置）。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            _LEAF_CHAIN_SQL + "SELECT id, role, content FROM chain ORDER BY id DESC LIMIT :lim",
            {"sid": session_id, "lim": limit},
        )
    rows.reverse()
    return [dict(row) for row in rows]


async def load_history(
    session_id: str, limit: int = HISTORY_LIMIT, db_path: str | None = None
) -> list[Message]:
    """按时间正序返回当前分支最近 limit 条历史（链是新到旧，翻正后截断）。"""
    rows = await load_history_rows(session_id, limit, db_path)
    return [Message(role=row["role"], content=row["content"] or "") for row in rows]


async def list_messages(session_id: str, db_path: str | None = None) -> list[dict]:
    """当前分支的完整消息链（根到叶），每条带分支位置信息供前端渲染 ◀ 2/3 ▶。"""
    async with get_db(db_path) as conn:
        chain = await conn.execute_fetchall(
            _LEAF_CHAIN_SQL + "SELECT id FROM chain",
            {"sid": session_id},
        )
        if not chain:
            return []
        path = [r["id"] for r in reversed(chain)]
        placeholders = ", ".join("?" for _ in path)
        rows = await conn.execute_fetchall(
            f"SELECT id, role, content, created_at, parent_id FROM messages "
            f"WHERE id IN ({placeholders})",
            path,
        )
        # 同 parent 的兄弟互为分支；roots（parent NULL）互为第一问的分支
        siblings = await conn.execute_fetchall(
            "SELECT id, parent_id FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    groups: dict[int | None, list[int]] = {}
    for s in siblings:
        groups.setdefault(s["parent_id"], []).append(s["id"])
    by_id = {r["id"]: dict(r) for r in rows}
    out = []
    for mid in path:
        m = by_id[mid]
        sibs = groups.get(m["parent_id"], [mid])
        m["branch_index"] = sibs.index(mid) + 1
        m["branch_count"] = len(sibs)
        out.append(m)
    return out


async def session_provider(session_id: str, db_path: str | None = None) -> str | None:
    """会话的供应商覆盖；NULL = 跟随全局 llm_provider。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT provider FROM sessions WHERE id = ?", (session_id,)
        )
    return rows[0]["provider"] if rows else None


async def session_model(session_id: str, db_path: str | None = None) -> str | None:
    """会话的模型覆盖；NULL = 跟随全局该供应商默认模型。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT model FROM sessions WHERE id = ?", (session_id,)
        )
    return rows[0]["model"] if rows else None


async def session_effort(session_id: str, db_path: str | None = None) -> str | None:
    """会话的思考强度覆盖；NULL = 跟随供应商默认。四档 off/low/high/max。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT effort FROM sessions WHERE id = ?", (session_id,)
        )
    return rows[0]["effort"] if rows else None


# ---------- 可配置人格（T2） ----------
#
# 人格原文是**用户数据**：存 app_settings（见 app/settings_store.py），不进 .env、不进
# EDITABLE_FIELDS。没设置过时加载仓库内的默认人格文件；空字符串表示用户明确不要人格。
# 默认人格的原则（有温度但不谄媚、守住边界与独立判断、事实/原理/证据优先、抽象用类比
# 落地、坦诚不确定、避免模板化收尾）参考了 liliMozi/openhanako 的 Hanako 模板
# （Apache-2.0）所体现的思路；文字为本项目原创中文重写，不保留其用户/访客身份设定，
# 也不在运行时访问外部仓库。
PERSONA_FILE = "app/agent/prompts/persona_default.md"


class PersonaUnavailableError(RuntimeError):
    """没设置过人格、默认人格文件又读不到。

    不静默返回空人格：那会把「人格文件没打进包」表现成「模型忽然不按人格说话」，从
    现象看不出因果。这里直接抛，交给 HTTP 层（/api/settings）或 run_agent 既有的错误
    路径报出来。
    """


def persona_default_path() -> Path:
    """默认人格文件的真实位置：源码环境与 PyInstaller 解包目录由 resource_path 统一解析。"""
    return resource_path(PERSONA_FILE)


async def load_persona(db_path: str | None = None) -> str:
    """本轮要用的人格原文。

    用户设置过就原样返回（包括空串 = 明确禁用人格，不回退默认文件）；没有记录才回退
    默认人格文件。文件读不到时抛 PersonaUnavailableError（见上）。
    """
    stored = await get_setting(PERSONA_KEY, db_path)
    if stored is not None:
        return stored
    path = persona_default_path()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PersonaUnavailableError(f"默认人格文件不可读：{path}（{exc}）") from exc


def assemble_messages(
    history: list[Message],
    user_message: str,
    memory: str | None = None,
    skill_prompt: str | None = None,
    mode: str = DEFAULT_MODE,
    persona: str | None = None,
) -> list[Message]:
    """组装发给模型的 prompt view。system prompt = SYSTEM_PROMPT + 模式 prompt；
    persona 非空时前置（T2 从 app_settings 读；空串表示用户明确不要人格，不回退默认）。
    召回的长期记忆作为一条 system 消息插在 system prompt 之后，命中的 skill 正文排在
    记忆之后（skill 比记忆更贴近本轮任务）。
    T8 的上下文治理（工具结果清理、历史压缩、token 预算）由 run_agent 在每次调用模型
    前对这里的产物走一遍 govern_context，不改 Agent 循环。

    memory 由 run_agent 先调 recall_memories 取好（本函数是同步的，召回是异步的）；
    skill_prompt 同理，由 run_agent 触发并渲染好（见 _apply_skill）。
    """
    system = SYSTEM_PROMPT + MODE_PROMPTS[_check_mode(mode)]
    if persona:
        system = f"{persona}\n\n{system}"
    messages = [Message(role="system", content=system)]
    if memory:
        messages.append(Message(role="system", content=memory))
    if skill_prompt:
        messages.append(Message(role="system", content=skill_prompt))
    return [*messages, *history, Message(role="user", content=user_message)]


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """检索结果格式化为「标题 + 内容 + chunk_id」文本，整体截断到 TOOL_RESULT_LIMIT。"""
    if not chunks:
        return "（知识库中没有检索到相关内容）"
    pieces = [
        f"[chunk {c.chunk_id}] {c.title or '未命名文档'}\n{c.content}" for c in chunks
    ]
    text = "\n\n".join(pieces)
    if len(text) > TOOL_RESULT_LIMIT:
        text = text[:TOOL_RESULT_LIMIT] + "\n…（检索结果已截断）"
    return text


async def execute_tool(
    call: ToolCall,
    db_path: str | None = None,
    skill_tools: dict[str, ToolFn] | None = None,
    mode: str = DEFAULT_MODE,
) -> tuple[str, str]:
    """执行一次工具调用，返回 (结果文本, 展示用摘要)。

    返回前记一条 kind='tool' 的 trace（detail 即展示用摘要）：失败路径（未知工具、
    缺 query、模式未开放）同样计入，工具层的失败在成本看板上要看得见。工具**抛异常**
    （检索本身出错）时不在这里记——那次调用没走完，run_agent 在降级分支里补记一条。

    mode 决定白名单（MODE_TOOLS）：模型偶尔会调用别的模式才开放的工具，这里再挡一次
    ——prompt 里的工具清单不是安全边界。

    skill_tools 是本轮触发的 skill 带来的专用工具（{工具名: 实现}）。默认空：HTTP 层
    与测试不会直接调用它，只有 run_agent 在触发了 skill 的那一轮传进来——skill 工具
    的可用范围严格限定在触发它的那一轮（brief 的「动态工具集」），也不受 mode 白名单
    约束（它们本就不在 MODE_TOOLS 里）。
    """
    result, label = await _dispatch_tool(call, db_path, skill_tools or {}, mode)
    record_tool(call.name, label, db_path)
    return result, label


async def _dispatch_tool(
    call: ToolCall, db_path: str | None, skill_tools: dict[str, ToolFn], mode: str
) -> tuple[str, str]:
    allowed = MODE_TOOLS[_check_mode(mode)]
    if call.name not in allowed:
        fn = skill_tools.get(call.name)
        if fn is not None:
            return await _call_skill_tool(call.name, fn, call.arguments, db_path)
        if call.name in _MODE_TOOL_NAMES:
            return (
                f"工具 {call.name} 在当前模式（{mode}）不可用",
                f"{call.name}（{mode} 模式不可用）",
            )
        return f"未知工具：{call.name}", f"未知工具 {call.name}"
    if call.name == SEARCH_TOOL.name:
        return await _search(call, db_path)
    # 漏洞复习工具（T3）：handler 自己把参数错误与「卡片不存在」写成结果文本返回，
    # 不抛异常——工具层能自行说清的失败不该走 run_agent 的降级分支
    fn = STUDY_TOOL_FNS.get(call.name)
    if fn is not None:
        return await fn(call.arguments, db_path)
    # 白名单里有名字、却没有实现：本期不会发生（MODE_TOOLS 里每一项都有 handler），
    # 留着是为了将来加模式/工具时，漏做实现表现为一句说明而不是一次 AttributeError
    return f"工具 {call.name} 尚未实现", f"{call.name}（尚未实现）"


async def _search(call: ToolCall, db_path: str | None) -> tuple[str, str]:
    query = call.arguments.get("query")
    if not isinstance(query, str) or not query:
        return "检索失败：缺少 query 参数", "search_knowledge（缺少 query 参数）"

    chunks = await hybrid_search(
        query, k=8, mode=settings.retrieval_mode, db_path=db_path
    )
    return format_chunks(chunks), f"search_knowledge({query}) → {len(chunks)} 条"


async def _call_skill_tool(
    name: str, fn: ToolFn, args: dict[str, Any], db_path: str | None
) -> tuple[str, str]:
    """调用 skill 的专用工具，把它的返回**规整**成 (结果文本, 展示用摘要)。

    tools.py 是用户手写的代码，返回什么形状都不意外。宽松收口的原因：这里一旦抛
    TypeError，run_agent 的降级分支会把它当成「工具失败」写进 SSE 与 trace，而真实
    原因（作者返回格式不对）反而看不出；规整后至少模型能拿到一句可读的说明继续走。
    """
    try:
        returned = await fn(args, db_path)
    except Exception as exc:
        logger.warning("skill 工具 %s 执行失败：%s: %s", name, type(exc).__name__, exc)
        return f"工具 {name} 执行失败：{exc}", f"{name} 失败：{exc}"
    result, label = returned, name
    if isinstance(returned, tuple) and len(returned) == 2:
        result, label = returned
    else:
        logger.warning("skill 工具 %s 的返回值不是 (结果文本, 摘要)，已按单值处理", name)
    return str(result), str(label)


def _assistant_message(text: str, calls: list[ToolCall]) -> Message:
    """助手轮次的完整内容：正文 + 工具调用，都要进历史供下一轮与前端展示。"""
    parts = [text] if text else []
    parts.extend(
        f"[调用工具 {c.name} {json.dumps(c.arguments, ensure_ascii=False)}]" for c in calls
    )
    return Message(role="assistant", content="\n".join(parts), tool_calls=calls)


def _apply_skill(
    user_message: str, db_path: str | None
) -> tuple[str | None, list[ToolDef], dict[str, ToolFn]]:
    """探测本轮触发的 skill，返回 (注入用正文, 本轮新增的工具定义, 工具实现)。

    skill 是纯增强项：任何一步出问题都退化为「本轮没有 skill」，不抛异常、不打断
    对话（与记忆召回同一口径）。返回值三件套一并给出，是因为注入正文、注册工具、
    执行工具必须来自**同一次**加载——分两次加载的话正文与实际工具可能对不上。

    skills_enabled 关掉时直接返回空三件套：连目录都不扫，省掉每轮一次 stat 遍历。

    整个探测包在宽 except 里，兑现上面那条承诺：skills/ 是用户可写的目录，
    「一个坏文件不该让这一轮对话报错」。少兜这一层的话，异常会穿透 run_agent 的
    try（那个 try 在 `path = db_path or ...` 之后才开始，且用户提问此时还没落库），
    用户看到的是 error 事件、提问也丢了；坏文件还留在盘上，之后每一轮都失败。

    **连渲染与埋点一起包**（不是只包加载）：render_skill_prompt 也是这条链路上的一环，
    它抛异常同样是「本轮没有 skill」该覆盖的失败。只包前半段的话，承诺与实现之间会
    留一道同型的缝——探测成功但渲染失败时照样整轮报错、提问落不了库。
    """
    if not settings.skills_enabled:
        return None, [], {}
    try:
        metas = load_skills(settings.skills_dir)
        matched = match_skill(user_message, metas)
        if matched is None:
            return None, [], {}
        skill: Skill | None = get_skill(matched.name, metas)
        if skill is None:
            return None, [], {}
        detail = "触发词：" + "、".join(matched.triggers)
        record_skill(skill.meta.name, detail, db_path)
        logger.debug("触发 skill：%s（%s）", skill.meta.name, detail)
        return render_skill_prompt(skill), list(skill.tools), dict(skill.tool_fns)
    except Exception as exc:
        logger.warning(
            "skill 探测失败，本轮按无 skill 处理：%s: %s", type(exc).__name__, exc
        )
        return None, [], {}


async def run_agent(
    session_id: str,
    user_message: str,
    db_path: str | None = None,
    *,
    user_saved: bool = False,
    mode: str | None = None,
) -> AsyncIterator[AgentEvent]:
    """ReAct 主循环：加载历史 → 流式生成 → 有工具调用则执行并回到生成。

    不建表：HTTP 层由 main.py 的 lifespan 调 init_db，测试自行初始化。
    db_path 仅测试用；HTTP 层走默认路径（app.config.settings.db_path）。

    user_saved=True 用于「回答已有提问」（编辑后重答 / 重新生成）：调用方已把提问
    落库为会话最后一条，这里从加载到的历史里摘掉它、不再重复插入。

    mode 只在**新建会话**的那次调用显式传入（新会话还没有可读的库记录）；其余情况传
    None，由这里从会话读——「会话模式」因此只有一个事实来源，续聊不会因为调用方忘传
    模式而跑错。读到的未知 mode 抛 UnknownModeError（不静默按 chat 跑）。

    本轮触发的 skill（T11）在开跑前一次性探测：正文进 system 消息，专用工具进本轮
    工具集与执行判据（见 _apply_skill）。探测只做一次，工具集在整轮里不变。

    人格（T2）在开跑前从 app_settings 读一次（见 load_persona），非空时前置到 system
    消息；设置面板保存后下一轮即生效，不用重建会话。
    """
    path = db_path or settings.db_path
    await ensure_session(session_id, path, mode=mode if mode is not None else DEFAULT_MODE)
    session_mode = mode if mode is not None else await get_session_mode(session_id, path)

    rows = await load_history_rows(session_id, HISTORY_LIMIT, path)
    if user_saved:
        # respond 端点已校验最后一条就是这条提问，直接摘掉，避免 prompt 里出现两遍
        if rows and rows[-1]["role"] == "user":
            rows = rows[:-1]
    # 持久化压缩（T8）：摘要有缓存时本轮不再调摘要 LLM。摘要是后台工具调用，
    # 用全局默认模型即可（会话级 provider/model/effort 覆盖是面向正式回答的）
    history = await compact_history_persistent(session_id, rows, db_path=path)
    first_turn = not history
    # 召回是同步等价的（要进本轮提示词），但失败只降级为无记忆，不抛
    memory = await recall_memories(user_message, path)
    # skill 与记忆同一口径：探测失败退化为「本轮无 skill」，正文进 system 消息、
    # 专用工具进本轮工具集。工具集与正文同寿命——只在触发它的这一轮可用
    skill_prompt, skill_tools, skill_fns = _apply_skill(user_message, path)
    # 人格是用户数据：设置过用设置过的（空串 = 不要人格），没设置过加载默认人格文件
    persona = await load_persona(path)
    messages = assemble_messages(
        history, user_message, memory, skill_prompt, session_mode, persona
    )
    tools = [*tools_for_mode(session_mode), *skill_tools]

    # 用户提问立刻落库，不等本轮结束：客户端中途关页面时任务会被取消
    # （CancelledError），清理阶段的 await 会被打断，只有提前写才能保证提问不丢。
    if not user_saved:
        await _save_message(session_id, "user", user_message, path)

    answer = ""
    calls: list[ToolCall] = []
    answer_saved = False

    async def save_answer(degraded: bool = False, interrupted: bool = False) -> None:
        """落库助手输出，只落一次。

        各终态（done / 两条 error）在 yield 之前就调用它——客户端收到 done 就会
        关闭 SSE，任务随即被取消，那时再写库会被 CancelledError 打断。
        finally 里的调用只兜底「没有任何终态到达」的中断（客户端提前断开）。

        degraded=True 表示本轮以错误收场（工具轮数上限 / LLM 异常），此时照样落库，
        但不做记忆抽取——从失败轮次里学到的「事实」正是记忆污染的主要来源。
        """
        nonlocal answer_saved
        if answer_saved:
            return
        answer_saved = True
        content = _assistant_message(answer, calls).content
        if interrupted:
            # 中断路径（客户端断开 / 生成器被关闭）没有终态事件，回答是半截的，
            # 留个标记，避免下次加载历史时被当成完整回答
            mark = "（回答未完成）"
            content = f"{content}\n{mark}" if content else mark
        await _save_message(session_id, "assistant", content, path)
        # 只在完整成功的轮次后抽取记忆：中断路径的后台任务会被 CancelledError 掐掉，
        # 半截或失败的问答本身就是噪声。整个写入 fire-and-forget，不阻塞 SSE。
        if not interrupted and not degraded:
            spawn_memory_write(session_id, user_message, answer, path)
            if first_turn:
                # 首轮成功后顺手起标题（会话列表不再永远是首条消息的前 30 字）
                spawn_title_write(session_id, user_message, path)

    try:
        try:
            # 会话有供应商/模型覆盖时用它建客户端（密钥仍取该供应商的全局配置）。
            # 不带参数调 get_llm() 是为了兼容测试里 zero-arg 的桩
            provider = await session_provider(session_id, path)
            model = await session_model(session_id, path)
            effort = await session_effort(session_id, path)
            override = provider or model or effort
            llm = (
                get_llm(provider=provider, model=model, effort=effort)
                if override
                else get_llm()
            )
            for _ in range(MAX_TOOL_ROUNDS):
                # 每轮都重新治理：轮内追加的工具结果同样要进预算。治理结果接着用作
                # 下一轮的基底，历史摘要因此只生成一次；摘要持久化在 sessions 表
                # （compact_history_persistent），跨用户回合也只在增量越窗时才重压。
                messages = await govern_context(messages, llm=llm)
                text = ""
                final_calls: list[ToolCall] = []
                async for chunk in llm.chat_stream(messages, tools=tools):
                    if chunk.text_delta:
                        text += chunk.text_delta
                        # 立刻累加到 answer：客户端可能在流中途断开，
                        # 那时循环体的收尾语句不会执行，只有这里能保住已生成的部分
                        answer += chunk.text_delta
                        yield AgentEvent("text_delta", {"text": chunk.text_delta})
                    if chunk.finish:
                        final_calls = chunk.tool_calls

                if not final_calls:
                    await save_answer()
                    yield AgentEvent("done", {"session_id": session_id, "text": answer})
                    return

                calls.extend(final_calls)
                messages.append(_assistant_message(text, final_calls))
                for call in final_calls:
                    yield AgentEvent(
                        "tool_start",
                        {"id": call.id, "name": call.name, "arguments": call.arguments},
                    )
                    try:
                        result, label = await execute_tool(call, path, skill_fns, session_mode)
                    except Exception as exc:  # 工具失败降级为一段说明，让模型自行收尾
                        result = f"工具失败：{exc}"
                        label = f"{call.name} 失败：{exc}"
                        # 这条路径 execute_tool 内部没机会记 trace（异常穿透），在这里补：
                        # 检索故障正是看板上最该看见的东西，不能只留在 SSE 事件里
                        record_tool(call.name, label, path)
                    yield AgentEvent(
                        "tool_end", {"id": call.id, "name": call.name, "summary": label}
                    )
                    messages.append(
                        Message(role="tool", tool_call_id=call.id, content=result)
                    )

            await save_answer(degraded=True)
            yield AgentEvent(
                "error",
                {
                    "message": f"达到工具调用轮数上限（{MAX_TOOL_ROUNDS} 轮），已停止",
                    "session_id": session_id,
                    "text": answer,
                },
            )
        except Exception as exc:
            await save_answer(degraded=True)
            yield AgentEvent(
                "error",
                {
                    "message": f"{type(exc).__name__}: {exc}",
                    "session_id": session_id,
                    "text": answer,
                },
            )
    finally:
        # 走到这里说明没有终态事件发出过（客户端断开、生成器被关闭），
        # 尽力落库一次；任务已被取消时这次写库可能被打断，丢的只是半截回答。
        await save_answer(interrupted=True)