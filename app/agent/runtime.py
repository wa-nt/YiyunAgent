from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from app.config import settings
from app.db import get_db
from app.llm import get_llm
from app.llm.types import Message, ToolCall, ToolDef
from app.memory.recall import recall_memories
from app.memory.writer import extract_and_store
from app.retrieval.hybrid import hybrid_search
from app.retrieval.types import RetrievedChunk

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


@dataclass
class AgentEvent:
    type: str  # text_delta | tool_start | tool_end | done | error
    data: dict[str, Any] = field(default_factory=dict)


# 未完成的记忆写入任务。fire-and-forget 不能裸 create_task：任务只被事件循环弱引用，
# 随时可能被 GC 掉；同时测试与 CLI 需要在同一事件循环里 await 到写入结束。
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


async def drain_memory_writes(timeout: float = DRAIN_TIMEOUT) -> None:
    """等所有在途的记忆写入结束（测试、CLI 与服务退出时用，不影响 HTTP 流）。

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


async def ensure_session(session_id: str | None, db_path: str | None = None) -> str:
    """session_id 为空时新建会话；传入未知 id 时补建对应行（外键约束要求）。"""
    if session_id is None:
        session_id = uuid.uuid4().hex
    async with get_db(db_path) as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )
        await conn.commit()
    return session_id


async def _save_message(session_id: str, role: str, content: str, db_path: str | None) -> None:
    async with get_db(db_path) as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        await conn.commit()


async def load_history(
    session_id: str, limit: int = HISTORY_LIMIT, db_path: str | None = None
) -> list[Message]:
    """按时间正序返回最近 limit 条历史（子查询倒序取，再翻正）。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM ("
            "  SELECT id, role, content FROM messages WHERE session_id = ?"
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY id",
            (session_id, limit),
        )
    return [Message(role=row["role"], content=row["content"] or "") for row in rows]


async def list_messages(session_id: str, db_path: str | None = None) -> list[dict]:
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    return [dict(row) for row in rows]


def assemble_messages(
    history: list[Message], user_message: str, memory: str | None = None
) -> list[Message]:
    """组装发给模型的 prompt view。召回的长期记忆作为一条 system 消息插在 system
    prompt 之后。T8 的上下文治理（compaction、工具结果清理、token 预算）同样从这里
    接入，替换本函数即可，不改 Agent 循环。

    memory 由 run_agent 先调 recall_memories 取好（本函数是同步的，召回是异步的）。
    """
    messages = [Message(role="system", content=SYSTEM_PROMPT)]
    if memory:
        messages.append(Message(role="system", content=memory))
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


async def execute_tool(call: ToolCall, db_path: str | None = None) -> tuple[str, str]:
    """执行一次工具调用，返回 (结果文本, 展示用摘要)。"""
    if call.name != SEARCH_TOOL.name:
        return f"未知工具：{call.name}", f"未知工具 {call.name}"

    query = call.arguments.get("query")
    if not isinstance(query, str) or not query:
        return "检索失败：缺少 query 参数", "search_knowledge（缺少 query 参数）"

    chunks = await hybrid_search(query, k=8, db_path=db_path)
    return format_chunks(chunks), f"search_knowledge({query}) → {len(chunks)} 条"


def _assistant_message(text: str, calls: list[ToolCall]) -> Message:
    """助手轮次的完整内容：正文 + 工具调用，都要进历史供下一轮与前端展示。"""
    parts = [text] if text else []
    parts.extend(
        f"[调用工具 {c.name} {json.dumps(c.arguments, ensure_ascii=False)}]" for c in calls
    )
    return Message(role="assistant", content="\n".join(parts), tool_calls=calls)


async def run_agent(
    session_id: str, user_message: str, db_path: str | None = None
) -> AsyncIterator[AgentEvent]:
    """ReAct 主循环：加载历史 → 流式生成 → 有工具调用则执行并回到生成。

    不建表：HTTP 层由 main.py 的 lifespan 调 init_db，CLI 与测试自行初始化。
    db_path 仅测试与 CLI 用；HTTP 层走默认路径（app.config.settings.db_path）。
    """
    path = db_path or settings.db_path
    await ensure_session(session_id, path)

    history = await load_history(session_id, HISTORY_LIMIT, path)
    # 召回是同步等价的（要进本轮提示词），但失败只降级为无记忆，不抛
    memory = await recall_memories(user_message, path)
    messages = assemble_messages(history, user_message, memory)

    # 用户提问立刻落库，不等本轮结束：客户端中途关页面时任务会被取消
    # （CancelledError），清理阶段的 await 会被打断，只有提前写才能保证提问不丢。
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

    try:
        try:
            llm = get_llm()
            for _ in range(MAX_TOOL_ROUNDS):
                text = ""
                final_calls: list[ToolCall] = []
                async for chunk in llm.chat_stream(messages, tools=[SEARCH_TOOL]):
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
                        result, label = await execute_tool(call, path)
                    except Exception as exc:  # 工具失败降级为一段说明，让模型自行收尾
                        result = f"检索失败：{exc}"
                        label = f"{call.name} 失败：{exc}"
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