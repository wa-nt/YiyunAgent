"""T9 可观测：LLM / 工具调用埋点与成本看板。

设计取舍（见 .superpowers/sdd/design/task-9-brief.md）：

- 埋点是 fire-and-forget：调用方只调度一个后台写库任务就继续，不 await、不上抛，
  失败只记 warning（与 app/memory/writer.py 同一口径）——观测不能拖慢或拖垮对话。
- 落在 T1 建好的 traces 表（SQLite），不做 OpenTelemetry：单进程项目，SQLite 足够，
  brief 明确划掉了 OTEL 与分布式追踪。
- 成本是**估算**：按 provider 定价表（app/config.py）乘 provider 返回的 usage，只做
  量级归因，不追求与账单一致。
- 流式调用只有收尾 chunk 才带 usage，所以 chat_stream 在产完最后一个 chunk 之后才记；
  消费者中途弃用生成器（客户端断开）时这条调用不计——拿不到 usage 就不编造 token 数。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db import get_db
from app.llm.types import Usage

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 100
# 收尾等待在途写入的上限：写库卡住时不能拖住进程退出与测试收尾（同 runtime.DRAIN_TIMEOUT）
DRAIN_TIMEOUT = 5.0

# 每 1K tokens 的单价所在的配置字段，按 provider 分档。表里没有的 provider（如通义）
# 回落到 openai 档：定价本身就是估算，多一档等于多编一组数字
_PRICE_FIELDS = {
    "openai": ("price_openai_input", "price_openai_output"),
    "anthropic": ("price_anthropic_input", "price_anthropic_output"),
    "deepseek": ("price_deepseek_input", "price_deepseek_output"),
}

# 在途的写库任务。fire-and-forget 不能裸 create_task：任务只被事件循环弱引用，
# 随时可能被 GC 掉；同时测试与服务收尾需要能等到它们结束（同 runtime._pending_writes）
_pending: set[asyncio.Task] = set()


def _now() -> str:
    """毫秒精度：一轮对话会产生多条 trace，秒级时间戳不足以稳定排序。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def estimate_cost(provider: str, tokens_in: int, tokens_out: int) -> float:
    """按 provider 定价表估算成本（美元）。单价每次从 settings 读，改配置立即生效。"""
    fields = _PRICE_FIELDS.get(provider, _PRICE_FIELDS["openai"])
    price_in, price_out = (getattr(settings, field) for field in fields)
    return round(tokens_in / 1000 * price_in + tokens_out / 1000 * price_out, 6)


def record_trace(
    kind: str,
    name: str,
    detail: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost: float = 0.0,
    db_path: str | None = None,
) -> None:
    """埋点入口：把一次调用记进 traces 表，fire-and-forget。

    调度后台任务后立刻返回；tracing_enabled 关掉时连任务都不建（T10 的观测开销对照）。
    """
    if not settings.tracing_enabled:
        return
    task = asyncio.create_task(
        _insert(kind, name, detail, tokens_in, tokens_out, cost, db_path),
        name=f"trace:{kind}:{name}",
    )
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def record_llm(
    provider: str, model: str, usage: Usage | None = None, db_path: str | None = None
) -> None:
    """LLM 调用埋点：name=provider/model，成本按定价表估算。

    usage 为 None（部分兼容端点的流式响应不带 usage）时 token 与成本记 0：调用次数
    仍然计入，只是这一次没有 token 归因。
    """
    tokens_in = usage.tokens_in if usage else 0
    tokens_out = usage.tokens_out if usage else 0
    record_trace(
        "llm",
        f"{provider}/{model}",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=estimate_cost(provider, tokens_in, tokens_out),
        db_path=db_path,
    )


def record_tool(name: str, detail: str = "", db_path: str | None = None) -> None:
    """工具调用埋点：token 与成本都是 0——工具自己不烧 LLM 的钱。

    detail 放展示用摘要（工具名 + query + 命中条数）；失败路径（未知工具、缺 query）
    同样记录，工具层的失败在成本看板上要看得见，而不是只躺在日志里。
    """
    record_trace("tool", name, detail=detail, db_path=db_path)


async def _insert(
    kind: str,
    name: str,
    detail: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    db_path: str | None,
) -> None:
    """真正落库的一步，只被 record_trace 调度。任何失败都只记日志、不上抛。"""
    try:
        async with get_db(db_path) as conn:
            await conn.execute(
                "INSERT INTO traces (ts, kind, name, detail, tokens_in, tokens_out, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_now(), kind, name, detail, tokens_in, tokens_out, cost),
            )
            await conn.commit()
    except Exception as exc:  # 观测是增强项：写不进去也不能影响对话
        logger.warning(
            "trace 写入失败，已跳过：%s: %s（%s %s）", type(exc).__name__, exc, kind, name
        )


async def drain_traces(timeout: float = DRAIN_TIMEOUT) -> None:
    """等在途的 trace 写入结束（测试收尾、服务退出时用，不影响 HTTP 流程）。

    有超时上限：写库卡住时不能让进程退出或测试收尾无限等下去，超时后放弃并告警——
    丢的只是观测数据，不影响业务结果。
    """
    while _pending:
        done, pending = await asyncio.wait(list(_pending), timeout=timeout)
        # 显式摘掉本轮看到的任务，不依赖 done_callback 的调度时机，保证循环必然收敛
        for task in done:
            _pending.discard(task)
        if not pending:
            continue
        logger.warning("trace 写入在 %.1fs 内未完成，放弃等待：%d 个", timeout, len(pending))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in pending:
            _pending.discard(task)


# ---------- 成本看板查询 ----------


def _where(
    kind: str | None, name: str | None, start: str | None, end: str | None
) -> tuple[str, list[Any]]:
    """过滤条件拼装（SQL 片段 + 参数）。

    时间是 ISO 字符串的字典序比较，所以 start/end 要传带偏移量的完整时间戳：
    只传日期（"2026-09-25"）时 end 会漏掉当天的记录。
    """
    clauses: list[str] = []
    params: list[Any] = []
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if name:
        clauses.append("name = ?")
        params.append(name)
    if start:
        clauses.append("ts >= ?")
        params.append(start)
    if end:
        clauses.append("ts <= ?")
        params.append(end)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _group_row(field: str, row: Any) -> dict[str, Any]:
    """一条分组统计。分组键的列名按 kind/name 命名，读起来比统一的 key 直观。"""
    return {
        field: row["key"],
        "calls": row["calls"],
        "tokens_in": row["tokens_in"],
        "tokens_out": row["tokens_out"],
        "cost": round(row["cost"], 6),
    }


async def list_traces(
    kind: str | None = None,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    db_path: str | None = None,
) -> dict[str, Any]:
    """trace 分页列表，最新在前。name 为精确匹配（不做前缀/模糊匹配）。

    total 是同一过滤条件下的总条数，不随 limit/offset 变，供前端翻页。
    """
    where, params = _where(kind, name, start, end)
    async with get_db(db_path) as conn:
        counted = await conn.execute_fetchall(
            f"SELECT COUNT(*) AS n FROM traces{where}", params
        )
        rows = await conn.execute_fetchall(
            "SELECT id, ts, kind, name, detail, tokens_in, tokens_out, cost "
            f"FROM traces{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
    return {
        "items": [dict(row) for row in rows],
        "total": counted[0]["n"],
        "limit": limit,
        "offset": offset,
    }


async def summarize_traces(
    kind: str | None = None,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """成本看板聚合：总调用次数 / 总 tokens / 总成本，外加按 kind、按 name 分组。

    分组按成本降序——看板的用途就是看钱花在哪（成本相同时调用多的在前）。
    """
    where, params = _where(kind, name, start, end)
    async with get_db(db_path) as conn:
        totals = (
            await conn.execute_fetchall(
                "SELECT COUNT(*) AS calls, COALESCE(SUM(tokens_in), 0) AS tokens_in, "
                "COALESCE(SUM(tokens_out), 0) AS tokens_out, COALESCE(SUM(cost), 0) AS cost "
                f"FROM traces{where}",
                params,
            )
        )[0]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for field in ("kind", "name"):
            rows = await conn.execute_fetchall(
                f"SELECT {field} AS key, COUNT(*) AS calls, "
                "COALESCE(SUM(tokens_in), 0) AS tokens_in, "
                "COALESCE(SUM(tokens_out), 0) AS tokens_out, "
                "COALESCE(SUM(cost), 0) AS cost "
                f"FROM traces{where} GROUP BY {field} ORDER BY cost DESC, calls DESC, key",
                params,
            )
            grouped[field] = [_group_row(field, row) for row in rows]
    return {
        "total_calls": totals["calls"],
        "tokens_in": totals["tokens_in"],
        "tokens_out": totals["tokens_out"],
        "tokens_total": totals["tokens_in"] + totals["tokens_out"],
        "cost": round(totals["cost"], 6),
        "by_kind": grouped["kind"],
        "by_name": grouped["name"],
    }