"""T9 可观测：LLM / 工具调用埋点与成本看板。

设计取舍（见 .superpowers/sdd/design/task-9-brief.md）：

- 埋点是 fire-and-forget：调用方只调度一个后台写库任务就继续，不 await、不上抛，
  失败只记 warning（与 app/memory/writer.py 同一口径）——观测不能拖慢或拖垮对话。
- 落在 T1 建好的 traces 表（SQLite），不做 OpenTelemetry：单进程项目，SQLite 足够，
  brief 明确划掉了 OTEL 与分布式追踪。
- 成本是**估算**：按 provider 定价表（app/config.py，每 1M tokens 的美元单价）乘
  provider 返回的 usage，只做量级归因，不追求与账单一致。
- 流式调用只有收尾 chunk 才带 usage，所以 chat_stream 在产完最后一个 chunk 之后才记；
  消费者中途弃用生成器（客户端断开）时这条调用不计——拿不到 usage 就不编造 token 数。

已知取舍（个人量级可接受，量级上来了再动）：

- 每条 trace 单开一个连接写一行：本机实测中位 ~7ms、最大 ~9ms（含加载 sqlite-vec 扩展）。
  比攒批写慢，但省掉了写队列/重试/背压这一整套机制；_pending 也没有背压上限，极端
  情况下（一次几百个并发调用）任务集会短暂膨胀。真要压开销就改成单写者队列。
- traces 表已按 (kind, name, ts) 建索引（schema.sql 的 idx_traces_kind_name_ts）：
  看板的 kind/name 过滤与时间窗查询都走索引，不再全表扫描。
- 看板的总计与分组是两条独立查询、没有显式事务，并发写入时两者可能短暂不一致
  （差一条极新的记录）。看板是人工看的量级视图，这种瞬时偏差可接受。
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


class InvalidTimestamp(ValueError):
    """start / end 不是「带时区的 ISO 8601 时间戳」。HTTP 层据此回 422。"""


DEFAULT_LIMIT = 100
# 收尾等待在途写入的上限：写库卡住时不能拖住进程退出与测试收尾（同 runtime.DRAIN_TIMEOUT）
DRAIN_TIMEOUT = 5.0
# 成本保留的小数位。定价是 per-Mtok，单次调用常在 1e-4~1e-6 量级，6 位会把小额抹平；
# 8 位既留得住精度，又能盖掉浮点求和末尾的噪声（如 0.00012300000000000001）
COST_DECIMALS = 8

# 每 1M tokens 的单价所在的配置字段，按 provider 分档。表里没有的 provider（如通义）
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
    """trace 时间戳：UTC、毫秒精度。

    格式即时间窗过滤的比较基准（见 _normalize_ts），毫秒精度决定时间窗粒度；
    列表排序用 id 而不是 ts（同毫秒的多条记录靠 id 保持稳定顺序）。
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalize_ts(value: str) -> str:
    """把过滤用的时间戳归一化成与 _now() 同格式的 UTC 字符串，供字典序比较。

    解析用 datetime.fromisoformat：3.11+ 接受 Z 后缀与任意偏移（+08:00 等），
    归一化到 UTC 后「同一瞬时」才比较得出同一个结果——直接拿原串比字典序的话，
    `…T10:00:00Z` 会既不大于也不小于 `…T10:00:00.000+00:00`，静默给出错误结果。

    必须带时区：不带时区的时间点本身有歧义，与其替调用方猜，不如报错让人说清楚。
    解析不了、缺时区、或（能解析但）换算到 UTC 越界都抛 InvalidTimestamp，
    由 HTTP 层转 422——这三种都是「调用方给的值不可用」，不该变成 500。
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise InvalidTimestamp(
            f"时间戳无法解析：{value!r}，需要带时区的 ISO 8601（如 2026-09-25T10:00:00Z）"
        ) from exc
    if parsed.tzinfo is None:
        raise InvalidTimestamp(
            f"时间戳缺少时区：{value!r}，需要带时区的 ISO 8601"
            "（如 2026-09-25T10:00:00Z 或 2026-09-25T18:00:00+08:00）"
        )
    try:
        return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    except OverflowError as exc:
        # datetime 能表示的范围挡不住偏移换算：0001-01-01T00:00:00+08:00 减 8 小时就出界。
        # 换算这一步必须也在守卫里，否则极值时间戳会以 OverflowError 冒到 HTTP 层变 500
        raise InvalidTimestamp(
            f"时间戳超出可比较范围：{value!r}（换算到 UTC 越界），请改用更接近当下的时间"
        ) from exc


def estimate_cost(provider: str, tokens_in: int, tokens_out: int) -> float:
    """按 provider 定价表估算成本（美元）。单价每次从 settings 读，改配置立即生效。

    定价是**每 1M tokens** 的单价（与各家官方报价一致），所以这里除以 1e6。
    """
    fields = _PRICE_FIELDS.get(provider, _PRICE_FIELDS["openai"])
    price_in, price_out = (getattr(settings, field) for field in fields)
    return round(
        (tokens_in * price_in + tokens_out * price_out) / 1_000_000, COST_DECIMALS
    )


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

    调度后台任务后立刻返回（不 await、不碰数据库，所以主流程的延迟不受影响）；
    tracing_enabled 关掉时连任务都不建（T10 的观测开销对照）。任何失败都只记 warning。
    """
    if not settings.tracing_enabled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as exc:  # 没有运行中的事件循环：从同步上下文误调时走这里
        logger.warning(
            "trace 未记录（无运行中的事件循环）：%s（%s %s）", exc, kind, name
        )
        return
    # 先拿到 loop 再造协程：create_task 直接抛的话会留下一个没人 await 的协程对象
    task = loop.create_task(
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

    db_path 由**调用方的 client** 带来：LLM 客户端在构造时存下它（见 app/llm 的
    get_llm），record_llm 原样透传给 record_trace。用自定义库创建 client 的地方
    （评测 / CLI）因此能把 llm trace 落进那个库；client 没传 db_path 时落 None，
    由 get_db 兜到默认库 settings.db_path——HTTP 主流程走的正是这条。
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


def record_skill(name: str, detail: str = "", db_path: str | None = None) -> None:
    """skill 触发埋点（kind='skill'）：token 与成本都是 0。

    detail 放本轮命中的触发词，形如 `触发词：简历、求职`。触发命中率是 T11 的
    核心指标，落在 traces 表就能按 kind='skill' 直接统计，不用另建表。
    """
    record_trace("skill", name, detail=detail, db_path=db_path)


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

    timeout 是**软上限**：每轮最多等这么久，超时就放弃剩余任务并告警（最坏情况还要加上
    sqlite 的 busy timeout，因为取消一个卡在写库上的任务也要等它退出）。丢的只是观测
    数据，不影响业务结果。
    """
    loop = asyncio.get_running_loop()
    while _pending:
        # 上一个事件循环留下的任务（如 asyncio.run 收尾时的残项）在当前循环里
        # wait/cancel/gather 都会抛 RuntimeError("attached to a different loop")，
        # 而它的循环已经没了、永远等不到结果——直接丢弃并告警，否则 _pending 会永久
        # 留着这些死任务，之后每次 drain 都崩
        aliens = [task for task in _pending if task.get_loop() is not loop]
        for task in aliens:
            _pending.discard(task)
        if aliens:
            logger.warning("丢弃 %d 个来自其他事件循环的 trace 任务", len(aliens))
        if not _pending:
            return
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

    start / end 必须是带时区的 ISO 8601 时间戳（任意偏移或 Z 均可），归一化到 UTC
    后与 ts 做字典序比较（两边格式一致，所以字典序等于时间序）；闭区间 [start, end]。
    不规范的值抛 InvalidTimestamp，不会静默当成「无结果」。
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
        params.append(_normalize_ts(start))
    if end:
        clauses.append("ts <= ?")
        params.append(_normalize_ts(end))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _group_row(field: str, row: Any) -> dict[str, Any]:
    """一条分组统计。分组键的列名按 kind/name 命名，读起来比统一的 key 直观。"""
    return {
        field: row["key"],
        "calls": row["calls"],
        "tokens_in": row["tokens_in"],
        "tokens_out": row["tokens_out"],
        "cost": round(row["cost"], COST_DECIMALS),
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
        "cost": round(totals["cost"], COST_DECIMALS),
        "by_kind": grouped["kind"],
        "by_name": grouped["name"],
    }
