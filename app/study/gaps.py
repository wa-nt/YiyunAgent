"""知识漏洞（间隔重复卡片）的领域函数。

纯 SQL：没有需要常驻的状态，函数拿 db_path 直接读写 `knowledge_gaps`，不引入服务层、
不缓存。时间统一 UTC（`_now()` 与 app/memory/writer.py 同一口径），所以
`next_review_at <= now` 可以直接用字符串比较——ISO 8601 带 `+00:00` 偏移的定长格式，
字典序就是时间序。

间隔规则（brief 定的，改动即变更产品行为）：

- 新卡片：1 天后复习；
- 复习通过：`interval = min(interval * 2, 180)` 天；
- 复习未通过：重置回 1 天；
- 到期：`next_review_at <= now`。

本期**不**实现归档/恢复，表里因此没有 `resolved` 一类的字段：一张卡片只有「记录」
「复习」「删除」三个动作。留一个没有写入路径的伪状态，比不做这个功能更难排查——
UI 上看起来有归档，点下去什么也没发生。

同主题重复记录是允许的（本期不做语义去重）：用户两次卡在同一个概念上，就是两张
卡片，各自有自己的间隔。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db import get_db

INITIAL_INTERVAL_DAYS = 1
# 复习通过的间隔上限：再长的间隔在「复习卡片」这个场景里等于不再复习
MAX_INTERVAL_DAYS = 180

# 列表/回读统一用这一列组，避免几处 SELECT 的列顺序漂移（返回值直接进 API 与工具结果）
_COLUMNS = "id, topic, detail, interval_days, next_review_at, created_at, updated_at"


class GapNotFoundError(LookupError):
    """复习/删除一个不存在的漏洞 id。

    不静默返回 None：调用方（工具 handler、HTTP 层）要能把它翻译成「这张卡片不存在」，
    而不是让模型以为自己复习成功了。
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _in_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")


async def record_gap(topic: str, detail: str, db_path: str | None = None) -> dict:
    """记一张漏洞卡片，首次复习时间 = now + 1 天。返回库里那一行（含新 id）。"""
    now = _now()
    async with get_db(db_path) as conn:
        cursor = await conn.execute(
            "INSERT INTO knowledge_gaps "
            "(topic, detail, interval_days, next_review_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (topic, detail, INITIAL_INTERVAL_DAYS, _in_days(INITIAL_INTERVAL_DAYS), now, now),
        )
        await conn.commit()
        rows = await conn.execute_fetchall(
            f"SELECT {_COLUMNS} FROM knowledge_gaps WHERE id = ?", (cursor.lastrowid,)
        )
    return dict(rows[0])


async def review_gap(gap_id: int, passed: bool, db_path: str | None = None) -> dict:
    """复习一张卡片并按结果排下一次：通过翻倍（封顶 180 天），未通过重置为 1 天。

    返回更新后的整行。id 不存在时抛 GapNotFoundError。
    """
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            f"SELECT {_COLUMNS} FROM knowledge_gaps WHERE id = ?", (gap_id,)
        )
        if not rows:
            raise GapNotFoundError(f"漏洞 #{gap_id} 不存在")
        interval = (
            min(rows[0]["interval_days"] * 2, MAX_INTERVAL_DAYS)
            if passed
            else INITIAL_INTERVAL_DAYS
        )
        now = _now()
        next_review_at = _in_days(interval)
        await conn.execute(
            "UPDATE knowledge_gaps SET interval_days = ?, next_review_at = ?, updated_at = ? "
            "WHERE id = ?",
            (interval, next_review_at, now, gap_id),
        )
        await conn.commit()
    return {
        **dict(rows[0]),
        "interval_days": interval,
        "next_review_at": next_review_at,
        "updated_at": now,
    }


async def list_gaps(
    db_path: str | None = None, *, due_only: bool = False
) -> list[dict]:
    """卡片列表，按 next_review_at 升序（过期的排最前，正好是该复习的顺序）。

    due_only=True 只返回已到期的，False 返回全部（含还没到期的）。
    """
    sql = f"SELECT {_COLUMNS} FROM knowledge_gaps"
    params: tuple = ()
    if due_only:
        sql += " WHERE next_review_at <= ?"
        params = (_now(),)
    sql += " ORDER BY next_review_at, id"
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(sql, params)
    return [dict(row) for row in rows]


async def due_gaps(db_path: str | None = None) -> list[dict]:
    """到期的卡片（定时复习任务的输入，T5 消费）。"""
    return await list_gaps(db_path, due_only=True)


async def delete_gap(gap_id: int, db_path: str | None = None) -> bool:
    """删除一张卡片，返回是否真的删掉了（False = 本来就没有这一行）。"""
    async with get_db(db_path) as conn:
        cursor = await conn.execute("DELETE FROM knowledge_gaps WHERE id = ?", (gap_id,))
        await conn.commit()
    return cursor.rowcount > 0
