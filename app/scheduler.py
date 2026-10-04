"""定时任务核心（T5）：cron 表、单进程调度器、自动会话与通知。

时间口径
    cron 表达式的语义是「任务创建时那台机器的本地墙钟时间」，所以每个任务在创建时记下
    本机 IANA 时区标识（`timezone`，见 local_timezone_name），此后永远按它解释——机器
    时区后来变了也不迁移既有任务（UI 显示实际解释时区并提示这一点）。库里存的时间
    （created_at / last_fired_at / occurrence / 通知）一律是 UTC ISO 8601，定长且带
    `+00:00`，字典序即时间序，所以「同一 occurrence」可以直接用字符串比较判断。

幂等
    `run_scheduler_tick` 先用任务保存的时区算出 `prev_occurrence(now)`，再用
    「任务 ID + occurrence」条件 UPDATE 事务性占位 `last_fired_at`；只有占位成功的那个
    调用者才去创建 fire task。因此并发 tick、重复 tick、同一进程里的多次扫描都只会触发
    一次。占位在跑 Agent **之前**写入且失败不回滚：一次失败不重试同一 occurrence，只写
    一条 error 通知。

不补跑
    只有落在 `[started_at - grace_window, now]` 内的 occurrence 才会触发（grace window
    等于一个 tick 周期）。休眠/关机期间错过的更早 occurrence 直接跳过，因为窗口下界固定、
    occurrence 只会向后推进，所以它永远进不了窗口。夏令时：缺失的墙钟时间不产生虚假时刻
    （croniter 把那一次落到跳变瞬间），回拨重复的墙钟时间只算一次（occurrence 统一按
    fold=0 换算成 UTC，第二次经过同一墙钟时刻得到同一个 UTC 瞬间，被占位挡下）。

失败隔离
    库里已存的坏 cron/坏时区在 tick 里被捕获 → 禁用该任务并只写一条通知，继续处理其他
    任务；fire task 的异常与超时写经清洗截断的 error 通知后吞掉，绝不抛回调度层，循环不死。

范围与已知限制
    - 单进程、单 worker：这里没有跨进程调度锁（数据库占位只防同一进程内的重复触发），
      多 worker / 多实例会重复触发，桌面入口负责单实例。
    - 通知表只增不清理：这是已知的长期增长点，后台清理推迟；查询侧一律只取最近 N 条。
    - prompt 里的 `{due_gaps}` 占位符在触发时替换成当前到期的复习卡片（T3 的 due_gaps），
      复习模板见 REVIEW_QUIZ_TEMPLATE。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tzlocal
from croniter import croniter

from app.agent.runtime import (
    SUPPORTED_MODES,
    UnknownModeError,
    ensure_session,
    run_agent,
)
from app.db import get_db
from app.study.gaps import due_gaps

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 扫描周期与 grace window：invariant 1 要求 grace window 等于一个 tick 周期
TICK_INTERVAL = 30.0
GRACE_WINDOW = TICK_INTERVAL

# fire task 的全局并发上限与单任务超时；shutdown 等它们结束的最大时间
MAX_FIRE_CONCURRENCY = 2
FIRE_TIMEOUT = 300.0
SHUTDOWN_WAIT = 30.0

MAX_NAME_LEN = 60
MAX_PROMPT_LEN = 4000

# 通知只查最近 N 条（默认），上限防住一个查询把整张表读出来
NOTIFICATION_LIMIT = 100
MAX_NOTIFICATION_LIMIT = 500
NOTIFICATION_BODY_LIMIT = 500

KIND_SUCCESS = "success"
KIND_ERROR = "error"

# 复习任务模板：占位符在触发时才替换（创建任务时卡片列表还是空的/已过期）
DUE_GAPS_PLACEHOLDER = "{due_gaps}"
# 一次最多列这么多张卡片：prompt 长度有上限，几百张卡片塞进去只会把上下文顶掉
MAX_DUE_GAPS = 20
REVIEW_QUIZ_TEMPLATE = (
    "复习一下到期的知识漏洞。先看卡片，再挑一张最该复习的考我，等我答完再判断对错，"
    "不要替我回答。\n\n"
    f"{DUE_GAPS_PLACEHOLDER}\n\n"
    "卡片看完后按复习结果调 review_knowledge_gap：答对传 passed=true，答错或答不上来传 false。"
)

_COLUMNS = (
    "id, name, cron, prompt, mode, timezone, enabled, last_fired_at, created_at"
)

# 通知正文要进的可能是托盘气泡与 UI 列表，长度与敏感串两头都要收：先洗掉密钥形状的串，
# 再按长度截断。清洗是尽力而为（第三方 SDK 的错误文本不受我们控制）。
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|access[_-]?token)\b\s*[:=]?\s*[A-Za-z0-9._\-]{8,}"),
)


class TaskConfigError(ValueError):
    """任务定义本身不可用（cron / 时区 / 名称 / 内容）。

    创建时拒绝（API 回 400）；库里已存的坏值在 tick 时被捕获并禁用该任务。
    """


class InvalidCronError(TaskConfigError):
    pass


class InvalidTimezoneError(TaskConfigError):
    pass


@dataclass(frozen=True)
class TaskRow:
    id: int
    name: str
    cron: str
    prompt: str
    mode: str
    timezone: str
    enabled: bool
    last_fired_at: str | None
    created_at: str


def _row_to_task(row) -> TaskRow:
    return TaskRow(
        id=row["id"],
        name=row["name"],
        cron=row["cron"],
        prompt=row["prompt"],
        mode=row["mode"],
        timezone=row["timezone"],
        enabled=bool(row["enabled"]),
        last_fired_at=row["last_fired_at"],
        created_at=row["created_at"],
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """统一成 UTC：naive 的按 UTC 处理（调用方都该给 aware，这里只是不让它静默算错）。"""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds")


# ---------- cron / 时区 ----------


def validate_cron(expr: str) -> str:
    """校验并归一（去空白）cron 表达式，非法时抛 InvalidCronError。

    不为「好看」收窄语法：croniter 认的 5/6/7 段表达式与 @daily 这类别名都放行，
    校验口径只有 croniter 一个来源。
    """
    expr = (expr or "").strip()
    if not expr or not croniter.is_valid(expr):
        raise InvalidCronError(f"不是有效的 cron 表达式：{expr!r}")
    return expr


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise InvalidTimezoneError(f"未知时区：{name}") from exc


def local_timezone_name() -> str:
    """本机时区标识（IANA 名，如 Asia/Shanghai），任务创建时保存。

    TZ 环境变量优先，方便 Linux 部署与测试钉住解释时区；其次交给 tzlocal——Windows
    注册表里存的是 "China Standard Time" 这类 Windows 时区 ID，zoneinfo 不认，stdlib
    也没有这张映射表。两条路都拿不到就报错而不是猜一个偏移：猜错会把 cron 整体平移，
    用户看到的触发时间和设置面板对不上。
    """
    candidates = [os.environ.get("TZ")]
    try:
        candidates.append(tzlocal.get_localzone_name())
    except Exception as exc:  # tzlocal 在异常环境里会抛各种东西，这里只降级不中断
        logger.warning("tzlocal 取本机时区失败：%s: %s", type(exc).__name__, exc)
    for name in candidates:
        if not name:
            continue
        try:
            resolve_timezone(name)
        except InvalidTimezoneError:
            continue
        return name
    raise InvalidTimezoneError("无法确定本机时区，请设置 TZ 环境变量（如 TZ=Asia/Shanghai）")


def prev_occurrence(cron: str, tz_name: str, now: datetime) -> datetime:
    """now 之前最近一次触发时刻（含正好落在 occurrence 上的那一秒），返回 UTC。

    - cron 按 tz_name 的本地墙钟解释。
    - croniter 的 get_prev 严格早于起点，所以先把 now 截到整秒、再从「+1 秒」往回找：
      正好命中 occurrence 的那一刻因此被算进来（否则每个整点都会退回到上一周期）。
      截秒不能省——croniter 的最小粒度是一秒，带着亚秒部分直接 +1 秒会跨过当前的这一秒，
      秒级 cron（如 `* * * * * *`）就会永远算到未来而永不触发。
    - 回拨的重复墙钟时间只算一次：匹配到的本地时间统一按 fold=0（第一次出现的那个偏移）
      换算成 UTC，第二次经过同一墙钟时得到同一个 UTC 瞬间。
    - 跳变的缺失墙钟时间不产生虚假本地时刻：croniter 把那一次落到跳变瞬间。
    """
    expr = validate_cron(cron)
    zone = resolve_timezone(tz_name)
    start = (_as_utc(now).replace(microsecond=0) + timedelta(seconds=1)).astimezone(zone)
    prev = croniter(expr, start).get_prev(datetime)
    wall = prev.replace(tzinfo=None)
    return wall.replace(tzinfo=zone, fold=0).astimezone(UTC)


# ---------- 任务 CRUD ----------


def task_to_dict(task: TaskRow, machine_timezone: str | None = None) -> dict:
    """API 输出。machine_timezone 是本机当前时区：与任务保存的不同就说明解释时区不再是
    本机时区（任务仍按创建时的那个走），UI 据此提示。"""
    return {
        "id": task.id,
        "name": task.name,
        "cron": task.cron,
        "prompt": task.prompt,
        "mode": task.mode,
        "timezone": task.timezone,
        "enabled": task.enabled,
        "last_fired_at": task.last_fired_at,
        "created_at": task.created_at,
        "machine_timezone": machine_timezone,
        "timezone_changed": bool(machine_timezone and machine_timezone != task.timezone),
    }


async def create_task(
    name: str,
    cron: str,
    prompt: str,
    mode: str = "work",
    timezone_name: str | None = None,
    db_path: str | None = None,
) -> TaskRow:
    """新建任务。校验全部在这里收口，坏输入不落库（不留下一条永远跑不起来的任务）。"""
    name = (name or "").strip()
    prompt = (prompt or "").strip()
    if not name:
        raise TaskConfigError("任务名不能为空")
    if len(name) > MAX_NAME_LEN:
        raise TaskConfigError(f"任务名不能超过 {MAX_NAME_LEN} 字")
    if not prompt:
        raise TaskConfigError("任务内容不能为空")
    if len(prompt) > MAX_PROMPT_LEN:
        raise TaskConfigError(f"任务内容不能超过 {MAX_PROMPT_LEN} 字")
    if mode not in SUPPORTED_MODES:
        raise UnknownModeError(f"不支持的模式：{mode}（可选：{'、'.join(SUPPORTED_MODES)}）")
    expr = validate_cron(cron)
    tz_name = timezone_name or local_timezone_name()
    resolve_timezone(tz_name)
    async with get_db(db_path) as conn:
        cursor = await conn.execute(
            "INSERT INTO scheduled_tasks "
            "(name, cron, prompt, mode, timezone, enabled, last_fired_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, 1, NULL, ?)",
            (name, expr, prompt, mode, tz_name, _iso(_now())),
        )
        await conn.commit()
        rows = await conn.execute_fetchall(
            f"SELECT {_COLUMNS} FROM scheduled_tasks WHERE id = ?", (cursor.lastrowid,)
        )
    return _row_to_task(rows[0])


async def list_tasks(db_path: str | None = None) -> list[TaskRow]:
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(f"SELECT {_COLUMNS} FROM scheduled_tasks ORDER BY id")
    return [_row_to_task(row) for row in rows]


async def get_task(task_id: int, db_path: str | None = None) -> TaskRow | None:
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            f"SELECT {_COLUMNS} FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
    return _row_to_task(rows[0]) if rows else None


async def set_task_enabled(task_id: int, enabled: bool, db_path: str | None = None) -> bool:
    """启停任务，返回是否命中一行（False = 任务不存在）。只影响后续 occurrence：在途的
    fire task 不被打断，跑完照常写通知（删除同理）。"""
    async with get_db(db_path) as conn:
        cursor = await conn.execute(
            "UPDATE scheduled_tasks SET enabled = ? WHERE id = ?", (1 if enabled else 0, task_id)
        )
        await conn.commit()
    return cursor.rowcount > 0


async def delete_task(task_id: int, db_path: str | None = None) -> bool:
    """删任务本身。历史会话与通知都是用户可见的结果，不跟着删。"""
    async with get_db(db_path) as conn:
        cursor = await conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await conn.commit()
    return cursor.rowcount > 0


async def _disable_broken_task(task: TaskRow, exc: Exception, db_path: str | None) -> None:
    """禁用库里已经坏掉的任务（cron/时区），并只写一条通知。

    禁用是为了不去重写同一条通知：每轮 tick 都报一次同样的坏 cron 只会把通知列表刷满。
    """
    logger.warning("任务 #%s 定义不可用，已禁用：%s", task.id, exc)
    await set_task_enabled(task.id, False, db_path)
    await add_notification(
        task.id,
        None,
        KIND_ERROR,
        f"任务「{task.name}」已停用",
        _sanitize(f"任务定义不可用，已自动停用：{exc}"),
        db_path,
    )


# ---------- 通知 ----------


def _sanitize(text: str, limit: int = NOTIFICATION_BODY_LIMIT) -> str:
    """通知正文：洗掉密钥形状的串再截断，保证长度可预期、不把内部细节整段抛给托盘。"""
    out = (text or "").strip()
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("***", out)
    if len(out) > limit:
        out = out[: max(1, limit - 1)] + "…"
    return out or "（没有可显示的正文）"


async def add_notification(
    task_id: int | None,
    session_id: str | None,
    kind: str,
    title: str,
    body: str,
    db_path: str | None = None,
) -> int:
    async with get_db(db_path) as conn:
        cursor = await conn.execute(
            "INSERT INTO notifications (task_id, session_id, kind, title, body, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, session_id, kind, title, body, _iso(_now())),
        )
        await conn.commit()
    return cursor.lastrowid


async def list_notifications(
    limit: int = NOTIFICATION_LIMIT, db_path: str | None = None
) -> list[dict]:
    """最近 N 条通知（新的在前）。托盘/UI 都从这里取，按 id 倒序即可（id 单调递增）。"""
    limit = max(1, min(int(limit), MAX_NOTIFICATION_LIMIT))
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, task_id, session_id, kind, title, body, created_at FROM notifications "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    return [dict(row) for row in rows]


# ---------- 触发 ----------


def _format_due_gaps(gaps: list[dict]) -> str:
    if not gaps:
        return "（今天没有到期的复习卡片）"
    lines = [
        f"- #{gap['id']} {gap['topic']}：{gap['detail']}（间隔 {gap['interval_days']} 天）"
        for gap in gaps[:MAX_DUE_GAPS]
    ]
    if len(gaps) > MAX_DUE_GAPS:
        lines.append(f"（另有 {len(gaps) - MAX_DUE_GAPS} 张未列出）")
    return "\n".join(lines)


async def render_prompt(prompt: str, db_path: str | None = None) -> str:
    """把 prompt 里的 {due_gaps} 替换成当前到期的复习卡片；没有占位符就原样返回。

    占位符在**触发时**才求值（不是创建时）：卡片会被复习、间隔会变，创建时算出来的
    列表到点已经过期了。
    """
    if DUE_GAPS_PLACEHOLDER not in prompt:
        return prompt
    return prompt.replace(DUE_GAPS_PLACEHOLDER, _format_due_gaps(await due_gaps(db_path)))


async def _session_for_occurrence(
    task_id: int, occurrence: str, db_path: str | None
) -> str | None:
    """这次 occurrence 已经建出来的会话（超时/异常通知要能指到它，用户才好去看半截结果）。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id FROM sessions WHERE scheduled_task_id = ? AND scheduled_occurrence_at = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (str(task_id), occurrence),
        )
    return rows[0]["id"] if rows else None


async def _notify_failure(
    task: TaskRow, occurrence: str, message: str, db_path: str | None = None
) -> None:
    session_id = await _session_for_occurrence(task.id, occurrence, db_path)
    try:
        await add_notification(
            task.id,
            session_id,
            KIND_ERROR,
            f"任务「{task.name}」执行失败",
            _sanitize(message),
            db_path,
        )
    except Exception:  # 写通知失败不能再往调度层抛：否则一次失败会被放大成循环崩溃
        logger.exception("写任务失败通知出错：task=%s occurrence=%s", task.id, occurrence)


async def _run_task(task: TaskRow, occurrence: str, db_path: str | None) -> None:
    """一次任务的完整执行：建自动会话 → 消费 run_agent 的完整事件流 → 写通知。

    只有「收到 done 且没有 error」才算成功；error 事件、事件流没有结束事件、抛异常
    都写 error 通知（结果本身不落通知表以外的地方——会话里已有完整记录）。
    """
    prompt = await render_prompt(task.prompt, db_path)
    session_id = await ensure_session(
        None,
        db_path,
        mode=task.mode,
        source="scheduled",
        scheduled_task_id=str(task.id),
        scheduled_occurrence_at=occurrence,
    )
    answer: str | None = None
    error: str | None = None
    # 不用 mode= 调用 run_agent：会话刚建好，让它从会话读模式（单一事实来源）
    async for event in run_agent(session_id, prompt, db_path):
        if event.type == "done":
            answer = event.data.get("text") or ""
        elif event.type == "error":
            error = event.data.get("message") or "Agent 报错"
    if error is not None:
        await add_notification(
            task.id,
            session_id,
            KIND_ERROR,
            f"任务「{task.name}」执行失败",
            _sanitize(error),
            db_path,
        )
    elif answer is None:
        await add_notification(
            task.id,
            session_id,
            KIND_ERROR,
            f"任务「{task.name}」执行失败",
            _sanitize("Agent 事件流没有结束事件（done/error），按失败处理"),
            db_path,
        )
    else:
        await add_notification(
            task.id,
            session_id,
            KIND_SUCCESS,
            f"任务「{task.name}」已完成",
            _sanitize(answer),
            db_path,
        )


async def fire_task(task: TaskRow, occurrence: str, db_path: str | None = None) -> None:
    """执行一次任务。业务异常与超时都在这里消化（写 error 通知），不抛回调度层。

    并发上限是全局 semaphore，超时是单任务 timeout。关闭时被取消的 fire task 由
    stop_scheduler 负责写通知，这里不吞 CancelledError。
    """
    try:
        async with _fire_semaphore():
            await asyncio.wait_for(_run_task(task, occurrence, db_path), FIRE_TIMEOUT)
    except TimeoutError:
        await _notify_failure(
            task, occurrence, f"任务执行超过 {FIRE_TIMEOUT:g} 秒未完成，已取消", db_path
        )
    except Exception as exc:
        await _notify_failure(task, occurrence, f"{type(exc).__name__}: {exc}", db_path)


# ---------- 进程内调度状态 ----------
#
# 单进程单 worker 是本期明确约束（见模块说明），所以状态就放在模块级。fire task 与
# semaphore 都记在这里：stop_scheduler 靠 _active_fires 等在途任务、超时取消并写通知；
# 同一任务已有在途 fire 时 tick 跳过重入。

_active_fires: dict[asyncio.Task, tuple[TaskRow, str]] = {}
_scheduler_task: asyncio.Task | None = None
_accepting = True
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _fire_semaphore() -> asyncio.Semaphore:
    """全局并发 semaphore。按事件循环惰性创建：asyncio 原语一旦绑定到某个循环就不能再被
    另一个循环使用，而测试每个用例都是新循环。"""
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore, _semaphore_loop = asyncio.Semaphore(MAX_FIRE_CONCURRENCY), loop
    return _semaphore


def active_fire_count() -> int:
    return len(_active_fires)


def is_running() -> bool:
    return _scheduler_task is not None and not _scheduler_task.done()


def _has_active_fire(task_id: int) -> bool:
    return any(task.id == task_id for task, _ in _active_fires.values())


def _spawn_fire(task: TaskRow, occurrence: str, db_path: str | None) -> asyncio.Task:
    fire = asyncio.create_task(
        fire_task(task, occurrence, db_path), name=f"fire-task:{task.id}:{occurrence}"
    )
    _active_fires[fire] = (task, occurrence)

    def _done(finished: asyncio.Task) -> None:
        _active_fires.pop(finished, None)
        if not finished.cancelled() and finished.exception() is not None:
            # fire_task 理应吞掉所有业务异常；这里只兜住我们自己写的 bug
            logger.error("fire task 抛出未处理异常", exc_info=finished.exception())

    fire.add_done_callback(_done)
    return fire


async def _claim_occurrence(task_id: int, occurrence: str, db_path: str | None) -> bool:
    """事务性占位：只有把 last_fired_at 从更早的值推到本次 occurrence 的调用者得到 True。

    条件更新本身是原子的（同一进程内两个连接也只有一个能改到行），失败方拿 0 行。
    """
    async with get_db(db_path) as conn:
        cursor = await conn.execute(
            "UPDATE scheduled_tasks SET last_fired_at = ? WHERE id = ? "
            "AND (last_fired_at IS NULL OR last_fired_at < ?)",
            (occurrence, task_id, occurrence),
        )
        await conn.commit()
    return cursor.rowcount == 1


async def run_scheduler_tick(
    now: datetime,
    started_at: datetime,
    grace_window: float = GRACE_WINDOW,
    db_path: str | None = None,
) -> list[int]:
    """扫描一轮：到点的任务各触发一次，返回本轮真正触发的任务 id。

    窗口是 `[started_at - grace_window, now]`（闭区间）：更早的 occurrence 直接跳过，
    不补跑。occurrence 用任务保存的时区算；占位先于创建 fire task，所以并发/重复 tick
    里只有一个赢家真的开会话。串行扫描：一个任务的坏数据不影响其他任务。
    """
    now = _as_utc(now)
    lower = _as_utc(started_at) - timedelta(seconds=max(0.0, grace_window))
    fired: list[int] = []
    for task in await list_tasks(db_path):
        if not _accepting:
            # 关闭流程已开始：不再创建新的 fire task（在途的由 stop_scheduler 处理）
            break
        if not task.enabled or _has_active_fire(task.id):
            continue
        try:
            occurrence = prev_occurrence(task.cron, task.timezone, now)
        except TaskConfigError as exc:
            await _disable_broken_task(task, exc, db_path)
            continue
        if occurrence < lower or occurrence > now:
            continue
        key = _iso(occurrence)
        if not await _claim_occurrence(task.id, key, db_path):
            continue
        _spawn_fire(task, key, db_path)
        fired.append(task.id)
    return fired


async def scheduler_loop(interval: float = TICK_INTERVAL) -> None:
    """后台扫描循环：串行扫描、按 tick 周期睡。单轮异常只记日志，循环不死。

    started_at 固定为循环启动时刻：grace window 因此锚在「调度器启动」而不是「这一轮」，
    长时间运行也不会把窗口一路推到现在。
    """
    started_at = _now()
    while True:
        try:
            await run_scheduler_tick(_now(), started_at, GRACE_WINDOW)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("调度扫描失败，下一轮继续")
        await asyncio.sleep(interval)


def start_scheduler(interval: float = TICK_INTERVAL) -> asyncio.Task:
    """启动调度：打开 accepting 并（首次）起一个后台循环。重复调用返回同一个循环。

    单 worker 由这里保证：进程里最多只有一个扫描循环。
    """
    global _scheduler_task, _accepting
    _accepting = True
    if is_running():
        return _scheduler_task
    _scheduler_task = asyncio.create_task(scheduler_loop(interval), name="scheduler-loop")
    return _scheduler_task


async def wait_for_active_fires(timeout: float = SHUTDOWN_WAIT) -> list[asyncio.Task]:
    """等在途 fire task 结束，返回超时还没结束的那些（不取消、不写通知）。"""
    pending = [fire for fire in list(_active_fires) if not fire.done()]
    if not pending:
        return []
    _, still = await asyncio.wait(pending, timeout=timeout)
    return sorted(still, key=lambda fire: fire.get_name())


async def stop_scheduler(wait: float = SHUTDOWN_WAIT) -> None:
    """关闭调度：先不再接收新工作，再取消扫描循环，最后等在途 fire task。

    超时还没结束的取消掉并各写一条通知（用户至少知道那次任务没跑完）。drain（记忆/trace）
    不在这里做，由 main.py 的 lifespan 用 try/finally 保证。
    """
    global _accepting, _scheduler_task
    _accepting = False
    loop_task, _scheduler_task = _scheduler_task, None
    if loop_task is not None:
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
    leftover = await wait_for_active_fires(wait)
    if not leftover:
        return
    known = [(fire, *_active_fires[fire]) for fire in leftover if fire in _active_fires]
    for fire in leftover:
        fire.cancel()
    await asyncio.gather(*leftover, return_exceptions=True)
    for _, task, occurrence in known:
        await _notify_failure(
            task, occurrence, f"应用退出时任务还在执行，已取消（等待超过 {wait:g} 秒）"
        )
