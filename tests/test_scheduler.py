"""定时任务核心（T5）的用例：cron/时区/DST 计算、幂等 tick、失败隔离、CRUD/API 与 lifespan。

隔离方式与 tests/test_api.py 一致：db 夹具把 settings.db_path 指到 tmp 并 init_db
（ASGITransport 不跑 lifespan，建表的责任在夹具）。fire task 真正执行的 Agent 一律换成
桩（scheduler.run_agent），只有一条端到端用例走真实 run_agent（LLM 也是桩，不联网）。

模块里的调度状态是进程内的（单进程单 worker 是本期的明确约束），所以每个用例用
clean_scheduler 夹具把 accepting 重新打开、收尾时等在途 fire 结束。
"""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app import scheduler
from app.agent import runtime
from app.agent.runtime import AgentEvent
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk
from app.main import app
from app.memory import writer as memory_writer

DIM = 8
UTC = timezone.utc
SHANGHAI = "Asia/Shanghai"
NEW_YORK = "America/New_York"
DAILY = "0 9 * * *"
EVERY_MINUTE = "* * * * *"


# ---------- 夹具 ----------


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def clean_scheduler(monkeypatch):
    """每个用例从「accepting=true、无在途 fire」开始。

    stop_scheduler 会把 accepting 置 false（这是关闭语义，不能复原），所以用例开始时
    显式重新打开。收尾只等在途任务结束并取消残留，不写通知——此时 monkeypatch 已还原，
    写通知会落到开发库上。
    """
    monkeypatch.setattr(scheduler, "_accepting", True)
    yield
    leftover = await scheduler.wait_for_active_fires(2.0)
    for fire in leftover:
        fire.cancel()
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)


# ---------- 桩 ----------


def fake_agent(monkeypatch, events, *, gate: asyncio.Event | None = None) -> list[dict]:
    """把 scheduler.run_agent 换成按脚本吐事件的异步生成器，返回收到的调用列表。"""
    calls: list[dict] = []

    async def _run(session_id, message, db_path=None, **kwargs):
        calls.append(
            {"session_id": session_id, "message": message, "db_path": db_path, "kwargs": kwargs}
        )
        if gate is not None:
            await gate.wait()
        for event in events:
            yield event

    monkeypatch.setattr(scheduler, "run_agent", _run)
    return calls


def raising_agent(monkeypatch, message: str = "boom") -> None:
    async def _run(session_id, message_, db_path=None, **kwargs):
        raise RuntimeError(message)
        yield  # pragma: no cover —— 让它成为异步生成器

    monkeypatch.setattr(scheduler, "run_agent", _run)


def hanging_agent(monkeypatch) -> None:
    async def _run(session_id, message, db_path=None, **kwargs):
        await asyncio.sleep(30)
        yield AgentEvent("done", {"session_id": session_id, "text": "太晚了"})

    monkeypatch.setattr(scheduler, "run_agent", _run)


def done_event(text: str = "复习完成") -> AgentEvent:
    return AgentEvent("done", {"session_id": "s", "text": text})


# ---------- 建数据与读数据的辅助 ----------


async def seed_task(
    db,
    *,
    name: str = "每日复习",
    cron: str = DAILY,
    prompt: str = "复习一下今天的漏洞",
    mode: str = "work",
    timezone_name: str = SHANGHAI,
    enabled: int = 1,
    last_fired_at: str | None = None,
) -> int:
    """绕过校验直接落一行，用来构造库里已有的坏 cron/坏时区/禁用任务。返回新 id。"""
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO scheduled_tasks "
            "(name, cron, prompt, mode, timezone, enabled, last_fired_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                cron,
                prompt,
                mode,
                timezone_name,
                enabled,
                last_fired_at,
                "2026-05-01T00:00:00+00:00",
            ),
        )
        await conn.commit()
        return cursor.lastrowid


async def task_row(db, task_id: int) -> dict:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,))
    return dict(rows[0]) if rows else {}


async def sessions_of(db) -> list[dict]:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, mode, source, scheduled_task_id, scheduled_occurrence_at FROM sessions "
            "ORDER BY created_at, id"
        )
    return [dict(r) for r in rows]


async def notifications_of(db) -> list[dict]:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall("SELECT * FROM notifications ORDER BY id")
    return [dict(r) for r in rows]


async def seed_gap(db, topic: str, detail: str, next_review_at: str) -> int:
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO knowledge_gaps "
            "(topic, detail, interval_days, next_review_at, created_at, updated_at) "
            "VALUES (?, ?, 1, ?, ?, ?)",
            (topic, detail, next_review_at, next_review_at, next_review_at),
        )
        await conn.commit()
        return cursor.lastrowid


# ---------- cron 校验 / occurrence 计算 ----------


@pytest.mark.parametrize("expr", ["0 9 * * *", "*/15 * * * *", "@daily", "0 9 * * 1-5"])
def test_validate_cron_accepts_standard_expressions(expr):
    assert scheduler.validate_cron(expr) == expr


@pytest.mark.parametrize("expr", ["", "  ", "bogus", "0 9 * *", "*/0 * * * *", "60 * * * *"])
def test_validate_cron_rejects_garbage(expr):
    with pytest.raises(scheduler.InvalidCronError):
        scheduler.validate_cron(expr)


async def test_create_task_rejects_invalid_cron(db):
    with pytest.raises(scheduler.InvalidCronError):
        await scheduler.create_task("坏任务", "bogus", "内容", db_path=db)
    assert await scheduler.list_tasks(db) == []


async def test_create_task_rejects_unknown_timezone(db):
    with pytest.raises(scheduler.InvalidTimezoneError):
        await scheduler.create_task("坏时区", DAILY, "内容", timezone_name="Mars/Olympus", db_path=db)


async def test_prev_occurrence_uses_the_tasks_timezone(db):
    now = datetime(2026, 5, 1, 2, 0, tzinfo=UTC)  # 上海 10:00，纽约 22:00（前一天）
    assert scheduler.prev_occurrence(DAILY, SHANGHAI, now) == datetime(2026, 5, 1, 1, 0, tzinfo=UTC)
    assert scheduler.prev_occurrence(DAILY, NEW_YORK, now) == datetime(
        2026, 4, 30, 13, 0, tzinfo=UTC
    )


async def test_prev_occurrence_includes_the_exact_moment(db):
    exact = datetime(2026, 5, 1, 1, 0, tzinfo=UTC)  # 上海 09:00 整
    assert scheduler.prev_occurrence(DAILY, SHANGHAI, exact) == exact
    # 差一秒就不算发生过：返回前一天那次
    assert scheduler.prev_occurrence(DAILY, SHANGHAI, exact - timedelta(seconds=1)) == datetime(
        2026, 4, 30, 1, 0, tzinfo=UTC
    )
    assert scheduler.prev_occurrence(DAILY, SHANGHAI, exact + timedelta(seconds=1)) == exact


async def test_prev_occurrence_skips_the_missing_hour_on_spring_forward(db):
    """2026-03-08 纽约 02:00→03:00；"30 2 * * *" 这天没有 02:30，落到跳变瞬间（03:00 EDT）。"""
    before = datetime(2026, 3, 8, 6, 59, tzinfo=UTC)
    after = datetime(2026, 3, 8, 8, 30, tzinfo=UTC)
    assert scheduler.prev_occurrence("30 2 * * *", NEW_YORK, before) == datetime(
        2026, 3, 7, 7, 30, tzinfo=UTC
    )
    assert scheduler.prev_occurrence("30 2 * * *", NEW_YORK, after) == datetime(
        2026, 3, 8, 7, 0, tzinfo=UTC
    )
    # 跳变当天只有这一次：更晚再算还是同一个瞬间，不会多出一次
    later = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
    assert scheduler.prev_occurrence("30 2 * * *", NEW_YORK, later) == datetime(
        2026, 3, 8, 7, 0, tzinfo=UTC
    )


async def test_prev_occurrence_counts_the_repeated_hour_once(db):
    """2026-11-01 纽约 02:00→01:00；本地 01:30 出现两次，只算第一次那个 UTC 瞬间。"""
    first_pass = datetime(2026, 11, 1, 5, 35, tzinfo=UTC)  # 01:35 EDT
    second_pass = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)  # 01:30 EST（回拨后的同一墙钟）
    assert scheduler.prev_occurrence("30 1 * * *", NEW_YORK, first_pass) == datetime(
        2026, 11, 1, 5, 30, tzinfo=UTC
    )
    assert scheduler.prev_occurrence("30 1 * * *", NEW_YORK, second_pass) == datetime(
        2026, 11, 1, 5, 30, tzinfo=UTC
    )


async def test_prev_occurrence_rejects_unknown_timezone(db):
    with pytest.raises(scheduler.InvalidTimezoneError):
        scheduler.prev_occurrence(DAILY, "Mars/Olympus", datetime(2026, 5, 1, tzinfo=UTC))


def test_grace_window_equals_one_tick_interval():
    """invariant 1：grace window 等于一个 tick 周期。"""
    assert scheduler.GRACE_WINDOW == scheduler.TICK_INTERVAL


async def test_local_timezone_name_is_a_usable_zone(db):
    name = scheduler.local_timezone_name()
    assert scheduler.resolve_timezone(name) is not None


# ---------- tick：幂等、窗口、隔离 ----------


async def test_tick_fires_due_task_once(db, monkeypatch):
    calls = fake_agent(monkeypatch, [done_event()])
    task = await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == [task.id]
    await scheduler.wait_for_active_fires(2.0)

    assert len(calls) == 1
    assert calls[0]["message"] == "干活"
    assert [s["source"] for s in await sessions_of(db)] == ["scheduled"]
    assert (await task_row(db, task.id))["last_fired_at"] == "2026-05-01T10:00:00+00:00"

    # 同一个 occurrence 再来一轮（无论 now 前进多少）都不再触发
    later = now + timedelta(seconds=5)
    assert await scheduler.run_scheduler_tick(later, now, 30.0, db_path=db) == []
    await scheduler.wait_for_active_fires(2.0)
    assert len(calls) == 1
    assert len(await sessions_of(db)) == 1
    assert len(await notifications_of(db)) == 1


async def test_tick_window_is_inclusive_on_the_lower_bound(db, monkeypatch):
    calls = fake_agent(monkeypatch, [done_event()])
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 30, tzinfo=UTC)  # prev = 10:00:00 = started_at - grace

    assert len(await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db)) == 1
    await scheduler.wait_for_active_fires(2.0)
    assert len(calls) == 1


async def test_tick_does_not_catch_up_missed_occurrences(db, monkeypatch):
    calls = fake_agent(monkeypatch, [done_event()])
    task = await scheduler.create_task("每日九点", DAILY, "复习", db_path=db)
    # 上海 18:00 启动：今天的 09:00（01:00Z）已经是 8 小时前，早于 grace 窗口
    now = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == []
    assert await scheduler.run_scheduler_tick(now + timedelta(minutes=1), now, 30.0, db_path=db) == []
    await scheduler.wait_for_active_fires(1.0)

    assert calls == []
    assert await sessions_of(db) == []
    assert (await task_row(db, task.id))["last_fired_at"] is None


async def test_concurrent_ticks_fire_the_occurrence_once(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    results = await asyncio.gather(
        scheduler.run_scheduler_tick(now, now, 30.0, db_path=db),
        scheduler.run_scheduler_tick(now, now, 30.0, db_path=db),
    )
    await scheduler.wait_for_active_fires(2.0)

    assert sum(len(r) for r in results) == 1
    assert len(await sessions_of(db)) == 1
    assert len(await notifications_of(db)) == 1


async def test_tick_skips_reentrant_fire_of_the_same_task(db, monkeypatch):
    gate = asyncio.Event()
    calls = fake_agent(monkeypatch, [done_event()], gate=gate)
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    assert len(await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db)) == 1
    # 第一个 fire 还挂着：同一任务不能有第二个在途任务
    assert await scheduler.run_scheduler_tick(now + timedelta(seconds=1), now, 30.0, db_path=db) == []
    gate.set()
    await scheduler.wait_for_active_fires(2.0)

    assert len(calls) == 1
    assert len(await sessions_of(db)) == 1


async def test_tick_skips_disabled_tasks(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    task = (await scheduler.list_tasks(db))[0]
    assert await scheduler.set_task_enabled(task.id, False, db_path=db) is True
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == []
    assert await sessions_of(db) == []


async def test_tick_disables_broken_cron_and_keeps_going(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    broken_id = await seed_task(db, name="坏任务", cron="bogus")
    good = await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == [good.id]
    await scheduler.wait_for_active_fires(2.0)

    assert (await task_row(db, broken_id))["enabled"] == 0
    errors = [row for row in await notifications_of(db) if row["kind"] == "error"]
    assert len(errors) == 1
    assert errors[0]["task_id"] == broken_id
    assert "cron" in errors[0]["body"]
    # 坏 cron 之前的任务被禁用后，同一轮里的下一个任务照常触发：坏数据不杀循环
    assert [row["kind"] for row in await notifications_of(db)] == ["error", "success"]

    # 已禁用的坏任务不会每轮重复写通知，其他任务继续跑
    assert await scheduler.run_scheduler_tick(now + timedelta(minutes=1), now, 30.0, db_path=db) == [
        good.id
    ]
    await scheduler.wait_for_active_fires(2.0)
    assert len([row for row in await notifications_of(db) if row["kind"] == "error"]) == 1


async def test_tick_disables_task_with_broken_timezone(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    broken_id = await seed_task(db, name="坏时区", timezone_name="Mars/Olympus")
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == []
    assert (await task_row(db, broken_id))["enabled"] == 0
    assert len(await notifications_of(db)) == 1


async def test_tick_does_not_spawn_fires_after_stop(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)
    monkeypatch.setattr(scheduler, "_accepting", False)

    assert await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db) == []
    assert await sessions_of(db) == []


# ---------- fire_task ----------


async def test_fire_task_creates_scheduled_session_with_task_metadata(db, monkeypatch):
    calls = fake_agent(monkeypatch, [done_event("复习完成：RAG")])
    task = await scheduler.create_task("每日复习", DAILY, "复习", mode="work", db_path=db)
    occurrence = "2026-05-01T01:00:00+00:00"

    await scheduler.fire_task(task, occurrence, db_path=db)

    sessions = await sessions_of(db)
    assert len(sessions) == 1
    assert sessions[0]["mode"] == "work"
    assert sessions[0]["source"] == "scheduled"
    assert sessions[0]["scheduled_task_id"] == str(task.id)
    assert sessions[0]["scheduled_occurrence_at"] == occurrence
    # 会话模式由任务决定：给 run_agent 不传 mode，让它从会话读
    assert calls[0]["kwargs"] == {}
    assert calls[0]["session_id"] == sessions[0]["id"]

    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "success"
    assert rows[0]["task_id"] == task.id
    assert rows[0]["session_id"] == sessions[0]["id"]
    assert "复习完成：RAG" in rows[0]["body"]


async def test_fire_task_keeps_the_tasks_mode_for_chat_tasks(db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    task = await scheduler.create_task("聊两句", DAILY, "聊聊", mode="chat", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    assert [s["mode"] for s in await sessions_of(db)] == ["chat"]


async def test_fire_task_renders_due_gaps_into_the_prompt(db, monkeypatch):
    calls = fake_agent(monkeypatch, [done_event()])
    await seed_gap(db, "向量检索", "分不清 HNSW 与 IVF", "2000-01-01T00:00:00+00:00")
    await seed_gap(db, "明天再说", "还没到期", "2999-01-01T00:00:00+00:00")
    task = await scheduler.create_task(
        "复习小测", DAILY, scheduler.REVIEW_QUIZ_TEMPLATE, db_path=db
    )

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    message = calls[0]["message"]
    assert scheduler.DUE_GAPS_PLACEHOLDER not in message
    assert "向量检索" in message and "分不清 HNSW 与 IVF" in message
    assert "明天再说" not in message


async def test_render_prompt_without_placeholder_is_untouched(db):
    task = await scheduler.create_task("随手记", DAILY, "把今天的想法整理成一条笔记", db_path=db)
    assert await scheduler.render_prompt(task.prompt, db) == "把今天的想法整理成一条笔记"


async def test_render_prompt_with_no_due_gaps_says_so(db):
    assert scheduler.DUE_GAPS_PLACEHOLDER in scheduler.REVIEW_QUIZ_TEMPLATE

    rendered = await scheduler.render_prompt(scheduler.REVIEW_QUIZ_TEMPLATE, db)

    assert scheduler.DUE_GAPS_PLACEHOLDER not in rendered
    assert "没有到期" in rendered


async def test_fire_task_error_event_writes_error_notification_and_swallows(db, monkeypatch):
    fake_agent(
        monkeypatch,
        [
            AgentEvent("text_delta", {"text": "开始"}),
            AgentEvent("error", {"message": "LLM 超时", "session_id": "s"}),
        ],
    )
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)  # 不抛

    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "error"
    assert "LLM 超时" in rows[0]["body"]
    assert rows[0]["session_id"] == (await sessions_of(db))[0]["id"]


async def test_fire_task_exception_writes_error_notification_and_swallows(db, monkeypatch):
    raising_agent(monkeypatch, "检索后端挂了")
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "error"
    assert "检索后端挂了" in rows[0]["body"]
    assert rows[0]["session_id"] == (await sessions_of(db))[0]["id"]


async def test_fire_task_stream_without_done_is_a_failure(db, monkeypatch):
    fake_agent(monkeypatch, [AgentEvent("text_delta", {"text": "半截回答"})])
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "error"


async def test_fire_task_timeout_writes_error_notification_and_swallows(db, monkeypatch):
    hanging_agent(monkeypatch)
    monkeypatch.setattr(scheduler, "FIRE_TIMEOUT", 0.05)
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)  # 不抛

    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "error"
    assert "超时" in rows[0]["body"] or "未完成" in rows[0]["body"]
    assert rows[0]["session_id"] is not None


async def test_error_notification_body_is_truncated_and_cleaned(db, monkeypatch):
    secret = "sk-abcdef1234567890"
    raising_agent(monkeypatch, f"调用失败 api_key={secret} " + "长" * 5000)
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)

    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    body = (await notifications_of(db))[0]["body"]
    assert len(body) <= scheduler.NOTIFICATION_BODY_LIMIT
    assert secret not in body
    assert "***" in body


async def test_fire_task_end_to_end_with_the_real_agent(db, monkeypatch):
    """不换 run_agent 的一条：真实 ReAct 循环 + 桩 LLM，验证任务真的能跑完并落通知。"""

    class _StubLLM:
        async def chat_stream(self, messages, tools=None):
            yield StreamChunk(text_delta="今天该复习 RAG")
            yield StreamChunk(finish=True, tool_calls=[])

        async def chat(self, messages, tools=None):
            return ChatResult(text="复习")

    class _SilentWriterLLM:
        async def chat(self, messages, tools=None):
            return ChatResult(text="[]")

    async def _no_chunks(query, k=8, mode="hybrid", db_path=None):
        return []

    monkeypatch.setattr(runtime, "get_llm", lambda *a, **kw: _StubLLM())
    monkeypatch.setattr(runtime, "hybrid_search", _no_chunks)
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    task = await scheduler.create_task("每日复习", DAILY, "复习一下", db_path=db)

    try:
        await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)
    finally:
        await runtime.drain_memory_writes()

    rows = await notifications_of(db)
    assert [r["kind"] for r in rows] == ["success"]
    assert "今天该复习 RAG" in rows[0]["body"]
    assert [s["source"] for s in await sessions_of(db)] == ["scheduled"]


# ---------- 通知查询 ----------


async def test_list_notifications_returns_newest_first_with_limit(db):
    for i in range(3):
        await scheduler.add_notification(
            None, None, "success", f"任务 {i}", f"结果 {i}", db_path=db
        )

    rows = await scheduler.list_notifications(db_path=db)
    assert [r["title"] for r in rows] == ["任务 2", "任务 1", "任务 0"]

    assert [r["title"] for r in await scheduler.list_notifications(limit=1, db_path=db)] == ["任务 2"]


# ---------- API ----------


async def test_create_task_api_stores_local_timezone_and_default_mode(client, db, monkeypatch):
    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: SHANGHAI)

    resp = await client.post(
        "/api/tasks", json={"name": "每日复习", "cron": DAILY, "prompt": "复习一下"}
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["mode"] == "work"
    assert body["enabled"] is True
    assert body["timezone"] == SHANGHAI
    assert body["machine_timezone"] == SHANGHAI
    assert body["timezone_changed"] is False
    assert body["last_fired_at"] is None

    row = await task_row(db, body["id"])
    assert row["timezone"] == SHANGHAI and row["enabled"] == 1


async def test_create_task_api_accepts_chat_mode(client, db, monkeypatch):
    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: SHANGHAI)
    resp = await client.post(
        "/api/tasks",
        json={"name": "聊两句", "cron": DAILY, "prompt": "聊聊", "mode": "chat"},
    )
    assert resp.status_code == 201
    assert resp.json()["mode"] == "chat"


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"name": "a", "cron": "bogus", "prompt": "p"}, "cron"),
        ({"name": "  ", "cron": DAILY, "prompt": "p"}, "任务名"),
        ({"name": "x" * (scheduler.MAX_NAME_LEN + 1), "cron": DAILY, "prompt": "p"}, "任务名"),
        ({"name": "a", "cron": DAILY, "prompt": "   "}, "任务内容"),
        (
            {"name": "a", "cron": DAILY, "prompt": "x" * (scheduler.MAX_PROMPT_LEN + 1)},
            "任务内容",
        ),
        ({"name": "a", "cron": DAILY, "prompt": "p", "mode": "code"}, "code 模式"),
        ({"name": "a", "cron": DAILY, "prompt": "p", "mode": "weird"}, "不支持的模式"),
    ],
)
async def test_create_task_api_rejects_bad_input(client, db, monkeypatch, payload, fragment):
    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: SHANGHAI)
    resp = await client.post("/api/tasks", json=payload)
    assert resp.status_code == 400
    assert fragment in resp.json()["detail"]
    assert await scheduler.list_tasks(db) == []


async def test_tasks_api_survives_an_undetectable_machine_timezone(client, db, monkeypatch):
    def _boom() -> str:
        raise scheduler.InvalidTimezoneError("无法确定本机时区")

    monkeypatch.setattr(scheduler, "local_timezone_name", _boom)

    resp = await client.post("/api/tasks", json={"name": "a", "cron": DAILY, "prompt": "p"})
    assert resp.status_code == 400
    assert "时区" in resp.json()["detail"]

    await seed_task(db, name="旧任务")
    rows = (await client.get("/api/tasks")).json()
    assert rows[0]["machine_timezone"] is None
    assert rows[0]["timezone_changed"] is False


async def test_tasks_api_lists_stored_timezone_and_reports_machine_change(client, db, monkeypatch):
    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: SHANGHAI)
    created = (await client.post(
        "/api/tasks", json={"name": "每日复习", "cron": DAILY, "prompt": "复习"}
    )).json()

    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: "Asia/Tokyo")
    rows = (await client.get("/api/tasks")).json()

    assert len(rows) == 1
    assert rows[0]["id"] == created["id"]
    assert rows[0]["timezone"] == SHANGHAI  # 机器时区变了也不迁移既有任务
    assert rows[0]["machine_timezone"] == "Asia/Tokyo"
    assert rows[0]["timezone_changed"] is True


async def test_disable_enable_and_delete_task_api(client, db, monkeypatch):
    monkeypatch.setattr(scheduler, "local_timezone_name", lambda: SHANGHAI)
    task_id = (await client.post(
        "/api/tasks", json={"name": "每日复习", "cron": DAILY, "prompt": "复习"}
    )).json()["id"]

    assert (await client.post(f"/api/tasks/{task_id}/disable")).status_code == 200
    assert (await task_row(db, task_id))["enabled"] == 0
    assert (await client.get("/api/tasks")).json()[0]["enabled"] is False

    assert (await client.post(f"/api/tasks/{task_id}/enable")).status_code == 200
    assert (await task_row(db, task_id))["enabled"] == 1

    assert (await client.post("/api/tasks/999/enable")).status_code == 404
    assert (await client.post("/api/tasks/999/disable")).status_code == 404
    assert (await client.delete("/api/tasks/999")).status_code == 404


async def test_delete_task_keeps_its_sessions_and_notifications(client, db, monkeypatch):
    fake_agent(monkeypatch, [done_event()])
    task = await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)
    await scheduler.fire_task(task, "2026-05-01T01:00:00+00:00", db_path=db)

    resp = await client.delete(f"/api/tasks/{task.id}")

    assert resp.status_code == 200 and resp.json() == {"deleted": task.id}
    assert await scheduler.list_tasks(db) == []
    assert len(await sessions_of(db)) == 1
    assert len(await notifications_of(db)) == 1


async def test_notifications_api_returns_recent_first(client, db):
    for i in range(3):
        await scheduler.add_notification(1, "s", "success", f"任务 {i}", f"结果 {i}", db_path=db)

    rows = (await client.get("/api/notifications")).json()
    assert [r["title"] for r in rows] == ["任务 2", "任务 1", "任务 0"]
    assert set(rows[0]) == {"id", "task_id", "session_id", "kind", "title", "body", "created_at"}

    assert len((await client.get("/api/notifications", params={"limit": 1})).json()) == 1
    assert (await client.get("/api/notifications", params={"limit": 0})).status_code == 422
    assert (
        await client.get("/api/notifications", params={"limit": scheduler.MAX_NOTIFICATION_LIMIT + 1})
    ).status_code == 422


async def test_export_covers_tasks_and_notifications(client, db):
    await scheduler.create_task("每日复习", DAILY, "复习", db_path=db)
    await scheduler.add_notification(1, "s1", "success", "任务", "结果", db_path=db)

    data = (await client.get("/api/export")).json()
    assert [t["name"] for t in data["scheduled_tasks"]] == ["每日复习"]
    assert [n["title"] for n in data["notifications"]] == ["任务"]


# ---------- 生命周期 ----------


async def test_lifespan_starts_and_stops_the_scheduler(db):
    async with app.router.lifespan_context(app):
        assert scheduler.is_running() is True
        assert scheduler._accepting is True
    assert scheduler.is_running() is False
    assert scheduler._accepting is False


async def test_start_scheduler_is_single_worker(db):
    first = scheduler.start_scheduler()
    try:
        second = scheduler.start_scheduler()
        assert first is second
        assert scheduler.is_running() is True
    finally:
        await scheduler.stop_scheduler(wait=2.0)
    assert scheduler.is_running() is False


async def test_scheduler_loop_keeps_scanning_after_errors(db, monkeypatch):
    calls: list[datetime] = []

    async def _tick(now, started_at, grace_window, db_path=None):
        calls.append(now)
        if len(calls) == 1:
            raise RuntimeError("第一轮扫描炸了")
        return []

    monkeypatch.setattr(scheduler, "run_scheduler_tick", _tick)
    loop = asyncio.create_task(scheduler.scheduler_loop(interval=0.01))
    try:
        for _ in range(50):
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.01)
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)

    assert len(calls) >= 3


async def test_scheduler_loop_fires_a_task_end_to_end(db, monkeypatch):
    """不手工调 tick：起真循环，用秒级 cron 等它自己触发并写通知。"""
    fake_agent(monkeypatch, [done_event("循环触发的复习")])
    task = await scheduler.create_task("每秒任务", "* * * * * *", "干活", db_path=db)

    scheduler.start_scheduler(interval=0.05)
    try:
        for _ in range(200):
            if await notifications_of(db):
                break
            await asyncio.sleep(0.05)
    finally:
        await scheduler.stop_scheduler(wait=2.0)

    rows = await notifications_of(db)
    assert rows, "调度循环没有触发任务"
    assert rows[0]["kind"] == "success"
    assert "循环触发的复习" in rows[0]["body"]
    assert [s["mode"] for s in await sessions_of(db)] == ["work"]
    assert (await task_row(db, task.id))["last_fired_at"] is not None
    assert scheduler.is_running() is False


async def test_stop_scheduler_cancels_inflight_fire_and_notifies(db, monkeypatch):
    gate = asyncio.Event()
    fake_agent(monkeypatch, [done_event()], gate=gate)
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)
    await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db)

    await scheduler.stop_scheduler(wait=0.05)

    assert scheduler.active_fire_count() == 0
    rows = await notifications_of(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "error"
    assert "取消" in rows[0]["body"]


async def test_stop_scheduler_waits_for_a_quick_fire_without_error_notification(db, monkeypatch):
    gate = asyncio.Event()
    fake_agent(monkeypatch, [done_event()], gate=gate)
    await scheduler.create_task("每分钟", EVERY_MINUTE, "干活", db_path=db)
    now = datetime(2026, 5, 1, 10, 0, 20, tzinfo=UTC)
    await scheduler.run_scheduler_tick(now, now, 30.0, db_path=db)
    gate.set()

    await scheduler.stop_scheduler(wait=2.0)

    rows = await notifications_of(db)
    assert [r["kind"] for r in rows] == ["success"]
