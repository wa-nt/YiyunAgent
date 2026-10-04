"""T3 知识漏洞（间隔重复卡片）的用例：领域函数、work 模式工具、HTTP 接口。

隔离方式与 tests/test_api.py 一致：db 夹具把 settings.db_path 指到 tmp 并 init_db
（ASGITransport 不跑 lifespan，建表的责任在夹具），完全不碰 data/app.db。

时间断言一律给容差：_now() 的粒度是秒，用例算出的「期望时刻」与库里的时刻不可能
逐字相等。到期的用例不靠 monkeypatch 时钟，而是直接把 next_review_at 改到过去——
那条 SQL 比较（`<= now`）本来就是被测对象。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ToolCall
from app.main import app
from app.study.gaps import (
    GapNotFoundError,
    delete_gap,
    due_gaps,
    list_gaps,
    record_gap,
    review_gap,
)

DIM = 8


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_near(value: str, expected: datetime, tol: timedelta = timedelta(minutes=1)) -> None:
    delta = abs(datetime.fromisoformat(value) - expected)
    assert delta <= tol, f"{value} 与期望 {expected.isoformat()} 相差 {delta}"


async def _set_next_review(db, gap_id: int, when: datetime) -> None:
    """直接把某张卡片的到期时间改到指定时刻（到期的用例用它造「已过期」状态）。"""
    async with get_db(db) as conn:
        await conn.execute(
            "UPDATE knowledge_gaps SET next_review_at = ? WHERE id = ?",
            (when.isoformat(timespec="seconds"), gap_id),
        )
        await conn.commit()


async def _record(db, topic="向量检索", detail="分不清 HNSW 与 IVF") -> dict:
    return await record_gap(topic, detail, db)


# ---------- 领域函数：记录与到期 ----------


async def test_new_gap_is_not_due_before_a_day_passes(db):
    gap = await _record(db)

    assert gap["interval_days"] == 1
    _assert_near(gap["next_review_at"], _now() + timedelta(days=1))
    assert await due_gaps(db) == []
    assert [row["id"] for row in await list_gaps(db)] == [gap["id"]]


async def test_overdue_gap_shows_up_in_due_list(db):
    gap = await _record(db)
    await _set_next_review(db, gap["id"], _now() - timedelta(minutes=1))

    due = await due_gaps(db)

    assert [row["id"] for row in due] == [gap["id"]]
    assert due[0]["topic"] == "向量检索"
    assert due[0]["detail"] == "分不清 HNSW 与 IVF"


async def test_same_topic_can_be_recorded_twice(db):
    """本期不做语义去重：同一个主题记两次就是两张卡片（brief 明确允许）。"""
    first = await _record(db)
    second = await _record(db, detail="又忘了")

    assert first["id"] != second["id"]
    assert len(await list_gaps(db)) == 2


async def test_review_pass_doubles_the_interval(db):
    gap = await _record(db)

    updated = await review_gap(gap["id"], True, db)

    assert updated["interval_days"] == 2
    _assert_near(updated["next_review_at"], _now() + timedelta(days=2))
    assert updated["updated_at"] >= updated["created_at"]

    again = await review_gap(gap["id"], True, db)
    assert again["interval_days"] == 4


async def test_review_failure_resets_the_interval_to_one_day(db):
    gap = await _record(db)
    for _ in range(3):  # 1 → 2 → 4 → 8
        await review_gap(gap["id"], True, db)

    updated = await review_gap(gap["id"], False, db)

    assert updated["interval_days"] == 1
    _assert_near(updated["next_review_at"], _now() + timedelta(days=1))


async def test_interval_is_capped_at_180_days(db):
    gap = await _record(db)

    for _ in range(20):  # 1→2→4→…→128，再往后一律钉在 180
        updated = await review_gap(gap["id"], True, db)

    assert updated["interval_days"] == 180
    _assert_near(updated["next_review_at"], _now() + timedelta(days=180))


async def test_review_unknown_gap_raises(db):
    with pytest.raises(GapNotFoundError):
        await review_gap(999, True, db)


async def test_review_does_not_touch_other_gaps(db):
    reviewed = await _record(db, topic="A", detail="a")
    untouched = await _record(db, topic="B", detail="b")

    await review_gap(reviewed["id"], True, db)

    rows = {row["id"]: row for row in await list_gaps(db)}
    assert rows[untouched["id"]]["interval_days"] == 1


async def test_delete_gap_reports_whether_a_row_was_removed(db):
    gap = await _record(db)

    assert await delete_gap(gap["id"], db) is True
    assert await list_gaps(db) == []
    assert await delete_gap(gap["id"], db) is False


async def test_utc_timestamps_have_an_explicit_offset(db):
    """时间统一 UTC：带偏移量才能保证字符串比较与真实时刻同一顺序。"""
    gap = await _record(db)

    for field in ("next_review_at", "created_at", "updated_at"):
        assert datetime.fromisoformat(gap[field]).utcoffset() == timedelta(0)


# ---------- work 模式的工具 ----------


async def test_work_mode_record_tool_writes_a_gap(db):
    result, label = await runtime.execute_tool(
        ToolCall(
            id="c1",
            name="record_knowledge_gap",
            arguments={"topic": "向量检索", "detail": "分不清 HNSW 与 IVF"},
        ),
        str(db),
        mode="work",
    )

    rows = await list_gaps(db)
    assert len(rows) == 1
    assert rows[0]["topic"] == "向量检索"
    assert f"#{rows[0]['id']}" in result
    assert "向量检索" in label
    assert "1 天后" in result


async def test_work_mode_review_tool_reschedules_the_gap(db):
    gap = await _record(db)

    result, label = await runtime.execute_tool(
        ToolCall(
            id="c1",
            name="review_knowledge_gap",
            arguments={"gap_id": gap["id"], "passed": True},
        ),
        str(db),
        mode="work",
    )

    assert (await list_gaps(db))[0]["interval_days"] == 2
    assert "间隔更新为 2 天" in result
    assert "间隔 2 天" in label


async def test_record_tool_rejects_missing_or_blank_args(db):
    cases = [
        ({"topic": "只给了主题"}, "detail"),
        ({"topic": "   ", "detail": "有内容"}, "topic"),
        ({"detail": "只给了细节"}, "topic"),
        ({}, "topic"),
    ]
    for arguments, missing in cases:
        result, label = await runtime.execute_tool(
            ToolCall(id="c1", name="record_knowledge_gap", arguments=arguments),
            str(db),
            mode="work",
        )
        assert missing in result and missing in label, arguments

    assert await list_gaps(db) == []


async def test_record_tool_rejects_non_string_args(db):
    result, _ = await runtime.execute_tool(
        ToolCall(
            id="c1",
            name="record_knowledge_gap",
            arguments={"topic": 42, "detail": ["x"]},
        ),
        str(db),
        mode="work",
    )

    assert "topic" in result and "detail" in result
    assert await list_gaps(db) == []


async def test_review_tool_reports_unknown_gap(db):
    result, label = await runtime.execute_tool(
        ToolCall(
            id="c1",
            name="review_knowledge_gap",
            arguments={"gap_id": 999, "passed": True},
        ),
        str(db),
        mode="work",
    )

    assert "999" in result and "不存在" in result
    assert "999" in label


async def test_review_tool_rejects_bad_args(db):
    cases = [
        {"gap_id": "abc", "passed": True},  # gap_id 不是整数
        {"passed": True},  # 缺 gap_id
        {"gap_id": 1},  # 缺 passed
        {"gap_id": 1, "passed": "也许吧"},  # passed 不是真假值
    ]
    for arguments in cases:
        result, label = await runtime.execute_tool(
            ToolCall(id="c1", name="review_knowledge_gap", arguments=arguments),
            str(db),
            mode="work",
        )
        assert "参数" in result or "gap_id" in result, arguments
        assert "参数" in label or "gap_id" in label, arguments


async def test_tools_are_registered_only_for_work_mode():
    names = [tool.name for tool in runtime.tools_for_mode("work")]
    assert "record_knowledge_gap" in names
    assert "review_knowledge_gap" in names
    assert [tool.name for tool in runtime.tools_for_mode("chat")] == ["search_knowledge"]


# ---------- HTTP 接口 ----------


async def test_api_gaps_defaults_to_due_only(client, db):
    overdue = await _record(db, topic="到期卡片", detail="该复习了")
    await _set_next_review(db, overdue["id"], _now() - timedelta(hours=1))
    await _record(db, topic="未来卡片", detail="还早")

    resp = await client.get("/api/gaps")
    explicit = await client.get("/api/gaps", params={"all": "false"})

    assert resp.status_code == 200
    assert [row["id"] for row in resp.json()] == [overdue["id"]]
    assert resp.json() == explicit.json()


async def test_api_gaps_all_lists_every_card_soonest_first(client, db):
    overdue = await _record(db, topic="过期", detail="a")
    await _set_next_review(db, overdue["id"], _now() - timedelta(hours=1))
    soon = await _record(db, topic="一天后", detail="b")
    later = await _record(db, topic="很久以后", detail="c")
    await _set_next_review(db, later["id"], _now() + timedelta(days=30))

    rows = (await client.get("/api/gaps", params={"all": "true"})).json()

    assert [row["id"] for row in rows] == [overdue["id"], soon["id"], later["id"]]
    assert set(rows[0]) == {
        "id",
        "topic",
        "detail",
        "interval_days",
        "next_review_at",
        "created_at",
        "updated_at",
    }


async def test_api_gaps_rejects_invalid_all_param(client, db):
    assert (await client.get("/api/gaps", params={"all": "也许"})).status_code == 422


async def test_api_delete_gap(client, db):
    gap = await _record(db)

    resp = await client.delete(f"/api/gaps/{gap['id']}")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": gap["id"]}
    assert await list_gaps(db) == []


async def test_api_delete_missing_gap_is_404(client, db):
    assert (await client.delete("/api/gaps/999")).status_code == 404


async def test_gaps_are_part_of_the_user_data_export(client, db):
    """漏洞卡片是用户的学习进度，导出（备份）必须带上。"""
    gap = await _record(db)

    exported = (await client.get("/api/export")).json()

    assert [row["id"] for row in exported["knowledge_gaps"]] == [gap["id"]]
