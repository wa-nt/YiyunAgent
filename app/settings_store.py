"""`app_settings` 的最小读写：只放「用户在界面上编辑的多行文本」这类用户数据。

为什么不是 `.env`：`.env` 是单行 `KEY=VALUE` 的部署配置口径，多行原文（含换行、引号、
中文）写进去会被写坏；而 persona 是用户数据，不该跟供应商密钥混在一个文件里。所以它
不进 `config.EDITABLE_FIELDS`，也不进 `/api/settings` 的 `.env` 写入路径。

**无记录 / NULL 与空字符串是两种语义**：前者表示「没设置过」，调用方据此回退默认人格
文件；后者表示用户明确不要人格，按空字符串 round-trip，不回退。`get_setting` 因此用
None 表示无记录，不用空串兜底。
"""

from __future__ import annotations

from pathlib import Path

from app.db import get_db

PERSONA_KEY = "persona"


async def get_setting(key: str, db_path: str | Path | None = None) -> str | None:
    """读一条设置；没有这条记录时返回 None。表不存在等真实故障照常抛，不静默吞掉。"""
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        )
    return rows[0]["value"] if rows else None


async def set_setting(key: str, value: str, db_path: str | Path | None = None) -> None:
    """整条覆盖写入（UPSERT）。value 原样落库，不做 strip/规范化。"""
    async with get_db(db_path) as conn:
        await conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await conn.commit()
