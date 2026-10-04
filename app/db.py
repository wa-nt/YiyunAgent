import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
import sqlite_vec

from app.config import settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

CHUNK_VECTORS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{embed_dim}]
);
"""


def render_schema(embed_dim: int) -> str:
    base = SCHEMA_PATH.read_text(encoding="utf-8")
    return base + CHUNK_VECTORS_SQL.format(embed_dim=embed_dim)


async def _load_vec_extension(conn: aiosqlite.Connection) -> None:
    await conn.enable_load_extension(True)
    try:
        await conn.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
    finally:
        await conn.enable_load_extension(False)


async def connect(db_path: str | Path | None = None) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(str(db_path or settings.db_path))
    conn.row_factory = aiosqlite.Row
    await _load_vec_extension(conn)
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


async def init_db(db_path: str | Path | None = None, embed_dim: int | None = None) -> None:
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await connect(path)
    try:
        await conn.executescript(render_schema(embed_dim or settings.embed_dim))
        # 存量库补列：CREATE TABLE IF NOT EXISTS 不会给已存在的表加新列
        session_cols = {r["name"] for r in await conn.execute_fetchall("PRAGMA table_info(sessions)")}
        if "title" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        if "provider" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN provider TEXT")
        if "model" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        if "active_leaf" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN active_leaf INTEGER")
            # 回填：旧库都是线性消息，叶子 = 每个会话的最后一条
            await conn.execute(
                "UPDATE sessions SET active_leaf = "
                "(SELECT MAX(id) FROM messages WHERE session_id = sessions.id)"
            )
        # 模式框架（T1）：mode 是会话创建时固定的运行时选择，其余三列标记「这条会话由
        # 定时任务创建」及其归属。新库同样走这段迁移（schema.sql 里不写这四列），
        # 保证新老安装只有一条建列路径。
        # 延迟 import：runtime 在模块层 import 本模块，这里再模块层 import 回去会成环；
        # init_db 一定在应用/测试启动后调用，那时 runtime 早已可用。
        from app.agent.runtime import DEFAULT_MODE, SUPPORTED_MODES

        if "mode" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN mode TEXT")
        null_modes = (
            await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM sessions WHERE mode IS NULL OR mode = ''"
            )
        )[0]["n"]
        if null_modes:
            logger.warning("归一化 %d 个缺失 mode 的存量为 %s", null_modes, DEFAULT_MODE)
            await conn.execute(
                "UPDATE sessions SET mode = ? WHERE mode IS NULL OR mode = ''",
                (DEFAULT_MODE,),
            )
        placeholders = ", ".join("?" for _ in SUPPORTED_MODES)
        dirty = await conn.execute_fetchall(
            f"SELECT id, mode FROM sessions WHERE mode NOT IN ({placeholders})",
            SUPPORTED_MODES,
        )
        if dirty:
            logger.warning(
                "归一化 %d 个未知 mode 的会话为 %s：%s",
                len(dirty),
                DEFAULT_MODE,
                "、".join(f"{r['id']}={r['mode']}" for r in dirty),
            )
            await conn.execute(
                f"UPDATE sessions SET mode = ? WHERE mode NOT IN ({placeholders})",
                (DEFAULT_MODE, *SUPPORTED_MODES),
            )
        if "source" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN source TEXT")
        if "scheduled_task_id" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN scheduled_task_id TEXT")
        if "scheduled_occurrence_at" not in session_cols:
            await conn.execute("ALTER TABLE sessions ADD COLUMN scheduled_occurrence_at TEXT")
        # NULL/空 source = 非定时任务建立的会话（旧数据与直接 INSERT 的行都算手动）
        await conn.execute(
            "UPDATE sessions SET source = 'manual' WHERE source IS NULL OR source = ''"
        )
        msg_cols = {r["name"] for r in await conn.execute_fetchall("PRAGMA table_info(messages)")}
        if "parent_id" not in msg_cols:
            await conn.execute("ALTER TABLE messages ADD COLUMN parent_id INTEGER")
            # 回填成线性链：每条消息的 parent 是同会话里它前面的最后一条
            await conn.execute(
                "UPDATE messages SET parent_id = (SELECT MAX(m2.id) FROM messages m2 "
                "WHERE m2.session_id = messages.session_id AND m2.id < messages.id)"
            )
        await conn.commit()
    finally:
        await conn.close()


@asynccontextmanager
async def get_db(db_path: str | Path | None = None) -> AsyncIterator[aiosqlite.Connection]:
    conn = await connect(db_path)
    try:
        yield conn
    finally:
        await conn.close()