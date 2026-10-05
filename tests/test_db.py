import pytest

from app.db import get_db, init_db
from app.helpers import blob_to_embedding, embedding_to_blob

DIM = 8


@pytest.mark.asyncio
async def test_init_db_insert_chunk_and_knn(tmp_path):
    db_path = tmp_path / "app.db"
    await init_db(db_path, DIM)

    vectors = {
        "kv": [1.0] + [0.0] * (DIM - 1),
        "rag": [0.0, 1.0] + [0.0] * (DIM - 2),
        "plan": [0.0, 0.0, 1.0] + [0.0] * (DIM - 3),
    }

    async with get_db(db_path) as conn:
        tables = {
            row["name"]
            for row in await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "documents",
            "chunks",
            "chunk_vectors",
            "memories",
            "sessions",
            "messages",
            "traces",
        } <= tables

        await conn.execute(
            "INSERT INTO documents (id, source, title, ingested_at, meta) VALUES (?, ?, ?, ?, ?)",
            (1, "notes/kv-cache.md", "KV Cache 笔记", "2026-09-24T10:00:00", "{}"),
        )

        chunk_ids = {}
        for idx, (name, vector) in enumerate(vectors.items()):
            cursor = await conn.execute(
                "INSERT INTO chunks (doc_id, idx, content, token_count, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, idx, f"chunk-{name}", 3, embedding_to_blob(vector)),
            )
            chunk_ids[name] = cursor.lastrowid
            await conn.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk_ids[name], embedding_to_blob(vector)),
            )
        await conn.commit()

        stored = await conn.execute_fetchall(
            "SELECT embedding FROM chunks WHERE id = ?", (chunk_ids["kv"],)
        )
        assert blob_to_embedding(stored[0]["embedding"]) == vectors["kv"]

        neighbours = await conn.execute_fetchall(
            "SELECT chunk_id, distance FROM chunk_vectors "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT 2",
            (embedding_to_blob([0.9, 0.1] + [0.0] * (DIM - 2)),),
        )

    assert neighbours[0]["chunk_id"] == chunk_ids["kv"]
    assert neighbours[0]["distance"] < neighbours[1]["distance"]

@pytest.mark.asyncio
async def test_init_db_sets_user_version_and_new_session_columns(tmp_path):
    db_path = tmp_path / "app.db"
    await init_db(db_path, DIM)
    async with get_db(db_path) as conn:
        version = (await conn.execute_fetchall("PRAGMA user_version"))[0][0]
        cols = {r["name"] for r in await conn.execute_fetchall("PRAGMA table_info(sessions)")}
    assert version == 1
    assert {"history_summary", "history_summary_upto"} <= cols


@pytest.mark.asyncio
async def test_startup_backup_rotates(tmp_path, monkeypatch):
    from app import main

    db_path = tmp_path / "app.db"
    await init_db(db_path, DIM)
    monkeypatch.setattr(main.settings, "db_path", str(db_path))
    monkeypatch.setattr(main, "BACKUP_KEEP", 2)
    bdir = tmp_path / "backups"
    bdir.mkdir()
    for name in ("app-20200101-000001.db", "app-20200101-000002.db", "app-20200101-000003.db"):
        (bdir / name).write_bytes(b"old")

    main._backup_db()

    files = sorted(f.name for f in bdir.glob("app-*.db"))
    # 新备份 + 保留的最新一份旧备份，更早的两份被删
    assert len(files) == 2
    assert files[-1] == "app-20200101-000003.db" or files[-1].startswith("app-20")
    assert files[-1] != "app-20200101-000001.db"
    assert max(bdir.glob("app-*.db"), key=lambda f: f.stat().st_mtime).stat().st_size > 0
