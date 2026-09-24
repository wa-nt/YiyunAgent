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