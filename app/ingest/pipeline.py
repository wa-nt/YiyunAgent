import json
from datetime import datetime, timezone
from pathlib import Path

from app.db import get_db, init_db
from app.helpers import embedding_to_blob
from app.ingest.chunker import chunk_text
from app.ingest.loaders import load
from app.llm.embed import embed_texts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def ingest(source: str | Path, db_path: str | Path | None = None) -> int:
    title, text = load(source)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    await init_db(db_path)
    async with get_db(db_path) as conn:
        try:
            cursor = await conn.execute(
                "INSERT INTO documents (source, title, ingested_at) VALUES (?, ?, ?)",
                (str(source), title, _now()),
            )
            doc_id = cursor.lastrowid
            vectors = await embed_texts(chunks)
            for idx, (content, vector) in enumerate(zip(chunks, vectors, strict=True)):
                cursor = await conn.execute(
                    "INSERT INTO chunks (doc_id, idx, content, token_count, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, idx, content, len(content), embedding_to_blob(vector)),
                )
                await conn.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, json.dumps(vector)),
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return len(chunks)


async def list_documents(db_path: str | Path | None = None) -> list[dict]:
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(
            "SELECT d.id, d.source, d.title, d.ingested_at, COUNT(c.id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY d.id ORDER BY d.id"
        )
    return [dict(row) for row in rows]


async def delete_document(doc_id: int, db_path: str | Path | None = None) -> None:
    async with get_db(db_path) as conn:
        chunk_ids = [
            row["id"]
            for row in await conn.execute_fetchall(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        ]
        await conn.executemany(
            "DELETE FROM chunk_vectors WHERE chunk_id = ?", [(cid,) for cid in chunk_ids]
        )
        await conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        await conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await conn.commit()