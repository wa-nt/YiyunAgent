import json
from pathlib import Path

from app.db import get_db
from app.retrieval.types import RetrievedChunk

# chunk_vectors 是 vec0 虚拟表。KNN 的 k 必须在虚拟表这一层给出（`k = ?` 或 LIMIT）：
# FROM 里一旦带 JOIN，SQLite 就不再把它当作 KNN 查询，直接写 LIMIT 会报
# "A LIMIT or 'k = ?' constraint is required on vec0 knn queries"。
# 所以先在子查询里取 top-k，再 JOIN 回 chunks/documents 补正文和标题。
KNN_SQL = """
SELECT v.chunk_id, v.distance, c.doc_id, c.content, d.title
FROM (
    SELECT chunk_id, distance FROM chunk_vectors
    WHERE embedding MATCH ? AND k = ?
) v
JOIN chunks c ON c.id = v.chunk_id
LEFT JOIN documents d ON d.id = c.doc_id
ORDER BY v.distance
"""


async def vector_search(
    query_vec: list[float], k: int = 20, db_path: str | Path | None = None
) -> list[RetrievedChunk]:
    async with get_db(db_path) as conn:
        rows = await conn.execute_fetchall(KNN_SQL, (json.dumps(query_vec), k))
    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            content=row["content"],
            title=row["title"],
            score=float(row["distance"]),
        )
        for row in rows
    ]