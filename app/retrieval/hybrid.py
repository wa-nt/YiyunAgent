import asyncio
from dataclasses import replace
from pathlib import Path

from app.llm.embed import embed_texts
from app.retrieval.bm25_search import bm25_search
from app.retrieval.entity_search import entity_search
from app.retrieval.types import RetrievedChunk
from app.retrieval.vector_search import vector_search

CANDIDATES = 20
RRF_K = 60
MODES = ("vector", "bm25", "hybrid")


async def hybrid_search(
    query: str, k: int = 8, mode: str = "hybrid", db_path: str | Path | None = None
) -> list[RetrievedChunk]:
    """三路混合检索。mode 取 "vector" / "bm25" 时只走单路（评测基线），
    取 "hybrid" 时三路各取 top-20 后用 RRF 融合再取 top-k，score 为 RRF 分。
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {MODES}")

    if mode == "bm25":
        return await bm25_search(query, k, db_path)

    query_vec = (await embed_texts([query]))[0]
    if mode == "vector":
        return await vector_search(query_vec, k, db_path)

    routes = await asyncio.gather(
        vector_search(query_vec, CANDIDATES, db_path),
        bm25_search(query, CANDIDATES, db_path),
        entity_search(query, CANDIDATES, db_path),
    )

    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for results in routes:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            chunks.setdefault(chunk.chunk_id, chunk)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [replace(chunks[chunk_id], score=score) for chunk_id, score in ranked]