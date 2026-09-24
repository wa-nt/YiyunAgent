import json
import math
from collections import Counter

import pytest

from app.config import settings
from app.db import get_db, init_db
from app.ingest import pipeline
from app.retrieval import hybrid as hybrid_module
from app.retrieval.bm25_search import bm25_search, invalidate, tokenize
from app.retrieval.entity_search import entity_search, extract_entities
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_search import vector_search

DIM = 8

# idx → chunk_id 均为 id 顺序（1..6）；两个文档用于验证 doc/title JOIN
CHUNKS = [
    (1, "KV Cache 缓存机制显著降低大模型推理延迟"),
    (1, "BM25 是基于词频的稀疏检索算法"),
    (1, "混合检索融合向量与关键词两路召回结果"),
    (2, "asyncio 提供异步事件循环与协程调度"),
    (2, "向量检索依赖 embedding 模型缓存文本向量"),
    (2, "Retrieval augmented generation combines retrieval with generation"),
]
TITLES = {1: "KV 笔记", 2: "检索笔记"}


def _ray(degrees: float) -> list[float]:
    """与第 0 维夹角为 degrees 的单位向量，到 QUERY_VEC 的距离随角度单调递增。"""
    theta = math.radians(degrees)
    return [math.cos(theta), math.sin(theta)] + [0.0] * (DIM - 2)


VECTORS = [_ray(20 * i) for i in range(len(CHUNKS))]
QUERY_VEC = [1.0] + [0.0] * (DIM - 1)


async def _insert_chunk(
    conn, doc_id: int, idx: int, content: str, vector: list[float]
) -> int:
    cursor = await conn.execute(
        "INSERT INTO chunks (doc_id, idx, content, token_count) VALUES (?, ?, ?, ?)",
        (doc_id, idx, content, len(content)),
    )
    chunk_id = cursor.lastrowid
    await conn.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, json.dumps(vector)),
    )
    return chunk_id


@pytest.fixture
async def db(tmp_path):
    invalidate()
    path = tmp_path / "app.db"
    await init_db(path, DIM)
    async with get_db(path) as conn:
        for doc_id, title in TITLES.items():
            await conn.execute(
                "INSERT INTO documents (id, source, title, ingested_at) VALUES (?, ?, ?, ?)",
                (doc_id, f"notes/note{doc_id}.md", title, "2026-09-24T10:00:00"),
            )
        for i, (doc_id, content) in enumerate(CHUNKS):
            await _insert_chunk(conn, doc_id, i, content, VECTORS[i])
        await conn.commit()
    yield path
    invalidate()


async def test_vector_search_returns_nearest_first(db):
    hits = await vector_search(QUERY_VEC, k=4, db_path=db)

    assert [h.chunk_id for h in hits] == [1, 2, 3, 4]
    distances = [h.score for h in hits]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-5)

    assert hits[0].content == CHUNKS[0][1]
    assert hits[0].doc_id == 1 and hits[0].title == "KV 笔记"
    assert hits[3].doc_id == 2 and hits[3].title == "检索笔记"


async def test_vector_search_k_truncates(db):
    hits = await vector_search(QUERY_VEC, k=2, db_path=db)

    assert [h.chunk_id for h in hits] == [1, 2]


def test_tokenize_english_and_cjk_bigram():
    assert tokenize("KV Cache 缓存") == ["kv", "cache", "缓存"]
    assert tokenize("推理加速") == ["推理", "理加", "加速"]
    assert tokenize("") == []


async def test_bm25_hits_chinese_keyword(db):
    hits = await bm25_search("缓存机制", k=5, db_path=db)
    assert hits[0].chunk_id == 1
    assert "缓存" in hits[0].content
    assert hits[0].score > 0

    hits = await bm25_search("混合检索", k=5, db_path=db)
    assert hits[0].chunk_id == 3

    hits = await bm25_search("asyncio 事件循环", k=5, db_path=db)
    assert hits[0].chunk_id == 4

    # 「检索」在 6 个 chunk 里命中 3 个，idf 正好退化为 0；命中判定不能依赖 score > 0
    hits = await bm25_search("检索", k=5, db_path=db)
    assert {h.chunk_id for h in hits} == {2, 3, 5}
    assert len(await bm25_search("检索", k=1, db_path=db)) == 1

    # 无共同 token 的查询不返回候选，而不是返回零分词
    assert await bm25_search("量子引力波", k=5, db_path=db) == []


async def test_extract_entities_keeps_names_only():
    assert extract_entities("KV Cache 与 asyncio") == ["KV", "Cache", "asyncio"]
    assert extract_entities("缓存机制") == ["缓存机制"]
    # 单字中文、超长中文串（句子片段）、纯标点都不当实体
    assert extract_entities("的 分布式训练需要多卡同步梯度") == []
    assert extract_entities("!!!") == []


async def test_entity_search_ranks_by_hit_count(db):
    hits = await entity_search("向量 缓存", k=5, db_path=db)

    assert hits[0].chunk_id == 5
    assert hits[0].score == 2.0
    assert {h.chunk_id for h in hits} == {1, 3, 5}

    assert [h.chunk_id for h in await entity_search("asyncio", db_path=db)] == [4]
    assert await entity_search("!!!", db_path=db) == []


async def test_hybrid_vector_mode_only_uses_vector_route(db, monkeypatch):
    demands: list[list[str]] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        demands.append(list(texts))
        return [QUERY_VEC for _ in texts]

    async def forbidden(*args, **kwargs):
        raise AssertionError("vector 模式不应调用其它检索路")

    monkeypatch.setattr(hybrid_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(hybrid_module, "bm25_search", forbidden)
    monkeypatch.setattr(hybrid_module, "entity_search", forbidden)

    expected = await vector_search(QUERY_VEC, 4, db)
    hits = await hybrid_search("缓存", k=4, mode="vector", db_path=db)

    assert demands == [["缓存"]]
    assert [h.chunk_id for h in hits] == [h.chunk_id for h in expected]
    assert [h.score for h in hits] == [h.score for h in expected]


async def test_hybrid_bm25_mode_skips_embedding(db, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("bm25 模式不应走向量 / 实体路")

    monkeypatch.setattr(hybrid_module, "embed_texts", forbidden)
    monkeypatch.setattr(hybrid_module, "entity_search", forbidden)

    expected = await bm25_search("缓存机制", 3, db)
    hits = await hybrid_search("缓存机制", k=3, mode="bm25", db_path=db)

    assert [h.chunk_id for h in hits] == [h.chunk_id for h in expected]


async def test_hybrid_rejects_unknown_mode(db):
    with pytest.raises(ValueError):
        await hybrid_search("缓存", mode="entity", db_path=db)


async def test_hybrid_fusion_ranks_dual_route_hits_first(db, monkeypatch):
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [QUERY_VEC for _ in texts]

    monkeypatch.setattr(hybrid_module, "embed_texts", fake_embed_texts)

    query = "检索"
    routes = [
        await vector_search(QUERY_VEC, 20, db),
        await bm25_search(query, 20, db),
        await entity_search(query, 20, db),
    ]
    counts = Counter(h.chunk_id for route in routes for h in route)
    dual = {cid for cid, n in counts.items() if n >= 2}
    single = {cid for cid, n in counts.items() if n == 1}

    hits = await hybrid_search(query, k=8, mode="hybrid", db_path=db)
    ordered = [h.chunk_id for h in hits]

    # 「检索」命中 chunk 2/3/5，它们同时出现在向量 top-20 里
    assert dual == {2, 3, 5}
    assert single == {1, 4, 6}
    assert set(ordered) == set(counts)

    scores = {h.chunk_id: h.score for h in hits}
    assert min(scores[cid] for cid in dual) > max(scores[cid] for cid in single)
    assert max(ordered.index(cid) for cid in dual) < min(
        ordered.index(cid) for cid in single
    )
    assert ordered[0] in dual


async def test_invalidate_picks_up_newly_inserted_chunk(db):
    assert [h.chunk_id for h in await bm25_search("协程调度", 5, db)] == [4]

    async with get_db(db) as conn:
        new_id = await _insert_chunk(
            conn, 2, 9, "协程调度可以配合事件循环做并发", _ray(5)
        )
        await conn.commit()

    # 索引是缓存的，未失效前搜不到新 chunk
    cached = await bm25_search("协程调度", 5, db)
    assert new_id not in [h.chunk_id for h in cached]

    invalidate()
    refreshed = await bm25_search("协程调度", 5, db)
    assert {h.chunk_id for h in refreshed} == {4, new_id}


async def test_ingest_refreshes_bm25_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [QUERY_VEC for _ in texts]

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)

    path = tmp_path / "app.db"
    await init_db(path, DIM)
    assert await bm25_search("分布式训练", 5, path) == []

    note = tmp_path / "note.md"
    note.write_text("# 训练笔记\n\n分布式训练需要多卡同步梯度", encoding="utf-8")
    await pipeline.ingest(str(note), path)

    # 单文档语料里 idf 全为负，命中判定同样不能依赖 score > 0
    hits = await bm25_search("分布式训练", 5, path)
    assert hits and "分布式训练" in hits[0].content

    await pipeline.delete_document(1, path)
    assert await bm25_search("分布式训练", 5, path) == []