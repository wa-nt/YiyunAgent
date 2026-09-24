import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings
from app.db import get_db
from app.retrieval.types import RetrievedChunk

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """英文按非字母数字切词，中文按 bigram 切分（不引分词依赖）。"""
    tokens = _WORD.findall(text.lower())
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """chunks 表全量语料的 BM25 索引，首次检索时懒加载。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        self._loaded = False
        self._ids: list[int] = []
        self._terms: list[frozenset[str]] = []
        self._meta: dict[int, tuple[int, str, str | None]] = {}
        self._bm25: BM25Okapi | None = None

    async def load(self) -> None:
        if self._loaded:
            return
        async with get_db(self._db_path) as conn:
            rows = await conn.execute_fetchall(
                "SELECT c.id, c.doc_id, c.content, d.title FROM chunks c "
                "LEFT JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
            )
        corpus: list[list[str]] = []
        for row in rows:
            content = row["content"] or ""
            self._ids.append(row["id"])
            self._meta[row["id"]] = (row["doc_id"], content, row["title"])
            corpus.append(tokenize(content))
        self._terms = [frozenset(tokens) for tokens in corpus]
        # 空语料或全部切不出 token（avgdl = 0）会让 BM25Okapi 除零
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._loaded = True

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        terms = set(query_tokens)
        if not terms:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # 命中判定用 token 交集，不能用 score > 0：token 出现在半数以上文档时
        # BM25 的 idf 会退化为 0 甚至负数（单文档语料必然如此）
        ranked = sorted(
            (
                (score, chunk_id)
                for chunk_id, score, doc_terms in zip(self._ids, scores, self._terms)
                if terms & doc_terms
            ),
            key=lambda item: (-item[0], item[1]),
        )
        results = []
        for score, chunk_id in ranked[:k]:
            doc_id, content, title = self._meta[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=content,
                    title=title,
                    score=float(score),
                )
            )
        return results


_indexes: dict[str, BM25Index] = {}


def _cache_key(db_path: str | Path | None) -> str:
    return str(Path(db_path or settings.db_path).resolve())


async def _get_index(db_path: str | Path | None) -> BM25Index:
    key = _cache_key(db_path)
    index = _indexes.get(key)
    if index is None:
        index = BM25Index(db_path)
        _indexes[key] = index
    await index.load()
    return index


def invalidate(db_path: str | Path | None = None) -> None:
    """丢弃 BM25 索引缓存；不传 db_path 时丢弃全部。chunks 表变更后必须调用。"""
    if db_path is None:
        _indexes.clear()
    else:
        _indexes.pop(_cache_key(db_path), None)


async def bm25_search(
    query: str, k: int = 20, db_path: str | Path | None = None
) -> list[RetrievedChunk]:
    index = await _get_index(db_path)
    return index.search(query, k)