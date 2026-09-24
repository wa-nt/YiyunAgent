import re
from pathlib import Path

from app.db import get_db
from app.retrieval.types import RetrievedChunk

# 保守启发式：英文词（2+ 字符）与中文片段当作候选实体。2-8 字的中文串
# 整体保留，再叠加 2 字子串参与 LIKE 匹配 —— 超过 8 字的长串（通常是句子
# 片段）不再整段弃用，靠 bigram 子串召回包含其中专名的正文。
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def extract_entities(query: str) -> list[str]:
    candidates = _WORD.findall(query)
    for run in _CJK.findall(query):
        if len(run) < 2:
            continue
        if len(run) <= 8:
            candidates.append(run)
        candidates.extend(run[i : i + 2] for i in range(len(run) - 1))
    seen: dict[str, None] = {}
    for token in candidates:
        seen.setdefault(token, None)
    return list(seen)


async def entity_search(
    query: str, k: int = 20, db_path: str | Path | None = None
) -> list[RetrievedChunk]:
    """逐 token LIKE 匹配正文，按命中 token 数排序。"""
    tokens = extract_entities(query)
    if not tokens:
        return []

    hits: dict[int, int] = {}
    meta: dict[int, tuple[int, str, str | None]] = {}
    async with get_db(db_path) as conn:
        for token in tokens:
            # token 由白名单正则产生，不含 LIKE 通配符，无需转义
            rows = await conn.execute_fetchall(
                "SELECT c.id, c.doc_id, c.content, d.title FROM chunks c "
                "LEFT JOIN documents d ON d.id = c.doc_id "
                "WHERE c.content LIKE ?",
                (f"%{token}%",),
            )
            for row in rows:
                hits[row["id"]] = hits.get(row["id"], 0) + 1
                meta[row["id"]] = (row["doc_id"], row["content"] or "", row["title"])

    ranked = sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta[chunk_id][0],
            content=meta[chunk_id][1],
            title=meta[chunk_id][2],
            score=float(count),
        )
        for chunk_id, count in ranked
    ]