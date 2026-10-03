from __future__ import annotations

import logging

from app.config import settings
from app.db import get_db
from app.retrieval.bm25_search import tokenize

logger = logging.getLogger(__name__)

RECALL_HEADER = "以下是关于用户的一些长期记忆，供参考："
MAX_RECALL_TOP_K = 50  # 召回上限：配置写错（负数/极大值）时不至于把整张表灌进提示词
CANDIDATE_FACTOR = 3  # 候选池：先按 confidence 取 top_k*3，再用 query 重叠度重排


async def recall_memories(user_message: str, db_path: str | None = None) -> str | None:
    """召回长期记忆，返回可直接作为 system 消息的文本；无可用记忆时返回 None。

    先取 active 状态按 confidence 倒序的 top_k*CANDIDATE_FACTOR 条候选（top_k =
    settings.memory_recall_top_k，钳制在 0..MAX_RECALL_TOP_K：负数会让 SQLite 的
    LIMIT -1 变成全量注入，N=0 表示本轮不注入记忆），再用 user_message 与候选
    content 的 token 重叠度（CJK bigram，见 bm25_search.tokenize）重排后取 top_k。
    memories 表没有 embedding 字段，所以这里是词面重叠而非语义召回。
    """
    if not settings.memory_enabled:
        return None
    top_k = min(max(settings.memory_recall_top_k, 0), MAX_RECALL_TOP_K)
    if top_k == 0:
        return None
    try:
        async with get_db(db_path) as conn:
            rows = await conn.execute_fetchall(
                "SELECT kind, content, confidence FROM memories WHERE status = 'active' "
                "ORDER BY confidence DESC, id DESC LIMIT ?",
                (top_k * CANDIDATE_FACTOR,),
            )
    except Exception as exc:  # 记忆是增强项，召回失败退化为无记忆，不能拖垮回答
        logger.warning("记忆召回失败，本轮按无记忆处理：%s: %s", type(exc).__name__, exc)
        return None
    if not rows:
        return None
    lines = "\n".join(
        f"- [{row['kind']}] {row['content']}" for row in _rerank(user_message, rows)[:top_k]
    )
    return f"{RECALL_HEADER}\n{lines}"


def _rerank(user_message: str, rows) -> list:
    """按 query 词面重叠加成重排：`confidence + token Jaccard`。

    confidence 仍是主序、重叠只做同尺度（0..1）的小幅加成，所以与 query 无关的
    高置信记忆不会被相关但低置信的记忆无理由挤掉；两边都无重叠时加成恒为 0，
    结果与纯 confidence 排序一致（sorted 是稳定排序，保留 SQL 的次序）。
    """
    query_terms = set(tokenize(user_message))
    if not query_terms:
        return rows
    scored = []
    for row in rows:
        terms = set(tokenize(row["content"]))
        overlap = len(query_terms & terms) / len(query_terms | terms) if terms else 0.0
        scored.append((row["confidence"] + overlap, row))
    return [row for _, row in sorted(scored, key=lambda item: -item[0])]
