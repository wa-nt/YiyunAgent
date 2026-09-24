from __future__ import annotations

import logging

from app.config import settings
from app.db import get_db

logger = logging.getLogger(__name__)

RECALL_HEADER = "以下是关于用户的一些长期记忆，供参考："


async def recall_memories(user_message: str, db_path: str | None = None) -> str | None:
    """召回长期记忆，返回可直接作为 system 消息的文本；无可用记忆时返回 None。

    取 active 状态按 confidence 倒序的 top-N（N = settings.memory_recall_top_k）。
    memories 表没有 embedding 字段，所以 user_message 暂不参与排序，语义召回
    留给 T10（按 query 加权的三路召回）；参数先按接口契约保留。
    """
    if not settings.memory_enabled:
        return None
    try:
        async with get_db(db_path) as conn:
            rows = await conn.execute_fetchall(
                "SELECT kind, content FROM memories WHERE status = 'active' "
                "ORDER BY confidence DESC, id DESC LIMIT ?",
                (settings.memory_recall_top_k,),
            )
    except Exception as exc:  # 记忆是增强项，召回失败退化为无记忆，不能拖垮回答
        logger.warning("记忆召回失败，本轮按无记忆处理：%s: %s", type(exc).__name__, exc)
        return None
    if not rows:
        return None
    lines = "\n".join(f"- [{row['kind']}] {row['content']}" for row in rows)
    return f"{RECALL_HEADER}\n{lines}"
