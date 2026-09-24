from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

import aiosqlite

from app.config import settings
from app.db import get_db
from app.llm import get_llm
from app.llm.types import Message

logger = logging.getLogger(__name__)

KINDS = ("preference", "fact", "goal", "project")
# 冲突判定每次最多送进 LLM 的同 kind 旧记忆条数：同 kind 活跃记忆不超过这个数时
# 全部参与判定，超出后只判最相似的一批（控制单次写入的 token 成本）
CONFLICT_WINDOW = 5
ANSWER_LIMIT = 2000  # 参与抽取的助手回答最大字符数，避免长回答撑爆抽取 prompt

EXTRACT_PROMPT = (
    "你是记忆抽取器。从接下来这轮对话里抽取值得长期记住的、关于「用户」的信息，"
    "只输出 JSON 数组，不要输出解释或代码块标记。每项字段：\n"
    "  kind：preference（偏好）| fact（事实）| goal（目标）| project（项目）\n"
    "  content：一句话，第三人称陈述用户，例如「用户偏好用 Markdown 记笔记」\n"
    "  importance：1-5 的整数，5 表示最值得长期保留\n"
    "没有值得长期记住的内容时输出 []。不要抽取寒暄、一次性提问和助手自己的观点。"
)

CONFLICT_PROMPT = (
    "你是记忆冲突检测器。判断「新记忆」是否与「已有记忆」语义矛盾——"
    "指同一件事上相互否定（例如对同一事物一个说喜欢一个说讨厌、同一时间说法不同）。"
    "仅仅主题相近、可以同时成立的不算矛盾。\n"
    '只输出 JSON：{"conflicts": [与之矛盾的记忆 id，整数]}；没有矛盾时输出 {"conflicts": []}。'
)

# 重要性规则兜底用的信号：数字/时间/专有名词、偏好动词、篇幅
_VALUE_HINTS = re.compile(r"\d|[一二三四五六七八九十](?:年|月|日|岁|次|个|点)|[A-Za-z]{2,}")
_PREFERENCE_HINTS = re.compile("我喜欢|我讨厌|我想|我需要")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_importance(content: str) -> int:
    """LLM 没给分（或给的分非法）时的规则兜底：命中一类信号 +1，基准 2，上限 5。"""
    score = 2
    if _VALUE_HINTS.search(content):
        score += 1
    if _PREFERENCE_HINTS.search(content):
        score += 1
    if len(content) > 50:
        score += 1
    return min(score, 5)


def _loads(raw: str) -> Any:
    """容错解析 LLM 输出：允许 ``` 围栏，也允许 JSON 前后夹带说明文字。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for start, end in (("[", "]"), ("{", "}")):
        i, j = text.find(start), text.rfind(end)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _normalize(payload: Any) -> list[tuple[str, str, int]]:
    """把 LLM 输出规整为 (kind, content, importance)：非法项丢弃，缺分走规则兜底。"""
    items = payload.get("memories") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    out: list[tuple[str, str, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "fact"
        importance = item.get("importance")
        if not isinstance(importance, int) or isinstance(importance, bool):
            importance = estimate_importance(content)
        out.append((kind, content, min(max(importance, 1), 5)))
    return out


def _similarity(a: str, b: str) -> float:
    """memories 表暂无 embedding 字段，去重与排序先用文本相似度替代 cosine。"""
    return SequenceMatcher(None, a, b).ratio()


async def _extract(llm, user_message: str, assistant_answer: str) -> list[tuple[str, str, int]]:
    answer = assistant_answer[:ANSWER_LIMIT]
    result = await llm.chat(
        [
            Message(role="system", content=EXTRACT_PROMPT),
            Message(role="user", content=f"用户：{user_message}\n\n助手：{answer}"),
        ]
    )
    return _normalize(_loads(result.text))


async def _active_of_kind(conn: aiosqlite.Connection, kind: str) -> list[dict]:
    rows = await conn.execute_fetchall(
        "SELECT id, content, confidence FROM memories "
        "WHERE kind = ? AND status = 'active' ORDER BY id",
        (kind,),
    )
    return [dict(row) for row in rows]


async def _detect_conflicts(llm, content: str, existing: list[dict]) -> list[int]:
    """让 LLM 判断候选记忆与同 kind 旧记忆是否语义矛盾，返回矛盾记忆的 id。"""
    listing = "\n".join(f"{m['id']}. {m['content']}" for m in existing)
    result = await llm.chat(
        [
            Message(role="system", content=CONFLICT_PROMPT),
            Message(
                role="user",
                content=f"新记忆：{content}\n\n已有记忆：\n{listing}",
            ),
        ]
    )
    payload = _loads(result.text)
    ids = payload.get("conflicts") if isinstance(payload, dict) else payload
    if not isinstance(ids, list):
        return []
    known = {m["id"] for m in existing}
    return [
        i for i in ids if isinstance(i, int) and not isinstance(i, bool) and i in known
    ]


async def _insert(
    conn: aiosqlite.Connection,
    kind: str,
    content: str,
    confidence: float,
    status: str,
    supersedes: int | None,
    session_id: str,
) -> int:
    now = _now()
    cursor = await conn.execute(
        "INSERT INTO memories "
        "(kind, content, confidence, source, created_at, updated_at, status, supersedes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (kind, content, confidence, session_id, now, now, status, supersedes),
    )
    return cursor.lastrowid


async def _store(
    conn: aiosqlite.Connection,
    llm,
    session_id: str,
    kind: str,
    content: str,
    importance: int,
    existing: list[dict],
) -> tuple[dict | None, int | None]:
    """单条候选的落库决策，返回 (新增的活跃记忆, 被取代的旧记忆 id)。

    冲突 → 新记忆挂 conflict 态并衰减旧记忆；重复 → 保留 confidence 高的那条；
    其余情况新增一条 active 记忆。
    """
    confidence = importance / 5
    if not existing:
        new_id = await _insert(conn, kind, content, confidence, "active", None, session_id)
        return {"id": new_id, "content": content, "confidence": confidence}, None

    window = sorted(existing, key=lambda m: -_similarity(content, m["content"]))[
        :CONFLICT_WINDOW
    ]
    conflicts = await _detect_conflicts(llm, content, window)
    now = _now()
    if conflicts:
        # 冲突不覆盖旧记忆：新记忆单独入册挂冲突态，旧记忆降权后仍可召回
        await _insert(conn, kind, content, confidence, "conflict", None, session_id)
        conflicted = set(conflicts)
        for member in window:
            if member["id"] not in conflicted:
                continue
            member["confidence"] = round(member["confidence"] * settings.memory_decay, 4)
            await conn.execute(
                "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
                (member["confidence"], now, member["id"]),
            )
        return None, None

    best = window[0]
    if _similarity(content, best["content"]) >= settings.memory_dedup_threshold:
        if confidence <= best["confidence"]:
            return None, None
        # 追加式版本链：新记忆 supersede 旧的，不直接改写历史记录
        new_id = await _insert(
            conn, kind, content, confidence, "active", best["id"], session_id
        )
        await conn.execute(
            "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
            (now, best["id"]),
        )
        return {"id": new_id, "content": content, "confidence": confidence}, best["id"]

    new_id = await _insert(conn, kind, content, confidence, "active", None, session_id)
    return {"id": new_id, "content": content, "confidence": confidence}, None


async def extract_and_store(
    session_id: str, user_message: str, assistant_answer: str, db_path: str | None = None
) -> None:
    """从一轮对话中抽取候选记忆并治理后写入长期记忆。

    fire-and-forget：调用方不等待，任何失败只记日志、不影响主流程。工作记忆
    （本轮候选）只在内存中流转，最终要么进长期记忆要么丢弃。
    """
    if not settings.memory_enabled:
        return
    try:
        llm = get_llm()
        candidates = await _extract(llm, user_message, assistant_answer)
        if not candidates:
            return
        async with get_db(db_path) as conn:
            # 同 kind 的活跃记忆读一次后随写入维护，同一批候选之间也能正确判重
            cache: dict[str, list[dict]] = {}
            for kind, content, importance in candidates:
                if kind not in cache:
                    cache[kind] = await _active_of_kind(conn, kind)
                existing = cache[kind]
                inserted, retired = await _store(
                    conn, llm, session_id, kind, content, importance, existing
                )
                if retired is not None:
                    existing[:] = [m for m in existing if m["id"] != retired]
                if inserted is not None:
                    existing.append(inserted)
                await conn.commit()
    except Exception as exc:  # 记忆是增强项，写入失败不拖累对话
        logger.warning("记忆写入失败，已跳过：%s: %s", type(exc).__name__, exc)
