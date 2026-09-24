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
from app.llm.types import LLMClient, Message

logger = logging.getLogger(__name__)

KINDS = ("preference", "fact", "goal", "project")
# 冲突判定每次最多送进 LLM 的同 kind 旧记忆条数：同 kind 活跃记忆不超过这个数时
# 全部参与判定，超出后只判最相似的一批（控制单次写入的 token 成本）
CONFLICT_WINDOW = 5
CONTEXT_LIMIT = 2000  # 参与抽取的用户消息 / 助手回答各自最大字符数，避免撑爆抽取 prompt
# 省 LLM 调用的旁路：与最相似的旧记忆共享字符都不到两成时，几乎不可能在讲同一件事，
# 直接跳过冲突判定。这是「省调用」与「漏判」的权衡——措辞差别很大的矛盾
# （「用户是素食主义者」vs「用户每周吃两次牛排」，相似度 0）同样会被跳过，
# 阈值刻意压得很低就是为了少漏一些。若 T10 评测显示记忆污染率偏高，优先调高它。
CONFLICT_MIN_SIMILARITY = 0.2
# 冲突衰减的下限：不设下限时反复冲突会把 confidence 压到 0，记忆等于废掉
MIN_CONFIDENCE = 0.1

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

# 重要性规则兜底用的信号。content 按 EXTRACT_PROMPT 的约定是第三人称，
# 所以偏好正则同时覆盖第一/第三人称写法（LLM 偶尔不守约定）
_VALUE_HINTS = re.compile(r"\d|[一二三四五六七八九十](?:年|月|日|岁|次|个|点)")
_PREFERENCE_HINTS = re.compile(
    "(?:我|用户|本人)(?:喜欢|讨厌|偏好|想|需要|习惯|爱|不爱)"
)
# 拉丁词信号只认长度 ≥3 且非停用词的 token（避免把 "the"/"for" 当成专有名词）
_LATIN_TOKEN = re.compile(r"[A-Za-z]{3,}")
_LATIN_STOPWORDS = frozenset(
    {"the", "and", "for", "you", "are", "was", "with", "that", "this", "not", "but"}
)

# 归一化去噪：去掉称呼前缀与标点空白，让「同一句话的两种写法」在比较前先对齐
_PREFIX = re.compile(r"^(?:用户|本人|我)+")
_NOISE = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()\[\]【】\-—…~·]+")

# 敏感信息：命中即整条丢弃，绝不落库（否则每轮都会回灌进 system 提示）
_API_KEY = re.compile(
    r"\bsk-[A-Za-z0-9_\-]{8,}|\bAKIA[0-9A-Z]{12,}|\bghp_[A-Za-z0-9]{20,}"
    r"|\bBearer\s+[A-Za-z0-9._\-]{12,}"
)
_SECRET_WORD = re.compile(
    r"(?:password|passwd|secret|token|api[_\-\s]?key|private[_\-\s]?key"
    r"|密码|密钥|口令|凭证)\s*[:：=是为]?\s*\S{4,}",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _has_latin_name(content: str) -> bool:
    return any(
        word.lower() not in _LATIN_STOPWORDS for word in _LATIN_TOKEN.findall(content)
    )


def estimate_importance(content: str) -> int:
    """LLM 没给分（或给的分非法）时的规则兜底：命中一类信号 +1，基准 2，上限 5。"""
    score = 2
    if _VALUE_HINTS.search(content) or _has_latin_name(content):
        score += 1
    if _PREFERENCE_HINTS.search(content):
        score += 1
    if len(content) > 50:
        score += 1
    return min(score, 5)


def normalize(content: str) -> str:
    """去称呼前缀与标点空白，用于相似度比较（不改变落库的原文）。"""
    return _NOISE.sub("", _PREFIX.sub("", content.strip()))


_CJK_DIGITS = {
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}
_NUMBER = re.compile(r"\d+|[一二两三四五六七八九十]")


def _numbers(content: str) -> list[str]:
    """抽出归一化文本里的数字（含中文数字），用于判断两条记忆的「事实」是否一致。"""
    tokens = _NUMBER.findall(normalize(content))
    return sorted(_CJK_DIGITS.get(token, token) for token in tokens)


def same_facts(a: str, b: str) -> bool:
    """数字/时间是否一致。日期、数量这类 token 只差一个字就是另一条事实
    （「2026 年 3 月投简历」vs「2026 年 6 月投简历」相似度 0.93，光靠阈值拦不住），
    所以数字不同一律不算重复。粗粒度映射（「十」→"10"）会把「二十」与「23」判成不同，
    代价只是多存一行 active 记忆，方向上是安全的。
    """
    return _numbers(a) == _numbers(b)


def has_secret(content: str) -> bool:
    """粗筛敏感信息：API key 形状，或「密码/token/密钥」后跟一段值。"""
    return bool(_API_KEY.search(content) or _SECRET_WORD.search(content))


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
    """把 LLM 输出规整为 (kind, content, importance)：非法项与敏感信息丢弃。

    过滤发生在这里而不是落库后，避免密码/密钥这类内容进了 memories 表又被
    每轮召回回灌进 system 提示（删库才能补救，代价高得多）。
    """
    items = payload.get("memories") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    out: list[tuple[str, str, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("content")
        if not isinstance(raw, str):
            continue
        content = raw.strip()
        if not content:
            continue
        if has_secret(content):
            logger.info("候选记忆命中敏感信息，已丢弃：%s", content)
            continue
        kind = item.get("kind")
        kind = kind.lower() if isinstance(kind, str) else ""
        if kind not in KINDS:
            kind = "fact"
        importance = item.get("importance")
        if not isinstance(importance, int) or isinstance(importance, bool):
            importance = estimate_importance(content)
        out.append((kind, content, min(max(importance, 1), 5)))
    return out


def _similarity(a: str, b: str) -> float:
    """memories 表暂无 embedding 字段，去重与排序先用归一化后的文本相似度替代 cosine。

    阈值口径见 settings.memory_dedup_ratio：比的是「去掉称呼前缀与标点后的整句序列
    相似度」，只对措辞差异鲁棒，一词之差改变事实的句子（上学/上班）刻意判为不同。
    """
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


async def _extract(
    llm: LLMClient, user_message: str, assistant_answer: str
) -> list[tuple[str, str, int]]:
    question = user_message[:CONTEXT_LIMIT]
    answer = assistant_answer[:CONTEXT_LIMIT]
    result = await llm.chat(
        [
            Message(role="system", content=EXTRACT_PROMPT),
            Message(role="user", content=f"用户：{question}\n\n助手：{answer}"),
        ]
    )
    return _normalize(_loads(result.text))


async def _of_kind(conn: aiosqlite.Connection, kind: str) -> list[dict]:
    """同 kind 下所有「未作废」的记忆（active + conflict），供冲突与去重判重使用。

    superseded 的行是版本链上的历史，不参与判重，否则新候选会反复撞上旧版本。
    """
    rows = await conn.execute_fetchall(
        "SELECT id, content, confidence, status FROM memories "
        "WHERE kind = ? AND status IN ('active', 'conflict') ORDER BY id",
        (kind,),
    )
    return [dict(row) for row in rows]


async def _detect_conflicts(
    llm: LLMClient, content: str, window: list[tuple[float, dict]]
) -> list[int]:
    """让 LLM 判断候选记忆与同 kind 旧记忆是否语义矛盾，返回矛盾记忆的 id。

    语义矛盾只能由模型判断：措辞完全不同的两句（「用户是素食主义者」与
    「用户每周吃两次牛排」）逐字相似度接近 0，却是最典型的冲突。
    """
    listing = "\n".join(f"{m['id']}. {m['content']}" for _, m in window)
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
    known = {m["id"] for _, m in window}
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
) -> dict:
    now = _now()
    cursor = await conn.execute(
        "INSERT INTO memories "
        "(kind, content, confidence, source, created_at, updated_at, status, supersedes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (kind, content, confidence, session_id, now, now, status, supersedes),
    )
    return {
        "id": cursor.lastrowid,
        "content": content,
        "confidence": confidence,
        "status": status,
    }


async def _decay(
    conn: aiosqlite.Connection, member: dict, decayed: set[int], now: str
) -> None:
    """置信度衰减并按 MIN_CONFIDENCE 兜底；同一轮内每个 id 只衰减一次。"""
    if member["id"] in decayed:
        return
    decayed.add(member["id"])
    member["confidence"] = max(
        MIN_CONFIDENCE, round(member["confidence"] * settings.memory_decay, 4)
    )
    await conn.execute(
        "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
        (member["confidence"], now, member["id"]),
    )


async def _store(
    conn: aiosqlite.Connection,
    llm: LLMClient,
    session_id: str,
    kind: str,
    content: str,
    importance: int,
    existing: list[dict],
    decayed: set[int],
) -> tuple[dict | None, int | None]:
    """单条候选的落库决策，返回 (新增的记忆行, 被取代的旧记忆 id)。

    先由 LLM 判语义矛盾：矛盾 → 新记忆挂 conflict 态、旧记忆衰减（不覆盖旧记忆）；
    再按归一化相似度判重：命中且语义等价 → 保留 confidence 高的那条；
    其余情况新增一条 active 记忆（措辞不同但不等价的事实并存，不静默丢弃）。
    """
    confidence = importance / 5
    if not existing:
        inserted = await _insert(
            conn, kind, content, confidence, "active", None, session_id
        )
        return inserted, None

    # 逐条打分：相似度 + 数字是否一致。数字不同的两条即使措辞几乎一样也是不同事实，
    # 不参与判重（见 same_facts 的说明）。同一份打分结果供冲突窗口与两处判重复用。
    scored: list[tuple[float, bool, dict]] = []
    for member in existing:
        scored.append(
            (
                _similarity(content, member["content"]),
                same_facts(content, member["content"]),
                member,
            )
        )
    scored.sort(key=lambda item: -item[0])

    window = [(sim, member) for sim, _, member in scored[:CONFLICT_WINDOW]]
    conflicts: list[int] = []
    if window[0][0] >= CONFLICT_MIN_SIMILARITY:
        conflicts = await _detect_conflicts(llm, content, window)
    else:
        logger.info(
            "候选与最相似的旧记忆相似度过低，跳过冲突判定：%.3f < %.2f（%s）",
            window[0][0],
            CONFLICT_MIN_SIMILARITY,
            content,
        )

    def find_dup(status: str) -> tuple[float, dict] | None:
        return next(
            (
                (sim, member)
                for sim, aligned, member in scored
                if member["status"] == status
                and aligned
                and sim >= settings.memory_dedup_ratio
            ),
            None,
        )

    now = _now()
    if conflicts:
        # 冲突不覆盖旧记忆：新记忆单独入册挂 conflict 态，旧记忆降权后仍可召回
        conflict_dup = find_dup("conflict")
        if conflict_dup is None:
            inserted = await _insert(
                conn, kind, content, confidence, "conflict", None, session_id
            )
        else:
            # 同一条矛盾语句反复出现时不再堆 conflict 行，只重新衰减旧记忆
            inserted = None
            logger.info(
                "候选与已有冲突记忆重复，不再插入：%s（相似 id=%s，相似度 %.3f）",
                content,
                conflict_dup[1]["id"],
                conflict_dup[0],
            )
        for _, member in window:
            if member["id"] in conflicts:
                await _decay(conn, member, decayed, now)
        return inserted, None

    best = find_dup("active")
    if best is not None:
        sim, member = best
        if confidence <= member["confidence"]:
            # 不静默丢弃：留下候选内容、比较对象与相似度，便于事后排查误判
            logger.info(
                "候选与已有记忆重复，丢弃：%s（相似 id=%s，相似度 %.3f，置信度 %.2f <= %.2f）",
                content,
                member["id"],
                sim,
                confidence,
                member["confidence"],
            )
            return None, None
        # 追加式版本链：新记忆 supersede 旧的，不直接改写历史记录
        inserted = await _insert(
            conn, kind, content, confidence, "active", member["id"], session_id
        )
        await conn.execute(
            "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
            (now, member["id"]),
        )
        return inserted, member["id"]

    inserted = await _insert(
        conn, kind, content, confidence, "active", None, session_id
    )
    return inserted, None


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
            # 同 kind 的未作废记忆读一次后随写入维护，同一批候选之间也能正确判重
            cache: dict[str, list[dict]] = {}
            # 一轮内每个旧记忆最多衰减一次，避免同批两条矛盾候选把它连乘两次
            decayed: set[int] = set()
            for kind, content, importance in candidates:
                if kind not in cache:
                    cache[kind] = await _of_kind(conn, kind)
                existing = cache[kind]
                inserted, retired = await _store(
                    conn, llm, session_id, kind, content, importance, existing, decayed
                )
                if retired is not None:
                    existing[:] = [m for m in existing if m["id"] != retired]
                if inserted is not None:
                    existing.append(inserted)
                await conn.commit()
    except Exception as exc:  # 记忆是增强项，写入失败不拖累对话
        logger.warning("记忆写入失败，已跳过：%s: %s", type(exc).__name__, exc)
