"""work 模式的两个内置工具：记录 / 复习知识漏洞。

handler 约定与 skills/<name>/tools.py 一致（见 app/skills/loader.py 的 ToolFn）：

    async def fn(args: dict, db_path: str | None) -> tuple[str, str]

返回 (给模型看的结果文本, 前端展示用摘要)。

参数一律**自己校验**并把失败写成一句可读说明，不抛异常：工具参数由模型生成，格式
不对是常态；抛出去会走 run_agent 的降级分支，模型只得到一句「工具失败」，拿不到
「缺哪个参数」这条能直接纠正的信息。领域错误（id 不存在）同理——它也不是异常情况，
用户删过的卡片再被复习一次是正常会发生的事。
"""

from __future__ import annotations

from typing import Any

from app.llm.types import ToolDef
from app.skills.loader import ToolFn
from app.study.gaps import GapNotFoundError, record_gap, review_gap

RECORD_TOOL = ToolDef(
    name="record_knowledge_gap",
    description=(
        "记录一个用户尚未掌握的知识点，生成一张间隔重复的复习卡片（默认 1 天后复习）。"
        "当用户明确表示没听懂、答错、或对某个概念只有模糊印象时调用。"
        "topic 是能一眼认出的主题，detail 写清楚具体卡在哪里——越具体，之后复习越有用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "漏洞主题，如「向量检索的索引选型」",
            },
            "detail": {
                "type": "string",
                "description": "具体没掌握的部分，如「分不清 HNSW 与 IVF 的取舍」",
            },
        },
        "required": ["topic", "detail"],
    },
)

REVIEW_TOOL = ToolDef(
    name="review_knowledge_gap",
    description=(
        "复习一张知识漏洞卡片并按结果排下一次复习：用户答对（passed=true）间隔翻倍，"
        "答错或答不上来（passed=false）间隔重置为 1 天。"
        "先由你出题让用户实际回答，拿到结果再调用本工具，不要替用户判断对错。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "gap_id": {
                "type": "integer",
                "description": "卡片 id（来自记录结果或漏洞列表）",
            },
            "passed": {"type": "boolean", "description": "用户本次是否真的答对"},
        },
        "required": ["gap_id", "passed"],
    },
)

TOOLS: list[ToolDef] = [RECORD_TOOL, REVIEW_TOOL]


async def record_knowledge_gap(args: dict[str, Any], db_path: str | None) -> tuple[str, str]:
    topic = _text_arg(args, "topic")
    detail = _text_arg(args, "detail")
    missing = "、".join(
        key for key, value in (("topic", topic), ("detail", detail)) if not value
    )
    if missing:
        return (
            f"记录失败：缺少 {missing} 参数（两者都是必填，且不能只有空白）。",
            f"record_knowledge_gap（缺少 {missing} 参数）",
        )

    gap = await record_gap(topic, detail, db_path)
    result = (
        f"已记录知识漏洞 #{gap['id']}：{topic}\n"
        f"细节：{detail}\n"
        f"下次复习：{gap['next_review_at']}（{gap['interval_days']} 天后）"
    )
    return result, f"record_knowledge_gap(#{gap['id']} {topic}) → {gap['interval_days']} 天后"


async def review_knowledge_gap(args: dict[str, Any], db_path: str | None) -> tuple[str, str]:
    gap_id = _int_arg(args, "gap_id")
    passed = _bool_arg(args, "passed")
    missing = "、".join(
        key for key, value in (("gap_id", gap_id), ("passed", passed)) if value is None
    )
    if missing:
        return (
            f"复习失败：参数不对（{missing}）——gap_id 要是卡片 id 的整数，"
            "passed 要是 true/false。",
            f"review_knowledge_gap（参数不对：{missing}）",
        )

    try:
        gap = await review_gap(gap_id, passed, db_path)
    except GapNotFoundError:
        return (
            f"复习失败：漏洞 #{gap_id} 不存在（可能已被删除）。"
            "可以用记录工具重新记录一张。",
            f"review_knowledge_gap（#{gap_id} 不存在）",
        )

    verdict = "通过" if passed else "未通过"
    result = (
        f"漏洞 #{gap['id']}（{gap['topic']}）复习{verdict}："
        f"间隔更新为 {gap['interval_days']} 天，下次复习 {gap['next_review_at']}"
    )
    return result, f"review_knowledge_gap(#{gap['id']}，{verdict}) → 间隔 {gap['interval_days']} 天"


TOOL_FNS: dict[str, ToolFn] = {
    RECORD_TOOL.name: record_knowledge_gap,
    REVIEW_TOOL.name: review_knowledge_gap,
}


def _text_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int_arg(args: dict[str, Any], key: str) -> int | None:
    """整数参数。接受 int 与纯数字字符串（不少模型把 id 写成 `"3"`）；bool 不算——
    它是 int 的子类，`True` 当成 gap_id 1 是纯粹的误解。"""
    value = args.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return None


def _bool_arg(args: dict[str, Any], key: str) -> bool | None:
    """真假参数。接受 bool、0/1 与 `"true"`/`"false"`：模型的 JSON 参数偶尔会是字符串，
    为此拒掉一次复习不值得（辨别不了的写法返回 None，由调用方报参数错误）。"""
    value = args.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    return None
