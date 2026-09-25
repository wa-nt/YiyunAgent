"""T11 Skill 系统的触发侧：关键词匹配 → 加载正文 → 渲染成注入用的 system 消息。

触发方式选关键词匹配而不是 LLM 分类（brief 允许）：一次子串扫描，零额外延迟与成本，
而且行为完全可解释——命中了哪个词、为什么触发，都能从 traces 里反查。LLM 分类是
brief 里标注的可选路径（留给 T10 做对比），`detect_skill` 保持 async 就是为了将来
接上它时不用改调用方。

阈值（settings.skills_trigger_threshold）在关键词路径下的含义是**意图占优程度**：
命中多个 skill 时，赢家占全部命中的比例低于阈值就视为意图不明，本轮不加载 skill
（把话交给模型与检索自己判断）。这样 0.7 这个默认值是有判别力的——「帮我优化简历」
（1/1=1.0）触发，「简历和面试分别要准备什么」（1/2=0.5）不触发。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.config import settings
from app.skills.loader import Skill, SkillMeta

logger = logging.getLogger(__name__)

# 注入用的系统消息开头，测试与用户都靠它判别「这轮加载了哪个技能」
SKILL_PREFIX = "[已启用技能"


@dataclass(frozen=True)
class SkillMatch:
    """一次触发的结果：命中哪个 skill、命中了哪些触发词、意图占优程度。"""

    name: str
    triggers: tuple[str, ...]
    confidence: float


def match_skill(user_message: str, skills: dict[str, SkillMeta]) -> SkillMatch | None:
    """关键词匹配：返回得分最高且足够占优的 skill，没有则返回 None。

    得分 = 该 skill 命中的**不同**触发词个数（一个词重复出现只算一次）。并列时取
    加载顺序里的第一个（load_skills 按目录名排序，所以结果是确定的，不依赖字典
    哈希）。

    总开关在这里也查一次（runtime._apply_skill 另有更早的一次，为了连目录都不扫）：
    detect_skill / match_skill 是 app.skills 对外的触发入口，直接调用它们的人拿到的
    行为必须与 Agent 链路一致，否则「关掉 skill」这件事会有两个口径。
    """
    if not settings.skills_enabled or not skills or not user_message:
        return None
    threshold = _threshold()
    haystack = user_message.casefold()
    scored: list[tuple[int, SkillMeta, tuple[str, ...]]] = []
    for meta in skills.values():
        hits = tuple(t for t in meta.triggers if t.casefold() in haystack)
        if hits:
            scored.append((len(hits), meta, hits))
    if not scored:
        return None

    best_score = max(score for score, _, _ in scored)
    total = sum(score for score, _, _ in scored)
    # 严格大于：并列时保留先出现的那个（load_skills 的目录名顺序）
    score, meta, hits = next(s for s in scored if s[0] == best_score)
    confidence = score / total
    if confidence < threshold:
        logger.debug(
            "skill 触发意图不明，本轮不加载：%s（占优 %.2f < 阈值 %.2f）",
            [m.name for _, m, _ in scored],
            confidence,
            threshold,
        )
        return None
    return SkillMatch(name=meta.name, triggers=hits, confidence=confidence)


def _threshold() -> float:
    """触发阈值，钳制在 [0, 1]。配错了当最严（1.0）处理并告警。

    静默按原值比较的话，70 会让**任何** skill 都触发不了，而现象只是「skill 好像
    没生效」，最难从结果反推——所以这里偏向喊一声。

    NaN 必须单独判：`0 <= nan <= 1` 是 False，而 `min(max(nan, 0), 1)` 仍是 nan，
    接着 `confidence < nan` 恒为 False——阈值被**静默关成最松**（谁都触发），与
    「配错当最严」正好相反，且日志里看不出异常。
    """
    value = settings.skills_trigger_threshold
    if math.isnan(value):
        logger.warning("skills_trigger_threshold 是 NaN，已按 1.0（最严）处理")
        return 1.0
    if 0.0 <= value <= 1.0:
        return value
    logger.warning(
        "skills_trigger_threshold 应在 [0, 1] 内，当前 %r，已按钳制后的值处理", value
    )
    return min(max(value, 0.0), 1.0)


async def detect_skill(user_message: str, skills: dict[str, SkillMeta]) -> str | None:
    """检测用户消息触发的 skill 名；未命中或意图不明时返回 None（brief 的接口契约）。

    只要名字的调用方用它；需要命中的触发词（写 trace）用 match_skill。
    """
    match = match_skill(user_message, skills)
    return match.name if match else None


def render_skill_prompt(skill: Skill) -> str:
    """把 skill 渲染成一条 system 消息的内容：正文 + 本轮新增的工具清单。

    工具清单必须写进提示词：动态注册的工具对模型来说是「这一轮突然多出来的」，
    不说清有哪些、什么时候用，它既不会调也不敢调，注册就等于白注册。
    """
    parts = [
        f"{SKILL_PREFIX} {skill.meta.name}] {skill.meta.description}".rstrip(),
        "",
        skill.content,
    ]
    if skill.tools:
        names = "、".join(tool.name for tool in skill.tools)
        parts += ["", f"本轮额外提供该技能的专用工具：{names}。需要时直接调用。"]
    return "\n".join(parts).strip()
