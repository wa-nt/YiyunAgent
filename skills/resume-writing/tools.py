"""resume-writing 的专用工具。

约定（与 app/skills/loader.py 的 _load_tools 一致）：

    TOOLS = [ToolDef(name="...", ...)]

    async def analyze_resume(args: dict, db_path: str | None) -> tuple[str, str]
        # 返回 (给模型看的结果文本, 前端展示用摘要)

两个工具都是**纯字面**分析：能查出「哪条 bullet 没有数字」「JD 里的词简历里有没有」，
判断不了「这段经历够不够硬」。结论要如实说明口径，别让模型把关键词覆盖率当成通过率。
"""

from __future__ import annotations

import re
from typing import Any

from app.llm.types import ToolDef

# 弱动词/职责式开头：简历里最典型的「写了等于没写」信号
WEAK_STARTS = (
    "负责", "参与", "协助", "帮助", "配合", "支持", "跟进", "熟悉", "了解",
    "负责了", "参加了",
)
# 数字或量化单位：命中任一即视为这条 bullet 有量化
_QUANTIFIED = re.compile(r"\d|百分之|%|倍|万|亿|千")
# bullet 行长阈值：超过就说明一句话塞了太多事，面试官不会读完
LONG_BULLET = 120
# 每种问题最多列几条例子（全列出来等于没重点）
MAX_EXAMPLES = 5

# 列表符号：`- item` / `• item` / `1. item` / `2) item` / `3、item`。
# 必须是**真正的**列表符号才剥——不能用 lstrip("0123456789.、) ") 这种字符集合，
# 那会把 bullet 正文开头的数字一起吃掉（「3 年 Go 后端开发经验」→「年 Go 后端开发
# 经验」）：既篡改了回显给用户看的内容，又让这条本来含量化的经历被判成「无量化」。
_BULLET_MARK = re.compile(r"^\s*(?:[-•*·>]|\d+[.、)）])\s*")

# JD 里提取英文技术词的字符集。取词后再剥掉尾部标点（见 _jd_keywords）
_EN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#._/-]{1,}")
# 词尾标点：英文逗号句点分号冒号，以及中文顿号逗号句号
_TRAILING_PUNCT = ".,;:、，。！？!?）)"

# JD 关键词里要忽略的词：英文虚词 + JD 的**结构用语**（职位名、年限、招聘套话）。
# 后一类不忽略的话，每份 JD 都会稳定贡献 5-8 个「永远不可能覆盖」的词，覆盖率被
# 系统性压低，看起来像简历缺了很多东西——而它们根本不是技能。
_STOPWORDS = {
    # 虚词 / 连接词
    "and", "or", "the", "with", "for", "you", "your", "our", "are", "will", "have",
    "has", "that", "this", "from", "not", "but", "all", "any", "can", "must", "we",
    "is", "to", "of", "in", "on", "at", "by", "as", "be", "a", "an", "it", "its",
    "if", "using", "use", "used", "good", "plus", "etc", "such", "while", "when",
    "who", "which", "what", "more", "than", "other", "also", "well", "new", "able",
    # 招聘结构用语
    "job", "role", "team", "work", "working", "years", "year", "experience",
    "experienced", "requirements", "requirement", "responsibilities", "responsibility",
    "include", "including", "include", "familiar", "familiarity", "preferred",
    "plus", "nice", "required", "require", "strong", "solid", "excellent",
    "proficient", "proficiency", "knowledge", "understanding", "ability", "skills",
    "skill", "candidate", "candidates", "position", "company", "opportunity",
    "benefits", "offer", "salary", "we're", "you'll", "our", "join", "looking",
    "seeking", "hiring", "description", "qualifications", "background", "degree",
    "bachelor", "master", "related", "field", "building", "improving", "reliability",
    "backend", "frontend", "fullstack", "engineer", "developer", "development",
    "senior", "junior", "lead", "staff", "principal", "intern", "engineers",
    "design", "develop", "maintain", "scale", "scaling", "ensure", "help", "support",
    "reliable", "reliability", "services", "service", "scalable", "quality",
    "practices", "environment", "tools", "technologies", "solutions", "deliver",
    "delivery", "collaborate", "collaboration", "communication", "problem",
    "solving", "fast", "paced", "complex", "large", "world", "best", "great",
}
# JD 里值得匹配的中文技能词。ATS 做的是字面匹配，这里保持一致：只认出现在表里的词，
# 不用分词器猜（猜出来的「关键词」会淹掉真正的信号）
CN_KEYWORDS = (
    "分布式", "高并发", "高可用", "微服务", "中间件", "消息队列", "容器化", "云原生",
    "机器学习", "深度学习", "大模型", "大语言模型", "推荐系统", "搜索", "向量检索",
    "知识图谱", "数据仓库", "数据挖掘", "特征工程", "模型训练", "模型推理", "量化",
    "性能优化", "监控", "告警", "单元测试", "自动化测试", "CI/CD", "代码review",
)

TOOLS: list[ToolDef] = [
    ToolDef(
        name="analyze_resume",
        description=(
            "对简历文本做字面体检：统计 bullet 数、无量化条目、弱动词开头、超长条目，"
            "并列出具体例子。用户贴出简历或要求分析简历时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "resume_text": {
                    "type": "string",
                    "description": "简历全文纯文本。用户没贴简历时留空，本工具会返回提示",
                }
            },
            "required": [],
        },
    ),
    ToolDef(
        name="jd_keyword_gap",
        description=(
            "比对岗位 JD 与简历的**字面**关键词覆盖：列出 JD 里出现、简历里没出现的技能词。"
            "用于判断简历是否会被 ATS/HR 的粗筛漏掉。需要同时提供 jd 与 resume_text。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "jd": {"type": "string", "description": "目标岗位的 JD 原文"},
                "resume_text": {"type": "string", "description": "简历全文纯文本"},
            },
            "required": ["jd", "resume_text"],
        },
    ),
]


async def analyze_resume(args: dict[str, Any], db_path: str | None) -> tuple[str, str]:
    """简历字面体检。db_path 不用（纯文本分析，不碰知识库）。"""
    text = _text_arg(args, "resume_text")
    if not text:
        return (
            "没有拿到简历文本。请让用户把简历内容贴进对话（或先说明简历已入库，"
            "由你先用 search_knowledge 取回相关片段），再把文本传给本工具。",
            "analyze_resume（缺少 resume_text）",
        )

    bullets = _bullets(text)
    if not bullets:
        return "这段文本里没识别出简历条目（每行一个条目，或以 -/•/数字开头）。", (
            "analyze_resume（未识别到条目）"
        )

    unquantified = [b for b in bullets if not _QUANTIFIED.search(b)]
    weak = [b for b in bullets if b.startswith(WEAK_STARTS)]
    long_ones = [b for b in bullets if len(b) > LONG_BULLET]

    lines = [f"简历体检（字面统计，共 {len(bullets)} 条条目）"]
    lines.append(
        f"- 无任何量化：{len(unquantified)} 条"
        f"（{len(unquantified) * 100 // len(bullets)}%）"
    )
    lines += _examples(unquantified)
    lines.append(f"- 弱动词/职责式开头：{len(weak)} 条")
    lines += _examples(weak)
    lines.append(f"- 超过 {LONG_BULLET} 字的条目：{len(long_ones)} 条")
    lines += _examples(long_ones)
    lines.append(
        "\n口径说明：这里只做字面判断。「有数字」不等于「有说服力」"
        "（上亿用户和节省了 2 秒是两回事），「职责式开头」也不必然是硬伤——"
        "请结合 STAR-C 逐条重写，而不是机械地把弱动词替换掉。"
    )
    summary = (
        f"analyze_resume：{len(bullets)} 条，无量化 {len(unquantified)}、"
        f"弱开头 {len(weak)}、超长 {len(long_ones)}"
    )
    return "\n".join(lines), summary


async def jd_keyword_gap(args: dict[str, Any], db_path: str | None) -> tuple[str, str]:
    """JD 与简历的字面关键词覆盖比对。db_path 不用。"""
    jd = _text_arg(args, "jd")
    resume = _text_arg(args, "resume_text")
    if not jd or not resume:
        return (
            "需要同时提供 jd 与 resume_text 两个参数（硬性要求，不做猜测）。",
            "jd_keyword_gap（缺少参数）",
        )

    keywords = _jd_keywords(jd)
    if not keywords:
        return (
            "这段 JD 里没提取到可匹配的技能词（本工具的词汇表有限，只认常见技术栈与"
            "中文技能词）。请改用语义对比：直接逐条读 JD 的要求，对照简历找缺口。",
            "jd_keyword_gap（未提取到关键词）",
        )

    lower_resume = resume.casefold()
    missing = [k for k in keywords if k.casefold() not in lower_resume]
    hit = [k for k in keywords if k.casefold() in lower_resume]

    lines = [
        f"JD 关键词覆盖（字面匹配）：{len(hit)}/{len(keywords)}，"
        f"覆盖率 {len(hit) * 100 // len(keywords)}%",
        f"- 已覆盖：{'、'.join(hit) if hit else '（无）'}",
        f"- 未覆盖：{'、'.join(missing) if missing else '（无）'}",
        "",
        "口径说明：这是 ATS 式字面匹配，**不**代表能力匹配。未覆盖的词分两类——"
        "真没做过（不要为了过筛编造，面试会穿）与做过但简历里没写这个说法"
        "（这类才值得补进简历，用真实经历的具体写法）。请先和用户确认每个缺口属于哪一类，"
        "再给修改建议。",
    ]
    summary = f"jd_keyword_gap：{len(hit)}/{len(keywords)} 覆盖，缺 {len(missing)} 个词"
    return "\n".join(lines), summary


# ---------- 内部工具 ----------


def _text_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return value.strip() if isinstance(value, str) else ""


def _bullets(text: str) -> list[str]:
    """按行拆条目，只剥**真正的**列表符号与首尾空白；太短的行（分节标题等）不算条目。

    剥符号用 `_BULLET_MARK`（要求符号后跟空白/标点），所以行首的数字与正文一起保留：
    「3 年 Go 后端开发经验」原样返回，「1. 负责支付网关」剥成「负责支付网关」。
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = _BULLET_MARK.sub("", raw).strip()
        if len(line) >= 8:  # 8 字以内多是「教育经历」「技能」这类分节标题
            out.append(line)
    return out


def _examples(items: list[str]) -> list[str]:
    shown = [f"    · {item[:60]}" for item in items[:MAX_EXAMPLES]]
    if len(items) > MAX_EXAMPLES:
        shown.append(f"    · …还有 {len(items) - MAX_EXAMPLES} 条")
    return shown


def _jd_keywords(jd: str) -> list[str]:
    """从 JD 里提取待匹配的关键词：英文技术词 + 词汇表里命中的中文技能词。

    英文词**剥掉词尾标点后**才进结果：`out` 里的词会原样拿去和简历做子串比较，
    留一个「Kubernetes.」的话，简历里明明写着 Kubernetes 也判成未覆盖，覆盖率
    系统性偏低（JD 的句末词几乎都带标点，一份 JD 能少算七八个词）。
    比对时统一 casefold（Python 与 python 是同一个词）。
    """
    seen: set[str] = set()
    out: list[str] = []
    for token in _EN_TOKEN.findall(jd):
        word = token.strip(_TRAILING_PUNCT)
        key = word.casefold()
        if len(key) < 2 or key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        out.append(word)
    for word in CN_KEYWORDS:
        if word in jd and word.casefold() not in seen:
            seen.add(word.casefold())
            out.append(word)
    return out
