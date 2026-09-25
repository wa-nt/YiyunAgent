"""T11 Skill 系统的加载侧：扫描 `skills/` 目录，解析 `SKILL.md`。

渐进式披露的三层，对应本模块的三个动作：

1. `load_skills`：只解析 frontmatter（name / description / triggers），长期常驻内存
2. `get_skill`：触发时才读正文，正文不进注册表也不进提示词
3. `get_skill` 内部 `_load_tools`：触发时才 import 该 skill 的 `tools.py`

取舍：

- frontmatter 用 pyproject 已声明的 python-frontmatter（它带的 PyYAML 也是现成依赖），
  不手写 YAML 解析器：多一个只认三个字段的解析器，比多用一个已有依赖更容易出错。
- 不做热更新（brief 的边界）：每轮重新扫目录、现读现解析，普通文件量级下开销可忽略，
  也省掉了缓存失效那一套。改 SKILL.md 下一轮就生效，反而比缓存更好用。
- skills 目录是用户可写的地方，所以单个 skill 的格式问题只跳过它并告警，不抛异常：
  一个手误不该让整个 Agent 起不来。
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import frontmatter

from app.llm.types import ToolDef

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
TOOLS_FILE = "tools.py"

# skill 专用工具的调用约定，与 app/agent/runtime.py 的 execute_tool 同一口径：
# 入参 (工具参数, db_path)，返回 (结果文本, 展示用摘要)
ToolFn = Callable[[dict[str, Any], "str | None"], Awaitable[tuple[str, str]]]


@dataclass(frozen=True)
class SkillMeta:
    """常驻内存的那一层：只有元数据，没有正文（正文按需读，见 get_skill）。"""

    name: str
    description: str
    triggers: tuple[str, ...]
    dir: Path
    has_tools: bool


@dataclass(frozen=True)
class Skill:
    """触发后才组装出来的完整 skill：正文 + 专用工具定义与实现。"""

    meta: SkillMeta
    content: str
    tools: tuple[ToolDef, ...] = ()
    tool_fns: dict[str, ToolFn] = field(default_factory=dict)


def load_skills(skills_dir: str | Path) -> dict[str, SkillMeta]:
    """扫描 `skills_dir/<skill-name>/SKILL.md`，返回 {name: SkillMeta}。

    注册顺序 = 目录名排序（触发词命中数并列时的兜底顺序依赖它，所以是显式的）。
    目录不存在返回 {}：没配 skill 的部署不该在每轮对话里报错或刷日志。
    """
    root = Path(skills_dir)
    if not root.is_dir():
        return {}
    metas: dict[str, SkillMeta] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        meta = _read_meta(entry)
        if meta is None:
            continue
        if meta.name in metas:
            logger.warning("skill 名称重复，已跳过后一个：%s（%s）", meta.name, entry)
            continue
        metas[meta.name] = meta
    return metas


def get_skill(name: str, skills: dict[str, SkillMeta]) -> Skill | None:
    """按需加载完整 skill：读正文 + import 专用工具。name 未知或读失败时返回 None。"""
    meta = skills.get(name)
    if meta is None:
        return None
    try:
        loaded = frontmatter.loads((meta.dir / SKILL_FILE).read_text(encoding="utf-8"))
    except Exception as exc:  # 文件被删/编码不对：本轮当没触发，不影响对话
        logger.warning(
            "skill %s 正文加载失败，本轮跳过：%s: %s", name, type(exc).__name__, exc
        )
        return None
    tools, tool_fns = _load_tools(meta) if meta.has_tools else ([], {})
    return Skill(
        meta=meta, content=loaded.content.strip(), tools=tuple(tools), tool_fns=tool_fns
    )


# ---------- frontmatter 解析 ----------


def _read_meta(skill_dir: Path) -> SkillMeta | None:
    """解析一个 skill 目录的 frontmatter；不是 skill 目录或格式不对则返回 None。"""
    path = skill_dir / SKILL_FILE
    if not path.is_file():
        return None  # 没有 SKILL.md：当普通目录跳过，不告警（resources/ 之类很常见）
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("SKILL.md 读取失败，已跳过 %s：%s", path, exc)
        return None
    try:
        metadata = frontmatter.loads(text).metadata
    except Exception as exc:
        logger.warning(
            "SKILL.md 的 frontmatter 解析失败，已跳过 %s：%s: %s",
            path,
            type(exc).__name__,
            exc,
        )
        return None
    if not isinstance(metadata, dict):  # YAML 顶层不是映射时 frontmatter 给的是 {}
        metadata = {}
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.warning("SKILL.md 缺少 name，已跳过 %s", path)
        return None
    return SkillMeta(
        name=name.strip(),
        description=_as_text(metadata.get("description")),
        triggers=_as_triggers(metadata.get("triggers"), path),
        dir=skill_dir,
        has_tools=(skill_dir / TOOLS_FILE).is_file(),
    )


def _as_text(raw: Any) -> str:
    return raw.strip() if isinstance(raw, str) else ""


def _as_triggers(raw: Any, path: Path) -> tuple[str, ...]:
    """触发词：接受 YAML 列表，也接受单个字符串（手写 SKILL.md 时的常见写法）。

    非字符串条目丢弃并告警——YAML 把 `triggers: 简历` 解析成字符串、把
    `triggers: [简历, 1]` 里的数字原样带出来，这两种都只是写得随意，不该让整个
    skill 加载失败。
    """
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        logger.warning("skill 的 triggers 类型不支持，按无触发词处理：%r（%s）", raw, path)
        items = []
    triggers: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            triggers.append(item.strip())
        else:
            logger.warning("skill 触发词不是非空字符串，已丢弃：%r（%s）", item, path)
    if not triggers:
        logger.warning("skill 没有任何可用触发词，永远不会被触发：%s", path)
    return tuple(triggers)


# ---------- tools.py 动态加载 ----------


def _load_tools(meta: SkillMeta) -> tuple[list[ToolDef], dict[str, ToolFn]]:
    """import skill 目录下的 tools.py，按 `TOOLS` 里的 ToolDef 名字配对同名函数。

    tools.py 的约定（见 skills/resume-writing/tools.py 的注释）：

        TOOLS = [ToolDef(name="analyze_resume", ...)]

        async def analyze_resume(args, db_path): ...   # 返回 (结果文本, 摘要)

    用 importlib 而不是 exec（brief 里提的是 exec）：exec 把整段代码塞进当前命名
    空间，出错只有行号、定位不到文件；import 有独立的模块命名空间，也不污染调用方。

    权衡：模块不注册进 sys.modules，所以 tools.py 不能 `import` 同目录的其它模块
    （要共享代码就放进同一个 tools.py）。当前两个内置 skill 的工具都是自包含的。
    """
    path = meta.dir / TOOLS_FILE
    module_name = f"_skill_tools_{meta.name.replace('-', '_')}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 {path} 构造模块规格")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # 工具是增强项：加载失败退化为「该技能没有专用工具」
        logger.warning(
            "skill %s 的 tools.py 加载失败，本轮无专用工具：%s: %s",
            meta.name,
            type(exc).__name__,
            exc,
        )
        return [], {}

    declared = getattr(module, "TOOLS", None)
    if not isinstance(declared, (list, tuple)):
        logger.warning("skill %s 的 tools.py 缺少 TOOLS 列表，本轮无专用工具", meta.name)
        return [], {}
    tools: list[ToolDef] = []
    fns: dict[str, ToolFn] = {}
    for tool in declared:
        if not isinstance(tool, ToolDef):
            logger.warning("skill %s 的 TOOLS 里有非 ToolDef 条目，已跳过：%r", meta.name, tool)
            continue
        fn = getattr(module, tool.name, None)
        if not callable(fn):
            logger.warning(
                "skill %s 的 tools.py 里没有可调用的 %s，该工具未注册", meta.name, tool.name
            )
            continue
        tools.append(tool)
        fns[tool.name] = fn
    return tools, fns
