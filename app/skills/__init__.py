"""T11 Skill 系统：SKILL.md 的加载、触发与动态工具注册。

目录契约：

    skills/<skill-name>/
      SKILL.md      # frontmatter: name / description / triggers；正文是技能说明
      tools.py      # 可选：该技能专用的工具（ToolDef + 同名 async 函数）

**这是 docs/design.md 第 7 节的子集**，差异是有意的（brief 的边界划掉了后两项）：

- design.md 还规定 `scripts/` 与 `resources/` 两目录，以及「脚本和资源必须由注册表
  允许后才能执行或读取」的权限闸门。本实现没有这两项目前也不需要：现有 skill 的能力
  全部落在 SKILL.md 正文与 tools.py 的函数里。目录里若出现这两个子目录会被当作无关
  内容忽略（没有 SKILL.md 的目录直接跳过）。
- SKILL.md 的编码需为 UTF-8（可带 BOM）；GBK/UTF-16 会告警并跳过该 skill，
  不做编码猜测（猜错会把乱码注入提示词，比明确跳过更难排查）。

渐进式披露体现在三层加载上：`load_skills` 只常驻 name/description/triggers，触发后
`get_skill` 才读正文，正文里点名的工具才 import。接入点在 app/agent/runtime.py 的
run_agent：命中后把正文作为 system 消息注入（排在记忆之后），本轮工具集加上该技能
的专用工具，并记一条 kind='skill' 的 trace。
"""

from app.skills.loader import Skill, SkillMeta, ToolFn, get_skill, load_skills
from app.skills.trigger import SkillMatch, detect_skill, match_skill, render_skill_prompt

__all__ = [
    "Skill",
    "SkillMatch",
    "SkillMeta",
    "ToolFn",
    "detect_skill",
    "get_skill",
    "load_skills",
    "match_skill",
    "render_skill_prompt",
]
