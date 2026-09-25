"""T11 Skill 系统：SKILL.md 的加载、触发与动态工具注册。

目录契约（与 docs/design.md 第 7 节一致）：

    skills/<skill-name>/
      SKILL.md      # frontmatter: name / description / triggers；正文是技能说明
      tools.py      # 可选：该技能专用的工具（ToolDef + 同名 async 函数）

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
