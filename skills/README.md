# skills/ —— Skill 目录契约

每个子目录是一个 skill，Agent 在用户消息命中触发词时按需加载它。

> **安全提示**：SKILL.md 的正文会作为 system 消息注入提示词，tools.py 会被直接
> import 执行——两者都**等价于本机代码**，且提示词注入还能撬动 Agent 的其它工具。
> 不要拷贝来源不明的 skill 进这个目录。

```text
skills/<skill-name>/
  SKILL.md      # 必需
  tools.py      # 可选：该 skill 专用的工具
```

## SKILL.md

```markdown
---
name: resume-writing          # 必需，必须与目录名一致（重复时按目录名排序取先出现的）
description: 一句话说明这个技能做什么
triggers:                     # 触发词，大小写不敏感的子串匹配（按 casefold 比较）
  - 简历
  - resume
---

正文：技能的能力、流程、输出格式、示例。触发后作为 system 消息注入提示词。
```

- `triggers` 写成单个字符串也可以（`triggers: 简历`）。
- 正文只在**触发后**才读进提示词（渐进式披露的第一层是 name + description）。
- 文件编码需为 **UTF-8**（带不带 BOM 都行）。编码不对的 skill 会被跳过并告警，
  不影响同一目录里的其它 skill，也不影响对话。

## tools.py

```python
from app.llm.types import ToolDef

TOOLS = [ToolDef(name="analyze_resume", description="…", parameters={…})]

async def analyze_resume(args: dict, db_path: str | None) -> tuple[str, str]:
    # 返回 (给模型看的结果文本, 前端展示用摘要)
    return "结果", "analyze_resume：…"
```

- 工具名与 `TOOLS` 里的 `ToolDef.name` 必须一致，否则该工具不会注册（只告警）。
- 不能与内置工具同名（`search_knowledge`）——重名会被拒绝并告警，因为那会让 skill
  的实现永远不可达。
- 工具只在触发该 skill 的那一轮可用，不会进其它轮的对话。
- 工具是用户手写的代码：抛异常会被收口成「工具失败」并照常记 trace，不会打断对话。
- **每次触发都会重新执行模块顶层代码**（不做缓存，换来「改完下一轮就生效」）：
  顶层放常量、正则、纯函数定义没问题；要连数据库、起客户端、读大文件的初始化请放进
  工具函数内部或做模块级惰性缓存。

## 内置 skill

| 名称 | 用途 |
|---|---|
| `resume-writing` | 简历分析与优化（含 `analyze_resume` / `jd_keyword_gap` 两个工具） |
| `interview-prep` | 面试准备（自我介绍、项目深挖、算法、反问） |
| `note-organizer` | 笔记归类与周复习计划、知识盲区 |
| `learning-planner` | 学习目标拆解为带检查点的周计划 |
| `project-ideas` | 从现有笔记出发找可落地的练手项目 |

## 配置

- `SKILLS_ENABLED`：总开关，关掉后不扫目录、不注入、不注册工具
- `SKILLS_DIR`：skill 根目录，默认 `skills`
- `SKILLS_TRIGGER_THRESHOLD`：命中多个 skill 时，赢家命中数占全部命中的比例低于
  此值就视为意图不明，本轮不加载（默认 0.7）。「帮我优化简历」= 1/1 触发，
  「简历和面试分别要准备什么」= 1/2 不触发

改完 `SKILL.md` 下一轮对话就生效（每轮重新扫目录，不做缓存，见
`app/skills/loader.py` 的取舍说明）。

## 与设计文档的差异

`docs/design.md` 第 7 节还规定了 `scripts/`、`resources/` 两个目录，以及「脚本和资源
必须由注册表允许后才能执行或读取」的权限闸门。本实现是那个契约的**子集**，这两项未
实现（brief 的边界划掉了它们）：现有 skill 的能力全部落在 SKILL.md 正文与 tools.py
里。若目录中出现这两个子目录，会被当作无关内容忽略。
