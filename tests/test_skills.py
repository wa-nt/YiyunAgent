"""T11 Skill 系统测试：SKILL.md 加载 / 关键词触发 / 动态工具注册 / 接入 run_agent。

全部 LLM 交互都是桩（chat_stream 由 FakeLLM 提供），不发真实 API 请求。
skills_dir 默认指向 tmp：内置 skills/ 目录是项目的一部分，用例断言的是**可控的**
skill 集合，让内置目录的触发词改动把这里判红是假警报。内置 skill 另有专门的用例
（见文件末尾），验的是「能被加载、正文非空、工具真能跑」。

tracing 总开关由 tests/conftest.py 的 autouse 夹具钉住为 False，断言 trace 的用例
自己显式打开并从 tmp 库里读。
"""

import json
import logging

import pytest

from app import tracing
from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk, ToolCall, ToolDef
from app.memory import writer as memory_writer
from app.memory.recall import RECALL_HEADER
from app.retrieval.bm25_search import invalidate
from app.skills import trigger
from app.skills.loader import get_skill, load_skills

DIM = 8
SESSION = "s-skill"


# ---------- 桩与夹具 ----------


class _SilentWriterLLM:
    """记忆抽取桩：不发请求、不产出记忆。"""

    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


class FakeLLM:
    """按脚本逐轮返回 chunk；记录每轮收到的 messages 与 tools。"""

    def __init__(self, rounds: list[list[StreamChunk]]):
        self.rounds = rounds
        self.calls: list[list] = []
        self.tools: list[list[ToolDef]] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        self.tools.append(list(tools or []))
        index = min(len(self.calls) - 1, len(self.rounds) - 1)
        for c in self.rounds[index]:
            yield c


def final(calls: list[ToolCall] | None = None) -> StreamChunk:
    return StreamChunk(finish=True, tool_calls=calls or [])


def answer(text: str = "好") -> list[StreamChunk]:
    return [StreamChunk(text_delta=text), final()]


def call_round(name: str, arguments: dict | None = None) -> list[StreamChunk]:
    return [final([ToolCall(id="c1", name=name, arguments=arguments or {})])]


TOOLS_PY = '''\
from app.llm.types import ToolDef

TOOLS = [
    ToolDef(
        name="echo_tool",
        description="回显入参",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )
]


async def echo_tool(args, db_path):
    return f"看到：{args.get('text')}", "echo_tool(演示)"
'''


def write_skill(
    root,
    dir_name: str,
    *,
    name: str | None = None,
    description: str = "演示用技能",
    triggers: str = "  - 演示\n",
    body: str = "# 正文\n\n只有触发后才读进来的一段指导。\n",
    raw: str | None = None,
    tools_py: str | None = None,
) -> None:
    """在 root 下造一个 skill 目录。raw 非空时 SKILL.md 整份由它决定（测畸形输入用）。"""
    skill_dir = root / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    text = raw or (
        "---\n"
        f"name: {name or dir_name}\n"
        f"description: {description}\n"
        f"triggers:\n{triggers}"
        "---\n\n"
        f"{body}"
    )
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    if tools_py is not None:
        (skill_dir / "tools.py").write_text(tools_py, encoding="utf-8")


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    """tmp 的 skills 目录 + 指向它的配置。"""
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(settings, "skills_dir", str(root))
    monkeypatch.setattr(settings, "skills_enabled", True)
    monkeypatch.setattr(settings, "skills_trigger_threshold", 0.7)
    return root


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """tmp 数据库 + 隔离的 BM25 缓存；记忆抽取换成空桩（不发真实请求）。"""
    monkeypatch.setattr(runtime.settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    invalidate()
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"
    await runtime.drain_memory_writes()
    await tracing.drain_traces()
    invalidate()


async def add_memory(db, content: str = "用户在准备大模型实习面试") -> None:
    async with get_db(db) as conn:
        await conn.execute(
            "INSERT INTO memories (kind, content, confidence, source, created_at, "
            "updated_at, status) VALUES ('fact', ?, 0.9, 't', '2026-09-24', '2026-09-24', 'active')",
            (content,),
        )
        await conn.commit()


async def trace_rows(db) -> list[dict]:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT kind, name, detail FROM traces ORDER BY id", ()
        )
    return [dict(r) for r in rows]


# ---------- 加载：frontmatter ----------


def test_load_skills_parses_frontmatter(skills_root):
    write_skill(skills_root, "demo-skill")

    metas = load_skills(skills_root)

    assert list(metas) == ["demo-skill"]
    meta = metas["demo-skill"]
    assert meta.name == "demo-skill"
    assert meta.description == "演示用技能"
    assert meta.triggers == ("演示",)
    assert meta.has_tools is False
    # 渐进式披露：常驻的元数据里没有正文
    assert not hasattr(meta, "content")


def test_load_skills_missing_dir_is_empty(tmp_path):
    assert load_skills(tmp_path / "nope") == {}


def test_dirs_without_skill_md_are_skipped_quietly(skills_root, caplog):
    """resources/ 这类没有 SKILL.md 的目录是正常情况，不该刷告警。"""
    (skills_root / "resources").mkdir()
    write_skill(skills_root, "demo-skill")

    with caplog.at_level(logging.WARNING, logger="app.skills.loader"):
        metas = load_skills(skills_root)

    assert list(metas) == ["demo-skill"]
    assert caplog.text == ""


def test_load_skills_skips_bad_frontmatter(skills_root, caplog):
    write_skill(skills_root, "broken-yaml", raw="---\nname: [a\n---\n正文\n")
    write_skill(skills_root, "no-name", raw="---\ndescription: 没有 name\n---\n正文\n")
    write_skill(skills_root, "no-frontmatter", raw="# 只有正文\n")
    write_skill(skills_root, "demo-skill")

    with caplog.at_level(logging.WARNING, logger="app.skills.loader"):
        metas = load_skills(skills_root)

    assert list(metas) == ["demo-skill"]  # 坏的那些只跳过，不影响其它 skill
    assert "解析失败" in caplog.text and "缺少 name" in caplog.text


def test_triggers_accept_string_list_and_inline(skills_root):
    write_skill(skills_root, "single", raw="---\nname: single\ntriggers: 简历\n---\n正文\n")
    write_skill(
        skills_root, "inline", raw="---\nname: inline\ntriggers: [面试, 面经]\n---\n正文\n"
    )
    write_skill(
        skills_root,
        "bad-items",
        raw="---\nname: bad-items\ntriggers: [简历, 3, '  ']\n---\n正文\n",
    )

    metas = load_skills(skills_root)

    assert metas["single"].triggers == ("简历",)
    assert metas["inline"].triggers == ("面试", "面经")
    assert metas["bad-items"].triggers == ("简历",)  # 非字符串条目丢弃


def test_skill_without_triggers_warns_but_loads(skills_root, caplog):
    write_skill(skills_root, "no-trigger", triggers="")

    with caplog.at_level(logging.WARNING, logger="app.skills.loader"):
        metas = load_skills(skills_root)

    assert metas["no-trigger"].triggers == ()
    assert "永远不会被触发" in caplog.text


def test_duplicate_name_keeps_first(skills_root, caplog):
    write_skill(skills_root, "a-dir", name="same")
    write_skill(skills_root, "b-dir", name="same")

    with caplog.at_level(logging.WARNING, logger="app.skills.loader"):
        metas = load_skills(skills_root)

    assert list(metas) == ["same"]
    assert metas["same"].dir.name == "a-dir"  # 目录名排序里先出现的胜出
    assert "名称重复" in caplog.text


# ---------- 加载：按需读正文与 tools.py ----------


def test_get_skill_loads_body_on_demand(skills_root):
    write_skill(skills_root, "demo-skill", body="# 正文\n\n只有触发后才读。\n")
    metas = load_skills(skills_root)

    skill = get_skill("demo-skill", metas)

    assert skill is not None
    assert "只有触发后才读。" in skill.content
    assert "---" not in skill.content  # frontmatter 不进正文
    assert skill.tools == () and skill.tool_fns == {}


def test_get_skill_unknown_name_is_none(skills_root):
    assert get_skill("不存在", load_skills(skills_root)) is None


def test_get_skill_returns_none_when_body_unreadable(skills_root):
    """注册后 SKILL.md 被删（人手动改了目录）：本轮当没触发，不抛。"""
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)
    (skills_root / "demo-skill" / "SKILL.md").unlink()

    assert get_skill("demo-skill", metas) is None


def test_get_skill_loads_tools_py(skills_root):
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    metas = load_skills(skills_root)

    assert metas["demo-skill"].has_tools is True
    skill = get_skill("demo-skill", metas)

    assert [t.name for t in skill.tools] == ["echo_tool"]
    assert skill.tools[0].description == "回显入参"
    assert set(skill.tool_fns) == {"echo_tool"}


def test_tools_py_failures_degrade_to_no_tools(skills_root, caplog):
    """写了 TOOLS 却没有同名函数 / tools.py 语法错误：只告警，不抛、不注册。"""
    write_skill(
        skills_root,
        "missing-fn",
        tools_py=(
            'from app.llm.types import ToolDef\nTOOLS=[ToolDef(name="ghost",description="x")]\n'
        ),
    )
    write_skill(skills_root, "broken-py", tools_py="def (:\n")
    write_skill(skills_root, "no-tools-list", tools_py="X = 1\n")

    with caplog.at_level(logging.WARNING, logger="app.skills.loader"):
        metas = load_skills(skills_root)
        assert get_skill("missing-fn", metas).tools == ()
        assert get_skill("broken-py", metas).tools == ()
        assert get_skill("no-tools-list", metas).tools == ()

    assert "没有可调用的 ghost" in caplog.text
    assert "tools.py 加载失败" in caplog.text
    assert "缺少 TOOLS 列表" in caplog.text


# ---------- 触发：关键词匹配 ----------


def test_match_skill_hits_keyword(skills_root):
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)

    match = trigger.match_skill("帮我看看这个演示", metas)

    assert match is not None
    assert match.name == "demo-skill" and match.triggers == ("演示",)
    assert match.confidence == 1.0


def test_match_skill_no_keyword_returns_none(skills_root):
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)

    assert trigger.match_skill("今天天气不错", metas) is None
    assert trigger.match_skill("", metas) is None


def test_match_skill_is_case_insensitive(skills_root):
    write_skill(skills_root, "en", raw="---\nname: en\ntriggers: [resume]\n---\n正文\n")
    metas = load_skills(skills_root)

    assert trigger.match_skill("帮我看看 resume", metas).name == "en"
    assert trigger.match_skill("我的 RESUME 有问题", metas).name == "en"


def test_same_trigger_twice_counts_once(skills_root):
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)

    match = trigger.match_skill("演示、演示、还是演示", metas)

    assert match.triggers == ("演示",) and match.confidence == 1.0


def test_ambiguous_intent_stays_below_threshold(skills_root):
    """两个 skill 各命中一个词 → 占优 0.5 < 0.7，本轮不加载任何 skill。"""
    write_skill(skills_root, "a-one", name="a", triggers="  - 简历\n")
    write_skill(skills_root, "b-two", name="b", triggers="  - 面试\n")
    metas = load_skills(skills_root)

    assert trigger.match_skill("简历和面试分别要准备什么", metas) is None
    # 只提一个时占优 1.0，正常触发
    assert trigger.match_skill("帮我优化简历", metas).name == "a"


def test_dominant_skill_wins_over_partial_hits(skills_root):
    """3:1 的占优（0.75 ≥ 0.7）算意图明确，赢家通吃。"""
    write_skill(skills_root, "a-one", name="a", triggers="  - 简历\n")
    write_skill(
        skills_root,
        "b-three",
        name="b",
        triggers="  - 简历优化\n  - 简历修改\n  - 改简历\n",
    )
    metas = load_skills(skills_root)

    match = trigger.match_skill("帮我做简历优化、简历修改，顺便改简历", metas)

    assert match.name == "b"
    assert len(match.triggers) == 3 and match.confidence == 0.75


def test_threshold_out_of_range_is_clamped(skills_root, caplog, monkeypatch):
    """阈值按百分数写成 70 不能静默让所有 skill 失效，也不能静默全放行。"""
    write_skill(skills_root, "a-one", name="a", triggers="  - 简历\n")
    write_skill(skills_root, "b-two", name="b", triggers="  - 面试\n")
    metas = load_skills(skills_root)

    with caplog.at_level(logging.WARNING, logger="app.skills.trigger"):
        monkeypatch.setattr(settings, "skills_trigger_threshold", 70.0)
        assert trigger.match_skill("简历和面试", metas) is None  # 钳到 1.0
        monkeypatch.setattr(settings, "skills_trigger_threshold", -3.0)
        assert trigger.match_skill("简历和面试", metas).name == "a"  # 钳到 0.0

    assert caplog.text.count("应在 [0, 1] 内") == 2


async def test_detect_skill_returns_name(skills_root):
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)

    assert await trigger.detect_skill("来个演示", metas) == "demo-skill"
    assert await trigger.detect_skill("无关内容", metas) is None


async def test_detect_skill_respects_disabled_switch(skills_root, monkeypatch):
    write_skill(skills_root, "demo-skill")
    metas = load_skills(skills_root)
    monkeypatch.setattr(settings, "skills_enabled", False)

    assert await trigger.detect_skill("来个演示", metas) is None
    assert trigger.match_skill("来个演示", metas) is None


def test_render_skill_prompt_includes_body_and_tool_names(skills_root):
    write_skill(
        skills_root, "demo-skill", tools_py=TOOLS_PY, body="# 正文\n\n指导内容。\n"
    )
    skill = get_skill("demo-skill", load_skills(skills_root))

    prompt = trigger.render_skill_prompt(skill)

    assert prompt.startswith("[已启用技能 demo-skill]")
    assert "演示用技能" in prompt and "指导内容。" in prompt
    assert "echo_tool" in prompt  # 动态工具要说清存在，否则模型不会调


def test_render_skill_prompt_without_tools_omits_tool_line(skills_root):
    write_skill(skills_root, "demo-skill")

    skill = get_skill("demo-skill", load_skills(skills_root))

    assert "专用工具" not in trigger.render_skill_prompt(skill)


# ---------- 注入：run_agent 的接线 ----------


async def messages_of(session_id: str, db) -> list[dict]:
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    return [dict(r) for r in rows]


async def test_triggered_skill_is_injected_after_memory(db, skills_root, monkeypatch):
    """正文作为 system 消息插在记忆之后；专用工具进本轮工具集。"""
    write_skill(skills_root, "demo-skill", body="# 正文\n\n技能指导内容。\n")
    await add_memory(db)
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    sent = llm.calls[0]
    assert [m.role for m in sent] == ["system", "system", "system", "user"]
    assert sent[1].content.startswith(RECALL_HEADER)  # 记忆在前
    assert sent[2].content.startswith("[已启用技能 demo-skill]")  # skill 更贴近本轮任务
    assert "技能指导内容。" in sent[2].content
    assert sent[-1].content == "来个演示"


async def test_untriggered_message_keeps_prompt_and_tools_unchanged(
    db, skills_root, monkeypatch
):
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "今天天气不错", db)]

    assert [m.role for m in llm.calls[0]] == ["system", "user"]
    assert [t.name for t in llm.tools[0]] == ["search_knowledge"]


async def test_triggered_skill_registers_tools(db, skills_root, monkeypatch):
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    assert [t.name for t in llm.tools[0]] == ["search_knowledge", "echo_tool"]


async def test_skills_disabled_never_triggers(db, skills_root, monkeypatch):
    """skills_enabled=False：即使消息命中触发词也不注入、不注册工具、不记 trace。"""
    monkeypatch.setattr(settings, "tracing_enabled", True)
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    monkeypatch.setattr(settings, "skills_enabled", False)
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]
    await tracing.drain_traces()

    assert [m.role for m in llm.calls[0]] == ["system", "user"]
    assert [t.name for t in llm.tools[0]] == ["search_knowledge"]
    assert await trace_rows(db) == []


async def test_missing_skills_dir_degrades_silently(db, tmp_path, monkeypatch):
    """目录不存在（没配 skill 的部署）：正常回答，不报错。"""
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "no-such-dir"))
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    assert [e.type for e in events] == ["text_delta", "done"]
    assert [m.role for m in llm.calls[0]] == ["system", "user"]


async def test_trigger_recorded_as_skill_trace(db, skills_root, monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", True)
    write_skill(skills_root, "demo-skill")
    monkeypatch.setattr(runtime, "get_llm", lambda: FakeLLM([answer()]))

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]
    await tracing.drain_traces()

    rows = await trace_rows(db)
    assert [r["kind"] for r in rows] == ["skill"]
    assert rows[0]["name"] == "demo-skill"
    assert rows[0]["detail"] == "触发词：演示"


async def test_no_skill_trace_when_not_triggered(db, skills_root, monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", True)
    write_skill(skills_root, "demo-skill")
    monkeypatch.setattr(runtime, "get_llm", lambda: FakeLLM([answer()]))

    [e async for e in runtime.run_agent(SESSION, "今天天气不错", db)]
    await tracing.drain_traces()

    assert await trace_rows(db) == []


# ---------- 执行：skill 的专用工具 ----------


async def test_skill_tool_runs_when_triggered(db, skills_root, monkeypatch):
    """模型调用专用工具 → 执行、结果进 tool 消息、记一条 tool trace。"""
    monkeypatch.setattr(settings, "tracing_enabled", True)
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    llm = FakeLLM([call_round("echo_tool", {"text": "你好"}), answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [e async for e in runtime.run_agent(SESSION, "来个演示", db)]
    await tracing.drain_traces()

    assert [e.type for e in events] == ["tool_start", "tool_end", "text_delta", "done"]
    assert events[1].data["summary"] == "echo_tool(演示)"
    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert tool_msg.content == "看到：你好"
    # 两条 trace 的写入顺序不保证（各自 fire-and-forget），只断言集合
    assert sorted((r["kind"], r["name"]) for r in await trace_rows(db)) == [
        ("skill", "demo-skill"),
        ("tool", "echo_tool"),
    ]


async def test_skill_tool_receives_db_path(db, skills_root, monkeypatch):
    write_skill(
        skills_root,
        "demo-skill",
        tools_py=(
            "from app.llm.types import ToolDef\n"
            'TOOLS=[ToolDef(name="show_db",description="x")]\n'
            "async def show_db(args, db_path):\n    return str(db_path), 'show_db'\n"
        ),
    )
    llm = FakeLLM([call_round("show_db"), answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert tool_msg.content == str(db)


async def test_skill_tool_unavailable_without_trigger(db, skills_root, monkeypatch):
    """同一轮没触发 skill：专用工具不在工具集里，调用被当成未知工具。"""
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    llm = FakeLLM([call_round("echo_tool", {"text": "x"}), [final()]])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [e async for e in runtime.run_agent(SESSION, "今天天气不错", db)]

    assert "未知工具" in events[1].data["summary"]
    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert tool_msg.content == "未知工具：echo_tool"


async def test_skill_tool_is_not_available_next_round(db, skills_root, monkeypatch):
    """下一轮换了提问（不触发）：工具集回到只有检索工具。"""
    write_skill(skills_root, "demo-skill", tools_py=TOOLS_PY)
    llm = FakeLLM([answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    [e async for e in runtime.run_agent(SESSION, "来个演示", db)]
    [e async for e in runtime.run_agent(SESSION, "今天天气不错", db)]

    assert [t.name for t in llm.tools[0]] == ["search_knowledge", "echo_tool"]
    assert [t.name for t in llm.tools[1]] == ["search_knowledge"]


async def test_skill_tool_exception_degrades_without_breaking_turn(
    db, skills_root, monkeypatch
):
    write_skill(
        skills_root,
        "demo-skill",
        tools_py=(
            "from app.llm.types import ToolDef\n"
            'TOOLS=[ToolDef(name="boom_tool",description="x")]\n'
            "async def boom_tool(args, db_path):\n    raise RuntimeError('炸了')\n"
        ),
    )
    llm = FakeLLM([call_round("boom_tool"), answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    assert [e.type for e in events] == ["tool_start", "tool_end", "text_delta", "done"]
    assert "失败" in events[1].data["summary"] and "炸了" in events[1].data["summary"]
    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert "炸了" in tool_msg.content  # 对话继续，不中断整轮
    assert [m["role"] for m in await messages_of(SESSION, db)][:1] == ["user"]


async def test_skill_tool_bad_return_shape_is_wrapped(db, skills_root, monkeypatch):
    """工具返回单值（不是 (结果, 摘要)）：规整成可读结果，不炸成本轮错误。"""
    write_skill(
        skills_root,
        "demo-skill",
        tools_py=(
            "from app.llm.types import ToolDef\n"
            'TOOLS=[ToolDef(name="odd",description="x")]\n'
            "async def odd(args, db_path):\n    return '只有结果'\n"
        ),
    )
    llm = FakeLLM([call_round("odd"), answer()])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    events = [e async for e in runtime.run_agent(SESSION, "来个演示", db)]

    assert events[-1].type == "done"
    tool_msg = [m for m in llm.calls[1] if m.role == "tool"][0]
    assert tool_msg.content == "只有结果"


async def test_execute_tool_without_skill_fns_rejects_skill_tool(db):
    """execute_tool 的默认口径不变：不传 skill_tools 时专用工具一律未知。"""
    result, label = await runtime.execute_tool(
        ToolCall(id="c1", name="echo_tool", arguments={}), db
    )

    assert result == "未知工具：echo_tool" and label == "未知工具 echo_tool"


async def test_execute_tool_accepts_skill_fns(db):
    """skill_tools 显式传入时按约定执行（接口对调用方开放）。"""

    async def echo_tool(args, db_path):
        return f"db={db_path}", "echo_tool(手写)"

    result, label = await runtime.execute_tool(
        ToolCall(id="c1", name="echo_tool", arguments={}), db, {"echo_tool": echo_tool}
    )

    assert result == f"db={db}" and label == "echo_tool(手写)"


# ---------- 端到端（HTTP 层） ----------


async def test_chat_end_to_end_triggers_resume_writing(db, monkeypatch, tmp_path):
    """brief 的手动验收路径：对话里说「帮我优化简历」→ 触发 resume-writing。

    走真实 HTTP 链路（含 SSE 序列化），并用**内置** skills/ 目录——这条用例验的是
    「用户按 README 跑起来就能看到效果」，而不是可控的测试 fixture。
    """
    import httpx

    from app.main import app

    monkeypatch.setattr(settings, "skills_dir", "skills")
    monkeypatch.setattr(settings, "tracing_enabled", True)
    llm = FakeLLM([answer("我来分析你的简历。")])
    monkeypatch.setattr(runtime, "get_llm", lambda: llm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat", json={"session_id": SESSION, "message": "帮我优化简历"}
        )
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
    await runtime.drain_memory_writes()
    await tracing.drain_traces()

    assert [e["type"] for e in events] == ["text_delta", "done"]
    sent = llm.calls[0]
    # 注入的正文在记忆之后、历史之前
    skill_msg = next(m for m in sent if m.content.startswith("[已启用技能"))
    assert "resume-writing" in skill_msg.content and "STAR" in skill_msg.content
    assert "analyze_resume" in skill_msg.content  # 专用工具已注册并写在正文里
    assert [t.name for t in llm.tools[0]] == [
        "search_knowledge",
        "analyze_resume",
        "jd_keyword_gap",
    ]
    rows = await trace_rows(db)
    assert [(r["kind"], r["name"]) for r in rows] == [("skill", "resume-writing")]


# ---------- 内置 skill ----------


def test_builtin_skills_load():
    """项目自带的 skills/ 目录要能被加载：有元数据、有触发词、有正文。

    不钉数量（后续加 skill 不该让这里红），只钉结构性要求——写坏一个 SKILL.md
    会让它静默消失，这条用例就是防线。
    """
    metas = load_skills("skills")

    assert set(metas) >= {
        "resume-writing",
        "interview-prep",
        "note-organizer",
        "learning-planner",
        "project-ideas",
    }
    for name, meta in metas.items():
        assert meta.name == name
        assert meta.description, f"{name} 缺少 description"
        assert meta.triggers, f"{name} 没有触发词"
        skill = get_skill(name, metas)
        assert skill is not None and len(skill.content) > 200, f"{name} 正文太短"


def test_builtin_resume_writing_tools_load():
    """resume-writing 的 tools.py 要能加载出两个工具（手写工具最易被改坏）。"""
    metas = load_skills("skills")

    skill = get_skill("resume-writing", metas)

    assert skill is not None
    assert {t.name for t in skill.tools} == {"analyze_resume", "jd_keyword_gap"}
    assert set(skill.tool_fns) == {"analyze_resume", "jd_keyword_gap"}


async def test_builtin_analyze_resume_flags_weak_bullets():
    """内置工具真的能跑出字面体检结论（不是只 import 得进来）。"""
    metas = load_skills("skills")
    fn = get_skill("resume-writing", metas).tool_fns["analyze_resume"]

    result, label = await fn(
        {
            "resume_text": (
                "负责后端接口开发\n"
                "实现登录接口，将 P99 从 800ms 降到 120ms\n"
                "参与分布式改造项目\n"
            )
        },
        None,
    )

    assert "负责后端接口开发" in result
    assert "无任何量化：2 条" in result
    assert "弱动词/职责式开头：2 条" in result  # 「负责…」与「参与…」
    assert "analyze_resume" in label


async def test_builtin_analyze_resume_without_text_asks_for_input():
    metas = load_skills("skills")
    fn = get_skill("resume-writing", metas).tool_fns["analyze_resume"]

    result, label = await fn({}, None)

    assert "没有拿到简历文本" in result
    assert "缺少 resume_text" in label


async def test_builtin_jd_keyword_gap_reports_missing():
    metas = load_skills("skills")
    fn = get_skill("resume-writing", metas).tool_fns["jd_keyword_gap"]

    result, label = await fn(
        {"jd": "熟悉 Python 与 Kubernetes，有高并发经验", "resume_text": "用 Python 写过服务"},
        None,
    )

    missing_line = next(l for l in result.splitlines() if l.startswith("- 未覆盖"))
    assert "Kubernetes" in missing_line and "高并发" in missing_line
    assert "Python" not in missing_line  # 已覆盖的词不进缺口
    assert label == "jd_keyword_gap：1/3 覆盖，缺 2 个词"
