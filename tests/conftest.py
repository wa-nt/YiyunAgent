"""全套测试共用的夹具。

上下文治理（T8）让 run_agent 的行为依赖 CONTEXT_* 配置，而各用例断言的 prompt 内容
（历史条数、记忆是否注入）都是按 brief 的默认值写的。这里用 autouse 把默认值钉住，
测试结果不随 `.env` 或环境变量漂移：实测 CONTEXT_COMPACTION_ENABLED=false 会让压缩
用例变红、CONTEXT_MAX_TOKENS=50 会让 test_agent/test_memory 里断言历史与记忆的用例变红。

用例要改这些开关必须显式 monkeypatch，保证「关掉某策略」是它自己的意图而不是环境残留。

T9 的埋点开关走同一个夹具，但钉的是 **False**（与 app/config.py 的默认值相反）：
tests/test_llm.py 用真客户端 + 假传输调 chat/chat_stream，埋点开着就会 fire-and-forget
往默认库（data/app.db）写 trace——既污染开发库的成本数据，又留下没人 drain 的后台任务。
断言埋点的用例自己显式打开（见 tests/test_tracing.py）。

T11 的 skill 开关也钉 False，理由同 T9：run_agent 会按用户消息触发 skill，命中时给
prompt view 多加一条 system 消息（正文）与专用工具。skills_dir 默认是仓库里的
`skills/`，于是**整套测试都隐式耦合了内置 skill 的内容**——改一个触发词就可能打红
test_memory.py 里那些「断言注入了几条消息」的无关用例。钉死 False 后，需要真触发的
用例自己打开开关（见 tests/test_skills.py 的 skills_root 夹具）。
"""

import pytest

from app.config import settings

# T8 brief 的默认值，与 app/config.py 的 Settings 字段默认值保持一致
CONTEXT_DEFAULTS = {
    "context_compaction_enabled": True,
    "context_tool_clean_enabled": True,
    "context_token_budget_enabled": True,
    "context_max_tokens": 8000,
    "context_compaction_threshold": 10,
}

# T9 埋点总开关：测试里默认关，理由见模块 docstring
TRACING_DEFAULT = False

# T11 skill 总开关：测试里默认关，理由见模块 docstring
SKILLS_DEFAULT = False


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    for name, value in CONTEXT_DEFAULTS.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "tracing_enabled", TRACING_DEFAULT)
    monkeypatch.setattr(settings, "skills_enabled", SKILLS_DEFAULT)
