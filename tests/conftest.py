"""全套测试共用的夹具。

上下文治理（T8）让 run_agent 的行为依赖 CONTEXT_* 配置，而各用例断言的 prompt 内容
（历史条数、记忆是否注入）都是按 brief 的默认值写的。这里用 autouse 把默认值钉住，
测试结果不随 `.env` 或环境变量漂移：实测 CONTEXT_COMPACTION_ENABLED=false 会让压缩
用例变红、CONTEXT_MAX_TOKENS=50 会让 test_agent/test_memory 里断言历史与记忆的用例变红。

用例要改这些开关必须显式 monkeypatch，保证「关掉某策略」是它自己的意图而不是环境残留。
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


@pytest.fixture(autouse=True)
def pinned_context_settings(monkeypatch):
    for name, value in CONTEXT_DEFAULTS.items():
        monkeypatch.setattr(settings, name, value)
