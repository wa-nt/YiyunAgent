"""设置面板（/api/settings）：脱敏回显、.env 就地更新、保存即热生效。"""

import pytest
from fastapi.testclient import TestClient

from app.config import mask_secret, settings, update_env_file
from app.main import app


def test_mask_secret():
    assert mask_secret(None) == ""
    assert mask_secret("") == ""
    assert mask_secret("sk-1234567890abcdef") == "…cdef"
    assert mask_secret("abc") == "…"  # 太短就一位不露


def test_update_env_file_preserves_unrelated_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# 注释行\nOPENAI_API_KEY=old-key\nDB_PATH=data/app.db\n",
        encoding="utf-8",
    )
    update_env_file(path, {"openai_api_key": "new-key", "openai_model": "deepseek-chat"})
    text = path.read_text(encoding="utf-8")
    assert "# 注释行" in text
    assert "DB_PATH=data/app.db" in text
    assert "OPENAI_API_KEY=new-key" in text  # 已有键替换原行
    assert "OPENAI_MODEL=deepseek-chat" in text  # 新键追加末尾
    assert "old-key" not in text


@pytest.fixture
def client(tmp_path, monkeypatch):
    # .env 写到临时目录，绝不动仓库里真实的 .env
    monkeypatch.setattr("app.main.env_path", lambda: tmp_path / ".env")
    return TestClient(app)


def test_get_settings_masks_secrets(client, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-real-secret-1234")
    data = client.get("/api/settings").json()
    assert data["openai_api_key"] == "…1234"
    assert "sk-real-secret" not in str(data)


def test_update_settings_rejects_incomplete_provider(client, monkeypatch):
    # 切到没配齐密钥/模型的供应商 = 写出来一用就 401，必须在边界拒绝
    for name in ("openai_api_key", "openai_model"):
        monkeypatch.setattr(settings, name, None)
    r = client.post("/api/settings", json={"llm_provider": "openai_compat"})
    assert r.status_code == 422


def test_update_settings_applies_immediately_and_persists(client, tmp_path, monkeypatch):
    # monkeypatch 钉住会被接口改写的字段，防本用例污染后续用例的全局 settings
    monkeypatch.setattr(settings, "llm_provider", "openai_compat")
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_model", None)
    r = client.post(
        "/api/settings",
        json={
            "llm_provider": "openai_compat",
            "openai_api_key": "sk-test-9999",
            "openai_model": "deepseek-chat",
        },
    )
    assert r.status_code == 200, r.text
    assert settings.openai_model == "deepseek-chat"  # 热生效，无需重启
    assert "OPENAI_API_KEY=sk-test-9999" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_update_settings_blank_secret_keeps_current(client, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-keep-me-0000")
    monkeypatch.setattr(settings, "openai_model", "deepseek-chat")
    r = client.post("/api/settings", json={"openai_api_key": "", "openai_model": "deepseek-chat"})
    assert r.status_code == 200, r.text
    assert settings.openai_api_key == "sk-keep-me-0000"  # 留空没把密钥清掉


# ---- 拉取模型（POST /api/models） ----
# 端点必须转发到后端：浏览器直连供应商会跨域失败，密钥也只该存在服务端


class _FakeModel:
    def __init__(self, mid):
        self.id = mid


class _FakePage:
    def __init__(self, ids):
        self.data = [_FakeModel(i) for i in ids]


class _FakeModels:
    """对应真实客户端的 `client.models` 资源对象（同步属性 + 可 await 的 list()）。"""

    def __init__(self, ids, error):
        self.ids = ids
        self.error = error

    async def list(self):
        if self.error:
            raise self.error
        return _FakePage(self.ids)


class _FakeClient:
    """替身 AsyncOpenAI：记下构造参数，models.list() 返回预置清单或抛预置异常。"""

    def __init__(self, ids=None, error=None, **kwargs):
        self.kwargs = kwargs
        self.models = _FakeModels(ids or [], error)
        self.closed = False

    async def close(self):
        self.closed = True


def _install_fake(monkeypatch, ids=None, error=None):
    created = []

    def factory(**kwargs):
        c = _FakeClient(ids=ids, error=error, **kwargs)
        created.append(c)
        return c

    monkeypatch.setattr("app.main.AsyncOpenAI", factory)
    return created


def test_models_uses_form_values_and_dedupes_sorted(client, monkeypatch):
    created = _install_fake(monkeypatch, ids=["gpt-4o-mini", "bge-m3", "gpt-4o-mini"])
    r = client.post(
        "/api/models",
        json={"kind": "llm", "base_url": "https://relay.example/v1", "api_key": "sk-typed"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["models"] == ["bge-m3", "gpt-4o-mini"]  # 去重且有序
    assert created[0].kwargs["base_url"] == "https://relay.example/v1"
    assert created[0].kwargs["api_key"] == "sk-typed"
    assert created[0].closed  # 用完关掉，不泄漏连接


def test_models_falls_back_to_saved_config(client, monkeypatch):
    # 密钥框平时是空的，留空时应回落到已保存的配置而不是报「缺少 key」
    monkeypatch.setattr(settings, "openai_api_key", "sk-saved")
    monkeypatch.setattr(settings, "openai_base_url", "https://saved.example/v1")
    created = _install_fake(monkeypatch, ids=["m1"])
    r = client.post("/api/models", json={"kind": "llm"})
    assert r.status_code == 200, r.text
    assert created[0].kwargs["api_key"] == "sk-saved"
    assert created[0].kwargs["base_url"] == "https://saved.example/v1"


def test_models_embed_kind_reads_embed_config(client, monkeypatch):
    monkeypatch.setattr(settings, "embed_api_key", "sk-embed")
    monkeypatch.setattr(settings, "embed_base_url", "https://embed.example/v1")
    created = _install_fake(monkeypatch, ids=["BAAI/bge-m3"])
    r = client.post("/api/models", json={"kind": "embed"})
    assert r.status_code == 200, r.text
    assert created[0].kwargs["api_key"] == "sk-embed"
    assert created[0].kwargs["base_url"] == "https://embed.example/v1"


def test_models_bad_key_is_401(client, monkeypatch):
    import httpx
    from openai import AuthenticationError

    resp = httpx.Response(401, request=httpx.Request("GET", "https://x/v1/models"))
    _install_fake(monkeypatch, error=AuthenticationError("boom", response=resp, body=None))
    r = client.post("/api/models", json={"kind": "llm", "api_key": "sk-bad"})
    assert r.status_code == 401
    assert "API Key" in r.json()["detail"]


def test_models_unreachable_is_502(client, monkeypatch):
    import httpx
    from openai import APIConnectionError

    _install_fake(monkeypatch, error=APIConnectionError(request=httpx.Request("GET", "https://x")))
    r = client.post("/api/models", json={"kind": "llm", "api_key": "sk-x"})
    assert r.status_code == 502


def test_models_missing_key_is_422(client, monkeypatch):
    # 没配过 key 也没填：构造函数抛 OpenAIError，要落成 422 而不是 500
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_base_url", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/api/models", json={"kind": "llm"})
    assert r.status_code == 422
    assert "API Key" in r.json()["detail"]


def test_models_anthropic_not_supported(client, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    r = client.post("/api/models", json={"kind": "llm"})
    assert r.status_code == 400
