"""T7 开机自启（Windows Run 键）+ 启动期懒加载 + 桌面单实例。

自启是非 .env 的用户数据（同 persona）：`POST /api/settings` 特判它并调 `app/autostart.py`，
`GET` 读注册表当前状态。**注册表用例一律走假 winreg**：测试跑在开发机上，真写 HKCU 等于
给开发者装一个开机自启项，而且会污染跑测试的这台机器。

懒加载的自动化门槛只有一条（brief 明确不设毫秒阈值）：`import app.main` 之后，明确列出的
重型模块（pymupdf / app.ingest.pipeline / app.ingest.loaders / httpx）不得出现在 sys.modules。
观测到的 importtime 只写进报告，不作为通过条件。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app import autostart, desktop, scheduler
from app.config import EDITABLE_FIELDS, settings
from app.db import init_db
from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
DIM = 8

# 启动期不该付出的导入代价。pymupdf 是 app.ingest.loaders 的顶层依赖（~77ms），
# httpx 同源（~38ms）；两者都由 app.ingest.pipeline 带进来。
HEAVY_MODULES = ("pymupdf", "app.ingest.pipeline", "app.ingest.loaders", "httpx")


# ---------- 夹具 ----------


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    # 设置面板会写 .env（源码环境绝不动仓库里真实的那份）
    monkeypatch.setattr("app.main.env_path", lambda: tmp_path / "settings.env")
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def pin_provider(monkeypatch) -> None:
    """钉住供应商配置：/api/settings 会校验「保存后的完整配置」，不钉的话这些用例
    会先撞上 422（同 tests/test_persona.py 的做法）。"""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "deepseek-chat")


class _FakeKey:
    def __init__(self, fake: "FakeWinreg", path: str):
        self._fake = fake
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fake.calls.append(("CloseKey", self.path))
        return False


class FakeWinreg:
    """winreg 的最小替身：只实现 app/autostart.py 用到的那几个函数。

    store 是 {键路径: {值名: 值}}，calls 记录调用序列（用来断言「只动我们自己的值」），
    fail 注入注册表访问失败。
    """

    HKEY_CURRENT_USER = "HKCU"
    KEY_READ = 1
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self, store=None, fail: Exception | None = None):
        self.store: dict[str, dict[str, str]] = dict(store or {})
        self.fail = fail
        self.calls: list[tuple] = []

    def _record(self, *entry):
        self.calls.append(entry)
        if self.fail is not None:
            raise self.fail

    def CreateKeyEx(self, hive, path, reserved, access):
        self._record("CreateKeyEx", path)
        self.store.setdefault(path, {})
        return _FakeKey(self, path)

    def OpenKey(self, hive, path, reserved, access):
        self._record("OpenKey", path)
        if path not in self.store:
            raise FileNotFoundError(f"no such key: {path}")
        return _FakeKey(self, path)

    def SetValueEx(self, key, name, reserved, type_, value):
        self._record("SetValueEx", key.path, name)
        self.store[key.path][name] = value

    def QueryValueEx(self, key, name):
        self._record("QueryValueEx", key.path, name)
        try:
            return self.store[key.path][name], self.REG_SZ
        except KeyError:
            raise FileNotFoundError(name) from None

    def DeleteValue(self, key, name):
        self._record("DeleteValue", key.path, name)
        try:
            del self.store[key.path][name]
        except KeyError:
            raise FileNotFoundError(name) from None


@pytest.fixture
def registry(monkeypatch):
    """把 autostart 的 winreg 换成替身，返回替身供断言。"""
    fake = FakeWinreg()
    monkeypatch.setattr(autostart, "_winreg", lambda: fake)
    return fake


# ---------- 支持判定 ----------


def test_is_supported_is_false_in_the_source_tree():
    """接口钉死：源码环境不支持（is_supported 要求 sys.frozen）。"""
    assert not getattr(sys, "frozen", False)
    assert autostart.is_supported is False


# ---------- 注册表：enable / disable / is_enabled ----------


def test_enable_writes_a_fully_quoted_command_and_is_idempotent(registry, monkeypatch):
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\第二大脑\SecondBrainAgent.exe")

    autostart.enable()
    autostart.enable()  # 重复 enable 幂等

    assert registry.store[autostart.RUN_KEY] == {
        autostart.VALUE_NAME: subprocess.list2cmdline(
            [r"C:\Program Files\第二大脑\SecondBrainAgent.exe"]
        )
    }
    written = registry.store[autostart.RUN_KEY][autostart.VALUE_NAME]
    # 带空格的路径必须整体引用，否则 Run 键会把 "C:\Program" 当成可执行文件
    assert written.startswith('"') and written.endswith('"')
    assert autostart.RUN_KEY == r"Software\Microsoft\Windows\CurrentVersion\Run"


def test_enable_does_not_touch_other_run_entries(registry):
    registry.store[autostart.RUN_KEY] = {"其他软件": r'"C:\Other\other.exe"'}

    autostart.enable()

    assert registry.store[autostart.RUN_KEY]["其他软件"] == r'"C:\Other\other.exe"'


def test_is_enabled_reads_the_registry(registry):
    registry.store[autostart.RUN_KEY] = {autostart.VALUE_NAME: r'"C:\a.exe"'}
    assert autostart.is_enabled() is True

    registry.store[autostart.RUN_KEY] = {}
    assert autostart.is_enabled() is False  # 值不存在 = 没启用

    registry.store.clear()
    assert autostart.is_enabled() is False  # 连键都没有也只是 False，不是错误


def test_disable_removes_only_our_value_and_succeeds_when_absent(registry):
    registry.store[autostart.RUN_KEY] = {
        autostart.VALUE_NAME: r'"C:\a.exe"',
        "其他软件": r'"C:\Other\other.exe"',
    }

    autostart.disable()
    assert registry.store[autostart.RUN_KEY] == {"其他软件": r'"C:\Other\other.exe"'}

    autostart.disable()  # 值不存在也成功
    assert autostart.is_enabled() is False

    registry.store.clear()
    autostart.disable()  # 连 Run 键都没有也成功


@pytest.mark.parametrize("call", ["enable", "disable", "is_enabled"])
def test_registry_failure_raises_a_readable_error(registry, call):
    """注册表访问失败要变成一句人能看懂的话，HTTP 层据此回 500。"""
    registry.fail = PermissionError("拒绝访问")

    with pytest.raises(autostart.AutostartError) as exc:
        getattr(autostart, call)()

    assert "注册表" in str(exc.value)
    assert "拒绝访问" in str(exc.value)


# ---------- 设置 API：GET 状态 ----------


async def test_get_settings_reports_autostart_state_and_support(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", True)
    monkeypatch.setattr(autostart, "is_enabled", lambda: True)

    data = (await client.get("/api/settings")).json()

    assert data["autostart_supported"] is True
    assert data["autostart"] is True


async def test_get_settings_skips_the_registry_when_unsupported(client, monkeypatch):
    """源码环境不读注册表：开关在界面上是禁用的，读出来的值也没有意义。"""
    monkeypatch.setattr(autostart, "is_supported", False)

    def boom():
        raise AssertionError("源码环境不该访问注册表")

    monkeypatch.setattr(autostart, "is_enabled", boom)

    data = (await client.get("/api/settings")).json()

    assert data["autostart_supported"] is False
    assert data["autostart"] is False


async def test_get_settings_survives_an_unreadable_registry(client, monkeypatch):
    """注册表读不到不该把整个设置面板变成 500：自启是可选功能，面板还要能改模型配置。"""
    monkeypatch.setattr(autostart, "is_supported", True)

    def boom():
        raise autostart.AutostartError("注册表访问失败：拒绝访问")

    monkeypatch.setattr(autostart, "is_enabled", boom)

    resp = await client.get("/api/settings")

    assert resp.status_code == 200
    assert resp.json()["autostart"] is False


# ---------- 设置 API：POST 开关 ----------


async def test_post_settings_rejects_autostart_in_the_source_tree(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", False)
    called: list[str] = []
    monkeypatch.setattr(autostart, "enable", lambda: called.append("enable"))
    pin_provider(monkeypatch)

    resp = await client.post("/api/settings", json={"autostart": True})

    assert resp.status_code == 400
    assert "打包" in resp.json()["detail"]
    assert called == []  # 被拒的请求不许留下副作用


async def test_post_settings_accepts_autostart_false(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", False)
    monkeypatch.setattr(autostart, "enable", lambda: pytest.fail("不该调用 enable"))
    pin_provider(monkeypatch)

    resp = await client.post("/api/settings", json={"autostart": False})

    assert resp.status_code == 200, resp.text
    assert "autostart" in resp.json()["updated"]


async def test_post_settings_toggles_the_registry_when_supported(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", True)
    called: list[str] = []
    monkeypatch.setattr(autostart, "enable", lambda: called.append("enable"))
    monkeypatch.setattr(autostart, "disable", lambda: called.append("disable"))
    pin_provider(monkeypatch)

    assert (await client.post("/api/settings", json={"autostart": True})).status_code == 200
    assert (await client.post("/api/settings", json={"autostart": False})).status_code == 200

    assert called == ["enable", "disable"]


async def test_post_settings_surfaces_a_registry_failure_as_500(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", True)

    def boom():
        raise autostart.AutostartError("写注册表失败：拒绝访问")

    monkeypatch.setattr(autostart, "enable", boom)
    pin_provider(monkeypatch)

    resp = await client.post("/api/settings", json={"autostart": True})

    assert resp.status_code == 500
    assert "注册表" in resp.json()["detail"]


async def test_post_settings_without_autostart_leaves_it_alone(client, monkeypatch):
    monkeypatch.setattr(autostart, "is_supported", True)
    monkeypatch.setattr(autostart, "enable", lambda: pytest.fail("没提就别动"))
    monkeypatch.setattr(autostart, "disable", lambda: pytest.fail("没提就别动"))
    pin_provider(monkeypatch)

    resp = await client.post("/api/settings", json={"openai_model": "deepseek-chat"})

    assert resp.status_code == 200, resp.text
    assert "autostart" not in resp.json()["updated"]


async def test_autostart_never_lands_in_the_env_file(client, tmp_path, monkeypatch):
    """自启是注册表状态，不是配置项：进 .env 的话「关掉」就会变成一次静默回退。"""
    monkeypatch.setattr(autostart, "is_supported", False)
    pin_provider(monkeypatch)

    resp = await client.post(
        "/api/settings", json={"autostart": False, "openai_model": "deepseek-chat"}
    )

    assert resp.status_code == 200, resp.text
    text = (tmp_path / "settings.env").read_text(encoding="utf-8")
    assert "AUTOSTART" not in text.upper()
    assert "autostart" not in EDITABLE_FIELDS


# ---------- 懒加载：唯一的自动化门槛 ----------


def test_startup_does_not_load_the_pdf_and_http_stack():
    """`import app.main` 之后重型模块不得出现在 sys.modules。

    这是 T7 唯一的自动化门槛（brief 明确不设毫秒阈值）：pymupdf 与 httpx 是
    app.ingest.pipeline 的导入副作用，只有真正导入文档时才该付出这个代价。
    """
    code = (
        "import sys, app.main;"
        f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )

    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


async def test_ingest_endpoints_still_work_with_the_deferred_import(
    client, db, tmp_path, monkeypatch
):
    """延迟导入不能把功能改坏：/api/ingest 与 /api/documents 仍走真实管线。"""
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def fake_embed(texts):
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    monkeypatch.setattr("app.ingest.pipeline.embed_texts", fake_embed)
    note = tmp_path / "uploads" / "笔记.md"
    note.parent.mkdir()
    note.write_text("# 延迟导入\n内容", encoding="utf-8")

    resp = await client.post("/api/ingest", json={"source": "uploads/笔记.md"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"chunks": 1}
    docs = (await client.get("/api/documents")).json()
    assert [d["title"] for d in docs] == ["延迟导入"]
    assert (await client.delete(f"/api/documents/{docs[0]['id']}")).json() == {
        "deleted": docs[0]["id"]
    }


# ---------- 打包联动（Task 2 / Task 5 交接） ----------


def test_default_persona_and_schema_resources_are_readable():
    """打包 smoke 的源码侧一半：默认人格与 schema 必须按仓库相对布局解析得到。

    tests/test_persona.py 已有一条同主题用例（T2 交付）；这里保留最小一条是因为
    T7 动了打包清单，资源可读性必须与打包脚本的清单一起被钉住。
    """
    from app.agent import runtime
    from app.resources import resource_path

    assert runtime.persona_default_path().is_file()
    assert runtime.persona_default_path().read_text(encoding="utf-8").strip()
    assert resource_path("app/schema.sql").is_file()


def test_tzdata_zone_database_is_available():
    """Windows 没有系统 tz 库，zoneinfo 只能靠 tzdata 包（build_desktop.bat 用
    --collect-data tzdata 带上它）。这里钉住「tzdata 已安装 + 非本机时区能解析」。"""
    assert importlib.util.find_spec("tzdata") is not None
    assert scheduler.resolve_timezone("America/New_York").key == "America/New_York"


def test_local_timezone_name_smoke():
    """打包 smoke：本机时区标识要能拿到且能被 zoneinfo 解析（tzlocal + tzdata 都在位）。

    与 tests/test_scheduler.py 的同名用例重复是刻意的——那条守调度语义（任务解释时区），
    这条守打包资源（缺 tzdata 时 resolve_timezone 会直接抛）。
    """
    name = scheduler.local_timezone_name()

    assert scheduler.resolve_timezone(name) is not None


def test_build_script_ships_every_runtime_resource():
    """打包脚本是 T7 的交付物之一，把资源清单钉在测试里，防止下次改脚本时静默丢资源。"""
    script = (REPO_ROOT / "build_desktop.bat").read_text(encoding="utf-8")

    for needle in (
        '--add-data "web;web"',
        '--add-data "app\\schema.sql;app"',
        '--add-data "app\\agent\\prompts\\persona_default.md;app\\agent\\prompts"',
        "--collect-data tzdata",
        "xcopy /E /I /Y /Q skills",
    ):
        assert needle in script, f"打包脚本缺少：{needle}"


# ---------- 桌面单实例（T5/T6 交接：调度正确性依赖单进程） ----------


def test_first_launch_takes_the_mutex(monkeypatch):
    monkeypatch.setattr(desktop, "_instance_handle", None)
    created: list[str] = []
    monkeypatch.setattr(desktop, "_create_mutex", lambda name: (created.append(name) or "H", False))

    assert desktop.acquire_single_instance() is True

    assert created == [desktop.INSTANCE_MUTEX_NAME]


def test_second_launch_is_rejected_and_closes_its_handle(monkeypatch):
    monkeypatch.setattr(desktop, "_instance_handle", None)
    closed: list[str] = []
    monkeypatch.setattr(desktop, "_create_mutex", lambda name: ("H", True))
    monkeypatch.setattr(desktop, "_close_mutex", closed.append)

    assert desktop.acquire_single_instance() is False

    assert closed == ["H"]  # 已存在的句柄也要关，否则泄漏一个内核对象


def test_acquire_is_idempotent_within_one_process(monkeypatch):
    """同一进程里重复调用不该把自己当成「另一个实例」。"""
    monkeypatch.setattr(desktop, "_instance_handle", object())
    monkeypatch.setattr(
        desktop, "_create_mutex", lambda name: pytest.fail("已持有互斥体，不该再建")
    )

    assert desktop.acquire_single_instance() is True


def test_non_windows_skips_the_mutex(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        desktop, "_create_mutex", lambda name: pytest.fail("非 Windows 不该碰内核对象")
    )

    assert desktop.acquire_single_instance() is True
