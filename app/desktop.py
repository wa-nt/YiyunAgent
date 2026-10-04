"""桌面入口：原生窗口 + 系统托盘，复用现有 FastAPI 应用（叠加式，不动任何现有模块）。

形态：随机空闲端口后台起 uvicorn → pywebview 原生窗口指向它。点 X 是最小化到
托盘（微信/QQ 惯例），真退出走托盘菜单；退出时先置 uvicorn.should_exit，让
app.main 的 lifespan 收尾（记忆/trace drain）跑完，再销毁窗口。

配置口径：.env / data/ / skills/ 都按 CWD 相对解析（见 config.py），所以
PyInstaller 打包后先把 CWD 锚到 exe 所在目录——从源码跑时则不碰 CWD（按文档
约定从项目根启动）。

托盘通知（T6）
    通知渠道本期选的是托盘气泡，所以这里另起一个 daemon 轮询线程：用**同步** sqlite3
    只读连接查 notifications（T5 写的表），每条只弹一次。刻意不建 ASGI client、不调
    FastAPI route、不跨事件循环复用 aiosqlite 连接——轮询线程跑在 uvicorn 之外，
    跨循环用异步连接是未定义行为，而这条链路只是旁路，不值得为它引一套客户端。
    只读连接（mode=ro）保证「库还不存在」时报错而不是就地建一个空库；弹出失败只记日志，
    不影响窗口和 Web 端的任务管理。pystray 不可用时整个托盘功能降级为纯窗口模式——这时
    点 X 是真退出（藏进一个不存在的托盘 = 窗口和进程一起失联）。

用法：
    python -m app.desktop              启动桌面端
    python -m app.desktop --make-icon  生成 web/app.ico（build_desktop.bat 打包前用）
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 托盘通知轮询：周期、每次最多读多少条、读连接的 busy timeout。
# 读 50 条足够覆盖一次轮询间隔内新增的记录（一个 tick 最多触发 MAX_FIRE_CONCURRENCY 个任务），
# 又不会在库很大时白读一堆历史。
NOTIFY_POLL_INTERVAL = 5.0
NOTIFY_LIMIT = 50
NOTIFY_BUSY_TIMEOUT_MS = 5000


def _free_port() -> int:
    """让系统分配一个空闲端口。不固定 8765：与开发模式并存时不能互相抢端口。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _readonly_uri(path: str | Path) -> str:
    """只读连接用的 file: URI。

    `mode=ro` 是必须的：轮询线程可能先于 lifespan 的 init_db 跑起来，普通连接会在
    数据目录里就地建一个空库，把真正的库顶掉。用 as_uri() 而不是手拼字符串——路径里的
    空格、中文、`?` 都要百分号转义，否则会被当成 URI 语法。
    """
    return Path(path).resolve().as_uri() + "?mode=ro"


def read_notifications_sync(limit: int = NOTIFY_LIMIT, db_path: str | Path | None = None) -> list[dict]:
    """同步只读查询最近 N 条通知（新的在前，与 GET /api/notifications 同一形状）。

    库不存在 / 表还没建 / 库损坏一律返回空并记日志：托盘通知是旁路，读不到不该反过来
    把桌面端拖死。查询失败与「确实没有记录」对调用方是同一件事——本轮什么都不弹。
    """
    if db_path is None:
        from app.config import settings

        db_path = settings.db_path
    try:
        conn = sqlite3.connect(_readonly_uri(db_path), uri=True)
    except sqlite3.Error as exc:
        logger.warning("通知库不可读（本轮跳过）：%s: %s", type(exc).__name__, exc)
        return []
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {NOTIFY_BUSY_TIMEOUT_MS}")
        rows = conn.execute(
            "SELECT id, task_id, session_id, kind, title, body, created_at FROM notifications "
            "ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("读取通知失败（本轮跳过）：%s: %s", type(exc).__name__, exc)
        return []
    finally:
        conn.close()
    return [dict(row) for row in rows]


class TrayNotifier:
    """托盘通知轮询：只读 SQLite → 逐条 icon.notify → 记住已通知的最大 id。

    只弹「应用启动后新增」的记录：prime() 先记下启动那一刻的最大 id（id 是自增主键，
    单调递增，字符串/数字比较都即时间序），之后严格按 id 去重，所以重启不会把历史通知
    再弹一遍。每条只**尝试**弹一次——弹失败也只记日志并推进游标：托盘后端对同一条重复
    弹比漏一条更烦人（Windows 的气泡会重绘），而漏掉的那条在 Web 端的任务结果里还在。

    停止由 threading.Event 控制（stop()），run() 里用 event.wait 睡，退出最多等一个
    周期，不阻塞窗口关闭。
    """

    def __init__(
        self,
        icon,
        db_path: str | Path | None = None,
        interval: float = NOTIFY_POLL_INTERVAL,
        limit: int = NOTIFY_LIMIT,
    ):
        self._icon = icon
        self._db_path = db_path
        self._interval = interval
        self._limit = limit
        self._last_id = 0
        self._stop = threading.Event()

    @property
    def last_id(self) -> int:
        return self._last_id

    def prime(self) -> int:
        """记下启动时已有的最大通知 id，返回它。必须在库建好之后调用（见 main）。"""
        rows = read_notifications_sync(self._limit, self._db_path)
        self._last_id = max((int(row["id"]) for row in rows), default=0)
        return self._last_id

    def poll_once(self) -> list[dict]:
        """读一轮并把新记录按 id 升序弹出来（返回弹出过的那些）。

        升序是为了让最新的那条留在气泡里；同一条不会弹第二次，因为游标在弹之前就推进了。
        """
        fresh = sorted(
            (
                row
                for row in read_notifications_sync(self._limit, self._db_path)
                if int(row["id"]) > self._last_id
            ),
            key=lambda row: int(row["id"]),
        )
        for row in fresh:
            self._last_id = max(self._last_id, int(row["id"]))
            try:
                self._icon.notify(row["body"], row["title"])
            except Exception as exc:
                logger.warning("托盘通知弹出失败（跳过该条）：%s: %s", type(exc).__name__, exc)
        return fresh

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # 读库/弹窗都不该让轮询线程死掉
                logger.exception("托盘通知轮询失败，下一轮继续")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()


def _start_tray_notifications(icon) -> TrayNotifier:
    """起托盘通知轮询线程。

    调用点在 _start_server 之后：uvicorn 的 server.started 是 lifespan 启动完成后才置位的，
    所以此刻 scheduled_tasks / notifications 表已经建好，prime() 读到的最大 id 是可信的
    （prime 读早了会把启动前的历史通知全弹一遍）。
    """
    from app.config import settings

    notifier = TrayNotifier(icon, db_path=settings.db_path)
    notifier.prime()
    threading.Thread(target=notifier.run, daemon=True, name="tray-notifications").start()
    return notifier


def _start_server(port: int, timeout: float = 30.0):
    """后台线程跑 uvicorn，返回 (server, thread)。等 server.started 再返回，
    否则窗口可能赶在服务监听前加载，首屏就是连接拒绝。

    必须检测失败：uvicorn 启动失败（如 schema.sql 缺失）时 started 永远为 False，
    只等 started 会无限空转——进程活着、不监听、无输出，最难排查的一种「打不开」。
    """
    import uvicorn

    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("内置服务启动失败：uvicorn 已退出（多半是数据文件缺失或端口被占）")
        if time.monotonic() > deadline:
            raise RuntimeError(f"内置服务启动超时（{timeout:.0f} 秒内未就绪）")
        time.sleep(0.05)
    return server, thread


def _fatal(message: str) -> None:
    """启动失败要看得见：--windowed 打包没有控制台，弹一个系统对话框，
    否则用户只能看到「双击没反应」。ctypes 是标准库，不引依赖。"""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "第二大脑 Agent 启动失败", 0x10)
    except Exception:
        pass
    raise SystemExit(1)


def make_icon(size: int = 256):
    """珊瑚橙圆角方块 + 奶油色四芒星（Claude 配色的极简应用图标）。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 5, fill="#cc785c")
    c, r, w = size / 2, size * 0.30, size * 0.055
    d.polygon(
        [
            (c, c - r), (c + w, c - w), (c + r, c), (c + w, c + w),
            (c, c + r), (c - w, c + w), (c - r, c), (c - w, c - w),
        ],
        fill="#faf9f5",
    )
    return img


def main() -> None:
    if "--make-icon" in sys.argv:
        make_icon().save(
            "web/app.ico",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        return

    if getattr(sys, "frozen", False):
        # PyInstaller 包：把 .env / data/ / skills/ 的相对路径口径锚到 exe 旁边，
        # 这样无论从快捷方式、任务栏还是资源管理器启动行为都一致
        os.chdir(os.path.dirname(sys.executable))

    import webview

    port = _free_port()
    try:
        server, server_thread = _start_server(port)
    except Exception as exc:
        # windowed 打包没有控制台，异常不弹框就是「双击没反应」
        _fatal(f"{exc}\n\n请确认 .env 与 data 目录与程序在同一目录。")

    window = webview.create_window(
        "第二大脑 Agent",
        f"http://127.0.0.1:{port}/",
        width=1280,
        height=800,
        min_size=(900, 600),
    )

    quitting = False

    def on_closing():
        # 点 X = 藏到托盘，不是退出；返回 False 取消 pywebview 的默认关闭。
        # 没有托盘时必须真关：藏进一个不存在的托盘 = 窗口和进程一起失联，
        # 用户只能去任务管理器杀进程（降级路径的另一半，见下面的 try/except）
        if quitting or notifier is None:
            return True
        window.hide()
        return False

    window.events.closing += on_closing

    def show_window(icon=None, item=None):
        window.show()
        window.restore()

    # 托盘与托盘通知：pystray 起不来（缺后端/无桌面会话）时只记日志，窗口和 Web 端的
    # 任务管理照常可用——通知渠道是托盘，但它不该成为「打不开」的原因。
    # icon / notifier 都留在本函数（主线程）的帧里：图标实例的生命周期由主线程持有。
    icon = notifier = None
    try:
        import pystray
        from pystray import MenuItem as Item

        def quit_app(icon=None, item=None):
            nonlocal quitting
            quitting = True
            # 顺序不能反：先让 uvicorn 开始优雅停（lifespan 里 drain 记忆/trace），
            # 再销毁窗口让 webview.start() 返回；主线程最后 join 等 drain 收尾
            if notifier is not None:
                notifier.stop()
            server.should_exit = True
            icon.stop()
            window.destroy()

        icon = pystray.Icon(
            "second-brain-agent",
            make_icon(64),
            "第二大脑 Agent",
            menu=pystray.Menu(
                Item("打开主界面", show_window, default=True),
                Item("退出", quit_app),
            ),
        )
        threading.Thread(target=icon.run, daemon=True).start()
        notifier = _start_tray_notifications(icon)
    except Exception as exc:
        logger.warning("托盘不可用，已降级为纯窗口模式：%s: %s", type(exc).__name__, exc)
        icon = notifier = None

    webview.start()  # 主线程跑 GUI 消息循环，窗口销毁后返回
    if notifier is not None:
        notifier.stop()
    if icon is not None:
        icon.stop()  # 降级路径下没起图标，别在 None 上调 stop
    # 降级路径（无托盘）里用户点 X 是真关窗口，没走 quit_app，所以这里补一次停服，
    # 否则 lifespan 的 drain 跑不到，最后几轮记忆/trace 会随进程一起丢
    server.should_exit = True
    server_thread.join(timeout=10)  # 给 lifespan 的 drain 一个收尾窗口


if __name__ == "__main__":
    main()
