"""桌面入口：原生窗口 + 系统托盘，复用现有 FastAPI 应用（叠加式，不动任何现有模块）。

形态：随机空闲端口后台起 uvicorn → pywebview 原生窗口指向它。点 X 是最小化到
托盘（微信/QQ 惯例），真退出走托盘菜单；退出时先置 uvicorn.should_exit，让
app.main 的 lifespan 收尾（记忆/trace drain）跑完，再销毁窗口。

配置口径：.env / data/ / skills/ 都按 CWD 相对解析（见 config.py），所以
PyInstaller 打包后先把 CWD 锚到 exe 所在目录——从源码跑时则不碰 CWD（按文档
约定从项目根启动）。

用法：
    python -m app.desktop              启动桌面端
    python -m app.desktop --make-icon  生成 web/app.ico（build_desktop.bat 打包前用）
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time


def _free_port() -> int:
    """让系统分配一个空闲端口。不固定 8765：与开发模式并存时不能互相抢端口。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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

    import pystray
    import webview
    from pystray import MenuItem as Item

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
        # 点 X = 藏到托盘，不是退出；返回 False 取消 pywebview 的默认关闭
        if quitting:
            return True
        window.hide()
        return False

    window.events.closing += on_closing

    def show_window(icon=None, item=None):
        window.show()
        window.restore()

    def quit_app(icon=None, item=None):
        nonlocal quitting
        quitting = True
        # 顺序不能反：先让 uvicorn 开始优雅停（lifespan 里 drain 记忆/trace），
        # 再销毁窗口让 webview.start() 返回；主线程最后 join 等 drain 收尾
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

    webview.start()  # 主线程跑 GUI 消息循环，窗口销毁后返回
    server_thread.join(timeout=10)  # 给 lifespan 的 drain 一个收尾窗口


if __name__ == "__main__":
    main()
