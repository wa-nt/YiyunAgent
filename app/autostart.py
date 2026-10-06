"""Windows 开机自启：写 / 删 `HKCU\\...\\Run` 下的一个值。

为什么只支持打包版
    自启项要记的是一条**能在重启后独立启动**的命令。源码环境里那是 `python.exe`，得靠
    当前工作目录、虚拟环境和 `.env` 的相对路径口径才能跑起来——写进注册表大概率是一条
    开机就报错、用户又不知道去哪删的记录。所以源码环境明确拒绝启用（返回可读错误），
    而不是写一条看着成功、实际跑不起来的命令。

    判定与 UI 的禁用状态用同一个 `is_supported`，不在两处各写一份条件。

为什么用 HKCU
    只影响当前用户，不需要管理员权限，卸载时删自己的值即可（不需要额外的清理逻辑）。

为什么单独成模块
    它是唯一直接碰注册表的地方，测试用假 winreg 替换 `_winreg()` 就能完整覆盖
    （真写 HKCU 等于给跑测试的这台机器装一个开机自启项）。
"""

from __future__ import annotations

import subprocess
import sys

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
# 值名就是注册表里显示给用户看的那一行，用产品名而不是包名
VALUE_NAME = "YiyunAgent"
# 固定启动参数：桌面端不带参数就是正常启动。留成常量是为了让「自启时跑的命令」在一处
# 可见，将来真要加 `--tray` 之类只需改这里
STARTUP_ARGS: tuple[str, ...] = ()

# 支持判定：只有打包后的 Windows 程序才有稳定的自启命令（见模块 docstring）
is_supported = sys.platform == "win32" and bool(getattr(sys, "frozen", False))


class AutostartError(RuntimeError):
    """注册表访问失败。HTTP 层把它翻成 500 并原样带上原因。"""


def _winreg():
    """按需导入 winreg。单独成函数是让测试能整体替换成假实现。"""
    import winreg

    return winreg


def _command() -> str:
    """自启要执行的完整命令行。

    `list2cmdline` 负责引用：路径里有空格（`C:\\Program Files\\...`）时不引用的话，
    Run 键会把 `C:\\Program` 当成可执行文件。
    """
    return subprocess.list2cmdline([sys.executable, *STARTUP_ARGS])


def _wrap(exc: OSError) -> AutostartError:
    return AutostartError(f"注册表访问失败：{exc}")


def is_enabled() -> bool:
    """当前是否已启用自启。值不存在（含 Run 键本身不存在）就是没启用。"""
    winreg = _winreg()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, VALUE_NAME)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise _wrap(exc) from exc
    return True


def enable() -> None:
    """写入自启项。重复调用幂等（同一个值名覆盖同一个值）。

    只动自己的值：Run 键里还有别的软件，删键或清空键都会把它们的自启一起搞掉。
    """
    winreg = _winreg()
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, _command())
    except OSError as exc:
        raise _wrap(exc) from exc


def disable() -> None:
    """删除自启项。值或 Run 键不存在也算成功——「关掉一个本来就关着的开关」不该报错。"""
    winreg = _winreg()
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _wrap(exc) from exc
