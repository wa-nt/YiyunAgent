"""仓库内资源（schema / web / 默认人格）的位置解析。

源码环境 = 仓库根目录；PyInstaller 冻结后 = 解包目录（`sys._MEIPASS`），因为 spec 里的
datas 把资源按**与仓库相同的相对布局**放进去（`app/schema.sql`、`web/`、
`app/agent/prompts/persona_default.md`），两种环境用同一段相对路径就能命中。
单独成模块而不是各自 `Path(__file__)`：`.py` 在包内是冻结进 PYZ 的，靠 `__file__`
反推资源目录在打包后依赖 PyInstaller 的实现细节，散落成多份更容易漂移。
"""

from __future__ import annotations

import sys
from pathlib import Path


def bundle_root() -> Path:
    """资源的根目录：冻结后优先 `sys._MEIPASS`（onedir/onefile 都由它指向解包目录）。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(relative: str | Path) -> Path:
    """把仓库相对路径解析成当前环境的真实路径；绝对路径原样返回（允许外部覆盖）。"""
    path = Path(relative)
    return path if path.is_absolute() else bundle_root() / path
