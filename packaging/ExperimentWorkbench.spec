# PyInstaller の設定（packaging/build_release.py から使う）。
#
# 方針:
# - Qt の DLL を利用者が差し替えられるよう、1 ファイルにまとめないフォルダ形式（onedir）で作る（LGPL-3.0 第 4 条）。
# - 同梱するのは Python のパッケージと Python 本体のファイルだけにする。OS のライブラリ（Linux の /usr/lib、
#   Windows の System32 など）は同梱せず、利用者の OS のものを使う。同梱するとそのライセンスへの対応も必要になり、
#   OS と版が合わないライブラリで起動しなくなることもあるため。外したものは一覧を表示する。

import sys
import sysconfig
from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH は PyInstaller が渡す
APP_NAME = "ExperimentWorkbench"
ALLOWED_ROOTS = [
    Path(sysconfig.get_paths()["purelib"]).resolve(),
    Path(sysconfig.get_paths()["platlib"]).resolve(),
    Path(sys.base_prefix).resolve(),
]


def comes_from_python(source):
    """同梱してよいファイル（Python のパッケージか Python 本体のもの）なら True を返す。"""
    path = Path(source).resolve()
    return any(path.is_relative_to(root) for root in ALLOWED_ROOTS)


a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    excludes=[
        # 使わないもの（サイズを減らすため）。
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "pyqtgraph.examples",
        "pyqtgraph.opengl",
        "pyqtgraph.jupyter",
        # 開発・ビルドの道具（pydantic の mypy 用プラグインなどが、入っていれば連れてくる）。
        "mypy",
        "pydantic.mypy",
        "setuptools",
        "pkg_resources",
        "distutils",
        # 対話用の入力補完（Linux では GPL の GNU readline と結び付くことがあるので入れない）。
        "readline",
    ],
    noarchive=False,
)

dropped = [entry for entry in a.binaries if not comes_from_python(entry[1])]
a.binaries = [entry for entry in a.binaries if comes_from_python(entry[1])]
print(f"[spec] OS のライブラリとして同梱しないもの: {len(dropped)} 個")
for name, source, _ in sorted(dropped):
    print(f"[spec]   {name} <- {source}")

pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    upx=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    name=APP_NAME,
    upx=False,
)
