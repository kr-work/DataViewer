"""配布用の実行ファイルを作り、ライセンス文書を同梱して圧縮する（Windows と Linux）。

使い方（リポジトリの直下で）::

    uv sync --group build
    uv run python packaging/build_release.py [--expected-version v0.1.0] [--skip-smoke-test]

行うこと:
    1. PyInstaller でフォルダ形式の実行ファイルを作る（設定は ``ExperimentWorkbench.spec``）。
    2. 同梱した全ファイルの出どころを、パッケージの記録（RECORD）と Python 本体の
       ファイルで照らし合わせる。出どころの分からないファイルがあれば止める
       （ライセンスの確認漏れを防ぐため）。
    3. 出どころのパッケージごとにライセンス文書を ``licenses/`` に写す。PySide6 には
       wheel に含まれない LGPL-3.0 と GPL-3.0 の全文を加え、同じ版の Qt と PySide6 の
       ソースの入手先を ``licenses/README.txt`` に書く。
    4. 実行ファイルを ``--smoke-test`` で起動し、ファイルを開いて処理を実行できることを確かめる。
    5. ``dist/`` に圧縮ファイル（Windows は zip、Linux は tar.gz）と、Release の説明文を作る。

この手順の理由は ``THIRD_PARTY_NOTICES.md`` の「配布物を作るとき」を参照。
"""

import argparse
import ast
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "ExperimentWorkbench"
PROJECT_NAME = "experiment-workbench"
SPEC = ROOT / "packaging" / f"{APP_NAME}.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build"
GNU_TEXTS = (
    ROOT / "packaging" / "licenses" / "LGPL-3.0.txt",
    ROOT / "packaging" / "licenses" / "GPL-3.0.txt",
)
LGPL_PACKAGES = frozenset({"pyside6-essentials", "shiboken6"})
"""LGPL-3.0 を選んで使うパッケージ（wheel に LGPL と GPL の全文がないので補う）。"""

LICENSE_FILE = re.compile(r"(?i)^(licen[sc]e|copying|notice)|licen[sc]e")
"""ライセンス文書とみなすファイル名。"""

CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyi", ".pyd", ".so", ".dll"})
"""名前に license を含んでもライセンス文書ではないファイルの拡張子。"""

SMOKE_TIMEOUT_S = 600
"""起動の確認を待つ最大時間 [s]。"""

QT_SOURCE_PAGES = (
    "https://download.qt.io/official_releases/qt/{minor}/{version}/single/",
    "https://download.qt.io/archive/qt/{minor}/{version}/single/",
)
"""Qt のソースの置き場所（新しい版は official_releases、古くなると archive に移る）。"""

PYTHON_LICENSE_RST = "https://raw.githubusercontent.com/python/cpython/v{version}/Doc/license.rst"
"""Python に組み込まれたソフトウェアのライセンス文書（使った版と同じタグのもの）。"""

PYTHON_INCORPORATED = ("OpenSSL", "expat", "libffi", "zlib", "libmpdec")
"""``Doc/license.rst`` に含まれているはずの文書（含まれていなければ止める）。"""

PYSIDE_SOURCE_PAGES = (
    "https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{version}-src/",
    "https://download.qt.io/archive/qtforpython/pyside6/PySide6-{version}-src/",
)
"""PySide6 のソースの置き場所。"""


def normalized(name: str) -> str:
    """パッケージ名を比べられる形にする（PEP 503 の正規化）。

    Args:
        name: パッケージ名。

    Returns:
        小文字にし、``-_.`` の並びを ``-`` にそろえた名前。
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def path_key(relative: str) -> str:
    """同梱したファイルと記録上のパスを比べるための形にする。

    Args:
        relative: ``/`` 区切りの相対パス。

    Returns:
        比べるための文字列（Windows はファイル名の大文字・小文字を区別しないので小文字にする）。
    """
    key = relative.replace("\\", "/")
    return key.lower() if sys.platform == "win32" else key


def platform_tag() -> str:
    """圧縮ファイルの名前に付ける OS と CPU の名前を返す。

    Returns:
        例: "windows-x64"、"linux-x64"。
    """
    system = "windows" if sys.platform == "win32" else sys.platform.removesuffix("2")
    machine = platform.machine().lower()
    cpu = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return f"{system}-{cpu}"


def check_version(expected: str) -> str:
    """プロジェクトの版を返し、タグの版と違えば止める。

    Args:
        expected: タグの名前（例: "v0.1.0"）。空ならタグと比べない。

    Returns:
        プロジェクトの版（例: "0.1.0"）。

    Raises:
        SystemExit: タグと ``pyproject.toml`` の版が違う場合。
    """
    version = metadata.version(PROJECT_NAME)
    if expected and expected.removeprefix("v") != version:
        raise SystemExit(
            f"タグ {expected} と pyproject.toml の版 {version} が違います。"
            "版をそろえてからタグを付けてください"
        )
    return version


def run_pyinstaller() -> Path:
    """PyInstaller で実行ファイルのフォルダを作る。

    Returns:
        作ったフォルダ（``dist/ExperimentWorkbench``）。
    """
    command = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm", "--clean"]
    command += ["--distpath", str(DIST), "--workpath", str(WORK)]
    subprocess.run(command, check=True, cwd=ROOT)
    return DIST / APP_NAME


def distribution_index() -> dict[str, metadata.Distribution]:
    """インストール済みパッケージのファイルの記録から、パス → パッケージの表を作る。

    Returns:
        site-packages からの相対パス（``path_key`` の形）→ そのファイルを入れたパッケージ。
    """
    index: dict[str, metadata.Distribution] = {}
    for distribution in metadata.distributions():
        for file in distribution.files or []:
            relative = str(file)
            if not relative.startswith(".."):  # スクリプトなど site-packages の外のファイルは除く
                index[path_key(relative)] = distribution
    return index


def python_runtime_names() -> set[str]:
    """Python 本体に含まれるファイルの名前を返す（DLL や拡張モジュールの出どころの判定に使う）。

    Returns:
        ``sys.base_prefix`` の下にあるファイルの名前（``path_key`` の形）。
    """
    base = Path(sys.base_prefix)
    return {path_key(path.name) for path in base.rglob("*") if path.is_file()}


def archived_module_sources() -> list[Path]:
    """実行ファイル内のアーカイブ（PYZ）に入れた Python モジュールの、元のファイルを返す。

    Python だけで書かれたパッケージ（polars・typing-extensions など）は
    アーカイブに入り、フォルダにはファイルとして現れない。ライセンスの確認から
    漏れないよう、PyInstaller の一覧（PYZ-00.toc）から拾う。

    Returns:
        元のファイルのパス（組み込みモジュールなど、ファイルがないものは除く）。
    """
    listing = WORK / APP_NAME / "PYZ-00.toc"
    _, entries = ast.literal_eval(listing.read_text(encoding="utf-8"))
    return [
        Path(source)
        for _, source, _ in entries
        if isinstance(source, str) and Path(source).is_file()
    ]


def owner_of_module(
    source: Path, index: dict[str, metadata.Distribution]
) -> tuple[str, metadata.Distribution | None] | None:
    """アーカイブに入れたモジュールの出どころを返す。

    Args:
        source: モジュールの元のファイル。
        index: ``distribution_index`` の表。

    Returns:
        (出どころの名前, パッケージ。Python 本体と本体のコードは None)。分からなければ None。
    """
    path = source.resolve()
    for site in {
        Path(p).resolve()
        for p in (sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"])
    }:
        if path.is_relative_to(site):
            distribution = index.get(path_key(path.relative_to(site).as_posix()))
            return (
                None
                if distribution is None
                else (normalized(distribution.metadata["Name"]), distribution)
            )
    if path.is_relative_to(Path(sys.base_prefix).resolve()):
        return "python", None
    if path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "packaging"):
        return PROJECT_NAME, None
    return None


def take_inventory(bundle: Path) -> tuple[dict[str, metadata.Distribution | None], list[str]]:
    """同梱した全ファイルと、アーカイブに入れた全モジュールの出どころを調べる。

    実行ファイル本体は PyInstaller のブートローダー、``base_library.zip`` は
    Python の標準ライブラリとみなす。
    シンボリックリンクは指す先のファイルで数えるので飛ばす。

    Args:
        bundle: 実行ファイルのフォルダ。

    Returns:
        (出どころの名前 → パッケージ（Python 本体と本体のコードは None）,
        出どころの分からないもののリスト)。
    """
    index = distribution_index()
    runtime = python_runtime_names()
    contents = bundle / "_internal"
    owners: dict[str, metadata.Distribution | None] = {"python": None, PROJECT_NAME: None}
    owners["pyinstaller"] = metadata.distribution("pyinstaller")
    unknown: list[str] = []
    for source in archived_module_sources():
        owner = owner_of_module(source, index)
        if owner is None:
            unknown.append(f"（アーカイブ）{source}")
        else:
            owners[owner[0]] = owner[1]
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.relative_to(bundle).parts[0] == "licenses":
            continue  # 前回の実行で写したライセンス文書
        if path.parent == bundle:
            continue  # 実行ファイル本体（PyInstaller のブートローダー）と、後で置く文書
        relative = path.relative_to(contents).as_posix()
        distribution = index.get(path_key(relative))
        if distribution is not None:
            owners[normalized(distribution.metadata["Name"])] = distribution
        elif relative == "base_library.zip" or path_key(path.name) in runtime:
            continue  # Python 本体（owners の "python"）
        else:
            unknown.append(relative)
    return owners, unknown


def license_files(distribution: metadata.Distribution) -> list[Path]:
    """パッケージのライセンス文書を返す（dist-info の中と、パッケージに同梱された文書）。

    Args:
        distribution: パッケージ。

    Returns:
        ライセンス文書のパス。
    """
    found: list[Path] = []
    for file in distribution.files or []:
        name = file.name
        if Path(name).suffix.lower() in CODE_SUFFIXES or not LICENSE_FILE.search(name):
            continue
        path = Path(str(distribution.locate_file(file)))
        if path.is_file():
            found.append(path)
    return found


def python_license() -> Path:
    """Python 本体のライセンス文書を返す。

    Returns:
        LICENSE.txt のパス。

    Raises:
        SystemExit: 見つからない場合（Python の配布形態によって置き場所が違うため、
            黙って省かない）。
    """
    base = Path(sys.base_prefix)
    candidates = [base / "LICENSE.txt", base / "LICENSE"]
    candidates += sorted(base.glob("lib/python3*/LICENSE.txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Python のライセンス文書が {base} に見つかりません")


def python_incorporated_licenses(version: str) -> bytes:
    """Python に組み込まれたソフトウェア（OpenSSL・libffi など）の文書を取得する。

    Python の配布物の LICENSE.txt には、これらの文書が入っていないことがある。CPython は同じ内容を
    ``Doc/license.rst`` に載せているので、使った版と同じタグのものを取得する。

    Args:
        version: Python の版（例: "3.12.13"）。

    Returns:
        ``Doc/license.rst`` の内容。

    Raises:
        SystemExit: 取得できない、または想定した文書が含まれていない場合（黙って省かないため）。
    """
    url = PYTHON_LICENSE_RST.format(version=version)
    request = urllib.request.Request(url, headers={"User-Agent": "release-build"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise SystemExit(
            f"Python の同梱ソフトウェアの文書を取得できません: {url}（{error}）"
        ) from None
    text = content.decode("utf-8", errors="replace")
    missing = [name for name in PYTHON_INCORPORATED if name not in text]
    if missing:
        raise SystemExit(f"{url} に {missing} の文書が見つかりません")
    return content


def license_expression(distribution: metadata.Distribution) -> str:
    """パッケージのライセンスの表記を返す（一覧に書くため）。

    Args:
        distribution: パッケージ。

    Returns:
        SPDX の式、ライセンス欄の 1 行目、または分類の表記。
    """
    meta = distribution.metadata
    expression = meta.get("License-Expression") or ""
    if expression:
        return expression
    # License 欄には著作権の行や全文が入っていることがあるので、分類の表記を優先する。
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License")
    ]
    if classifiers:
        return ", ".join(classifiers)
    field = (meta.get("License") or "").strip().splitlines()
    if field and len(field[0]) <= 80 and not field[0].lower().startswith("copyright"):
        return field[0]
    return "同梱の文書を参照"


def copy_licenses(bundle: Path, owners: dict[str, metadata.Distribution | None]) -> list[str]:
    """出どころのパッケージごとに、ライセンス文書を ``licenses/`` に写す。

    Args:
        bundle: 実行ファイルのフォルダ。
        owners: 出どころのパッケージ名 → パッケージ（Python 本体は None）。

    Returns:
        一覧の行（"名前 版: ライセンス"）。

    Raises:
        SystemExit: ライセンス文書が 1 つもないパッケージがある場合。
    """
    folder = bundle / "licenses"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    python_version = platform.python_version()
    python_folder = folder / f"python-{python_version}"
    python_folder.mkdir()
    shutil.copy2(python_license(), python_folder / "LICENSE.txt")
    (python_folder / "incorporated-software.rst").write_bytes(
        python_incorporated_licenses(python_version)
    )
    lines = [
        f"Experiment Workbench {metadata.version(PROJECT_NAME)}（本体）: MIT（LICENSE.txt）",
        f"Python {python_version}: PSF-2.0（LICENSE.txt）。組み込まれた OpenSSL・"
        "libffi・zlib・expat・"
        "libmpdec などの文書は incorporated-software.rst",
    ]
    missing: list[str] = []
    for name, distribution in sorted(owners.items()):
        if distribution is None or name == PROJECT_NAME:
            # Python 本体と本体は上で扱った（本体のパッケージ情報には LICENSE の記録がない）。
            continue
        version = distribution.version
        target = folder / f"{name}-{version}"
        files = license_files(distribution)
        # PySide6 の wheel のライセンス文書は商用ライセンスの説明だけで、Linux 版には
        # 1 つもないこともある。
        # LGPL-3.0 で使うので、LGPL-3.0 と GPL-3.0 の全文を必ず入れる。
        extra = GNU_TEXTS if name in LGPL_PACKAGES else ()
        if not files and not extra:
            missing.append(name)
            continue
        target.mkdir(parents=True)
        site = Path(str(distribution.locate_file("")))
        for file in files:
            destination = target / file.relative_to(site)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        for text in extra:
            shutil.copy2(text, target / text.name)
        lines.append(f"{name} {version}: {license_expression(distribution)}")
    if missing:
        raise SystemExit(f"ライセンス文書が見つからないパッケージがあります: {missing}")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", folder / "THIRD_PARTY_NOTICES.md")
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE.txt")
    return lines


def reachable(url: str) -> bool:
    """ページが開けるかを返す。

    Args:
        url: ページの URL。

    Returns:
        開ければ True。
    """
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "release-build"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, TimeoutError):
        return False


def qt_source_links() -> tuple[str, str, str]:
    """同梱した版の Qt と PySide6 のソースの置き場所を返す（実在を確かめる）。

    Returns:
        (Qt の版, Qt のソースの URL, PySide6 のソースの URL)。

    Raises:
        SystemExit: どちらかの置き場所が見つからない場合（入手先を示せないまま配布しないため）。
    """
    from PySide6 import QtCore, __version__

    qt_version = QtCore.qVersion()
    minor = ".".join(qt_version.split(".")[:2])
    qt_pages = [page.format(minor=minor, version=qt_version) for page in QT_SOURCE_PAGES]
    pyside_pages = [page.format(version=__version__) for page in PYSIDE_SOURCE_PAGES]
    qt_url = next((page for page in qt_pages if reachable(page)), "")
    pyside_url = next((page for page in pyside_pages if reachable(page)), "")
    if not qt_url or not pyside_url:
        raise SystemExit(
            f"Qt {qt_version} または PySide6 {__version__} のソースの置き場所が見つかりません"
        )
    return qt_version, qt_url, pyside_url


def write_notice(bundle: Path, lines: list[str], links: tuple[str, str, str]) -> None:
    """``licenses/README.txt`` に、同梱物の一覧と Qt（LGPL）の表示を書く。

    Args:
        bundle: 実行ファイルのフォルダ。
        lines: 同梱物の一覧の行。
        links: (Qt の版, Qt のソースの URL, PySide6 のソースの URL)。
    """
    qt_version, qt_url, pyside_url = links
    text = [
        "Experiment Workbench に同梱したソフトウェアのライセンス",
        "=" * 40,
        "",
        "本体のライセンスは LICENSE.txt（MIT）。",
        "同梱したライブラリのライセンス文書は、このフォルダの",
        "「名前-版」のフォルダにある。",
        "",
        "Qt / PySide6",
        "-" * 12,
        f"This software uses Qt {qt_version} and PySide6 under the GNU LGPL version 3.",
        f"このソフトウェアは Qt {qt_version} と PySide6 を GNU LGPL 第 3 版のもとで使っています。",
        "LGPL-3.0 と GPL-3.0 の全文は pyside6-essentials と shiboken6 のフォルダにあります。",
        "Qt のライブラリは _internal/PySide6 に別ファイルのまま置いているので、",
        "互換のある版に差し替えられます。",
        "同梱した版のソースの入手先:",
        f"  Qt:      {qt_url}",
        f"  PySide6: {pyside_url}",
        "",
        "同梱したもの",
        "-" * 12,
        *lines,
        "PyInstaller のブートローダー（実行ファイル本体）:",
        "  GPL-2.0 にブートローダーの例外が付いたもの",
        "",
    ]
    (bundle / "licenses" / "README.txt").write_text("\n".join(text), encoding="utf-8")


def write_release_notes(version: str, links: tuple[str, str, str]) -> Path:
    """Release の説明文を書く。

    Args:
        version: プロジェクトの版。
        links: (Qt の版, Qt のソースの URL, PySide6 のソースの URL)。

    Returns:
        説明文のファイル。
    """
    qt_version, qt_url, pyside_url = links
    notes = f"""## Experiment Workbench {version}

圧縮ファイルを展開し、中の `{APP_NAME}`（Windows は `{APP_NAME}.exe`）を起動します。
Ubuntu では Qt が使うシステムのライブラリが必要です（README の「セットアップ」を参照）。

### ライセンス

本体は MIT ライセンスです。同梱したライブラリのライセンス文書は、
展開したフォルダの `licenses/` にあります。

This software uses Qt {qt_version} and PySide6 under the GNU LGPL version 3.
Qt と PySide6 のソース: [Qt {qt_version}]({qt_url}) / [PySide6]({pyside_url})
"""
    path = DIST / "release-notes.md"
    path.write_text(notes, encoding="utf-8")
    return path


def run_smoke_test(bundle: Path) -> None:
    """作った実行ファイルを起動の確認のモードで動かす。

    Args:
        bundle: 実行ファイルのフォルダ。

    Raises:
        SystemExit: 起動や処理に失敗した場合。
    """
    executable = bundle / (f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)
    log = WORK / "smoke-test.log"
    log.unlink(missing_ok=True)
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [str(executable), "--smoke-test", str(log)],
        env=environment,
        timeout=SMOKE_TIMEOUT_S,
        check=False,
    )
    print(log.read_text(encoding="utf-8") if log.exists() else "（ログがありません）")
    if result.returncode != 0:
        raise SystemExit(f"起動の確認に失敗しました（終了コード {result.returncode}）")


def make_archive(bundle: Path, version: str) -> Path:
    """配布用の圧縮ファイルを作る（Linux は実行権限を保つ tar.gz）。

    Args:
        bundle: 実行ファイルのフォルダ。
        version: プロジェクトの版。

    Returns:
        圧縮ファイル。
    """
    base = DIST / f"{APP_NAME}-{version}-{platform_tag()}"
    archive_format = "zip" if sys.platform == "win32" else "gztar"
    return Path(shutil.make_archive(str(base), archive_format, root_dir=DIST, base_dir=bundle.name))


def main() -> None:
    """配布用の実行ファイルを作る。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expected-version", default="", help="タグの名前（例: v0.1.0）。版の一致を確かめる"
    )
    parser.add_argument("--skip-smoke-test", action="store_true", help="起動の確認を省く（調査用）")
    arguments = parser.parse_args()
    version = check_version(arguments.expected_version)
    bundle = run_pyinstaller()
    owners, unknown = take_inventory(bundle)
    if unknown:
        listing = "\n  ".join(unknown)
        raise SystemExit(
            f"出どころの分からないファイルがあります（ライセンスを確認できません）:\n  {listing}"
        )
    print(f"同梱物の出どころ: {', '.join(sorted(owners))}")
    lines = copy_licenses(bundle, owners)
    links = qt_source_links()
    write_notice(bundle, lines, links)
    if not arguments.skip_smoke_test:
        run_smoke_test(bundle)
    archive = make_archive(bundle, version)
    notes = write_release_notes(version, links)
    print(f"作成: {archive}（{archive.stat().st_size / 1e6:.0f} MB）、{notes}")


if __name__ == "__main__":
    main()
