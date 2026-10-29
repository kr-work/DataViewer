"""Experiment Workbench: 研究実験の時系列データ向け汎用ワークベンチ。

パッケージ構成（Phase 0 時点）:
    - ``model``: 型付きドメインモデル（pydantic）。データ受け渡し用のモデルはここに集約する。
    - ``storage``: プロジェクトフォルダと SQLite メタデータの永続化。Qt に依存しない。
    - ``synthetic``: 検証用の合成データ生成。

GUI（PySide6 / PyQtGraph）は Phase 1 以降で ``ui`` サブパッケージとして追加する。
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # インストール済みメタデータからバージョンを取得し、pyproject.toml と二重管理しない。
    __version__ = version("experiment-workbench")
except PackageNotFoundError:  # pragma: no cover - ソースツリーを直接参照した場合のみ
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
