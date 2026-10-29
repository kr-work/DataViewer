"""プロジェクトフォルダ（SPEC §5.3）の作成・オープン。

フォルダ構成::

    my-experiment/
    ├── project.json       # プロジェクトの基本情報と保存形式バージョン
    ├── metadata.sqlite    # 実験/試技/Source/Channel/Event
    ├── originals/         # インポートした元データのコピー
    ├── datasets/          # 正規化データ（Parquet）
    ├── derived/           # 処理結果のキャッシュ（Parquet）
    ├── pipelines/         # 処理定義（JSON）
    ├── views/             # 表示設定（JSON）
    └── exports/           # CSV、画像等

メタデータ内のファイルパスはすべてプロジェクトルートからの相対パスで保持し、
フォルダを移動・別 OS で再オープンしても壊れないようにする。
このモジュールは Qt に依存しない。
"""

import json
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from pydantic import ValidationError

from experiment_workbench.model import PROJECT_FORMAT_VERSION, Project
from experiment_workbench.storage.database import MetadataDatabase, SchemaVersionError, StorageError
from experiment_workbench.validation import validate_relative_posix_path

PROJECT_FILE_NAME = "project.json"
METADATA_DB_NAME = "metadata.sqlite"


class ProjectDirectory(StrEnum):
    """プロジェクトフォルダ直下のサブディレクトリ。"""

    ORIGINALS = "originals"
    DATASETS = "datasets"
    DERIVED = "derived"
    PIPELINES = "pipelines"
    VIEWS = "views"
    EXPORTS = "exports"


class ProjectStoreError(StorageError):
    """プロジェクトフォルダの作成・オープンに失敗した。"""


class ProjectStore:
    """1 つのプロジェクトフォルダを表し、project.json と metadata.sqlite を管理するクラス。

    インスタンスはコンストラクタではなく ``create()`` / ``open()`` で生成する。
    使い終わったら ``close()`` するか、``with`` 文で使うこと。

    使用例::

        with ProjectStore.create(Path("my-experiment"), name="歩行実験") as store:
            store.database.insert(experiment)

    Attributes:
        root: プロジェクトフォルダの絶対パス。
        project: project.json の内容。
        database: metadata.sqlite への接続。
    """

    def __init__(self, root: Path, project: Project, database: MetadataDatabase) -> None:
        """インスタンスを初期化する（通常は ``create()`` / ``open()`` を使う）。

        Args:
            root: プロジェクトフォルダの絶対パス。
            project: 読み込み済みの project.json の内容。
            database: 開いた metadata.sqlite。
        """
        self.root = root
        self.project = project
        self.database = database

    # --- 生成 ----------------------------------------------------------------

    @classmethod
    def create(cls, root: Path, name: str, description: str = "") -> "ProjectStore":
        """新しいプロジェクトフォルダを作成して開く。

        既存データを上書きしないよう、空でないフォルダには作成しない。

        Args:
            root: 作成するプロジェクトフォルダ。存在しないか、空のディレクトリであること。
            name: プロジェクト名。
            description: プロジェクトの説明。

        Returns:
            開いた状態の ProjectStore。

        Raises:
            ProjectStoreError: ``root`` がファイル、または空でないディレクトリの場合。
            pydantic.ValidationError: ``name`` が空白のみなど、不正な値の場合。
        """
        root = root.absolute()
        if root.exists():
            if not root.is_dir():
                raise ProjectStoreError(f"同名のファイルが存在します: {root}")
            if any(root.iterdir()):
                raise ProjectStoreError(f"空でないフォルダにはプロジェクトを作成できません: {root}")
        # フォルダを作る前に検証し、不正な名前で空フォルダだけが残らないようにする。
        project = Project(name=name, description=description)
        root.mkdir(parents=True, exist_ok=True)
        _write_project_file(root, project)
        for directory in ProjectDirectory:
            (root / directory).mkdir()
        database = MetadataDatabase(root / METADATA_DB_NAME)
        return cls(root, project, database)

    @classmethod
    def open(cls, root: Path) -> "ProjectStore":
        """既存のプロジェクトフォルダを開く。

        project.json の形式バージョンを確認してから DB を開く。DB はここで最新スキーマへ移行される。
        Git や ZIP では空ディレクトリが失われることがあるため、データを含まない
        サブディレクトリが欠けている場合は作り直す。一方、metadata.sqlite が欠けている場合は
        データ消失の可能性があるため、空の DB を作らずにエラーにする。

        Args:
            root: プロジェクトフォルダ。

        Returns:
            開いた状態の ProjectStore。

        Raises:
            ProjectStoreError: project.json / metadata.sqlite がない場合、
                または project.json の内容が不正な場合。
            SchemaVersionError: project.json または DB がこのアプリより新しい形式の場合。
        """
        root = root.absolute()
        project = _read_project_file(root)
        db_path = root / METADATA_DB_NAME
        if not db_path.is_file():
            raise ProjectStoreError(f"{METADATA_DB_NAME} が見つかりません: {db_path}")
        for directory in ProjectDirectory:
            (root / directory).mkdir(exist_ok=True)
        database = MetadataDatabase(db_path)
        return cls(root, project, database)

    # --- パス ----------------------------------------------------------------

    def directory(self, kind: ProjectDirectory) -> Path:
        """プロジェクト直下のサブディレクトリの絶対パスを返す。

        Args:
            kind: サブディレクトリの種類。

        Returns:
            サブディレクトリの絶対パス。
        """
        return self.root / kind

    def resolve(self, relpath: str) -> Path:
        """メタデータに保存された相対パスを、プロジェクト内の絶対パスに変換する。

        文字列としての検証（".." や絶対パスの禁止）に加えて、シンボリックリンクを辿った
        実体がプロジェクトフォルダの外に出ないことも確認する。

        Args:
            relpath: プロジェクトルートからの "/" 区切り相対パス。

        Returns:
            プロジェクトフォルダ内の絶対パス（ファイルの存在は確認しない）。

        Raises:
            ValueError: 相対パスとして不正な場合、または実体がプロジェクト外を指す場合。
        """
        validate_relative_posix_path(relpath)
        path = self.root.joinpath(*relpath.split("/"))
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"プロジェクトフォルダの外を指すパスです: {relpath!r}")
        return path

    # --- 接続管理 ------------------------------------------------------------

    def close(self) -> None:
        """DB 接続を閉じる。"""
        self.database.close()

    def __enter__(self) -> "ProjectStore":
        """``with`` 文の開始時に自身を返す。

        Returns:
            自身。
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """``with`` 文の終了時に DB 接続を閉じる。

        Args:
            exc_type: 発生した例外の型（なければ None）。
            exc: 発生した例外（なければ None）。
            tb: トレースバック（なければ None）。
        """
        self.close()


# ---------------------------------------------------------------------------
# project.json の読み書き（モジュール内部用）
# ---------------------------------------------------------------------------


def _write_project_file(root: Path, project: Project) -> None:
    """project.json を書き込む。

    書き込み途中でアプリが落ちても壊れたファイルが残らないよう、一時ファイルに書いてから
    置き換える。

    Args:
        root: プロジェクトフォルダ。
        project: 書き込む内容。
    """
    path = root / PROJECT_FILE_NAME
    tmp_path = path.with_name(f"{PROJECT_FILE_NAME}.tmp")
    tmp_path.write_text(project.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_project_file(root: Path) -> Project:
    """project.json を読み込んで検証する。

    モデルは未知のフィールドを拒否するため、新しい形式のファイルを読むと
    分かりにくい検証エラーになる。そこで、モデル検証の前に形式バージョンだけを確認する。

    Args:
        root: プロジェクトフォルダ。

    Returns:
        検証済みの Project。

    Raises:
        ProjectStoreError: ファイルがない、JSON として不正、または内容が不正な場合。
        SchemaVersionError: 形式バージョンがこのアプリより新しい場合。
    """
    path = root / PROJECT_FILE_NAME
    if not path.is_file():
        raise ProjectStoreError(
            f"プロジェクトフォルダではありません（{PROJECT_FILE_NAME} がありません）: {root}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectStoreError(f"{path} を JSON として読み込めません: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectStoreError(f"{path} の最上位が JSON オブジェクトではありません")

    version = raw.get("format_version")
    # bool は int のサブクラスなので明示的に除外する。
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectStoreError(f"{path} の format_version が整数ではありません: {version!r}")
    if version > PROJECT_FORMAT_VERSION:
        raise SchemaVersionError(
            f"{path} の形式バージョン {version} は、このアプリが扱えるバージョン "
            f"{PROJECT_FORMAT_VERSION} より新しいため開けません。アプリを更新してください。"
        )
    # 古い形式の変換処理は、形式バージョン 2 を導入するときにここへ追加する。

    try:
        return Project.model_validate(raw)
    except ValidationError as exc:
        raise ProjectStoreError(f"{path} の内容が不正です:\n{exc}") from exc
