"""プロジェクトフォルダ（SPEC §5.3）の作成・オープンと、フォルダ内のファイルの読み書き。

元データのコピー（originals/）、正規化データ（datasets/ の Parquet）、表示設定（views/）の
読み書きもここで行う。DB に触れないメソッド（ファイルの読み書き）は、
ワーカースレッドからも呼べる。

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

import hashlib
import io
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from uuid import UUID

import numpy as np
import polars as pl
from pydantic import ValidationError

from experiment_workbench.model import (
    ANALYSIS_FORMAT_VERSION,
    DATASET_TIME_COLUMN,
    PROJECT_FORMAT_VERSION,
    VIEW_FORMAT_VERSION,
    AnalysisSpec,
    Channel,
    ChannelData,
    PlotView,
    Project,
    Source,
)
from experiment_workbench.storage.database import MetadataDatabase, SchemaVersionError, StorageError
from experiment_workbench.validation import VIEW_NAME_PATTERN, validate_relative_posix_path

# ファイルのハッシュを計算するときに一度に読む大きさ [byte]。
# 大きなファイルでもメモリを使いすぎないため。
_HASH_CHUNK_BYTES = 1024 * 1024

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

    # --- 元データ・正規化データ ------------------------------------------------

    def write_original(self, relpath: str, data: bytes) -> str:
        """元ファイルの内容を originals/ 以下に書き込み、保存されたファイルの SHA-256 を返す。

        既存のファイルは上書きしない。書き込み後に保存されたファイルを読み直してハッシュを
        計算するので、呼び出し側は元の内容のハッシュと比べてコピーの完全性を確認できる。

        Args:
            relpath: 書き込み先の相対パス（originals/ 以下）。
            data: 元ファイルの内容。

        Returns:
            保存されたファイルの SHA-256（16 進小文字）。

        Raises:
            FileExistsError: 書き込み先が既に存在する場合。
        """
        _write_bytes_atomically(self.resolve(relpath), data, overwrite=False)
        return self.file_sha256(relpath)

    def write_dataset(self, relpath: str, frame: pl.DataFrame) -> None:
        """正規化データを Parquet として datasets/ 以下に書き込む。

        Args:
            relpath: 書き込み先の相対パス（datasets/ 以下）。
            frame: 書き込む表（先頭が時刻列 ``time_s``、残りがチャンネル）。

        Raises:
            FileExistsError: 書き込み先が既に存在する場合（正規化データも上書きしない）。
        """
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        _write_bytes_atomically(self.resolve(relpath), buffer.getvalue(), overwrite=False)

    def file_sha256(self, relpath: str) -> str:
        """プロジェクト内のファイルの SHA-256 を計算する。

        Args:
            relpath: ファイルの相対パス。

        Returns:
            SHA-256（16 進小文字）。
        """
        digest = hashlib.sha256()
        with self.resolve(relpath).open("rb") as file:
            while chunk := file.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    def remove_file(self, relpath: str) -> None:
        """取り込みに失敗したときに、書きかけのファイルを削除する。

        ファイルを置いていたディレクトリ（originals/<Source ID>/ など）が空になった場合は、
        それも削除する。プロジェクト直下のサブディレクトリ自体は削除しない。

        Args:
            relpath: 削除するファイルの相対パス。存在しなくてもよい。
        """
        path = self.resolve(relpath)
        path.unlink(missing_ok=True)
        parent = path.parent
        top_level = {self.directory(kind) for kind in ProjectDirectory}
        if parent not in top_level and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def read_channel_data(self, source: Source, channel: Channel) -> ChannelData:
        """正規化データから 1 チャンネル分の時刻と値を読み込む。

        Args:
            source: チャンネルが属する Source。
            channel: 読み込むチャンネル。

        Returns:
            時刻 [s] と値（欠損は NaN）。

        Raises:
            ProjectStoreError: 正規化データのファイルや列が見つからない場合。
        """
        try:
            frame = pl.read_parquet(
                self.resolve(source.dataset_relpath), columns=[DATASET_TIME_COLUMN, channel.name]
            )
        except (OSError, pl.exceptions.PolarsError) as exc:
            raise ProjectStoreError(
                f"正規化データを読み込めません"
                f"（{source.dataset_relpath}、列 {channel.name!r}）: {exc}"
            ) from exc
        # Float64 の欠損（null）は to_numpy で NaN になる。
        return ChannelData(
            channel=channel,
            source=source,
            time_s=frame[DATASET_TIME_COLUMN].to_numpy().astype(np.float64),
            values=frame[channel.name].to_numpy().astype(np.float64),
        )

    # --- 表示設定 ------------------------------------------------------------

    def save_view(self, view: PlotView) -> None:
        """表示設定を views/<名前>.json に保存する（同名のファイルは置き換える）。

        Args:
            view: 保存する表示設定。
        """
        path = self.directory(ProjectDirectory.VIEWS) / f"{view.name}.json"
        data = (view.model_dump_json(indent=2) + "\n").encode("utf-8")
        _write_bytes_atomically(path, data, overwrite=True)

    def load_view(self, name: str = "main") -> PlotView | None:
        """表示設定を読み込む。

        Args:
            name: 表示設定の名前（英数字・"_"・"-" のみ）。

        Returns:
            表示設定。保存されていなければ None。

        Raises:
            ValueError: 名前に使えない文字が含まれる場合（パスの指定を防ぐため）。
            SchemaVersionError: このアプリより新しい形式で保存されている場合。
            ProjectStoreError: ファイルの内容が不正な場合。
        """
        if not re.fullmatch(VIEW_NAME_PATTERN, name):
            raise ValueError(f"表示設定の名前に使えない文字が含まれています: {name!r}")
        path = self.directory(ProjectDirectory.VIEWS) / f"{name}.json"
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectStoreError(f"{path} を JSON として読み込めません: {exc}") from exc
        version = raw.get("format_version", 1) if isinstance(raw, dict) else None
        if isinstance(version, int) and version > VIEW_FORMAT_VERSION:
            raise SchemaVersionError(
                f"{path} の形式バージョン {version} は、このアプリが扱えるバージョン "
                f"{VIEW_FORMAT_VERSION} より新しいため開けません"
            )
        if isinstance(raw, dict) and version == 1:
            raw = _upgrade_view_v1(raw)
        try:
            return PlotView.model_validate(raw)
        except ValidationError as exc:
            raise ProjectStoreError(f"{path} の内容が不正です:\n{exc}") from exc

    # --- 保存した解析と計算結果のキャッシュ ----------------------------------------

    def save_analysis(self, spec: AnalysisSpec) -> None:
        """解析の条件を pipelines/<ID>.json に保存する（同じ ID のファイルは置き換える）。

        Args:
            spec: 保存する解析。
        """
        path = self.directory(ProjectDirectory.PIPELINES) / f"{spec.id}.json"
        data = (spec.model_dump_json(indent=2) + "\n").encode("utf-8")
        _write_bytes_atomically(path, data, overwrite=True)

    def load_analyses(self) -> tuple[list[AnalysisSpec], list[str]]:
        """保存されている解析をすべて読み込む（作成日時の順）。

        1 つのファイルが壊れていても他の解析は使えるよう、読めないファイルは飛ばして
        理由を返す（黙って捨てない）。

        Returns:
            (解析のリスト, 読めなかったファイルの説明のリスト)。
        """
        specs: list[AnalysisSpec] = []
        problems: list[str] = []
        for path in sorted(self.directory(ProjectDirectory.PIPELINES).glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                version = raw.get("format_version") if isinstance(raw, dict) else None
                if isinstance(version, int) and version > ANALYSIS_FORMAT_VERSION:
                    problems.append(f"{path.name} はこのアプリより新しい形式のため読み込みません")
                    continue
                specs.append(AnalysisSpec.model_validate(raw))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
                # OSError: 他のアプリが開いている・権限がないなど。1 つのファイルのために
                # プロジェクト全体を開けなくしないよう、他の読めない理由と同じく飛ばして知らせる。
                problems.append(f"{path.name} を読み込めません: {exc}")
        specs.sort(key=_created_at)
        return specs, problems

    def delete_analysis(self, analysis_id: UUID) -> None:
        """解析の条件と、その計算結果のキャッシュを削除する（生データは変更しない）。

        Args:
            analysis_id: 削除する解析の ID。
        """
        (self.directory(ProjectDirectory.PIPELINES) / f"{analysis_id}.json").unlink(missing_ok=True)
        cache_dir = self.directory(ProjectDirectory.DERIVED) / str(analysis_id)
        if cache_dir.is_dir():
            for cache_file in cache_dir.glob("*.parquet"):
                cache_file.unlink()
            if not any(cache_dir.iterdir()):
                cache_dir.rmdir()

    def save_derived(self, analysis_id: UUID, fingerprint: str, frame: pl.DataFrame) -> None:
        """解析の計算結果をキャッシュとして derived/<解析 ID>/<指紋>.parquet に保存する。

        同じ解析の、条件が違う（指紋が違う）古いキャッシュは使われないので削除する。

        Args:
            analysis_id: 解析の ID。
            fingerprint: 計算結果を決める条件の指紋（runner.analysis_fingerprint）。
            frame: 計算結果の表（runner.result_to_frame）。
        """
        cache_dir = self.directory(ProjectDirectory.DERIVED) / str(analysis_id)
        target = cache_dir / f"{fingerprint}.parquet"
        if cache_dir.is_dir():
            for old in cache_dir.glob("*.parquet"):
                if old != target:
                    old.unlink()
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        _write_bytes_atomically(target, buffer.getvalue(), overwrite=True)

    def load_derived(self, analysis_id: UUID, fingerprint: str) -> pl.DataFrame | None:
        """解析の計算結果のキャッシュを読み込む。

        Args:
            analysis_id: 解析の ID。
            fingerprint: 計算結果を決める条件の指紋。

        Returns:
            計算結果の表。キャッシュがない、または読めない場合は None（再計算させる）。
        """
        path = (
            self.directory(ProjectDirectory.DERIVED) / str(analysis_id) / f"{fingerprint}.parquet"
        )
        if not path.is_file():
            return None
        try:
            return pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            return None

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
    data = (project.model_dump_json(indent=2) + "\n").encode("utf-8")
    _write_bytes_atomically(root / PROJECT_FILE_NAME, data, overwrite=True)


def _upgrade_view_v1(raw: dict[str, object]) -> dict[str, object]:
    """表示設定の形式 v1 を v2 に変換する（パネルの channel_ids を series_ids に改名する）。

    Phase 3 で解析の出力もパネルに置けるようにしたため、項目名を一般化した。

    Args:
        raw: v1 の表示設定（JSON を読み込んだ辞書）。

    Returns:
        v2 の表示設定の辞書。
    """
    upgraded = dict(raw)
    panels = raw.get("panels", [])
    if isinstance(panels, list):
        upgraded["panels"] = [
            {("series_ids" if key == "channel_ids" else key): value for key, value in panel.items()}
            if isinstance(panel, dict)
            else panel
            for panel in panels
        ]
    upgraded["format_version"] = VIEW_FORMAT_VERSION
    return upgraded


def _created_at(spec: AnalysisSpec) -> datetime:
    """解析の作成日時を返す（並べ替えのキー用）。

    Args:
        spec: 解析。

    Returns:
        作成日時。
    """
    return spec.created_at


def _write_bytes_atomically(path: Path, data: bytes, *, overwrite: bool) -> None:
    """一時ファイルに書いてから置き換えることで、書きかけのファイルを残さずに書き込む。

    Args:
        path: 書き込み先。親ディレクトリがなければ作る。
        data: 書き込む内容。
        overwrite: False なら、既存のファイルがあるときにエラーにする（上書き防止）。

    Raises:
        FileExistsError: ``overwrite`` が False で、書き込み先が既に存在する場合。
    """
    if not overwrite and path.exists():
        raise FileExistsError(f"既に存在するため書き込みません: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
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
