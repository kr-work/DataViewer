"""SQLite によるメタデータ（実験・試技・Source・Channel・Event）の永続化。

スキーマのバージョン管理とマイグレーション方針（docs/DECISIONS.md にも記録）:
    1. スキーマバージョンは SQLite 標準の ``PRAGMA user_version`` に保存する。
    2. ``_MIGRATIONS[i]`` はバージョン i から i+1 へ上げる SQL。追加のみ行い、既存の要素は
       書き換えない（配布済みの DB を同じ手順で再現できるようにするため）。
    3. 各マイグレーションは 1 トランザクションで実行し、失敗したら ROLLBACK して元の
       バージョンのまま例外を送出する。
    4. 既存データのある DB（バージョン 1 以上）を移行する前に、バックアップファイルを作る。
    5. アプリより新しいバージョンの DB は開かない（古いアプリで壊さないため）。

このモジュールは Qt に依存しない。
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from uuid import UUID

from experiment_workbench.model import Channel, Entity, Event, Experiment, Session, Source, Trial

# ---------------------------------------------------------------------------
# スキーマ定義
# ---------------------------------------------------------------------------

# 各テーブルの列名は対応するモデルのフィールド名と一致させる（tests で一致を検査している）。
# 日時は ISO 8601 文字列、UUID は文字列、入れ子のモデル・リスト・辞書は JSON 文字列で保存する。
_SCHEMA_V1 = """
CREATE TABLE experiments (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE sessions (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    experiment_id TEXT NOT NULL REFERENCES experiments (id),
    name          TEXT NOT NULL,
    recorded_on   TEXT,
    notes         TEXT NOT NULL,
    UNIQUE (experiment_id, name)
);

CREATE TABLE trials (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions (id),
    name       TEXT NOT NULL,
    tags       TEXT NOT NULL,  -- JSON 配列
    notes      TEXT NOT NULL,
    UNIQUE (session_id, name)
);

CREATE TABLE sources (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    trial_id            TEXT NOT NULL REFERENCES trials (id),
    name                TEXT NOT NULL,
    format              TEXT NOT NULL,
    original_filename   TEXT NOT NULL,
    original_relpath    TEXT NOT NULL UNIQUE,
    original_sha256     TEXT NOT NULL,
    original_size_bytes INTEGER NOT NULL,
    dataset_relpath     TEXT NOT NULL UNIQUE,
    time_column         TEXT,
    time_base           TEXT NOT NULL,  -- JSON オブジェクト（TimeBase）
    UNIQUE (trial_id, name),
    -- events の複合外部キー (source_id, trial_id) の参照先として必要
    UNIQUE (id, trial_id)
);

CREATE TABLE channels (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    source_id     TEXT NOT NULL REFERENCES sources (id),
    name          TEXT NOT NULL,
    original_name TEXT NOT NULL,
    unit          TEXT,
    original_unit TEXT,
    sample_rate   TEXT NOT NULL,  -- JSON オブジェクト（SampleRate）
    description   TEXT NOT NULL,
    UNIQUE (source_id, name)
);

CREATE TABLE events (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    trial_id          TEXT NOT NULL REFERENCES trials (id),
    source_id         TEXT NOT NULL,
    label             TEXT NOT NULL,
    time_s            REAL NOT NULL,
    detection_method  TEXT NOT NULL,
    detection_params  TEXT NOT NULL,  -- JSON オブジェクト
    note              TEXT NOT NULL,
    -- イベント時刻の基準 Source は、必ず同じ Trial に属していなければならない
    FOREIGN KEY (source_id, trial_id) REFERENCES sources (id, trial_id)
);

CREATE INDEX idx_sessions_experiment ON sessions (experiment_id);
CREATE INDEX idx_trials_session ON trials (session_id);
CREATE INDEX idx_sources_trial ON sources (trial_id);
CREATE INDEX idx_channels_source ON channels (source_id);
CREATE INDEX idx_events_trial ON events (trial_id);
CREATE INDEX idx_events_source ON events (source_id, trial_id);
"""

# マイグレーション一覧。_MIGRATIONS[i] でバージョン i → i+1 に上げる。追加のみ行うこと。
_MIGRATIONS: tuple[str, ...] = (_SCHEMA_V1,)

# モデルクラス → テーブル名。ここに登録したモデルだけを保存できる。
ENTITY_TABLES: Mapping[type[Entity], str] = {
    Experiment: "experiments",
    Session: "sessions",
    Trial: "trials",
    Source: "sources",
    Channel: "channels",
    Event: "events",
}

# JSON 文字列として保存する列。読み出し時に json.loads してからモデルへ渡す。
_JSON_COLUMNS: Mapping[str, frozenset[str]] = {
    "experiments": frozenset(),
    "sessions": frozenset(),
    "trials": frozenset({"tags"}),
    "sources": frozenset({"time_base"}),
    "channels": frozenset({"sample_rate"}),
    "events": frozenset({"detection_params"}),
}


def latest_schema_version() -> int:
    """このアプリが扱える最新のスキーマバージョンを返す。

    定数ではなく関数にしているのは、テストでマイグレーション一覧を差し替えたときにも
    正しい値を返すため。

    Returns:
        マイグレーション数（= 最新のスキーマバージョン）。
    """
    return len(_MIGRATIONS)


# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """保存・読み込み処理で発生するエラーの基底クラス。"""


class SchemaVersionError(StorageError):
    """保存形式・スキーマのバージョンがこのアプリと互換でない。"""


class MigrationError(StorageError):
    """スキーマのマイグレーションに失敗した（DB は元のバージョンのまま）。"""


class EntityNotFoundError(StorageError):
    """指定 ID のエンティティが存在しない。"""


class IntegrityViolationError(StorageError):
    """外部キー・一意制約などの整合性制約に違反した。"""


# ---------------------------------------------------------------------------
# MetadataDatabase
# ---------------------------------------------------------------------------


class MetadataDatabase:
    """metadata.sqlite への接続と、エンティティの保存・取得を担うクラス。

    開いた時点でスキーマを最新バージョンへ移行する。生成後は ``close()`` するか、
    ``with`` 文で使うこと。

    使用例::

        with MetadataDatabase(path) as db:
            db.insert(experiment)
            trials = db.list_entities(Trial, session_id=session.id)

    Attributes:
        path: SQLite ファイルのパス。
    """

    def __init__(self, path: Path) -> None:
        """DB を開き（なければ作成し）、スキーマを最新バージョンへ移行する。

        Args:
            path: SQLite ファイルのパス。親ディレクトリは存在している必要がある。

        Raises:
            SchemaVersionError: DB のスキーマがこのアプリより新しい場合。
            MigrationError: マイグレーションに失敗した場合。
        """
        self.path = path
        # isolation_level=None で sqlite3 モジュールの暗黙トランザクションを無効にし、
        # BEGIN/COMMIT を transaction() で明示的に管理する。
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        try:
            # SQLite の外部キー制約は接続ごとに有効化が必要。
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    # --- 接続管理 ----------------------------------------------------------

    def close(self) -> None:
        """DB 接続を閉じる。"""
        self._conn.close()

    def __enter__(self) -> "MetadataDatabase":
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
        """``with`` 文の終了時に接続を閉じる。

        Args:
            exc_type: 発生した例外の型（なければ None）。
            exc: 発生した例外（なければ None）。
            tb: トレースバック（なければ None）。
        """
        self.close()

    @property
    def schema_version(self) -> int:
        """現在の DB のスキーマバージョン（``PRAGMA user_version``）。"""
        row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """複数の書き込みを 1 つのトランザクションにまとめる。

        例外が発生した場合は ROLLBACK し、例外をそのまま再送出する。
        既にトランザクション中の場合は外側のトランザクションに参加する（入れ子にしない）。

        Yields:
            なし。ブロック内で insert などを呼ぶ。
        """
        if self._conn.in_transaction:
            yield
            return
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    # --- マイグレーション --------------------------------------------------

    def _migrate(self) -> None:
        """スキーマを最新バージョンへ順に移行する。

        Raises:
            SchemaVersionError: DB のスキーマがこのアプリより新しい場合。
            MigrationError: いずれかのマイグレーションに失敗した場合。
        """
        current = self.schema_version
        latest = latest_schema_version()
        if current > latest:
            raise SchemaVersionError(
                f"{self.path} のスキーマバージョン {current} は、このアプリが扱える"
                f"バージョン {latest} より新しいため開けません。アプリを更新してください。"
            )
        if current == latest:
            return
        # 新規作成（バージョン 0）ではデータがないためバックアップ不要。
        if current > 0:
            self._backup(current)
        for version in range(current, latest):
            self._apply_migration(version)

    def _apply_migration(self, version: int) -> None:
        """バージョン ``version`` → ``version + 1`` のマイグレーションを 1 つ適用する。

        executescript は暗黙のトランザクション制御をしないため、BEGIN/COMMIT と
        user_version の更新を同じスクリプトに含めて原子的に実行する。

        Args:
            version: 適用前のスキーマバージョン。

        Raises:
            MigrationError: SQL の実行に失敗した場合（ROLLBACK 済み）。
        """
        target = version + 1
        script = f"BEGIN;\n{_MIGRATIONS[version]}\nPRAGMA user_version = {target};\nCOMMIT;"
        try:
            self._conn.executescript(script)
        except sqlite3.Error as exc:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise MigrationError(
                f"スキーマをバージョン {version} から {target} へ移行できませんでした: {exc}"
            ) from exc

    def _backup(self, current_version: int) -> Path:
        """マイグレーション前に DB ファイルのバックアップを作る。

        既存のバックアップは上書きしない（過去の状態を失わないため）。同名がある場合は
        連番を付ける。

        Args:
            current_version: 移行前のスキーマバージョン（ファイル名に含める）。

        Returns:
            作成したバックアップファイルのパス。
        """
        base = self.path.with_name(f"{self.path.name}.v{current_version}.bak")
        backup_path = base
        counter = 1
        while backup_path.exists():
            backup_path = base.with_name(f"{base.name}.{counter}")
            counter += 1
        # Connection.backup は接続中でも一貫したスナップショットを取れる。
        # sqlite3 の接続を with に渡しても close されないため、明示的に閉じる。
        dest = sqlite3.connect(backup_path)
        try:
            self._conn.backup(dest)
        finally:
            dest.close()
        return backup_path

    # --- エンティティの保存・取得 -------------------------------------------

    def insert(self, entity: Entity) -> None:
        """エンティティを 1 件保存する。

        Args:
            entity: 保存するエンティティ（``ENTITY_TABLES`` に登録されたモデル）。

        Raises:
            IntegrityViolationError: ID の重複、親の不在、名前の重複などで制約に違反した場合。
            TypeError: 保存対象として登録されていないモデルの場合。
        """
        self.insert_many([entity])

    def insert_many(self, entities: Iterable[Entity]) -> None:
        """複数のエンティティを 1 つのトランザクションで保存する。

        途中で失敗した場合はすべて取り消す（Source とその Channel のように、
        まとめて登録しないと整合しないデータのため）。

        Args:
            entities: 保存するエンティティ。親から子の順に並べること（外部キー制約のため）。

        Raises:
            IntegrityViolationError: 制約に違反した場合（全件取り消し済み）。
            TypeError: 保存対象として登録されていないモデルが含まれる場合。
        """
        try:
            with self.transaction():
                for entity in entities:
                    table = _table_for(type(entity))
                    row = _to_row(entity)
                    columns = ", ".join(row)
                    placeholders = ", ".join(f":{name}" for name in row)
                    self._conn.execute(
                        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", row
                    )
        except sqlite3.IntegrityError as exc:
            raise IntegrityViolationError(f"整合性制約に違反したため保存できません: {exc}") from exc

    def get[EntityT: Entity](self, model: type[EntityT], entity_id: UUID) -> EntityT:
        """ID を指定してエンティティを 1 件取得する。

        Args:
            model: 取得するモデルクラス（例: ``Trial``）。
            entity_id: エンティティの ID。

        Returns:
            検証済みのモデルインスタンス。

        Raises:
            EntityNotFoundError: 指定 ID のエンティティが存在しない場合。
        """
        table = _table_for(model)
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (str(entity_id),)
        ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"{model.__name__} (id={entity_id}) が見つかりません")
        return _from_row(model, table, row)

    def list_entities[EntityT: Entity](
        self, model: type[EntityT], **filters: UUID | str
    ) -> list[EntityT]:
        """条件に一致するエンティティを登録順に取得する。

        Args:
            model: 取得するモデルクラス（例: ``Trial``）。
            **filters: 列名 = 値 の完全一致条件（例: ``session_id=session.id``）。
                複数指定した場合は AND 条件。省略時は全件。

        Returns:
            検証済みのモデルインスタンスのリスト（登録順）。

        Raises:
            ValueError: モデルに存在しないフィールド名を条件に指定した場合。
        """
        table = _table_for(model)
        # 列名は SQL に直接埋め込むため、モデルのフィールド名であることを必ず確認する。
        unknown = sorted(set(filters) - set(model.model_fields))
        if unknown:
            raise ValueError(f"{model.__name__} に存在しないフィールドです: {unknown}")
        where = " AND ".join(f"{name} = :{name}" for name in filters)
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + " ORDER BY rowid"
        params = {name: str(value) for name, value in filters.items()}
        return [_from_row(model, table, row) for row in self._conn.execute(sql, params)]

    def table_columns(self, table: str) -> list[str]:
        """テーブルの列名一覧を返す（スキーマとモデルの一致検査用）。

        Args:
            table: テーブル名。``ENTITY_TABLES`` に登録されたテーブルのみ指定できる。

        Returns:
            定義順の列名リスト。

        Raises:
            ValueError: 登録されていないテーブル名の場合（SQL に直接埋め込むため確認する）。
        """
        if table not in ENTITY_TABLES.values():
            raise ValueError(f"未登録のテーブルです: {table}")
        return [str(row["name"]) for row in self._conn.execute(f"PRAGMA table_info({table})")]


# ---------------------------------------------------------------------------
# モデル ⇔ 行 の変換（モジュール内部用）
# ---------------------------------------------------------------------------


def _table_for(model: type[Entity]) -> str:
    """モデルクラスに対応するテーブル名を返す。

    Args:
        model: モデルクラス。

    Returns:
        テーブル名。

    Raises:
        TypeError: 保存対象として登録されていないモデルの場合。
    """
    try:
        return ENTITY_TABLES[model]
    except KeyError:
        raise TypeError(f"{model.__name__} は DB に保存できるモデルではありません") from None


def _to_row(entity: Entity) -> dict[str, object]:
    """モデルを SQLite に渡せる値の辞書へ変換する。

    ``model_dump(mode="json")`` で UUID・日時・列挙型を文字列にし、
    入れ子のモデル・リスト・辞書は JSON 文字列にする。

    Args:
        entity: 変換するエンティティ。

    Returns:
        列名 → 値 の辞書。
    """
    dumped = entity.model_dump(mode="json")
    return {
        name: json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list) else value
        for name, value in dumped.items()
    }


def _from_row[EntityT: Entity](model: type[EntityT], table: str, row: sqlite3.Row) -> EntityT:
    """SQLite の行をモデルへ変換する。

    JSON 列を復元してから pydantic で検証するため、DB が手で書き換えられていても
    不正な値は ValidationError として検出される。

    Args:
        model: 変換先のモデルクラス。
        table: 行を取得したテーブル名（JSON 列の判定に使う）。
        row: SQLite の行。

    Returns:
        検証済みのモデルインスタンス。
    """
    json_columns = _JSON_COLUMNS[table]
    # sqlite3.Row は反復すると「値」を返すため、列名を得るには keys() が必要。
    column_names = row.keys()
    data = {
        name: json.loads(row[name]) if name in json_columns else row[name] for name in column_names
    }
    return model.model_validate(data)
