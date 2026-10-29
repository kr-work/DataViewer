"""storage/database.py（SQLite メタデータ）のテスト。"""

import sqlite3
from pathlib import Path

import pytest

from experiment_workbench.model import Entity, Event, Experiment, Project, Session, Trial
from experiment_workbench.storage import database
from experiment_workbench.storage.database import (
    ENTITY_TABLES,
    EntityNotFoundError,
    IntegrityViolationError,
    MetadataDatabase,
    MigrationError,
    SchemaVersionError,
    latest_schema_version,
)
from tests.conftest import Hierarchy


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """テスト用の DB ファイルのパス。

    Args:
        tmp_path: pytest の一時ディレクトリ。

    Returns:
        まだ存在しない DB ファイルのパス。
    """
    return tmp_path / "metadata.sqlite"


def _user_version(path: Path) -> int:
    """MetadataDatabase を介さずに DB のスキーマバージョンを読む。

    Args:
        path: DB ファイルのパス。

    Returns:
        ``PRAGMA user_version`` の値。
    """
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


# --- スキーマとマイグレーション ------------------------------------------------


def test_new_database_is_migrated_to_latest_version(db_path: Path) -> None:
    """新規作成した DB は最新スキーマになる。"""
    with MetadataDatabase(db_path) as db:
        assert db.schema_version == latest_schema_version() >= 1


def test_reopening_keeps_data_and_does_not_create_backup(
    db_path: Path, hierarchy: Hierarchy
) -> None:
    """最新スキーマの DB を開き直してもデータは残り、不要なバックアップは作られない。"""
    with MetadataDatabase(db_path) as db:
        db.insert_many(hierarchy)
    with MetadataDatabase(db_path) as db:
        assert db.get(Experiment, hierarchy[0].id) == hierarchy[0]
    assert list(db_path.parent.glob("*.bak*")) == []


@pytest.mark.parametrize(("model", "table"), list(ENTITY_TABLES.items()))
def test_table_columns_match_model_fields(db_path: Path, model: type[Entity], table: str) -> None:
    """テーブルの列名とモデルのフィールド名が一致している（変換処理の前提）。"""
    with MetadataDatabase(db_path) as db:
        assert set(db.table_columns(table)) == set(model.model_fields)


def test_newer_schema_version_is_refused(db_path: Path) -> None:
    """アプリより新しいスキーマの DB は開かない（古いアプリで壊さないため）。"""
    MetadataDatabase(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute(f"PRAGMA user_version = {latest_schema_version() + 1}")
    conn.close()
    with pytest.raises(SchemaVersionError):
        MetadataDatabase(db_path)


def test_migration_backs_up_existing_database(
    db_path: Path, hierarchy: Hierarchy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """既存データのある DB を移行する前に、移行前の状態のバックアップが作られる。"""
    with MetadataDatabase(db_path) as db:
        db.insert_many(hierarchy)
    old_version = latest_schema_version()
    monkeypatch.setattr(
        database, "_MIGRATIONS", (*database._MIGRATIONS, "CREATE TABLE extra (id TEXT);")
    )

    with MetadataDatabase(db_path) as db:
        assert db.schema_version == old_version + 1
        assert db.get(Experiment, hierarchy[0].id) == hierarchy[0]

    backups = list(db_path.parent.glob(f"{db_path.name}.v{old_version}.bak*"))
    assert len(backups) == 1
    assert _user_version(backups[0]) == old_version


def test_failed_migration_is_rolled_back(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """マイグレーションが途中で失敗したら、途中までの変更も取り消される。"""
    MetadataDatabase(db_path).close()
    old_version = latest_schema_version()
    broken_script = "CREATE TABLE half_done (id TEXT);\nTHIS IS NOT SQL;"
    monkeypatch.setattr(database, "_MIGRATIONS", (*database._MIGRATIONS, broken_script))

    with pytest.raises(MigrationError):
        MetadataDatabase(db_path)

    assert _user_version(db_path) == old_version
    conn = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert "half_done" not in tables


# --- 保存と取得 ----------------------------------------------------------------


def test_every_entity_round_trips(db_path: Path, hierarchy: Hierarchy) -> None:
    """全エンティティが、保存後に同じ値で取得できる（入れ子の JSON 列を含む）。"""
    with MetadataDatabase(db_path) as db:
        db.insert_many(hierarchy)
        for entity in hierarchy:
            assert db.get(type(entity), entity.id) == entity


def test_list_entities_filters_and_keeps_insertion_order(
    db_path: Path, hierarchy: Hierarchy
) -> None:
    """一覧は条件で絞り込まれ、登録順に並ぶ。"""
    experiment, session, trial, *_ = hierarchy
    other_session = Session(experiment_id=experiment.id, name="session_02")
    later_trial = Trial(session_id=session.id, name="trial_002")
    other_trial = Trial(session_id=other_session.id, name="trial_001")
    with MetadataDatabase(db_path) as db:
        db.insert_many([experiment, session, other_session, trial, later_trial, other_trial])
        assert db.list_entities(Trial, session_id=session.id) == [trial, later_trial]
        assert len(db.list_entities(Trial)) == 3


def test_list_entities_rejects_unknown_filter(db_path: Path) -> None:
    """モデルにないフィールド名で絞り込もうとするとエラー（SQL への埋め込み防止）。"""
    with MetadataDatabase(db_path) as db, pytest.raises(ValueError, match="存在しない"):
        db.list_entities(Trial, **{"name; DROP TABLE trials": "x"})


def test_get_missing_entity_raises(db_path: Path, hierarchy: Hierarchy) -> None:
    """存在しない ID を取得するとエラーになる。"""
    with MetadataDatabase(db_path) as db, pytest.raises(EntityNotFoundError):
        db.get(Experiment, hierarchy[0].id)


def test_unregistered_model_cannot_be_stored(db_path: Path) -> None:
    """DB に保存しないモデル（Project は project.json に保存）は insert できない。"""
    with MetadataDatabase(db_path) as db, pytest.raises(TypeError):
        db.insert(Project(name="p"))


# --- 整合性制約 ----------------------------------------------------------------


def test_child_without_parent_is_rejected(db_path: Path) -> None:
    """存在しない親を参照する子は保存できない（外部キー制約）。"""
    orphan = Session(experiment_id=Experiment(name="未登録").id, name="s")
    with MetadataDatabase(db_path) as db, pytest.raises(IntegrityViolationError):
        db.insert(orphan)


def test_sibling_names_must_be_unique(db_path: Path, hierarchy: Hierarchy) -> None:
    """同じ親の下で名前が重複する子は保存できない。"""
    _, session, trial, *_ = hierarchy
    duplicate = Trial(session_id=session.id, name=trial.name)
    with MetadataDatabase(db_path) as db:
        db.insert_many(hierarchy)
        with pytest.raises(IntegrityViolationError):
            db.insert(duplicate)


def test_event_source_must_belong_to_same_trial(db_path: Path, hierarchy: Hierarchy) -> None:
    """イベント時刻の基準 Source は、同じ Trial のものでなければならない。"""
    _, session, _, source, _, event = hierarchy
    other_trial = Trial(session_id=session.id, name="trial_002")
    mismatched = Event(trial_id=other_trial.id, source_id=source.id, label="x", time_s=event.time_s)
    with MetadataDatabase(db_path) as db:
        db.insert_many([*hierarchy, other_trial])
        with pytest.raises(IntegrityViolationError):
            db.insert(mismatched)


def test_insert_many_is_all_or_nothing(db_path: Path, hierarchy: Hierarchy) -> None:
    """insert_many の途中で失敗したら、それまでの分も保存されない。"""
    experiment = hierarchy[0]
    orphan = Session(experiment_id=Experiment(name="未登録").id, name="s")
    with MetadataDatabase(db_path) as db:
        with pytest.raises(IntegrityViolationError):
            db.insert_many([experiment, orphan])
        assert db.list_entities(Experiment) == []


def _insert_then_fail(db: MetadataDatabase, entities: Hierarchy) -> None:
    """transaction() 内で書き込んだ後に例外を送出する（ロールバックの確認用）。

    Args:
        db: 書き込み先の DB。
        entities: 書き込むエンティティ（親から子の順）。

    Raises:
        RuntimeError: 常に送出する。
    """
    with db.transaction():
        db.insert_many(entities[:3])
        raise RuntimeError("中断")


def test_transaction_rolls_back_on_exception(db_path: Path, hierarchy: Hierarchy) -> None:
    """transaction() 内で例外が起きたら、ブロック内の書き込みはすべて取り消される。"""
    with MetadataDatabase(db_path) as db:
        with pytest.raises(RuntimeError, match="中断"):
            _insert_then_fail(db, hierarchy)
        assert db.list_entities(Experiment) == []
        assert db.list_entities(Trial) == []
