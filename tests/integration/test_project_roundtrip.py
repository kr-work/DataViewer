"""プロジェクトの作成 → 保存 → 移動 → 再オープンの結合テスト（F-01 の保存層部分）。"""

from pathlib import Path

from experiment_workbench.model import Channel, Event, Experiment, Session, Source, Trial
from experiment_workbench.storage import ProjectStore, latest_schema_version
from tests.conftest import Hierarchy


def test_project_survives_close_move_and_reopen(tmp_path: Path, hierarchy: Hierarchy) -> None:
    """閉じて別の場所へ移動したプロジェクトを開き直しても、全メタデータと親子関係が復元される。"""
    experiment, session, trial, source, channel, event = hierarchy
    second_trial = Trial(session_id=session.id, name="trial_002", tags=["failure"])

    original_root = tmp_path / "original"
    with ProjectStore.create(original_root, name="移動テスト") as store:
        store.database.insert_many([*hierarchy, second_trial])
        project = store.project

    moved_root = tmp_path / "elsewhere" / "project"
    moved_root.parent.mkdir()
    original_root.rename(moved_root)

    with ProjectStore.open(moved_root) as store:
        db = store.database
        assert store.project == project
        assert db.schema_version == latest_schema_version()
        assert db.list_entities(Experiment) == [experiment]
        assert db.list_entities(Session, experiment_id=experiment.id) == [session]
        assert db.list_entities(Trial, session_id=session.id) == [trial, second_trial]
        assert db.list_entities(Source, trial_id=trial.id) == [source]
        assert db.list_entities(Channel, source_id=source.id) == [channel]
        assert db.list_entities(Event, trial_id=trial.id) == [event]
        assert db.list_entities(Source, trial_id=second_trial.id) == []
        # 保存した相対パスは、移動後のフォルダを基準に解決される。
        assert store.resolve(source.dataset_relpath).is_relative_to(moved_root.absolute())
