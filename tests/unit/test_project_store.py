"""storage/project_store.py（プロジェクトフォルダ）のテスト。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from experiment_workbench.model import (
    ANALYSIS_FORMAT_VERSION,
    DATASET_TIME_COLUMN,
    PROJECT_FORMAT_VERSION,
    VIEW_FORMAT_VERSION,
    AnalysisKind,
    AnalysisSpec,
    ButterworthKind,
    ButterworthParams,
    PlotPanelState,
    PlotView,
    Project,
)
from experiment_workbench.storage import (
    METADATA_DB_NAME,
    PROJECT_FILE_NAME,
    ProjectDirectory,
    ProjectStore,
    ProjectStoreError,
    SchemaVersionError,
)
from tests.conftest import Hierarchy


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """作成済み（クローズ済み）のプロジェクトフォルダ。

    Args:
        tmp_path: pytest の一時ディレクトリ。

    Returns:
        プロジェクトフォルダのパス。
    """
    root = tmp_path / "my-experiment"
    ProjectStore.create(root, name="テスト", description="説明").close()
    return root


def _rewrite_project_json(root: Path, **changes: object) -> None:
    """project.json の一部の値を書き換える（異常系テスト用）。

    Args:
        root: プロジェクトフォルダ。
        **changes: 上書きするキーと値。
    """
    path = root / PROJECT_FILE_NAME
    data = json.loads(path.read_text(encoding="utf-8")) | changes
    path.write_text(json.dumps(data), encoding="utf-8")


# --- 作成 ----------------------------------------------------------------------


def test_create_builds_folder_layout(tmp_path: Path) -> None:
    """作成すると project.json・metadata.sqlite・各サブディレクトリができる。"""
    root = tmp_path / "new-project"
    with ProjectStore.create(root, name="歩行解析") as store:
        assert store.root == root.absolute()
        assert store.project.name == "歩行解析"
    assert (root / PROJECT_FILE_NAME).is_file()
    assert (root / METADATA_DB_NAME).is_file()
    for directory in ProjectDirectory:
        assert (root / directory).is_dir()
    saved = Project.model_validate_json((root / PROJECT_FILE_NAME).read_text(encoding="utf-8"))
    assert saved.format_version == PROJECT_FORMAT_VERSION


def test_create_accepts_existing_empty_directory(tmp_path: Path) -> None:
    """既存の空ディレクトリにはプロジェクトを作成できる。"""
    root = tmp_path / "empty"
    root.mkdir()
    ProjectStore.create(root, name="p").close()
    assert (root / PROJECT_FILE_NAME).is_file()


def test_create_refuses_non_empty_directory(project_root: Path) -> None:
    """既存データを上書きしないよう、空でないフォルダには作成しない。"""
    with pytest.raises(ProjectStoreError, match="空でない"):
        ProjectStore.create(project_root, name="上書き")


def test_create_refuses_existing_file(tmp_path: Path) -> None:
    """同名のファイルがある場所には作成しない。"""
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="ファイル"):
        ProjectStore.create(path, name="p")


def test_create_with_invalid_name_leaves_nothing(tmp_path: Path) -> None:
    """名前が不正なら、フォルダを作る前にエラーになる。"""
    root = tmp_path / "invalid"
    with pytest.raises(ValidationError):
        ProjectStore.create(root, name="  ")
    assert not root.exists()


# --- オープン ------------------------------------------------------------------


def test_open_restores_project(project_root: Path) -> None:
    """作成したプロジェクトを開き直すと、同じ内容が復元される。"""
    saved = Project.model_validate_json(
        (project_root / PROJECT_FILE_NAME).read_text(encoding="utf-8")
    )
    with ProjectStore.open(project_root) as store:
        assert store.project == saved


def test_open_requires_project_file(tmp_path: Path) -> None:
    """project.json がないフォルダはプロジェクトとして開けない。"""
    with pytest.raises(ProjectStoreError, match=PROJECT_FILE_NAME):
        ProjectStore.open(tmp_path)


def test_open_does_not_silently_recreate_missing_database(project_root: Path) -> None:
    """metadata.sqlite が消えていたら、空の DB を作らずにエラーにする（データ消失を隠さない）。"""
    (project_root / METADATA_DB_NAME).unlink()
    with pytest.raises(ProjectStoreError, match=METADATA_DB_NAME):
        ProjectStore.open(project_root)
    assert not (project_root / METADATA_DB_NAME).exists()


def test_open_recreates_missing_empty_directories(project_root: Path) -> None:
    """Git や ZIP で失われがちな空のサブディレクトリは作り直す。"""
    (project_root / ProjectDirectory.EXPORTS).rmdir()
    ProjectStore.open(project_root).close()
    assert (project_root / ProjectDirectory.EXPORTS).is_dir()


def test_open_refuses_newer_format_version(project_root: Path) -> None:
    """アプリより新しい形式の project.json は開かない。新形式の未知フィールドがあっても同様。"""
    _rewrite_project_json(
        project_root, format_version=PROJECT_FORMAT_VERSION + 1, new_field="future"
    )
    with pytest.raises(SchemaVersionError):
        ProjectStore.open(project_root)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "JSON"),
        ("[1, 2]", "オブジェクト"),
        ('{"name": "p"}', "format_version"),
        ('{"format_version": true}', "format_version"),
    ],
)
def test_open_reports_malformed_project_file(
    project_root: Path, content: str, message: str
) -> None:
    """壊れた project.json は、原因が分かるエラーになる。"""
    (project_root / PROJECT_FILE_NAME).write_text(content, encoding="utf-8")
    with pytest.raises(ProjectStoreError, match=message):
        ProjectStore.open(project_root)


def test_open_reports_invalid_project_fields(project_root: Path) -> None:
    """形式バージョンが正しくても、フィールドの値が不正ならエラーになる。"""
    _rewrite_project_json(project_root, name="")
    with pytest.raises(ProjectStoreError, match="不正"):
        ProjectStore.open(project_root)


# --- パス ----------------------------------------------------------------------


def test_moved_project_resolves_paths_from_new_location(project_root: Path, tmp_path: Path) -> None:
    """フォルダを移動しても、相対パスは新しい場所を基準に解決される。"""
    moved = tmp_path / "moved" / "project"
    moved.parent.mkdir()
    project_root.rename(moved)
    with ProjectStore.open(moved) as store:
        resolved = store.resolve("datasets/trial_001/data.parquet")
        assert resolved == moved.absolute() / "datasets" / "trial_001" / "data.parquet"
        assert store.directory(ProjectDirectory.DATASETS) == moved.absolute() / "datasets"


@pytest.mark.parametrize("relpath", ["../outside.csv", "/etc/passwd", "C:/data.csv", "a\\b"])
def test_resolve_rejects_paths_outside_project(project_root: Path, relpath: str) -> None:
    """プロジェクト外を指しうる相対パスは解決しない。"""
    with ProjectStore.open(project_root) as store, pytest.raises(ValueError, match=r"."):
        store.resolve(relpath)


def test_resolve_rejects_symlink_escaping_project(project_root: Path, tmp_path: Path) -> None:
    """シンボリックリンクでプロジェクト外を指している場合も拒否する。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project_root / ProjectDirectory.ORIGINALS / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("この環境ではシンボリックリンクを作成できない（Windows の権限設定など）")
    with ProjectStore.open(project_root) as store, pytest.raises(ValueError, match="外"):
        store.resolve("originals/link/file.csv")


# --- 元データ・正規化データ ------------------------------------------------------


def test_write_original_returns_hash_and_refuses_overwrite(project_root: Path) -> None:
    """元データのコピーは保存した内容の SHA-256 を返し、既存のファイルは上書きしない。"""
    data = "時刻,値\n0,1\n".encode()
    with ProjectStore.open(project_root) as store:
        digest = store.write_original("originals/abc/data.csv", data)
        assert digest == hashlib.sha256(data).hexdigest()
        with pytest.raises(FileExistsError):
            store.write_original("originals/abc/data.csv", b"other")
        assert store.resolve("originals/abc/data.csv").read_bytes() == data


def test_dataset_round_trip_turns_null_into_nan(project_root: Path, hierarchy: Hierarchy) -> None:
    """正規化データを書いて読み戻すと同じ値になり、欠損（null）は NaN として読める。"""
    source, channel = hierarchy[3], hierarchy[4]
    frame = pl.DataFrame(
        {DATASET_TIME_COLUMN: [0.0, 0.1, 0.2], channel.name: [1.0, None, float("nan")]}
    )
    with ProjectStore.open(project_root) as store:
        store.write_dataset(source.dataset_relpath, frame)
        data = store.read_channel_data(source, channel)
    np.testing.assert_array_equal(data.time_s, [0.0, 0.1, 0.2])
    assert data.values[0] == 1.0
    assert np.isnan(data.values[1:]).all()


def test_read_channel_data_reports_missing_column(project_root: Path, hierarchy: Hierarchy) -> None:
    """正規化データに列がない場合は、原因が分かるエラーになる。"""
    source, channel = hierarchy[3], hierarchy[4]
    with ProjectStore.open(project_root) as store:
        store.write_dataset(source.dataset_relpath, pl.DataFrame({DATASET_TIME_COLUMN: [0.0]}))
        with pytest.raises(ProjectStoreError, match=channel.name):
            store.read_channel_data(source, channel)


def test_remove_file_cleans_empty_folder_but_keeps_top_level(project_root: Path) -> None:
    """書きかけのファイルを消すと空になったフォルダも消えるが、直下のサブディレクトリは残る。"""
    with ProjectStore.open(project_root) as store:
        store.write_original("originals/abc/data.csv", b"x")
        store.remove_file("originals/abc/data.csv")
        assert not (project_root / "originals" / "abc").exists()
        assert (project_root / "originals").is_dir()
        store.remove_file("originals/abc/data.csv")  # 存在しなくてもエラーにしない


# --- 表示設定 ------------------------------------------------------------------


def test_view_round_trip(project_root: Path) -> None:
    """表示設定を保存して読み込むと同じ内容になる。"""
    view = PlotView(
        panels=[PlotPanelState(series_ids=[uuid4()], y_range=(-1.0, 1.0)), PlotPanelState()],
        x_range=(10.0, 20.0),
        cursor_time_s=12.5,
    )
    with ProjectStore.open(project_root) as store:
        assert store.load_view() is None
        store.save_view(view)
        assert store.load_view() == view


def test_load_view_rejects_unsafe_name(project_root: Path) -> None:
    """表示設定の名前にパスの区切りなどは使えない（プロジェクト外のファイルを読ませない）。"""
    with ProjectStore.open(project_root) as store, pytest.raises(ValueError, match="名前"):
        store.load_view("../project")


def test_load_view_reports_broken_or_newer_file(project_root: Path) -> None:
    """壊れた表示設定はエラー、新しい形式の表示設定はバージョンのエラーになる。"""
    views = project_root / ProjectDirectory.VIEWS
    with ProjectStore.open(project_root) as store:
        (views / "main.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(ProjectStoreError, match="JSON"):
            store.load_view()
        (views / "main.json").write_text(
            json.dumps({"format_version": VIEW_FORMAT_VERSION + 1}), encoding="utf-8"
        )
        with pytest.raises(SchemaVersionError):
            store.load_view()


@pytest.mark.parametrize("explicit_version", [True, False])
def test_load_view_upgrades_version_1(project_root: Path, *, explicit_version: bool) -> None:
    """形式 v1 の表示設定（パネルの channel_ids）は、v2 の series_ids に読み替える。

    形式バージョンを書いていない最初期のファイルも v1 として扱う。
    """
    channel_id = uuid4()
    raw: dict[str, object] = {"panels": [{"channel_ids": [str(channel_id)], "y_range": [0, 1]}]}
    if explicit_version:
        raw["format_version"] = 1
    (project_root / ProjectDirectory.VIEWS / "main.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    with ProjectStore.open(project_root) as store:
        view = store.load_view()
    assert view is not None
    assert view.format_version == VIEW_FORMAT_VERSION
    assert view.panels == [PlotPanelState(series_ids=[channel_id], y_range=(0.0, 1.0))]


# --- 保存した解析と計算結果のキャッシュ -----------------------------------------


def _lowpass_spec(name: str, created_at: datetime) -> AnalysisSpec:
    """保存のテストに使う低域通過フィルタの解析を作る。

    Args:
        name: 解析の名前。
        created_at: 作成日時（読み込み順の確認に使う）。

    Returns:
        解析。
    """
    return AnalysisSpec(
        name=name,
        created_at=created_at,
        input_channel_id=uuid4(),
        kind=AnalysisKind.BUTTERWORTH,
        butterworth=ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0),
    )


def test_analyses_round_trip_in_creation_order(project_root: Path) -> None:
    """保存した解析は作成日時の順に読み込め、同じ ID で保存し直すと置き換わる。"""
    older = _lowpass_spec("older", datetime(2026, 1, 1, tzinfo=UTC))
    newer = _lowpass_spec("newer", datetime(2026, 1, 2, tzinfo=UTC))
    with ProjectStore.open(project_root) as store:
        assert store.load_analyses() == ([], [])
        store.save_analysis(newer)
        store.save_analysis(older)
        assert store.load_analyses() == ([older, newer], [])
        renamed = older.model_copy(update={"name": "renamed"})
        store.save_analysis(renamed)
        specs, problems = store.load_analyses()
    assert specs == [renamed, newer]
    assert problems == []


def test_load_analyses_skips_broken_and_newer_files_with_reason(project_root: Path) -> None:
    """壊れた・新しい形式の解析ファイルは飛ばして理由を返し、他の解析は読み込む。"""
    spec = _lowpass_spec("ok", datetime(2026, 1, 1, tzinfo=UTC))
    pipelines = project_root / ProjectDirectory.PIPELINES
    with ProjectStore.open(project_root) as store:
        store.save_analysis(spec)
        (pipelines / "broken.json").write_text("{broken", encoding="utf-8")
        newer = json.loads(spec.model_copy(update={"id": uuid4()}).model_dump_json())
        newer["format_version"] = ANALYSIS_FORMAT_VERSION + 1
        newer["future_field"] = True
        (pipelines / "newer.json").write_text(json.dumps(newer), encoding="utf-8")
        specs, problems = store.load_analyses()
    assert specs == [spec]
    assert len(problems) == 2
    assert any("broken.json" in problem for problem in problems)
    assert any("newer.json" in problem and "新しい形式" in problem for problem in problems)


def test_derived_cache_keeps_only_latest_fingerprint(project_root: Path) -> None:
    """計算結果のキャッシュは指紋ごとに読み書きでき、条件が変わると古いキャッシュを消す。"""
    analysis_id = uuid4()
    first = pl.DataFrame({DATASET_TIME_COLUMN: [0.0, 1.0], "output": [1.0, 2.0]})
    second = pl.DataFrame({DATASET_TIME_COLUMN: [0.0, 1.0], "output": [3.0, 4.0]})
    cache_dir = project_root / ProjectDirectory.DERIVED / str(analysis_id)
    with ProjectStore.open(project_root) as store:
        assert store.load_derived(analysis_id, "aaaa") is None
        store.save_derived(analysis_id, "aaaa", first)
        loaded = store.load_derived(analysis_id, "aaaa")
        assert loaded is not None
        assert loaded.equals(first)
        store.save_derived(analysis_id, "bbbb", second)
        assert store.load_derived(analysis_id, "aaaa") is None
        assert sorted(path.name for path in cache_dir.iterdir()) == ["bbbb.parquet"]


def test_broken_derived_cache_is_treated_as_missing(project_root: Path) -> None:
    """読めないキャッシュはエラーにせず、ないものとして扱う（再計算させる）。"""
    analysis_id = uuid4()
    cache_dir = project_root / ProjectDirectory.DERIVED / str(analysis_id)
    cache_dir.mkdir(parents=True)
    (cache_dir / "aaaa.parquet").write_bytes(b"not parquet")
    with ProjectStore.open(project_root) as store:
        assert store.load_derived(analysis_id, "aaaa") is None


def test_delete_analysis_removes_spec_and_cache_but_not_others(project_root: Path) -> None:
    """解析を削除すると条件のファイルとキャッシュが消え、他の解析は残る。"""
    removed = _lowpass_spec("removed", datetime(2026, 1, 1, tzinfo=UTC))
    kept = _lowpass_spec("kept", datetime(2026, 1, 2, tzinfo=UTC))
    frame = pl.DataFrame({DATASET_TIME_COLUMN: [0.0], "output": [1.0]})
    derived = project_root / ProjectDirectory.DERIVED
    with ProjectStore.open(project_root) as store:
        for spec in (removed, kept):
            store.save_analysis(spec)
            store.save_derived(spec.id, "aaaa", frame)
        store.delete_analysis(removed.id)
        assert store.load_analyses() == ([kept], [])
        store.delete_analysis(uuid4())  # 存在しない解析を削除してもエラーにしない
    assert not (derived / str(removed.id)).exists()
    assert (derived / str(kept.id) / "aaaa.parquet").is_file()
