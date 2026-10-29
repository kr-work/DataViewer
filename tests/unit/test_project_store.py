"""storage/project_store.py（プロジェクトフォルダ）のテスト。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from experiment_workbench.model import PROJECT_FORMAT_VERSION, Project
from experiment_workbench.storage import (
    METADATA_DB_NAME,
    PROJECT_FILE_NAME,
    ProjectDirectory,
    ProjectStore,
    ProjectStoreError,
    SchemaVersionError,
)


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
