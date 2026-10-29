"""importer.py（データファイルの取り込み）のテスト。

合成データ（正常な試技ファイルと、目録に異常が書かれた失敗ファイル）を使う。
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiment_workbench.importer import DataImporter, ImportBlockedError, ImportTargetError
from experiment_workbench.model import (
    DATASET_TIME_COLUMN,
    Channel,
    ChannelImportSpec,
    DataIssueKind,
    Experiment,
    ImportRequest,
    IssueSeverity,
    ReadOptions,
    Session,
    Source,
    SourceFormat,
    SyntheticBrokenFileRecord,
    TimeAxisKind,
    TimeUnit,
    Trial,
)
from experiment_workbench.storage import ProjectDirectory, ProjectStore, StorageError
from tests.conftest import SyntheticDataset, high_rate_request, low_rate_request


def _sha256(path: Path) -> str:
    """ファイルの SHA-256 を返す。

    Args:
        path: ファイル。

    Returns:
        SHA-256（16 進小文字）。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_under(directory: Path) -> list[Path]:
    """ディレクトリ以下のファイルの一覧を返す。

    Args:
        directory: ディレクトリ。

    Returns:
        ファイルのパスのリスト。
    """
    return [path for path in directory.rglob("*") if path.is_file()]


def _broken_request(record: SyntheticBrokenFileRecord, columns: list[str]) -> ImportRequest:
    """失敗ファイルを取り込む指定を作る（時刻列 time_s と、それ以外の列をチャンネルにする）。

    Args:
        record: 失敗ファイルの記録。
        columns: プレビューで分かった列名。

    Returns:
        取り込みの指定。
    """
    channels = [name for name in dict.fromkeys(columns) if name != "time_s" and name.strip()]
    return ImportRequest(
        file_format=SourceFormat.CSV,
        read_options=ReadOptions(encoding=record.encoding),
        time_column="time_s",
        channels=[ChannelImportSpec(column=name) for name in channels or ["value_a"]],
        experiment_name="合成",
        session_name="失敗データ",
        trial_name="t",
        source_name=Path(record.relpath).stem,
    )


# --- プレビュー ----------------------------------------------------------------


def test_preview_shows_columns_rows_and_numeric_kinds(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """プレビューで列名・数値かどうか・先頭行が分かり、正常なファイルには異常がない。"""
    data, _ = synthetic_dataset
    preview = DataImporter(project).preview(
        data / "trial_001/high_rate.csv", SourceFormat.CSV, ReadOptions(), n_rows=5
    )
    assert [column.name for column in preview.columns] == [
        "time_s",
        "accel",
        "true_accel",
        "true_velocity",
        "true_position",
    ]
    assert all(column.is_numeric for column in preview.columns)
    assert len(preview.rows) == 5
    assert preview.issues == []


def test_preview_treats_spaced_numbers_as_numeric(project: ProjectStore, tmp_path: Path) -> None:
    """区切り文字の後に空白がある数値（"1.0, 2.0"）も数値の列として扱う。列名は書かれたまま。"""
    path = tmp_path / "spaced.csv"
    path.write_text("time_s, value\n0.0, 1.5\n0.1, 2.5\n", encoding="utf-8")
    preview = DataImporter(project).preview(path, SourceFormat.CSV, ReadOptions())
    assert [(column.name, column.is_numeric) for column in preview.columns] == [
        ("time_s", True),
        (" value", True),
    ]


def test_preview_of_shift_jis_file_with_correct_encoding(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """Shift_JIS のファイルは、文字コードに cp932 を指定すれば日本語の列名を正しく読める。"""
    data, _ = synthetic_dataset
    importer = DataImporter(project)
    path = data / "broken/shift_jis_header.csv"
    wrong = importer.preview(path, SourceFormat.CSV, ReadOptions())
    assert [issue.kind for issue in wrong.issues] == [DataIssueKind.NON_UTF8_ENCODING]
    right = importer.preview(path, SourceFormat.CSV, ReadOptions(encoding="cp932"))
    assert [column.name for column in right.columns] == ["時刻[s]", "値A", "値B"]
    assert right.issues == []


# --- 失敗データ ----------------------------------------------------------------


@pytest.mark.parametrize(
    "file_name",
    [
        "empty.csv",
        "header_only.csv",
        "duplicate_columns.csv",
        "shift_jis_header.csv",
        "nan_values.csv",
        "timestamp_reversed.csv",
        "duplicate_timestamps.csv",
    ],
)
def test_broken_files_report_expected_issues(
    project: ProjectStore, synthetic_dataset: SyntheticDataset, file_name: str
) -> None:
    """失敗データごとに、目録に書かれた異常が検出される（エラーなら取り込まず、何も書かない）。

    目録にない異常が追加で報告されることはある（時刻の逆行の直後は時刻が大きく進むので
    「時刻の飛び」も報告されるなど）。
    """
    data, manifest = synthetic_dataset
    record = next(r for r in manifest.broken_files if Path(r.relpath).name == file_name)
    importer = DataImporter(project)
    path = data / record.relpath
    preview = importer.preview(path, SourceFormat.CSV, ReadOptions(encoding="utf-8"))
    request = _broken_request(record, [column.name for column in preview.columns])
    expected = {issue.kind for issue in record.expected_issues}
    blocking = {
        DataIssueKind.EMPTY_FILE,
        DataIssueKind.HEADER_ONLY,
        DataIssueKind.DUPLICATE_COLUMN,
        DataIssueKind.NON_UTF8_ENCODING,
    }
    if expected & blocking:
        with pytest.raises(ImportBlockedError) as excinfo:
            importer.import_file(path, request.model_copy(update={"read_options": ReadOptions()}))
        detected = {issue.kind for issue in excinfo.value.issues}
        assert expected <= detected
        assert all(
            issue.severity is IssueSeverity.ERROR
            for issue in excinfo.value.issues
            if issue.kind in expected
        )
        assert _files_under(project.directory(ProjectDirectory.ORIGINALS)) == []
        assert _files_under(project.directory(ProjectDirectory.DATASETS)) == []
    else:
        prepared = importer.import_file(path, request)
        detected = {issue.kind for issue in prepared.source.issues}
        assert expected <= detected
        assert all(issue.severity is IssueSeverity.WARNING for issue in prepared.source.issues)


def test_reversed_timestamps_are_kept_in_original_order(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """時刻が逆行していても並べ替えず、ファイルの順のまま正規化データに保存する。"""
    data, manifest = synthetic_dataset
    record = next(r for r in manifest.broken_files if r.relpath.endswith("timestamp_reversed.csv"))
    importer = DataImporter(project)
    request = _broken_request(record, ["time_s", "value_a", "value_b"])
    prepared = importer.import_file(data / record.relpath, request)
    channel = prepared.channels[0]
    stored = project.read_channel_data(prepared.source, channel)
    original = pl.read_csv(data / record.relpath)
    np.testing.assert_array_equal(stored.time_s, original["time_s"].to_numpy())
    assert np.any(np.diff(stored.time_s) < 0)


# --- 正常な取り込み ------------------------------------------------------------


def test_import_copies_original_and_normalizes_without_modifying_source(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """取り込みで元ファイルは変わらず、同じ内容のコピーとハッシュ、正規化データが保存される。"""
    data, _ = synthetic_dataset
    path = data / "trial_001/high_rate.csv"
    before = _sha256(path)
    prepared = DataImporter(project).import_file(path, high_rate_request())

    assert _sha256(path) == before
    source = prepared.source
    copy = project.resolve(source.original_relpath)
    assert copy.read_bytes() == path.read_bytes()
    assert source.original_sha256 == before
    assert source.original_size_bytes == path.stat().st_size
    assert source.read_options == ReadOptions()

    dataset = pl.read_parquet(project.resolve(source.dataset_relpath))
    assert dataset.columns == [DATASET_TIME_COLUMN, "accel", "true_velocity"]
    original = pl.read_csv(path)
    np.testing.assert_array_equal(dataset[DATASET_TIME_COLUMN], original["time_s"])
    np.testing.assert_array_equal(dataset["accel"], original["accel"])


def test_import_registers_hierarchy_and_metadata(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """実験〜チャンネルが登録され、時間軸・単位・推定周波数が記録される。"""
    data, _ = synthetic_dataset
    prepared = DataImporter(project).import_file(
        data / "trial_001/high_rate.csv", high_rate_request(shares_trial_clock=True)
    )
    database = project.database
    experiment = database.list_entities(Experiment)[0]
    session = database.list_entities(Session, experiment_id=experiment.id)[0]
    trial = database.list_entities(Trial, session_id=session.id)[0]
    source = database.get(Source, prepared.source.id)
    assert (experiment.name, session.name, trial.name) == ("合成", "s1", "trial_001")
    assert source.trial_id == trial.id
    assert source.time_base.kind is TimeAxisKind.RELATIVE_SECONDS
    assert source.time_base.original_unit is TimeUnit.SECOND
    assert source.time_base.shares_trial_clock
    channels = database.list_entities(Channel, source_id=source.id)
    assert [(c.name, c.unit) for c in channels] == [("accel", "m/s^2"), ("true_velocity", "m/s")]
    assert channels[0].sample_rate.estimated_hz == pytest.approx(1000.0, rel=1e-6)


def test_millisecond_time_is_converted_to_seconds(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """ミリ秒の時刻列は秒に換算して保存し、元の単位はメタデータに残す。"""
    data, _ = synthetic_dataset
    path = data / "trial_001/low_rate.csv"
    prepared = DataImporter(project).import_file(path, low_rate_request())
    stored = project.read_channel_data(prepared.source, prepared.channels[0])
    original = pl.read_csv(path)
    np.testing.assert_allclose(stored.time_s, original["time_ms"].to_numpy() / 1000.0)
    assert prepared.source.time_base.original_unit is TimeUnit.MILLISECOND
    assert prepared.channels[0].sample_rate.estimated_hz == pytest.approx(100.0, rel=1e-6)


def test_missing_segment_and_nan_are_recorded_as_warnings(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """欠損区間（2 試技目）と NaN（3 試技目）は警告として記録し、データはそのまま保存する。"""
    data, _ = synthetic_dataset
    importer = DataImporter(project)
    gap = importer.import_file(data / "trial_002/high_rate.csv", high_rate_request("trial_002"))
    assert [issue.kind for issue in gap.source.issues] == [DataIssueKind.MISSING_SEGMENT]
    nan = importer.import_file(data / "trial_003/low_rate.csv", low_rate_request("trial_003"))
    assert [(issue.kind, issue.column) for issue in nan.source.issues] == [
        (DataIssueKind.NAN_VALUES, "position")
    ]
    stored = project.read_channel_data(nan.source, nan.channels[0])
    assert np.isnan(stored.values).sum() > 0


def test_file_without_time_column_uses_sample_interval(
    project: ProjectStore, tmp_path: Path
) -> None:
    """時刻列のないファイルは、指定したサンプル間隔とサンプル番号から時刻を作る。"""
    path = tmp_path / "no_time.csv"
    path.write_text("value\n1\n2\n3\n4\n", encoding="utf-8")
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column=None,
        sample_interval_s=0.5,
        channels=[ChannelImportSpec(column="value")],
        experiment_name="e",
        session_name="s",
        trial_name="t",
        source_name="no_time",
    )
    prepared = DataImporter(project).import_file(path, request)
    stored = project.read_channel_data(prepared.source, prepared.channels[0])
    np.testing.assert_allclose(stored.time_s, [0.0, 0.5, 1.0, 1.5])
    assert prepared.source.time_base.kind is TimeAxisKind.SAMPLE_INDEX
    assert prepared.source.time_base.sample_interval_s == 0.5
    assert prepared.channels[0].sample_rate.nominal_hz == pytest.approx(2.0)


def test_blank_lines_are_removed_with_warning(project: ProjectStore, tmp_path: Path) -> None:
    """空行は値を含まないため取り除き、その事実を警告として記録する（行番号はファイル上の位置）。"""
    path = tmp_path / "blank.csv"
    path.write_text("t,v\n0,1\n\n1,2\n2,3\n", encoding="utf-8")
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="t",
        channels=[ChannelImportSpec(column="v")],
        experiment_name="e",
        session_name="s",
        trial_name="t",
        source_name="blank",
    )
    prepared = DataImporter(project).import_file(path, request)
    stored = project.read_channel_data(prepared.source, prepared.channels[0])
    np.testing.assert_array_equal(stored.values, [1.0, 2.0, 3.0])
    blank_issue = prepared.source.issues[0]
    assert blank_issue.row_index == 1
    assert "空行" in blank_issue.description


# --- エラーと取り込み先 ----------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    """テキストファイルを書いてパスを返す。

    Args:
        path: 書き込み先。
        text: 内容。

    Returns:
        書き込んだファイルのパス。
    """
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("t,v\n0,1\n1,abc\n", DataIssueKind.NON_NUMERIC_VALUES),  # 数値でない値
        ("t,v\n0,1\n,2\n", DataIssueKind.NAN_VALUES),  # 時刻の欠損
        ("t,w\n0,1\n", DataIssueKind.INVALID_COLUMN),  # 指定した列 v がない
    ],
)
def test_data_errors_block_import_and_write_nothing(
    project: ProjectStore, tmp_path: Path, text: str, kind: DataIssueKind
) -> None:
    """数値でない値・時刻の欠損・存在しない列はエラーとし、何も書き込まない。"""
    path = _write(tmp_path / "bad.csv", text)
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="t",
        channels=[ChannelImportSpec(column="v")],
        experiment_name="e",
        session_name="s",
        trial_name="t",
        source_name="bad",
    )
    with pytest.raises(ImportBlockedError) as excinfo:
        DataImporter(project).import_file(path, request)
    assert kind in {issue.kind for issue in excinfo.value.issues}
    assert _files_under(project.directory(ProjectDirectory.ORIGINALS)) == []
    assert project.database.list_entities(Source) == []


def test_resolve_targets_reuses_existing_entities(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """同じ名前の実験・セッション・試技があれば、新しく作らずにそれを使う。"""
    data, _ = synthetic_dataset
    importer = DataImporter(project)
    importer.import_file(data / "trial_001/high_rate.csv", high_rate_request())
    targets = importer.resolve_targets(low_rate_request())
    assert targets.new_entities == []
    importer.import_file(data / "trial_001/low_rate.csv", low_rate_request())
    assert len(project.database.list_entities(Trial)) == 1
    assert len(project.database.list_entities(Source)) == 2


def test_duplicate_source_name_in_trial_is_rejected(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """同じ試技に同じ名前の Source は登録できない（ファイルを書く前に分かる）。"""
    data, _ = synthetic_dataset
    importer = DataImporter(project)
    importer.import_file(data / "trial_001/high_rate.csv", high_rate_request())
    with pytest.raises(ImportTargetError, match="high_rate"):
        importer.import_file(data / "trial_001/high_rate.csv", high_rate_request())
    assert len(_files_under(project.directory(ProjectDirectory.ORIGINALS))) == 1


def test_failed_commit_removes_written_files(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """DB への登録に失敗したら、書き込んだコピーと正規化データを削除する。"""
    data, _ = synthetic_dataset
    importer = DataImporter(project)
    targets = importer.resolve_targets(high_rate_request())
    prepared = importer.prepare(
        data / "trial_001/high_rate.csv", high_rate_request(), targets.trial.id
    )
    assert len(_files_under(project.directory(ProjectDirectory.ORIGINALS))) == 1
    # 解決後に同名の実験が別に登録された状況を作り、一意制約違反で登録を失敗させる。
    project.database.insert(Experiment(name="合成"))
    with pytest.raises(StorageError):
        importer.commit(prepared, targets)
    assert _files_under(project.directory(ProjectDirectory.ORIGINALS)) == []
    assert _files_under(project.directory(ProjectDirectory.DATASETS)) == []
    assert project.database.list_entities(Source) == []
