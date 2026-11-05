"""importer.py の自動判定（ファイルを開くだけで始めるための suggest）のテスト。

判定できるときは確認なしで取り込める指定を返し、判定できないときは推測で取り込まずに確認を
求めることを確かめる。
"""

from pathlib import Path

import pytest

from experiment_workbench.importer import (
    DEFAULT_GROUP_NAME,
    DataImporter,
    file_format_of,
    file_sha256,
    guess_time_column,
    parse_time_unit,
)
from experiment_workbench.model import (
    Channel,
    ColumnPreview,
    FilePreview,
    SourceFormat,
    TimeUnit,
)
from experiment_workbench.storage import ProjectStore
from tests.conftest import SyntheticDataset


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """テスト用のファイルを書く。

    Args:
        path: 書き込み先。
        text: 内容。
        encoding: 文字コード。

    Returns:
        書き込んだファイル。
    """
    path.write_bytes(text.encode(encoding))
    return path


def test_synthetic_file_opens_without_questions(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """時刻列のある普通の CSV は、確認なしで全チャンネルを取り込める。"""
    output, _ = synthetic_dataset
    path = output / "trial_001" / "high_rate.csv"
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.request is not None
    assert suggestion.review_reasons == []
    assert suggestion.assumptions == []
    assert suggestion.request.time_column == "time_s"
    assert suggestion.request.time_unit is TimeUnit.SECOND
    columns = [spec.column for spec in suggestion.request.channels]
    assert "accel" in columns
    assert "time_s" not in columns
    assert suggestion.request.source_name == "high_rate"
    assert suggestion.request.experiment_name == DEFAULT_GROUP_NAME


def test_suggestion_can_be_imported_as_is(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """自動で決めた指定のまま、実際に取り込める。"""
    output, _ = synthetic_dataset
    path = output / "trial_001" / "low_rate.csv"
    importer = DataImporter(project)
    suggestion = importer.suggest(path)
    assert suggestion.request is not None
    assert suggestion.request.time_unit is TimeUnit.MILLISECOND
    importer.import_file(path, suggestion.request)
    assert importer.has_same_file(path)


def test_time_unit_unknown_is_reported_as_assumption(project: ProjectStore, tmp_path: Path) -> None:
    """単位が列名から分からない時刻列は秒とみなし、その仮定を必ず知らせる。"""
    path = _write(tmp_path / "a.csv", "time,force\n0,1\n1,2\n2,3\n")
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.request is not None
    assert suggestion.request.time_unit is TimeUnit.SECOND
    assert any("秒として読みました" in text for text in suggestion.assumptions)


def test_no_time_column_asks_for_review(project: ProjectStore, tmp_path: Path) -> None:
    """時刻列が分からないファイルは、推測で取り込まずに確認を求める。"""
    path = _write(tmp_path / "a.csv", "alpha,beta\n1,2\n3,4\n")
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.request is None
    assert any("時刻の列" in text for text in suggestion.review_reasons)


def test_broken_file_asks_for_review(project: ProjectStore, tmp_path: Path) -> None:
    """読めないファイルは、理由を添えて確認を求める。"""
    path = _write(tmp_path / "a.csv", "time,a,a\n0,1,2\n")
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.request is None
    assert any("回現れます" in text for text in suggestion.review_reasons)


def test_time_only_file_asks_for_review(project: ProjectStore, tmp_path: Path) -> None:
    """時刻以外の数値の列がなければ、確認を求める。"""
    path = _write(tmp_path / "a.csv", "time_s,label\n0,a\n1,b\n")
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.request is None
    assert any("数値の列" in text for text in suggestion.review_reasons)


def test_shift_jis_semicolon_and_preamble_are_detected(
    project: ProjectStore, tmp_path: Path
) -> None:
    """日本語の装置が書き出す形式（Shift_JIS・セミコロン・前置きの行）を判定する。"""
    text = "装置: ロガー\n記録日: 2026-11-05\n時刻(ms);張力[N]\n0;1.5\n1;1.6\n2;1.7\n"
    path = _write(tmp_path / "a.csv", text, "cp932")
    importer = DataImporter(project)
    suggestion = importer.suggest(path)
    assert suggestion.read_options.encoding == "cp932"
    assert suggestion.read_options.delimiter == ";"
    assert suggestion.read_options.skip_rows == 2
    assert suggestion.request is not None
    assert suggestion.request.time_column == "時刻(ms)"
    assert suggestion.request.time_unit is TimeUnit.MILLISECOND
    assert suggestion.request.channels[0].unit == "N"
    assert any("2 行" in text for text in suggestion.assumptions)
    prepared = importer.import_file(path, suggestion.request)
    assert prepared.channels[0].unit == "N"


def test_tab_separated_text_is_detected(project: ProjectStore, tmp_path: Path) -> None:
    """拡張子が .txt でも、タブ区切りを判定する。"""
    path = _write(tmp_path / "a.txt", "t\tv\n0\t1\n1\t2\n2\t3\n")
    suggestion = DataImporter(project).suggest(path)
    assert suggestion.read_options.delimiter == "\t"
    assert suggestion.request is not None
    assert suggestion.request.time_column == "t"


def test_same_stem_gets_a_unique_source_name(
    project: ProjectStore, synthetic_dataset: SyntheticDataset, tmp_path: Path
) -> None:
    """別のフォルダの同じ名前のファイルは、名前を変えて取り込める。"""
    output, _ = synthetic_dataset
    importer = DataImporter(project)
    first = output / "trial_001" / "low_rate.csv"
    suggestion = importer.suggest(first)
    assert suggestion.request is not None
    importer.import_file(first, suggestion.request)
    other = _write(tmp_path / "low_rate.csv", "time_s,v\n0,1\n1,2\n")
    second = importer.suggest(other)
    assert second.request is not None
    assert second.request.source_name == "low_rate_2"
    assert second.request.trial_name == suggestion.request.trial_name


def test_same_file_is_recognized_by_content(
    project: ProjectStore, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """取り込み済みかどうかは、ファイル名ではなく内容で判定する。"""
    output, _ = synthetic_dataset
    original = output / "trial_001" / "low_rate.csv"
    importer = DataImporter(project)
    suggestion = importer.suggest(original)
    assert suggestion.request is not None
    importer.import_file(original, suggestion.request)
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(original.read_bytes())
    assert importer.has_same_file(renamed)
    assert not importer.has_same_file(_write(tmp_path / "b.csv", "time_s,v\n0,1\n1,2\n"))
    assert not importer.has_same_file(tmp_path / "missing.csv")
    assert file_sha256(renamed) == file_sha256(original)
    assert project.database.list_entities(Channel)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("time_s", TimeUnit.SECOND),
        ("time_ms", TimeUnit.MILLISECOND),
        ("t [us]", TimeUnit.MICROSECOND),
        ("time(ns)", TimeUnit.NANOSECOND),
        ("時刻(ミリ秒)", TimeUnit.MILLISECOND),
        ("elapsed-sec", TimeUnit.SECOND),
        ("time", None),
        ("times", None),
        ("timestamp", None),
        ("items", None),
    ],
)
def test_time_unit_is_read_from_the_column_name(column: str, expected: TimeUnit | None) -> None:
    """時刻の単位は単語の区切りを見て読み取る（"times" の s を秒と読まない）。"""
    assert parse_time_unit(column) is expected


def test_guess_time_column_falls_back_to_first_numeric() -> None:
    """時刻らしい列がなければ、確認画面の初期値は最初の数値の列にする。"""
    preview = FilePreview(
        columns=[
            ColumnPreview(name="label", dtype="数値以外", is_numeric=False),
            ColumnPreview(name="alpha", dtype="数値", is_numeric=True),
        ],
        rows=[],
        issues=[],
    )
    assert guess_time_column(preview) == "alpha"
    assert guess_time_column(FilePreview(columns=[], rows=[], issues=[])) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.csv", SourceFormat.CSV),
        ("a.TXT", SourceFormat.CSV),
        ("a.tsv", SourceFormat.TSV),
        ("a.Parquet", SourceFormat.PARQUET),
    ],
)
def test_file_format_is_decided_by_suffix(name: str, expected: SourceFormat) -> None:
    """ファイル形式は拡張子で判定する。"""
    assert file_format_of(Path(name)) is expected
