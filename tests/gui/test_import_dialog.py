"""ui/import_dialog.py（取り込みダイアログ）のテスト。"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox
from pytestqt.qtbot import QtBot

from experiment_workbench.importer import DataImporter
from experiment_workbench.model import SourceFormat, TimeUnit
from experiment_workbench.storage import ProjectStore
from experiment_workbench.ui.import_dialog import ImportDialog
from tests.conftest import SyntheticDataset


def _dialog(qtbot: QtBot, project: ProjectStore, path: Path) -> ImportDialog:
    """ファイルを指定した取り込みダイアログを作る（表示はしない）。

    Args:
        qtbot: pytest-qt のヘルパー。
        project: 取り込み先のプロジェクト。
        path: 取り込むファイル。

    Returns:
        取り込みダイアログ。
    """
    dialog = ImportDialog(DataImporter(project), path)
    qtbot.addWidget(dialog)
    return dialog


def _ok_enabled(dialog: ImportDialog) -> bool:
    """「取り込む」ボタンが押せるかを返す。

    Args:
        dialog: 取り込みダイアログ。

    Returns:
        押せれば True。
    """
    return dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def _preview_headers(dialog: ImportDialog) -> list[str]:
    """プレビューの表の列見出し（列名と種類）を返す。

    Args:
        dialog: 取り込みダイアログ。

    Returns:
        各列の見出しの文字列。
    """
    headers: list[str] = []
    for index in range(dialog.preview_table.columnCount()):
        item = dialog.preview_table.horizontalHeaderItem(index)
        headers.append(item.text() if item is not None else "")
    return headers


def _issue_texts(dialog: ImportDialog) -> list[str]:
    """異常の一覧の表示内容を返す。

    Args:
        dialog: 取り込みダイアログ。

    Returns:
        各行の文字列。
    """
    return [dialog.issues_list.item(i).text() for i in range(dialog.issues_list.count())]


def test_dialog_previews_file_and_preselects_time_column(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """ファイルを開くとプレビューが表示され、時刻らしい列が時刻列に、他の数値列がチャンネルになる。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "trial_001/high_rate.csv")
    assert dialog.preview_table.columnCount() == 5
    assert dialog.preview_table.rowCount() > 0
    assert dialog.time_column_combo.currentData() == "time_s"
    assert dialog.channel_table.rowCount() == 4
    assert dialog.source_name_edit.text() == "high_rate"
    assert _ok_enabled(dialog)
    request = dialog.build_request()
    assert request.time_unit is TimeUnit.SECOND
    assert request.file_format is SourceFormat.CSV


def test_millisecond_time_column_unit_is_guessed(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """列名が "time_ms" の時刻列は、単位の既定値がミリ秒になる。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "trial_001/low_rate.csv")
    assert dialog.time_column_combo.currentData() == "time_ms"
    assert dialog.build_request().time_unit is TimeUnit.MILLISECOND


def test_selected_format_and_delimiter_are_used(
    qtbot: QtBot, project: ProjectStore, tmp_path: Path
) -> None:
    """拡張子が .tsv のファイルは TSV・タブ区切りとして読み、その選択が指定に反映される。"""
    path = tmp_path / "data.tsv"
    path.write_text("time_s\tvalue\n0.0\t1.0\n0.1\t2.0\n", encoding="utf-8")
    dialog = _dialog(qtbot, project, path)
    assert _preview_headers(dialog) == ["time_s\n数値", "value\n数値"]
    request = dialog.build_request()
    assert request.file_format is SourceFormat.TSV
    assert request.read_options.delimiter == "\t"


def test_file_with_structural_error_cannot_be_imported(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """列名が重複したファイルはエラーを表示し、「取り込む」を押せない。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "broken/duplicate_columns.csv")
    assert any(text.startswith("エラー") and "value" in text for text in _issue_texts(dialog))
    assert not _ok_enabled(dialog)


def test_changing_encoding_fixes_shift_jis_file(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """Shift_JIS のファイルは、文字コードを cp932 にしてプレビューを更新すると読める。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "broken/shift_jis_header.csv")
    assert not _ok_enabled(dialog)
    dialog.encoding_combo.setEditText("cp932")
    dialog.refresh_preview()
    assert _issue_texts(dialog) == []
    assert dialog.time_column_combo.currentData() == "時刻[s]"
    assert _ok_enabled(dialog)


def test_build_request_reflects_choices(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """選んだチャンネル・単位・登録先・同期の有無が、取り込みの指定に反映される。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "trial_001/high_rate.csv")
    for row in range(dialog.channel_table.rowCount()):
        name_item = dialog.channel_table.item(row, 1)
        check_item = dialog.channel_table.item(row, 0)
        unit_item = dialog.channel_table.item(row, 3)
        assert name_item is not None
        assert check_item is not None
        assert unit_item is not None
        keep = name_item.text() == "accel"
        check_item.setCheckState(Qt.CheckState.Checked if keep else Qt.CheckState.Unchecked)
        if keep:
            unit_item.setText("m/s^2")
    dialog.experiment_combo.setEditText("実験X")
    dialog.session_combo.setEditText("セッションY")
    dialog.trial_combo.setEditText("試技Z")
    dialog.shares_clock_check.setChecked(True)
    request = dialog.build_request()
    assert [(c.column, c.unit) for c in request.channels] == [("accel", "m/s^2")]
    assert (request.experiment_name, request.session_name, request.trial_name) == (
        "実験X",
        "セッションY",
        "試技Z",
    )
    assert request.shares_trial_clock
    assert request.time_column == "time_s"


def test_no_time_column_uses_sample_interval(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """時刻列なしを選ぶとサンプル間隔を入力でき、指定に反映される。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "trial_001/high_rate.csv")
    dialog.time_column_combo.setCurrentIndex(0)
    assert dialog.interval_spin.isEnabled()
    dialog.interval_spin.setValue(0.002)
    request = dialog.build_request()
    assert request.time_column is None
    assert request.sample_interval_s == 0.002
    assert "time_s" in [channel.column for channel in request.channels]


def test_accept_with_invalid_name_keeps_dialog_open(
    qtbot: QtBot, project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """名前が空のまま「取り込む」を押しても閉じず、エラーを表示する。"""
    data, _ = synthetic_dataset
    dialog = _dialog(qtbot, project, data / "trial_001/high_rate.csv")
    dialog.source_name_edit.setText("")
    dialog.accept()
    assert dialog.import_request is None
    assert any(text.startswith("エラー") for text in _issue_texts(dialog))
