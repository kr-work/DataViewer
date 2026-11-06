"""ui/main_window.py（メイン画面）のテスト。

ファイルを開く → すぐグラフに表示 → 一覧のチェックと処理 → 開き直して続きから、の流れを確かめる。
"""

import json
import shutil
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget, QTreeWidgetItem
from pytestqt.qtbot import QtBot

from experiment_workbench.model import AnalysisKind, ImportRequest, Source
from experiment_workbench.storage import ProjectDirectory, ProjectStore
from experiment_workbench.ui import main_window as main_window_module
from experiment_workbench.ui.import_dialog import ImportDialog
from experiment_workbench.ui.main_window import (
    MAX_AUTO_PANELS,
    MainWindow,
    application_settings,
    main,
    workspace_root,
)
from experiment_workbench.ui.plot_workspace import series_id_from_mime
from tests.conftest import SyntheticDataset, high_rate_request, low_rate_request

IMPORT_TIMEOUT_MS = 20_000


def _import_finished(window: MainWindow) -> bool:
    """取り込みの処理が終わったかを返す（qtbot.waitUntil 用）。

    Args:
        window: メイン画面。

    Returns:
        取り込み中でなければ True。
    """
    return not window.is_importing


def _window(qtbot: QtBot) -> MainWindow:
    """メイン画面を作る（表示はしない）。

    Args:
        qtbot: pytest-qt のヘルパー。

    Returns:
        メイン画面。
    """
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _messages(window: MainWindow) -> list[str]:
    """メッセージ欄の内容を返す。

    Args:
        window: メイン画面。

    Returns:
        各行の文字列。
    """
    return [window.messages.item(i).text() for i in range(window.messages.count())]


def _import(qtbot: QtBot, window: MainWindow, path: Path, request: ImportRequest) -> None:
    """取り込みを開始し、終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
        path: 取り込むファイル。
        request: 取り込みの指定。
    """
    assert window.start_import(path, request)
    qtbot.waitUntil(partial(_import_finished, window), timeout=IMPORT_TIMEOUT_MS)


def test_import_display_save_and_restore(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """取り込んだチャンネルを縦に並べ、保存して開き直すと同じ表示が復元される。"""
    data, _ = synthetic_dataset
    root = tmp_path / "project"
    window = _window(qtbot)
    assert window.create_project(root, "GUI テスト")
    _import(qtbot, window, data / "trial_001/high_rate.csv", high_rate_request())
    _import(qtbot, window, data / "trial_001/low_rate.csv", low_rate_request())

    accel, velocity, position = window.explorer_channel_ids()
    panels = window.workspace.panels
    assert [panel.series_ids() for panel in panels] == [[accel], [velocity], [position]]
    # 別の時計のデータ（low_rate、同期済みでない）は時間軸を連動させず、警告を出す。
    assert [panel.warning_label.isVisibleTo(window) for panel in panels] == [False, False, True]
    start = float(window.workspace.panels[0].series[0].time_s[0])
    panels[0].set_x_range((start + 0.5, start + 1.0))
    window.close()

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    restored = reopened.workspace.panels
    assert [panel.series_ids() for panel in restored] == [[accel], [velocity], [position]]
    # 連動するパネルの間では、画面上の位置を揃えるための微小な補正が入ることがある。
    assert restored[0].x_range()[0] == pytest.approx(start + 0.5, abs=0.01)
    reopened.close()


def _wait_open(qtbot: QtBot, window: MainWindow) -> None:
    """ファイルを開く処理が終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
    """
    qtbot.waitUntil(partial(_import_finished, window), timeout=IMPORT_TIMEOUT_MS)


def _write_csv(path: Path, text: str) -> Path:
    """CSV ファイルを書く。

    Args:
        path: 書き込み先。
        text: 内容。

    Returns:
        書き込んだファイル。
    """
    path.write_text(text, encoding="utf-8")
    return path


def _analysis_finished(window: MainWindow) -> bool:
    """処理の計算が終わったかを返す（qtbot.waitUntil 用）。

    Args:
        window: メイン画面。

    Returns:
        計算中でなければ True。
    """
    return not window.analysis.is_running


def _has_analyses(window: MainWindow) -> bool:
    """一覧に処理の行があるかを返す（qtbot.waitUntil 用）。

    Args:
        window: メイン画面。

    Returns:
        あれば True。
    """
    return bool(window.explorer_analysis_ids())


def _has_channels(window: MainWindow, count: int) -> bool:
    """一覧のチャンネルの行の数が指定の数かを返す（qtbot.waitUntil 用）。

    Args:
        window: メイン画面。
        count: 期待する数。

    Returns:
        指定の数なら True。
    """
    return len(window.explorer_channel_ids()) == count


def _row_has_text(item: QTreeWidgetItem, text: str) -> bool:
    """一覧の行の名前が指定の文字列かを返す（qtbot.waitUntil 用）。

    Args:
        item: 一覧の行。
        text: 期待する名前。

    Returns:
        一致すれば True。
    """
    return item.text(0) == text


_shown_dialogs: list[tuple[Path, str]] = []
"""テスト中に開いた確認の画面の (ファイル, 確認が必要な理由)。"""


def _accept_dialog(dialog: ImportDialog) -> int:
    """確認の画面を開かずに受理する（ImportDialog.exec の代わり）。

    Args:
        dialog: 確認の画面。

    Returns:
        ダイアログの結果。
    """
    _shown_dialogs.append((Path(dialog.path_edit.text()), dialog.reason_label.text()))
    dialog.accept()
    return int(dialog.result())


def _reject_dialog(dialog: ImportDialog) -> int:
    """確認の画面を開かずに取り消す（ImportDialog.exec の代わり）。

    Args:
        dialog: 確認の画面。

    Returns:
        取り消しの結果。
    """
    _shown_dialogs.append((Path(dialog.path_edit.text()), dialog.reason_label.text()))
    return int(QDialog.DialogCode.Rejected)


class _ClockAnswer:
    """時計の質問に決まった答えを返し、尋ねられた内容を覚える（ask_same_clock の代わり）。

    Attributes:
        asked: 尋ねられたファイルの組。
    """

    def __init__(self, answer: bool | None) -> None:
        """返す答えを受け取る。

        Args:
            answer: 返す答え（None は取り消し）。
        """
        self._answer = answer
        self.asked: list[list[Path]] = []

    def __call__(self, paths: list[Path]) -> bool | None:
        """尋ねられた内容を覚えて答えを返す。

        Args:
            paths: 追加するファイル。

        Returns:
            決まった答え。
        """
        self.asked.append(paths)
        return self._answer


@pytest.fixture(autouse=True)
def clear_shown_dialogs() -> None:
    """前のテストで開いた確認の画面の記録を消す。"""
    _shown_dialogs.clear()


def _item(window: MainWindow, name: str) -> QTreeWidgetItem:
    """一覧から名前の行を探す。

    Args:
        window: メイン画面。
        name: 行の名前。

    Returns:
        見つかった行。
    """
    found: list[QTreeWidgetItem] = window.explorer.findItems(
        name, Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly
    )
    assert found, f"{name} の行がありません"
    return found[0]


# --- ファイルを開くだけで始める ------------------------------------------------------


def test_opening_a_file_shows_graph_without_any_question(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """ファイルを開くと、何も尋ねずに全チャンネルをグラフに出し、最初のチャンネルを対象にする。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    channels = window.explorer_channel_ids()
    assert [panel.series_ids() for panel in window.workspace.panels] == [[c] for c in channels]
    assert window.workspace.isVisibleTo(window)
    panel_input = window.analysis_panel.input_data
    assert panel_input is not None
    assert panel_input.channel.id == channels[0]
    store = window.store
    assert store is not None
    assert store.root.is_relative_to(workspace_root(application_settings()))
    window.close()


def test_original_file_is_not_modified_by_opening(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """開いても元のファイルは変わらず、ファイルのフォルダに何も作らない。"""
    data, _ = synthetic_dataset
    path = tmp_path / "data" / "io_pair.csv"
    path.parent.mkdir()
    shutil.copy(data / "trial_001/io_pair.csv", path)
    before = path.read_bytes()
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    window.close()
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_reopening_same_file_resumes_previous_work(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """同じファイルを開くと、取り込み直さず、処理と表示が前回の続きから始まる。"""
    data, _ = synthetic_dataset
    path = data / "trial_001/io_pair.csv"
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    window.analysis_panel.start_method(AnalysisKind.MOVING_AVERAGE)
    qtbot.waitUntil(partial(_analysis_finished, window), timeout=IMPORT_TIMEOUT_MS)
    spec_id = window.analysis_panel.analysis_id
    layout = [panel.series_ids() for panel in window.workspace.panels]
    store = window.store
    assert store is not None
    root = store.root
    window.close()

    again = _window(qtbot)
    again.open_files([path])
    _wait_open(qtbot, again)
    reopened = again.store
    assert reopened is not None
    assert reopened.root == root
    assert len(reopened.database.list_entities(Source)) == 1
    assert again.explorer_analysis_ids() == [spec_id]
    assert [panel.series_ids() for panel in again.workspace.panels] == layout
    again.close()


def test_reopening_after_abnormal_exit_still_shows_graph(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """異常終了で表示設定が残っていなくても、開いたときに空の画面にしない。"""
    data, _ = synthetic_dataset
    path = data / "trial_001/io_pair.csv"
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    store = window.store
    assert store is not None
    root = store.root
    window.close()
    (root / ProjectDirectory.VIEWS / "main.json").unlink()

    again = _window(qtbot)
    again.open_files([path])
    _wait_open(qtbot, again)
    assert again.workspace.displayed_series_ids()
    assert again.analysis_panel.input_data is not None
    again.close()


def test_changed_file_starts_separately(qtbot: QtBot, tmp_path: Path) -> None:
    """同じ名前でも内容が変わったファイルは別に保存し、前の結果と混ぜない。"""
    path = _write_csv(tmp_path / "a.csv", "time_s,v\n0,1\n1,2\n2,3\n")
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    store = window.store
    assert store is not None
    first = store.root
    window.close()
    _write_csv(path, "time_s,v\n0,5\n1,6\n2,7\n")
    second = _window(qtbot)
    second.open_files([path])
    _wait_open(qtbot, second)
    assert second.store is not None
    assert second.store.root != first
    second.close()


def test_many_channels_show_only_the_first_ones(qtbot: QtBot, tmp_path: Path) -> None:
    """チャンネルが多いときは最初のいくつかだけ表示し、残りはチェックで出せることを伝える。"""
    columns = [f"c{index}" for index in range(MAX_AUTO_PANELS + 3)]
    header = ",".join(["time_s", *columns])
    rows = "\n".join(",".join(str(row + c) for c in range(len(columns) + 1)) for row in range(5))
    path = _write_csv(tmp_path / "wide.csv", f"{header}\n{rows}\n")
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    assert len(window.workspace.panels) == MAX_AUTO_PANELS
    assert any("残りの 3 個" in text for text in _messages(window))
    assert _item(window, "c0").checkState(0) == Qt.CheckState.Checked
    assert _item(window, columns[-1]).checkState(0) == Qt.CheckState.Unchecked
    window.close()


def test_file_needing_review_opens_dialog_with_reason(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """時刻の列が分からないファイルは、理由を添えた確認の画面を出してから開く。"""
    path = _write_csv(tmp_path / "a.csv", "alpha,beta\n1,2\n3,4\n")
    monkeypatch.setattr(ImportDialog, "exec", _accept_dialog)
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    assert any("時刻の列" in reason for _, reason in _shown_dialogs)
    assert window.explorer_channel_ids()
    window.close()


def test_cancelling_review_opens_nothing(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """確認の画面を取り消すと、データは追加しない。"""
    path = _write_csv(tmp_path / "a.csv", "alpha,beta\n1,2\n3,4\n")
    monkeypatch.setattr(ImportDialog, "exec", _reject_dialog)
    window = _window(qtbot)
    window.open_files([path])
    assert not window.is_importing
    assert window.explorer_channel_ids() == []
    window.close()


def test_import_error_opens_review_with_reason(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """プレビューに出ない行のエラーで取り込めないときは、理由を見せて読み方を確認してもらう。"""
    rows = "\n".join(f"{index},{index}" for index in range(40))
    path = _write_csv(tmp_path / "a.csv", f"time_s,value\n{rows}\n39,abc\n")
    monkeypatch.setattr(ImportDialog, "exec", _reject_dialog)
    window = _window(qtbot)
    window.open_files([path])
    qtbot.waitUntil(partial(bool, _shown_dialogs), timeout=IMPORT_TIMEOUT_MS)
    assert "abc" in _shown_dialogs[0][1]
    window.close()


def test_review_option_always_shows_dialog(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """「読み込み設定を指定して開く」では、判定できるファイルでも確認の画面を出す。"""
    data, _ = synthetic_dataset
    monkeypatch.setattr(ImportDialog, "exec", _reject_dialog)
    window = _window(qtbot)
    path = data / "trial_001/io_pair.csv"
    window.open_files([path], review=True)
    assert [shown_path for shown_path, _ in _shown_dialogs] == [path]
    window.close()


def test_data_warnings_are_shown_not_hidden(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """データの異常（時刻の飛びなど）は確認なしで開いても隠さず、メッセージの欄を開いて見せる。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    assert not window.messages.isVisibleTo(window)
    window.open_files([data / "trial_002/high_rate.csv"])
    _wait_open(qtbot, window)
    assert any(text.startswith("警告") for text in _messages(window))
    assert window.messages.isVisibleTo(window)
    window.close()


def test_assumed_time_unit_is_shown(qtbot: QtBot, tmp_path: Path) -> None:
    """時刻の単位を仮定したときは、そのことをメッセージで知らせる。"""
    path = _write_csv(tmp_path / "a.csv", "time,v\n0,1\n1,2\n2,3\n")
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    assert any("秒として読みました" in text for text in _messages(window))
    window.close()


def test_missing_file_is_reported(qtbot: QtBot, tmp_path: Path) -> None:
    """見つからないファイルは理由を表示し、開始ページのままにする。"""
    window = _window(qtbot)
    window.open_files([tmp_path / "missing.csv"])
    assert window.store is None
    assert any("見つかりません" in text for text in _messages(window))
    window.close()


# --- 2 つ目のファイル -----------------------------------------------------------------


@pytest.fixture
def two_files(tmp_path: Path) -> tuple[Path, Path]:
    """同じ長さの 2 つの CSV を作る。

    Args:
        tmp_path: pytest の一時ディレクトリ。

    Returns:
        (1 つ目, 2 つ目)。
    """
    first = _write_csv(tmp_path / "first.csv", "time_s,a\n0,1\n1,2\n2,3\n")
    second = _write_csv(tmp_path / "second.csv", "time_s,b\n0,4\n1,5\n2,6\n")
    return first, second


def test_second_file_on_same_clock_is_linked(
    qtbot: QtBot, two_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 つ目のファイルは時計を尋ね、同じ時計なら時間軸を連動して同じ画面に追加する。"""
    first, second = two_files
    window = _window(qtbot)
    window.open_files([first])
    _wait_open(qtbot, window)
    answer = _ClockAnswer(answer=True)
    monkeypatch.setattr(window, "ask_same_clock", answer)
    window.open_files([second])
    _wait_open(qtbot, window)
    assert answer.asked == [[second]]
    panels = window.workspace.panels
    assert len(panels) == 2
    assert not panels[1].warning_label.isVisibleTo(window)
    panels[0].set_x_range((0.5, 1.5))
    assert panels[1].x_range() == pytest.approx((0.5, 1.5), abs=0.05)
    window.close()


def test_second_file_on_other_clock_is_not_aligned(
    qtbot: QtBot, two_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """別の時計と答えたら時刻を揃えず、別の時間軸として警告を出す。"""
    first, second = two_files
    window = _window(qtbot)
    window.open_files([first])
    _wait_open(qtbot, window)
    monkeypatch.setattr(window, "ask_same_clock", _ClockAnswer(answer=False))
    window.open_files([second])
    _wait_open(qtbot, window)
    assert window.workspace.panels[1].warning_label.isVisibleTo(window)
    window.close()


def test_cancelling_clock_question_adds_nothing(
    qtbot: QtBot, two_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """時計の質問を取り消すと、ファイルを追加しない。"""
    first, second = two_files
    window = _window(qtbot)
    window.open_files([first])
    _wait_open(qtbot, window)
    monkeypatch.setattr(window, "ask_same_clock", _ClockAnswer(answer=None))
    window.open_files([second])
    assert len(window.explorer_channel_ids()) == 1
    window.close()


def test_two_files_dropped_together_ask_once(
    qtbot: QtBot, two_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 つをまとめて開くと、時計を 1 回だけ尋ねて両方を開く。"""
    first, second = two_files
    window = _window(qtbot)
    answer = _ClockAnswer(answer=True)
    monkeypatch.setattr(window, "ask_same_clock", answer)
    window.open_files([first, second])
    qtbot.waitUntil(partial(_has_channels, window, 2), timeout=IMPORT_TIMEOUT_MS)
    _wait_open(qtbot, window)
    assert answer.asked == [[second]]
    assert len(window.workspace.panels) == 2
    window.close()


def test_same_file_is_not_added_twice(
    qtbot: QtBot, two_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """開いているファイルをもう一度開いても、尋ねずに何も増やさない。"""
    first, _ = two_files
    window = _window(qtbot)
    window.open_files([first])
    _wait_open(qtbot, window)
    answer = _ClockAnswer(answer=True)
    monkeypatch.setattr(window, "ask_same_clock", answer)
    window.open_files([first])
    assert answer.asked == []
    assert len(window.explorer_channel_ids()) == 1
    window.close()


# --- データの一覧 ---------------------------------------------------------------------


def test_unit_in_column_name_is_not_repeated(qtbot: QtBot, tmp_path: Path) -> None:
    """列名に単位が書かれたファイルでは、名前と単位を分けて表示し、処理の名前も短くなる。"""
    rows = "\n".join(f"{index},{index % 7}" for index in range(40))
    path = _write_csv(tmp_path / "a.csv", f"time_s,張力[N]\n{rows}\n")
    window = _window(qtbot)
    window.open_files([path])
    _wait_open(qtbot, window)
    heading = window.workspace.panels[0].readout_label.text()
    assert "<b>張力</b> [N]" in heading
    assert "張力[N]" not in heading
    assert _item(window, "張力").text(1).startswith("N")
    panel = window.analysis_panel
    assert panel.input_label.text() == "対象: 張力 [N]"
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    qtbot.waitUntil(partial(_has_analyses, window), timeout=IMPORT_TIMEOUT_MS)
    assert _item(window, "移動平均 11").text(0) == "移動平均 11"
    window.close()


def test_list_shows_files_and_channels_without_hierarchy(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """一覧の最上位はファイルで、実験・セッション・試技の階層は出さない。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    top = window.explorer.topLevelItem(0)
    assert top is not None
    assert top.text(0) == "io_pair"
    assert window.explorer.topLevelItemCount() == 1
    assert top.child(0) is not None
    assert (
        window.explorer.findItems("データ", Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly)
        == []
    )
    window.close()


def test_checkbox_shows_and_hides_channel(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """チェックでグラフの表示が切り替わり、空になったパネルは閉じる。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    item = _item(window, "input")
    channel = window.explorer_channel_ids()[0]
    assert item.checkState(0) == Qt.CheckState.Checked
    count = len(window.workspace.panels)
    item.setCheckState(0, Qt.CheckState.Unchecked)
    assert not window.workspace.panels_showing(channel)
    assert len(window.workspace.panels) == count - 1
    item.setCheckState(0, Qt.CheckState.Checked)
    assert window.workspace.panels_showing(channel)
    window.close()


def test_checks_follow_changes_made_on_the_graph(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """パネルの × で閉じると、一覧のチェックも外れる（表示と一覧をずらさない）。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    item = _item(window, "input")
    window.workspace.panels[0].close_button.click()
    assert item.checkState(0) == Qt.CheckState.Unchecked
    window.close()


def test_clicking_rows_switches_target_and_editor(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """チャンネルの行で対象が変わり、処理の行で調整の画面が開く。戻った後の同じ行のクリックも効く。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    panel = window.analysis_panel
    output_row = _item(window, "output")
    window.explorer.setCurrentItem(output_row)
    assert panel.input_data is not None
    assert panel.input_data.channel.name == "output"
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    qtbot.waitUntil(partial(_has_analyses, window), timeout=IMPORT_TIMEOUT_MS)
    spec_id = panel.analysis_id
    current = window.explorer.currentItem()
    assert current is not None
    assert current.text(0) == "移動平均 11"  # 作った処理の行を選んでいる

    panel.back_button.click()
    assert not panel.is_editing
    window.explorer.itemClicked.emit(current, 0)
    assert panel.is_editing
    assert panel.analysis_id == spec_id
    window.explorer.setCurrentItem(_item(window, "input"))
    assert not panel.is_editing
    assert panel.input_data.channel.name == "input"
    window.close()


def test_condition_change_updates_row_without_rebuilding(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """条件を変えると処理の行の名前が更新され、一覧は作り直さない（選択が動かないように）。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    panel = window.analysis_panel
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    qtbot.waitUntil(partial(_has_analyses, window), timeout=IMPORT_TIMEOUT_MS)
    row = _item(window, "移動平均 11")
    with qtbot.assertNotEmitted(window.analysis.analyses_changed):
        panel.moving_average_page.window_spin.setValue(21)
        panel.request_run()
        qtbot.waitUntil(partial(_row_has_text, row, "移動平均 21"), timeout=IMPORT_TIMEOUT_MS)
    window.close()


def test_kalman_outputs_get_their_own_checkboxes(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """出力が複数ある処理は、行の下の出力ごとにチェックで表示できる。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    panel = window.analysis_panel
    panel.start_method(AnalysisKind.KALMAN)
    qtbot.waitUntil(partial(_has_analyses, window), timeout=IMPORT_TIMEOUT_MS)
    count = len(window.workspace.panels)
    item = _item(window, "1 階微分")
    assert item.checkState(0) == Qt.CheckState.Unchecked
    item.setCheckState(0, Qt.CheckState.Checked)
    assert len(window.workspace.panels) == count + 1
    window.close()


def test_hidden_bottom_docks_leave_height_to_plots(
    qtbot: QtBot, synthetic_dataset: SyntheticDataset
) -> None:
    """問題がなければ下の欄は出さず、プロットが高さを使う。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    window.open_files([data / "trial_001/io_pair.csv"])
    _wait_open(qtbot, window)
    qtbot.wait(20)
    assert not window.messages.isVisibleTo(window)
    assert not window.result_view.isVisibleTo(window)
    window.close()


def test_blocked_import_reports_errors_and_registers_nothing(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """エラーのあるファイルは取り込まず、理由をメッセージ欄に表示し、何も登録・書き込みしない。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.create_project(tmp_path / "project", "GUI テスト")
    request = high_rate_request().model_copy(update={"source_name": "duplicate_columns"})
    _import(qtbot, window, data / "broken/duplicate_columns.csv", request)
    assert any("列名 'value'" in text for text in _messages(window))
    store = window.store
    assert store is not None
    assert store.database.list_entities(Source) == []
    assert not any(store.directory(ProjectDirectory.ORIGINALS).rglob("*.csv"))
    window.close()


def test_duplicate_source_name_is_rejected_before_import(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """同じ試技に同じ名前の Source を取り込もうとすると、開始前に理由を表示して断る。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.create_project(tmp_path / "project", "GUI テスト")
    _import(qtbot, window, data / "trial_001/high_rate.csv", high_rate_request())
    assert not window.start_import(data / "trial_001/high_rate.csv", high_rate_request())
    assert any("同じ名前の Source" in text for text in _messages(window))
    window.close()


def test_open_project_reports_missing_channel_in_saved_view(qtbot: QtBot, tmp_path: Path) -> None:
    """保存された表示に存在しないチャンネルがあれば、表示から外したことを知らせる。"""
    root = tmp_path / "project"
    ProjectStore.create(root, name="p").close()
    missing = uuid4()
    view = {"panels": [{"channel_ids": [str(missing)]}]}
    (root / ProjectDirectory.VIEWS / "main.json").write_text(json.dumps(view), encoding="utf-8")
    window = _window(qtbot)
    assert window.open_project(root)
    assert any(str(missing) in text for text in _messages(window))
    window.close()


def test_closing_saves_view(qtbot: QtBot, tmp_path: Path) -> None:
    """画面を閉じると表示状態が保存される。"""
    root = tmp_path / "project"
    window = _window(qtbot)
    window.create_project(root, "p")
    window.workspace.add_panel()
    window.close()
    saved = json.loads((root / ProjectDirectory.VIEWS / "main.json").read_text(encoding="utf-8"))
    assert len(saved["panels"]) == 2


def _close_application() -> None:
    """開いているすべての画面を閉じて、イベントループを終える（main() の起動確認用）。"""
    for widget in QApplication.topLevelWidgets():
        widget.close()
    QApplication.quit()


def test_main_starts_and_opens_project_given_as_argument(tmp_path: Path) -> None:
    """起動コマンドの main() は画面を開き、引数で渡したプロジェクトを開く。"""
    root = tmp_path / "project"
    ProjectStore.create(root, name="起動確認").close()
    QTimer.singleShot(300, _close_application)
    assert main(["ewb", str(root)]) == 0
    # 最後の画面を閉じたときの終了の指示が残っていると、後のテストのイベントループがすぐ終わって
    # ワーカースレッドからの通知が届かなくなるので、取り除く。
    application = QApplication.instance()
    assert application is not None
    QApplication.removePostedEvents(application, QEvent.Type.Quit.value)
    assert (root / ProjectDirectory.VIEWS / "main.json").is_file()  # 閉じるときに表示を保存した


# --- 画面の使いやすさ -------------------------------------------------------------


def test_start_page_is_shown_until_project_is_open(qtbot: QtBot, tmp_path: Path) -> None:
    """プロジェクトを開く前は開始ページだけを見せ、開くと縦並べ表示に切り替わる。"""
    window = _window(qtbot)
    assert window.welcome_page.isVisibleTo(window)
    assert not window.workspace.isVisibleTo(window)
    window.create_project(tmp_path / "p", "p")
    assert window.workspace.isVisibleTo(window)
    assert not window.welcome_page.isVisibleTo(window)
    window.close_project()
    assert window.welcome_page.isVisibleTo(window)


def test_recent_projects_can_be_reopened(qtbot: QtBot, tmp_path: Path) -> None:
    """開いたプロジェクトは開始ページの一覧に出て、選ぶと開ける（消えたフォルダは出さない）。"""
    root = tmp_path / "recent"
    gone = tmp_path / "gone"
    window = _window(qtbot)
    window.create_project(gone, "消えるプロジェクト")
    window.create_project(root, "残るプロジェクト")
    window.close()
    shutil.rmtree(gone)
    second = _window(qtbot)
    names = [
        second.welcome_page.recent_list.item(i).text()
        for i in range(second.welcome_page.recent_list.count())
    ]
    assert names[0] == "recent"
    assert "gone" not in names
    second.welcome_page.recent_list.itemClicked.emit(second.welcome_page.recent_list.item(0))
    assert second.store is not None
    assert second.store.root == root.absolute()
    second.close()


def test_channels_can_be_dragged_from_tree(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """チャンネルの項目はドラッグでき、ドラッグのデータから系列 ID が分かる。"""
    data, _ = synthetic_dataset
    window = _window(qtbot)
    window.create_project(tmp_path / "p", "p")
    _import(qtbot, window, data / "trial_001/high_rate.csv", high_rate_request())
    accel = window.explorer_channel_ids()[0]
    found: list[QTreeWidgetItem] = window.explorer.findItems(
        "accel", Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly
    )
    assert series_id_from_mime(window.explorer.mimeData(found)) == accel
    assert found[0].flags() & Qt.ItemFlag.ItemIsDragEnabled
    target = window.workspace.panels[1]
    window.workspace.series_dropped.emit(accel, target)  # 別のチャンネルのパネルに重ねる
    assert target in window.workspace.panels_showing(accel)
    window.close()


def test_dropped_files_are_opened(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ファイルをウィンドウへドロップすると、そのまま開く（複数でもよい）。"""
    window = _window(qtbot)
    opened: list[list[Path]] = []
    monkeypatch.setattr(window, "open_files", opened.append)
    mime = QMimeData()
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    window.dropEvent(
        QDropEvent(
            QPointF(5, 5),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert opened == [paths]


def test_default_workspace_root_is_in_user_data_folder() -> None:
    """既定の保存先は、利用者のデータのフォルダの下にする。"""
    settings = application_settings()
    settings.remove(main_window_module.WORKSPACE_ROOT_KEY)
    root = workspace_root(settings)
    assert root.parts[-2:] == (main_window_module.SETTINGS_APPLICATION, "workspaces")


def test_settings_follow_the_redirected_folder(qtbot: QtBot, isolated_settings: Path) -> None:
    """設定の保存先の指定が効き、テストが利用者の設定（レジストリ）に書き込まない。

    形式を省いた ``QSettings(組織名, アプリ名)`` はこの指定を無視してレジストリへ書き込む。
    その状態に戻ると、利用者の設定がテストで変わり、古い値でテストの結果まで左右される。
    """
    file_name = Path(application_settings().fileName())
    assert file_name.is_relative_to(isolated_settings)
    window = _window(qtbot)
    window.close()
    settings = application_settings()
    # QSettings はファイルへ遅れて書き込むので、確かめる前に書き出す。
    settings.sync()
    assert "window_geometry" in settings.allKeys()
    assert file_name.is_file()


def test_closing_the_project_keeps_dock_widths(qtbot: QtBot, tmp_path: Path) -> None:
    """プロジェクトを閉じてから終了しても、ドックの幅を覚えている。"""
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    window.create_project(tmp_path / "p", "p")
    qtbot.wait(50)
    dock = window.findChild(QDockWidget, "explorerDock")
    assert isinstance(dock, QDockWidget)
    width = dock.width()
    window.close_project()
    assert str(application_settings().value("left_dock_width")) == str(width)


def test_dock_widths_survive_restart(qtbot: QtBot, tmp_path: Path) -> None:
    """左右のドックの幅は次の起動に引き継がれ、壊れた値でも細く潰れない。"""
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    window.resize(1500, 900)
    window.create_project(tmp_path / "p", "p")
    qtbot.wait(50)
    # 既定の幅に広がっている（最小の幅のままではない）。
    assert window.explorer.width() > 220
    assert window.analysis_panel.width() > 320
    window.close()

    settings = application_settings()
    assert str(settings.value("left_dock_width")).isdigit()
    settings.setValue("left_dock_width", "abc")
    settings.setValue("right_dock_width", 1)
    settings.sync()
    second = _window(qtbot)
    second.show()
    qtbot.waitExposed(second)
    assert second.open_project(tmp_path / "p")
    qtbot.wait(50)
    assert second.explorer.width() > 100
    assert second.analysis_panel.width() > 150
    second.close()
