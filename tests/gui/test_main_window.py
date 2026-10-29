"""ui/main_window.py（メイン画面）のテスト。

取り込み（ワーカースレッド）→ 縦並べ表示 → 保存 → 開き直して復元、の流れを確かめる。
"""

import json
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from experiment_workbench.model import ImportRequest, Source
from experiment_workbench.storage import ProjectDirectory, ProjectStore
from experiment_workbench.ui.main_window import MainWindow, main
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
    assert window.add_channel_to_workspace(accel)
    assert window.add_channel_to_workspace(velocity, new_panel=True)
    assert window.add_channel_to_workspace(position, new_panel=True)
    panels = window.workspace.panels
    assert [panel.channel_ids() for panel in panels] == [[accel], [velocity], [position]]
    # 別の時計のデータ（low_rate、同期済みでない）は時間軸を連動させず、警告を出す。
    assert [panel.warning_label.isVisibleTo(window) for panel in panels] == [False, False, True]
    start = float(window.workspace.panels[0].series[0].time_s[0])
    panels[0].set_x_range((start + 0.5, start + 1.0))
    window.close()

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    restored = reopened.workspace.panels
    assert [panel.channel_ids() for panel in restored] == [[accel], [velocity], [position]]
    # 連動するパネルの間では、画面上の位置を揃えるための微小な補正が入ることがある。
    assert restored[0].x_range()[0] == pytest.approx(start + 0.5, abs=0.01)
    reopened.close()


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
    assert (root / ProjectDirectory.VIEWS / "main.json").is_file()  # 閉じるときに表示を保存した
