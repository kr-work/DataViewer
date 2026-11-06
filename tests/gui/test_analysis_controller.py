"""ui/analysis_controller.py（解析の実行・表示・保存・復元）のテスト。

メイン画面を通して、処理を選ぶ → ワーカースレッドで計算 → 縦並べ表示に反映 → 自動で保存
→ 開き直して復元、の流れを確かめる。
"""

import shutil
from functools import partial
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from experiment_workbench.analysis import output_series_id, run_analysis
from experiment_workbench.model import (
    AnalysisKind,
    AnalysisResult,
    AnalysisSpec,
    Channel,
    ChannelImportSpec,
    ImportRequest,
    SourceFormat,
    TimeUnit,
)
from experiment_workbench.storage import ProjectDirectory
from experiment_workbench.ui import analysis_controller
from experiment_workbench.ui.main_window import MainWindow
from tests.conftest import SyntheticDataset, high_rate_request, low_rate_request

TIMEOUT_MS = 20_000
"""取り込み・計算の終了を待つ最大時間 [ms]。"""


def _idle(window: MainWindow) -> bool:
    """取り込みも解析の計算もしていないかを返す（qtbot.waitUntil 用）。

    Args:
        window: メイン画面。

    Returns:
        どちらもしていなければ True。
    """
    return not window.is_importing and not window.analysis.is_running


def _forbidden_run(*args: object, **kwargs: object) -> AnalysisResult:
    """呼ばれたらテストを失敗させる run_analysis の代わり（キャッシュから復元したことの確認用）。

    Args:
        *args: 使わない。
        **kwargs: 使わない。

    Returns:
        返さない。

    Raises:
        AssertionError: 必ず送出する。
    """
    raise AssertionError("計算結果のキャッシュがあるのに再計算した")


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


def _wait_idle(qtbot: QtBot, window: MainWindow) -> None:
    """取り込み・計算が終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
    """
    qtbot.waitUntil(partial(_idle, window), timeout=TIMEOUT_MS)


def _import(qtbot: QtBot, window: MainWindow, path: Path, request: ImportRequest) -> None:
    """取り込みを開始し、終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
        path: 取り込むファイル。
        request: 取り込みの指定。
    """
    assert window.start_import(path, request)
    _wait_idle(qtbot, window)


def _project_with_data(
    qtbot: QtBot, root: Path, synthetic_dataset: SyntheticDataset
) -> tuple[MainWindow, UUID, UUID]:
    """合成データ high_rate を取り込んだプロジェクトを作る。

    Args:
        qtbot: pytest-qt のヘルパー。
        root: プロジェクトフォルダ。
        synthetic_dataset: 合成データ。

    Returns:
        (メイン画面, accel のチャンネル ID, true_velocity のチャンネル ID)。
    """
    data, _ = synthetic_dataset
    window = _window(qtbot)
    assert window.create_project(root, "解析テスト")
    _import(qtbot, window, data / "trial_001/high_rate.csv", high_rate_request())
    accel, velocity = window.explorer_channel_ids()
    return window, accel, velocity


def _run(qtbot: QtBot, window: MainWindow) -> None:
    """解析パネルの条件で実行し、計算が終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
    """
    window.analysis_panel.request_run()
    assert window.analysis.is_running
    _wait_idle(qtbot, window)


def _start(qtbot: QtBot, window: MainWindow, kind: AnalysisKind) -> None:
    """処理の一覧で処理を選び、計算が終わるまで待つ。

    Args:
        qtbot: pytest-qt のヘルパー。
        window: メイン画面。
        kind: 解析の種類。
    """
    window.analysis_panel.start_method(kind)
    _wait_idle(qtbot, window)


def _layout(window: MainWindow) -> list[list[UUID]]:
    """縦並べ表示の、パネルごとの系列 ID を上から順に返す。

    Args:
        window: メイン画面。

    Returns:
        パネルごとの系列 ID のリスト。
    """
    return [panel.series_ids() for panel in window.workspace.panels]


# --- 実行と表示 -----------------------------------------------------------------


def test_filter_output_overlays_input_and_updates_in_place(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """結果は入力と同じパネルに重ね、条件を変えると自動で再計算して同じ系列を描き直す。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    assert window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    output = output_series_id(panel.analysis_id, "output")
    assert _layout(window) == [[accel, output], [velocity]]
    first = window.workspace.panels[0].series[1].values.copy()

    with qtbot.waitSignal(panel.run_requested, timeout=3000):
        panel.butterworth_page.order_spin.setValue(2)
    _wait_idle(qtbot, window)
    assert _layout(window) == [[accel, output], [velocity]]
    assert not np.array_equal(window.workspace.panels[0].series[1].values, first)
    window.close()


def test_result_is_shown_with_input_when_input_was_hidden(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """入力を表示していなくても、入力と結果を並べて表示する（比べられないと意味がないため）。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    window.workspace.hide_series(accel)
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    output = output_series_id(window.analysis_panel.analysis_id, "output")
    assert sorted(map(sorted, _layout(window))) == sorted(
        map(sorted, [[velocity], [accel, output]])
    )
    window.close()


def test_kalman_estimate_is_overlaid_and_extras_are_on_request(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """推定値は入力に帯付きで重ね、微分などの追加の出力は選んだときだけ別のパネルに出す。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.KALMAN)
    spec_id = panel.analysis_id
    estimate = output_series_id(spec_id, "state_0")
    derivative = output_series_id(spec_id, "state_1")
    assert _layout(window) == [[accel, estimate], [velocity]]
    drawn = window.workspace.panels[0].series[1]
    assert drawn.band_lower is not None
    assert drawn.band_upper is not None
    assert "イノベーションの RMS" in panel.status_label.text()

    assert window.analysis.show_series(derivative)
    expected = [[accel, estimate], [derivative], [velocity]]  # 入力のすぐ下に並べる
    assert _layout(window) == expected
    assert not any(p.warning_label.isVisibleTo(window) for p in window.workspace.panels)
    _run(qtbot, window)  # もう一度計算しても、パネルや系列は増えない
    assert _layout(window) == expected
    window.close()


@pytest.mark.parametrize("kind", [AnalysisKind.DERIVATIVE, AnalysisKind.INTEGRAL])
def test_result_of_other_unit_goes_below_input(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset, kind: AnalysisKind
) -> None:
    """単位が変わる結果は重ねず、入力のすぐ下に並べる（元の線の縦軸を壊さないため）。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    window.select_channel(accel)
    _start(qtbot, window, kind)
    output = output_series_id(window.analysis_panel.analysis_id, "output")
    assert _layout(window) == [[accel], [output], [velocity]]
    window.close()


def test_removed_part_is_placed_below_input(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """取り除いた成分は、入力のすぐ下のパネルに出る。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    output = output_series_id(panel.analysis_id, "output")
    removed = output_series_id(panel.analysis_id, "removed")
    assert window.analysis.show_series(removed)
    assert _layout(window) == [[accel, output], [removed], [velocity]]
    window.close()


def test_unchecked_output_stays_hidden_after_recalculation(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """外した出力を再計算で勝手に戻さず、もう一度選ぶと入力のパネルに重なる。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    output = output_series_id(panel.analysis_id, "output")
    window.workspace.hide_series(output)
    assert _layout(window) == [[accel], [velocity]]
    panel.butterworth_page.order_spin.setValue(2)
    _run(qtbot, window)
    assert _layout(window) == [[accel], [velocity]]
    assert window.analysis.show_series(output)
    assert _layout(window) == [[accel, output], [velocity]]
    window.close()


def test_latest_run_wins(qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset) -> None:
    """計算中に条件を変えて実行し直すと、前の計算の結果は捨てて最後の条件の結果を表示する。"""
    window, accel, _ = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    panel.butterworth_page.low_edit.set_value(5.0)
    panel.request_run()
    panel.butterworth_page.low_edit.set_value(50.0)
    panel.request_run()
    _wait_idle(qtbot, window)
    data = window.load_channel_data(accel)
    assert data is not None
    expected = run_analysis(panel.build_spec(), data).series[0].values
    np.testing.assert_array_equal(window.workspace.panels[0].series[1].values, expected)
    window.close()


def test_cancel_stops_long_kalman_run(qtbot: QtBot, tmp_path: Path) -> None:
    """長いカルマンフィルタの計算は「中止」で止まり、結果を表示しない。"""
    time_s = np.arange(200_000) / 1000.0  # 1 kHz で 200 秒（計算に数秒かかる長さ）
    values = np.sin(2 * np.pi * time_s) + np.random.default_rng(0).normal(0, 0.1, time_s.size)
    path = tmp_path / "long.csv"
    np.savetxt(path, np.column_stack([time_s, values]), delimiter=",", header="time_s,value")
    path.write_text(path.read_text(encoding="utf-8").removeprefix("# "), encoding="utf-8")
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="time_s",
        time_unit=TimeUnit.SECOND,
        channels=[ChannelImportSpec(column="value")],
        experiment_name="e",
        session_name="s",
        trial_name="t",
        source_name="long",
    )
    window = _window(qtbot)
    window.create_project(tmp_path / "p", "中止テスト")
    _import(qtbot, window, path, request)
    (channel,) = window.explorer_channel_ids()
    panel = window.analysis_panel
    window.select_channel(channel)
    panel.start_method(AnalysisKind.KALMAN)
    assert panel.cancel_button.isEnabled()
    panel.cancel_button.click()
    _wait_idle(qtbot, window)
    assert panel.status_label.text() == "計算を中止しました"
    assert not panel.cancel_button.isEnabled()
    assert _layout(window) == [[channel]]
    assert window.explorer_analysis_ids() == []  # 中止した処理は保存しない
    window.close()


# --- 保存・復元・削除 ------------------------------------------------------------


def test_result_is_saved_automatically_and_keeps_following_changes(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """計算できた処理は自動で保存され、条件を変えると保存も更新される。"""
    root = tmp_path / "p"
    window, accel, _ = _project_with_data(qtbot, root, synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    spec_id = panel.analysis_id
    path = root / ProjectDirectory.PIPELINES / f"{spec_id}.json"
    assert path.is_file()
    panel.butterworth_page.order_spin.setValue(2)
    _run(qtbot, window)
    saved = AnalysisSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.butterworth is not None
    assert saved.butterworth.order == 2
    assert window.explorer_analysis_ids() == [spec_id]
    window.close()


def test_failed_calculation_is_not_saved(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """計算できない条件の処理は保存せず、理由を見せる（一覧に壊れた行を増やさない）。"""
    root = tmp_path / "p"
    window, accel, _ = _project_with_data(qtbot, root, synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    panel.butterworth_page.low_edit.set_value(1e6)  # ナイキスト周波数を超える
    panel.start_method(AnalysisKind.BUTTERWORTH)
    _wait_idle(qtbot, window)
    assert "計算できません" in panel.status_label.text()
    assert window.explorer_analysis_ids() == []
    assert not any((root / ProjectDirectory.PIPELINES).glob("*.json"))
    window.close()


def test_duplicate_overlays_second_result_for_comparison(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """「複製して比べる」と、元の結果を残したまま 2 つ目の結果が同じパネルに重なる。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    first = panel.analysis_id
    panel.duplicate_button.click()
    _wait_idle(qtbot, window)
    second = panel.analysis_id
    assert first != second
    assert _layout(window) == [
        [accel, output_series_id(first, "output"), output_series_id(second, "output")],
        [velocity],
    ]
    assert window.explorer_analysis_ids() == [first, second]
    window.close()


def test_saved_kalman_analysis_is_restored_from_cache(
    qtbot: QtBot,
    tmp_path: Path,
    synthetic_dataset: SyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """処理は一覧に出て、開き直すと計算し直さずにキャッシュから表示が戻る。"""
    root = tmp_path / "p"
    window, accel, _ = _project_with_data(qtbot, root, synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.KALMAN)
    spec_id = panel.analysis_id
    assert window.explorer_analysis_ids() == [spec_id]
    assert (root / ProjectDirectory.PIPELINES / f"{spec_id}.json").is_file()
    assert any((root / ProjectDirectory.DERIVED / str(spec_id)).glob("*.parquet"))
    layout = _layout(window)
    estimate = window.workspace.panels[0].series[1]
    window.close()

    monkeypatch.setattr(analysis_controller, "run_analysis", _forbidden_run)
    reopened = _window(qtbot)
    assert reopened.open_project(root)
    assert _layout(reopened) == layout
    restored = reopened.workspace.panels[0].series[1]
    np.testing.assert_array_equal(restored.values, estimate.values)
    assert restored.band_upper is not None
    assert estimate.band_upper is not None
    np.testing.assert_array_equal(restored.band_upper, estimate.band_upper)
    assert reopened.explorer_analysis_ids() == [spec_id]
    reopened.close()


def test_saved_analysis_without_cache_is_recomputed(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """キャッシュが消えていても、保存した条件から計算し直して表示を戻し、キャッシュを作り直す。"""
    root = tmp_path / "p"
    window, accel, velocity = _project_with_data(qtbot, root, synthetic_dataset)
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    spec_id = window.analysis_panel.analysis_id
    expected = window.workspace.panels[0].series[1].values.copy()
    window.close()
    shutil.rmtree(root / ProjectDirectory.DERIVED / str(spec_id))

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    assert _layout(reopened) == [[accel, output_series_id(spec_id, "output")], [velocity]]
    np.testing.assert_array_equal(reopened.workspace.panels[0].series[1].values, expected)
    assert any((root / ProjectDirectory.DERIVED / str(spec_id)).glob("*.parquet"))
    reopened.close()


def test_missing_analysis_output_is_reported_on_reopen(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """保存した条件のファイルが失われた出力は、開き直したときに表示から外し、そのことを知らせる。"""
    root = tmp_path / "p"
    window, accel, velocity = _project_with_data(qtbot, root, synthetic_dataset)
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    spec_id = window.analysis_panel.analysis_id
    output = output_series_id(spec_id, "output")
    window.close()
    (root / ProjectDirectory.PIPELINES / f"{spec_id}.json").unlink()

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    assert _layout(reopened) == [[accel], [velocity]]
    messages = [reopened.messages.item(i).text() for i in range(reopened.messages.count())]
    assert any(str(output) in text for text in messages)
    reopened.close()


def test_edit_analysis_loads_saved_conditions(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """保存した解析を調整のために読み込むと、入力・ID・条件が戻る。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    panel.butterworth_page.order_spin.setValue(2)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    spec_id = panel.analysis_id
    window.select_channel(velocity)
    panel.butterworth_page.order_spin.setValue(6)
    assert panel.analysis_id != spec_id

    assert window.edit_analysis(spec_id)
    assert panel.analysis_id == spec_id
    assert panel.butterworth_page.order_spin.value() == 2
    assert panel.input_data is not None
    assert panel.input_data.channel.id == accel
    window.close()


def test_deleting_analysis_removes_outputs_and_saved_files_only(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """解析を削除すると、表示中の出力・条件のファイル・キャッシュが消え、生データは残る。"""
    root = tmp_path / "p"
    window, accel, velocity = _project_with_data(qtbot, root, synthetic_dataset)
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.BUTTERWORTH)
    spec_id = window.analysis_panel.analysis_id

    window.delete_analysis(spec_id)
    assert window.explorer_analysis_ids() == []
    assert _layout(window) == [[accel], [velocity]]
    assert not window.analysis_panel.is_editing  # 消した処理の調整の画面に留まらない
    assert not (root / ProjectDirectory.PIPELINES / f"{spec_id}.json").exists()
    assert not (root / ProjectDirectory.DERIVED / str(spec_id)).exists()
    store = window.store
    assert store is not None
    assert store.database.get(Channel, accel).id == accel
    assert window.load_channel_data(accel) is not None
    window.close()


# --- 2 チャンネルの解析 -----------------------------------------------------------


def _project_with_io_pair(
    qtbot: QtBot, root: Path, synthetic_dataset: SyntheticDataset
) -> tuple[MainWindow, UUID, UUID, float]:
    """合成データ io_pair を取り込んだプロジェクトを作る。

    Args:
        qtbot: pytest-qt のヘルパー。
        root: プロジェクトフォルダ。
        synthetic_dataset: 合成データ。

    Returns:
        (メイン画面, 入力のチャンネル ID, 出力のチャンネル ID, 最初のステップのファイル上の時刻)。
    """
    data, manifest = synthetic_dataset
    trial = manifest.trials[0]
    window = _window(qtbot)
    assert window.create_project(root, "入出力テスト")
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="time_s",
        time_unit=TimeUnit.SECOND,
        channels=[
            ChannelImportSpec(column="input", unit="N"),
            ChannelImportSpec(column="output", unit="V"),
        ],
        experiment_name="e",
        session_name="s",
        trial_name="t",
        source_name="io_pair",
    )
    _import(qtbot, window, data / "trial_001/io_pair.csv", request)
    io_input, io_output = window.explorer_channel_ids()
    step_time = trial.io_pair.clock_offset_s + trial.io_system.step_times_s[0]
    return window, io_input, io_output, step_time


def _metric_text(window: MainWindow, label: str) -> str:
    """解析結果の表で、項目名の行の値の文字列を返す。

    Args:
        window: メイン画面。
        label: 項目名。

    Returns:
        値の文字列。
    """
    table = window.result_view.metrics_table
    for row in range(table.rowCount()):
        name, value = table.item(row, 0), table.item(row, 1)
        if name is not None and value is not None and name.text() == label:
            return value.text()
    raise AssertionError(f"項目 {label} がありません")


def test_step_response_shows_metrics_markers_and_both_channels(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """ステップ応答を実行すると、入力と出力を縦に並べ、指標・グラフ・時刻の印を表示する。"""
    window, io_input, io_output, step_time = _project_with_io_pair(
        qtbot, tmp_path / "p", synthetic_dataset
    )
    panel = window.analysis_panel
    window.select_channel(io_input)
    _start(qtbot, window, AnalysisKind.STEP_RESPONSE)
    panel.alignment_page.set_range(step_time - 0.1, step_time + 0.2)
    _run(qtbot, window)
    assert _layout(window) == [[io_input], [io_output]]
    assert window.result_view.analysis_id == panel.analysis_id
    assert window.result_view.chart_tabs.count() == 1
    assert _metric_text(window, "時定数 τ") != "—"
    assert [p.marker_count() for p in window.workspace.panels] == [4, 4]
    assert window.workspace.panels[0].region_range() == pytest.approx(
        (step_time - 0.1, step_time + 0.2), abs=1e-6
    )
    assert panel.status_label.text() == ""  # 要約は解析結果の欄に出る
    window.close()


def test_range_selector_appears_and_drives_recalculation(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """範囲の指定を始めると入力を表示して青い帯を出し、帯を動かすと欄と計算に反映する。"""
    window, io_input, _, step_time = _project_with_io_pair(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(io_input)
    _start(qtbot, window, AnalysisKind.STEP_RESPONSE)
    plot_panel = window.workspace.panels[0]
    assert plot_panel.series_ids() == [io_input]
    assert plot_panel.region_range() is not None
    with qtbot.waitSignal(panel.run_requested, timeout=3000) as blocker:
        region = plot_panel.region_item()
        assert region is not None
        region.setRegion((step_time - 0.1, step_time + 0.2))
    spec = blocker.args[0] if blocker.args else None
    assert spec is not None
    assert spec.alignment.start_s == pytest.approx(step_time - 0.1, abs=1e-4)
    _wait_idle(qtbot, window)
    assert window.result_view.analysis_id == panel.analysis_id
    _start(qtbot, window, AnalysisKind.INTEGRAL)
    assert plot_panel.region_range() is None
    window.close()


def test_range_starts_in_the_middle_of_data_that_begins_at_zero(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """時刻 0 から始まるデータでも、欄の仮の値 0〜1 ではなくデータの中央から始める。"""
    rows = "\n".join(f"{i / 100:.2f},{i / 1000:.3f}" for i in range(1001))
    path = tmp_path / "from_zero.csv"
    path.write_text(f"time_s,value\n{rows}\n", encoding="utf-8")
    window = _window(qtbot)
    window.open_files([path])
    _wait_idle(qtbot, window)
    _start(qtbot, window, AnalysisKind.CROP)
    assert window.analysis_panel.range_values() == pytest.approx((3.0, 7.0), abs=0.5)
    window.close()


def test_leaving_two_channel_analysis_clears_overlays_and_result(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """処理の一覧に戻ると、時刻の印・青い帯・解析結果の欄を片付ける。"""
    window, io_input, _, step_time = _project_with_io_pair(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.show()
    qtbot.waitExposed(window)
    window.select_channel(io_input)
    _start(qtbot, window, AnalysisKind.STEP_RESPONSE)
    panel.alignment_page.set_range(step_time - 0.1, step_time + 0.2)
    _run(qtbot, window)
    plot_panel = window.workspace.panels[0]
    assert plot_panel.marker_count() > 0
    assert window.result_view.isVisibleTo(window)
    panel.back_button.click()
    assert plot_panel.marker_count() == 0
    assert plot_panel.region_range() is None
    assert not window.result_view.isVisibleTo(window)
    window.close()


def test_saved_two_channel_analysis_is_recomputed_when_opened(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """保存した 2 チャンネルの解析は、ツリーから開くと計算して結果を表示し、削除で表示も消える。"""
    root = tmp_path / "p"
    window, io_input, _, _ = _project_with_io_pair(qtbot, root, synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(io_input)
    _start(qtbot, window, AnalysisKind.HYSTERESIS)
    spec_id = panel.analysis_id
    assert window.explorer_analysis_ids() == [spec_id]
    assert not (root / ProjectDirectory.DERIVED / str(spec_id)).exists()  # 結果は保存しない
    window.close()

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    assert reopened.result_view.analysis_id is None
    assert reopened.edit_analysis(spec_id)
    _wait_idle(qtbot, reopened)
    assert reopened.result_view.analysis_id == spec_id
    reopened.delete_analysis(spec_id)
    assert reopened.result_view.analysis_id is None
    reopened.close()


def test_different_clocks_are_refused_with_reason_until_offset_is_given(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """時計が違うチャンネルは理由を表示して計算せず、時刻オフセットを指定すると計算する。"""
    data, manifest = synthetic_dataset
    high, low = manifest.trials[0].sources
    window, accel, _ = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    _import(qtbot, window, data / "trial_001/low_rate.csv", low_rate_request())
    position = window.explorer_channel_ids()[2]
    window.select_channel(accel)
    page = panel.alignment_page
    page.output_combo.setCurrentIndex(page.output_combo.findData(str(position)))
    _start(qtbot, window, AnalysisKind.HYSTERESIS)
    assert "時刻の対応" in panel.status_label.text()
    assert window.result_view.analysis_id is None
    page.offset_check.setChecked(True)
    page.offset_edit.set_value(high.clock_offset_s - low.clock_offset_s)
    _run(qtbot, window)
    assert window.result_view.analysis_id == panel.analysis_id
    window.close()


# --- 値の変換・時間の処理・統計 -------------------------------------------------------


def test_crop_overlays_selected_section_on_input(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """切り出しは青い帯で区間を選び、結果を生データに重ねる。時刻は元の時間軸のまま。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.CROP)
    output = output_series_id(panel.analysis_id, "output")
    assert _layout(window) == [[accel, output], [velocity]]
    plot_panel = window.workspace.panels[0]
    selected = plot_panel.region_range()
    assert selected is not None
    cut, raw = plot_panel.series[1], plot_panel.series[0]
    assert cut.time_s.size < raw.time_s.size
    assert float(cut.time_s[0]) >= selected[0] - 0.01
    window.close()


def test_saved_crop_is_restored_with_shorter_time_axis(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """切り出した結果は、長さが変わっていても開き直すと復元される。"""
    root = tmp_path / "p"
    window, accel, _ = _project_with_data(qtbot, root, synthetic_dataset)
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.CROP)
    expected = window.workspace.panels[0].series[1]
    spec_id = window.analysis_panel.analysis_id
    window.close()

    reopened = _window(qtbot)
    assert reopened.open_project(root)
    restored = reopened.workspace.panels[0].series[1]
    np.testing.assert_array_equal(restored.time_s, expected.time_s)
    np.testing.assert_array_equal(restored.values, expected.values)
    assert reopened.edit_analysis(spec_id)
    assert reopened.analysis_panel.range_values() is not None
    reopened.close()


def test_statistics_show_result_without_new_series(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """区間統計は解析結果の欄に表と分布を出し、時系列は増やさない。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.STATISTICS)
    data = window.load_channel_data(accel)
    assert data is not None
    assert _layout(window) == [[accel], [velocity]]
    assert window.result_view.analysis_id == panel.analysis_id
    assert float(_metric_text(window, "平均値")) == pytest.approx(
        float(np.mean(data.values)), rel=1e-3
    )
    assert window.result_view.headline_label.text().startswith("平均")
    assert window.workspace.panels[0].marker_count() == 2  # 最大と最小の時刻
    assert window.explorer_analysis_ids() == [panel.analysis_id]
    window.close()


def test_scale_result_goes_below_input_with_its_unit(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """線形変換は大きさが変わるので重ねず、指定した単位で入力のすぐ下に出す。"""
    window, accel, velocity = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    panel.scale_page.gain_edit.set_value(1000.0)
    panel.scale_page.unit_edit.setText("mm/s^2")
    _start(qtbot, window, AnalysisKind.SCALE)
    output = output_series_id(panel.analysis_id, "output")
    assert _layout(window) == [[accel], [output], [velocity]]
    series = window.workspace.panels[1].series[0]
    assert series.unit == "mm/s^2"
    data = window.load_channel_data(accel)
    assert data is not None
    np.testing.assert_allclose(series.values, data.values * 1000.0)
    window.close()


def test_fill_gaps_says_when_nothing_is_missing(
    qtbot: QtBot, tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """欠損がないデータでは、そのことを結論の文で知らせる。"""
    window, accel, _ = _project_with_data(qtbot, tmp_path / "p", synthetic_dataset)
    panel = window.analysis_panel
    window.select_channel(accel)
    _start(qtbot, window, AnalysisKind.FILL_GAPS)
    assert "欠損はありません" in panel.status_label.text()
    window.close()
