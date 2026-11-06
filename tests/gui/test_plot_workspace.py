"""ui/plot_workspace.py（縦並べ比較）のテスト。"""

from uuid import UUID, uuid4

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from experiment_workbench.model import (
    PlotSeries,
    Source,
    SourceFormat,
    TimeAxisKind,
    TimeBase,
    TimeMarker,
    TimeUnit,
)
from experiment_workbench.timebase import time_domain_key
from experiment_workbench.ui.plot_workspace import (
    EMPTY_PANEL_HINT,
    SERIES_MIME_TYPE,
    PlotPanel,
    StackedPlotWorkspace,
)
from experiment_workbench.ui.theme import RAW_BEHIND_RESULT, SERIES_COLORS

TIME = np.arange(0.0, 10.0, 0.01)
"""テスト用の時刻（0〜10 秒、100 Hz）。"""


def _source(trial_id: UUID, name: str, *, shares_trial_clock: bool = False) -> Source:
    """表示のテストに必要な項目だけを持つ Source を作る。

    Args:
        trial_id: 所属する試技の ID。
        name: Source 名。
        shares_trial_clock: 同期済みなら True。

    Returns:
        Source。
    """
    return Source(
        trial_id=trial_id,
        name=name,
        format=SourceFormat.CSV,
        original_filename=f"{name}.csv",
        original_relpath=f"originals/x/{name}.csv",
        original_sha256="0" * 64,
        original_size_bytes=1,
        dataset_relpath=f"datasets/{name}.parquet",
        time_column="t",
        time_base=TimeBase(
            kind=TimeAxisKind.RELATIVE_SECONDS,
            original_unit=TimeUnit.SECOND,
            shares_trial_clock=shares_trial_clock,
        ),
    )


def _series(
    source: Source,
    name: str,
    frequency_hz: float = 1.0,
    *,
    band: float | None = None,
    markers: list[float] | None = None,
) -> PlotSeries:
    """正弦波の表示用系列を作る。

    Args:
        source: 時間軸を決める Source。
        name: 系列の名前。
        frequency_hz: 正弦波の周波数 [Hz]。
        band: 帯の半幅（None なら帯なし）。
        markers: 印を付ける時刻（None なら印なし）。

    Returns:
        表示用の系列。
    """
    values = np.sin(2 * np.pi * frequency_hz * TIME)
    return PlotSeries(
        series_id=uuid4(),
        label=f"{source.name} / {name}",
        unit="V",
        source=source,
        time_s=TIME.copy(),
        values=values,
        band_lower=None if band is None else values - band,
        band_upper=None if band is None else values + band,
        marker_times_s=None if markers is None else np.array(markers),
    )


@pytest.fixture
def workspace(qtbot: QtBot) -> StackedPlotWorkspace:
    """空の縦並べ表示を作る。

    Args:
        qtbot: pytest-qt のヘルパー。

    Returns:
        縦並べ表示。
    """
    widget = StackedPlotWorkspace()
    qtbot.addWidget(widget)
    widget.resize(800, 600)
    return widget


def test_panels_of_same_source_share_time_axis(workspace: StackedPlotWorkspace) -> None:
    """同じ Source の系列を別のパネルに置くと、時間軸のズームが連動する。"""
    source = _source(uuid4(), "logger")
    assert workspace.add_series(_series(source, "voltage"))
    second = workspace.add_panel()
    assert workspace.add_series(_series(source, "tension"), second)
    first = workspace.panels[0]
    first.set_x_range((2.0, 3.0))
    assert second.x_range() == pytest.approx((2.0, 3.0), abs=0.05)
    assert not first.warning_label.isVisibleTo(workspace)
    assert not second.warning_label.isVisibleTo(workspace)


def test_cursor_moves_together_and_reads_nearest_sample(workspace: StackedPlotWorkspace) -> None:
    """カーソルを動かすと連動するパネルのカーソルも動き、最も近いサンプルの値を表示する。"""
    source = _source(uuid4(), "logger")
    workspace.add_series(_series(source, "voltage"))
    second = workspace.add_panel()
    workspace.add_series(_series(source, "tension", frequency_hz=0.5), second)
    first = workspace.panels[0]
    first.cursor_line.setValue(2.503)  # ドラッグと同じく位置変更のシグナルが出る
    assert second.cursor_time() == pytest.approx(2.503)
    expected = np.sin(2 * np.pi * 0.5 * 2.50)  # 最も近いサンプルは t = 2.50 s
    assert f"{expected:.5g} V" in second.readout_label.text()
    assert "t = 2.5 s" in second.readout_label.text()


def test_unsynchronized_source_is_not_linked_and_warned(workspace: StackedPlotWorkspace) -> None:
    """同期していない別の Source のパネルは基準パネルと時間軸を連動させず、警告を表示する。"""
    trial_id = uuid4()
    workspace.add_series(_series(_source(trial_id, "a"), "voltage"))
    other = workspace.add_panel()
    assert workspace.add_series(_series(_source(trial_id, "b"), "tension"), other)
    assert other.warning_label.isVisibleTo(workspace)
    before = other.x_range()
    workspace.panels[0].set_x_range((2.0, 3.0))
    assert other.x_range() == pytest.approx(before)


def test_panels_of_another_time_axis_are_linked_to_each_other(
    workspace: StackedPlotWorkspace,
) -> None:
    """基準パネルと比較できない時間軸でも、同じ時間軸のパネルどうしは連動する。"""
    trial_id = uuid4()
    other_source = _source(trial_id, "b")
    workspace.add_series(_series(_source(trial_id, "a"), "voltage"))
    second = workspace.add_panel()
    workspace.add_series(_series(other_source, "position"), second)
    third = workspace.add_panel()
    workspace.add_series(_series(other_source, "velocity"), third)
    second.set_x_range((5.0, 6.0))
    assert third.x_range() == pytest.approx((5.0, 6.0), abs=0.05)
    assert workspace.panels[0].x_range() != pytest.approx((5.0, 6.0), abs=0.05)
    assert second.warning_label.isVisibleTo(workspace)
    assert third.warning_label.isVisibleTo(workspace)


def test_incomparable_series_is_refused_in_same_panel(workspace: StackedPlotWorkspace) -> None:
    """時刻を比較できない系列は同じパネルに置けず、理由を知らせる。"""
    trial_id = uuid4()
    messages: list[str] = []
    workspace.message.connect(messages.append)
    workspace.add_series(_series(_source(trial_id, "a"), "voltage"))
    assert not workspace.add_series(_series(_source(trial_id, "b"), "tension"))
    assert len(workspace.panels[0].series_ids()) == 1
    assert "時刻を比較できない" in messages[-1]


def test_synchronized_sources_are_linked(workspace: StackedPlotWorkspace) -> None:
    """同じ試技で同期済みと明示した Source どうしは、同じパネルにも別のパネルにも並べられる。"""
    trial_id = uuid4()
    first_source = _source(trial_id, "a", shares_trial_clock=True)
    second_source = _source(trial_id, "b", shares_trial_clock=True)
    workspace.add_series(_series(first_source, "voltage"))
    assert workspace.add_series(_series(second_source, "tension"))
    other = workspace.add_panel()
    assert workspace.add_series(_series(second_source, "temperature"), other)
    workspace.panels[0].set_x_range((4.0, 5.0))
    assert other.x_range() == pytest.approx((4.0, 5.0), abs=0.05)
    assert not other.warning_label.isVisibleTo(workspace)


def test_duplicate_series_in_panel_is_refused(workspace: StackedPlotWorkspace) -> None:
    """同じ系列を同じパネルに 2 回は置けない。"""
    series = _series(_source(uuid4(), "a"), "voltage")
    assert workspace.add_series(series)
    assert not workspace.add_series(series)


def test_band_and_markers_are_drawn_and_read(workspace: StackedPlotWorkspace) -> None:
    """帯と印のある系列（カルマンフィルタの推定値など）は、帯の半幅もカーソル位置に表示する。"""
    series = _series(_source(uuid4(), "a"), "estimate", band=0.25, markers=[1.0, 1.5])
    workspace.add_series(series)
    panel = workspace.panels[0]
    panel.set_cursor(1.0)
    assert "± 0.25" in panel.readout_label.text()
    workspace.remove_series(panel, series.series_id)
    assert panel.series_ids() == []


def test_update_series_redraws_in_place(workspace: StackedPlotWorkspace) -> None:
    """同じ ID の新しいデータで描き直すと、表示順と色を保ったまま値が変わる。"""
    source = _source(uuid4(), "a")
    raw = _series(source, "raw")
    result = _series(source, "result", frequency_hz=1.0)
    workspace.add_series(raw)
    workspace.add_series(result)
    recomputed = result.model_copy(update={"values": result.values * 2.0, "label": "new label"})
    assert workspace.update_series(recomputed) == 1
    panel = workspace.panels[0]
    assert panel.series_ids() == [raw.series_id, result.series_id]
    np.testing.assert_array_equal(panel.series[1].values, result.values * 2.0)
    assert workspace.update_series(_series(source, "not shown")) == 0


def test_view_state_round_trip(qtbot: QtBot, workspace: StackedPlotWorkspace) -> None:
    """表示状態を保存用のモデルにして別の画面で復元すると、パネル・時間軸・カーソルが戻る。"""
    source = _source(uuid4(), "logger")
    voltage, tension = _series(source, "voltage"), _series(source, "tension")
    workspace.add_series(voltage)
    workspace.add_series(tension, workspace.add_panel())
    workspace.panels[0].set_x_range((1.0, 4.0))
    workspace.panels[0].set_cursor(2.0)
    view = workspace.view_state()

    restored = StackedPlotWorkspace()
    qtbot.addWidget(restored)
    by_id = {voltage.series_id: voltage, tension.series_id: tension}
    messages = restored.restore(view, by_id.get)
    assert messages == []
    assert [panel.series_ids() for panel in restored.panels] == [
        [voltage.series_id],
        [tension.series_id],
    ]
    assert restored.panels[0].x_range() == pytest.approx((1.0, 4.0), abs=0.05)
    assert restored.panels[1].cursor_time() == pytest.approx(2.0)


def test_restore_reports_missing_series(qtbot: QtBot) -> None:
    """保存されていた系列が見つからない場合は、表示から外して理由を返す。"""
    voltage = _series(_source(uuid4(), "logger"), "voltage")
    workspace = StackedPlotWorkspace()
    qtbot.addWidget(workspace)
    workspace.add_series(voltage)
    view = workspace.view_state()
    restored = StackedPlotWorkspace()
    qtbot.addWidget(restored)
    nothing: dict[UUID, PlotSeries] = {}
    messages = restored.restore(view, nothing.get)
    assert len(messages) == 1
    assert str(voltage.series_id) in messages[0]
    assert restored.panels[0].series_ids() == []


def test_removing_last_panel_keeps_one_empty_panel(workspace: StackedPlotWorkspace) -> None:
    """系列を外すと表示が消え、最後のパネルを削除しても空のパネルが 1 つ残る。"""
    series = _series(_source(uuid4(), "a"), "voltage")
    workspace.add_series(series)
    panel = workspace.panels[0]
    workspace.remove_series(panel, series.series_id)
    assert panel.series_ids() == []
    assert EMPTY_PANEL_HINT in panel.readout_label.text()
    workspace.remove_panel(panel)
    assert len(workspace.panels) == 1
    assert workspace.panels[0].series_ids() == []


# --- 見やすさと操作 ---------------------------------------------------------------


def _time_axis_shown(panel: PlotPanel) -> bool:
    """パネルに時刻の目盛りが表示されているかを返す。

    Args:
        panel: パネル。

    Returns:
        表示されていれば True。
    """
    return bool(panel.plot_item.getAxis("bottom").style["showValues"])


def test_time_axis_is_shown_only_at_bottom_of_linked_panels(
    workspace: StackedPlotWorkspace,
) -> None:
    """同じ時間軸のパネルでは、時刻の目盛りを一番下だけに出す（時間軸が変わる所では出す）。"""
    trial_id = uuid4()
    first, second = _source(trial_id, "a"), _source(trial_id, "b")
    workspace.add_series(_series(first, "x"))
    workspace.add_series(_series(first, "y"), workspace.add_panel())
    workspace.add_series(_series(second, "z"), workspace.add_panel())
    workspace.add_series(_series(second, "w"), workspace.add_panel())
    assert [_time_axis_shown(panel) for panel in workspace.panels] == [False, True, False, True]


def test_colors_restart_in_each_panel_and_stay_distinct(workspace: StackedPlotWorkspace) -> None:
    """パネルの中の系列は別々の色になり、どのパネルでも先頭の系列は同じ色になる。"""
    source = _source(uuid4(), "a")
    workspace.add_series(_series(source, "x"))
    workspace.add_series(_series(source, "y"))
    second = workspace.add_panel()
    workspace.add_series(_series(source, "z"), second)
    first_text = workspace.panels[0].readout_label.text()
    assert SERIES_COLORS[0] in first_text
    assert SERIES_COLORS[1] in first_text
    assert SERIES_COLORS[0] in second.readout_label.text()
    assert SERIES_COLORS[1] not in second.readout_label.text()


def test_legend_click_toggles_series_visibility(workspace: StackedPlotWorkspace) -> None:
    """凡例の色の四角をクリックすると、系列の表示・非表示が切り替わる（再計算しても保つ）。"""
    series = _series(_source(uuid4(), "a"), "x", band=0.1)
    workspace.add_series(series)
    panel = workspace.panels[0]
    panel.readout_label.linkActivated.emit(str(series.series_id))
    assert panel.is_series_hidden(series.series_id)
    assert "□" in panel.readout_label.text()
    workspace.update_series(series.model_copy(update={"values": series.values * 2}))
    assert panel.is_series_hidden(series.series_id)
    panel.readout_label.linkActivated.emit(str(series.series_id))
    assert not panel.is_series_hidden(series.series_id)
    assert "■" in panel.readout_label.text()


def test_place_cursor_updates_linked_panels(workspace: StackedPlotWorkspace) -> None:
    """クリックなどでカーソルを置くと、同じ時間軸のパネルのカーソルも動く。"""
    source = _source(uuid4(), "a")
    workspace.add_series(_series(source, "x"))
    second = workspace.add_panel()
    workspace.add_series(_series(source, "y"), second)
    workspace.panels[0].place_cursor(4.0)
    assert second.cursor_time() == pytest.approx(4.0)


def test_mouse_moves_time_axis_only_and_y_follows_view(workspace: StackedPlotWorkspace) -> None:
    """マウスは時間軸だけを動かし、縦軸は見えている範囲のデータに合わせる。"""
    panel = workspace.panels[0]
    view_box = panel.plot_item.getViewBox()
    assert view_box.state["mouseEnabled"] == [True, False]
    assert view_box.state["autoVisibleOnly"] == [False, True]
    assert view_box.autoRangeEnabled()[1]


def _wheel(panel: PlotPanel, position: QPointF, modifiers: Qt.KeyboardModifier) -> None:
    """パネルのプロット領域にホイールのイベントを送る。

    Args:
        panel: パネル。
        position: イベントの位置（プロット領域の座標）。
        modifiers: 押しているキー。
    """
    viewport = panel.plot_widget.viewport()
    event = QWheelEvent(
        position,
        QPointF(viewport.mapToGlobal(position.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(viewport, event)


def test_shift_wheel_zooms_y_and_fit_all_restores(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> None:
    """Shift + ホイールで縦軸をズームでき、「全体を表示」で全体と縦軸の自動調整に戻る。"""
    workspace.show()
    qtbot.waitExposed(workspace)
    source = _source(uuid4(), "a")
    workspace.add_series(_series(source, "x"))
    panel = workspace.panels[0]
    center = QPointF(panel.plot_widget.viewport().rect().center())
    assert panel.y_range_state() is None
    _wheel(panel, center, Qt.KeyboardModifier.ShiftModifier)
    assert panel.y_range_state() is not None
    panel.set_x_range((2.0, 3.0))
    workspace.fit_all()
    assert panel.y_range_state() is None
    low, high = panel.x_range()
    assert low < TIME[0]
    assert high > TIME[-1]


def test_fit_all_covers_data_of_all_linked_panels(workspace: StackedPlotWorkspace) -> None:
    """「全体を表示」は、連動しているパネルのどのデータも入る範囲にする。"""
    source = _source(uuid4(), "a")
    longer = _series(source, "x").model_copy(update={"time_s": TIME + 5.0})
    workspace.add_series(_series(source, "y"))
    second = workspace.add_panel()
    workspace.add_series(longer, second)
    workspace.fit_all()
    low, high = workspace.panels[0].x_range()
    assert low <= TIME[0]
    assert high >= TIME[-1] + 5.0


def test_time_markers_follow_panels_of_their_time_axis(workspace: StackedPlotWorkspace) -> None:
    """時刻の印は、その時間軸のパネルにだけ引き、後から加えた同じ時間軸のパネルにも引く。"""
    trial_id = uuid4()
    first, second = _source(trial_id, "a"), _source(trial_id, "b")
    workspace.add_series(_series(first, "x"))
    workspace.add_series(_series(second, "y"), workspace.add_panel())
    markers = [TimeMarker(time_s=1.0, label="開始"), TimeMarker(time_s=2.0, label="終了")]
    workspace.show_time_markers(time_domain_key(first), markers)
    assert [panel.marker_count() for panel in workspace.panels] == [2, 0]
    third = workspace.add_panel()
    workspace.add_series(_series(first, "z"), third)
    assert third.marker_count() == 2
    # 名前は一番上のパネルだけに添える（低いパネルで重なって読めなくならないように）。
    assert workspace.panels[0].has_marker_labels()
    assert not third.has_marker_labels()
    workspace.clear_time_markers()
    assert [panel.marker_count() for panel in workspace.panels] == [0, 0, 0]


def test_range_selector_is_shared_by_linked_panels(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> None:
    """解析する範囲の選択は同じ時間軸のパネルに出て、動かすと揃い、動かし終えると知らせる。"""
    source = _source(uuid4(), "a")
    workspace.add_series(_series(source, "x"))
    second = workspace.add_panel()
    workspace.add_series(_series(source, "y"), second)
    workspace.show_range_selector(time_domain_key(source), 2.0, 4.0)
    first = workspace.panels[0]
    assert first.region_range() == pytest.approx((2.0, 4.0))
    with qtbot.waitSignal(workspace.range_selected, timeout=1000) as blocker:
        region = first.region_item()
        assert region is not None
        region.setRegion((3.0, 5.0))  # 端のドラッグと同じシグナルが出る
    assert blocker.args == pytest.approx([3.0, 5.0])
    assert second.region_range() == pytest.approx((3.0, 5.0))
    workspace.hide_range_selector()
    assert first.region_range() is None


def _drop_event(series_id: UUID) -> QDropEvent:
    """系列のドロップのイベントを作る。

    Args:
        series_id: 系列 ID。

    Returns:
        ドロップのイベント。
    """
    mime = QMimeData()
    mime.setData(SERIES_MIME_TYPE, QByteArray(str(series_id).encode("ascii")))
    # イベントの間 mime が消えないよう、Python の参照を持たせておく。
    event = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    event.setProperty = mime  # type: ignore[attr-defined]
    return event


def test_dropping_series_on_panel_requests_it_there(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> None:
    """ツリーからパネルへドロップすると、そのパネルへの追加を知らせ、選択中にする。"""
    second = workspace.add_panel()
    first = workspace.panels[0]
    workspace.set_active_panel(first)
    series_id = uuid4()
    with qtbot.waitSignal(workspace.series_dropped, timeout=1000) as blocker:
        second.dropEvent(_drop_event(series_id))
    assert blocker.args == [series_id, second]
    assert workspace.active_panel is second


def test_new_panels_get_equal_plot_height(qtbot: QtBot, workspace: StackedPlotWorkspace) -> None:
    """パネルを足すと、一番下の時間軸の分を除いたプロットの高さが揃う。"""
    workspace.show()
    qtbot.waitExposed(workspace)
    source = _source(uuid4(), "a")
    for name in ("x", "y", "z"):
        workspace.show_series(_series(source, name))
    qtbot.wait(20)
    heights = [panel.height() for panel in workspace.panels]
    assert abs(heights[0] - heights[1]) <= 2
    assert 20 < heights[2] - heights[0] < 70


# --- 表示に合わせて増減するパネル ---------------------------------------------------


def test_show_series_uses_empty_panel_then_new_panels(workspace: StackedPlotWorkspace) -> None:
    """系列を表示すると、空のパネルを使い、なければパネルが増える。"""
    source = _source(uuid4(), "a")
    assert workspace.show_series(_series(source, "x"))
    assert len(workspace.panels) == 1
    assert workspace.show_series(_series(source, "y"))
    assert len(workspace.panels) == 2


def test_hide_series_closes_empty_panels(workspace: StackedPlotWorkspace) -> None:
    """系列を外して空になったパネルは閉じ、最後は空のパネルが 1 つ残る。"""
    source = _source(uuid4(), "a")
    first, second = _series(source, "x"), _series(source, "y")
    workspace.show_series(first)
    workspace.show_series(second)
    workspace.hide_series(first.series_id)
    assert len(workspace.panels) == 1
    assert workspace.panels[0].series_ids() == [second.series_id]
    workspace.hide_series(second.series_id)
    assert len(workspace.panels) == 1
    assert workspace.panels[0].series_ids() == []


def test_hide_series_keeps_panel_with_other_series(workspace: StackedPlotWorkspace) -> None:
    """重ねて表示している系列を外しても、残りの系列があるパネルは閉じない。"""
    source = _source(uuid4(), "a")
    raw, result = _series(source, "raw"), _series(source, "result")
    workspace.add_series(raw)
    workspace.add_series(result)
    workspace.hide_series(result.series_id)
    assert workspace.displayed_series_ids() == {raw.series_id}


def test_series_changed_is_emitted_for_every_way_of_changing(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> None:
    """表示が変わる操作のどれでも知らせる（一覧のチェックがずれないように）。"""
    source = _source(uuid4(), "a")
    series = _series(source, "x")
    with qtbot.waitSignal(workspace.series_changed, timeout=1000):
        workspace.show_series(series)
    panel = workspace.panels[0]
    with qtbot.waitSignal(workspace.series_changed, timeout=1000):
        panel.close_button.click()
    assert workspace.displayed_series_ids() == set()


def test_many_panels_keep_their_height_and_scroll(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> None:
    """パネルが多くても潰れず、画面の外にはスクロールで届く。"""
    workspace.resize(800, 300)
    workspace.show()
    qtbot.waitExposed(workspace)
    source = _source(uuid4(), "a")
    for index in range(8):
        workspace.show_series(_series(source, f"s{index}"))
    qtbot.wait(20)
    assert min(panel.height() for panel in workspace.panels) >= 80
    assert workspace.height() <= 310


# --- ドラッグ・ダブルクリック・スクロールバー -----------------------------------------


def _plot_point(panel: PlotPanel, fraction: float) -> QPoint:
    """プロットの描画領域の、横方向の割合の位置（縦は中央）を返す。

    Args:
        panel: パネル。
        fraction: 左端を 0、右端を 1 とした位置。

    Returns:
        プロット領域の座標。
    """
    view_rect = panel.plot_item.getViewBox().sceneBoundingRect()
    scene_x = view_rect.left() + fraction * view_rect.width()
    scene_point = QPointF(scene_x, view_rect.center().y())
    point: QPoint = panel.plot_widget.mapFromScene(scene_point)
    return point


def _drag(panel: PlotPanel, start: float, end: float, button: Qt.MouseButton) -> None:
    """プロットの上を実際のマウスのイベントでドラッグする。

    Args:
        panel: パネル。
        start: 開始の位置（左端 0、右端 1）。
        end: 終了の位置。
        button: 押すボタン。
    """
    viewport = panel.plot_widget.viewport()
    first, last = _plot_point(panel, start), _plot_point(panel, end)
    QTest.mousePress(viewport, button, Qt.KeyboardModifier.NoModifier, first)
    QTest.mouseMove(viewport, QPoint((first.x() + last.x()) // 2, first.y()))
    QTest.mouseMove(viewport, last)
    QTest.mouseRelease(viewport, button, Qt.KeyboardModifier.NoModifier, last)


@pytest.fixture
def shown_panels(
    qtbot: QtBot, workspace: StackedPlotWorkspace
) -> tuple[StackedPlotWorkspace, PlotPanel, PlotPanel]:
    """同じ時間軸の 2 パネルを表示した状態を作る。

    Args:
        qtbot: pytest-qt のヘルパー。
        workspace: 空の縦並べ表示。

    Returns:
        (縦並べ表示, 1 つ目のパネル, 2 つ目のパネル)。
    """
    workspace.show()
    qtbot.waitExposed(workspace)
    source = _source(uuid4(), "a")
    workspace.show_series(_series(source, "x"))
    workspace.show_series(_series(source, "y"))
    workspace.fit_all()
    qtbot.wait(20)
    return workspace, workspace.panels[0], workspace.panels[1]


def test_left_drag_zooms_to_selected_range_in_all_linked_panels(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """左ドラッグで選んだ範囲を拡大し、連動するパネルも揃う。縦軸は自動のまま。"""
    _, first, second = shown_panels
    low, high = first.x_range()
    _drag(first, 0.25, 0.5, Qt.MouseButton.LeftButton)
    width = high - low
    # 押した位置は、最初のドラッグ判定の分だけ動き出しが遅れることがあるので幅で確かめる。
    assert first.x_range()[1] - first.x_range()[0] == pytest.approx(0.25 * width, rel=0.1)
    assert second.x_range() == pytest.approx(first.x_range(), abs=0.05)
    assert first.y_range_state() is None


def test_right_to_left_drag_also_zooms(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """右から左へのドラッグでも同じように拡大する。"""
    _, first, _ = shown_panels
    low, high = first.x_range()
    _drag(first, 0.75, 0.5, Qt.MouseButton.LeftButton)
    assert first.x_range()[1] - first.x_range()[0] == pytest.approx(0.25 * (high - low), rel=0.1)


def test_tiny_drag_does_not_zoom(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """手が少し動いただけのドラッグでは拡大しない。"""
    _, first, _ = shown_panels
    before = first.x_range()
    _drag(first, 0.5, 0.504, Qt.MouseButton.LeftButton)
    assert first.x_range() == pytest.approx(before)


def test_right_drag_moves_time_axis(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """右ドラッグは表示の幅を変えずに時間軸を動かす（マウスと同じ向きに内容が動く）。"""
    _, first, _ = shown_panels
    first.set_x_range((4.0, 6.0))
    _drag(first, 0.25, 0.75, Qt.MouseButton.RightButton)
    low, high = first.x_range()
    assert high - low == pytest.approx(2.0, abs=0.05)
    assert low < 4.0


def test_double_click_fits_all(
    qtbot: QtBot, shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel]
) -> None:
    """ダブルクリックで全体の表示に戻る。"""
    _, first, second = shown_panels
    first.set_x_range((4.0, 5.0))
    viewport = first.plot_widget.viewport()
    point = _plot_point(first, 0.5)
    left, no_key = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    # 実際の順序（押す・離す・ダブルクリック・離す）で送る。PyQtGraph は離したときに処理する。
    QTest.mouseClick(viewport, left, no_key, point)
    QTest.mouseDClick(viewport, left, no_key, point)
    QTest.mouseRelease(viewport, left, no_key, point)
    qtbot.wait(20)
    low, high = second.x_range()
    assert low < TIME[0]
    assert high > TIME[-1]


def test_click_places_cursor_without_zooming(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """クリックだけならカーソルが移り、表示の範囲は変わらない。"""
    _, first, second = shown_panels
    before = first.x_range()
    viewport = first.plot_widget.viewport()
    QTest.mouseClick(
        viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _plot_point(first, 0.5)
    )
    assert first.x_range() == pytest.approx(before)
    assert second.cursor_time() == pytest.approx(sum(before) / 2, abs=0.3)


def test_time_scroll_follows_zoom_and_moves_view(
    shown_panels: tuple[StackedPlotWorkspace, PlotPanel, PlotPanel],
) -> None:
    """スクロールバーは表示の位置を示し、動かすと幅を保ったまま移動する。"""
    workspace, first, second = shown_panels
    scroll = workspace.time_scroll
    assert scroll.isVisible()
    assert scroll.maximum() == 0  # 全体を表示している間は動かせない
    first.set_x_range((2.0, 3.0))
    assert scroll.maximum() > 0
    assert scroll.value() == pytest.approx(scroll.maximum() * 2.0 / 9.0, rel=0.05)
    scroll.setValue(scroll.maximum() // 2)
    low, high = second.x_range()
    assert high - low == pytest.approx(1.0, abs=0.02)
    assert low > 4.0


def test_time_scroll_is_hidden_without_data(workspace: StackedPlotWorkspace) -> None:
    """データがなければスクロールバーは出さない。"""
    assert not workspace.time_scroll.isVisibleTo(workspace)
    series = _series(_source(uuid4(), "a"), "x")
    workspace.show_series(series)
    assert workspace.time_scroll.isVisibleTo(workspace)
    workspace.hide_series(series.series_id)
    assert not workspace.time_scroll.isVisibleTo(workspace)


# --- 図を見れば何の図かが分かること ---------------------------------------------------


def _named(source: Source, channel: str, unit: str = "m/s^2") -> PlotSeries:
    """チャンネル名と単位を持つ生データの系列を作る。

    Args:
        source: 時間軸を決める Source。
        channel: チャンネル名。
        unit: 単位。

    Returns:
        生データの系列。
    """
    series = _series(source, channel)
    return series.model_copy(
        update={"channel_name": channel, "unit": unit, "label": f"{source.name} / {channel}"}
    )


def _result_of(raw: PlotSeries, name: str) -> PlotSeries:
    """生データに対する処理の結果の系列を作る。

    Args:
        raw: 生データの系列。
        name: 処理の名前（"accel LPF 10 Hz" の形）。

    Returns:
        結果の系列。
    """
    return raw.model_copy(update={"series_id": uuid4(), "label": name, "analysis_id": uuid4()})


def test_panel_heading_shows_channel_and_unit_without_numbers(
    workspace: StackedPlotWorkspace,
) -> None:
    """見出しはチャンネル名と単位で、クリックするまで説明のない数値を出さない。"""
    workspace.show_series(_named(_source(uuid4(), "logger"), "accel"))
    text = workspace.panels[0].readout_label.text()
    assert "<b>accel</b> [m/s^2]" in text
    assert "t =" not in text
    assert "logger" not in text  # ファイルが 1 つなら、ファイル名は見分けに要らない
    assert workspace.panels[0].cursor_time() is None


def test_click_shows_value_only_then(workspace: StackedPlotWorkspace) -> None:
    """カーソルを置くと、時刻と値が出る。"""
    raw = _named(_source(uuid4(), "logger"), "accel")
    workspace.show_series(raw)
    panel = workspace.panels[0]
    panel.place_cursor(2.0)
    assert panel.cursor_time() == pytest.approx(2.0)
    assert "t = 2 s" in panel.readout_label.text()
    assert "m/s^2" in panel.readout_label.text()


def test_result_makes_raw_line_grey_and_named(workspace: StackedPlotWorkspace) -> None:
    """結果を重ねると、元の線は「生データ」としてグレーになり、結果は条件の名前で出る。"""
    raw = _named(_source(uuid4(), "logger"), "accel")
    result = _result_of(raw, "accel LPF 10 Hz")
    workspace.add_series(raw)
    panel = workspace.panels[0]
    workspace.add_series(result)
    text = panel.readout_label.text()
    assert "<b>accel</b> [m/s^2]" in text
    assert "生データ" in text
    assert "LPF 10 Hz" in text
    assert "accel LPF" not in text
    assert RAW_BEHIND_RESULT in text
    workspace.hide_series(result.series_id)
    text = panel.readout_label.text()
    assert RAW_BEHIND_RESULT not in text  # 比べる相手がなければ元の色に戻す
    assert "生データ" not in text


def test_result_alone_has_full_name(workspace: StackedPlotWorkspace) -> None:
    """結果だけのパネル（取り除いた成分など）は、何の結果かが分かる名前を見出しにする。"""
    raw = _named(_source(uuid4(), "logger"), "accel")
    workspace.show_series(_result_of(raw, "accel LPF 10 Hz: 取り除いた成分"))
    text = workspace.panels[0].readout_label.text()
    assert "accel LPF 10 Hz: 取り除いた成分" in text


def test_two_channels_in_a_panel_keep_their_names(workspace: StackedPlotWorkspace) -> None:
    """チャンネルを重ねたパネルでは、どちらも「生データ」と呼ばず名前で見分ける。"""
    source = _source(uuid4(), "logger")
    workspace.add_series(_named(source, "accel"))
    workspace.add_series(_named(source, "force", "N"))
    text = workspace.panels[0].readout_label.text()
    assert "<b>accel</b>" in text
    assert "<b>force</b>" in text
    assert "生データ" not in text


def test_file_name_is_added_when_two_files_are_shown(workspace: StackedPlotWorkspace) -> None:
    """2 つのファイルを並べたときだけ、見出しにファイル名を添える。"""
    trial_id = uuid4()
    workspace.show_series(_named(_source(trial_id, "first"), "accel"))
    other = _named(_source(trial_id, "second"), "accel")
    workspace.show_series(other)
    assert "<b>first / accel</b>" in workspace.panels[0].readout_label.text()
    assert "<b>second / accel</b>" in workspace.panels[1].readout_label.text()
    workspace.hide_series(other.series_id)
    assert "<b>accel</b>" in workspace.panels[0].readout_label.text()


def test_range_band_is_labeled_only_on_top_panel(workspace: StackedPlotWorkspace) -> None:
    """解析する範囲の帯には、何の帯かを示す文字を一番上のパネルにだけ添える。"""
    source = _source(uuid4(), "logger")
    workspace.show_series(_named(source, "a"))
    workspace.show_series(_named(source, "b"))
    workspace.show_range_selector(time_domain_key(source), 2.0, 4.0)
    first, second = workspace.panels[0].region_item(), workspace.panels[1].region_item()
    assert first is not None
    assert second is not None
    assert len(first.lines[0].childItems()) > len(second.lines[0].childItems())
