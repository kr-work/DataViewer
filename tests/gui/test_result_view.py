"""ui/result_view.py（解析結果の指標とグラフの表示）のテスト。"""

from uuid import uuid4

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from experiment_workbench.model import (
    AnalysisChart,
    AnalysisMetric,
    AnalysisResult,
    ChartAnnotation,
    ChartSeries,
    ChartStyle,
)
from experiment_workbench.ui.result_view import AnalysisResultView


@pytest.fixture
def view(qtbot: QtBot) -> AnalysisResultView:
    """空の解析結果の表示を作る。

    Args:
        qtbot: pytest-qt のヘルパー。

    Returns:
        解析結果の表示。
    """
    widget = AnalysisResultView()
    qtbot.addWidget(widget)
    return widget


def _chart(title: str = "グラフ") -> AnalysisChart:
    """すべての描き方の系列を持つグラフを作る。

    Args:
        title: グラフの題名。

    Returns:
        グラフ。
    """
    x = np.linspace(1.0, 10.0, 20)
    return AnalysisChart(
        title=title,
        x_label="入力 [N]",
        y_label="出力 [V]",
        series=[
            ChartSeries(label="点", x=x, y=x * 2, style=ChartStyle.POINTS, color="#d62728"),
            ChartSeries(label="線", x=x, y=x),
            ChartSeries(label="破線", x=x, y=x + 1, style=ChartStyle.DASHED),
            ChartSeries(label="印", x=x[:1], y=x[:1], style=ChartStyle.MARKERS),
        ],
    )


def _cell(view: AnalysisResultView, row: int, column: int) -> str:
    """指標の表のセルの文字列を返す。

    Args:
        view: 解析結果の表示。
        row: 行。
        column: 列。

    Returns:
        セルの文字列。
    """
    item = view.metrics_table.item(row, column)
    assert item is not None
    return item.text()


def test_metrics_are_listed_with_short_time_in_ms(view: AnalysisResultView) -> None:
    """指標を表に並べ、1 秒未満の時間はミリ秒で、求められない値は「—」で表示する。"""
    analysis_id = uuid4()
    result = AnalysisResult(
        analysis_id=analysis_id,
        series=[],
        summary=["範囲: 0〜1 s"],
        metrics=[
            AnalysisMetric(key="delay", label="遅れ", value=0.01234, unit="s", note="定義"),
            AnalysisMetric(key="long", label="長い時間", value=12.5, unit="s"),
            AnalysisMetric(key="gain", label="ゲイン", value=0.5, unit="V/N"),
            AnalysisMetric(key="none", label="整定時間", value=None, unit="s"),
        ],
    )
    view.show_result(analysis_id, "応答", result)
    assert view.analysis_id == analysis_id
    assert view.title_label.text() == "応答"
    assert (_cell(view, 0, 1), _cell(view, 0, 2)) == ("12.34", "ms")
    assert (_cell(view, 1, 1), _cell(view, 1, 2)) == ("12.5", "s")
    assert (_cell(view, 2, 1), _cell(view, 2, 2)) == ("0.5", "V/N")
    assert _cell(view, 3, 1) == "—"
    item = view.metrics_table.item(0, 0)
    assert item is not None
    assert item.toolTip() == "定義"
    assert view.summary_label.text() == "範囲: 0〜1 s"


def test_charts_are_replaced_by_next_result(view: AnalysisResultView) -> None:
    """結果を表示し直すと、グラフは新しい結果のものだけになる（すべての描き方を描ける）。"""
    first = AnalysisResult(analysis_id=uuid4(), series=[], charts=[_chart("a"), _chart("b")])
    view.show_result(first.analysis_id, "a", first)
    assert view.chart_tabs.count() == 2
    assert not view.chart_tabs.tabBar().isHidden()
    second = AnalysisResult(analysis_id=uuid4(), series=[], charts=[_chart("c")])
    view.show_result(second.analysis_id, "c", second)
    assert view.chart_tabs.count() == 1
    assert view.chart_tabs.tabBar().isHidden()  # 1 つならタブは出さない


def test_clear_shows_guidance(view: AnalysisResultView) -> None:
    """表示を消すと、指標とグラフがなくなり、案内を表示する。"""
    result = AnalysisResult(
        analysis_id=uuid4(),
        series=[],
        metrics=[AnalysisMetric(key="a", label="a", value=1.0)],
        charts=[_chart()],
    )
    view.show_result(result.analysis_id, "a", result)
    view.clear()
    assert view.analysis_id is None
    assert view.metrics_table.rowCount() == 0
    assert view.chart_tabs.count() == 0
    assert "ステップ応答" in view.summary_label.text()


def _shown_plot_item(view: AnalysisResultView, index: int = 0) -> pg.PlotItem:
    """表示しているグラフの PlotItem を返す。

    Args:
        view: 解析結果の表示。
        index: グラフのタブの番号。

    Returns:
        PlotItem。
    """
    widget = view.chart_tabs.widget(index)
    assert isinstance(widget, pg.PlotWidget)
    return widget.getPlotItem()


def test_headline_is_shown_first(view: AnalysisResultView) -> None:
    """結論の文を見出しに出し、消すと空に戻る。"""
    analysis_id = uuid4()
    result = AnalysisResult(analysis_id=analysis_id, series=[], headline="出力は遅れました。")
    view.show_result(analysis_id, "応答", result)
    assert view.headline_label.text() == "出力は遅れました。"
    view.clear()
    assert view.headline_label.text() == ""


def test_chart_title_is_drawn_in_the_figure(view: AnalysisResultView) -> None:
    """題名を図の中に書く（タブが 1 つのときは見出しが隠れるため）。"""
    analysis_id = uuid4()
    result = AnalysisResult(analysis_id=analysis_id, series=[], charts=[_chart("何の図か")])
    view.show_result(analysis_id, "応答", result)
    assert "何の図か" in _shown_plot_item(view).titleLabel.text


def test_only_marked_series_appear_in_legend(view: AnalysisResultView) -> None:
    """背景の点や基準線を凡例に出さず、読み取る線だけを出す。"""
    x = np.linspace(1.0, 10.0, 20)
    chart = AnalysisChart(
        title="図",
        x_label="x",
        y_label="y",
        series=[
            ChartSeries(label="背景", x=x, y=x, style=ChartStyle.POINTS, show_in_legend=False),
            ChartSeries(label="基準", x=x, y=x, style=ChartStyle.DASHED, show_in_legend=False),
            ChartSeries(label="読み取る線", x=x, y=x),
        ],
    )
    analysis_id = uuid4()
    view.show_result(
        analysis_id, "図", AnalysisResult(analysis_id=analysis_id, series=[], charts=[chart])
    )
    legend = _shown_plot_item(view).legend
    assert [label.text for _, label in legend.items] == ["読み取る線"]


def test_annotations_are_written_on_the_figure(view: AnalysisResultView) -> None:
    """注記は、図の上に縦線と文字で書き込む。"""
    chart = _chart().model_copy(
        update={
            "annotations": [
                ChartAnnotation(x=2.0, text="② 動き始め 9 ms"),
                ChartAnnotation(x=3.0, text="④ 落ち着く 75 ms"),
            ]
        }
    )
    analysis_id = uuid4()
    view.show_result(
        analysis_id, "図", AnalysisResult(analysis_id=analysis_id, series=[], charts=[chart])
    )
    lines = [item for item in _shown_plot_item(view).items if isinstance(item, pg.InfiniteLine)]
    assert [line.value() for line in lines] == [2.0, 3.0]
    assert [line.label.format for line in lines] == ["② 動き始め 9 ms", "④ 落ち着く 75 ms"]


def test_conditions_are_folded_and_summary_is_always_shown(view: AnalysisResultView) -> None:
    """結果に関わる補足は見せ、定型の計算の条件は畳んでおく。"""
    analysis_id = uuid4()
    result = AnalysisResult(
        analysis_id=analysis_id,
        series=[],
        summary=["補間: 線形補間（出力）"],
        conditions=["範囲: 0〜1 s"],
    )
    view.show_result(analysis_id, "応答", result)
    assert view.summary_label.isVisibleTo(view)
    assert view.conditions_section.isVisibleTo(view)
    assert not view.conditions_label.isVisibleTo(view)
    view.conditions_section.toggle_button.click()
    assert view.conditions_label.isVisibleTo(view)


def test_empty_summary_takes_no_space(view: AnalysisResultView) -> None:
    """補足がなければ欄を出さない。"""
    analysis_id = uuid4()
    view.show_result(analysis_id, "応答", AnalysisResult(analysis_id=analysis_id, series=[]))
    assert not view.summary_label.isVisibleTo(view)
    assert not view.conditions_section.isVisibleTo(view)


def test_annotation_text_has_no_background(view: AnalysisResultView) -> None:
    """注記の文字の背景は塗らず、指定の側に置く（線を隠さないように）。"""
    chart = _chart().model_copy(
        update={"annotations": [ChartAnnotation(x=2.0, text="左", height=0.5, on_left=True)]}
    )
    analysis_id = uuid4()
    view.show_result(
        analysis_id, "図", AnalysisResult(analysis_id=analysis_id, series=[], charts=[chart])
    )
    (line,) = [item for item in _shown_plot_item(view).items if isinstance(item, pg.InfiniteLine)]
    assert line.label.fill.style() == Qt.BrushStyle.NoBrush
    assert line.label.orthoPos == pytest.approx(0.5)
    assert line.label.anchors == [(1, 0.5), (1, 0.5)]
