"""解析結果の表示（SPEC F-12・F-13）: 結論の一文と、説明の入ったグラフと、指標の表。

時系列を出力しない解析（ステップ応答・ヒステリシス・相互相関）の結果を表示する。
画面の下のドックに置く。まず結論の文を読み、グラフの題名と注記で確かめ、細かい数値は右の表で
見る順に並べている（用語の表から読み解かなくても、何が起きたかが分かるように）。
表とグラフの中身は AnalysisResult にすべて入っており、このモジュールは描くだけを受け持つ。
解析の種類ごとの分岐は持たない（Phase 5 の周波数解析も同じ部品で表示するため）。
"""

from uuid import UUID

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.model import (
    AnalysisChart,
    AnalysisMetric,
    AnalysisResult,
    ChartAnnotation,
    ChartSeries,
    ChartStyle,
)
from experiment_workbench.ui.analysis_panel import CollapsibleSection
from experiment_workbench.ui.theme import (
    BORDER,
    MUTED,
    TEXT,
    TICK_FONT_POINT_SIZE,
    series_color,
    ui_font,
)

_NO_VALUE = "—"
"""求められなかった指標の表示。"""
_LINE_WIDTH = 1.6
_POINT_SIZE = 3
_POINT_ALPHA = 90
_MARKER_SIZE = 11
_TABLE_WIDTH = 330
_TABLE_MIN_HEIGHT = 40
_ANNOTATION_POSITIONS = (0.94, 0.8, 0.66, 0.52)
"""グラフの注記の文字を置く高さ（近い位置の注記が重ならないよう、順に変える）。"""


class AnalysisResultView(QWidget):
    """解析の結果の指標とグラフを表示する部品。

    Attributes:
        title_label: 解析の名前を表示するラベル。
        headline_label: 結果を一文で述べたラベル。
        metrics_table: 指標の表（項目・値・単位）。行のツールチップに指標の定義を出す。
        summary_label: 読み手に必ず見せる補足（補間などの変換や注意点）のラベル。
        conditions_label: 計算の条件のラベル（普段は畳んでおく）。
        chart_tabs: グラフのタブ。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる（結果はまだ表示しない）。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._analysis_id: UUID | None = None
        self.title_label = QLabel()
        self.title_label.setStyleSheet(f"color: {MUTED};")
        self.headline_label = QLabel()
        self.headline_label.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.headline_label.setWordWrap(True)
        self.headline_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["項目", "値", "単位"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.metrics_table.setAlternatingRowColors(True)
        # 既定の最小の高さのままだと、結果がない間も下のドックが大きくなりプロットが狭くなる。
        self.metrics_table.setMinimumHeight(_TABLE_MIN_HEIGHT)
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {MUTED};")
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.conditions_label = QLabel()
        self.conditions_label.setWordWrap(True)
        self.conditions_label.setStyleSheet(f"color: {MUTED};")
        self.conditions_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.conditions_section = CollapsibleSection("計算の条件", self.conditions_label)
        self.chart_tabs = QTabWidget()
        self.chart_tabs.setDocumentMode(True)

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(4, 0, 4, 4)
        details_layout.addWidget(self.metrics_table, stretch=1)
        details_layout.addWidget(self.summary_label)
        details_layout.addWidget(self.conditions_section)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.chart_tabs)
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([2 * _TABLE_WIDTH, _TABLE_WIDTH])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 0)
        layout.setSpacing(2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.headline_label)
        layout.addWidget(splitter, stretch=1)
        self.clear()

    @property
    def analysis_id(self) -> UUID | None:
        """表示している解析の ID（何も表示していなければ None）。"""
        return self._analysis_id

    def show_result(self, analysis_id: UUID, title: str, result: AnalysisResult) -> None:
        """解析の結果を表示する（前の結果は消える）。

        Args:
            analysis_id: 解析の ID。
            title: 解析の名前。
            result: 解析の結果。
        """
        self._analysis_id = analysis_id
        self.title_label.setText(title)
        self.headline_label.setText(result.headline)
        self.metrics_table.setRowCount(len(result.metrics))
        for row, metric in enumerate(result.metrics):
            value, unit = _display_value(metric)
            cells = [metric.label, value, unit]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if metric.note:
                    item.setToolTip(metric.note)
                if column == 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.metrics_table.setItem(row, column, item)
        self.metrics_table.resizeRowsToContents()
        self.summary_label.setText("\n".join(result.summary))
        self.summary_label.setVisible(bool(result.summary))
        self.conditions_label.setText("\n".join(result.conditions))
        self.conditions_section.setVisible(bool(result.conditions))
        self._remove_charts()
        for chart in result.charts:
            self.chart_tabs.addTab(_chart_widget(chart), chart.title)
        self.chart_tabs.tabBar().setVisible(self.chart_tabs.count() > 1)

    def clear(self) -> None:
        """表示を消して、案内を出す。"""
        self._analysis_id = None
        self.title_label.setText("ここに解析の結果を表示します")
        self.headline_label.clear()
        self.metrics_table.setRowCount(0)
        self.summary_label.setText(
            "右の処理で「ステップ応答」「ヒステリシス」「相互相関」を選ぶと結果が出ます。"
        )
        self.summary_label.setVisible(True)
        self.conditions_label.clear()
        self.conditions_section.setVisible(False)
        self._remove_charts()

    def _remove_charts(self) -> None:
        """グラフのタブをすべて取り除く。"""
        while self.chart_tabs.count():
            widget = self.chart_tabs.widget(0)
            self.chart_tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()


def _display_value(metric: AnalysisMetric) -> tuple[str, str]:
    """指標の値を表に出す文字列にする（1 秒未満の時間はミリ秒で表す）。

    遅れや時定数はミリ秒の桁で読むことが多いため。保存される値は秒のまま。

    Args:
        metric: 指標。

    Returns:
        (値の文字列, 単位の文字列)。
    """
    unit = metric.unit or ""
    if metric.value is None:
        return _NO_VALUE, unit
    if unit == "s" and abs(metric.value) < 1.0:
        return f"{metric.value * 1000.0:.4g}", "ms"
    return f"{metric.value:.5g}", unit


def _chart_widget(chart: AnalysisChart) -> pg.PlotWidget:
    """グラフ 1 つを描いた部品を作る。

    系列の順に重ねて描くので、点（散布図）を先に、線や印を後に並べると線が手前に出る。
    題名を図の上に書く。タブが 1 つのときはタブの見出しを隠すので、図の中にないと何の図か
    分からないため。

    Args:
        chart: グラフの内容。

    Returns:
        プロットの部品。
    """
    widget = pg.PlotWidget()
    item = widget.getPlotItem()
    item.setMenuEnabled(False)
    item.showGrid(x=True, y=True, alpha=0.15)
    item.setTitle(chart.title, color=TEXT, size="11pt")
    item.setLabel("bottom", chart.x_label)
    item.setLabel("left", chart.y_label)
    item.setLogMode(chart.log_x, chart.log_y)
    tick_font = ui_font(TICK_FONT_POINT_SIZE)
    for side in ("left", "bottom"):
        axis = item.getAxis(side)
        axis.setTickFont(tick_font)
        axis.setPen(pg.mkPen(BORDER))
        axis.setTextPen(pg.mkPen(TEXT))
        axis.enableAutoSIPrefix(enable=False)
    # 注記の文字は図の上に並ぶので、凡例は右下に置いて重ならないようにする。
    legend = item.addLegend(offset=(-8, -8), brush=pg.mkBrush(255, 255, 255, 215))
    legend.setLabelTextColor(TEXT)
    for index, series in enumerate(chart.series):
        _draw_series(item, series, series.color or series_color(index))
    for index, annotation in enumerate(chart.annotations):
        _draw_annotation(item, annotation, index)
    return widget


def _draw_annotation(item: pg.PlotItem, annotation: ChartAnnotation, index: int) -> None:
    """グラフに注記（縦線と文字）を書き込む。

    Args:
        item: 描く先の PlotItem。
        annotation: 注記。
        index: 注記の番号（文字の高さを順に変える）。
    """
    position = _ANNOTATION_POSITIONS[index % len(_ANNOTATION_POSITIONS)]
    line = pg.InfiniteLine(
        pos=annotation.x,
        angle=90,
        movable=False,
        pen=pg.mkPen(MUTED, width=1, style=Qt.PenStyle.DotLine),
        label=annotation.text,
        labelOpts={
            "position": position,
            "color": TEXT,
            "fill": (255, 255, 255, 225),
            "anchors": [(0, 0.5), (0, 0.5)],
        },
    )
    item.addItem(line, ignoreBounds=True)


def _draw_series(item: pg.PlotItem, series: ChartSeries, color: str) -> None:
    """グラフの系列を、指定の描き方で描く。

    Args:
        item: 描く先の PlotItem。
        series: 系列。
        color: 色。
    """
    x, y = np.asarray(series.x), np.asarray(series.y)
    name = series.label if series.show_in_legend else None
    if series.style is ChartStyle.POINTS:
        faded = QColor(color)
        faded.setAlpha(_POINT_ALPHA)
        item.plot(
            x,
            y,
            pen=None,
            symbol="o",
            symbolSize=_POINT_SIZE,
            symbolPen=None,
            symbolBrush=pg.mkBrush(faded),
            name=name,
        )
    elif series.style is ChartStyle.MARKERS:
        item.plot(
            x,
            y,
            pen=None,
            symbol="o",
            symbolSize=_MARKER_SIZE,
            symbolPen=pg.mkPen("#ffffff", width=1.5),
            symbolBrush=pg.mkBrush(color),
            name=name,
        )
    else:
        style = Qt.PenStyle.DashLine if series.style is ChartStyle.DASHED else Qt.PenStyle.SolidLine
        pen = pg.mkPen(color, width=_LINE_WIDTH, style=style)
        item.plot(x, y, pen=pen, connect="finite", name=name)
