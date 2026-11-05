"""縦並べ比較（SPEC F-10）: 時間軸を連動させた複数のプロットパネル。

パネルに置くのは表示用の系列（PlotSeries）で、生データのチャンネルと解析の出力の両方を扱う。

時間軸の連動の規則（判定は timebase.time_domain_key に任せる）:
    - 1 つのパネルには、時刻を比較できる系列だけを置ける。
      比較できない系列は理由を示して断る。
    - 時刻を比較できるパネルどうしは、時間軸のズーム・パン・カーソルを連動させる
      （その時間軸で一番上のパネルに合わせる）。空のパネルは基準パネルに合わせる。
    - 基準パネル（上から見て最初にデータを置いたパネル）と時刻を比較できないパネルは
      基準パネルと連動させず、パネル上部に警告を表示する（勝手に揃えない）。
解析の出力は入力チャンネルの Source の時間軸を持つので、入力と同じパネルに重ねられる。

見やすさと操作の方針:
    - プロットの高さを優先する。凡例とカーソルの値はパネル上部の 1 行にまとめ、
      時刻の目盛りは、同じ時間軸のパネルのまとまりの一番下だけに出す。
    - 図を見れば何の図かが分かるようにする。パネルの見出しにチャンネル名と単位を書き、処理の結果を
      重ねたら、元の線を「元データ」としてグレーで後ろに退かせる（処理の前後の違いが見えるように）。
    - 説明のない数値を出さない。カーソルと値は、クリックして読もうとしたときだけ出す。
    - 凡例の色の四角をクリックすると、その系列の表示・非表示を切り替える。
    - 時系列を見るときの操作は、覚えなくても使えるようにする。左ドラッグで選んだ範囲を拡大し、
      ダブルクリックで全体に戻す。ホイールはズーム、下のスクロールバーと右ドラッグは移動。
      動かすのは時間軸だけで、縦軸は見えている範囲に自動で合わせる。
      縦軸のズームは、Shift + ホイール、または縦軸の上でのホイールで行う。
    - プロットをクリックするとカーソルがそこに移る。右クリックで操作のメニューを出す。
    - パネルは系列の表示に合わせて増減する（系列を外して空になったパネルは消す）。
      パネルの数が多いときは、高さを保ったまま縦にスクロールできる。

描画には PyQtGraph の表示用間引き（ピーク保存型）を使う。間引きは表示だけに使い、
カーソルの値は元のデータの最も近いサンプルから読む。帯（カルマンフィルタの ±σ など）は
半透明の塗り、印（観測更新しなかった時刻など）は × で表示する。
解析の時刻の印（TimeMarker）と、解析する範囲の選択は、入力チャンネルの時間軸のパネルに重ねて描く。
pyqtgraph には型情報がないため、pyqtgraph のクラスは継承せずに部品として使う。
"""

import html
from collections.abc import Callable
from uuid import UUID

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent
from PySide6.QtCore import QEvent, QMimeData, QObject, QPointF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.model import PlotPanelState, PlotSeries, PlotView, TimeMarker
from experiment_workbench.timebase import time_domain_key
from experiment_workbench.ui.theme import (
    ACCENT,
    BORDER,
    MUTED,
    RAW_BEHIND_RESULT,
    TEXT,
    TICK_FONT_POINT_SIZE,
    WARNING,
    series_color,
    ui_font,
)

SeriesLoader = Callable[[UUID], PlotSeries | None]
"""系列 ID（チャンネル ID または解析の出力 ID）から表示用の系列を作る関数（なければ None）。"""

SERIES_MIME_TYPE = "application/x-experiment-workbench-series"
"""一覧からパネルへドラッグするデータの種類（中身は系列 ID の文字列）。"""

UNLINKED_WARNING = "⚠ 時間軸が別"
"""基準パネルと時間軸を連動しないパネルの警告（プロットの高さを削らないよう短くする）。"""

UNLINKED_WARNING_DETAIL = (
    "このパネルの系列は、基準のパネル（データのある一番上のパネル）の系列と時刻を比較できない"
    "ため、時間軸を基準のパネルと連動させていません（時刻の原点を勝手に揃えないため）。\n"
    "同じ時間軸のパネルどうしは連動します。同じ試技で「同期済み」として取り込んだファイル"
    "どうしは、同じ時間軸になります。"
)
"""警告のツールチップに出す詳しい説明。"""

EMPTY_PANEL_HINT = "左の一覧でチャンネルにチェックを入れると、ここに表示します"
"""空のパネルに出す案内。"""

GRAPH_HINT = (
    "ドラッグ: 範囲を拡大　／　ダブルクリック: 全体　／　ホイール: ズーム　／　右ドラッグ: 移動"
    "　／　クリック: 値を読む"
)
"""グラフの操作の案内（覚えなくても使えるよう、ステータスバーに常に出す）。"""

_PANEL_STYLE = f"#plotPanel {{ background: #ffffff; border: 1px solid {BORDER}; }}"
_DROP_STYLE = f"#plotPanel {{ background: #eff4ff; border: 2px dashed {ACCENT}; }}"
_BAND_ALPHA = 45
_RAW_LINE_WIDTH = 1.0
_RESULT_LINE_WIDTH = 2.0
_RESULT_Z_VALUE = 10
_CURSOR_Z_VALUE = 50
_LEFT_AXIS_WIDTH = 50
_WHEEL_ZOOM_BASE = 1.02
_MIN_PANEL_HEIGHT = 80
_REGION_ALPHA = 26
_ZOOM_BAND_ALPHA = 60
_MIN_ZOOM_PIXELS = 5
"""ドラッグで拡大する最小の幅 [px]（手が少し動いただけで拡大しないように）。"""
_SCROLL_STEPS = 10_000
"""時間軸のスクロールバーの刻みの数。"""
_MARKER_LABEL_POSITION = 0.9
"""時刻の印の番号を置く高さ。"""
_RESULT_LEGEND_NAME = "元データ"
"""処理の結果を重ねたときの、元の線の凡例の名前。"""
_RANGE_LABEL = "解析する範囲"
"""解析する範囲の帯に添える文字。"""
_TIME_AXIS_HEIGHT = 46
"""時刻の目盛りと軸の名前の高さの目安 [px]（一番下のパネルのプロットが低くならないように足す）。"""
_FIT_MARGIN = 0.01
"""「全体を表示」で、データの両端に空ける余白（範囲に対する割合）。"""


class _DrawnSeries:
    """パネルに描画した 1 系列の部品をまとめて持つ（モジュール内部用）。

    Attributes:
        series: 表示している系列。
        color_index: 線の色の番号（再計算で描き直しても色を変えないため）。
        curve: 値の線（PyQtGraph の PlotDataItem）。
        band_items: 帯の部品（下端・上端の見えない線と塗り）。帯がなければ空。
        marker_item: 印の部品（PyQtGraph の ScatterPlotItem）。印がなければ None。
        hidden: 利用者が非表示にしていれば True。
    """

    def __init__(self, series: PlotSeries, color_index: int, curve: pg.PlotDataItem) -> None:
        """描画した部品を受け取る。

        Args:
            series: 表示している系列。
            color_index: 線の色の番号。
            curve: 値の線。
        """
        self.series = series
        self.color_index = color_index
        self.curve = curve
        self.band_items: list[pg.GraphicsObject] = []
        self.marker_item: pg.ScatterPlotItem | None = None
        self.hidden = False


class PlotPanel(QFrame):
    """縦並べの 1 パネル（時刻を比較できる系列の集まり）。

    Attributes:
        plot_widget: PyQtGraph のプロット領域。
        plot_item: プロット領域の PlotItem（軸・カーソルを持つ）。
        cursor_line: 時刻を指すカーソル（縦線）。ドラッグで動かせる。
            （QWidget の cursor() メソッドと衝突しないよう、この名前にしている）
        readout_label: 凡例とカーソル位置の値を 1 行にまとめて表示するラベル。
        warning_label: 時間軸を連動していないときの警告ラベル。
        close_button: パネルを閉じるボタン。
    """

    activated = Signal(object)
    """パネルがクリックされた（引数: 自身）。"""
    cursor_moved = Signal(object, float)
    """カーソルが利用者の操作で動いた（引数: 自身, 時刻 [s]）。"""
    remove_requested = Signal(object)
    """パネルの削除が求められた（引数: 自身）。"""
    series_remove_requested = Signal(object, object)
    """系列の取り外しが求められた（引数: 自身, 系列 ID）。"""
    fit_requested = Signal()
    """全体の表示が求められた。"""
    series_dropped = Signal(object, object)
    """系列がドロップされた（引数: 自身, 系列 ID）。"""
    region_moved = Signal(object, float, float)
    """解析する範囲の選択が動いている（引数: 自身, 開始 [s], 終了 [s]）。"""
    region_finished = Signal(float, float)
    """解析する範囲の選択を動かし終えた（引数: 開始 [s], 終了 [s]）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """パネルの部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self.setMinimumHeight(_MIN_PANEL_HEIGHT)
        self.setAcceptDrops(True)
        self.setStyleSheet(_PANEL_STYLE)
        self._drawn: list[_DrawnSeries] = []
        self._updating_cursor = False
        self._updating_region = False
        # 時間軸を連動させている相手（PyQtGraph の PlotItem）。連動していなければ None。
        self._linked_to: object | None = None
        self._marker_lines: list[pg.InfiniteLine] = []
        self._markers_labeled = False
        self._source_shown = False
        self._region: pg.LinearRegionItem | None = None
        self._zoom_band: pg.LinearRegionItem | None = None

        self.readout_label = QLabel()
        self.readout_label.setTextFormat(Qt.TextFormat.RichText)
        self.readout_label.setWordWrap(True)
        self.readout_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.readout_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.readout_label.linkActivated.connect(self._on_legend_clicked)
        self.warning_label = QLabel(UNLINKED_WARNING)
        self.warning_label.setToolTip(UNLINKED_WARNING_DETAIL)
        self.warning_label.setStyleSheet(f"color: {WARNING};")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.warning_label.setVisible(False)
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.close_button.setAutoRaise(True)
        self.close_button.setToolTip("このパネルを閉じる（データは消えません）")
        self.close_button.clicked.connect(self._on_remove_clicked)
        header = QHBoxLayout()
        header.setContentsMargins(8, 3, 3, 0)
        header.setSpacing(8)
        header.addWidget(self.readout_label, stretch=1)
        header.addWidget(self.warning_label)
        header.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignTop)

        self.plot_widget = pg.PlotWidget()
        # ドロップはパネル全体で受ける（プロット領域がドラッグを横取りしないように）。
        self.plot_widget.setAcceptDrops(False)
        self.plot_widget.viewport().setAcceptDrops(False)
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_item = self.plot_widget.getPlotItem()
        self._configure_plot()
        self.cursor_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(MUTED, width=1))
        self.cursor_line.setZValue(_CURSOR_Z_VALUE)
        self.cursor_line.setVisible(False)
        self.plot_item.addItem(self.cursor_line, ignoreBounds=True)
        self.cursor_line.sigPositionChanged.connect(self._on_cursor_dragged)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.plot_widget, stretch=1)
        self._show_empty_hint()

    def _configure_plot(self) -> None:
        """プロット領域の軸・グリッド・マウス操作を設定する。

        PyQtGraph の標準の右クリックメニューは英語で項目も多いため使わず、
        自前の日本語のメニューに置き換える。縦軸の名前は出さない（単位はパネル上部の凡例に
        値と一緒に出る。回転した「V」は読みにくく、横幅も取るため）。
        """
        item = self.plot_item
        item.showGrid(x=True, y=True, alpha=0.15)
        item.setLabel("bottom", "時刻", units="s")
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setContentsMargins(2, 2, 8, 2)
        tick_font = ui_font(TICK_FONT_POINT_SIZE)
        for side in ("left", "bottom"):
            axis = item.getAxis(side)
            axis.setTickFont(tick_font)
            axis.setPen(pg.mkPen(BORDER))
            axis.setTextPen(pg.mkPen(TEXT))
        left_axis = item.getAxis("left")
        left_axis.setWidth(_LEFT_AXIS_WIDTH)
        left_axis.enableAutoSIPrefix(enable=False)
        view_box = item.getViewBox()
        view_box.setMouseEnabled(x=True, y=False)
        view_box.setAutoVisible(y=True)
        view_box.enableAutoRange(axis="y", enable=True)
        # ドラッグの意味だけを変える。クラスは継承せず、この部品のドラッグの処理を差し替える
        # （カーソルや範囲の端のドラッグは、その部品が先に受けるのでここには来ない）。
        view_box.mouseDragEvent = self._on_view_dragged

    # --- 状態の参照 ----------------------------------------------------------

    @property
    def series(self) -> list[PlotSeries]:
        """パネルに表示している系列（表示順のコピー）。"""
        return [drawn.series for drawn in self._drawn]

    def series_ids(self) -> list[UUID]:
        """パネルに表示している系列の ID を表示順に返す。

        Returns:
            系列 ID のリスト。
        """
        return [drawn.series.series_id for drawn in self._drawn]

    def is_series_hidden(self, series_id: UUID) -> bool:
        """系列を非表示にしているかを返す。

        Args:
            series_id: 系列 ID。

        Returns:
            非表示なら True（パネルにない系列も False）。
        """
        return any(d.hidden for d in self._drawn if d.series.series_id == series_id)

    def time_domain(self) -> str | None:
        """パネルの時間軸の識別子を返す（空のパネルは None）。

        Returns:
            timebase.time_domain_key の値、または None。
        """
        return time_domain_key(self._drawn[0].series.source) if self._drawn else None

    def accepts(self, series: PlotSeries) -> bool:
        """系列をこのパネルに置けるか（時刻を比較できるか）を返す。

        Args:
            series: 置きたい系列。

        Returns:
            空のパネル、または時間軸が同じなら True。
        """
        domain = self.time_domain()
        return domain is None or domain == time_domain_key(series.source)

    def cursor_time(self) -> float | None:
        """カーソルの時刻を返す（カーソルを表示していなければ None）。

        Returns:
            時刻 [s]、または None。
        """
        return float(self.cursor_line.value()) if self.cursor_line.isVisible() else None

    def x_range(self) -> tuple[float, float]:
        """表示している時間軸の範囲を返す。

        Returns:
            (最小, 最大) [s]。
        """
        low, high = self.plot_item.getViewBox().viewRange()[0]
        return float(low), float(high)

    def y_range_state(self) -> tuple[float, float] | None:
        """縦軸の範囲を保存用に返す（自動調整中なら None）。

        Returns:
            (最小, 最大)、または None。
        """
        view_box = self.plot_item.getViewBox()
        if view_box.autoRangeEnabled()[1]:
            return None
        low, high = view_box.viewRange()[1]
        return float(low), float(high)

    def data_range(self) -> tuple[float, float] | None:
        """表示している系列の時刻の範囲を返す。

        Returns:
            (最小, 最大) [s]。系列がなければ None。
        """
        if not self._drawn:
            return None
        lows = [float(np.min(drawn.series.time_s)) for drawn in self._drawn]
        highs = [float(np.max(drawn.series.time_s)) for drawn in self._drawn]
        return min(lows), max(highs)

    def marker_count(self) -> int:
        """表示している時刻の印の数を返す。

        Returns:
            時刻の印の数。
        """
        return len(self._marker_lines)

    def has_marker_labels(self) -> bool:
        """時刻の印に名前を添えているかを返す。

        Returns:
            名前を添えていれば True。
        """
        return self._markers_labeled and bool(self._marker_lines)

    def region_range(self) -> tuple[float, float] | None:
        """解析する範囲の選択を返す。

        Returns:
            (開始, 終了) [s]。選択を表示していなければ None。
        """
        if self._region is None:
            return None
        low, high = self._region.getRegion()
        return float(low), float(high)

    def region_item(self) -> pg.LinearRegionItem | None:
        """解析する範囲の選択の部品を返す。

        Returns:
            範囲の選択の部品。表示していなければ None。
        """
        return self._region

    # --- 操作 ----------------------------------------------------------------

    def add_series(self, series: PlotSeries) -> None:
        """系列を描画する（時間軸の判定は呼び出し側で済ませておく）。

        線の色は、パネルの中でまだ使っていない番号のうち最も小さいものにする
        （どのパネルでも、最初の系列は同じ色になる）。
        欠損（NaN）の区間は線をつながずに空白として表示する。時刻が単調増加でないデータでは、
        表示範囲の切り出し（clipToView）が正しく働かないため使わない。

        Args:
            series: 描画する系列。
        """
        used = {drawn.color_index for drawn in self._drawn}
        color_index = next(index for index in range(len(used) + 1) if index not in used)
        self._drawn.append(self._draw(series, color_index))
        self._restyle_raw_lines()

    def update_series(self, series: PlotSeries) -> bool:
        """表示中の系列を、同じ ID の新しいデータで描き直す（色・順序・非表示は変えない）。

        Args:
            series: 新しいデータ。

        Returns:
            表示していた系列なら True（描き直した）、表示していなければ False。
        """
        for index, drawn in enumerate(self._drawn):
            if drawn.series.series_id == series.series_id:
                self._erase(drawn)
                redrawn = self._draw(series, drawn.color_index)
                redrawn.hidden = drawn.hidden
                self._apply_visibility(redrawn)
                self._drawn[index] = redrawn
                self._restyle_raw_lines()
                self._refresh_readout()
                return True
        return False

    def remove_series(self, series_id: UUID) -> None:
        """系列の描画を取り除く。

        Args:
            series_id: 取り除く系列の ID。表示していなければ何もしない。
        """
        for drawn in list(self._drawn):
            if drawn.series.series_id == series_id:
                self._erase(drawn)
                self._drawn.remove(drawn)
        if not self._drawn:
            self.cursor_line.setVisible(False)
            self._show_empty_hint()
        else:
            self._restyle_raw_lines()
            self._refresh_readout()

    def set_series_hidden(self, series_id: UUID, *, hidden: bool) -> None:
        """系列の表示・非表示を切り替える（データはパネルに残る）。

        Args:
            series_id: 系列 ID。
            hidden: 非表示にするなら True。
        """
        for drawn in self._drawn:
            if drawn.series.series_id == series_id:
                drawn.hidden = hidden
                self._apply_visibility(drawn)
        self._refresh_readout()

    def link_to(self, master: "PlotPanel | None") -> None:
        """時間軸を基準パネルに連動させる（None なら連動を解除する）。

        連動中は時間軸の自動調整が止まるため、連動を解除したときは自動調整を戻して
        自分のデータ全体が見えるようにする（連動していた範囲のまま残さない）。

        Args:
            master: 基準パネル。自身または None なら連動しない。
        """
        target = None if master is None or master is self else master.plot_item
        if target is self._linked_to:
            return
        was_linked = self._linked_to is not None
        self.plot_item.setXLink(target)
        self._linked_to = target
        if target is None and was_linked:
            self.plot_item.enableAutoRange(axis="x", enable=True)

    def set_source_shown(self, *, shown: bool) -> None:
        """見出しにファイル名を添えるかを切り替える（複数のファイルを見分けるため）。

        Args:
            shown: 添えるなら True。
        """
        if shown != self._source_shown:
            self._source_shown = shown
            self._refresh_readout()

    def set_unlinked_warning(self, *, visible: bool) -> None:
        """時間軸を連動していない警告の表示を切り替える。

        Args:
            visible: 表示するなら True。
        """
        self.warning_label.setVisible(visible)

    def set_time_axis_visible(self, *, visible: bool) -> None:
        """時刻の目盛りと軸の名前の表示を切り替える（グリッドは残す）。

        同じ時間軸のパネルでは、一番下のパネルだけに出してプロットの高さを確保する。

        Args:
            visible: 表示するなら True。
        """
        axis = self.plot_item.getAxis("bottom")
        axis.setStyle(showValues=visible)
        axis.showLabel(visible)
        axis.setHeight(None if visible else 4)

    def set_x_range(self, x_range: tuple[float, float]) -> None:
        """時間軸の範囲を設定する。

        Args:
            x_range: (最小, 最大) [s]。
        """
        self.plot_item.setXRange(x_range[0], x_range[1], padding=0)

    def set_y_range(self, y_range: tuple[float, float]) -> None:
        """縦軸の範囲を設定する（自動調整は解除される）。

        Args:
            y_range: (最小, 最大)。
        """
        self.plot_item.setYRange(y_range[0], y_range[1], padding=0)

    def fit_y(self) -> None:
        """縦軸を、見えている範囲のデータに合わせる自動調整に戻す。"""
        self.plot_item.enableAutoRange(axis="y", enable=True)

    def set_cursor(self, time_s: float) -> None:
        """カーソルを指定の時刻に置き、値の表示を更新する（cursor_moved は送らない）。

        Args:
            time_s: 時刻 [s]。
        """
        self._updating_cursor = True
        try:
            self.cursor_line.setValue(time_s)
            self.cursor_line.setVisible(True)
        finally:
            self._updating_cursor = False
        self.update_readout(time_s)

    def place_cursor(self, time_s: float) -> None:
        """利用者の操作としてカーソルを動かす（同じ時間軸のパネルにも反映される）。

        Args:
            time_s: 時刻 [s]。
        """
        self.cursor_line.setVisible(True)
        self.cursor_line.setValue(time_s)
        self.update_readout(time_s)

    def set_time_markers(self, markers: list[TimeMarker], *, labeled: bool = True) -> None:
        """時刻の印を引き直す。

        Args:
            markers: 時刻の印（空なら消す）。
            labeled: 印に番号を添えるなら True。
        """
        for line in self._marker_lines:
            self.plot_item.removeItem(line)
        self._marker_lines = []
        self._markers_labeled = labeled
        pen = pg.mkPen(TEXT, width=1, style=Qt.PenStyle.DashLine)
        for marker in markers:
            line = pg.InfiniteLine(
                pos=marker.time_s,
                angle=90,
                movable=False,
                pen=pen,
                label=marker.label if labeled else None,
                labelOpts={
                    "position": _MARKER_LABEL_POSITION,
                    "color": TEXT,
                    "fill": (255, 255, 255, 220),
                    "anchors": [(0, 0.5), (0, 0.5)],
                },
            )
            self.plot_item.addItem(line, ignoreBounds=True)
            self._marker_lines.append(line)

    def set_region(self, region: tuple[float, float] | None, *, labeled: bool = False) -> None:
        """解析する範囲の選択を表示する（None なら消す）。

        範囲の両端の線だけをドラッグできるようにする。範囲の中でも、パンやカーソルの操作が
        できるようにするため（本体もドラッグできると、範囲の中でパンできなくなる）。

        Args:
            region: (開始, 終了) [s]。None なら消す。
            labeled: 帯が何かを示す文字を添えるなら True（作るときだけ効く）。
        """
        if region is None:
            if self._region is not None:
                self.plot_item.removeItem(self._region)
                self._region = None
            return
        if self._region is not None:
            self._updating_region = True
            try:
                self._region.setRegion(region)
            finally:
                self._updating_region = False
            return
        color = QColor(ACCENT)
        color.setAlpha(_REGION_ALPHA)
        item = pg.LinearRegionItem(
            values=region,
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(ACCENT, width=2),
            hoverPen=pg.mkPen(ACCENT, width=3),
        )
        for line in item.lines:
            line.setMovable(True)
        if labeled:
            pg.InfLineLabel(
                item.lines[0],
                text=_RANGE_LABEL,
                position=0.12,
                color=ACCENT,
                anchors=[(0, 0.5), (0, 0.5)],
            )
        item.setZValue(-5)
        self.plot_item.addItem(item, ignoreBounds=True)
        item.sigRegionChanged.connect(self._on_region_changed)
        item.sigRegionChangeFinished.connect(self._on_region_finished)
        self._region = item

    def update_readout(self, time_s: float | None) -> None:
        """見出しと凡例を表示する（カーソルがあれば、その位置の各系列の値も添える）。

        補間はせず、各系列で最も近いサンプルの値を表示する。先頭の系列のサンプルの時刻を
        最後に示し、それと違う時刻のサンプルを読んだ系列には、その時刻を添える（帯がある系列は
        「値 ± 半幅」）。時刻が単調増加でないデータにも対応するため、二分探索ではなく全体から
        最も近いものを探す。色の四角は、系列の表示・非表示を切り替えるリンクにする。

        Args:
            time_s: カーソルの時刻 [s]。カーソルを出していなければ None。
        """
        if not self._drawn:
            self._show_empty_hint()
            return
        first_time: float | None = None
        parts: list[str] = []
        for drawn in self._drawn:
            if time_s is None:
                parts.append(self._legend_entry(drawn))
                continue
            index = int(np.argmin(np.abs(drawn.series.time_s - time_s)))
            sample_time = float(drawn.series.time_s[index])
            if first_time is None:
                first_time = sample_time
            parts.append(self._legend_entry(drawn, index, sample_time, first_time))
        heading = self._heading()
        tail = (
            []
            if first_time is None
            else [f'<span style="color:{MUTED}">t = {first_time:.6g} s</span>']
        )
        self.readout_label.setText(" &nbsp;&nbsp; ".join([*heading, *parts, *tail]))

    # --- ドラッグ＆ドロップ ----------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """一覧からの系列のドラッグを受け入れ、ドロップできることを枠で示す。

        Args:
            event: ドラッグのイベント。
        """
        if event.mimeData().hasFormat(SERIES_MIME_TYPE):
            self.setStyleSheet(_DROP_STYLE)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """ドラッグが離れたら、枠の表示を戻す。

        Args:
            event: ドラッグのイベント。
        """
        self.setStyleSheet(_PANEL_STYLE)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """ドロップされた系列を知らせる。

        Args:
            event: ドロップのイベント。
        """
        self.setStyleSheet(_PANEL_STYLE)
        series_id = series_id_from_mime(event.mimeData())
        if series_id is not None:
            event.acceptProposedAction()
            self.series_dropped.emit(self, series_id)

    # --- 内部処理 ------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の上書き
        """Shift + ホイール、または縦軸の上でのホイールを、縦軸のズームにする。

        縦軸はマウスで動かさない設定（時間軸だけを動かすため）なので、ここで受ける。

        Args:
            watched: イベントの対象。
            event: イベント。

        Returns:
            イベントを処理したら True。
        """
        if event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            position = self.plot_widget.mapToScene(event.position().toPoint())
            view_rect = self.plot_item.getViewBox().sceneBoundingRect()
            over_left_axis = position.x() < view_rect.left()
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if shift or over_left_axis:
                delta = event.angleDelta().y() or event.angleDelta().x()
                self._zoom_y(delta, position)
                return True
        return super().eventFilter(watched, event)

    def _on_view_dragged(self, event: MouseDragEvent, axis: int | None = None) -> None:
        """プロットのドラッグを処理する（PyQtGraph の ViewBox.mouseDragEvent の代わり）。

        左ドラッグは選んだ時間の範囲を拡大する。右・中ボタンのドラッグと、時間軸の上のドラッグは
        時間軸を移動する。縦軸の上のドラッグでは何もしない（縦軸は自動で合わせるため）。

        Args:
            event: PyQtGraph のドラッグのイベント。
            axis: 軸の上のドラッグなら軸の番号（0: 時間軸、1: 縦軸）。プロットの上なら None。
        """
        event.accept()
        if axis == 1:
            return
        if axis == 0 or event.button() != Qt.MouseButton.LeftButton:
            view_box = self.plot_item.getViewBox()
            now = view_box.mapSceneToView(event.scenePos())
            before = view_box.mapSceneToView(event.lastScenePos())
            view_box.translateBy(x=before.x() - now.x())
            return
        self._drag_zoom(event)

    def _drag_zoom(self, event: MouseDragEvent) -> None:
        """ドラッグしている時間の範囲を帯で示し、離したらその範囲を拡大する。

        縦軸の範囲は指定せず、見えている範囲への自動調整に任せる。

        Args:
            event: PyQtGraph のドラッグのイベント。
        """
        view_box = self.plot_item.getViewBox()
        start = float(view_box.mapSceneToView(event.buttonDownScenePos()).x())
        end = float(view_box.mapSceneToView(event.scenePos()).x())
        selected = (min(start, end), max(start, end))
        if self._zoom_band is None:
            color = QColor(ACCENT)
            color.setAlpha(_ZOOM_BAND_ALPHA)
            self._zoom_band = pg.LinearRegionItem(
                values=selected,
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(ACCENT, width=1),
            )
            self.plot_item.addItem(self._zoom_band, ignoreBounds=True)
        self._zoom_band.setRegion(selected)
        if not event.isFinish():
            return
        self.plot_item.removeItem(self._zoom_band)
        self._zoom_band = None
        pixels = abs(event.scenePos().x() - event.buttonDownScenePos().x())
        if pixels >= _MIN_ZOOM_PIXELS:
            self.set_x_range(selected)

    def _zoom_y(self, delta: int, scene_position: QPointF) -> None:
        """縦軸を、マウスの位置を中心にズームする（PyQtGraph のホイールの倍率に合わせる）。

        Args:
            delta: ホイールの回転量（Qt の angleDelta）。
            scene_position: マウスの位置（シーンの座標）。
        """
        view_box = self.plot_item.getViewBox()
        center = view_box.mapSceneToView(scene_position)
        view_box.scaleBy(y=_WHEEL_ZOOM_BASE ** (-delta / 8.0), center=center)

    def _compared_raw(self) -> _DrawnSeries | None:
        """処理の結果と比べている元の線を返す。

        Returns:
            チャンネルが 1 つで処理の結果を重ねているパネルなら、その元の線。それ以外は None。
        """
        raws = [drawn for drawn in self._drawn if drawn.series.analysis_id is None]
        return raws[0] if len(raws) == 1 and len(self._drawn) > 1 else None

    def _channel_title(self, series: PlotSeries) -> str:
        """チャンネル名と単位の見出しの HTML を作る。

        Args:
            series: 系列。

        Returns:
            HTML（複数のファイルを開いているときはファイル名を添える）。
        """
        name = series.channel_name or series.label
        if self._source_shown and series.channel_name:
            name = f"{series.source.name} / {name}"
        unit = f" [{series.unit}]" if series.unit else ""
        return f"<b>{html.escape(name)}</b>{html.escape(unit)}"

    def _heading(self) -> list[str]:
        """パネルの見出しを返す。

        Returns:
            処理の結果と比べているパネルならチャンネルの見出し。それ以外は凡例が見出しを兼ねる
            ので空。
        """
        compared = self._compared_raw()
        return [] if compared is None else [self._channel_title(compared.series)]

    def _legend_name(self, drawn: _DrawnSeries) -> str:
        """凡例の名前の HTML を作る。

        処理の結果と比べているパネルでは、見出しにチャンネル名があるので、元の線は「元データ」、
        結果は条件だけの名前（"LPF 10 Hz"）にする。

        Args:
            drawn: 系列の部品。

        Returns:
            HTML。
        """
        series = drawn.series
        if self._compared_raw() is None:
            if series.analysis_id is None:
                return self._channel_title(series)
            unit = f" [{series.unit}]" if series.unit else ""
            return f"<b>{html.escape(series.label)}</b>{html.escape(unit)}"
        if series.analysis_id is None:
            return _RESULT_LEGEND_NAME
        return html.escape(series.label.removeprefix(f"{series.channel_name} "))

    def _line_color(self, drawn: _DrawnSeries) -> str:
        """線の色を返す。

        Args:
            drawn: 系列の部品。

        Returns:
            色。処理の結果と比べている元の線はグレーにする。
        """
        return (
            RAW_BEHIND_RESULT if drawn is self._compared_raw() else series_color(drawn.color_index)
        )

    def _restyle_raw_lines(self) -> None:
        """元の線の色を、処理の結果と比べているかどうかに合わせる。"""
        for drawn in self._drawn:
            if drawn.series.analysis_id is None:
                drawn.curve.setPen(pg.mkPen(self._line_color(drawn), width=_RAW_LINE_WIDTH))

    def _legend_entry(
        self,
        drawn: _DrawnSeries,
        index: int | None = None,
        sample_time: float | None = None,
        first_time: float | None = None,
    ) -> str:
        """凡例の 1 系列分の HTML を作る。

        Args:
            drawn: 系列の部品。
            index: カーソルに最も近いサンプルの番号。カーソルがなければ None。
            sample_time: そのサンプルの時刻 [s]。
            first_time: 先頭の系列のサンプルの時刻 [s]。

        Returns:
            HTML。
        """
        series = drawn.series
        color = MUTED if drawn.hidden else self._line_color(drawn)
        box = "□" if drawn.hidden else "■"
        link = f'<a href="{series.series_id}" style="color:{color};text-decoration:none;">{box}</a>'
        entry = f"{link}&nbsp;{self._legend_name(drawn)}"
        if index is None:
            return entry
        value = float(series.values[index])
        text = "欠損" if np.isnan(value) else f"{value:.5g}"
        if series.band_lower is not None and series.band_upper is not None:
            half_width = float(series.band_upper[index] - series.band_lower[index]) / 2.0
            if np.isfinite(half_width) and not np.isnan(value):
                text += f" ± {half_width:.3g}"
        unit = f" {series.unit}" if series.unit else ""
        time_note = "" if sample_time == first_time else f" (t = {sample_time:.6g} s)"
        return f"{entry}&nbsp;&nbsp;{html.escape(text + unit)}{time_note}"

    def _refresh_readout(self) -> None:
        """見出しと凡例を表示し直す（カーソルがあれば値も）。"""
        self.update_readout(self.cursor_time())

    def _show_empty_hint(self) -> None:
        """空のパネルの案内を表示する。"""
        self.readout_label.setText(f'<span style="color:{MUTED}">{EMPTY_PANEL_HINT}</span>')

    def _draw(self, series: PlotSeries, color_index: int) -> _DrawnSeries:
        """系列の線・帯・印を描画する。

        解析の出力は、生データより太い線で手前に描く（重ねたときに結果が埋もれないように）。
        太い線のアンチエイリアスは描画が 10 倍以上遅くなり（1 kHz × 180 秒・4 パネルの全体表示で
        1 回 140〜340 ms）、パンやズームが引っかかるため、太い線だけアンチエイリアスを切る。

        Args:
            series: 描画する系列。
            color_index: 線の色の番号。

        Returns:
            描画した部品。
        """
        monotonic = bool(np.all(np.diff(series.time_s) > 0))
        color = series_color(color_index)
        is_result = series.analysis_id is not None
        pen = pg.mkPen(color, width=_RESULT_LINE_WIDTH if is_result else _RAW_LINE_WIDTH)
        curve = self.plot_item.plot(
            series.time_s,
            series.values,
            pen=pen,
            connect="finite",
            antialias=not is_result,
        )
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(monotonic)
        if is_result:
            curve.setZValue(_RESULT_Z_VALUE)
        drawn = _DrawnSeries(series, color_index, curve)
        if series.band_lower is not None and series.band_upper is not None:
            edges = []
            for edge_values in (series.band_lower, series.band_upper):
                edge = pg.PlotDataItem(series.time_s, edge_values, pen=None, connect="finite")
                # 間引き・切り出しの設定はプロットに追加した後に行う（追加前に設定すると、
                # 追加の瞬間に PyQtGraph の内部でエラーになるため）。
                self.plot_item.addItem(edge)
                edge.setDownsampling(auto=True, method="peak")
                edge.setClipToView(monotonic)
                edges.append(edge)
            band_color = QColor(color)
            band_color.setAlpha(_BAND_ALPHA)
            fill = pg.FillBetweenItem(edges[0], edges[1], brush=pg.mkBrush(band_color))
            self.plot_item.addItem(fill)
            drawn.band_items = [*edges, fill]
        if series.marker_times_s is not None and series.marker_times_s.size:
            marker_values = np.interp(series.marker_times_s, series.time_s, series.values)
            markers = pg.ScatterPlotItem(
                x=series.marker_times_s,
                y=marker_values,
                symbol="x",
                size=7,
                pen=pg.mkPen(color, width=1),
                brush=None,
            )
            self.plot_item.addItem(markers)
            drawn.marker_item = markers
        return drawn

    def _erase(self, drawn: _DrawnSeries) -> None:
        """系列の線・帯・印をプロットから取り除く。

        Args:
            drawn: 取り除く部品。
        """
        self.plot_item.removeItem(drawn.curve)
        for item in drawn.band_items:
            self.plot_item.removeItem(item)
        if drawn.marker_item is not None:
            self.plot_item.removeItem(drawn.marker_item)

    def _apply_visibility(self, drawn: _DrawnSeries) -> None:
        """系列の線・帯・印の表示を、非表示の設定に合わせる。

        Args:
            drawn: 系列の部品。
        """
        visible = not drawn.hidden
        drawn.curve.setVisible(visible)
        for item in drawn.band_items:
            item.setVisible(visible)
        if drawn.marker_item is not None:
            drawn.marker_item.setVisible(visible)

    def _open_menu(self) -> None:
        """パネルの操作のメニューを、マウスの位置に開く。"""
        menu = QMenu(self)
        fit_action = menu.addAction("全体を表示")
        fit_action.triggered.connect(self._on_fit_triggered)
        menu.addSeparator()
        for drawn in self._drawn:
            action = menu.addAction(f"「{drawn.series.label}」を表示")
            action.setCheckable(True)
            action.setChecked(not drawn.hidden)
            action.setData(("toggle", str(drawn.series.series_id)))
        if self._drawn:
            remove_menu = menu.addMenu("系列を外す")
            for drawn in self._drawn:
                action = remove_menu.addAction(drawn.series.label)
                action.setData(("remove", str(drawn.series.series_id)))
            menu.addSeparator()
        remove_panel = menu.addAction("パネルを削除")
        remove_panel.triggered.connect(self._on_remove_clicked)
        menu.triggered.connect(self._on_menu_triggered)
        menu.exec(QCursor.pos())

    def _on_menu_triggered(self, action: QAction) -> None:
        """メニューで選んだ系列の操作を行う。

        Args:
            action: 選ばれたメニュー項目（系列の操作なら data に (操作, 系列 ID) を持つ）。
        """
        data = action.data()
        if not isinstance(data, tuple):
            return
        operation, series_id = data[0], UUID(data[1])
        if operation == "toggle":
            self.set_series_hidden(series_id, hidden=not action.isChecked())
        else:
            self.series_remove_requested.emit(self, series_id)

    def _on_fit_triggered(self) -> None:
        """全体の表示を求める。"""
        self.fit_requested.emit()

    def _on_remove_clicked(self) -> None:
        """パネルの削除を求める。"""
        self.remove_requested.emit(self)

    def _on_legend_clicked(self, link: str) -> None:
        """凡例の色の四角がクリックされたら、その系列の表示・非表示を切り替える。

        Args:
            link: リンクの内容（系列 ID の文字列）。
        """
        series_id = UUID(link)
        self.set_series_hidden(series_id, hidden=not self.is_series_hidden(series_id))

    def _on_cursor_dragged(self, line: object) -> None:
        """カーソルが利用者の操作で動いたことを知らせる（プログラムからの移動では知らせない）。

        Args:
            line: 動いたカーソル（PyQtGraph の InfiniteLine）。
        """
        if self._updating_cursor:
            return
        self.cursor_moved.emit(self, float(self.cursor_line.value()))

    def _on_region_changed(self, region: object) -> None:
        """範囲の選択が動いたら、他のパネルに合わせるよう知らせる。

        Args:
            region: 動いた範囲の選択（PyQtGraph の LinearRegionItem）。
        """
        if self._updating_region:
            return
        selected = self.region_range()
        if selected is not None:
            self.region_moved.emit(self, selected[0], selected[1])

    def _on_region_finished(self, region: object) -> None:
        """範囲の選択を動かし終えたことを知らせる。

        Args:
            region: 動いた範囲の選択（PyQtGraph の LinearRegionItem）。
        """
        if self._updating_region:
            return
        selected = self.region_range()
        if selected is not None:
            self.region_finished.emit(selected[0], selected[1])

    def _on_scene_clicked(self, event: MouseClickEvent) -> None:
        """クリックでカーソルを移し、ダブルクリックで全体を表示する（右クリックならメニュー）。

        Args:
            event: PyQtGraph のマウスイベント。
        """
        self.activated.emit(self)
        button = event.button()
        if button == Qt.MouseButton.RightButton:
            self._open_menu()
            return
        if button != Qt.MouseButton.LeftButton or not self._drawn:
            return
        if event.double():
            self.fit_requested.emit()
            return
        view_box = self.plot_item.getViewBox()
        position = event.scenePos()
        if view_box.sceneBoundingRect().contains(position):
            self.place_cursor(float(view_box.mapSceneToView(position).x()))


def series_id_from_mime(mime: QMimeData) -> UUID | None:
    """ドラッグのデータから系列 ID を取り出す。

    Args:
        mime: ドラッグのデータ。

    Returns:
        系列 ID。系列のドラッグでなければ None。
    """
    if not mime.hasFormat(SERIES_MIME_TYPE):
        return None
    return UUID(bytes(mime.data(SERIES_MIME_TYPE).data()).decode("ascii"))


class StackedPlotWorkspace(QWidget):
    """縦に並べたプロットパネルの集まり（縦並べ比較の本体）。

    系列の表示は ``show_series()`` と ``hide_series()`` で行い、パネルは表示に合わせて増減する。
    パネルの間の時間軸の連動は、系列の追加・削除のたびに ``_update_links()`` で決め直す。

    Attributes:
        time_scroll: 表示している時間の位置を示し、動かすスクロールバー。
    """

    message = Signal(str)
    """利用者に知らせるメッセージ（系列を追加できなかった理由など）。"""
    series_dropped = Signal(object, object)
    """系列がドロップされた（引数: 系列 ID, 追加先のパネル）。"""
    series_changed = Signal()
    """表示している系列の組が変わった（一覧のチェックを合わせるために使う）。"""
    range_selected = Signal(float, float)
    """解析する範囲の選択が動かされた（引数: 開始 [s], 終了 [s]）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """空のパネルを 1 つ持つ状態で初期化する。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(4)
        # パネルが多いときは、高さを保ったままスクロールする（窓より高くならないように）。
        panels_area = QScrollArea()
        panels_area.setWidgetResizable(True)
        panels_area.setFrameShape(QFrame.Shape.NoFrame)
        panels_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panels_area.setWidget(self._splitter)
        self.time_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.time_scroll.setToolTip("表示している時間の位置（ドラッグで移動）")
        self.time_scroll.setRange(0, 0)
        self.time_scroll.setPageStep(_SCROLL_STEPS)
        self.time_scroll.setVisible(False)
        self.time_scroll.valueChanged.connect(self._on_time_scrolled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(panels_area, stretch=1)
        layout.addWidget(self.time_scroll)
        self._updating_scroll = False
        self._panels: list[PlotPanel] = []
        self._active: PlotPanel | None = None
        # 表示中の時刻の印・解析する範囲の選択（対象の時間軸と一緒に覚えておき、
        # パネルや系列が変わるたびに、その時間軸のパネルへ描き直す）。
        self._markers: tuple[str, list[TimeMarker]] | None = None
        self._range: tuple[str, tuple[float, float]] | None = None
        self.add_panel()

    # --- パネル --------------------------------------------------------------

    @property
    def panels(self) -> list[PlotPanel]:
        """上から順のパネル（コピー）。"""
        return list(self._panels)

    @property
    def active_panel(self) -> PlotPanel | None:
        """選択中のパネル。"""
        return self._active

    def add_panel(self, after: PlotPanel | None = None) -> PlotPanel:
        """空のパネルを追加し、選択中にする（パネルの高さは揃え直す）。

        Args:
            after: このパネルのすぐ下に追加する（元の線と並べて比べるため）。None なら一番下。

        Returns:
            追加したパネル。
        """
        panel = PlotPanel()
        panel.plot_item.sigXRangeChanged.connect(self._on_x_range_changed)
        panel.activated.connect(self.set_active_panel)
        panel.cursor_moved.connect(self._on_cursor_moved)
        panel.remove_requested.connect(self.remove_panel)
        panel.series_remove_requested.connect(self.remove_series)
        panel.fit_requested.connect(self.fit_all)
        panel.series_dropped.connect(self._on_panel_dropped)
        panel.region_moved.connect(self._on_region_moved)
        panel.region_finished.connect(self.range_selected)
        index = len(self._panels) if after is None else self._panels.index(after) + 1
        self._splitter.insertWidget(index, panel)
        self._panels.insert(index, panel)
        self.set_active_panel(panel)
        self._update_links()
        self._equalize_panels()
        return panel

    def remove_panel(self, panel: PlotPanel) -> None:
        """パネルを削除する。最後の 1 つを削除した場合は、空のパネルを 1 つ残す。

        Args:
            panel: 削除するパネル。
        """
        if panel not in self._panels:
            return
        self._discard(panel)
        if not self._panels:
            self.add_panel()
            return
        if self._active is panel:
            self.set_active_panel(self._panels[-1])
        self._update_links()
        self._equalize_panels()

    def set_active_panel(self, panel: PlotPanel) -> None:
        """系列の追加先にするパネルを選ぶ。

        Args:
            panel: 選ぶパネル。
        """
        self._active = panel

    def clear(self) -> None:
        """すべてのパネルを削除し、空のパネルを 1 つだけ残す。"""
        for panel in list(self._panels):
            self._discard(panel)
        self._active = None
        self._markers = None
        self._range = None
        self.add_panel()

    def panels_showing(self, series_id: UUID) -> list[PlotPanel]:
        """系列を表示しているパネルを返す。

        Args:
            series_id: 系列 ID。

        Returns:
            上から順のパネルのリスト（表示していなければ空）。
        """
        return [panel for panel in self._panels if series_id in panel.series_ids()]

    def displayed_series_ids(self) -> set[UUID]:
        """いずれかのパネルに置いている系列の ID を返す（凡例で隠した系列も含む）。

        Returns:
            系列 ID の集合。
        """
        return {series_id for panel in self._panels for series_id in panel.series_ids()}

    def fit_all(self) -> None:
        """すべてのパネルで、データ全体が見えるように表示する（縦軸は自動調整に戻す）。

        連動している系列はまとめて 1 つの時間軸を持つので、同じ時間軸のパネルのデータ全体が
        入る範囲を、その時間軸で一番上のパネルに設定する。
        """
        groups: dict[str, list[PlotPanel]] = {}
        for panel in self._panels:
            domain = panel.time_domain()
            if domain is not None:
                groups.setdefault(domain, []).append(panel)
            panel.fit_y()
        for panels in groups.values():
            ranges = [panel.data_range() for panel in panels]
            lows = [item[0] for item in ranges if item is not None]
            highs = [item[1] for item in ranges if item is not None]
            margin = _FIT_MARGIN * (max(highs) - min(lows))
            panels[0].set_x_range((min(lows) - margin, max(highs) + margin))

    # --- 系列 ----------------------------------------------------------------

    def show_series(self, series: PlotSeries) -> bool:
        """系列を、空のパネル、なければ新しいパネルに表示する。

        Args:
            series: 表示する系列。

        Returns:
            表示できたら True。
        """
        empty = next((panel for panel in self._panels if not panel.series), None)
        return self.add_series(series, empty or self.add_panel())

    def hide_series(self, series_id: UUID) -> None:
        """系列をすべてのパネルから外し、空になったパネルを閉じる。

        Args:
            series_id: 外す系列の ID。
        """
        for panel in self.panels_showing(series_id):
            panel.remove_series(series_id)
            if not panel.series:
                self.remove_panel(panel)
        self._update_links()

    def add_series(self, series: PlotSeries, panel: PlotPanel | None = None) -> bool:
        """系列をパネルに追加する。追加できない場合は理由を ``message`` で知らせる。

        Args:
            series: 追加する系列。
            panel: 追加先。None なら選択中のパネル。

        Returns:
            追加できたら True。
        """
        reason = self._try_add_series(series, panel)
        if reason is not None:
            self.message.emit(reason)
        return reason is None

    def update_series(self, series: PlotSeries) -> int:
        """表示中の系列を、同じ ID の新しいデータで描き直す（解析の再計算の結果など）。

        Args:
            series: 新しいデータ。

        Returns:
            描き直したパネルの数（表示していなければ 0）。
        """
        return sum(1 for panel in self._panels if panel.update_series(series))

    def remove_series(self, panel: PlotPanel, series_id: UUID) -> None:
        """パネルから系列を取り除き、時間軸の連動を決め直す。

        Args:
            panel: 系列を表示しているパネル。
            series_id: 取り除く系列の ID。
        """
        panel.remove_series(series_id)
        self._update_links()

    def _try_add_series(self, series: PlotSeries, panel: PlotPanel | None) -> str | None:
        """系列を追加する（追加できない場合は理由を返し、何もしない）。

        Args:
            series: 追加する系列。
            panel: 追加先。None なら選択中のパネル。

        Returns:
            追加できなかった理由。追加できたら None。
        """
        target = panel or self._active or self.add_panel()
        if series.series_id in target.series_ids():
            return f"{series.label} は既にこのパネルに表示しています"
        if not target.accepts(series):
            return (
                f"{series.label} は、このパネルの系列と時刻を比較できないため、"
                "同じパネルには重ねられません（別の時計で記録したファイルです）。"
                "左の一覧のチェックで、別のパネルに表示できます"
            )
        target.add_series(series)
        self._update_links()
        self._sync_cursor(target)
        return None

    def _discard(self, panel: PlotPanel) -> None:
        """パネルを一覧と画面から取り除く。

        Args:
            panel: 取り除くパネル。
        """
        self._panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()

    def _equalize_panels(self) -> None:
        """パネルのプロットの高さを揃える。

        一番下のパネルは時刻の目盛りの分だけ全体を高くする（プロットの高さを揃えるため）。
        """
        count = len(self._panels)
        height = max(self._splitter.height(), count * _MIN_PANEL_HEIGHT)
        base = max(round((height - _TIME_AXIS_HEIGHT) / count), 1)
        self._splitter.setSizes([base] * (count - 1) + [base + _TIME_AXIS_HEIGHT])

    # --- 時刻の印と解析する範囲 ------------------------------------------------

    def show_time_markers(self, domain: str, markers: list[TimeMarker]) -> None:
        """時刻の印を、指定した時間軸のパネルに引く（前の印は消える）。

        Args:
            domain: 印の時刻の時間軸（timebase.time_domain_key の値）。
            markers: 時刻の印。
        """
        self._markers = (domain, markers)
        self._apply_overlays()

    def clear_time_markers(self) -> None:
        """時刻の印を消す。"""
        self._markers = None
        self._apply_overlays()

    def show_range_selector(self, domain: str, start: float, end: float) -> None:
        """解析する範囲の選択を、指定した時間軸のパネルに表示する。

        Args:
            domain: 範囲の時間軸（timebase.time_domain_key の値）。
            start: 開始 [s]。
            end: 終了 [s]。
        """
        self._range = (domain, (start, end))
        self._apply_overlays()

    def hide_range_selector(self) -> None:
        """解析する範囲の選択を消す。"""
        self._range = None
        self._apply_overlays()

    def _apply_overlays(self) -> None:
        """時刻の印と範囲の選択を、それぞれの時間軸のパネルに描き直す。

        印の番号と帯の文字は、その時間軸の一番上のパネルだけに添える（線は全パネルに引いて
        見比べられるようにする。すべてに出すと波形が隠れるため）。
        """
        labeled = False
        range_labeled = False
        for panel in self._panels:
            domain = panel.time_domain()
            in_marker_domain = self._markers is not None and self._markers[0] == domain
            markers = self._markers[1] if in_marker_domain and self._markers else []
            panel.set_time_markers(markers, labeled=not labeled)
            labeled = labeled or bool(markers)
            in_range_domain = self._range is not None and self._range[0] == domain
            region = self._range[1] if in_range_domain and self._range else None
            panel.set_region(region, labeled=not range_labeled)
            range_labeled = range_labeled or region is not None

    def _on_region_moved(self, source: PlotPanel, start: float, end: float) -> None:
        """動かしている範囲の選択を、同じ時間軸の他のパネルに反映する。

        Args:
            source: 範囲を動かしているパネル。
            start: 開始 [s]。
            end: 終了 [s]。
        """
        if self._range is None:
            return
        self._range = (self._range[0], (start, end))
        for panel in self._panels:
            if panel is not source and panel.region_range() is not None:
                panel.set_region((start, end))

    # --- 時間軸の連動とカーソル ------------------------------------------------

    def _reference_panel(self) -> PlotPanel | None:
        """基準パネル（上から見て最初にデータを置いたパネル）を返す。

        Returns:
            基準パネル。データのあるパネルがなければ None。
        """
        return next((panel for panel in self._panels if panel.series), None)

    def _update_links(self) -> None:
        """各パネルの時間軸の連動を決め直し、警告と時刻の目盛りの表示を更新する。

        同じ時間軸のパネルどうしは、その時間軸で一番上のパネルに連動させる（基準パネルと
        時刻を比較できない組どうしでも、組の中では一緒にズームできるようにするため）。
        空のパネルは基準パネルに連動させる。基準パネルと時刻を比較できないパネルには警告を出す。
        時刻の目盛りは、続けて並んだ同じ時間軸のパネルのまとまりの一番下だけに出す。
        """
        reference = self._reference_panel()
        reference_domain = reference.time_domain() if reference is not None else None
        masters: dict[str, PlotPanel] = {}
        effective: list[str | None] = []
        for panel in self._panels:
            domain = panel.time_domain()
            effective.append(domain or reference_domain)
            if domain is None:
                panel.link_to(reference)
                panel.set_unlinked_warning(visible=False)
                continue
            panel.link_to(masters.setdefault(domain, panel))
            panel.set_unlinked_warning(visible=domain != reference_domain)
        for index, panel in enumerate(self._panels):
            is_last = index == len(self._panels) - 1
            panel.set_time_axis_visible(visible=is_last or effective[index + 1] != effective[index])
        self._apply_overlays()
        self._sync_time_scroll()
        sources = {series.source.id for panel in self._panels for series in panel.series}
        for panel in self._panels:
            panel.set_source_shown(shown=len(sources) > 1)
        self.series_changed.emit()

    def _reference_span(self, reference: PlotPanel) -> tuple[float, float] | None:
        """基準パネルと同じ時間軸のパネルのデータ全体の範囲を返す。

        Args:
            reference: 基準パネル。

        Returns:
            (最小, 最大) [s]。データがなければ None。
        """
        domain = reference.time_domain()
        ranges = [panel.data_range() for panel in self._panels if panel.time_domain() == domain]
        spans = [item for item in ranges if item is not None]
        if not spans:
            return None
        return min(item[0] for item in spans), max(item[1] for item in spans)

    def _sync_time_scroll(self) -> None:
        """スクロールバーを、基準パネルの表示範囲に合わせる。

        データがないときは隠す。ズームのたびに表示したり隠したりするとプロットの高さが変わって
        しまうので、データがある間は出しておき、全体を表示しているときは動かせない状態にする。
        """
        reference = self._reference_panel()
        span = self._reference_span(reference) if reference is not None else None
        if reference is None or span is None or span[1] <= span[0]:
            self.time_scroll.setVisible(False)
            return
        low, high = reference.x_range()
        total = span[1] - span[0]
        visible = min(1.0, (high - low) / total)
        page = max(1, round(_SCROLL_STEPS * visible))
        position = min(max((low - span[0]) / total, 0.0), 1.0)
        self._updating_scroll = True
        try:
            self.time_scroll.setPageStep(page)
            self.time_scroll.setRange(0, max(0, _SCROLL_STEPS - page))
            self.time_scroll.setValue(round(_SCROLL_STEPS * position))
        finally:
            self._updating_scroll = False
        self.time_scroll.setVisible(True)

    def _on_x_range_changed(self, *args: object) -> None:
        """時間軸の表示範囲が変わったら、スクロールバーを合わせる。

        Args:
            *args: PyQtGraph のシグナルの引数（使わない）。
        """
        self._sync_time_scroll()

    def _on_time_scrolled(self, value: int) -> None:
        """スクロールバーの位置に合わせて、表示の幅を変えずに時間軸を動かす。

        Args:
            value: スクロールバーの値。
        """
        reference = self._reference_panel()
        if self._updating_scroll or reference is None:
            return
        span = self._reference_span(reference)
        if span is None:
            return
        low, high = reference.x_range()
        start = span[0] + (span[1] - span[0]) * value / _SCROLL_STEPS
        reference.set_x_range((start, start + high - low))

    def _sync_cursor(self, panel: PlotPanel) -> None:
        """データを追加したパネルのカーソルを、同じ時間軸のパネルに合わせる。

        カーソルはクリックで読もうとしたときだけ出すので、どこにもなければ出さない。

        Args:
            panel: データを追加したパネル。
        """
        domain = panel.time_domain()
        for other in self._panels:
            other_time = other.cursor_time()
            if other is not panel and other.time_domain() == domain and other_time is not None:
                panel.set_cursor(other_time)
                return
        panel.update_readout(panel.cursor_time())

    def _on_cursor_moved(self, source: PlotPanel, time_s: float) -> None:
        """動いたカーソルの時刻を、同じ時間軸のパネルに反映する。

        Args:
            source: カーソルが動いたパネル。
            time_s: カーソルの時刻 [s]。
        """
        source.update_readout(time_s)
        domain = source.time_domain()
        for panel in self._panels:
            if panel is not source and panel.series and panel.time_domain() == domain:
                panel.set_cursor(time_s)

    def _on_panel_dropped(self, panel: PlotPanel, series_id: UUID) -> None:
        """パネルへのドロップを知らせる。

        Args:
            panel: ドロップされたパネル。
            series_id: 系列 ID。
        """
        self.set_active_panel(panel)
        self.series_dropped.emit(series_id, panel)

    # --- 保存と復元 ----------------------------------------------------------

    def view_state(self, name: str = "main") -> PlotView:
        """現在の表示状態を保存用のモデルにする。

        Args:
            name: 表示設定の名前。

        Returns:
            表示状態。
        """
        reference = self._reference_panel()
        return PlotView(
            name=name,
            panels=[
                PlotPanelState(series_ids=panel.series_ids(), y_range=panel.y_range_state())
                for panel in self._panels
            ],
            x_range=reference.x_range() if reference is not None else None,
            cursor_time_s=reference.cursor_time() if reference is not None else None,
        )

    def restore(self, view: PlotView, loader: SeriesLoader) -> list[str]:
        """保存した表示状態を復元する。

        見つからない系列や、時刻を比較できずに置けない系列は表示から外し、
        その理由を返す（黙って捨てない）。

        Args:
            view: 表示状態。
            loader: 系列 ID から表示用の系列を作る関数。

        Returns:
            復元できなかった内容の説明。
        """
        messages: list[str] = []
        for panel in list(self._panels):
            self._discard(panel)
        self._active = None
        for state in view.panels:
            panel = self.add_panel()
            for series_id in state.series_ids:
                series = loader(series_id)
                if series is None:
                    messages.append(
                        f"保存されていた系列（ID {series_id}）が見つからないため、"
                        "表示から外しました（保存していない解析の結果は復元できません）"
                    )
                    continue
                reason = self._try_add_series(series, panel)
                if reason is not None:
                    messages.append(reason)
            if state.y_range is not None:
                panel.set_y_range(state.y_range)
        if not self._panels:
            self.add_panel()
        reference = self._reference_panel()
        if reference is not None and view.x_range is not None:
            reference.set_x_range(view.x_range)
        if reference is not None and view.cursor_time_s is not None:
            reference.set_cursor(view.cursor_time_s)
            self._on_cursor_moved(reference, view.cursor_time_s)
        return messages
