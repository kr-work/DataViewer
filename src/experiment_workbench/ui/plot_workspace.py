"""縦並べ比較（SPEC F-10）: 時間軸を連動させた複数のプロットパネル。

時間軸の連動の規則（判定は timebase.time_domain_key に任せる）:
    - 1 つのパネルには、時刻を比較できるチャンネルだけを置ける。
      比較できないチャンネルは理由を示して断る。
    - 基準パネル（上から見て最初にデータを置いたパネル）と時刻を比較できるパネル、
      および空のパネルは、時間軸のズーム・パン・カーソルを基準パネルと連動させる。
    - 比較できないパネルは連動させず、パネル上部に警告を表示する（勝手に揃えない）。

描画には PyQtGraph の表示用間引き（ピーク保存型）を使う。間引きは表示だけに使い、
カーソルの値は元のデータの最も近いサンプルから読む。
pyqtgraph には型情報がないため、pyqtgraph のクラスは継承せずに部品として使う。
"""

from collections.abc import Callable
from uuid import UUID

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.model import ChannelData, PlotPanelState, PlotView
from experiment_workbench.timebase import time_domain_key

ChannelLoader = Callable[[UUID], ChannelData | None]
"""チャンネル ID から表示用のデータを読み込む関数（見つからなければ None）。"""

UNLINKED_WARNING = (
    "⚠ 他のパネルと時刻を比較できないデータのため、時間軸を連動していません"
    "（同期済みとして取り込んだファイルどうしだけが同じ時間軸に並びます）"
)
"""時間軸を連動しないパネルに表示する警告。"""

_ACTIVE_STYLE = "#plotPanel { border: 2px solid #1f77b4; }"
_INACTIVE_STYLE = "#plotPanel { border: 1px solid #c8c8c8; }"


def series_label(data: ChannelData) -> str:
    """凡例やカーソル表示に使うチャンネルの表示名（"Source 名 / チャンネル名"）を返す。

    Args:
        data: チャンネルのデータ。

    Returns:
        表示名。
    """
    return f"{data.source.name} / {data.channel.name}"


class PlotPanel(QFrame):
    """縦並べの 1 パネル（時刻を比較できるチャンネルの集まり）。

    Attributes:
        plot_widget: PyQtGraph のプロット領域。
        plot_item: プロット領域の PlotItem（軸・凡例・カーソルを持つ）。
        cursor_line: 時刻を指すカーソル（縦線）。ドラッグで動かせる。
            （QWidget の cursor() メソッドと衝突しないよう、この名前にしている）
        title_label: パネルの番号を表示するラベル。
        warning_label: 時間軸を連動していないときの警告ラベル。
        readout_label: カーソル位置の値を表示するラベル。
    """

    activated = Signal(object)
    """パネルがクリックされた（引数: 自身）。"""
    cursor_moved = Signal(object, float)
    """カーソルがドラッグされた（引数: 自身, 時刻 [s]）。"""
    remove_requested = Signal(object)
    """パネルの削除が求められた（引数: 自身）。"""
    channel_remove_requested = Signal(object, object)
    """チャンネルの取り外しが求められた（引数: 自身, チャンネル ID）。"""

    def __init__(self, number: int, parent: QWidget | None = None) -> None:
        """パネルの部品を組み立てる。

        Args:
            number: 上から何番目のパネルか（表示用、1 始まり）。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self._series: list[ChannelData] = []
        self._curves: dict[UUID, object] = {}
        self._updating_cursor = False
        # 時間軸を連動させている相手（PyQtGraph の PlotItem）。連動していなければ None。
        self._linked_to: object | None = None

        self.title_label = QLabel()
        self.warning_label = QLabel(UNLINKED_WARNING)
        self.warning_label.setStyleSheet("color: #b54708;")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        self._channel_menu = QMenu(self)
        self._channel_menu.aboutToShow.connect(self._rebuild_channel_menu)
        self._channel_menu.triggered.connect(self._on_channel_menu_triggered)
        remove_channel_button = QToolButton()
        remove_channel_button.setText("チャンネルを外す")
        remove_channel_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        remove_channel_button.setMenu(self._channel_menu)
        remove_panel_button = QToolButton()
        remove_panel_button.setText("パネルを削除")
        remove_panel_button.clicked.connect(self._on_remove_clicked)
        header = QHBoxLayout()
        header.addWidget(self.title_label)
        header.addWidget(self.warning_label, stretch=1)
        header.addWidget(remove_channel_button)
        header.addWidget(remove_panel_button)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.setLabel("bottom", "時刻", units="s")
        self.plot_item.addLegend(offset=(10, 10))
        self.cursor_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#d62728", width=1))
        self.cursor_line.setVisible(False)
        self.plot_item.addItem(self.cursor_line, ignoreBounds=True)
        self.cursor_line.sigPositionChanged.connect(self._on_cursor_dragged)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

        self.readout_label = QLabel()
        self.readout_label.setTextFormat(Qt.TextFormat.PlainText)
        self.readout_label.setWordWrap(True)
        self.readout_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(header)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.readout_label)
        self.set_number(number)
        self.set_active(active=False)

    # --- 状態の参照 ----------------------------------------------------------

    @property
    def series(self) -> list[ChannelData]:
        """パネルに表示しているチャンネルのデータ（表示順のコピー）。"""
        return list(self._series)

    def channel_ids(self) -> list[UUID]:
        """パネルに表示しているチャンネルの ID を表示順に返す。

        Returns:
            チャンネル ID のリスト。
        """
        return [data.channel.id for data in self._series]

    def time_domain(self) -> str | None:
        """パネルの時間軸の識別子を返す（空のパネルは None）。

        Returns:
            timebase.time_domain_key の値、または None。
        """
        return time_domain_key(self._series[0].source) if self._series else None

    def accepts(self, data: ChannelData) -> bool:
        """チャンネルをこのパネルに置けるか（時刻を比較できるか）を返す。

        Args:
            data: 置きたいチャンネル。

        Returns:
            空のパネル、または時間軸が同じなら True。
        """
        domain = self.time_domain()
        return domain is None or domain == time_domain_key(data.source)

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

    # --- 操作 ----------------------------------------------------------------

    def set_number(self, number: int) -> None:
        """パネルの番号の表示を更新する。

        Args:
            number: 上から何番目のパネルか（1 始まり）。
        """
        self.title_label.setText(f"パネル {number}")

    def set_active(self, *, active: bool) -> None:
        """選択中のパネルとして枠を強調するかを切り替える。

        Args:
            active: 選択中なら True。
        """
        self.setStyleSheet(_ACTIVE_STYLE if active else _INACTIVE_STYLE)

    def add_series(self, data: ChannelData, color_index: int) -> None:
        """チャンネルを描画する（時間軸の判定は呼び出し側で済ませておく）。

        欠損（NaN）の区間は線をつながずに空白として表示する。時刻が単調増加でないデータでは、
        表示範囲の切り出し（clipToView）が正しく働かないため使わない。

        Args:
            data: 描画するチャンネル。
            color_index: 線の色の番号（ワークスペース全体で通し番号にする）。
        """
        pen = pg.mkPen(pg.intColor(color_index, hues=9), width=1)
        curve = self.plot_item.plot(
            data.time_s, data.values, pen=pen, name=series_label(data), connect="finite"
        )
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(bool(np.all(np.diff(data.time_s) > 0)))
        self._curves[data.channel.id] = curve
        self._series.append(data)
        self._update_axis_label()

    def remove_series(self, channel_id: UUID) -> None:
        """チャンネルの描画を取り除く。

        Args:
            channel_id: 取り除くチャンネルの ID。表示していなければ何もしない。
        """
        curve = self._curves.pop(channel_id, None)
        if curve is None:
            return
        self.plot_item.removeItem(curve)
        self._series = [data for data in self._series if data.channel.id != channel_id]
        if not self._series:
            self.cursor_line.setVisible(False)
            self.readout_label.clear()
        self._update_axis_label()

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

    def set_unlinked_warning(self, *, visible: bool) -> None:
        """時間軸を連動していない警告の表示を切り替える。

        Args:
            visible: 表示するなら True。
        """
        self.warning_label.setVisible(visible)

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

    def update_readout(self, time_s: float) -> None:
        """カーソル位置の各チャンネルの値を表示する。

        補間はせず、各チャンネルで最も近いサンプルの時刻と値を表示する。
        時刻が単調増加でないデータにも対応するため、二分探索ではなく全体から最も近いものを探す。

        Args:
            time_s: カーソルの時刻 [s]。
        """
        parts = [f"t = {time_s:.6g} s"]
        for data in self._series:
            index = int(np.argmin(np.abs(data.time_s - time_s)))
            value = float(data.values[index])
            text = "欠損" if np.isnan(value) else f"{value:.6g}"
            unit = f" {data.channel.unit}" if data.channel.unit else ""
            parts.append(f"{series_label(data)}: {text}{unit}（t = {data.time_s[index]:.6g} s）")
        self.readout_label.setText("   |   ".join(parts))

    # --- 内部処理 ------------------------------------------------------------

    def _update_axis_label(self) -> None:
        """縦軸のラベルを、表示中のチャンネルの単位がすべて同じならその単位にする。"""
        units = {data.channel.unit for data in self._series}
        unit = units.pop() if len(units) == 1 else None
        self.plot_item.setLabel("left", unit or "")

    def _rebuild_channel_menu(self) -> None:
        """「チャンネルを外す」メニューを、表示中のチャンネルで作り直す。"""
        self._channel_menu.clear()
        if not self._series:
            empty = self._channel_menu.addAction("（表示中のチャンネルはありません）")
            empty.setEnabled(False)
            return
        for data in self._series:
            action = self._channel_menu.addAction(series_label(data))
            action.setData(str(data.channel.id))

    def _on_channel_menu_triggered(self, action: QAction) -> None:
        """メニューで選んだチャンネルの取り外しを求める。

        Args:
            action: 選ばれたメニュー項目（data にチャンネル ID の文字列を持つ）。
        """
        data = action.data()
        if isinstance(data, str):
            self.channel_remove_requested.emit(self, UUID(data))

    def _on_remove_clicked(self) -> None:
        """パネルの削除を求める。"""
        self.remove_requested.emit(self)

    def _on_cursor_dragged(self, line: object) -> None:
        """カーソルがドラッグされたことを知らせる（プログラムからの移動では知らせない）。

        Args:
            line: 動いたカーソル（PyQtGraph の InfiniteLine）。
        """
        if self._updating_cursor:
            return
        self.cursor_moved.emit(self, float(self.cursor_line.value()))

    def _on_scene_clicked(self, event: object) -> None:
        """プロット領域がクリックされたら、このパネルを選択中にするよう知らせる。

        Args:
            event: PyQtGraph のマウスイベント（使わない）。
        """
        self.activated.emit(self)


class StackedPlotWorkspace(QWidget):
    """縦に並べたプロットパネルの集まり（縦並べ比較の本体）。

    チャンネルは選択中のパネル（枠が青いパネル）に追加する。パネルの間の時間軸の連動は、
    チャンネルの追加・削除のたびに ``_update_links()`` で決め直す。
    """

    message = Signal(str)
    """利用者に知らせるメッセージ（チャンネルを追加できなかった理由など）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """空のパネルを 1 つ持つ状態で初期化する。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)
        self._panels: list[PlotPanel] = []
        self._active: PlotPanel | None = None
        self._color_counter = 0
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

    def add_panel(self) -> PlotPanel:
        """一番下に空のパネルを追加し、選択中にする。

        Returns:
            追加したパネル。
        """
        panel = PlotPanel(len(self._panels) + 1)
        panel.activated.connect(self.set_active_panel)
        panel.cursor_moved.connect(self._on_cursor_moved)
        panel.remove_requested.connect(self.remove_panel)
        panel.channel_remove_requested.connect(self.remove_channel)
        self._splitter.addWidget(panel)
        self._panels.append(panel)
        self.set_active_panel(panel)
        self._update_links()
        return panel

    def remove_panel(self, panel: PlotPanel) -> None:
        """パネルを削除する。最後の 1 つを削除した場合は、空のパネルを 1 つ残す。

        Args:
            panel: 削除するパネル。
        """
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()
        for number, remaining in enumerate(self._panels, start=1):
            remaining.set_number(number)
        if not self._panels:
            self.add_panel()
            return
        if self._active is panel:
            self.set_active_panel(self._panels[-1])
        self._update_links()

    def set_active_panel(self, panel: PlotPanel) -> None:
        """チャンネルの追加先にするパネルを選ぶ。

        Args:
            panel: 選ぶパネル。
        """
        self._active = panel
        for candidate in self._panels:
            candidate.set_active(active=candidate is panel)

    def clear(self) -> None:
        """すべてのパネルを削除し、空のパネルを 1 つだけ残す。"""
        for panel in list(self._panels):
            self._panels.remove(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._active = None
        self._color_counter = 0
        self.add_panel()

    # --- チャンネル ----------------------------------------------------------

    def add_channel(self, data: ChannelData, panel: PlotPanel | None = None) -> bool:
        """チャンネルをパネルに追加する。追加できない場合は理由を ``message`` で知らせる。

        Args:
            data: 追加するチャンネルのデータ。
            panel: 追加先。None なら選択中のパネル。

        Returns:
            追加できたら True。
        """
        reason = self._try_add_channel(data, panel)
        if reason is not None:
            self.message.emit(reason)
        return reason is None

    def remove_channel(self, panel: PlotPanel, channel_id: UUID) -> None:
        """パネルからチャンネルを取り除き、時間軸の連動を決め直す。

        Args:
            panel: チャンネルを表示しているパネル。
            channel_id: 取り除くチャンネルの ID。
        """
        panel.remove_series(channel_id)
        self._update_links()

    def _try_add_channel(self, data: ChannelData, panel: PlotPanel | None) -> str | None:
        """チャンネルを追加する（追加できない場合は理由を返し、何もしない）。

        Args:
            data: 追加するチャンネルのデータ。
            panel: 追加先。None なら選択中のパネル。

        Returns:
            追加できなかった理由。追加できたら None。
        """
        target = panel or self._active or self.add_panel()
        number = self._panels.index(target) + 1
        if data.channel.id in target.channel_ids():
            return f"{series_label(data)} は既にパネル {number} に表示しています"
        if not target.accepts(data):
            return (
                f"{series_label(data)} は、パネル {number} のチャンネルと時刻を比較できないため、"
                "同じパネルには追加できません。別のパネルに追加してください"
                "（同期済みとして取り込んだファイルどうしだけが同じ時間軸に並びます）"
            )
        target.add_series(data, self._color_counter)
        self._color_counter += 1
        self._update_links()
        self._place_cursor(target, float(data.time_s[0]))
        return None

    # --- 時間軸の連動とカーソル ------------------------------------------------

    def _reference_panel(self) -> PlotPanel | None:
        """基準パネル（上から見て最初にデータを置いたパネル）を返す。

        Returns:
            基準パネル。データのあるパネルがなければ None。
        """
        return next((panel for panel in self._panels if panel.series), None)

    def _update_links(self) -> None:
        """各パネルの時間軸を基準パネルに連動させるか決め直し、警告の表示を更新する。"""
        reference = self._reference_panel()
        domain = reference.time_domain() if reference is not None else None
        for panel in self._panels:
            comparable = panel.time_domain() is None or panel.time_domain() == domain
            panel.link_to(reference if comparable else None)
            panel.set_unlinked_warning(visible=not comparable)

    def _place_cursor(self, panel: PlotPanel, default_time: float) -> None:
        """データを追加したパネルのカーソルを置く。

        同じ時間軸のパネルに既にカーソルがあればその時刻に合わせ、なければデータの先頭に置く。

        Args:
            panel: データを追加したパネル。
            default_time: 合わせる先がない場合の時刻（追加したデータの先頭）[s]。
        """
        current = panel.cursor_time()
        if current is not None:
            panel.update_readout(current)
            return
        domain = panel.time_domain()
        for other in self._panels:
            other_time = other.cursor_time()
            if other is not panel and other.time_domain() == domain and other_time is not None:
                panel.set_cursor(other_time)
                return
        panel.set_cursor(default_time)

    def _on_cursor_moved(self, source: PlotPanel, time_s: float) -> None:
        """ドラッグされたカーソルの時刻を、同じ時間軸のパネルに反映する。

        Args:
            source: カーソルがドラッグされたパネル。
            time_s: カーソルの時刻 [s]。
        """
        source.update_readout(time_s)
        domain = source.time_domain()
        for panel in self._panels:
            if panel is not source and panel.series and panel.time_domain() == domain:
                panel.set_cursor(time_s)

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
                PlotPanelState(channel_ids=panel.channel_ids(), y_range=panel.y_range_state())
                for panel in self._panels
            ],
            x_range=reference.x_range() if reference is not None else None,
            cursor_time_s=reference.cursor_time() if reference is not None else None,
        )

    def restore(self, view: PlotView, loader: ChannelLoader) -> list[str]:
        """保存した表示状態を復元する。

        見つからないチャンネルや、時刻を比較できずに置けないチャンネルは表示から外し、
        その理由を返す（黙って捨てない）。

        Args:
            view: 表示状態。
            loader: チャンネル ID から表示用のデータを読み込む関数。

        Returns:
            復元できなかった内容の説明。
        """
        messages: list[str] = []
        for panel in list(self._panels):
            self._panels.remove(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._active = None
        self._color_counter = 0
        for state in view.panels:
            panel = self.add_panel()
            for channel_id in state.channel_ids:
                data = loader(channel_id)
                if data is None:
                    messages.append(
                        f"保存されていたチャンネル（ID {channel_id}）が見つからないため、"
                        "表示から外しました"
                    )
                    continue
                reason = self._try_add_channel(data, panel)
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
