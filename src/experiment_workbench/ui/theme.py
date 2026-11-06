"""GUI の見た目（色・文字・スタイルシート）をまとめたモジュール。

色や文字の指定が画面ごとにばらつかないよう、ここに集める。
    - 系列の色は、白い背景で見分けやすい配色にする（色覚の違いでも区別しやすいよう、
      明るい黄色や黄緑を避け、青と橙を先頭に置く）。
    - 日本語の文字は、OS の既定に任せると読みにくいフォントが選ばれることがあるため、
      読みやすいフォントを優先順に指定する（入っていなければ次の候補を使う）。
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

SERIES_COLORS = (
    "#1f6fb2",
    "#d9480f",
    "#2b8a3e",
    "#7048e8",
    "#c2255c",
    "#0b7285",
    "#8d6e00",
    "#495057",
)
"""系列の線の色（パネルの中で先頭から順に使う）。"""

ACCENT = "#2563eb"
"""選択中のパネルなど、注目させる部分の色。"""
TEXT = "#1f2328"
"""文字の色。"""
MUTED = "#6b7280"
"""補足の文字の色。"""
BORDER = "#dfe2e6"
"""枠線の色。"""
BACKGROUND = "#f3f4f6"
"""ウィンドウの背景の色。"""
WARNING = "#b54708"
"""警告の色。"""
ERROR = "#b42318"
"""エラーの色。"""

FONT_FAMILIES = [
    "Yu Gothic UI",
    "Meiryo UI",
    "Segoe UI",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "sans-serif",
]
"""画面の文字に使うフォントの候補（優先順）。Windows と Ubuntu の両方を含める。"""

FONT_POINT_SIZE = 10
"""画面の文字の大きさ [pt]。"""

TICK_FONT_POINT_SIZE = 8
"""グラフの目盛りの文字の大きさ [pt]。"""

_STYLE_SHEET = f"""
QMainWindow, QDockWidget {{ background: {BACKGROUND}; }}
QDockWidget {{ titlebar-close-icon: none; font-weight: 600; color: {TEXT}; }}
QDockWidget::title {{ background: {BACKGROUND}; padding: 5px 8px; }}
QToolBar {{
    background: #ffffff;
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 3px 6px;
    spacing: 2px;
}}
QToolBar QToolButton {{ padding: 4px 10px; border-radius: 4px; color: {TEXT}; }}
QToolBar QToolButton:hover {{ background: #e8edf5; }}
QToolBar QToolButton:disabled {{ color: #a6acb5; }}
QTreeWidget, QListWidget, QTableWidget {{
    background: #ffffff;
    border: 1px solid {BORDER};
    font-weight: 400;
}}
QTreeWidget::item {{ padding: 2px 0; }}
QTabBar::tab {{ padding: 5px 14px; font-weight: 400; }}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QMenuBar {{ background: #ffffff; color: {TEXT}; }}
QMenuBar::item:selected {{ background: #e8edf5; }}
QPushButton#primaryButton {{
    background: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 18px;
}}
QPushButton#primaryButton:hover {{ background: #1d4ed8; }}
QStatusBar {{ background: #ffffff; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QSplitter::handle {{ background: {BACKGROUND}; }}
"""


RAW_BEHIND_RESULT = "#a8b0bb"
"""処理の結果を重ねたときの生データの線の色（処理の前後の違いが見えるよう、後ろに退かせる）。"""


def series_color(index: int) -> str:
    """系列の番号から線の色を返す（色が足りなければ先頭に戻る）。

    Args:
        index: 系列の番号（0 始まり）。

    Returns:
        色（"#rrggbb"）。
    """
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def ui_font(point_size: int = FONT_POINT_SIZE) -> QFont:
    """画面で使うフォントを返す。

    Args:
        point_size: 文字の大きさ [pt]。

    Returns:
        フォント。
    """
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(point_size)
    return font


def apply_theme(application: QApplication) -> None:
    """アプリケーション全体の見た目を設定する（画面を作る前に呼ぶ）。

    OS のダークモードでも、グラフの白い背景と画面の明るさが揃うよう、明るい配色に固定する。

    Args:
        application: アプリケーション。
    """
    pg.setConfigOptions(background="w", foreground=TEXT, antialias=True)
    application.setStyle("Fusion")
    application.styleHints().setColorScheme(Qt.ColorScheme.Light)
    application.setFont(ui_font())
    application.setStyleSheet(_STYLE_SHEET)
