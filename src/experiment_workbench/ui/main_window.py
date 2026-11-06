"""メイン画面（SPEC §6）と GUI の起動。

使い方の流れ（覚えることを減らすため、ファイルを開くだけで始められるようにしている）:
    1. ファイルをドロップするか開くと、保存先のフォルダを自動で用意して取り込み、すぐにグラフに
       表示する。読み方は中身から判定し、判定できないときだけ確認を求める。
    2. 左の一覧のチェックでグラフへの表示を切り替える。クリックすると処理の対象になる。
    3. 右で処理を選ぶと結果がグラフに重なり、条件を動かすとその場で更新される。保存は自動。
    4. 同じ内容のファイルを開くと、前回の続きから始まる。

画面構成:
    - 上: ツールバー（よく使う操作）
    - 左: データの一覧（ファイル → チャンネル → 処理の結果）
    - 中央: 縦並べ表示（StackedPlotWorkspace）。ファイルを開く前は開始ページ
    - 右: 処理パネル（処理の一覧と、条件の調整）
    - 中央の下: 解析結果とメッセージ。必要なときだけ出す（普段はプロットの高さを優先する）
左右のドックは画面の高さいっぱいに使い、下のドックは中央の幅だけにする。
ファイルを開く前は、ドックと縦並べ表示をまとめて開始ページに置き換える。
ウィンドウの大きさと左右のドックの幅、最近開いたものは次回の起動のために覚えておく。

実験・セッション・試技の階層は、保存の形式としては残しているが画面には出さない
（SPEC は研究の管理の単位として定義しているが、データを見るだけなら意識しなくてよいため）。

取り込みの流れ（SPEC §8「インポートはワーカースレッドで実行し、UI スレッドをブロックしない」）:
    1. メインスレッドで登録先を解決する（DB を読む）。
    2. ワーカースレッドでファイルの検証・コピー・正規化を行う（DB に触れない）。
    3. メインスレッドで DB に登録する。SQLite の接続はスレッドをまたいで使えないため、DB の操作は
       メインスレッドだけで行う。
解析の実行・表示・保存・復元の流れは AnalysisController にまとめている。

起動: ``uv run ewb``（引数にファイルかプロジェクトフォルダを渡すと開く）
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import (
    QByteArray,
    QMimeData,
    QObject,
    QPoint,
    QRunnable,
    QSettings,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QIcon,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.analysis import display_channel_name
from experiment_workbench.importer import (
    DataImporter,
    ImportBlockedError,
    ImportTargetError,
    file_sha256,
)
from experiment_workbench.model import (
    METRIC_ANALYSIS_KINDS,
    AnalysisOutputInfo,
    AnalysisResult,
    AnalysisSpec,
    Channel,
    ChannelData,
    ImportRequest,
    ImportSuggestion,
    ImportTargets,
    IssueSeverity,
    PlotSeries,
    PreparedImport,
    Source,
    Trial,
)
from experiment_workbench.storage import ProjectStore, StorageError
from experiment_workbench.ui.analysis_controller import AnalysisController
from experiment_workbench.ui.analysis_panel import KIND_LABELS, AnalysisPanel
from experiment_workbench.ui.import_dialog import ImportDialog
from experiment_workbench.ui.plot_workspace import (
    GRAPH_HINT,
    SERIES_MIME_TYPE,
    PlotPanel,
    StackedPlotWorkspace,
)
from experiment_workbench.ui.result_view import AnalysisResultView
from experiment_workbench.ui.theme import ACCENT, BORDER, ERROR, MUTED, WARNING, apply_theme

APP_TITLE = "Experiment Workbench"

SETTINGS_ORGANIZATION = "ExperimentWorkbench"
"""ウィンドウの配置などを保存する設定の組織名。"""
SETTINGS_APPLICATION = "ExperimentWorkbench"
"""ウィンドウの配置などを保存する設定のアプリケーション名。"""

MAX_RECENT_PROJECTS = 8
"""覚えておく最近開いたものの数。"""

MAX_AUTO_PANELS = 6
"""ファイルを開いたときに自動で表示するチャンネルの数（残りは一覧のチェックで表示する）。"""

WORKSPACE_ROOT_KEY = "workspace_root"
"""自動で作る保存先の親フォルダの設定の名前（未設定なら OS の利用者データのフォルダ）。"""

DATA_FILE_FILTER = "データファイル (*.csv *.tsv *.txt *.parquet);;すべてのファイル (*)"
"""ファイルを開くダイアログの絞り込み。"""

_RECENT_KEY = "recent_projects"
_GEOMETRY_KEY = "window_geometry"
_LEFT_WIDTH_KEY = "left_dock_width"
_RIGHT_WIDTH_KEY = "right_dock_width"
_LEFT_DOCK_WIDTH = 270
_RIGHT_DOCK_WIDTH = 370
_MIN_LEFT_DOCK_WIDTH = 200
_MIN_RIGHT_DOCK_WIDTH = 300
_MESSAGE_DOCK_HEIGHT = 120
_MESSAGE_LIST_MIN_HEIGHT = 40
_RESULT_DOCK_HEIGHT = 340
_RESULT_DOCK_MAX_FRACTION = 0.4
"""解析結果の欄が、ウィンドウの高さに対して占める割合の上限。"""
_RECENT_LIST_HEIGHT = 220
_WORKSPACE_HASH_LENGTH = 12
"""自動で作る保存先の名前に付ける、内容のハッシュの文字数。"""

HELP_TEXT = """<h3>基本の流れ</h3>
<ol>
<li>ファイルをウィンドウへドロップ（または「ファイルを開く」）: すぐにグラフに表示します</li>
<li>左の一覧のチェック: グラフへの表示を切り替えます</li>
<li>左の一覧でチャンネルをクリック → 右で処理を選ぶ: 結果がグラフに重なります</li>
<li>右のスライダーや数値を変える: その場で結果が更新されます（保存は自動です）</li>
</ol>
<h3>図の読み方</h3>
<ul>
<li>パネルの見出しは、チャンネル名と単位です</li>
<li>処理の結果を重ねると、元の線は「生データ」としてグレーになります</li>
<li>単位や大きさが変わる結果（微分・積分・正規化など）は、生データのすぐ下のパネルに出ます</li>
<li>フィルタの除去成分（生データ − 結果）は、左の一覧のチェックで見られます</li>
<li>入出力解析と区間統計は、下の欄の一文が結論です。①〜④ の番号は、グラフの注記と対応しています</li>
</ul>
<h3>グラフの操作</h3>
<ul>
<li>左ドラッグ: 選んだ時間の範囲を拡大</li>
<li>ダブルクリック: 全体を表示（Ctrl+0 でも同じ）</li>
<li>マウスホイール: 時間軸のズーム</li>
<li>下のスクロールバー、または右ドラッグ: 時間軸の移動</li>
<li>クリック: その位置の値を読む（カーソルの移動）</li>
<li>Shift + ホイール、または縦軸の上でホイール: 縦軸のズーム</li>
<li>凡例の色の四角をクリック: 一時的に隠す</li>
<li>一覧のチャンネルをパネルへドラッグ: そのパネルに重ねて表示</li>
</ul>
<h3>複数のファイル</h3>
<ul>
<li>開いている状態でファイルを開くと、同じ画面に追加します</li>
<li>同じ時計で記録したファイルは時間軸を連動して比べられます。
別の時計のファイルは時刻を揃えず、別の時間軸で表示します</li>
</ul>"""
"""ヘルプに表示する操作の説明。"""

_CHANNEL_ID_ROLE = Qt.ItemDataRole.UserRole
"""一覧のチャンネルの行に、チャンネル ID（文字列）を持たせるためのデータの役割。"""
_ANALYSIS_ID_ROLE = Qt.ItemDataRole.UserRole + 1
"""一覧の処理の行に、解析の ID（文字列）を持たせるためのデータの役割。"""
_SERIES_ID_ROLE = Qt.ItemDataRole.UserRole + 2
"""表示できる項目（チャンネル・解析の出力）に、系列 ID（文字列）を持たせるためのデータの役割。"""

# プロジェクト名（= フォルダ名）に使えない文字（Windows のファイル名の禁止文字）。
_INVALID_NAME_CHARACTERS = set('\\/:*?"<>|')

_MESSAGE_COLORS = {"error": QColor(ERROR), "warning": QColor(WARNING)}
_MESSAGE_PREFIXES = {"error": "エラー: ", "warning": "警告: ", "info": ""}


def application_settings() -> QSettings:
    """ウィンドウの配置などを保存する設定を返す。

    形式を明示しているのは、``QSettings(組織名, アプリ名)`` が ``QSettings.setDefaultFormat`` を
    無視して、常に OS の標準の保存先（Windows ならレジストリ）を使うため。形式を省くと、テストで
    保存先を一時フォルダへ向けても効かず、利用者の設定を読み書きしてしまう。

    Returns:
        既定の形式（通常は OS の標準の保存先）の設定。
    """
    return QSettings(
        QSettings.defaultFormat(),
        QSettings.Scope.UserScope,
        SETTINGS_ORGANIZATION,
        SETTINGS_APPLICATION,
    )


def workspace_root(settings: QSettings) -> Path:
    """ファイルごとに自動で作る保存先の親フォルダを返す。

    データファイルの隣には作らない。データのフォルダを散らかさず、読み取り専用の場所や
    ネットワーク上のファイルも開けるようにするため。

    Args:
        settings: アプリケーションの設定。

    Returns:
        親フォルダのパス（まだ存在しないこともある）。
    """
    value = settings.value(WORKSPACE_ROOT_KEY)
    if isinstance(value, str) and value:
        return Path(value)
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    return Path(base) / SETTINGS_APPLICATION / "workspaces"


class _ImportSignals(QObject):
    """ワーカースレッドの取り込み処理から、メインスレッドへ結果を届けるシグナル。"""

    finished = Signal(object)
    """検証・書き込みが終わった（引数: PreparedImport）。"""
    failed = Signal(object)
    """失敗した（引数: 発生した例外）。"""


class _ImportTask(QRunnable):
    """ワーカースレッドで ``DataImporter.prepare()`` を実行するタスク（DB には触れない）。"""

    def __init__(
        self,
        importer: DataImporter,
        path: Path,
        request: ImportRequest,
        trial_id: UUID,
        signals: _ImportSignals,
    ) -> None:
        """実行に必要な値を受け取る。

        Args:
            importer: 取り込み処理。
            path: 取り込むファイル。
            request: 取り込みの指定。
            trial_id: 登録先の試技の ID。
            signals: 結果を届けるシグナル（メインスレッドで作ったもの）。
        """
        super().__init__()
        self._importer = importer
        self._path = path
        self._request = request
        self._trial_id = trial_id
        self._signals = signals

    def run(self) -> None:
        """ファイルの検証・コピー・正規化を行い、結果をシグナルで届ける。"""
        try:
            prepared = self._importer.prepare(self._path, self._request, self._trial_id)
        except Exception as exc:
            # どんな失敗も、メインスレッドで理由を表示するために届ける。
            self._signals.failed.emit(exc)
            return
        self._signals.finished.emit(prepared)


class _ProjectTree(QTreeWidget):
    """データの一覧。チャンネルと解析の出力は、パネルへドラッグできる。"""

    def mimeTypes(self) -> list[str]:  # noqa: N802 - Qt のメソッドの上書き
        """ドラッグで渡すデータの種類を返す。

        Returns:
            データの種類のリスト。
        """
        return [SERIES_MIME_TYPE]

    def mimeData(self, items: Sequence[QTreeWidgetItem]) -> QMimeData:  # noqa: N802 - Qt の上書き
        """ドラッグする項目の系列 ID をデータにする。

        Args:
            items: ドラッグする項目。

        Returns:
            ドラッグのデータ。
        """
        mime = QMimeData()
        series_id = items[0].data(0, _SERIES_ID_ROLE) if items else None
        if isinstance(series_id, str):
            mime.setData(SERIES_MIME_TYPE, QByteArray(series_id.encode("ascii")))
        return mime


class _WelcomePage(QWidget):
    """ファイルを開く前に出す開始ページ（ファイルのドロップ先と、最近開いたもの）。"""

    open_requested = Signal()
    """ファイルを開くことが求められた。"""
    recent_requested = Signal(str)
    """最近開いたものを開くことが求められた（引数: フォルダのパス）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 20pt; font-weight: 600;")
        drop_label = QLabel("ここにデータのファイルをドロップ")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet("font-size: 13pt;")
        kinds_label = QLabel("CSV・TSV・テキスト・Parquet（複数でも開けます）")
        kinds_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        kinds_label.setStyleSheet(f"color: {MUTED};")
        open_button = QPushButton("ファイルを開く...")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self._on_open_clicked)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(open_button)
        button_row.addStretch(1)
        drop_area = QFrame()
        drop_area.setObjectName("dropArea")
        drop_area.setStyleSheet(
            f"#dropArea {{ background: #ffffff; border: 2px dashed {BORDER}; border-radius: 8px; }}"
        )
        drop_layout = QVBoxLayout(drop_area)
        drop_layout.setContentsMargins(24, 36, 24, 36)
        drop_layout.setSpacing(10)
        drop_layout.addWidget(drop_label)
        drop_layout.addWidget(kinds_label)
        drop_layout.addSpacing(8)
        drop_layout.addLayout(button_row)
        self.recent_label = QLabel("最近開いたもの")
        self.recent_label.setStyleSheet("font-weight: 600;")
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(_RECENT_LIST_HEIGHT)
        self.recent_list.itemClicked.connect(self._on_recent_activated)
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addStretch(2)
        column.addWidget(title)
        column.addSpacing(8)
        column.addWidget(drop_area)
        column.addSpacing(16)
        column.addWidget(self.recent_label)
        column.addWidget(self.recent_list)
        column.addStretch(3)
        centered = QHBoxLayout(self)
        centered.addStretch(1)
        centered.addLayout(column, stretch=2)
        centered.addStretch(1)

    def set_recent_projects(self, paths: list[str]) -> None:
        """最近のプロジェクトの一覧を更新する。

        Args:
            paths: プロジェクトフォルダのパス（新しい順）。
        """
        self.recent_list.clear()
        for path in paths:
            item = QListWidgetItem(_display_name(Path(path)))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.recent_list.addItem(item)
        has_recent = bool(paths)
        self.recent_label.setVisible(has_recent)
        self.recent_list.setVisible(has_recent)

    def _on_open_clicked(self) -> None:
        """ファイルを開くことを求める。"""
        self.open_requested.emit()

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        """選ばれた最近開いたものを開くことを求める。

        Args:
            item: 選ばれた項目（データにフォルダのパスを持つ）。
        """
        self.recent_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))


class MainWindow(QMainWindow):
    """メイン画面。

    メニューから使う操作（開く・取り込み・解析）は、テストやスクリプトからも呼べるよう
    公開メソッドにしている。

    Attributes:
        workspace: 縦並べ表示。
        explorer: データの一覧。
        messages: メッセージの一覧。
        analysis_panel: 処理パネル。
        analysis: 解析の実行・表示・保存の流れ。
        result_view: 解析結果（指標とグラフ）の表示。
        welcome_page: プロジェクトを開く前の開始ページ。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """画面を組み立てる（プロジェクトは開かない）。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.setAcceptDrops(True)
        self._set_initial_size()
        self._store: ProjectStore | None = None
        self._importer: DataImporter | None = None
        self._pending_targets: ImportTargets | None = None
        self._import_signals: _ImportSignals | None = None
        # 取り込み中のファイルと、続けて開くファイルの待ち行列（1 つずつ取り込むため）。
        self._importing_path: Path | None = None
        self._open_queue: list[Path] = []
        self._queue_same_clock = True
        self._queue_review = False
        # 自動判定で始めた取り込みなら True（失敗したら読み方の確認を求める）。
        self._auto_import = False
        self._thread_pool = QThreadPool(self)
        self._settings = application_settings()
        self._results_enlarged = False
        self._work_area_shown = False

        self.workspace = StackedPlotWorkspace()
        self.workspace.message.connect(self._on_workspace_message)
        self.workspace.series_dropped.connect(self._on_series_dropped)
        self.workspace.series_changed.connect(self._sync_checks)
        self.welcome_page = _WelcomePage()
        self.welcome_page.open_requested.connect(self._on_open_files)
        self.welcome_page.recent_requested.connect(self._on_recent_requested)
        # 開始ページはドックも含めて画面全体を置き換える。ドックを 1 つずつ隠さなくて済むよう、
        # ドックと縦並べ表示を内側の QMainWindow にまとめ、開始ページと丸ごと切り替える。
        self._work_area = QMainWindow()
        self._work_area.setWindowFlags(Qt.WindowType.Widget)
        self._work_area.setCentralWidget(self.workspace)
        self._central = QStackedWidget()
        self._central.addWidget(self.welcome_page)
        self._central.addWidget(self._work_area)
        self.setCentralWidget(self._central)

        # 左右のドックを上下いっぱいに使う（下のドックは中央の幅だけにする）。
        area = self._work_area
        area.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        area.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        self.explorer = _ProjectTree()
        self.explorer.setColumnCount(2)
        self.explorer.setHeaderHidden(True)
        self.explorer.setIndentation(14)
        self.explorer.setDragEnabled(True)
        self.explorer.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        # 名前を切らさないことを優先する（単位などの列は、幅が足りなければ右端で切れてよい）。
        header = self.explorer.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.explorer.currentItemChanged.connect(self._on_explorer_current_changed)
        self.explorer.itemClicked.connect(self._on_explorer_clicked)
        self.explorer.itemChanged.connect(self._on_explorer_item_changed)
        self.explorer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.explorer.customContextMenuRequested.connect(self._on_explorer_context_menu)
        explorer_hint = QLabel("チェック: グラフに表示 / クリック: 処理の対象")
        explorer_hint.setToolTip("チャンネルをパネルへドラッグすると、重ねて表示できます")
        explorer_hint.setWordWrap(True)
        explorer_hint.setStyleSheet(f"color: {MUTED}; font-weight: 400; padding: 0 2px 2px 2px;")
        explorer_page = QWidget()
        explorer_layout = QVBoxLayout(explorer_page)
        explorer_layout.setContentsMargins(4, 0, 4, 4)
        explorer_layout.addWidget(explorer_hint)
        explorer_layout.addWidget(self.explorer, stretch=1)
        self._explorer_dock = QDockWidget("データ", area)
        self._explorer_dock.setObjectName("explorerDock")
        self._explorer_dock.setWidget(explorer_page)
        self._explorer_dock.setMinimumWidth(_MIN_LEFT_DOCK_WIDTH)
        area.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._explorer_dock)

        self.analysis_panel = AnalysisPanel()
        self._analysis_dock = QDockWidget("処理", area)
        self._analysis_dock.setObjectName("analysisDock")
        self._analysis_dock.setWidget(self.analysis_panel)
        self._analysis_dock.setMinimumWidth(_MIN_RIGHT_DOCK_WIDTH)
        area.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._analysis_dock)
        self.analysis = AnalysisController(
            self.workspace, self.analysis_panel, self._thread_pool, self
        )
        self.analysis.analyses_changed.connect(self.refresh_explorer)
        self.analysis.analysis_updated.connect(self._on_analysis_updated)
        self.analysis.message.connect(self.show_message)
        self.analysis.result_ready.connect(self._on_result_ready)
        self.analysis.result_removed.connect(self._on_result_removed)
        self.analysis_panel.delete_requested.connect(self._confirm_delete_analysis)
        self.analysis_panel.methods_shown.connect(self._hide_results)

        self.messages = QListWidget()
        self.messages.setMinimumHeight(_MESSAGE_LIST_MIN_HEIGHT)
        self._messages_dock = QDockWidget("メッセージ", area)
        self._messages_dock.setObjectName("messagesDock")
        self._messages_dock.setWidget(self.messages)
        area.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._messages_dock)
        self.result_view = AnalysisResultView()
        self._results_dock = QDockWidget("解析結果", area)
        self._results_dock.setObjectName("resultsDock")
        self._results_dock.setWidget(self.result_view)
        area.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._results_dock)
        area.tabifyDockWidget(self._results_dock, self._messages_dock)
        # 下のドックは必要なときだけ出す（普段はプロットの高さを優先する）。
        self._results_dock.setVisible(False)
        self._messages_dock.setVisible(False)

        self._graph_hint = QLabel(GRAPH_HINT)
        self._graph_hint.setStyleSheet(f"color: {MUTED};")
        self.statusBar().addPermanentWidget(self._graph_hint)
        self._build_menus()
        self._build_toolbar()
        self._restore_geometry()
        self.welcome_page.set_recent_projects(self._recent_projects())
        self._update_actions()

    # --- プロジェクト --------------------------------------------------------

    @property
    def store(self) -> ProjectStore | None:
        """開いているプロジェクト（開いていなければ None）。"""
        return self._store

    def open_files(self, paths: list[Path], *, review: bool = False) -> None:
        """ファイルを開いてグラフに表示する（何も開いていなければ保存先を自動で用意する）。

        既に開いていれば同じ画面に追加する。同じ内容のファイルは取り込み直さない
        （最初のファイルが同じなら、前回の続きから始まる）。フォルダを渡すと、プロジェクト
        フォルダとして開く。

        Args:
            paths: 開くファイル。
            review: True なら、判定できるファイルでも読み込み設定の確認を求める。
        """
        if self.is_importing:
            self.show_message("取り込みが終わってから開いてください", "warning")
            return
        files = self._existing_files(paths)
        if not files or (self._store is None and not self._open_workspace_for(files[0])):
            return
        importer = self._importer
        if importer is None:
            return
        pending = [path for path in files if not importer.has_same_file(path)]
        if not pending and not self.workspace.displayed_series_ids():
            # 異常終了などで表示設定が残っていないと、開いても何も見えなくなるので、表示し直す。
            self._show_imported(importer.store.database.list_entities(Channel))
        for path in files:
            if path not in pending and self._store is not None and len(files) > 1:
                self.show_message(f"{path.name} はすでに開いています")
        has_data = bool(importer.store.database.list_entities(Source))
        others = pending if has_data else pending[1:]
        same_clock = True
        if others:
            answer = self.ask_same_clock(others)
            if answer is None:
                if not has_data:
                    self.close_project()
                return
            same_clock = answer
        self._open_queue = list(pending)
        self._queue_same_clock = same_clock
        self._queue_review = review
        self._open_next()

    def ask_same_clock(self, paths: list[Path]) -> bool | None:
        """追加するファイルが、開いているデータと同じ時計で記録したものかを尋ねる。

        時刻の原点や時計の違いは推測しない（CLAUDE.md の原則 5）。同じ時計なら時間軸を連動させ、
        別の時計なら別の時間軸のまま表示するので、結果の解釈に関わる大事な確認として尋ねる。

        Args:
            paths: 追加するファイル。

        Returns:
            同じ時計なら True、別の時計なら False、取り消したら None。
        """
        names = "、".join(path.name for path in paths)
        box = QMessageBox(self)
        box.setWindowTitle("ファイルの追加")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"{names} は、開いているデータと同じ時計で記録しましたか？")
        box.setInformativeText(
            "同じ時計なら、時間軸を連動して比べます。\n"
            "別の時計なら、時刻を揃えずに別の時間軸で表示します。"
        )
        same = box.addButton("同じ時計", QMessageBox.ButtonRole.YesRole)
        different = box.addButton("別の時計", QMessageBox.ButtonRole.NoRole)
        box.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is same:
            return True
        return False if clicked is different else None

    def _existing_files(self, paths: list[Path]) -> list[Path]:
        """開けるファイルだけを返す（フォルダはプロジェクトとして開く）。

        Args:
            paths: 指定されたパス。

        Returns:
            存在するファイル。フォルダを開いた場合は空。
        """
        files: list[Path] = []
        for path in paths:
            if path.is_dir():
                self.open_project(path)
                return []
            if path.is_file():
                files.append(path)
            else:
                self.show_message(f"ファイルが見つかりません: {path}", "error")
        return files

    def _open_workspace_for(self, path: Path) -> bool:
        """最初に開くファイルの内容に対応する保存先を開く（なければ作る）。

        保存先の名前に内容のハッシュを使う。同じ内容のファイルを開くと同じ保存先になって前回の
        続きから始まり、ファイルが変わっていれば別の保存先になって混ざらない。

        Args:
            path: 最初に開くファイル。

        Returns:
            開けたら True（失敗の理由はメッセージに表示する）。
        """
        try:
            digest = file_sha256(path)
        except OSError as exc:
            self.show_message(f"ファイルを読み込めません: {exc}", "error")
            return False
        root = workspace_root(self._settings) / f"{path.stem}_{digest[:_WORKSPACE_HASH_LENGTH]}"
        if root.is_dir() and any(root.iterdir()):
            return self.open_project(root)
        try:
            store = ProjectStore.create(root, name=path.stem)
        except (StorageError, ValidationError, OSError) as exc:
            self.show_message(f"保存先を作成できません: {exc}", "error")
            return False
        self._set_store(store)
        self._remember_project(store.root)
        return True

    def _open_next(self) -> None:
        """待ち行列の次のファイルを開く。"""
        importer = self._importer
        if not self._open_queue or importer is None:
            return
        path = self._open_queue.pop(0)
        suggestion = importer.suggest(path, shares_trial_clock=self._queue_same_clock)
        if self._queue_review or suggestion.request is None:
            self._review_import(path, suggestion)
            return
        for text in suggestion.assumptions:
            self.show_message(f"{path.name}: {text}", "warning")
        self._auto_import = True
        if not self.start_import(path, suggestion.request):
            self._auto_import = False
            self._open_next()

    def _review_import(self, path: Path, suggestion: ImportSuggestion) -> None:
        """取り込みダイアログで確認を受けてから取り込む（取り消したら次のファイルへ進む）。

        Args:
            path: 開くファイル。
            suggestion: 判定の結果（確認が必要な理由を含む）。
        """
        importer = self._importer
        if importer is None:
            return
        dialog = ImportDialog(
            importer, path, self, suggestion=suggestion, same_clock=self._queue_same_clock
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        chosen_path, request = dialog.import_path, dialog.import_request
        started = False
        if accepted and chosen_path is not None and request is not None:
            started = self.start_import(chosen_path, request)
        if not started:
            self._open_next()

    def _show_imported(self, channels: list[Channel]) -> None:
        """取り込んだチャンネルをグラフに表示する（何もしなくても見えるように）。

        数が多いと 1 つあたりの高さが足りなくなるので、最初のいくつかだけを表示する。

        Args:
            channels: 取り込んだチャンネル。
        """
        for channel in channels[:MAX_AUTO_PANELS]:
            series = self.load_series(channel.id)
            if series is not None:
                self.workspace.show_series(series)
        self.workspace.fit_all()
        hidden = len(channels) - MAX_AUTO_PANELS
        if hidden > 0:
            self.show_message(
                f"チャンネルが多いので、最初の {MAX_AUTO_PANELS} 個を表示しました。"
                f"残りの {hidden} 個は左の一覧のチェックで表示できます"
            )
        if self.analysis_panel.input_data is None and channels:
            self.select_channel(channels[0].id)
            self._select_row(_CHANNEL_ID_ROLE, channels[0].id)

    def create_project(self, root: Path, name: str) -> bool:
        """新しいプロジェクトを、保存先を指定して作成して開く。

        Args:
            root: 作成するプロジェクトフォルダ（存在しないか空であること）。
            name: プロジェクト名。

        Returns:
            作成できたら True（失敗の理由はメッセージに表示する）。
        """
        if self._store is not None:
            self.close_project()
        try:
            store = ProjectStore.create(root, name=name)
        except (StorageError, ValidationError, OSError) as exc:
            self.show_message(f"プロジェクトを作成できません: {exc}", "error")
            return False
        self._set_store(store)
        self._remember_project(store.root)
        self.show_message(f"プロジェクト「{name}」を作成しました: {store.root}")
        return True

    def open_project(self, root: Path) -> bool:
        """既存のプロジェクトを開き、保存されていた表示状態を復元する。

        Args:
            root: プロジェクトフォルダ。

        Returns:
            開けたら True（失敗の理由はメッセージに表示する）。
        """
        if self._store is not None:
            self.close_project()
        try:
            store = ProjectStore.open(root)
        except (StorageError, ValueError, OSError) as exc:
            self.show_message(f"プロジェクトを開けません: {exc}", "error")
            return False
        self._set_store(store)
        self._remember_project(store.root)
        try:
            view = store.load_view()
        except (StorageError, ValueError) as exc:
            self.show_message(f"表示設定を読み込めないため、空の表示で開きます: {exc}", "warning")
            view = None
        if view is not None:
            for text in self.workspace.restore(view, self.load_series):
                self.show_message(text, "warning")
        self._select_first_channel()
        self.show_message(f"「{store.project.name}」を開きました")
        return True

    def close_project(self) -> None:
        """表示状態を保存してプロジェクトを閉じる（実行中の解析は中止する）。"""
        if self._store is None:
            return
        self.save_view()
        # プロジェクトを閉じてから終了しても、ドックの幅を失わないようにここでも覚える。
        self._save_layout()
        self.analysis.set_store(None)
        self._store.close()
        self._store = None
        self._importer = None
        self._open_queue = []
        self.workspace.clear()
        self.result_view.clear()
        self._results_dock.setVisible(False)
        self._messages_dock.setVisible(False)
        self.refresh_explorer()
        self.setWindowTitle(APP_TITLE)
        self.welcome_page.set_recent_projects(self._recent_projects())
        self._update_actions()

    def save_view(self) -> bool:
        """縦並べ表示の状態をプロジェクトに保存する。

        Returns:
            保存できたら True。
        """
        if self._store is None:
            return False
        try:
            self._store.save_view(self.workspace.view_state())
        except (StorageError, OSError) as exc:
            self.show_message(f"表示設定を保存できません: {exc}", "error")
            return False
        self.statusBar().showMessage("表示設定を保存しました")
        return True

    # --- 取り込み ------------------------------------------------------------

    @property
    def is_importing(self) -> bool:
        """取り込みの処理中なら True。"""
        return self._pending_targets is not None

    def start_import(self, path: Path, request: ImportRequest) -> bool:
        """ファイルの取り込みを開始する（検証・書き込みはワーカースレッドで行う）。

        Args:
            path: 取り込むファイル。
            request: 取り込みの指定。

        Returns:
            開始できたら True（登録先の指定に問題があるなどで開始できない場合は False）。
        """
        if self._importer is None or self.is_importing:
            self.show_message("取り込みが終わってから開いてください", "warning")
            return False
        try:
            targets = self._importer.resolve_targets(request)
        except ImportTargetError as exc:
            self.show_message(str(exc), "error")
            return False
        self._pending_targets = targets
        self._importing_path = path
        signals = _ImportSignals(self)
        signals.finished.connect(self._on_import_finished)
        signals.failed.connect(self._on_import_failed)
        self._import_signals = signals
        task = _ImportTask(self._importer, path, request, targets.trial.id, signals)
        self._thread_pool.start(task)
        self._update_actions()
        self.statusBar().showMessage(f"取り込み中: {path.name}")
        return True

    def _on_import_finished(self, prepared: PreparedImport) -> None:
        """ワーカースレッドの処理が終わったら、メインスレッドで DB に登録する。

        Args:
            prepared: 検証・書き込みが終わった取り込み結果。
        """
        self._release_import_signals()
        targets = self._pending_targets
        self._pending_targets = None
        self._importing_path = None
        self._auto_import = False
        self._update_actions()
        if self._importer is None or targets is None:
            return
        try:
            self._importer.commit(prepared, targets)
        except StorageError as exc:
            self.show_message(
                f"登録できませんでした（書き込んだファイルは削除しました）: {exc}", "error"
            )
            return
        source = prepared.source
        self.refresh_explorer()
        self._show_imported(prepared.channels)
        self.show_message(
            f"{source.original_filename} を開きました（チャンネル {len(prepared.channels)} 個、"
            f"警告 {len(source.issues)} 件）"
        )
        # データの異常は隠さない。メッセージの欄を開いて見えるようにする。
        for issue in source.issues:
            self.show_message(f"{source.name}: {issue.description}", "warning")
        self._open_next()

    def _on_import_failed(self, error: object) -> None:
        """取り込みに失敗した理由を表示する。

        Args:
            error: ワーカースレッドで発生した例外。
        """
        self._release_import_signals()
        self._pending_targets = None
        path = self._importing_path if self._auto_import else None
        self._importing_path = None
        self._auto_import = False
        self._update_actions()
        if isinstance(error, ImportBlockedError):
            for issue in error.issues:
                level = "error" if issue.severity is IssueSeverity.ERROR else "warning"
                self.show_message(issue.description, level)
            self.show_message(
                "エラーがあるため取り込みませんでした（ファイルは書き込んでいません）", "error"
            )
            if path is not None and self._importer is not None:
                # 読み方を変えれば取り込めることがあるので、理由を添えて確認の画面を出す。
                errors = [
                    issue.description
                    for issue in error.issues
                    if issue.severity is IssueSeverity.ERROR
                ]
                suggestion = self._importer.suggest(path, shares_trial_clock=self._queue_same_clock)
                self._review_import(
                    path, suggestion.model_copy(update={"request": None, "review_reasons": errors})
                )
                return
        else:
            self.show_message(f"取り込みに失敗しました: {error}", "error")
        self._open_next()

    def _release_import_signals(self) -> None:
        """結果が届いた取り込みのシグナルのオブジェクトを破棄する（取り込みのたびに溜まらないように）。

        ワーカースレッドは完了・失敗のシグナルを最後に 1 回だけ出すので、その後は使われない。
        """
        if self._import_signals is not None:
            self._import_signals.deleteLater()
            self._import_signals = None

    # --- 一覧と縦並べ表示 ----------------------------------------------------

    def refresh_explorer(self) -> None:
        """データの一覧と、解析の出力チャンネルの選択肢を作り直す。

        実験・セッション・試技の階層は出さず、ファイル → チャンネル → 処理の順に並べる。
        複数の試技があるプロジェクトだけ、区別できるようファイル名の前に試技名を付ける。
        """
        self.explorer.blockSignals(True)
        self.explorer.clear()
        choices: list[tuple[UUID, str, UUID]] = []
        if self._store is not None:
            database = self._store.database
            analyses = self.analysis.analyses
            trial_names = {trial.id: trial.name for trial in database.list_entities(Trial)}
            for source in database.list_entities(Source):
                prefix = f"{trial_names[source.trial_id]} / " if len(trial_names) > 1 else ""
                channels = database.list_entities(Channel, source_id=source.id)
                source_item = _source_item(source, f"{prefix}{source.name}")
                self.explorer.addTopLevelItem(source_item)
                for channel in channels:
                    label = f"{source.name} / {display_channel_name(channel)}"
                    choices.append((channel.id, label, source.id))
                    channel_item = _channel_item(channel)
                    source_item.addChild(channel_item)
                    for spec in analyses:
                        if spec.input_channel_id == channel.id:
                            outputs = self.analysis.outputs_of(spec)
                            channel_item.addChild(_analysis_item(spec, outputs, channel))
            self.explorer.expandAll()
        self.explorer.blockSignals(False)
        self._sync_checks()
        self.analysis_panel.set_channel_choices(choices)
        self._restore_row()

    def _restore_row(self) -> None:
        """作り直した一覧で、処理パネルが対象にしている行を選び直す。"""
        panel = self.analysis_panel
        if panel.is_editing:
            self._select_row(_ANALYSIS_ID_ROLE, panel.analysis_id)
        elif panel.input_data is not None:
            self._select_row(_CHANNEL_ID_ROLE, panel.input_data.channel.id)

    def _select_row(self, role: int, target: UUID) -> None:
        """指定した ID の行を選ぶ（選択の変更の処理は動かさない）。

        Args:
            role: データの役割。
            target: 選ぶ行の ID。
        """
        item = self._find_row(role, target)
        if item is not None:
            self.explorer.blockSignals(True)
            self.explorer.setCurrentItem(item)
            self.explorer.blockSignals(False)

    def _find_row(self, role: int, target: UUID) -> QTreeWidgetItem | None:
        """指定した ID の行を探す。

        Args:
            role: データの役割。
            target: 探す行の ID。

        Returns:
            見つかった行。なければ None。
        """
        for item in _all_items(self.explorer):
            if item.data(0, role) == str(target):
                return item
        return None

    def _sync_checks(self) -> None:
        """一覧のチェックを、グラフの実際の表示に合わせる。"""
        displayed = self.workspace.displayed_series_ids()
        self.explorer.blockSignals(True)
        for item in _all_items(self.explorer):
            series = item.data(0, _SERIES_ID_ROLE)
            if isinstance(series, str):
                shown = UUID(series) in displayed
                state = Qt.CheckState.Checked if shown else Qt.CheckState.Unchecked
                item.setCheckState(0, state)
        self.explorer.blockSignals(False)

    def _select_first_channel(self) -> None:
        """何も選ばれていなければ、最初のチャンネルを処理の対象にする。"""
        channel_ids = self.explorer_channel_ids()
        if self.analysis_panel.input_data is None and channel_ids:
            self.select_channel(channel_ids[0])
            self._select_row(_CHANNEL_ID_ROLE, channel_ids[0])

    def explorer_channel_ids(self) -> list[UUID]:
        """一覧に表示しているチャンネルの ID を、上から順に返す。

        Returns:
            チャンネル ID のリスト。
        """
        return self._collect_ids(_CHANNEL_ID_ROLE)

    def explorer_analysis_ids(self) -> list[UUID]:
        """一覧に表示している処理の ID を、上から順に返す。

        Returns:
            解析 ID のリスト。
        """
        return self._collect_ids(_ANALYSIS_ID_ROLE)

    def select_channel(self, channel_id: UUID) -> bool:
        """チャンネルを処理の対象にする（一覧でチャンネルを選んだときと同じ）。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            入力にできたら True。
        """
        data = self.load_channel_data(channel_id)
        if data is None:
            return False
        self.analysis_panel.set_input(data)
        return True

    def edit_analysis(self, analysis_id: UUID) -> bool:
        """保存した解析を、調整のために解析パネルへ読み込む。

        Args:
            analysis_id: 解析の ID。

        Returns:
            読み込めたら True。
        """
        spec = next((s for s in self.analysis.analyses if s.id == analysis_id), None)
        if spec is None:
            return False
        data = self.load_channel_data(spec.input_channel_id)
        if data is None:
            return False
        self.analysis_panel.load_spec(spec, data)
        if spec.kind in METRIC_ANALYSIS_KINDS:
            # 指標を返す解析は結果を保存しないので、読み込んだら計算して結果を見せる。
            self.analysis_panel.request_run()
        else:
            self._hide_results()
        return True

    def delete_analysis(self, analysis_id: UUID) -> None:
        """保存した解析を削除する（確認はしない。メニューからは確認してから呼ぶ）。

        Args:
            analysis_id: 解析の ID。
        """
        self.analysis.delete(analysis_id)

    def add_series_to_workspace(
        self, series_id: UUID, *, new_panel: bool = False, panel: PlotPanel | None = None
    ) -> bool:
        """系列（チャンネルまたは解析の出力）を縦並べ表示に追加する。

        Args:
            series_id: 追加する系列の ID。
            new_panel: True なら新しいパネルを作って追加する。
            panel: 追加先のパネル。どちらも指定しなければ選択中のパネルに追加する。

        Returns:
            追加できたら True（できない理由はメッセージに表示する）。
        """
        series = self.load_series(series_id)
        if series is None:
            return False
        target = self.workspace.add_panel() if new_panel else panel
        return self.workspace.add_series(series, target)

    def load_series(self, series_id: UUID) -> PlotSeries | None:
        """系列 ID から表示用の系列を作る（解析の出力はキャッシュまたは再計算から）。

        Args:
            series_id: 系列 ID（チャンネル ID または解析の出力 ID）。

        Returns:
            表示用の系列。見つからなければ None。
        """
        return self.analysis.load_series(series_id)

    def load_channel_data(self, channel_id: UUID) -> ChannelData | None:
        """チャンネルの時刻と値を正規化データから読み込む。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            チャンネルのデータ。見つからない・読めない場合は None。
        """
        return self.analysis.load_channel_data(channel_id)

    # --- メッセージ ----------------------------------------------------------

    def show_message(self, text: str, level: str = "info") -> None:
        """メッセージを一覧に追加し、ステータスバーにも表示する。

        Args:
            text: メッセージ。
            level: "info"・"warning"・"error" のいずれか（色と接頭辞に使う）。
        """
        item = QListWidgetItem(_MESSAGE_PREFIXES.get(level, "") + text)
        color = _MESSAGE_COLORS.get(level)
        if color is not None:
            item.setForeground(QBrush(color))
        self.messages.addItem(item)
        self.messages.scrollToBottom()
        self.statusBar().showMessage(text)
        if level in ("warning", "error"):
            self._reveal_messages()

    def _reveal_messages(self) -> None:
        """メッセージの欄を出す（警告やエラーを見落とさないように）。"""
        if self._messages_dock.isHidden():
            self._messages_dock.setVisible(True)
            # 表示した直後は大きさを変えても効かないので、次の周回で行う。
            QTimer.singleShot(0, self._shrink_messages)
        self._messages_dock.raise_()

    def _shrink_messages(self) -> None:
        """メッセージの欄を小さくする（プロットの高さを優先する）。"""
        self._work_area.resizeDocks(
            [self._messages_dock], [_MESSAGE_DOCK_HEIGHT], Qt.Orientation.Vertical
        )

    def _hide_results(self) -> None:
        """解析結果の欄を隠す（時系列の処理では使わないので、プロットを広く使う）。"""
        self._results_dock.setVisible(False)

    # --- 画面の組み立て・イベント ------------------------------------------------

    def _set_initial_size(self) -> None:
        """ウィンドウの大きさを、画面に収まるように決める。"""
        screen = QGuiApplication.primaryScreen()
        width, height = 1600, 950
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.95))
            height = min(height, int(available.height() * 0.92))
        self.resize(width, height)

    def _build_menus(self) -> None:
        """メニューとアクションを作る。"""
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        self._open_action = QAction("ファイルを開く...", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.setToolTip("開いていれば同じ画面に追加します")
        self._open_action.triggered.connect(self._on_open_files)
        self._review_action = QAction("読み込み設定を指定して開く...", self)
        self._review_action.triggered.connect(self._on_open_with_review)
        self._save_action = QAction("今の表示を保存", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.triggered.connect(self.save_view)
        self._close_action = QAction("閉じる", self)
        self._close_action.triggered.connect(self.close_project)
        self._new_action = QAction("保存先を指定して始める...", self)
        self._new_action.triggered.connect(self._on_new_project)
        self._open_project_action = QAction("保存先のフォルダを開く...", self)
        self._open_project_action.triggered.connect(self._on_open_project)
        quit_action = QAction("終了", self)
        quit_action.triggered.connect(self.close)
        for action in (
            self._open_action,
            self._review_action,
            self._close_action,
            self._save_action,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self._new_action)
        file_menu.addAction(self._open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("表示(&V)")
        self._fit_action = QAction("全体を表示", self)
        self._fit_action.setShortcut("Ctrl+0")
        self._fit_action.triggered.connect(self._on_fit_all)
        view_menu.addAction(self._fit_action)
        view_menu.addSeparator()
        for dock in (
            self._explorer_dock,
            self._analysis_dock,
            self._results_dock,
            self._messages_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("ヘルプ(&H)")
        self._help_action = QAction("操作方法", self)
        self._help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        self._help_action.triggered.connect(self._on_show_help)
        help_menu.addAction(self._help_action)

    def _build_toolbar(self) -> None:
        """よく使う操作のツールバーを作る（普段使うものだけを置く）。"""
        toolbar = QToolBar("ツールバー", self)
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons = (
            (self._open_action, QIcon.ThemeIcon.DocumentOpen),
            (self._fit_action, QIcon.ThemeIcon.ZoomFitBest),
            (self._help_action, QIcon.ThemeIcon.HelpAbout),
        )
        for action, icon in icons:
            action.setIcon(QIcon.fromTheme(icon))
            toolbar.addAction(action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _restore_geometry(self) -> None:
        """前回のウィンドウの大きさと位置を戻す（なければ既定のまま）。"""
        geometry = self._settings.value(_GEOMETRY_KEY)
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)

    def _save_layout(self) -> None:
        """ウィンドウの大きさと位置、左右のドックの幅を覚えておく。

        ドックの移動やタブの配置までは覚えない（既定の配置に戻せなくなるのを避け、単純に保つ）。
        """
        self._settings.setValue(_GEOMETRY_KEY, self.saveGeometry())
        if self._store is not None:
            self._settings.setValue(_LEFT_WIDTH_KEY, self._explorer_dock.width())
            self._settings.setValue(_RIGHT_WIDTH_KEY, self._analysis_dock.width())

    def _width_setting(self, key: str, default: int) -> int:
        """覚えておいたドックの幅を返す。

        Args:
            key: 設定の名前。
            default: 覚えていない、または読めないときの幅。

        Returns:
            ドックの幅 [px]。
        """
        value = self._settings.value(key, default)
        return int(value) if isinstance(value, int | str) and str(value).isdigit() else default

    def _apply_dock_sizes(self) -> None:
        """左右のドックを前回の幅にする。

        内側のウィンドウが隠れている間に ``resizeDocks`` を呼ぶと、すべてのドックが最小の幅に
        縮む。そのため最初に表示した後で呼ぶ。
        """
        self._work_area.resizeDocks(
            [self._explorer_dock, self._analysis_dock],
            [
                self._width_setting(_LEFT_WIDTH_KEY, _LEFT_DOCK_WIDTH),
                self._width_setting(_RIGHT_WIDTH_KEY, _RIGHT_DOCK_WIDTH),
            ],
            Qt.Orientation.Horizontal,
        )

    def _recent_projects(self) -> list[str]:
        """最近開いたもののフォルダを、存在するものだけ新しい順に返す。

        Returns:
            フォルダのパスのリスト。
        """
        # 設定の保存形式によって、1 件だけのリストは文字列で返ってくる。
        value = self._settings.value(_RECENT_KEY, [])
        entries = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        return [str(path) for path in entries if Path(str(path)).is_dir()]

    def _remember_project(self, root: Path) -> None:
        """開いたプロジェクトを、最近開いたものの先頭に加える。

        Args:
            root: プロジェクトフォルダ。
        """
        path = str(root)
        others = [item for item in self._recent_projects() if item != path]
        self._settings.setValue(_RECENT_KEY, [path, *others][:MAX_RECENT_PROJECTS])

    def _update_actions(self) -> None:
        """開いているかどうかと取り込み中かどうかで、画面と操作の有効・無効を切り替える。

        何も開いていないときは開始ページだけを見せる。
        """
        has_project = self._store is not None
        busy = self.is_importing
        for action in (
            self._open_action,
            self._review_action,
            self._new_action,
            self._open_project_action,
        ):
            action.setEnabled(not busy)
        self._save_action.setEnabled(has_project)
        self._close_action.setEnabled(has_project and not busy)
        self._fit_action.setEnabled(has_project)
        self._graph_hint.setVisible(has_project)
        self._central.setCurrentWidget(self._work_area if has_project else self.welcome_page)
        if has_project and not self._work_area_shown:
            self._work_area_shown = True
            # 内側のウィンドウが表示された後の周回で行う（理由は _apply_dock_sizes を参照）。
            QTimer.singleShot(0, self._apply_dock_sizes)

    def _set_store(self, store: ProjectStore) -> None:
        """開いたプロジェクトを画面に反映する。

        Args:
            store: 開いたプロジェクト。
        """
        self._store = store
        self._importer = DataImporter(store)
        self.workspace.clear()
        self.analysis.set_store(store)
        self.refresh_explorer()
        self.setWindowTitle(f"{APP_TITLE} - {store.project.name}")
        self._update_actions()

    def _collect_ids(self, role: int) -> list[UUID]:
        """一覧の行から、指定した役割の ID を上から順に集める。

        Args:
            role: データの役割。

        Returns:
            ID のリスト。
        """
        values = [item.data(0, role) for item in _all_items(self.explorer)]
        return [UUID(value) for value in values if isinstance(value, str)]

    def _on_series_dropped(self, series_id: UUID, panel: PlotPanel | None) -> None:
        """パネルにドロップされた系列を、そのパネルに重ねて表示する。

        Args:
            series_id: 系列 ID。
            panel: ドロップされたパネル。
        """
        self.add_series_to_workspace(series_id, panel=panel)

    def _on_result_ready(self, spec: AnalysisSpec, result: AnalysisResult) -> None:
        """指標を返す解析の結果を表示し、解析結果の欄を出す。

        最初の結果のときだけ、グラフが見える高さに広げる（その後は利用者の調整を変えない）。

        Args:
            spec: 解析の条件。
            result: 解析の結果。
        """
        self.result_view.show_result(spec.id, spec.name, result)
        self._results_dock.setVisible(True)
        self._results_dock.raise_()
        if not self._results_enlarged:
            self._results_enlarged = True
            # 小さい画面でもグラフの高さを残す。
            height = min(_RESULT_DOCK_HEIGHT, round(self.height() * _RESULT_DOCK_MAX_FRACTION))
            self._work_area.resizeDocks([self._results_dock], [height], Qt.Orientation.Vertical)

    def _on_result_removed(self, analysis_id: UUID) -> None:
        """削除された解析の結果を表示していたら、表示を消す。

        Args:
            analysis_id: 削除された解析の ID。
        """
        if self.result_view.analysis_id == analysis_id:
            self.result_view.clear()

    def _on_recent_requested(self, path: str) -> None:
        """開始ページで選ばれたものを開く。

        Args:
            path: プロジェクトフォルダのパス。
        """
        self.open_project(Path(path))

    def _on_show_help(self) -> None:
        """操作方法を表示する。"""
        QMessageBox.information(self, "操作方法", HELP_TEXT)

    def _on_fit_all(self) -> None:
        """すべてのパネルで全体を表示する。"""
        self.workspace.fit_all()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """ファイルのドラッグを受け入れる（開くため）。

        Args:
            event: ドラッグのイベント。
        """
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """ドロップされたファイルを開く。

        Args:
            event: ドラッグのイベント。
        """
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if not paths:
            return
        event.acceptProposedAction()
        self.open_files(paths)

    def _on_explorer_current_changed(
        self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None
    ) -> None:
        """一覧の選択が変わったら、その行を処理パネルに反映する（キーボードでの移動も含む）。

        Args:
            current: 新しく選ばれた項目。
            previous: 前に選ばれていた項目（使わない）。
        """
        if current is not None:
            self._activate_row(current)

    def _on_explorer_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """クリックされた行を処理パネルに反映する。

        選択が変わらないクリック（処理の一覧に戻った後で同じ行を押した場合）にも応えるため。

        Args:
            item: クリックされた項目。
            column: 列番号（使わない）。
        """
        self._activate_row(item)

    def _activate_row(self, item: QTreeWidgetItem) -> None:
        """チャンネルの行なら処理の対象にし、処理の行なら調整の画面を開く。

        既にその状態なら何もしない（選択の変更とクリックの両方から呼ばれるため）。

        Args:
            item: 選ばれた項目。
        """
        analysis = _analysis_id_of(item)
        channel = item.data(0, _CHANNEL_ID_ROLE)
        panel = self.analysis_panel
        if analysis is not None:
            if not (panel.is_editing and panel.analysis_id == analysis):
                self.edit_analysis(analysis)
        elif isinstance(channel, str):
            data = panel.input_data
            if panel.is_editing or data is None or str(data.channel.id) != channel:
                self.select_channel(UUID(channel))

    def _on_explorer_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """チェックが変わったら、グラフへの表示を切り替える。

        Args:
            item: チェックが変わった項目。
            column: 列番号（使わない）。
        """
        series = item.data(0, _SERIES_ID_ROLE)
        if not isinstance(series, str):
            return
        if item.checkState(0) == Qt.CheckState.Checked:
            self.analysis.show_series(UUID(series))
        else:
            self.workspace.hide_series(UUID(series))
        # 表示できなかったときにチェックが残らないよう、実際の表示に合わせ直す。
        self._sync_checks()

    def _on_analysis_updated(self, spec: AnalysisSpec) -> None:
        """処理の条件が変わったら、一覧の名前を更新する（一覧は作り直さない）。

        Args:
            spec: 更新された処理。
        """
        item = self._find_row(_ANALYSIS_ID_ROLE, spec.id)
        parent = item.parent() if item is not None else None
        channel_name = parent.text(0) if parent is not None else ""
        if item is not None:
            self.explorer.blockSignals(True)
            item.setText(0, spec.name.removeprefix(f"{channel_name} "))
            self.explorer.blockSignals(False)

    def _on_explorer_context_menu(self, position: QPoint) -> None:
        """処理の行の右クリックメニュー（削除）を出す。

        Args:
            position: 一覧の上のクリック位置。
        """
        item = self.explorer.itemAt(position)
        analysis = _analysis_id_of(item) if item is not None else None
        if analysis is None:
            return
        menu = QMenu(self)
        delete = menu.addAction("処理を削除...")
        chosen = menu.exec(self.explorer.viewport().mapToGlobal(position))
        if chosen is delete:
            self._confirm_delete_analysis(analysis)

    def _confirm_delete_analysis(self, analysis_id: UUID) -> None:
        """確認してから処理を削除する（生データは削除しない）。

        Args:
            analysis_id: 処理の ID。
        """
        answer = QMessageBox.question(
            self,
            "処理の削除",
            "この処理の条件と結果を削除します（元のデータは消えません）。よろしいですか。",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_analysis(analysis_id)

    def _on_workspace_message(self, text: str) -> None:
        """縦並べ表示からのメッセージ（系列を追加できない理由など）を警告として表示する。

        Args:
            text: メッセージ。
        """
        self.show_message(text, "warning")

    def _on_new_project(self) -> None:
        """プロジェクト名と保存先を尋ねて、新しいプロジェクトを作成する。"""
        name, accepted = QInputDialog.getText(self, "新規プロジェクト", "プロジェクト名:")
        name = name.strip()
        if not accepted or not name:
            return
        if _INVALID_NAME_CHARACTERS & set(name):
            self.show_message(
                f"プロジェクト名にフォルダ名として使えない文字が含まれています: {name}", "error"
            )
            return
        parent_dir = QFileDialog.getExistingDirectory(self, "プロジェクトを作成する場所を選択")
        if parent_dir:
            self.create_project(Path(parent_dir) / name, name)

    def _on_open_project(self) -> None:
        """プロジェクトフォルダを尋ねて開く。"""
        directory = QFileDialog.getExistingDirectory(self, "保存先のフォルダを開く")
        if directory:
            self.open_project(Path(directory))

    def _on_open_files(self) -> None:
        """開くファイルを尋ねて開く（何も開いていなくても使える）。"""
        selected, _ = QFileDialog.getOpenFileNames(self, "ファイルを開く", "", DATA_FILE_FILTER)
        if selected:
            self.open_files([Path(path) for path in selected])

    def _on_open_with_review(self) -> None:
        """開くファイルを尋ね、読み込み設定を確認してから開く。"""
        selected, _ = QFileDialog.getOpenFileNames(self, "ファイルを開く", "", DATA_FILE_FILTER)
        if selected:
            self.open_files([Path(path) for path in selected], review=True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """閉じるときに表示状態を保存してプロジェクトを閉じる（取り込み中は閉じない）。

        実行中の解析は中止し、止まるのを待ってから閉じる。

        Args:
            event: 閉じるイベント。
        """
        if self.is_importing:
            self.show_message(
                "取り込み中のため閉じられません。終わってから閉じてください", "warning"
            )
            event.ignore()
            return
        self.analysis.shutdown()
        self._save_layout()
        self.close_project()
        event.accept()


def _format_rate(rate_hz: float) -> str:
    """サンプリング周波数を読みやすい単位で表す。

    Args:
        rate_hz: サンプリング周波数 [Hz]。

    Returns:
        文字列（例: "1 kHz"、"100 Hz"）。
    """
    return f"{rate_hz / 1000:.4g} kHz" if rate_hz >= 1000 else f"{rate_hz:.4g} Hz"


def _display_name(root: Path) -> str:
    """保存先のフォルダの、一覧に出す名前を返す（自動で付けたハッシュの部分は見せない）。

    Args:
        root: 保存先のフォルダ。

    Returns:
        表示名。
    """
    name, separator, suffix = root.name.rpartition("_")
    automatic = separator and len(suffix) == _WORKSPACE_HASH_LENGTH
    return name if automatic else root.name


def _source_item(source: Source, label: str) -> QTreeWidgetItem:
    """ファイルの行を作る。

    行には警告の件数だけを短く表示し、詳しい内容（形式・元ファイル名・時刻列・時計・警告の内容）は
    ツールチップに出す。

    Args:
        source: Source。
        label: 行に出す名前。

    Returns:
        一覧の項目。
    """
    clock = "同じ時計のファイルと連動" if source.time_base.shares_trial_clock else "独立した時計"
    badge = f"⚠ {len(source.issues)}" if source.issues else ""
    item = QTreeWidgetItem([label, badge])
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
    font = item.font(0)
    font.setBold(True)
    item.setFont(0, font)
    time_text = (
        f"時刻列: {source.time_column}（{source.time_base.original_unit}）"
        if source.time_column is not None
        else f"時刻列なし（サンプル間隔 {source.time_base.sample_interval_s} s）"
    )
    tooltip = [
        f"元ファイル: {source.original_filename}（{source.format.value.upper()}）",
        time_text,
        f"時計: {clock}",
    ]
    tooltip.extend(f"警告: {issue.description}" for issue in source.issues)
    item.setToolTip(0, "\n".join(tooltip))
    item.setToolTip(1, "\n".join(tooltip))
    if source.issues:
        item.setForeground(1, QBrush(_MESSAGE_COLORS["warning"]))
    return item


def _channel_item(channel: Channel) -> QTreeWidgetItem:
    """チャンネルの行を作る（単位とサンプリング周波数を添える）。

    Args:
        channel: チャンネル。

    Returns:
        一覧の項目。
    """
    rate = channel.sample_rate.estimated_hz or channel.sample_rate.nominal_hz
    parts = [channel.unit or "", _format_rate(rate) if rate is not None else ""]
    name = display_channel_name(channel)
    item = QTreeWidgetItem([name, "  ".join(part for part in parts if part)])
    item.setForeground(1, QBrush(QColor(MUTED)))
    item.setData(0, _CHANNEL_ID_ROLE, str(channel.id))
    item.setData(0, _SERIES_ID_ROLE, str(channel.id))
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setToolTip(
        0, "チェックでグラフに表示。クリックで処理の対象に。パネルへドラッグすると重ねて表示"
    )
    return item


def _analysis_item(
    spec: AnalysisSpec, outputs: list[AnalysisOutputInfo], channel: Channel
) -> QTreeWidgetItem:
    """処理の行を作る。

    処理の行のチェックは主な結果（先頭の出力）の表示を切り替える。それ以外の出力は、
    行の下に並べる（フィルタの「除去成分」、カルマンフィルタの微分など）。

    Args:
        spec: 処理。
        outputs: 処理の出力。
        channel: 入力チャンネル。

    Returns:
        一覧の項目。
    """
    item = QTreeWidgetItem([spec.name.removeprefix(f"{display_channel_name(channel)} "), ""])
    item.setForeground(0, QBrush(QColor(ACCENT)))
    item.setData(0, _ANALYSIS_ID_ROLE, str(spec.id))
    item.setToolTip(0, f"{KIND_LABELS[spec.kind]}。クリックで条件を調整（右クリックで削除）")
    if not outputs:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
        return item
    item.setData(0, _SERIES_ID_ROLE, str(outputs[0].series_id))
    item.setText(1, outputs[0].unit or "")
    item.setForeground(1, QBrush(QColor(MUTED)))
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    for output in outputs[1:]:
        label = output.label.removeprefix(f"{spec.name}: ")
        child = QTreeWidgetItem([label, output.unit or ""])
        child.setForeground(1, QBrush(QColor(MUTED)))
        child.setData(0, _SERIES_ID_ROLE, str(output.series_id))
        child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        child.setToolTip(0, "チェックでグラフに表示")
        item.addChild(child)
    return item


def _analysis_id_of(item: QTreeWidgetItem) -> UUID | None:
    """行が属する処理の ID を返す（処理の行と、その出力の行）。

    Args:
        item: 一覧の項目。

    Returns:
        処理の ID。処理の行でなければ None。
    """
    parent = item.parent()
    for candidate in (item, parent):
        data = candidate.data(0, _ANALYSIS_ID_ROLE) if candidate is not None else None
        if isinstance(data, str):
            return UUID(data)
    return None


def _all_items(tree: QTreeWidget) -> list[QTreeWidgetItem]:
    """一覧のすべての行を、上から順に返す。

    Args:
        tree: 一覧。

    Returns:
        行のリスト。
    """
    items: list[QTreeWidgetItem] = []
    stack = [tree.topLevelItem(index) for index in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop(0)
        if item is None:
            continue
        items.append(item)
        stack[0:0] = [item.child(index) for index in range(item.childCount())]
    return items


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する。

    Args:
        argv: コマンドライン引数（プログラム名を含む）。2 つ目にファイルかプロジェクトフォルダを
            渡すと開く。None なら ``sys.argv``。

    Returns:
        アプリケーションの終了コード。
    """
    arguments = sys.argv if argv is None else argv
    application = QApplication.instance() or QApplication(arguments)
    if isinstance(application, QApplication):
        apply_theme(application)  # プロットの背景の設定があるので、画面を作る前に行う
    window = MainWindow()
    window.show()
    if len(arguments) > 1:
        window.open_files([Path(arguments[1])])
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
