"""メイン画面（SPEC §6）と GUI の起動。

画面構成:
    - 左: プロジェクトのツリー（実験 → セッション → 試技 → Source → チャンネル）
    - 中央: 縦並べ表示（StackedPlotWorkspace）
    - 下: メッセージ（取り込み時の異常、表示できなかった理由など）

取り込みの流れ（SPEC §8「インポートはワーカースレッドで実行し、UI スレッドをブロックしない」）:
    1. メインスレッドで登録先を解決する（DB を読む）。
    2. ワーカースレッドでファイルの検証・コピー・正規化を行う（DB に触れない）。
    3. メインスレッドで DB に登録する。SQLite の接続はスレッドをまたいで使えないため、DB の操作は
       メインスレッドだけで行う。

起動: ``uv run ewb``（引数にプロジェクトフォルダを渡すと開く）
"""

import sys
from pathlib import Path
from uuid import UUID

import pyqtgraph as pg
from pydantic import ValidationError
from PySide6.QtCore import QObject, QPoint, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from experiment_workbench.importer import DataImporter, ImportBlockedError, ImportTargetError
from experiment_workbench.model import (
    Channel,
    ChannelData,
    Experiment,
    ImportRequest,
    ImportTargets,
    IssueSeverity,
    PreparedImport,
    Session,
    Source,
    Trial,
)
from experiment_workbench.storage import EntityNotFoundError, ProjectStore, StorageError
from experiment_workbench.ui.import_dialog import ImportDialog
from experiment_workbench.ui.plot_workspace import StackedPlotWorkspace

APP_TITLE = "Experiment Workbench"

_CHANNEL_ID_ROLE = Qt.ItemDataRole.UserRole
"""ツリーのチャンネル項目に、チャンネル ID（文字列）を持たせるためのデータの役割。"""

# プロジェクト名（= フォルダ名）に使えない文字（Windows のファイル名の禁止文字）。
_INVALID_NAME_CHARACTERS = set('\\/:*?"<>|')

_MESSAGE_COLORS = {"error": QColor("#b42318"), "warning": QColor("#b54708")}
_MESSAGE_PREFIXES = {"error": "エラー: ", "warning": "警告: ", "info": ""}


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


class MainWindow(QMainWindow):
    """メイン画面。

    メニューから使う操作（新規作成・開く・取り込み）は、テストやスクリプトからも呼べるよう
    ``create_project()``・``open_project()``・``start_import()`` などの公開メソッドにしている。

    Attributes:
        workspace: 縦並べ表示。
        explorer: プロジェクトのツリー。
        messages: メッセージの一覧。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """画面を組み立てる（プロジェクトは開かない）。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self._store: ProjectStore | None = None
        self._importer: DataImporter | None = None
        self._pending_targets: ImportTargets | None = None
        self._import_signals: _ImportSignals | None = None
        self._thread_pool = QThreadPool(self)

        self.workspace = StackedPlotWorkspace()
        self.workspace.message.connect(self._on_workspace_message)
        self.setCentralWidget(self.workspace)

        self.explorer = QTreeWidget()
        self.explorer.setHeaderLabels(["名前", "情報"])
        self.explorer.setColumnWidth(0, 220)
        self.explorer.itemDoubleClicked.connect(self._on_explorer_double_clicked)
        self.explorer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.explorer.customContextMenuRequested.connect(self._on_explorer_context_menu)
        explorer_dock = QDockWidget("プロジェクト", self)
        explorer_dock.setObjectName("explorerDock")
        explorer_dock.setWidget(self.explorer)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, explorer_dock)

        self.messages = QListWidget()
        messages_dock = QDockWidget("メッセージ", self)
        messages_dock.setObjectName("messagesDock")
        messages_dock.setWidget(self.messages)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, messages_dock)

        self._build_menus()
        self._update_actions()
        self.statusBar().showMessage("プロジェクトを作成するか開いてください")

    # --- プロジェクト --------------------------------------------------------

    @property
    def store(self) -> ProjectStore | None:
        """開いているプロジェクト（開いていなければ None）。"""
        return self._store

    def create_project(self, root: Path, name: str) -> bool:
        """新しいプロジェクトを作成して開く。

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
        try:
            view = store.load_view()
        except (StorageError, ValueError) as exc:
            self.show_message(f"表示設定を読み込めないため、空の表示で開きます: {exc}", "warning")
            view = None
        if view is not None:
            for text in self.workspace.restore(view, self.load_channel_data):
                self.show_message(text, "warning")
        self.show_message(f"プロジェクト「{store.project.name}」を開きました: {store.root}")
        return True

    def close_project(self) -> None:
        """表示状態を保存してプロジェクトを閉じる。"""
        if self._store is None:
            return
        self.save_view()
        self._store.close()
        self._store = None
        self._importer = None
        self.workspace.clear()
        self.explorer.clear()
        self.setWindowTitle(APP_TITLE)
        self._update_actions()
        self.statusBar().showMessage("プロジェクトを閉じました")

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
            self.show_message("プロジェクトを開いてから取り込んでください", "warning")
            return False
        try:
            targets = self._importer.resolve_targets(request)
        except ImportTargetError as exc:
            self.show_message(str(exc), "error")
            return False
        self._pending_targets = targets
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
        targets = self._pending_targets
        self._pending_targets = None
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
        for issue in source.issues:
            self.show_message(f"{source.name}: {issue.description}", "warning")
        self.refresh_explorer()
        self.show_message(
            f"{source.original_filename} を取り込みました（チャンネル {len(prepared.channels)} 個、"
            f"警告 {len(source.issues)} 件）"
        )

    def _on_import_failed(self, error: object) -> None:
        """取り込みに失敗した理由を表示する。

        Args:
            error: ワーカースレッドで発生した例外。
        """
        self._pending_targets = None
        self._update_actions()
        if isinstance(error, ImportBlockedError):
            for issue in error.issues:
                level = "error" if issue.severity is IssueSeverity.ERROR else "warning"
                self.show_message(issue.description, level)
            self.show_message(
                "エラーがあるため取り込みませんでした（ファイルは書き込んでいません）", "error"
            )
        else:
            self.show_message(f"取り込みに失敗しました: {error}", "error")

    # --- ツリーと縦並べ表示 ----------------------------------------------------

    def refresh_explorer(self) -> None:
        """プロジェクトのツリーを DB の内容で作り直す。"""
        self.explorer.clear()
        if self._store is None:
            return
        database = self._store.database
        for experiment in database.list_entities(Experiment):
            experiment_item = QTreeWidgetItem([experiment.name, "実験"])
            self.explorer.addTopLevelItem(experiment_item)
            for session in database.list_entities(Session, experiment_id=experiment.id):
                session_item = QTreeWidgetItem([session.name, "セッション"])
                experiment_item.addChild(session_item)
                for trial in database.list_entities(Trial, session_id=session.id):
                    trial_item = QTreeWidgetItem([trial.name, "試技"])
                    session_item.addChild(trial_item)
                    for source in database.list_entities(Source, trial_id=trial.id):
                        channels = database.list_entities(Channel, source_id=source.id)
                        trial_item.addChild(_source_item(source, channels))
        self.explorer.expandAll()

    def explorer_channel_ids(self) -> list[UUID]:
        """ツリーに表示しているチャンネルの ID を、上から順に返す。

        Returns:
            チャンネル ID のリスト。
        """
        ids: list[UUID] = []
        for index in range(self.explorer.topLevelItemCount()):
            item = self.explorer.topLevelItem(index)
            if item is not None:
                _collect_channel_ids(item, ids)
        return ids

    def add_channel_to_workspace(self, channel_id: UUID, *, new_panel: bool = False) -> bool:
        """チャンネルを縦並べ表示に追加する。

        Args:
            channel_id: 追加するチャンネルの ID。
            new_panel: True なら新しいパネルを作って追加する。False なら選択中のパネルに追加する。

        Returns:
            追加できたら True（できない理由はメッセージに表示する）。
        """
        data = self.load_channel_data(channel_id)
        if data is None:
            return False
        panel = self.workspace.add_panel() if new_panel else None
        return self.workspace.add_channel(data, panel)

    def load_channel_data(self, channel_id: UUID) -> ChannelData | None:
        """チャンネルの時刻と値を正規化データから読み込む。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            チャンネルのデータ。見つからない・読めない場合は None（理由はメッセージに表示する）。
        """
        if self._store is None:
            return None
        database = self._store.database
        try:
            channel = database.get(Channel, channel_id)
            source = database.get(Source, channel.source_id)
        except EntityNotFoundError:
            return None
        try:
            return self._store.read_channel_data(source, channel)
        except StorageError as exc:
            self.show_message(str(exc), "error")
            return None

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

    # --- 画面の組み立て・イベント ------------------------------------------------

    def _build_menus(self) -> None:
        """メニューとアクションを作る。"""
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        self._new_action = QAction("新規プロジェクト...", self)
        self._new_action.setShortcut(QKeySequence.StandardKey.New)
        self._new_action.triggered.connect(self._on_new_project)
        self._open_action = QAction("プロジェクトを開く...", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._on_open_project)
        self._save_action = QAction("表示を保存", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.triggered.connect(self.save_view)
        self._close_action = QAction("プロジェクトを閉じる", self)
        self._close_action.triggered.connect(self.close_project)
        quit_action = QAction("終了", self)
        quit_action.triggered.connect(self.close)
        for action in (self._new_action, self._open_action, self._save_action, self._close_action):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        data_menu = self.menuBar().addMenu("データ(&D)")
        self._import_action = QAction("ファイルを取り込む...", self)
        self._import_action.setShortcut("Ctrl+I")
        self._import_action.triggered.connect(self._on_import)
        data_menu.addAction(self._import_action)

        view_menu = self.menuBar().addMenu("表示(&V)")
        add_panel_action = QAction("パネルを追加", self)
        add_panel_action.triggered.connect(self._on_add_panel)
        remove_panel_action = QAction("選択中のパネルを削除", self)
        remove_panel_action.triggered.connect(self._on_remove_active_panel)
        view_menu.addAction(add_panel_action)
        view_menu.addAction(remove_panel_action)

    def _update_actions(self) -> None:
        """プロジェクトの有無と取り込み中かどうかで、メニューの有効・無効を切り替える。"""
        has_project = self._store is not None
        busy = self.is_importing
        self._new_action.setEnabled(not busy)
        self._open_action.setEnabled(not busy)
        self._save_action.setEnabled(has_project)
        self._close_action.setEnabled(has_project and not busy)
        self._import_action.setEnabled(has_project and not busy)

    def _set_store(self, store: ProjectStore) -> None:
        """開いたプロジェクトを画面に反映する。

        Args:
            store: 開いたプロジェクト。
        """
        self._store = store
        self._importer = DataImporter(store)
        self.workspace.clear()
        self.refresh_explorer()
        self.setWindowTitle(f"{APP_TITLE} - {store.project.name}")
        self._update_actions()

    def _on_explorer_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """チャンネル項目をダブルクリックしたら、選択中のパネルに追加する。

        Args:
            item: ダブルクリックした項目。
            column: 列番号（使わない）。
        """
        data = item.data(0, _CHANNEL_ID_ROLE)
        if isinstance(data, str):
            self.add_channel_to_workspace(UUID(data))

    def _on_explorer_context_menu(self, position: QPoint) -> None:
        """チャンネル項目の右クリックメニュー（選択中のパネル・新しいパネルに追加）を出す。

        Args:
            position: ツリー上のクリック位置。
        """
        item = self.explorer.itemAt(position)
        data = item.data(0, _CHANNEL_ID_ROLE) if item is not None else None
        if not isinstance(data, str):
            return
        menu = QMenu(self)
        to_active = menu.addAction("選択中のパネルに追加")
        to_new = menu.addAction("新しいパネルに追加")
        chosen = menu.exec(self.explorer.viewport().mapToGlobal(position))
        if chosen is to_active:
            self.add_channel_to_workspace(UUID(data))
        elif chosen is to_new:
            self.add_channel_to_workspace(UUID(data), new_panel=True)

    def _on_workspace_message(self, text: str) -> None:
        """縦並べ表示からのメッセージ（チャンネルを追加できない理由など）を警告として表示する。

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
        directory = QFileDialog.getExistingDirectory(self, "プロジェクトフォルダを開く")
        if directory:
            self.open_project(Path(directory))

    def _on_import(self) -> None:
        """取り込むファイルを尋ね、取り込みダイアログで指定を受けて取り込みを開始する。"""
        if self._importer is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "取り込むファイル",
            "",
            "データファイル (*.csv *.tsv *.txt *.parquet);;すべてのファイル (*)",
        )
        if not selected:
            return
        dialog = ImportDialog(self._importer, Path(selected), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.import_path is not None and dialog.import_request is not None:
            self.start_import(dialog.import_path, dialog.import_request)

    def _on_add_panel(self) -> None:
        """空のパネルを追加する。"""
        self.workspace.add_panel()

    def _on_remove_active_panel(self) -> None:
        """選択中のパネルを削除する。"""
        panel = self.workspace.active_panel
        if panel is not None:
            self.workspace.remove_panel(panel)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt のメソッドの上書き
        """閉じるときに表示状態を保存してプロジェクトを閉じる（取り込み中は閉じない）。

        Args:
            event: 閉じるイベント。
        """
        if self.is_importing:
            self.show_message(
                "取り込み中のため閉じられません。終わってから閉じてください", "warning"
            )
            event.ignore()
            return
        self.close_project()
        event.accept()


def _collect_channel_ids(item: QTreeWidgetItem, ids: list[UUID]) -> None:
    """ツリー項目とその子孫から、チャンネル ID を上から順に集める。

    Args:
        item: 調べるツリー項目。
        ids: 見つけたチャンネル ID を追加するリスト。
    """
    data = item.data(0, _CHANNEL_ID_ROLE)
    if isinstance(data, str):
        ids.append(UUID(data))
    for index in range(item.childCount()):
        child = item.child(index)
        if child is not None:
            _collect_channel_ids(child, ids)


def _source_item(source: Source, channels: list[Channel]) -> QTreeWidgetItem:
    """Source とそのチャンネルのツリー項目を作る。

    Source の行には、形式・時刻の同期の有無・警告の件数を表示し、ツールチップに元ファイル名・
    時刻列・警告の内容を出す。チャンネルの行には、単位とサンプリング周波数を表示する。

    Args:
        source: Source。
        channels: Source のチャンネル。

    Returns:
        ツリー項目（子にチャンネル）。
    """
    clock = "同期済み" if source.time_base.shares_trial_clock else "独立した時計"
    warnings = f"・警告 {len(source.issues)} 件" if source.issues else ""
    item = QTreeWidgetItem([source.name, f"{source.format.value.upper()}・{clock}{warnings}"])
    time_text = (
        f"時刻列: {source.time_column}（{source.time_base.original_unit}）"
        if source.time_column is not None
        else f"時刻列なし（サンプル間隔 {source.time_base.sample_interval_s} s）"
    )
    tooltip = [f"元ファイル: {source.original_filename}", time_text]
    tooltip.extend(f"警告: {issue.description}" for issue in source.issues)
    item.setToolTip(0, "\n".join(tooltip))
    item.setToolTip(1, "\n".join(tooltip))
    if source.issues:
        item.setForeground(1, QBrush(_MESSAGE_COLORS["warning"]))
    for channel in channels:
        rate = channel.sample_rate.estimated_hz or channel.sample_rate.nominal_hz
        parts = [channel.unit or "", f"{rate:.6g} Hz" if rate is not None else ""]
        child = QTreeWidgetItem([channel.name, " ".join(part for part in parts if part)])
        child.setData(0, _CHANNEL_ID_ROLE, str(channel.id))
        child.setToolTip(
            0, "ダブルクリックで選択中のパネルに追加（右クリックで新しいパネルにも追加できます）"
        )
        item.addChild(child)
    return item


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する。

    Args:
        argv: コマンドライン引数（プログラム名を含む）。2 つ目にプロジェクトフォルダを渡すと開く。
            None なら ``sys.argv``。

    Returns:
        アプリケーションの終了コード。
    """
    arguments = sys.argv if argv is None else argv
    # 研究用の図として見やすいよう、背景を白にする（プロットを作る前に設定する必要がある）。
    pg.setConfigOptions(background="w", foreground="k")
    application = QApplication.instance() or QApplication(arguments)
    window = MainWindow()
    window.show()
    if len(arguments) > 1:
        window.open_project(Path(arguments[1]))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
