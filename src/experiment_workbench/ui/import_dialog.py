"""データファイルの取り込みダイアログ（SPEC F-02）。

ファイルのプレビューと、読み込み前に分かる異常を表示しながら、時刻列・チャンネル・登録先・
時刻の同期の有無を選ぶ。選んだ内容は ImportRequest（pydantic モデル）にまとめ、検証エラーは
ダイアログ内に表示して閉じない。ファイルの検証・コピー・登録は呼び出し側（メイン画面）が行う。
"""

from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.importer import DataImporter
from experiment_workbench.model import (
    DATASET_TIME_COLUMN,
    ChannelImportSpec,
    DataIssue,
    Experiment,
    FilePreview,
    ImportRequest,
    IssueSeverity,
    ReadOptions,
    Session,
    SourceFormat,
    TimeUnit,
    Trial,
)

NO_TIME_COLUMN_LABEL = "（時刻列なし: サンプル番号とサンプル間隔から作る）"
"""時刻列の選択肢のうち、「時刻列を使わない」を表す表示。"""

_DELIMITERS = [("カンマ（,）", ","), ("タブ", "\t"), ("セミコロン（;）", ";"), ("空白", " ")]
_ENCODINGS = ["utf-8", "cp932", "shift_jis", "euc-jp", "utf-16", "latin-1"]
_TIME_UNIT_LABELS = {
    TimeUnit.SECOND: "秒（s）",
    TimeUnit.MILLISECOND: "ミリ秒（ms）",
    TimeUnit.MICROSECOND: "マイクロ秒（us）",
    TimeUnit.NANOSECOND: "ナノ秒（ns）",
}
_FORMAT_LABELS = {SourceFormat.CSV: "CSV", SourceFormat.TSV: "TSV", SourceFormat.PARQUET: "Parquet"}

# 列の表の列番号
_COLUMN_CHECK, _COLUMN_NAME, _COLUMN_KIND, _COLUMN_UNIT = range(4)

_ERROR_COLOR = QColor("#b42318")
_WARNING_COLOR = QColor("#b54708")


class ImportDialog(QDialog):
    """データファイルの取り込みダイアログ。

    ``exec()`` が受理で終わったら、``import_path`` と ``import_request`` に選んだ内容が入る。
    部品はテストから操作できるよう公開属性にしている。

    Attributes:
        import_path: 取り込むファイル（受理時のみ）。
        import_request: 取り込みの指定（受理時のみ）。
    """

    def __init__(
        self, importer: DataImporter, path: Path | None = None, parent: QWidget | None = None
    ) -> None:
        """部品を組み立て、ファイルが指定されていればプレビューを表示する。

        Args:
            importer: 取り込み先のプロジェクトの取り込み処理（プレビューと登録先の候補に使う）。
            path: 取り込むファイル。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.setWindowTitle("データファイルの取り込み")
        self.resize(980, 760)
        self._importer = importer
        self._path: Path | None = None
        self._preview: FilePreview | None = None
        self.import_path: Path | None = None
        self.import_request: ImportRequest | None = None

        # --- ファイルと読み込み条件 ---
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        browse_button = QPushButton("参照...")
        browse_button.clicked.connect(self._on_browse)
        self.format_combo = QComboBox()
        for file_format, label in _FORMAT_LABELS.items():
            self.format_combo.addItem(label, file_format)
        self.delimiter_combo = QComboBox()
        for label, delimiter in _DELIMITERS:
            self.delimiter_combo.addItem(label, delimiter)
        self.encoding_combo = QComboBox()
        self.encoding_combo.setEditable(True)
        self.encoding_combo.addItems(_ENCODINGS)
        self.skip_rows_spin = QSpinBox()
        self.skip_rows_spin.setRange(0, 10_000)
        refresh_button = QPushButton("プレビューを更新")
        refresh_button.clicked.connect(self.refresh_preview)
        file_row = QHBoxLayout()
        file_row.addWidget(self.path_edit, stretch=1)
        file_row.addWidget(browse_button)
        options = QFormLayout()
        options.addRow("ファイル", file_row)
        options.addRow("形式", self.format_combo)
        options.addRow("区切り文字", self.delimiter_combo)
        options.addRow("文字コード", self.encoding_combo)
        options.addRow("ヘッダーの前に読み飛ばす行数", self.skip_rows_spin)
        options.addRow("", refresh_button)
        file_group = QGroupBox("ファイル")
        file_group.setLayout(options)

        # --- プレビュー ---
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setMinimumHeight(160)  # 先頭の数行が見える高さを確保する
        preview_group = QGroupBox("プレビュー（先頭の行）")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_table)

        # --- 時刻 ---
        self.time_column_combo = QComboBox()
        self.time_column_combo.currentIndexChanged.connect(self._on_time_column_changed)
        self.time_unit_combo = QComboBox()
        for unit, label in _TIME_UNIT_LABELS.items():
            self.time_unit_combo.addItem(label, unit)
        self.absolute_check = QCheckBox("UNIX 時刻（1970-01-01 UTC からの絶対時刻）")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setDecimals(9)
        self.interval_spin.setRange(1e-9, 1e6)
        self.interval_spin.setValue(0.001)
        self.interval_spin.setSuffix(" s")
        time_form = QFormLayout()
        time_form.addRow("時刻列", self.time_column_combo)
        time_form.addRow("時刻列の単位", self.time_unit_combo)
        time_form.addRow("", self.absolute_check)
        time_form.addRow("サンプル間隔（時刻列なしのとき）", self.interval_spin)
        time_group = QGroupBox("時刻")
        time_group.setLayout(time_form)

        # --- チャンネル ---
        self.channel_table = QTableWidget(0, 4)
        self.channel_table.setHorizontalHeaderLabels(["取り込む", "列名", "種類", "単位（任意）"])
        self.channel_table.itemChanged.connect(self._update_ok_button)
        channel_group = QGroupBox("チャンネル（数値の列のみ取り込めます）")
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.addWidget(self.channel_table)

        # --- 登録先 ---
        self.experiment_combo = QComboBox()
        self.experiment_combo.setEditable(True)
        self.session_combo = QComboBox()
        self.session_combo.setEditable(True)
        self.trial_combo = QComboBox()
        self.trial_combo.setEditable(True)
        self.source_name_edit = QLineEdit()
        self.shares_clock_check = QCheckBox(
            "同じ試技の他のファイルと同じ時計・同じ原点で記録した（時刻が同期済み）"
        )
        self.shares_clock_check.setToolTip(
            "チェックしたファイルどうしは、同じ時間軸で縦に並べて比較できます。"
            "チェックしないファイルの時刻は、他のファイルと比較しません（勝手に揃えないため）。"
        )
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.session_combo.currentTextChanged.connect(self._on_session_changed)
        target_form = QFormLayout()
        target_form.addRow("実験", self.experiment_combo)
        target_form.addRow("セッション", self.session_combo)
        target_form.addRow("試技", self.trial_combo)
        target_form.addRow("Source 名", self.source_name_edit)
        target_form.addRow("", self.shares_clock_check)
        target_group = QGroupBox("登録先")
        target_group.setLayout(target_form)

        # --- 異常とボタン ---
        self.issues_list = QListWidget()
        self.issues_list.setMaximumHeight(130)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("取り込む")
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setText("キャンセル")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        middle = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(time_group)
        left.addWidget(target_group)
        middle.addLayout(left, stretch=1)
        middle.addWidget(channel_group, stretch=1)
        layout = QVBoxLayout(self)
        layout.addWidget(file_group)
        layout.addWidget(preview_group, stretch=1)
        layout.addLayout(middle, stretch=1)
        layout.addWidget(self.issues_list)
        layout.addWidget(self.button_box)

        self._fill_experiment_names()
        if path is not None:
            self.set_path(path)
        self._update_ok_button()

    # --- 公開操作 ------------------------------------------------------------

    def set_path(self, path: Path) -> None:
        """取り込むファイルを設定し、形式などの既定値を決めてプレビューを表示する。

        Args:
            path: 取り込むファイル。
        """
        self._path = path
        self.path_edit.setText(str(path))
        suffix = path.suffix.lower()
        file_format = {".parquet": SourceFormat.PARQUET, ".tsv": SourceFormat.TSV}.get(
            suffix, SourceFormat.CSV
        )
        self.format_combo.setCurrentIndex(self.format_combo.findData(file_format))
        delimiter = "\t" if file_format is SourceFormat.TSV else ","
        self.delimiter_combo.setCurrentIndex(self.delimiter_combo.findData(delimiter))
        self.source_name_edit.setText(path.stem)
        if not self.trial_combo.currentText().strip():
            self.trial_combo.setEditText(path.parent.name or "試技1")
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """現在の読み込み条件でプレビューを読み直し、表示を更新する。"""
        self.issues_list.clear()
        self._preview = None
        if self._path is None:
            self._update_ok_button()
            return
        try:
            read_options = self._read_options()
        except ValidationError as exc:
            self._show_validation_error(exc)
            self._update_ok_button()
            return
        self._preview = self._importer.preview(self._path, self._file_format(), read_options)
        self._fill_preview_table(self._preview)
        # 時刻列の選択肢を作ると、時刻列の変更処理からチャンネルの表も作り直される。
        self._fill_time_columns(self._preview)
        for issue in self._preview.issues:
            self._add_issue(issue)
        self._update_ok_button()

    def build_request(self) -> ImportRequest:
        """ダイアログの内容から取り込みの指定を作る。

        Returns:
            取り込みの指定。

        Raises:
            pydantic.ValidationError: 指定が不正な場合（時刻列もサンプル間隔もない、名前が空など）。
        """
        time_column = self.time_column_combo.currentData()
        channels = [
            ChannelImportSpec(column=name, unit=unit or None)
            for name, unit in self._checked_channels()
        ]
        return ImportRequest(
            file_format=self._file_format(),
            read_options=self._read_options(),
            time_column=time_column if isinstance(time_column, str) else None,
            time_unit=self._time_unit(),
            time_is_absolute=self.absolute_check.isChecked(),
            sample_interval_s=None if isinstance(time_column, str) else self.interval_spin.value(),
            channels=channels,
            experiment_name=self.experiment_combo.currentText(),
            session_name=self.session_combo.currentText(),
            trial_name=self.trial_combo.currentText(),
            source_name=self.source_name_edit.text(),
            shares_trial_clock=self.shares_clock_check.isChecked(),
        )

    def accept(self) -> None:
        """取り込みの指定を検証し、問題がなければダイアログを閉じる（問題があれば閉じない）。"""
        if self._path is None:
            return
        try:
            request = self.build_request()
        except ValidationError as exc:
            self._show_validation_error(exc)
            return
        self.import_path = self._path
        self.import_request = request
        super().accept()

    # --- 表示の更新 ----------------------------------------------------------

    def _fill_preview_table(self, preview: FilePreview) -> None:
        """プレビューの表を更新する。

        Args:
            preview: プレビュー。
        """
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setRowCount(len(preview.rows))
        self.preview_table.setHorizontalHeaderLabels(
            [f"{column.name}\n{column.dtype}" for column in preview.columns]
        )
        for row_index, row in enumerate(preview.rows):
            for column_index, value in enumerate(row):
                self.preview_table.setItem(row_index, column_index, QTableWidgetItem(value))

    def _fill_time_columns(self, preview: FilePreview) -> None:
        """時刻列の選択肢を数値の列で作り直し、それらしい列を既定に選ぶ。

        名前に "time"・"時刻"・"timestamp" を含む数値の列があればそれを、なければ最初の数値の列を
        既定にする（利用者が確認して変えられる既定値であり、データは変更しない）。

        Args:
            preview: プレビュー。
        """
        numeric = [column.name for column in preview.columns if column.is_numeric]
        self.time_column_combo.blockSignals(True)
        self.time_column_combo.clear()
        self.time_column_combo.addItem(NO_TIME_COLUMN_LABEL, None)
        for name in numeric:
            self.time_column_combo.addItem(name, name)
        keywords = ("time", "時刻", "timestamp")
        guess = next(
            (name for name in numeric if any(word in name.lower() for word in keywords)),
            numeric[0] if numeric else None,
        )
        self.time_column_combo.setCurrentIndex(
            self.time_column_combo.findData(guess) if guess is not None else 0
        )
        self.time_column_combo.blockSignals(False)
        self._on_time_column_changed()

    def _fill_channel_table(self) -> None:
        """チャンネルの表を、時刻列以外の列で作り直す（チェックと単位は列名ごとに引き継ぐ）。"""
        previous = {
            name: (checked, unit) for name, checked, unit in self._channel_rows(include_all=True)
        }
        self.channel_table.blockSignals(True)
        self.channel_table.setRowCount(0)
        if self._preview is not None:
            time_column = self.time_column_combo.currentData()
            for column in self._preview.columns:
                if column.name == time_column or not column.name.strip():
                    continue
                row = self.channel_table.rowCount()
                self.channel_table.insertRow(row)
                importable = column.is_numeric and column.name != DATASET_TIME_COLUMN
                checked, unit = previous.get(column.name, (importable, ""))
                check_item = QTableWidgetItem()
                flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                if not importable:
                    flags = Qt.ItemFlag.NoItemFlags
                    checked = False
                    reserved = (
                        f"列名 {DATASET_TIME_COLUMN} は正規化データの時刻列として予約されています"
                    )
                    check_item.setToolTip(
                        "数値でない列は取り込めません" if not column.is_numeric else reserved
                    )
                check_item.setFlags(flags)
                check_item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                name_item = QTableWidgetItem(column.name)
                name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                kind_item = QTableWidgetItem(column.dtype)
                kind_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.channel_table.setItem(row, _COLUMN_CHECK, check_item)
                self.channel_table.setItem(row, _COLUMN_NAME, name_item)
                self.channel_table.setItem(row, _COLUMN_KIND, kind_item)
                self.channel_table.setItem(row, _COLUMN_UNIT, QTableWidgetItem(unit))
        self.channel_table.blockSignals(False)
        self._update_ok_button()

    def _fill_experiment_names(self) -> None:
        """登録先の実験の候補を、プロジェクトの既存の実験名で作る。"""
        names = [
            experiment.name
            for experiment in self._importer.store.database.list_entities(Experiment)
        ]
        self.experiment_combo.addItems(names)
        if not names:
            self.experiment_combo.setEditText("実験1")
        self._on_experiment_changed(self.experiment_combo.currentText())

    def _on_experiment_changed(self, name: str) -> None:
        """実験名が変わったら、その実験の既存のセッションを候補にする。

        Args:
            name: 実験名。
        """
        database = self._importer.store.database
        experiments = database.list_entities(Experiment, name=name)
        sessions = (
            database.list_entities(Session, experiment_id=experiments[0].id) if experiments else []
        )
        current = self.session_combo.currentText()
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems([session.name for session in sessions])
        self.session_combo.setEditText(current or (sessions[0].name if sessions else "セッション1"))
        self.session_combo.blockSignals(False)
        self._on_session_changed(self.session_combo.currentText())

    def _on_session_changed(self, name: str) -> None:
        """セッション名が変わったら、そのセッションの既存の試技を候補にする。

        Args:
            name: セッション名。
        """
        database = self._importer.store.database
        experiments = database.list_entities(Experiment, name=self.experiment_combo.currentText())
        sessions = (
            database.list_entities(Session, experiment_id=experiments[0].id, name=name)
            if experiments
            else []
        )
        trials = database.list_entities(Trial, session_id=sessions[0].id) if sessions else []
        current = self.trial_combo.currentText()
        self.trial_combo.blockSignals(True)
        self.trial_combo.clear()
        self.trial_combo.addItems([trial.name for trial in trials])
        self.trial_combo.setEditText(current)
        self.trial_combo.blockSignals(False)

    def _on_time_column_changed(self) -> None:
        """時刻列の選択に合わせて、単位の既定値・サンプル間隔の入力可否・チャンネルの表を更新する。"""
        time_column = self.time_column_combo.currentData()
        has_time = isinstance(time_column, str)
        self.time_unit_combo.setEnabled(has_time)
        self.absolute_check.setEnabled(has_time)
        self.interval_spin.setEnabled(not has_time)
        if isinstance(time_column, str):
            unit = _guess_time_unit(time_column)
            self.time_unit_combo.setCurrentIndex(self.time_unit_combo.findData(unit))
        self._fill_channel_table()

    def _on_browse(self) -> None:
        """ファイル選択ダイアログで取り込むファイルを選ぶ。"""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "取り込むファイル",
            "",
            "データファイル (*.csv *.tsv *.txt *.parquet);;すべてのファイル (*)",
        )
        if selected:
            self.set_path(Path(selected))

    def _update_ok_button(self) -> None:
        """エラーがなく、チャンネルを 1 つ以上選んでいるときだけ「取り込む」を押せるようにする。"""
        has_error = self._preview is None or any(
            issue.severity is IssueSeverity.ERROR for issue in self._preview.issues
        )
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setEnabled(not has_error and bool(self._checked_channels()))

    def _add_issue(self, issue: DataIssue) -> None:
        """異常を一覧に追加する（エラーは赤、警告は橙）。

        Args:
            issue: 異常。
        """
        is_error = issue.severity is IssueSeverity.ERROR
        item = QListWidgetItem(f"{'エラー' if is_error else '警告'}: {issue.description}")
        item.setForeground(QBrush(_ERROR_COLOR if is_error else _WARNING_COLOR))
        self.issues_list.addItem(item)

    def _show_validation_error(self, error: ValidationError) -> None:
        """指定の検証エラーを一覧に表示する。

        Args:
            error: pydantic の検証エラー。
        """
        for detail in error.errors():
            location = ".".join(str(part) for part in detail["loc"])
            text = f"エラー: {detail['msg']}" + (f"（{location}）" if location else "")
            item = QListWidgetItem(text)
            item.setForeground(QBrush(_ERROR_COLOR))
            self.issues_list.addItem(item)

    # --- 値の取り出し --------------------------------------------------------

    def _file_format(self) -> SourceFormat:
        """選んでいるファイル形式を返す。

        Qt のコンボボックスは列挙値を文字列として保持するため、列挙型に変換して返す。

        Returns:
            ファイル形式。
        """
        return SourceFormat(str(self.format_combo.currentData()))

    def _time_unit(self) -> TimeUnit:
        """選んでいる時刻列の単位を返す（文字列で保持されている値を列挙型に変換する）。

        Returns:
            時刻列の単位。
        """
        return TimeUnit(str(self.time_unit_combo.currentData()))

    def _read_options(self) -> ReadOptions:
        """選んでいる読み込み条件を返す。

        Returns:
            読み込み条件。

        Raises:
            pydantic.ValidationError: 文字コード名が不正な場合など。
        """
        delimiter = self.delimiter_combo.currentData()
        return ReadOptions(
            delimiter=delimiter if isinstance(delimiter, str) else ",",
            encoding=self.encoding_combo.currentText().strip(),
            skip_rows=self.skip_rows_spin.value(),
        )

    def _channel_rows(self, *, include_all: bool) -> list[tuple[str, bool, str]]:
        """チャンネルの表の内容を返す。

        Args:
            include_all: False ならチェックした行だけを返す。

        Returns:
            (列名, チェックしたか, 単位) のリスト。
        """
        rows: list[tuple[str, bool, str]] = []
        for row in range(self.channel_table.rowCount()):
            check_item = self.channel_table.item(row, _COLUMN_CHECK)
            name_item = self.channel_table.item(row, _COLUMN_NAME)
            unit_item = self.channel_table.item(row, _COLUMN_UNIT)
            if check_item is None or name_item is None:
                continue
            checked = check_item.checkState() == Qt.CheckState.Checked
            unit = unit_item.text().strip() if unit_item is not None else ""
            if include_all or checked:
                rows.append((name_item.text(), checked, unit))
        return rows

    def _checked_channels(self) -> list[tuple[str, str]]:
        """取り込むとチェックしたチャンネルを返す。

        Returns:
            (列名, 単位) のリスト。
        """
        return [(name, unit) for name, _, unit in self._channel_rows(include_all=False)]


def _guess_time_unit(column: str) -> TimeUnit:
    """時刻列の名前から単位の既定値を推定する（"time_ms" ならミリ秒など）。

    利用者が確認して変えられる既定値を決めるだけで、データには影響しない。

    Args:
        column: 時刻列の名前。

    Returns:
        推定した単位（分からなければ秒）。
    """
    name = column.lower().replace("[", "_").replace("]", "").replace("(", "_").replace(")", "")
    for suffix, unit in (
        ("ms", TimeUnit.MILLISECOND),
        ("us", TimeUnit.MICROSECOND),
        ("µs", TimeUnit.MICROSECOND),
        ("ns", TimeUnit.NANOSECOND),
    ):
        if name.endswith(suffix):
            return unit
    return TimeUnit.SECOND
