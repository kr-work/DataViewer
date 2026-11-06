"""データファイルの取り込み（SPEC F-02・F-03）: プレビュー、異常の検出、正規化、登録。

取り込みの手順（GUI では重い処理をワーカースレッドで行うため、DB の操作と分けている）:
    1. ``preview()``: 先頭行と、ヘッダー・文字コード・データ行の有無などの異常を確認する
       （読み取りのみ）。
    2. ``resolve_targets()``: 実験・セッション・試技を名前で探す。なければ新規作成する予定として返す
       （DB は読むだけ）。
    3. ``prepare()``: ファイル全体を検証し、元ファイルのコピー（originals/）と正規化データ
       （datasets/ の Parquet）を書く。DB に触れないので、ワーカースレッドで実行できる。
       エラーがあれば何も書かずに ``ImportBlockedError`` を送出する。
    4. ``commit()``: 新しい実験〜チャンネルを 1 トランザクションで DB に登録する。
       失敗したら 3 で書いたファイルを消す。
    ``import_file()`` は 1 を除く 2〜4 を同期的にまとめて実行する（スクリプト・テスト用）。

ファイルを開くだけで始められるよう、``suggest()`` がファイルの中身から指定を自動で決める。
判定するのは読み方（形式・文字コード・区切り文字・前置きの行数・時刻列・単位）だけで、データは
変えない。自信を持って決められないときは推測で取り込まず、利用者に確認を求める。

異常の扱い（SPEC 原則 6「隠して修正しない」）:
    - エラー（取り込みを中止する）: 空のファイル、ヘッダーのみ、列名の重複、文字コードの誤り、
      指定した列がない、数値として読めない値、時刻列の欠損。
    - 警告（取り込んで記録する）: 値の NaN・空欄、時刻の逆行・重複、時刻の飛び（欠損区間の可能性）、
      空行（値を何も含まないため取り除く）。データの並べ替え・補間・除去はしない。
    - 数値の前後の空白（"1.0, 2.0" のような書き方）は書式の違いとして無視して変換する。
      列名は書かれたまま保持する。

元ファイルは一度だけ読み込み、その内容（バイト列）を検証・コピー・ハッシュ計算のすべてに使う。
途中でファイルが変わっても、検証した内容とコピーした内容が食い違わないようにするため。
このモジュールは Qt に依存しない。
"""

import codecs
import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import polars as pl
from pydantic import ValidationError

from experiment_workbench.model import (
    DATASET_TIME_COLUMN,
    DEFAULT_MAX_GAP_FACTOR,
    Channel,
    ChannelImportSpec,
    ColumnPreview,
    DataIssue,
    DataIssueKind,
    Experiment,
    FilePreview,
    ImportRequest,
    ImportSuggestion,
    ImportTargets,
    IssueSeverity,
    PreparedImport,
    ReadOptions,
    SampleRate,
    Session,
    Source,
    SourceFormat,
    TimeAxisKind,
    TimeBase,
    TimeUnit,
    Trial,
)
from experiment_workbench.storage import ProjectDirectory, ProjectStore, StorageError
from experiment_workbench.timebase import to_seconds
from experiment_workbench.validation import FloatArray

PREVIEW_ROWS = 20
"""プレビューに表示する先頭の行数。"""

DEFAULT_GROUP_NAME = "データ"
"""ファイルを開くだけのときの登録先（実験・セッション・試技）の名前。画面の一覧には出さない。"""

_UTF8_BOM = b"\xef\xbb\xbf"
_HASH_CHUNK_BYTES = 1024 * 1024
_ENCODING_CANDIDATES = ("utf-8", "cp932")
"""文字コードの候補（UTF-8 は誤判定しにくいので先に試す）。"""
_DELIMITER_CANDIDATES = (",", "\t", ";", " ")
"""区切り文字の候補（同点なら先のものを選ぶ）。"""
_LAYOUT_SAMPLE_LINES = 50
"""区切り文字と前置きの行数の判定に使う、先頭の行数。"""
_TIME_KEYWORDS = ("time", "timestamp", "elapsed", "時刻", "時間", "経過")
"""時刻列の名前に含まれる語。"""
_TIME_BASE_NAMES = frozenset({"t", "sec", "second", "seconds", "秒"})
"""単位の部分を除いた列名がこれと一致すれば時刻列とみなす（"t"、"t_ms" など）。"""
_UNIT_IN_BRACKETS = re.compile(r"[\[(（［]\s*([^\[\]()（）［］]+?)\s*[\])）］]\s*$")
"""列名の末尾の括弧に書かれた単位（"force[N]"、"accel (m/s^2)" など）。"""
_TIME_UNIT_WORDS = {
    "ms": TimeUnit.MILLISECOND,
    "msec": TimeUnit.MILLISECOND,
    "ミリ秒": TimeUnit.MILLISECOND,
    "us": TimeUnit.MICROSECOND,
    "usec": TimeUnit.MICROSECOND,
    "µs": TimeUnit.MICROSECOND,
    "μs": TimeUnit.MICROSECOND,
    "マイクロ秒": TimeUnit.MICROSECOND,
    "ns": TimeUnit.NANOSECOND,
    "nsec": TimeUnit.NANOSECOND,
    "ナノ秒": TimeUnit.NANOSECOND,
    "s": TimeUnit.SECOND,
    "sec": TimeUnit.SECOND,
    "秒": TimeUnit.SECOND,
}
"""列名に書かれた時刻の単位。"""


class ImportBlockedError(Exception):
    """エラーの異常があるため取り込めない（何も書き込んでいない）。

    Attributes:
        issues: 検出した異常（エラーと、それまでに分かった警告）。
    """

    def __init__(self, issues: list[DataIssue]) -> None:
        """検出した異常を受け取り、エラーの説明をまとめたメッセージを作る。

        Args:
            issues: 検出した異常。1 つ以上のエラーを含むこと。
        """
        self.issues = issues
        errors = [issue.description for issue in issues if issue.severity is IssueSeverity.ERROR]
        super().__init__("取り込めません: " + " / ".join(errors))


class ImportTargetError(ValueError):
    """取り込み先の指定に問題がある（同じ試技に同名の Source が既にあるなど）。"""


class DataImporter:
    """1 つのプロジェクトへデータファイルを取り込むクラス。

    Attributes:
        store: 取り込み先のプロジェクト。
    """

    def __init__(self, store: ProjectStore) -> None:
        """取り込み先のプロジェクトを受け取って初期化する。

        Args:
            store: 取り込み先のプロジェクト。
        """
        self.store = store

    # --- 1. プレビュー -------------------------------------------------------

    def preview(
        self,
        path: Path,
        file_format: SourceFormat,
        read_options: ReadOptions,
        n_rows: int = PREVIEW_ROWS,
    ) -> FilePreview:
        """ファイルの先頭部分と、読み込み前に分かる異常を返す（ファイルは変更しない）。

        列が数値かどうかは、先頭行の値を取り込み時と同じ規則（前後の空白を除いて数値に変換できるか）で
        判定する。先頭行より後ろに数値でない値がある場合は、取り込み時にエラーとして分かる。

        Args:
            path: 取り込むファイル。
            file_format: ファイル形式。
            read_options: 読み込み条件（CSV / TSV で使う）。
            n_rows: 表示する先頭の行数。

        Returns:
            列の情報・先頭行・異常。構造のエラー（空・重複列名・文字コードなど）があれば、行は空になる。
        """
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return FilePreview(columns=[], rows=[], issues=[_read_error(f"読み込めません: {exc}")])
        header, table, issues = _inspect_structure(raw, file_format, read_options)
        if _has_error(issues) or table is None:
            return FilePreview(
                columns=[ColumnPreview(name=name, dtype="", is_numeric=False) for name in header],
                rows=[],
                issues=issues,
            )
        try:
            head = _read_table(table, file_format, read_options, columns=None, n_rows=n_rows)
        except pl.exceptions.PolarsError as exc:
            return FilePreview(
                columns=[], rows=[], issues=[*issues, _read_error(f"表として読み込めません: {exc}")]
            )
        columns = [
            ColumnPreview(
                name=name,
                dtype="数値" if _is_numeric_series(head[name]) else "数値以外",
                is_numeric=_is_numeric_series(head[name]),
            )
            for name in head.columns
        ]
        rows = [[_format_cell(value) for value in row] for row in head.rows()]
        return FilePreview(columns=columns, rows=rows, issues=issues)

    def suggest(self, path: Path, *, shares_trial_clock: bool = True) -> ImportSuggestion:
        """ファイルの中身から取り込みの指定を決める（ファイルは変更しない）。

        時刻列が特定できない、数値の列がない、構造にエラーがあるといった場合は、推測で取り込まず
        ``request`` を None にして確認を求める。時刻の単位が列名から分からない場合は秒とみなすが、
        誤っていると時間軸が丸ごとずれるので、仮定として必ず知らせる。

        Args:
            path: 開くファイル。
            shares_trial_clock: 同じ試技の他のファイルと同じ時計で記録したなら True。

        Returns:
            決めた指定、または確認が必要な理由。
        """
        file_format = file_format_of(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return ImportSuggestion(
                file_format=file_format, review_reasons=[f"ファイルを読み込めません: {exc}"]
            )
        read_options = _sniff_read_options(raw, file_format)
        preview = self.preview(path, file_format, read_options)
        suggestion = ImportSuggestion(file_format=file_format, read_options=read_options)
        errors = [i.description for i in preview.issues if i.severity is IssueSeverity.ERROR]
        if errors:
            return suggestion.model_copy(update={"review_reasons": errors})
        time_column = _confident_time_column(preview)
        if time_column is None:
            return suggestion.model_copy(
                update={"review_reasons": ["時刻の列を判定できませんでした。指定してください"]}
            )
        channels = _channel_specs(preview, time_column)
        if not channels:
            return suggestion.model_copy(
                update={"review_reasons": ["時刻以外の数値の列が見つかりませんでした"]}
            )
        assumptions: list[str] = []
        time_unit = parse_time_unit(time_column)
        if time_unit is None:
            time_unit = TimeUnit.SECOND
            assumptions.append(
                f"時刻列「{time_column}」の単位が列名から分からないため、秒として読みました。"
                "違うときは「読み込み設定を指定して開く」から指定してください"
            )
        if read_options.skip_rows > 0:
            assumptions.append(f"先頭の {read_options.skip_rows} 行は前置きとして読み飛ばしました")
        experiment_name, session_name, trial_name = self.default_target_names()
        request = ImportRequest(
            file_format=file_format,
            read_options=read_options,
            time_column=time_column,
            time_unit=time_unit,
            channels=channels,
            experiment_name=experiment_name,
            session_name=session_name,
            trial_name=trial_name,
            source_name=self.unique_source_name(path.stem, experiment_name, session_name),
            shares_trial_clock=shares_trial_clock,
        )
        return suggestion.model_copy(update={"request": request, "assumptions": assumptions})

    def default_target_names(self) -> tuple[str, str, str]:
        """ファイルを開くだけのときの登録先の名前を返す。

        既に試技があるプロジェクトでは最初の試技を使う（別の階層を増やさないため）。

        Returns:
            (実験名, セッション名, 試技名)。
        """
        database = self.store.database
        for experiment in database.list_entities(Experiment):
            for session in database.list_entities(Session, experiment_id=experiment.id):
                for trial in database.list_entities(Trial, session_id=session.id):
                    return experiment.name, session.name, trial.name
        return DEFAULT_GROUP_NAME, DEFAULT_GROUP_NAME, DEFAULT_GROUP_NAME

    def unique_source_name(self, stem: str, experiment_name: str, session_name: str) -> str:
        """同じ試技の中で重ならない Source 名を返す（"name"、"name_2"、"name_3" …）。

        別のフォルダの同じ名前のファイルも開けるようにするため。

        Args:
            stem: 名前の元（通常はファイル名の拡張子を除いた部分）。
            experiment_name: 登録先の実験名。
            session_name: 登録先のセッション名。

        Returns:
            重ならない名前。
        """
        database = self.store.database
        used: set[str] = set()
        for experiment in database.list_entities(Experiment, name=experiment_name):
            sessions = database.list_entities(
                Session, experiment_id=experiment.id, name=session_name
            )
            for session in sessions:
                for trial in database.list_entities(Trial, session_id=session.id):
                    sources = database.list_entities(Source, trial_id=trial.id)
                    used.update(source.name for source in sources)
        candidate = stem
        number = 2
        while candidate in used:
            candidate = f"{stem}_{number}"
            number += 1
        return candidate

    def has_same_file(self, path: Path) -> bool:
        """同じ内容のファイルをプロジェクトに取り込み済みかを返す。

        Args:
            path: 調べるファイル。

        Returns:
            取り込み済みなら True（読めないファイルは False。取り込みの段階で理由を示す）。
        """
        try:
            digest = file_sha256(path)
        except OSError:
            return False
        sources = self.store.database.list_entities(Source)
        return any(source.original_sha256 == digest for source in sources)

    # --- 2. 取り込み先の解決 ---------------------------------------------------

    def resolve_targets(self, request: ImportRequest) -> ImportTargets:
        """取り込み先の実験・セッション・試技を名前で探す（なければ新規作成の予定にする）。

        新しいエンティティはこの時点では DB に登録せず、``commit()`` でまとめて登録する。
        取り込みが途中で失敗したときに、空の実験などが残らないようにするため。

        Args:
            request: 取り込みの指定。

        Returns:
            取り込み先。

        Raises:
            ImportTargetError: 既存の試技に、同じ名前の Source が既にある場合。
        """
        database = self.store.database
        new_entities: list[Experiment | Session | Trial] = []

        experiment = _first(database.list_entities(Experiment, name=request.experiment_name))
        session: Session | None = None
        trial: Trial | None = None
        if experiment is None:
            experiment = Experiment(name=request.experiment_name)
            new_entities.append(experiment)
        else:
            session = _first(
                database.list_entities(
                    Session, experiment_id=experiment.id, name=request.session_name
                )
            )
        if session is None:
            session = Session(experiment_id=experiment.id, name=request.session_name)
            new_entities.append(session)
        else:
            trial = _first(
                database.list_entities(Trial, session_id=session.id, name=request.trial_name)
            )
        if trial is None:
            trial = Trial(session_id=session.id, name=request.trial_name)
            new_entities.append(trial)
        elif database.list_entities(Source, trial_id=trial.id, name=request.source_name):
            raise ImportTargetError(
                f"試技 {request.trial_name!r} には、同じ名前の Source "
                f"{request.source_name!r} が既にあります。別の名前を指定してください"
            )
        return ImportTargets(
            experiment=experiment, session=session, trial=trial, new_entities=new_entities
        )

    # --- 3. 検証・コピー・正規化 -----------------------------------------------

    def check(self, path: Path, request: ImportRequest) -> list[DataIssue]:
        """取り込む前に、ファイル全体を指定どおりに検証して異常を返す（何も書き込まない）。

        取り込みダイアログで、データの内容に関する警告やエラー（NaN、時刻の逆行、
        数値でない値など）を、取り込みを確定する前に見せるために使う。

        Args:
            path: 取り込むファイル（読み取りのみ）。
            request: 取り込みの指定。

        Returns:
            検出した異常（エラーと警告）。異常がなければ空。
        """
        try:
            _, _, _, issues = _validate_file(path, request)
        except ImportBlockedError as exc:
            return exc.issues
        return issues

    def prepare(self, path: Path, request: ImportRequest, trial_id: UUID) -> PreparedImport:
        """ファイル全体を検証し、元ファイルのコピーと正規化データを書き込む（DB には触れない）。

        Args:
            path: 取り込むファイル（読み取りのみ。変更しない）。
            request: 取り込みの指定。
            trial_id: 登録先の試技の ID（``resolve_targets()`` の結果）。

        Returns:
            DB への登録を待っている取り込み結果。

        Raises:
            ImportBlockedError: エラーの異常がある場合（何も書き込まない）。
            FileExistsError: 書き込み先が既に存在する場合（通常は起こらない）。
        """
        raw, numeric, time_s, issues = _validate_file(path, request)
        prepared = self._build_entities(path, raw, request, trial_id, time_s, issues)
        dataset = pl.DataFrame(
            {
                DATASET_TIME_COLUMN: time_s,
                **{spec.column: numeric[spec.column] for spec in request.channels},
            }
        )
        self._write_files(prepared, raw, dataset)
        return prepared

    def _build_entities(
        self,
        path: Path,
        raw: bytes,
        request: ImportRequest,
        trial_id: UUID,
        time_s: FloatArray,
        issues: list[DataIssue],
    ) -> PreparedImport:
        """登録する Source とチャンネルを組み立てる（ファイルを書く前に検証を済ませるため）。

        Args:
            path: 取り込むファイル。
            raw: ファイルの内容。
            request: 取り込みの指定。
            trial_id: 登録先の試技の ID。
            time_s: 秒に換算した時刻。
            issues: 検出した警告。

        Returns:
            書き込み予定のパスを含む取り込み結果（ファイルはまだ書いていない）。

        Raises:
            ImportBlockedError: ファイル名などが保存できない値の場合。
        """
        source_id = uuid4()
        original_relpath = f"{ProjectDirectory.ORIGINALS}/{source_id}/{path.name}"
        dataset_relpath = f"{ProjectDirectory.DATASETS}/{source_id}.parquet"
        sample_rate = _sample_rate(time_s, request)
        try:
            source = Source(
                id=source_id,
                trial_id=trial_id,
                name=request.source_name,
                format=request.file_format,
                original_filename=path.name,
                original_relpath=original_relpath,
                original_sha256=hashlib.sha256(raw).hexdigest(),
                original_size_bytes=len(raw),
                dataset_relpath=dataset_relpath,
                time_column=request.time_column,
                time_base=_time_base(request),
                read_options=None
                if request.file_format is SourceFormat.PARQUET
                else request.read_options,
                issues=issues,
            )
        except ValidationError as exc:
            raise ImportBlockedError(
                [*issues, _read_error(f"保存できない値があります（ファイル名など）: {exc}")]
            ) from exc
        channels = [
            Channel(
                source_id=source_id,
                name=spec.column,
                original_name=spec.column,
                unit=spec.unit,
                original_unit=spec.unit,
                sample_rate=sample_rate,
            )
            for spec in request.channels
        ]
        return PreparedImport(
            source=source, channels=channels, written_relpaths=[original_relpath, dataset_relpath]
        )

    def _write_files(self, prepared: PreparedImport, raw: bytes, dataset: pl.DataFrame) -> None:
        """元ファイルのコピーと正規化データを書き込み、コピーの完全性を確かめる。

        途中で失敗した場合は、書いたファイルを消してから例外を再送出する。

        Args:
            prepared: 書き込み先のパスを含む取り込み結果。
            raw: 元ファイルの内容。
            dataset: 正規化データ。

        Raises:
            OSError: コピーした内容のハッシュが元の内容と一致しない場合など。
        """
        source = prepared.source
        try:
            saved_sha256 = self.store.write_original(source.original_relpath, raw)
            if saved_sha256 != source.original_sha256:
                raise OSError(
                    f"コピーした元ファイルのハッシュが一致しません（{source.original_relpath}）"
                )
            self.store.write_dataset(source.dataset_relpath, dataset)
        except BaseException:
            self.discard(prepared)
            raise

    # --- 4. 登録 -------------------------------------------------------------

    def commit(self, prepared: PreparedImport, targets: ImportTargets) -> None:
        """新しい実験〜チャンネルを 1 トランザクションで DB に登録する。

        Args:
            prepared: ``prepare()`` の結果。
            targets: ``resolve_targets()`` の結果。

        Raises:
            StorageError: 登録に失敗した場合（書き込んだファイルは削除済み、DB は変更なし）。
        """
        try:
            self.store.database.insert_many(
                [*targets.new_entities, prepared.source, *prepared.channels]
            )
        except StorageError:
            self.discard(prepared)
            raise

    def discard(self, prepared: PreparedImport) -> None:
        """登録しない取り込み結果のファイルを削除する（キャンセル時・失敗時）。

        Args:
            prepared: ``prepare()`` の結果。
        """
        for relpath in prepared.written_relpaths:
            self.store.remove_file(relpath)

    def import_file(self, path: Path, request: ImportRequest) -> PreparedImport:
        """取り込み先の解決・検証・書き込み・登録をまとめて同期的に行う。

        Args:
            path: 取り込むファイル。
            request: 取り込みの指定。

        Returns:
            登録した取り込み結果。

        Raises:
            ImportTargetError: 取り込み先の指定に問題がある場合。
            ImportBlockedError: エラーの異常がある場合。
            StorageError: 登録に失敗した場合。
        """
        targets = self.resolve_targets(request)
        prepared = self.prepare(path, request, targets.trial.id)
        self.commit(prepared, targets)
        return prepared


# ---------------------------------------------------------------------------
# ファイル全体の検証（モジュール内部用）
# ---------------------------------------------------------------------------


def _validate_file(
    path: Path, request: ImportRequest
) -> tuple[bytes, pl.DataFrame, FloatArray, list[DataIssue]]:
    """ファイル全体を指定どおりに読み、取り込みの前提を満たすか検証する（何も書き込まない）。

    Args:
        path: 取り込むファイル。
        request: 取り込みの指定。

    Returns:
        (ファイルの内容, 数値に変換した選択列, 秒に換算した時刻, 警告)。

    Raises:
        ImportBlockedError: エラーの異常がある場合。
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ImportBlockedError([_read_error(f"読み込めません: {exc}")]) from exc

    header, table, issues = _inspect_structure(raw, request.file_format, request.read_options)
    # 構造が読めない（空・文字コードの誤りなど）場合は、
    # 列の有無を調べても意味がないので打ち切る。
    if _has_error(issues) or table is None:
        raise ImportBlockedError(issues)
    selected = [spec.column for spec in request.channels]
    if request.time_column is not None:
        selected = [request.time_column, *selected]
    issues.extend(
        DataIssue(
            kind=DataIssueKind.INVALID_COLUMN,
            description=f"列 {name!r} がファイルにありません",
            column=name,
            severity=IssueSeverity.ERROR,
        )
        for name in selected
        if name not in header
    )
    if _has_error(issues):
        raise ImportBlockedError(issues)

    try:
        frame = _read_table(
            table, request.file_format, request.read_options, columns=selected, n_rows=None
        )
    except pl.exceptions.PolarsError as exc:
        raise ImportBlockedError([*issues, _read_error(f"表として読み込めません: {exc}")]) from exc
    # row_numbers は、空行を除いた後の行からファイル上のデータ行番号への対応。
    # 報告する位置がファイルとずれないようにするため。
    frame, row_numbers, blank_issues = _drop_blank_rows(frame)
    numeric, conversion_issues = _to_float_columns(frame, selected, row_numbers)
    issues.extend(blank_issues)
    issues.extend(conversion_issues)
    if _has_error(issues):
        raise ImportBlockedError(issues)

    if request.time_column is not None:
        raw_time = numeric[request.time_column].to_numpy().astype(np.float64)
        time_s = to_seconds(raw_time, request.time_unit)
        issues.extend(_check_time(time_s, request.time_column, row_numbers))
    else:
        interval = request.sample_interval_s if request.sample_interval_s is not None else 1.0
        time_s = np.arange(numeric.height, dtype=np.float64) * interval
    for spec in request.channels:
        issues.extend(
            _check_channel(frame[spec.column], numeric[spec.column], spec.column, row_numbers)
        )
    if _has_error(issues):
        raise ImportBlockedError(issues)
    return raw, numeric, time_s, issues


# ---------------------------------------------------------------------------
# ファイルの読み方の判定
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """ファイルの SHA-256 を返す（大きなファイルでも少しずつ読む）。

    Args:
        path: ファイル。

    Returns:
        16 進数の文字列。

    Raises:
        OSError: 読めない場合。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_format_of(path: Path) -> SourceFormat:
    """ファイル名の拡張子から形式を判定する。

    Args:
        path: ファイル。

    Returns:
        ファイル形式（.parquet と .tsv 以外は CSV として読む）。
    """
    suffix = path.suffix.lower()
    return {".parquet": SourceFormat.PARQUET, ".tsv": SourceFormat.TSV}.get(
        suffix, SourceFormat.CSV
    )


def parse_time_unit(column: str) -> TimeUnit | None:
    """時刻列の名前に書かれた単位を読み取る（"time_ms"、"t [s]"、"時刻(ミリ秒)" など）。

    単語の区切りを見て判定する。"times" の末尾の s を秒と読まないようにするため。

    Args:
        column: 時刻列の名前。

    Returns:
        単位。名前から分からなければ None。
    """
    name = column.strip().lower()
    bracket = _UNIT_IN_BRACKETS.search(name)
    if bracket is not None:
        return _TIME_UNIT_WORDS.get(bracket.group(1))
    last_word = re.split(r"[\s_\-.]+", name)[-1]
    return _TIME_UNIT_WORDS.get(last_word) if last_word != name else None


def guess_time_column(preview: FilePreview) -> str | None:
    """時刻列の候補を返す（確認画面の初期値に使う）。

    Args:
        preview: ファイルのプレビュー。

    Returns:
        時刻らしい名前の数値の列。なければ最初の数値の列。数値の列がなければ None。
    """
    numeric = [column.name for column in preview.columns if column.is_numeric]
    return _confident_time_column(preview) or (numeric[0] if numeric else None)


def _confident_time_column(preview: FilePreview) -> str | None:
    """名前から時刻列だと判断できる数値の列を返す。

    Args:
        preview: ファイルのプレビュー。

    Returns:
        列名。判断できなければ None。
    """
    for column in preview.columns:
        if column.is_numeric and _looks_like_time(column.name):
            return column.name
    return None


def _looks_like_time(name: str) -> bool:
    """列名が時刻らしいかを返す。

    Args:
        name: 列名。

    Returns:
        時刻らしければ True。
    """
    lowered = name.strip().lower()
    if any(word in lowered for word in _TIME_KEYWORDS):
        return True
    base = re.split(r"[\s_\-.\[(（［]+", _UNIT_IN_BRACKETS.sub("", lowered))[0]
    return base in _TIME_BASE_NAMES


def _channel_specs(preview: FilePreview, time_column: str) -> list[ChannelImportSpec]:
    """時刻列以外の数値の列をチャンネルにする（列名の括弧に書かれた単位を読み取る）。

    Args:
        preview: ファイルのプレビュー。
        time_column: 時刻列の名前。

    Returns:
        チャンネルの指定（ファイル内の順）。
    """
    specs: list[ChannelImportSpec] = []
    for column in preview.columns:
        name = column.name
        if not column.is_numeric or name in (time_column, DATASET_TIME_COLUMN):
            continue
        if not name.strip():
            continue
        bracket = _UNIT_IN_BRACKETS.search(name)
        specs.append(ChannelImportSpec(column=name, unit=bracket.group(1) if bracket else None))
    return specs


def _sniff_read_options(raw: bytes, file_format: SourceFormat) -> ReadOptions:
    """ファイルの内容から、文字コード・区切り文字・前置きの行数を判定する。

    Args:
        raw: ファイルの内容。
        file_format: ファイル形式。

    Returns:
        読み込み条件。判定できない部分は既定値のまま（確認は後段の検証に任せる）。
    """
    if file_format is SourceFormat.PARQUET:
        return ReadOptions()
    encoding, text = _decode_with_candidates(raw)
    delimiters = ("\t",) if file_format is SourceFormat.TSV else _DELIMITER_CANDIDATES
    delimiter, skip_rows = _sniff_layout(text, delimiters)
    return ReadOptions(delimiter=delimiter, encoding=encoding, skip_rows=skip_rows)


def _decode_with_candidates(raw: bytes) -> tuple[str, str]:
    """候補の文字コードで順に読む。

    Args:
        raw: ファイルの内容。

    Returns:
        (文字コード, 読んだ文字列)。どれでも読めなければ ("utf-8", "") を返す
        （後段の検証で文字コードのエラーとして説明する）。
    """
    for encoding in _ENCODING_CANDIDATES:
        try:
            return encoding, raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return _ENCODING_CANDIDATES[0], ""


def _sniff_layout(text: str, delimiters: tuple[str, ...]) -> tuple[str, int]:
    """区切り文字と、ヘッダーの前の前置きの行数を判定する。

    データの行は区切り文字の数が揃うという性質を使う。候補ごとに先頭の行の区切り文字の数を数え、
    最も多くの行に共通する数（1 以上）を持つ候補を選び、その数を持つ最初の行をヘッダーとみなす。

    Args:
        text: ファイルの内容。
        delimiters: 区切り文字の候補（同点なら先のもの）。

    Returns:
        (区切り文字, ヘッダーの前の行数)。
    """
    lines = text.splitlines()[:_LAYOUT_SAMPLE_LINES]
    best = (delimiters[0], 0)
    best_score = 0
    for delimiter in delimiters:
        counts = [len(next(csv.reader([line], delimiter=delimiter), [])) - 1 for line in lines]
        common = Counter(count for count in counts if count > 0).most_common(1)
        if common and common[0][1] > best_score:
            best_score = common[0][1]
            best = (delimiter, counts.index(common[0][0]))
    return best


# ---------------------------------------------------------------------------
# 読み込みと構造の確認（モジュール内部用）
# ---------------------------------------------------------------------------


def _inspect_structure(
    raw: bytes, file_format: SourceFormat, read_options: ReadOptions
) -> tuple[list[str], bytes | None, list[DataIssue]]:
    """ファイル全体を表として読む前に、構造の異常（空・文字コード・列名・データ行の有無）を調べる。

    Polars は列名の重複を黙って改名し、文字コードの誤りを置換文字で読み進めるため、
    ここで先に調べてエラーにする。

    Args:
        raw: ファイルの内容。
        file_format: ファイル形式。
        read_options: 読み込み条件。

    Returns:
        (列名, Polars に渡す内容, 異常)。CSV / TSV の内容は UTF-8 に変換したもの。
        構造のエラーで読めない場合、Polars に渡す内容は None。
    """
    if not raw:
        return [], None, [_error(DataIssueKind.EMPTY_FILE, "ファイルが空です（0 バイト）")]
    if file_format is SourceFormat.PARQUET:
        return _inspect_parquet(raw)

    encoding = read_options.encoding
    # BOM は文字コードの印でありデータではないため、UTF-8 の場合は取り除いて読む。
    if codecs.lookup(encoding).name == "utf-8" and raw.startswith(_UTF8_BOM):
        encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        hint = (
            "Shift_JIS のファイルなら cp932 を指定してください。"
            if encoding.startswith("utf")
            else ""
        )
        return (
            [],
            None,
            [
                _error(
                    DataIssueKind.NON_UTF8_ENCODING,
                    f"文字コード {read_options.encoding} として読めません"
                    f"（{exc.start} バイト目）。{hint}",
                )
            ],
        )

    reader = csv.reader(io.StringIO(text), delimiter=read_options.delimiter)
    for _ in range(read_options.skip_rows):
        next(reader, None)
    header = next(reader, None)
    if header is None:
        return (
            [],
            None,
            [
                _error(
                    DataIssueKind.EMPTY_FILE,
                    "ヘッダー行がありません"
                    f"（読み飛ばす行数 {read_options.skip_rows} を確認してください）",
                )
            ],
        )
    issues = _check_header(header)
    if not any(any(field.strip() for field in row) for row in reader):
        issues.append(_error(DataIssueKind.HEADER_ONLY, "ヘッダー行だけでデータ行がありません"))
    return header, text.encode("utf-8"), issues


def _inspect_parquet(raw: bytes) -> tuple[list[str], bytes | None, list[DataIssue]]:
    """Parquet ファイルの列名と行の有無を調べる。

    Args:
        raw: ファイルの内容。

    Returns:
        (列名, Polars に渡す内容, 異常)。
    """
    try:
        header = list(pl.read_parquet_schema(io.BytesIO(raw)))
        height = pl.read_parquet(io.BytesIO(raw), n_rows=1).height
    except (OSError, pl.exceptions.PolarsError) as exc:
        return [], None, [_read_error(f"Parquet として読み込めません: {exc}")]
    issues = _check_header(header)
    if height == 0:
        issues.append(_error(DataIssueKind.HEADER_ONLY, "データ行がありません"))
    return header, raw, issues


def _check_header(header: list[str]) -> list[DataIssue]:
    """列名の重複（エラー）と空の列名（警告。その列は取り込めない）を調べる。

    Args:
        header: 列名（ファイル内の順）。

    Returns:
        異常。
    """
    issues = [
        _error(
            DataIssueKind.DUPLICATE_COLUMN,
            f"列名 {name!r} が {header.count(name)} 回現れます。"
            "どの列を指すか定まらないため取り込めません",
            column=name,
        )
        for name in sorted({name for name in header if header.count(name) > 1})
    ]
    issues.extend(
        DataIssue(
            kind=DataIssueKind.INVALID_COLUMN,
            description=f"{index + 1} 列目の列名が空です。この列は取り込めません",
        )
        for index, name in enumerate(header)
        if not name.strip()
    )
    return issues


def _read_table(
    table: bytes,
    file_format: SourceFormat,
    read_options: ReadOptions,
    *,
    columns: list[str] | None,
    n_rows: int | None,
) -> pl.DataFrame:
    """表を読み込む。CSV / TSV はすべての列を文字列として読む（数値への変換は後で明示的に行う）。

    型の推論に任せると、先頭が整数で後ろに小数がある列などで読み込みに失敗するため、
    文字列で読んでから ``_to_float_columns()`` で変換し、変換できない値を位置付きで報告する。

    Args:
        table: Polars に渡す内容（CSV / TSV は UTF-8 に変換済み）。
        file_format: ファイル形式。
        read_options: 読み込み条件。
        columns: 読み込む列。None ならすべて。
        n_rows: 読み込む行数。None ならすべて。

    Returns:
        読み込んだ表。

    Raises:
        polars.exceptions.PolarsError: 表として読み込めない場合（列数の不揃いなど）。
    """
    if file_format is SourceFormat.PARQUET:
        return pl.read_parquet(io.BytesIO(table), columns=columns, n_rows=n_rows)
    return pl.read_csv(
        table,
        separator=read_options.delimiter,
        skip_rows=read_options.skip_rows,
        has_header=True,
        columns=columns,
        n_rows=n_rows,
        infer_schema_length=0,  # 0 はすべての列を文字列として読む指定
    )


def _drop_blank_rows(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray[tuple[int], np.dtype[np.int64]], list[DataIssue]]:
    """選んだ列がすべて空の行（空行）を取り除き、その事実を警告として返す。

    空行は値を何も含まないため取り除いてもデータは失われない。ただし、ファイルの構造の
    異常である可能性があるので、隠さずに件数と位置を報告する。

    Args:
        frame: 読み込んだ表。

    Returns:
        (空行を除いた表, 各行のファイル上のデータ行番号（0 始まり）, 警告)。
    """
    blank = frame.select(pl.all_horizontal(pl.all().is_null())).to_series().to_numpy()
    row_numbers = np.flatnonzero(~blank).astype(np.int64)
    count = int(blank.sum())
    if count == 0:
        return frame, row_numbers, []
    first = int(np.argmax(blank))
    issue = DataIssue(
        kind=DataIssueKind.NAN_VALUES,
        description=f"空行が {count} 行あります（最初は{_row_label(first)}）。"
        "値を含まないため取り除きました",
        row_index=first,
    )
    return frame.filter(~pl.Series(blank)), row_numbers, [issue]


def _to_float_columns(
    frame: pl.DataFrame,
    columns: list[str],
    row_numbers: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> tuple[pl.DataFrame, list[DataIssue]]:
    """指定した列を Float64 に変換し、数値として読めない値をエラーとして報告する。

    文字列の列は前後の空白を除いてから変換する（"1.0, 2.0" のような書き方に対応するため）。
    "NaN"・"inf" などは数値として扱い、空欄は欠損（null）のままにする。

    Args:
        frame: 読み込んだ表。
        columns: 変換する列。
        row_numbers: 各行のファイル上のデータ行番号（報告用）。

    Returns:
        (変換後の表（指定した列のみ）, 異常)。
    """
    converted: dict[str, pl.Series] = {}
    issues: list[DataIssue] = []
    for name in columns:
        series = frame[name]
        if series.dtype == pl.String:
            series = series.str.strip_chars()
        values = series.cast(pl.Float64, strict=False)
        failed = values.is_null() & series.is_not_null()
        if failed.any():
            position = int(failed.arg_true()[0])
            row = int(row_numbers[position])
            issues.append(
                _error(
                    DataIssueKind.NON_NUMERIC_VALUES,
                    f"列 {name!r} に数値として読めない値が {int(failed.sum())} 個あります"
                    f"（最初は{_row_label(row)}の {series[position]!r}）",
                    column=name,
                    row_index=row,
                )
            )
        converted[name] = values
    return pl.DataFrame(converted), issues


def _is_numeric_series(series: pl.Series) -> bool:
    """列が数値として取り込めるかを、取り込み時と同じ規則で判定する（プレビュー用）。

    Args:
        series: 列。

    Returns:
        欠損でない値がすべて数値に変換できれば True（すべて欠損の列は False）。
    """
    if series.dtype.is_numeric():
        return True
    if series.dtype != pl.String:
        return False
    stripped = series.str.strip_chars()
    values = stripped.cast(pl.Float64, strict=False)
    has_value = stripped.is_not_null() & (stripped != "")
    return bool(has_value.any()) and not bool((values.is_null() & has_value).any())


# ---------------------------------------------------------------------------
# データの確認（モジュール内部用）
# ---------------------------------------------------------------------------


def _check_time(
    time_s: FloatArray,
    column: str,
    row_numbers: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> list[DataIssue]:
    """時刻の欠損（エラー）と、逆行・重複・飛び（警告）を調べる。

    Args:
        time_s: 秒に換算した時刻。
        column: 時刻列の名前（メッセージ用）。
        row_numbers: 各行のファイル上のデータ行番号（報告用）。

    Returns:
        異常。
    """
    not_finite = ~np.isfinite(time_s)
    if not_finite.any():
        row = int(row_numbers[int(np.argmax(not_finite))])
        return [
            _error(
                DataIssueKind.NAN_VALUES,
                f"時刻列 {column!r} に欠損・無限大が {int(not_finite.sum())} 個あります"
                f"（最初は{_row_label(row)}）。時刻のない行は取り込めません",
                column=column,
                row_index=row,
            )
        ]
    issues: list[DataIssue] = []
    steps = np.diff(time_s)
    decreasing = steps < 0
    if decreasing.any():
        row = int(row_numbers[int(np.argmax(decreasing)) + 1])
        issues.append(
            DataIssue(
                kind=DataIssueKind.NON_MONOTONIC_TIME,
                description=f"時刻が {int(decreasing.sum())} か所で逆行しています"
                f"（最初は{_row_label(row)}）。並べ替えずにそのまま取り込みます",
                column=column,
                row_index=row,
            )
        )
    repeated = steps == 0
    if repeated.any():
        row = int(row_numbers[int(np.argmax(repeated)) + 1])
        issues.append(
            DataIssue(
                kind=DataIssueKind.DUPLICATE_TIMESTAMP,
                description=f"同じ時刻が {int(repeated.sum())} か所で続いています"
                f"（最初は{_row_label(row)}）。そのまま取り込みます",
                column=column,
                row_index=row,
            )
        )
    positive = steps[steps > 0]
    if positive.size:
        median = float(np.median(positive))
        gaps = steps > DEFAULT_MAX_GAP_FACTOR * median
        if gaps.any():
            row = int(row_numbers[int(np.argmax(gaps)) + 1])
            issues.append(
                DataIssue(
                    kind=DataIssueKind.MISSING_SEGMENT,
                    description=f"時刻の間隔が中央値 {median:.6g} s の "
                    f"{DEFAULT_MAX_GAP_FACTOR} 倍を超える箇所が {int(gaps.sum())} か所あります"
                    f"（最大 {float(steps.max()):.6g} s、最初は{_row_label(row)}）。"
                    "欠損区間の可能性があります",
                    column=column,
                    row_index=row,
                )
            )
    return issues


def _check_channel(
    original: pl.Series,
    values: pl.Series,
    column: str,
    row_numbers: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> list[DataIssue]:
    """チャンネルの欠損（"NaN" などの値と空欄）を数えて警告にする。

    Args:
        original: 読み込んだままの列（空欄は null）。
        values: Float64 に変換した列。
        column: 列名（メッセージ用）。
        row_numbers: 各行のファイル上のデータ行番号（報告用）。

    Returns:
        異常（欠損がなければ空）。
    """
    blank = original.is_null()
    not_a_number = values.is_nan().fill_null(value=False)
    missing = blank | not_a_number
    if not missing.any():
        return []
    row = int(row_numbers[int(missing.arg_true()[0])])
    return [
        DataIssue(
            kind=DataIssueKind.NAN_VALUES,
            description=f"列 {column!r} に欠損が {int(missing.sum())} 個あります"
            f"（NaN などの値 {int(not_a_number.sum())} 個、空欄 {int(blank.sum())} 個。"
            f"最初は{_row_label(row)}）。欠損のまま保存します",
            column=column,
            row_index=row,
        )
    ]


# ---------------------------------------------------------------------------
# モデルの組み立て・小さな補助関数（モジュール内部用）
# ---------------------------------------------------------------------------


def _time_base(request: ImportRequest) -> TimeBase:
    """取り込みの指定から、Source の時間軸の定義を作る。

    Args:
        request: 取り込みの指定。

    Returns:
        時間軸の定義。
    """
    if request.time_column is None:
        return TimeBase(
            kind=TimeAxisKind.SAMPLE_INDEX,
            sample_interval_s=request.sample_interval_s,
            shares_trial_clock=request.shares_trial_clock,
        )
    if request.time_is_absolute:
        return TimeBase(
            kind=TimeAxisKind.ABSOLUTE_EPOCH_NS,
            original_unit=request.time_unit,
            origin="UNIX エポック（1970-01-01 00:00:00 UTC）",
            shares_trial_clock=request.shares_trial_clock,
        )
    return TimeBase(
        kind=TimeAxisKind.RELATIVE_SECONDS,
        original_unit=request.time_unit,
        shares_trial_clock=request.shares_trial_clock,
    )


def _sample_rate(time_s: FloatArray, request: ImportRequest) -> SampleRate:
    """チャンネルのサンプリング周波数を求める。

    時刻列がある場合は間隔の中央値から推定し（欠損区間の影響を受けにくい）、
    時刻列がない場合はユーザーが指定した間隔を公称値とする。

    Args:
        time_s: 秒に換算した時刻。
        request: 取り込みの指定。

    Returns:
        サンプリング周波数。
    """
    if request.time_column is None:
        interval = request.sample_interval_s
        return SampleRate(nominal_hz=None if interval is None else 1.0 / interval)
    steps = np.diff(time_s)
    positive = steps[steps > 0]
    if positive.size == 0:
        return SampleRate()
    return SampleRate(estimated_hz=float(1.0 / np.median(positive)))


def _has_error(issues: list[DataIssue]) -> bool:
    """エラーの異常が含まれるかを返す。

    Args:
        issues: 異常。

    Returns:
        エラーがあれば True。
    """
    return any(issue.severity is IssueSeverity.ERROR for issue in issues)


def _error(
    kind: DataIssueKind,
    description: str,
    *,
    column: str | None = None,
    row_index: int | None = None,
) -> DataIssue:
    """エラーの異常を作る。

    Args:
        kind: 異常の種類。
        description: 説明。
        column: 列名。
        row_index: 最初に該当した行。

    Returns:
        エラーの異常。
    """
    return DataIssue(
        kind=kind,
        description=description,
        column=column,
        severity=IssueSeverity.ERROR,
        row_index=row_index,
    )


def _read_error(description: str) -> DataIssue:
    """ファイルを読めないことを表すエラーを作る。

    Args:
        description: 説明。

    Returns:
        エラーの異常。
    """
    return _error(DataIssueKind.READ_ERROR, description)


def _row_label(row_index: int) -> str:
    """0 始まりのデータ行番号を、利用者向けの 1 始まりの表記にする。

    Args:
        row_index: データ行の番号（0 始まり、ヘッダーを除く）。

    Returns:
        例: " データ 1 行目"（前に空白を含む。文中に埋め込むため）。
    """
    return f" データ {row_index + 1} 行目"


def _format_cell(value: object) -> str:
    """プレビュー表示用に値を文字列にする（欠損は空文字）。

    Args:
        value: 値。

    Returns:
        表示用の文字列。
    """
    return "" if value is None else str(value)


def _first[EntityT](items: list[EntityT]) -> EntityT | None:
    """リストの先頭要素を返す（空なら None）。

    Args:
        items: リスト。

    Returns:
        先頭要素、または None。
    """
    return items[0] if items else None
