"""解析の実行・表示・保存・復元の流れ（SPEC F-11、Phase 3）。

メイン画面から解析に関する処理を切り出したもの。解析パネル（条件の編集）・縦並べ表示・
プロジェクト（保存先）の間をつなぐ。

実行の流れ:
    1. 解析パネルが条件（AnalysisSpec）を送る → ``run()``
    2. ワーカースレッドで ``analysis.run_analysis()`` を実行する（進捗を受け取り、中止できる）。
       新しい実行を始めると、前の実行は中止し、結果が届いても捨てる（世代番号で判別する）。
    3. 結果を縦並べ表示に反映する。表示中の出力はその場で描き直す。初めての結果は、入力と同じ量
       なら入力のパネルに重ね、単位が違う量（微分・積分など）なら入力のすぐ下のパネルに置く
       （重ねると縦軸が大きい方に合わせられて、元の線が読めなくなるため）。追加の出力（取り除いた
       成分・イノベーション・√R）は、左の一覧のチェックで必要なものだけを表示する。
       2 チャンネルの解析は時系列を出力しないので、入力と出力のチャンネルを表示し、
       指標とグラフを ``result_ready`` で知らせ、時刻の印を縦並べ表示に引く。
    4. 計算できた処理は自動で保存する（利用者が保存を意識しなくてよいように）。

解析する範囲は、グラフの上の青い帯（範囲の選択）と解析パネルの欄のどちらでも変えられる。

保存と復元:
    - ``_save()`` は条件を pipelines/<ID>.json に、計算結果を derived/ のキャッシュに保存する。
    - ``load_series()`` は、表示の復元などで系列 ID（チャンネル ID または解析の出力 ID）から
      系列を作る。解析の出力は、キャッシュがあれば読み、なければ計算し直してキャッシュに保存する。
"""

import threading
from uuid import UUID

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import QApplication

from experiment_workbench.analysis import (
    AnalysisCancelledError,
    AnalysisInputError,
    analysis_fingerprint,
    output_infos,
    overlays_input,
    result_from_frame,
    result_to_frame,
    run_analysis,
    series_from_channel,
)
from experiment_workbench.model import (
    TWO_CHANNEL_ANALYSIS_KINDS,
    AnalysisOutputInfo,
    AnalysisResult,
    AnalysisSpec,
    Channel,
    ChannelData,
    PlotSeries,
    Source,
)
from experiment_workbench.storage import EntityNotFoundError, ProjectStore, StorageError
from experiment_workbench.timebase import time_domain_key
from experiment_workbench.ui.analysis_panel import AnalysisPanel
from experiment_workbench.ui.plot_workspace import PlotPanel, StackedPlotWorkspace

_SHUTDOWN_WAIT_MS = 5000
"""終了時に、中止した計算が止まるのを待つ最大時間 [ms]。"""

_DEFAULT_RANGE_MARGIN = 0.3
"""範囲の選択の初期値で、表示範囲の両端から空ける割合（中央の 4 割を選ぶ）。"""

_PRIMARY_OUTPUT_KEYS = ("output", "state_0")
"""処理の主な結果の出力のキー。"""


class _AnalysisSignals(QObject):
    """ワーカースレッドの解析から、メインスレッドへ結果を届けるシグナル。"""

    finished = Signal(int, object, object)
    """計算が終わった（引数: 世代番号, AnalysisSpec, AnalysisResult）。"""
    failed = Signal(int, object)
    """失敗・中止した（引数: 世代番号, 発生した例外）。"""
    progress = Signal(int, float)
    """進捗（引数: 世代番号, 0〜1）。"""


class _ProgressRelay:
    """解析の進捗をシグナルで届け、中止の指示を伝える（KalmanFilter.run の progress に渡す）。

    関数内関数を使わずに、状態（世代番号・中止の合図）を持った呼び出し可能なオブジェクトにしている。
    """

    def __init__(self, generation: int, cancel: threading.Event, signals: _AnalysisSignals) -> None:
        """世代番号・中止の合図・シグナルを受け取る。

        Args:
            generation: 実行の世代番号。
            cancel: 中止の合図（セットされたら中止する）。
            signals: 進捗を届けるシグナル。
        """
        self._generation = generation
        self._cancel = cancel
        self._signals = signals

    def __call__(self, fraction: float) -> bool:
        """進捗を届け、続けるかどうかを返す。

        Args:
            fraction: 進捗（0〜1）。

        Returns:
            中止の合図がなければ True（続ける）。
        """
        self._signals.progress.emit(self._generation, fraction)
        return not self._cancel.is_set()


class _AnalysisTask(QRunnable):
    """ワーカースレッドで解析を実行するタスク。

    DB には触れず、入力データは渡されたものを使う（SQLite の接続はスレッドをまたげないため）。
    """

    def __init__(
        self,
        generation: int,
        spec: AnalysisSpec,
        data: ChannelData,
        output_data: ChannelData | None,
        progress: _ProgressRelay,
        signals: _AnalysisSignals,
    ) -> None:
        """実行に必要な値を受け取る。

        Args:
            generation: 実行の世代番号。
            spec: 解析の条件。
            data: 入力チャンネルのデータ。
            output_data: 出力チャンネルのデータ（2 チャンネルの解析だけ）。
            progress: 進捗の受け取りと中止の判定。
            signals: 結果を届けるシグナル（メインスレッドで作ったもの）。
        """
        super().__init__()
        self._generation = generation
        self._spec = spec
        self._data = data
        self._output_data = output_data
        self._progress = progress
        self._signals = signals

    def run(self) -> None:
        """解析を実行し、結果をシグナルで届ける。"""
        try:
            result = run_analysis(
                self._spec, self._data, self._progress, output_data=self._output_data
            )
        except Exception as exc:
            # どんな失敗も、メインスレッドで理由を表示するために届ける。
            self._signals.failed.emit(self._generation, exc)
            return
        self._signals.finished.emit(self._generation, self._spec, result)


class AnalysisController(QObject):
    """解析の実行・表示・保存・復元をまとめて扱うクラス。

    Attributes:
        workspace: 縦並べ表示。
        panel: 解析パネル。
    """

    analyses_changed = Signal()
    """処理の組や出力の構成が変わった（一覧の作り直しに使う）。"""
    analysis_updated = Signal(object)
    """保存した処理の条件だけが変わった（引数: AnalysisSpec。一覧の名前の更新に使う）。"""
    message = Signal(str, str)
    """利用者に知らせるメッセージ（引数: 本文, 水準 "info" / "warning" / "error"）。"""
    result_ready = Signal(object, object)
    """2 チャンネルの解析の結果が出た（引数: AnalysisSpec, AnalysisResult）。"""
    result_removed = Signal(object)
    """解析が削除された（引数: 解析の ID）。"""

    def __init__(
        self,
        workspace: StackedPlotWorkspace,
        panel: AnalysisPanel,
        thread_pool: QThreadPool,
        parent: QObject | None = None,
    ) -> None:
        """縦並べ表示・解析パネル・スレッドプールを受け取り、パネルの指示につなぐ。

        Args:
            workspace: 縦並べ表示。
            panel: 解析パネル。
            thread_pool: 解析を実行するスレッドプール。
            parent: 親オブジェクト。
        """
        super().__init__(parent)
        self.workspace = workspace
        self.panel = panel
        self._thread_pool = thread_pool
        self._store: ProjectStore | None = None
        self._analyses: list[AnalysisSpec] = []
        self._memo: dict[tuple[UUID, str], AnalysisResult] = {}
        self._last_run: tuple[AnalysisSpec, AnalysisResult] | None = None
        # 縦並べ表示に時刻の印を引いている解析の ID（解析を削除したら印も消すため）。
        self._marked_analysis: UUID | None = None
        self._generation = 0
        self._cancel: threading.Event | None = None
        self._running = False
        # 世代番号 → 実行中の計算の結果を届けるシグナル。計算の途中でシグナルのオブジェクトが
        # 消えないよう、結果（完了・失敗・中止）が届くまで持ち、届いたら破棄する。
        self._signals: dict[int, _AnalysisSignals] = {}
        panel.run_requested.connect(self.run)
        panel.cancel_requested.connect(self.cancel)
        panel.methods_shown.connect(self._on_methods_shown)
        panel.range_selection_changed.connect(self._on_range_selection_changed)
        panel.fit_range_requested.connect(self._fit_range_to_view)
        workspace.range_selected.connect(self._on_range_selected)

    # --- プロジェクト --------------------------------------------------------

    def set_store(self, store: ProjectStore | None) -> None:
        """プロジェクトを切り替える（実行中の計算は中止し、状態を初期化する）。

        Args:
            store: 開いたプロジェクト。閉じたときは None。
        """
        self.cancel()
        self._generation += 1  # 前のプロジェクトの計算結果が届いても捨てる
        self._running = False
        self.panel.set_running(running=False)
        self._store = store
        self._memo.clear()
        self._last_run = None
        self._marked_analysis = None
        self.workspace.clear_time_markers()
        self.panel.set_input(None)
        self._analyses = []
        if store is not None:
            self.reload_analyses()

    def reload_analyses(self) -> None:
        """保存した解析の一覧を読み込み直す（読めないファイルはメッセージで知らせる）。"""
        if self._store is None:
            self._analyses = []
            return
        self._analyses, problems = self._store.load_analyses()
        for problem in problems:
            self.message.emit(problem, "warning")

    @property
    def analyses(self) -> list[AnalysisSpec]:
        """保存した解析（作成日時の順、コピー）。"""
        return list(self._analyses)

    @property
    def is_running(self) -> bool:
        """解析の計算中なら True。"""
        return self._running

    def outputs_of(self, spec: AnalysisSpec) -> list[AnalysisOutputInfo]:
        """解析の出力の一覧を返す（入力チャンネルが見つからなければ空）。

        Args:
            spec: 解析。

        Returns:
            出力の情報。
        """
        channel = self._channel(spec.input_channel_id)
        return [] if channel is None else output_infos(spec, channel)

    # --- 実行 ----------------------------------------------------------------

    def run(self, spec: AnalysisSpec) -> bool:
        """解析をワーカースレッドで実行する（実行中の計算は中止して置き換える）。

        Args:
            spec: 解析の条件。

        Returns:
            開始できたら True（入力データが読めない場合は False）。
        """
        data = self.panel.input_data
        if data is None or data.channel.id != spec.input_channel_id:
            data = self.load_channel_data(spec.input_channel_id)
        if data is None:
            self.panel.show_error("入力チャンネルのデータを読み込めません")
            return False
        output_data: ChannelData | None = None
        if spec.output_channel_id is not None:
            output_data = self.load_channel_data(spec.output_channel_id)
            if output_data is None:
                self.panel.show_error("出力チャンネルのデータを読み込めません")
                return False
        self._sync_range_selector(spec, data)
        self.cancel()
        self._generation += 1
        cancel = threading.Event()
        self._cancel = cancel
        signals = _AnalysisSignals(self)
        signals.finished.connect(self._on_finished)
        signals.failed.connect(self._on_failed)
        signals.progress.connect(self._on_progress)
        self._signals[self._generation] = signals
        relay = _ProgressRelay(self._generation, cancel, signals)
        task = _AnalysisTask(self._generation, spec, data, output_data, relay, signals)
        self._thread_pool.start(task)
        self._running = True
        self.panel.set_running(running=True)
        return True

    def cancel(self) -> None:
        """実行中の計算に中止を指示する（計算は次の確認のときに止まる）。"""
        if self._cancel is not None:
            self._cancel.set()

    def shutdown(self) -> None:
        """終了時に、実行中の計算を中止して止まるのを待つ。"""
        self.cancel()
        self._generation += 1
        self._thread_pool.waitForDone(_SHUTDOWN_WAIT_MS)

    def _on_progress(self, generation: int, fraction: float) -> None:
        """現在の実行の進捗を解析パネルに表示する。

        Args:
            generation: 実行の世代番号。
            fraction: 進捗（0〜1）。
        """
        if generation == self._generation:
            self.panel.set_progress(fraction)

    def _release_signals(self, generation: int) -> None:
        """結果が届いた実行のシグナルのオブジェクトを破棄する（自動再計算で溜まり続けないように）。

        ワーカースレッドは完了・失敗のシグナルを最後に 1 回だけ出すので、その後は使われない。
        既に届いている進捗のシグナルは、破棄の前に処理される（deleteLater は後から処理されるため）。

        Args:
            generation: 実行の世代番号。
        """
        signals = self._signals.pop(generation, None)
        if signals is not None:
            signals.deleteLater()

    def _on_finished(self, generation: int, spec: AnalysisSpec, result: AnalysisResult) -> None:
        """計算が終わったら、結果を覚えて縦並べ表示に反映する（古い実行の結果は捨てる）。

        Args:
            generation: 実行の世代番号。
            spec: 解析の条件。
            result: 解析の結果。
        """
        self._release_signals(generation)
        if generation != self._generation:
            return
        self._running = False
        self.panel.set_running(running=False)
        self._last_run = (spec, result)
        self._memo[(spec.id, analysis_fingerprint(spec))] = result
        is_new = all(saved.id != spec.id for saved in self._analyses)
        self._display(spec, result, is_new=is_new)
        current = self.panel.is_editing and self.panel.analysis_id == spec.id
        if spec.kind not in TWO_CHANNEL_ANALYSIS_KINDS:
            if current:
                self.panel.show_summary([result.headline, *result.summary])
        elif current:
            # 要約は解析結果の欄に出るので、解析パネルには繰り返さない。
            # 別の画面に移った後で届いた結果は、その画面の表示を邪魔しないよう出さない。
            self.panel.show_summary([])
            self._publish_two_channel_result(spec, result)
        self._save(spec, is_new=is_new)

    def _publish_two_channel_result(self, spec: AnalysisSpec, result: AnalysisResult) -> None:
        """2 チャンネルの解析の時刻の印を縦並べ表示に引き、結果を知らせる。

        時刻の印は、入力チャンネルの時間軸のパネルに引く（印の時刻は入力の時間軸で表すため）。

        Args:
            spec: 解析の条件。
            result: 解析の結果。
        """
        source = self._source_of(spec.input_channel_id)
        if source is not None and result.time_markers:
            self.workspace.show_time_markers(time_domain_key(source), result.time_markers)
        else:
            self.workspace.clear_time_markers()
        self._marked_analysis = spec.id
        self.result_ready.emit(spec, result)

    def _on_failed(self, generation: int, error: object) -> None:
        """計算の失敗・中止を解析パネルに表示する（古い実行のものは無視する）。

        Args:
            generation: 実行の世代番号。
            error: 発生した例外。
        """
        self._release_signals(generation)
        if generation != self._generation:
            return
        self._running = False
        self.panel.set_running(running=False)
        if isinstance(error, AnalysisCancelledError):
            self.panel.show_status("計算を中止しました")
        elif isinstance(error, AnalysisInputError | ValueError):
            self.panel.show_error(f"計算できません: {error}")
            self.message.emit(f"解析を計算できません: {error}", "warning")
        else:
            self.panel.show_error(f"計算に失敗しました: {error}")
            self.message.emit(f"解析の計算に失敗しました: {error}", "error")

    # --- 表示 ----------------------------------------------------------------

    def _display(self, spec: AnalysisSpec, result: AnalysisResult, *, is_new: bool) -> None:
        """解析の結果を縦並べ表示に反映する。

        表示中の出力はその場で描き直す。新しい処理の結果だけを置く（利用者がチェックを外した
        出力を、再計算で勝手に戻さないため）。入力チャンネルを表示していなければ、入力も表示する。

        Args:
            spec: 解析の条件。
            result: 解析の結果。
            is_new: 新しい処理の最初の結果なら True。
        """
        if spec.kind in TWO_CHANNEL_ANALYSIS_KINDS:
            self._show_two_channels(spec)
            return
        for series in result.series:
            self.workspace.update_series(series)
        if not is_new:
            return
        infos = self.outputs_of(spec)
        by_key = {info.key: series for info, series in zip(infos, result.series, strict=False)}
        primary = by_key.get("output") or by_key.get("state_0")
        if primary is None:
            return
        self._panel_showing(spec.input_channel_id)
        if not self.workspace.panels_showing(primary.series_id):
            self._place_output(spec, primary, is_main=True)

    def _place_output(self, spec: AnalysisSpec, series: PlotSeries, *, is_main: bool) -> bool:
        """解析の出力を、元の線と比べやすい位置に置く。

        Args:
            spec: 解析。
            series: 置く出力。
            is_main: 主な結果なら True。

        Returns:
            置けたら True。
        """
        showing = self.workspace.panels_showing(spec.input_channel_id)
        if not showing:
            return self.workspace.show_series(series)
        if is_main and overlays_input(spec):
            return self.workspace.add_series(series, showing[0])
        return self.workspace.add_series(series, self.workspace.add_panel(after=showing[0]))

    def _panel_showing(self, channel_id: UUID) -> PlotPanel:
        """チャンネルを表示しているパネルを返す（解析の結果を重ねる先）。

        表示していなければ、空の選択中パネル、またはなければ新しいパネルに置く。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            チャンネルを表示しているパネル。
        """
        showing = self.workspace.panels_showing(channel_id)
        if showing:
            return showing[0]
        active = self.workspace.active_panel
        target = active if active is not None and not active.series else self.workspace.add_panel()
        raw = self.load_series(channel_id)
        if raw is not None:
            self.workspace.add_series(raw, target)
        return target

    def _show_two_channels(self, spec: AnalysisSpec) -> None:
        """2 チャンネルの解析の入力と出力を表示する（出力は別のパネルに置く）。

        縦並べで見比べられるようにするため。表示済みなら何もしない。
        新しいパネルを足しても、選択中のパネルは元に戻す。

        Args:
            spec: 解析の条件。
        """
        self._panel_showing(spec.input_channel_id)
        output_id = spec.output_channel_id
        if output_id is None or self.workspace.panels_showing(output_id):
            return
        previous = self.workspace.active_panel
        output = self.load_series(output_id)
        if output is not None:
            self.workspace.add_series(output, self.workspace.add_panel())
        if previous is not None:
            self.workspace.set_active_panel(previous)

    def show_series(self, series_id: UUID) -> bool:
        """一覧のチェックで選ばれた系列を表示する。

        入力と同じ量の結果は入力チャンネルのパネルに重ね、それ以外の出力は入力のすぐ下のパネルに
        置く。チャンネルは新しいパネルに置く。

        Args:
            series_id: 系列 ID。

        Returns:
            表示できたら True。
        """
        series = self.load_series(series_id)
        if series is None:
            return False
        spec = next((s for s in self._analyses if s.id == series.analysis_id), None)
        if spec is None:
            return self.workspace.show_series(series)
        infos = self.outputs_of(spec)
        primary = next((i for i in infos if i.key in _PRIMARY_OUTPUT_KEYS), None)
        is_main = primary is not None and primary.series_id == series_id
        return self._place_output(spec, series, is_main=is_main)

    def _on_methods_shown(self) -> None:
        """処理の一覧に戻ったら、調整していた処理の時刻の印を片付ける。"""
        self._marked_analysis = None
        self.workspace.clear_time_markers()

    # --- 解析する範囲の選択 ----------------------------------------------------

    def _on_range_selection_changed(self, active: bool) -> None:
        """解析パネルの指定に合わせて、グラフの上の範囲の選択を表示する・消す。

        欄の範囲が入力のデータと重ならなければ（入力を変えたときなど）、表示範囲の中央を選ぶ。

        Args:
            active: 範囲の選択を表示するなら True。
        """
        data = self.panel.input_data
        if not active or data is None:
            self.workspace.hide_range_selector()
            return
        target = self._panel_showing(data.channel.id)
        selected = self.panel.range_values()
        first, last = float(data.time_s[0]), float(data.time_s[-1])
        if selected is None or selected[1] <= first or selected[0] >= last:
            selected = _default_range(target)
            self.panel.set_range(*selected)
        self.workspace.show_range_selector(time_domain_key(data.source), *selected)

    def _on_range_selected(self, start: float, end: float) -> None:
        """グラフの上で動かした範囲を、解析パネルの欄に反映する（自動再計算の対象になる）。

        Args:
            start: 開始 [s]。
            end: 終了 [s]。
        """
        self.panel.set_range(start, end)

    def _fit_range_to_view(self) -> None:
        """解析する範囲を、入力チャンネルのパネルの表示中の範囲に合わせる。"""
        data = self.panel.input_data
        if data is None:
            return
        start, end = self._panel_showing(data.channel.id).x_range()
        self.panel.set_range(start, end)
        self.workspace.show_range_selector(time_domain_key(data.source), start, end)

    def _sync_range_selector(self, spec: AnalysisSpec, data: ChannelData) -> None:
        """欄に打ち込んだ範囲を、グラフの上の範囲の選択に反映する。

        Args:
            spec: 実行する解析の条件。
            data: 入力チャンネルのデータ。
        """
        alignment = spec.alignment
        if alignment is None or alignment.start_s is None or alignment.end_s is None:
            return
        domain = time_domain_key(data.source)
        self.workspace.show_range_selector(domain, alignment.start_s, alignment.end_s)

    # --- 保存・削除 ----------------------------------------------------------

    def _save(self, spec: AnalysisSpec, *, is_new: bool) -> bool:
        """計算できた処理の条件と結果を自動で保存し、一覧の更新を知らせる。

        スライダーを動かすたびに呼ばれるので、一覧の作り直しは処理が増えたときと出力の構成が
        変わったときだけにする（作り直すと一覧の選択やスクロールが動いてしまうため）。

        Args:
            spec: 保存する処理。
            is_new: 新しい処理なら True。

        Returns:
            保存できたら True（失敗の理由は解析パネルとメッセージで知らせる）。
        """
        if self._store is None:
            return False
        previous = next((saved for saved in self._analyses if saved.id == spec.id), None)
        try:
            self._store.save_analysis(spec)
            fingerprint = analysis_fingerprint(spec)
            result = self._memo.get((spec.id, fingerprint))
            channel = self._channel(spec.input_channel_id)
            if result is not None and channel is not None and result.series:
                frame = result_to_frame(result, output_infos(spec, channel))
                self._store.save_derived(spec.id, fingerprint, frame)
        except (StorageError, OSError) as exc:
            self.panel.show_error(f"保存できません: {exc}")
            self.message.emit(f"処理「{spec.name}」を保存できません: {exc}", "error")
            return False
        if previous is None:
            self._analyses.append(spec)
        else:
            self._analyses[self._analyses.index(previous)] = spec
        same_outputs = previous is not None and _output_ids(
            self.outputs_of(previous)
        ) == _output_ids(self.outputs_of(spec))
        if is_new or not same_outputs:
            self.analyses_changed.emit()
        else:
            self.analysis_updated.emit(spec)
        return True

    def delete(self, analysis_id: UUID) -> None:
        """保存した解析と計算結果のキャッシュを削除し、表示中の出力も取り除く。

        Args:
            analysis_id: 削除する解析の ID。
        """
        if self._store is None:
            return
        spec = next((s for s in self._analyses if s.id == analysis_id), None)
        if spec is not None:
            for info in self.outputs_of(spec):
                self.workspace.hide_series(info.series_id)
        self._store.delete_analysis(analysis_id)
        self._memo = {key: value for key, value in self._memo.items() if key[0] != analysis_id}
        if self._last_run is not None and self._last_run[0].id == analysis_id:
            self._last_run = None  # 削除した解析の出力を、ID から読み込めないようにする
        if self._marked_analysis == analysis_id:
            self._marked_analysis = None
            self.workspace.clear_time_markers()
        self.result_removed.emit(analysis_id)
        self.reload_analyses()
        self.analyses_changed.emit()
        if self.panel.is_editing and self.panel.analysis_id == analysis_id:
            self.panel.show_methods()
        if spec is not None:
            self.message.emit(f"処理「{spec.name}」を削除しました", "info")

    # --- 系列の読み込み ------------------------------------------------------

    def load_series(self, series_id: UUID) -> PlotSeries | None:
        """系列 ID（チャンネル ID または解析の出力 ID）から表示用の系列を作る。

        解析の出力は、直近の計算結果 → キャッシュ → 再計算 の順に探す。

        Args:
            series_id: 系列 ID。

        Returns:
            表示用の系列。見つからない・計算できない場合は None（理由はメッセージで知らせる）。
        """
        if self._store is None:
            return None
        data = self.load_channel_data(series_id)
        if data is not None:
            return series_from_channel(data)
        if self._last_run is not None:
            for series in self._last_run[1].series:
                if series.series_id == series_id:
                    return series
        for spec in self._analyses:
            infos = self.outputs_of(spec)
            if any(info.series_id == series_id for info in infos):
                result = self._result_for(spec)
                if result is None:
                    return None
                return next((s for s in result.series if s.series_id == series_id), None)
        return None

    def load_channel_data(self, channel_id: UUID) -> ChannelData | None:
        """チャンネルの時刻と値を正規化データから読み込む。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            チャンネルのデータ。チャンネルでない・読めない場合は None。
        """
        if self._store is None:
            return None
        channel = self._channel(channel_id)
        if channel is None:
            return None
        try:
            source = self._store.database.get(Source, channel.source_id)
            return self._store.read_channel_data(source, channel)
        except (EntityNotFoundError, StorageError) as exc:
            self.message.emit(str(exc), "error")
            return None

    def _result_for(self, spec: AnalysisSpec) -> AnalysisResult | None:
        """保存した解析の計算結果を得る（覚えている結果 → キャッシュ → 再計算）。

        再計算は表示の復元のために同期的に行う（キャッシュがあれば計算しない）。

        Args:
            spec: 保存した解析。

        Returns:
            計算結果。計算できない場合は None（理由はメッセージで知らせる）。
        """
        store = self._store
        if store is None:
            return None
        fingerprint = analysis_fingerprint(spec)
        memo = self._memo.get((spec.id, fingerprint))
        if memo is not None:
            return memo
        channel = self._channel(spec.input_channel_id)
        if channel is None:
            self.message.emit(f"解析「{spec.name}」の入力チャンネルが見つかりません", "warning")
            return None
        try:
            source = store.database.get(Source, channel.source_id)
        except EntityNotFoundError:
            return None
        frame = store.load_derived(spec.id, fingerprint)
        result = result_from_frame(spec, channel, source, frame) if frame is not None else None
        if result is None:
            result = self._compute_now(spec, channel, source, fingerprint)
        if result is not None:
            self._memo[(spec.id, fingerprint)] = result
        return result

    def _compute_now(
        self, spec: AnalysisSpec, channel: Channel, source: Source, fingerprint: str
    ) -> AnalysisResult | None:
        """キャッシュのない保存済みの解析を、その場で計算してキャッシュに保存する。

        Args:
            spec: 保存した解析。
            channel: 入力チャンネル。
            source: 入力チャンネルの Source。
            fingerprint: 解析の指紋。

        Returns:
            計算結果。計算できない場合は None。
        """
        store = self._store
        if store is None:
            return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data = store.read_channel_data(source, channel)
            result = run_analysis(spec, data)
            store.save_derived(
                spec.id, fingerprint, result_to_frame(result, output_infos(spec, channel))
            )
        except (AnalysisInputError, StorageError, ValueError, OSError) as exc:
            self.message.emit(f"解析「{spec.name}」を計算できません: {exc}", "warning")
            return None
        finally:
            QApplication.restoreOverrideCursor()
        return result

    def _source_of(self, channel_id: UUID) -> Source | None:
        """チャンネルの Source を返す。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            Source。見つからなければ None。
        """
        channel = self._channel(channel_id)
        if self._store is None or channel is None:
            return None
        try:
            return self._store.database.get(Source, channel.source_id)
        except EntityNotFoundError:
            return None

    def _channel(self, channel_id: UUID) -> Channel | None:
        """チャンネルを ID で探す。

        Args:
            channel_id: チャンネルの ID。

        Returns:
            チャンネル。なければ None。
        """
        if self._store is None:
            return None
        try:
            return self._store.database.get(Channel, channel_id)
        except EntityNotFoundError:
            return None


def _output_ids(outputs: list[AnalysisOutputInfo]) -> list[UUID]:
    """解析の出力の系列 ID を返す（出力の構成が変わったかどうかの比較に使う）。

    Args:
        outputs: 解析の出力。

    Returns:
        系列 ID のリスト。
    """
    return [output.series_id for output in outputs]


def _default_range(panel: PlotPanel) -> tuple[float, float]:
    """範囲の選択の初期値（パネルに見えているデータの中央の 4 割）を返す。

    Args:
        panel: 入力チャンネルを表示しているパネル。

    Returns:
        (開始, 終了) [s]。
    """
    view_low, view_high = panel.x_range()
    data_low, data_high = panel.data_range() or (view_low, view_high)
    low, high = max(view_low, data_low), min(view_high, data_high)
    if low >= high:  # 表示範囲がまだデータに合っていないとき
        low, high = data_low, data_high
    margin = _DEFAULT_RANGE_MARGIN * (high - low)
    return low + margin, high - margin
