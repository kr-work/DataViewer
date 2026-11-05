"""保存できる解析（AnalysisSpec）の実行と、結果の表示用データ（PlotSeries）への変換（SPEC F-11）。

GUI はこのモジュールを通して解析を実行する（解析の種類ごとの分岐をここに集める）。
    - ``output_infos()``: 計算しなくても分かる出力の一覧（ID・名前・単位）。
      ツリーや表示の復元に使う。
    - ``run_analysis()``: 解析を実行し、出力の系列と要約を返す。2 チャンネルの解析
      （ステップ応答・ヒステリシス・相互相関）は時系列を出力せず、指標・グラフ・時刻の印を返す。
    - ``result_to_frame()`` / ``result_from_frame()``: 結果とキャッシュ（derived/）の変換。
    - ``default_*()``: GUI の初期値（データから推定した観測雑音など）。利用者が確認して変える前提。

出力の ID は、解析の ID と出力名から決める（UUID5）。再計算・再起動しても同じ ID になるので、
表示設定（views/*.json）から解析の出力を参照できる。
このモジュールは Qt に依存しない。
"""

import hashlib
import math
from uuid import UUID, uuid5

import numpy as np
import polars as pl

from experiment_workbench.analysis.alignment import ChannelAligner
from experiment_workbench.analysis.input_output import (
    CrossCorrelationAnalyzer,
    HysteresisAnalyzer,
    StepResponseAnalyzer,
)
from experiment_workbench.analysis.kalman import (
    KalmanFilter,
    ProgressCallback,
    kinematic_process_noise,
    kinematic_transition_matrix,
    system_from_settings,
)
from experiment_workbench.analysis.signal_processing import (
    AnalysisInputError,
    ButterworthFilter,
    Differentiator,
    Integrator,
    MovingAverageFilter,
    SavitzkyGolayFilter,
)
from experiment_workbench.model import (
    DATASET_TIME_COLUMN,
    TWO_CHANNEL_ANALYSIS_KINDS,
    AlignedPair,
    AlignmentRecord,
    AnalysisChart,
    AnalysisKind,
    AnalysisMetric,
    AnalysisOutputInfo,
    AnalysisResult,
    AnalysisSpec,
    Channel,
    ChannelData,
    ChartAnnotation,
    ChartSeries,
    ChartStyle,
    CrossCorrelationResult,
    HysteresisResult,
    InterpolationMethod,
    KalmanInitialState,
    KalmanModelKind,
    KalmanSettings,
    KinematicSystemParams,
    LinearSystemParams,
    PlotSeries,
    PolynomialObservationParams,
    Source,
    StepResponseResult,
    TimeMarker,
)
from experiment_workbench.timebase import are_time_comparable
from experiment_workbench.validation import FloatArray

DEFAULT_BANDWIDTH_DIVISOR = 20.0
"""カルマンフィルタの初期値で、追従の速さ（帯域）をサンプリング周波数の 1/20 にする。"""

DEFAULT_SIGNIFICANT_DIGITS = 4
"""GUI の初期値（雑音の強さ・初期共分散など）を丸める有効数字の桁数。

初期値は推定に基づく目安で、利用者が GUI で見て調整する出発点なので、読みやすく打ち直しやすい
桁数に丸める（丸めによる変化は 0.05 % 以下で、推定の目安としての意味は変わらない）。
"""

MAX_CHART_POINTS = 20_000
"""解析結果のグラフの 1 系列に描く点の数の上限（表示だけを間引く。指標は全サンプルで計算する）。"""

_INCREASING_COLOR = "#d62728"
_DECREASING_COLOR = "#1f77b4"
_INCREASING_BRANCH_COLOR = "#7f1d1d"
_DECREASING_BRANCH_COLOR = "#0b3d6b"
_EXCLUDED_COLOR = "#b8b8b8"
_REFERENCE_COLOR = "#555555"
_INPUT_COLOR = "#8a8a8a"

_FILTER_KINDS = (
    AnalysisKind.BUTTERWORTH,
    AnalysisKind.MOVING_AVERAGE,
    AnalysisKind.SAVITZKY_GOLAY,
)
"""元データから一部の変動を取り除く処理（取り除いた成分を出力できるもの）。"""

_MILLISECOND_LIMIT_S = 2.0
"""グラフの時間軸を ms で表す上限 [s]（これより長い範囲は s で表す）。"""

_LOWER_SUFFIX = "__lower"
_UPPER_SUFFIX = "__upper"
_MARKER_SUFFIX = "__marker"


# ---------------------------------------------------------------------------
# 生データの系列・出力の情報
# ---------------------------------------------------------------------------


def series_from_channel(data: ChannelData) -> PlotSeries:
    """生データのチャンネルを、表示用の系列にする。

    Args:
        data: チャンネルのデータ。

    Returns:
        表示用の系列（ID はチャンネル ID）。
    """
    return PlotSeries(
        series_id=data.channel.id,
        label=f"{data.source.name} / {data.channel.name}",
        channel_name=data.channel.name,
        unit=data.channel.unit,
        source=data.source,
        time_s=data.time_s,
        values=data.values,
    )


def overlays_input(spec: AnalysisSpec) -> bool:
    """解析の主な結果を、入力チャンネルと同じ縦軸に重ねて比べられるかを返す。

    単位や大きさが違う量を重ねると、縦軸が大きい方に合わせられて元の線が読めなくなる。
    重ねられるのは、入力と同じ量を返す処理だけにする。微分・積分は単位が変わる。EKF の
    多項式の観測で観測の式が恒等でないとき、線形モデルで観測が先頭の状態そのものでないときは、
    推定する状態が入力とは別の量になる。

    Args:
        spec: 解析。

    Returns:
        重ねられるなら True。
    """
    if spec.kind in _FILTER_KINDS:
        return True
    if spec.kind is not AnalysisKind.KALMAN:
        return False
    settings = _require(spec.kalman, "kalman")
    if settings.model_kind is KalmanModelKind.KINEMATIC:
        return True
    if settings.model_kind is KalmanModelKind.POLYNOMIAL_OBSERVATION:
        polynomial = _require(settings.polynomial_observation, "polynomial_observation")
        return polynomial.coefficients[:2] == [0.0, 1.0] and not any(polynomial.coefficients[2:])
    observation = _require(settings.linear, "linear").observation[0]
    return observation[0] == 1.0 and not any(observation[1:])


def output_series_id(analysis_id: UUID, key: str) -> UUID:
    """解析の出力の ID を、解析の ID と出力名から決める（何度計算しても同じ値）。

    Args:
        analysis_id: 解析の ID。
        key: 出力名（例: "output"、"state_0"）。

    Returns:
        出力の ID。
    """
    return uuid5(analysis_id, key)


def output_infos(spec: AnalysisSpec, channel: Channel) -> list[AnalysisOutputInfo]:
    """解析の出力の一覧を返す（計算はしない）。

    微分・積分の出力は 1 つ（"output"）。フィルタの出力は、処理結果（"output"）と、元データから
    取り除いた成分（"removed"）。結果の線だけでは、何が取り除かれたかが見えないため。
    カルマンフィルタ系の出力は、状態ごとの推定値（"state_0"、"state_1" ...）、イノベーション
    （"innovation"）、観測雑音の標準偏差（"noise_std"）。先頭の出力が主な結果になる。

    Args:
        spec: 解析。
        channel: 入力チャンネル（単位の決定に使う）。

    Returns:
        出力の情報（run_analysis の結果の系列と同じ順）。
    """
    return [
        AnalysisOutputInfo(
            series_id=output_series_id(spec.id, key), key=key, label=label, unit=unit
        )
        for key, label, unit in _outputs(spec, channel.unit)
    ]


def _outputs(spec: AnalysisSpec, unit: str | None) -> list[tuple[str, str, str | None]]:
    """解析の種類ごとに、出力の (名前, 表示名, 単位) を決める。

    Args:
        spec: 解析。
        unit: 入力チャンネルの単位。

    Returns:
        (出力名, 表示名, 単位) のリスト。
    """
    if spec.kind in TWO_CHANNEL_ANALYSIS_KINDS:
        return []  # 時系列は出力しない（指標・グラフ・時刻の印を結果に入れる）
    if spec.kind is AnalysisKind.DERIVATIVE:
        order = spec.derivative.order if spec.derivative is not None else 1
        return [("output", spec.name, _per_second(unit, order))]
    if spec.kind is AnalysisKind.INTEGRAL:
        return [("output", spec.name, _times_second(unit))]
    if spec.kind in _FILTER_KINDS:
        return [("output", spec.name, unit), ("removed", f"{spec.name}: 取り除いた成分", unit)]
    settings = _require(spec.kalman, "kalman")
    state_labels, state_units = _state_descriptions(settings, unit)
    outputs = [
        (f"state_{index}", f"{spec.name}: {label}", state_unit)
        for index, (label, state_unit) in enumerate(zip(state_labels, state_units, strict=True))
    ]
    outputs.append(("innovation", f"{spec.name}: イノベーション", unit))
    outputs.append(("noise_std", f"{spec.name}: √R", unit))
    return outputs


def _state_descriptions(
    settings: KalmanSettings, unit: str | None
) -> tuple[list[str], list[str | None]]:
    """カルマンフィルタの状態ごとの表示名と単位を返す。

    運動学モデルの状態は [値, 1 階微分, ...] なので、入力の単位から決められる。
    多項式の観測モデルの状態 x は観測と単位が違い（電圧から推定した張力など）、
    線形モデルの状態の意味は行列による。どちらも単位は分からないので None にする。

    Args:
        settings: カルマンフィルタの条件。
        unit: 入力チャンネル（観測）の単位。

    Returns:
        (表示名のリスト, 単位のリスト)。
    """
    derivative_names = ["推定値", "1 階微分", "2 階微分"]
    if settings.model_kind is KalmanModelKind.KINEMATIC:
        params = _require(settings.kinematic, "kinematic")
        count = params.derivative_order + 1
        return derivative_names[:count], [_per_second(unit, order) for order in range(count)]
    if settings.model_kind is KalmanModelKind.POLYNOMIAL_OBSERVATION:
        polynomial = _require(settings.polynomial_observation, "polynomial_observation")
        count = polynomial.derivative_order + 1
        return [f"x の{name}" for name in derivative_names[:count]], [None] * count
    linear = _require(settings.linear, "linear")
    count = len(linear.transition)
    return [f"状態 {index + 1}" for index in range(count)], [None] * count


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run_analysis(
    spec: AnalysisSpec,
    data: ChannelData,
    progress: ProgressCallback | None = None,
    *,
    output_data: ChannelData | None = None,
) -> AnalysisResult:
    """解析を実行して、結果を返す。

    Args:
        spec: 解析。
        data: 入力チャンネルのデータ（``spec.input_channel_id`` のチャンネル）。
        progress: 進捗を受け取り、続けるなら True を返す関数（カルマンフィルタ系で使う）。
        output_data: 出力チャンネルのデータ（2 チャンネルの解析だけ。
            ``spec.output_channel_id`` のチャンネル）。

    Returns:
        解析の結果。

    Raises:
        AnalysisInputError: チャンネルが違う、または入力が解析の前提を満たさない場合。
        AnalysisCancelledError: ``progress`` が False を返して中止した場合。
    """
    if data.channel.id != spec.input_channel_id:
        raise AnalysisInputError("解析の入力チャンネルと、渡されたデータのチャンネルが違います")
    if spec.kind in TWO_CHANNEL_ANALYSIS_KINDS:
        return _run_two_channel(spec, data, output_data)
    infos = output_infos(spec, data.channel)
    if spec.kind is AnalysisKind.KALMAN:
        return _run_kalman(spec, data, infos, progress)
    values = _run_signal_processing(spec, data)
    series = [_series(infos[0], spec, data, values)]
    if spec.kind in _FILTER_KINDS:
        series.append(_series(infos[1], spec, data, data.values - values))
    return AnalysisResult(
        analysis_id=spec.id, series=series, headline=_signal_headline(spec, data, values, infos)
    )


def _signal_headline(
    spec: AnalysisSpec, data: ChannelData, values: FloatArray, infos: list[AnalysisOutputInfo]
) -> str:
    """フィルタ・微分・積分の結果を、数値入りの一文にする。

    Args:
        spec: 解析。
        data: 入力チャンネルのデータ。
        values: 処理後の値。
        infos: 出力の情報。

    Returns:
        結果の一文。求められる値がなければ空文字。
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    unit = _unit_text(infos[0].unit)
    if spec.kind is AnalysisKind.DERIVATIVE:
        return f"変化の速さは最大で {float(np.max(np.abs(finite))):.3g}{unit} でした。"
    if spec.kind is AnalysisKind.INTEGRAL:
        return f"積み上げた値は、最後に {float(finite[-1]):.3g}{unit} になりました。"
    removed = data.values - values
    rms = float(np.sqrt(np.nanmean(removed**2)))
    spread = float(np.nanstd(data.values))
    share = f"（元の変動の {100.0 * rms / spread:.3g} %）" if spread > 0 else ""
    return f"元データから大きさ {rms:.3g}{unit} の変動を取り除きました{share}。"


def _unit_text(unit: str | None) -> str:
    """文の中で値の後ろに続ける単位を返す。

    Args:
        unit: 単位。

    Returns:
        前に空白を付けた単位。単位がなければ空文字。
    """
    return f" {unit}" if unit else ""


def _time_text(seconds: float) -> str:
    """時間を読みやすい単位の文字列にする（1 秒未満は ms）。

    Args:
        seconds: 時間 [s]。

    Returns:
        文字列（例: "9.37 ms"、"1.2 s"）。
    """
    if abs(seconds) < 1.0:
        return f"{seconds * 1000.0:.3g} ms"
    return f"{seconds:.3g} s"


def _run_signal_processing(spec: AnalysisSpec, data: ChannelData) -> FloatArray:
    """フィルタ・微分・積分を実行する。

    Args:
        spec: 解析（カルマンフィルタ以外）。
        data: 入力チャンネルのデータ。

    Returns:
        処理後の値。
    """
    t, x = data.time_s, data.values
    if spec.kind is AnalysisKind.BUTTERWORTH:
        return ButterworthFilter(_require(spec.butterworth, "butterworth")).apply(t, x)
    if spec.kind is AnalysisKind.MOVING_AVERAGE:
        return MovingAverageFilter(_require(spec.moving_average, "moving_average")).apply(t, x)
    if spec.kind is AnalysisKind.SAVITZKY_GOLAY:
        return SavitzkyGolayFilter(_require(spec.savitzky_golay, "savitzky_golay")).apply(t, x)
    if spec.kind is AnalysisKind.DERIVATIVE:
        return Differentiator(_require(spec.derivative, "derivative")).apply(t, x)
    return Integrator(_require(spec.integral, "integral")).apply(t, x)


def _run_kalman(
    spec: AnalysisSpec,
    data: ChannelData,
    infos: list[AnalysisOutputInfo],
    progress: ProgressCallback | None,
) -> AnalysisResult:
    """カルマンフィルタ系を実行し、状態・イノベーション・観測雑音の系列にする。

    状態の推定値には ±σ（共分散の対角成分の平方根）の帯、イノベーションには ±√S の帯を付ける。
    観測が欠測（NaN）で更新しなかった時刻には印を付ける。

    Args:
        spec: 解析（カルマンフィルタ系）。
        data: 入力チャンネルのデータ。
        infos: 出力の情報（output_infos の結果）。
        progress: 進捗を受け取る関数。

    Returns:
        解析の結果。
    """
    settings = _require(spec.kalman, "kalman")
    system = system_from_settings(settings)
    result = KalmanFilter(system, settings.initial, adaptive=settings.adaptive).run(
        data.time_s, data.values, progress
    )
    skipped = ~result.updated
    markers = result.time_s[skipped] if skipped.any() else None
    by_key = {info.key: info for info in infos}
    series: list[PlotSeries] = []
    for index in range(result.states.shape[1]):
        estimate = result.states[:, index]
        sigma = np.sqrt(np.maximum(result.covariances[:, index, index], 0.0))
        series.append(_series(by_key[f"state_{index}"], spec, data, estimate, sigma, markers))
    innovation_sigma = np.sqrt(np.maximum(result.innovation_covariances[:, 0, 0], 0.0))
    series.append(
        _series(by_key["innovation"], spec, data, result.innovations[:, 0], innovation_sigma)
    )
    noise_std = np.sqrt(result.measurement_noise[:, 0, 0])
    series.append(_series(by_key["noise_std"], spec, data, noise_std))

    summary = [f"観測更新しなかった時刻: {int(skipped.sum())} 個"] if skipped.any() else []
    if settings.adaptive is not None:
        summary.append(f"推定した観測雑音の標準偏差 √R の最終値: {float(noise_std[-1]):.6g}")
        summary.append(f"R の正定値化を行った時刻: {int(result.noise_floor_applied.sum())} 個")
    unit = _unit_text(data.channel.unit)
    sigma = np.sqrt(np.maximum(result.covariances[:, 0, 0], 0.0))
    innovation_rms = float(np.sqrt(np.nanmean(result.innovations[:, 0] ** 2)))
    headline = (
        f"推定の不確かさは平均 ±{float(np.mean(sigma)):.3g}、"
        f"観測との差は平均して {innovation_rms:.3g}{unit} でした。"
    )
    return AnalysisResult(analysis_id=spec.id, series=series, headline=headline, summary=summary)


def _series(
    info: AnalysisOutputInfo,
    spec: AnalysisSpec,
    data: ChannelData,
    values: FloatArray,
    sigma: FloatArray | None = None,
    markers: FloatArray | None = None,
) -> PlotSeries:
    """出力の値から表示用の系列を作る（帯は値 ± sigma）。

    Args:
        info: 出力の情報。
        spec: 解析。
        data: 入力チャンネルのデータ（時刻と時間軸の Source に使う）。
        values: 出力の値。
        sigma: 帯の半幅。None なら帯なし。
        markers: 印を付ける時刻。None なら印なし。

    Returns:
        表示用の系列。
    """
    return PlotSeries(
        series_id=info.series_id,
        label=info.label,
        channel_name=data.channel.name,
        unit=info.unit,
        source=data.source,
        time_s=data.time_s,
        values=np.asarray(values, dtype=np.float64),
        band_lower=None if sigma is None else np.asarray(values - sigma, dtype=np.float64),
        band_upper=None if sigma is None else np.asarray(values + sigma, dtype=np.float64),
        marker_times_s=markers,
        analysis_id=spec.id,
    )


# ---------------------------------------------------------------------------
# 2 チャンネルの解析（ステップ応答・ヒステリシス・相互相関）
# ---------------------------------------------------------------------------


def _run_two_channel(
    spec: AnalysisSpec, data: ChannelData, output_data: ChannelData | None
) -> AnalysisResult:
    """入力と出力を同じ時刻に対応付けて解析し、指標・グラフ・時刻の印にする。

    Args:
        spec: 解析（2 チャンネルの種類）。
        data: 入力チャンネルのデータ。
        output_data: 出力チャンネルのデータ。

    Returns:
        解析の結果（時系列の出力はない）。

    Raises:
        AnalysisInputError: 出力チャンネルのデータがない、時刻の対応が明示されていない、
            または入力が解析の前提を満たさない場合。
    """
    if output_data is None or output_data.channel.id != spec.output_channel_id:
        raise AnalysisInputError("解析の出力チャンネルのデータが渡されていません")
    pair = ChannelAligner(_require(spec.alignment, "alignment")).align(
        data.time_s,
        data.values,
        output_data.time_s,
        output_data.values,
        time_comparable=are_time_comparable(data.source, output_data.source),
    )
    if spec.kind is AnalysisKind.STEP_RESPONSE:
        params = _require(spec.step_response, "step_response")
        step = StepResponseAnalyzer(params).analyze(pair)
        return _step_response_result(spec, data, output_data, pair, step, params.settling_band)
    if spec.kind is AnalysisKind.HYSTERESIS:
        loop = HysteresisAnalyzer(_require(spec.hysteresis, "hysteresis")).analyze(pair)
        return _hysteresis_result(spec, data, output_data, pair, loop)
    correlation_params = _require(spec.cross_correlation, "cross_correlation")
    correlation = CrossCorrelationAnalyzer(correlation_params).analyze(pair)
    return _cross_correlation_result(spec, pair, correlation)


def _step_response_result(
    spec: AnalysisSpec,
    data: ChannelData,
    output_data: ChannelData,
    pair: AlignedPair,
    step: StepResponseResult,
    settling_band: float,
) -> AnalysisResult:
    """ステップ応答の指標を、表・グラフ・時刻の印にする。

    Args:
        spec: 解析。
        data: 入力チャンネルのデータ。
        output_data: 出力チャンネルのデータ。
        pair: 対応付けた入力と出力。
        step: ステップ応答の指標。
        settling_band: 整定の幅（出力の変化量に対する割合）。

    Returns:
        解析の結果。
    """
    input_unit, output_unit = data.channel.unit, output_data.channel.unit
    metrics = [
        _metric(
            "response_delay_s",
            "動き始めるまでの遅れ",
            step.response_delay_s,
            "s",
            "入力の変化の開始から、出力が同じ割合だけ変化するまで",
        ),
        _metric(
            "delay_time_s",
            "半分に達するまで Td",
            step.delay_time_s,
            "s",
            "出力が変化量の 50% に達するまで",
        ),
        _metric(
            "rise_time_s",
            "立ち上がり時間 Tr",
            step.rise_time_s,
            "s",
            "出力が変化量の 10% から 90% になるまで",
        ),
        _metric(
            "time_constant_s",
            "時定数 τ",
            step.time_constant_s,
            "s",
            "一次遅れ＋むだ時間で近似（2 点法）",
        ),
        _metric(
            "dead_time_s", "むだ時間 L", step.dead_time_s, "s", "一次遅れ＋むだ時間で近似（2 点法）"
        ),
        _metric(
            "settling_time_s",
            "落ち着くまで Ts",
            step.settling_time_s,
            "s",
            f"出力が最終値の ±{settling_band * 100:.3g}% に入ったまま出なくなるまで",
        ),
        _metric(
            "overshoot_percent",
            "行き過ぎ（オーバーシュート）",
            _percent(step.overshoot),
            "%",
            "出力の変化量に対する割合",
        ),
        _metric(
            "gain",
            "感度（ゲイン）",
            step.gain,
            _ratio(output_unit, input_unit),
            "出力の変化量 ÷ 入力の変化量",
        ),
        _metric("input_onset_s", "入力の変化の開始", step.input_onset_s, "s", "入力の時間軸の時刻"),
        _metric("input_initial", "入力の初期値", step.input_initial, input_unit),
        _metric("input_final", "入力の最終値", step.input_final, input_unit),
        _metric("output_initial", "出力の初期値", step.output_initial, output_unit),
        _metric("output_final", "出力の最終値", step.output_final, output_unit),
    ]
    return AnalysisResult(
        analysis_id=spec.id,
        series=[],
        headline=_step_headline(step),
        summary=[*_alignment_notes(pair.record), *step.notes],
        conditions=_alignment_conditions(pair.record),
        metrics=metrics,
        charts=[_step_chart(step)],
        time_markers=_step_markers(step),
    )


def _step_headline(step: StepResponseResult) -> str:
    """ステップ応答の結果を、数値入りの一文にする。

    Args:
        step: ステップ応答の指標。

    Returns:
        結果の一文（求められなかった指標は文に入れない）。
    """
    parts = []
    if step.response_delay_s is not None:
        parts.append(f"入力の変化から {_time_text(step.response_delay_s)} 遅れて動き始め")
    if step.delay_time_s is not None:
        parts.append(f"{_time_text(step.delay_time_s)} で変化の半分に達し")
    if step.settling_time_s is not None:
        parts.append(f"{_time_text(step.settling_time_s)} で落ち着きました")
    else:
        parts.append("範囲の中では落ち着きませんでした")
    return "出力は" + "、".join(parts) + "。"


def _step_markers(step: StepResponseResult) -> list[TimeMarker]:
    """縦並べ表示に引く時刻の印を作る。

    時間が近い印の名前は重なって読めないので、番号だけを出し、意味は結果のグラフの注記で示す。

    Args:
        step: ステップ応答の指標。

    Returns:
        時刻の印。
    """
    markers = [TimeMarker(time_s=step.input_onset_s, label="①")]
    for offset, label in (
        (step.response_delay_s, "②"),
        (step.delay_time_s, "③"),
        (step.settling_time_s, "④"),
    ):
        if offset is not None:
            markers.append(TimeMarker(time_s=step.input_onset_s + offset, label=label))
    return markers


def _step_chart(step: StepResponseResult) -> AnalysisChart:
    """入力の変化を起点に、出力が追いつく様子のグラフを作る。

    変化前を 0 %、変化後を 100 % に揃えて、入力と出力を重ねる。

    Args:
        step: ステップ応答の指標。

    Returns:
        グラフ。
    """
    x, normalized_input = _thin(step.response_time_s, step.normalized_input)
    _, normalized_output = _thin(step.response_time_s, step.normalized_output)
    span = float(max(abs(x[0]), abs(x[-1])))
    scale, time_unit = (1000.0, "ms") if span < _MILLISECOND_LIMIT_S else (1.0, "s")
    ends = np.array([float(x[0]), float(x[-1])]) * scale
    series = [
        ChartSeries(label="入力", x=x * scale, y=normalized_input * 100.0, color=_INPUT_COLOR),
        ChartSeries(
            label="出力", x=x * scale, y=normalized_output * 100.0, color=_DECREASING_COLOR
        ),
        ChartSeries(
            label="50 %",
            x=ends,
            y=np.array([50.0, 50.0]),
            style=ChartStyle.DASHED,
            color=_REFERENCE_COLOR,
            show_in_legend=False,
        ),
    ]
    annotations = [ChartAnnotation(x=0.0, text="① 入力が変化")]
    for offset, label in (
        (step.response_delay_s, "② 動き始め"),
        (step.delay_time_s, "③ 半分"),
        (step.settling_time_s, "④ 落ち着く"),
    ):
        if offset is not None:
            annotations.append(
                ChartAnnotation(x=offset * scale, text=f"{label} {_time_text(offset)}")
            )
    return AnalysisChart(
        title="入力が変わってから、出力が追いつくまで",
        x_label=f"入力が変わってからの時間 [{time_unit}]",
        y_label="変化の割合 [%]",
        series=series,
        annotations=annotations,
    )


def _hysteresis_result(
    spec: AnalysisSpec,
    data: ChannelData,
    output_data: ChannelData,
    pair: AlignedPair,
    loop: HysteresisResult,
) -> AnalysisResult:
    """入出力特性とヒステリシスの指標を、表と X-Y のグラフにする。

    Args:
        spec: 解析。
        data: 入力チャンネルのデータ。
        output_data: 出力チャンネルのデータ。
        pair: 対応付けた入力と出力。
        loop: 入出力特性とヒステリシスの指標。

    Returns:
        解析の結果。
    """
    input_unit, output_unit = data.channel.unit, output_data.channel.unit
    metrics = [
        _metric("slope", "感度（回帰直線の傾き）", loop.slope, _ratio(output_unit, input_unit)),
        _metric("intercept", "オフセット（回帰直線の切片）", loop.intercept, output_unit),
        _metric("r_squared", "決定係数 R²", loop.r_squared),
        _metric(
            "full_scale_output",
            "フルスケール出力 FS",
            loop.full_scale_output,
            output_unit,
            "回帰直線の上で、入力が最小から最大まで動いたときの出力の変化",
        ),
        _metric(
            "max_hysteresis",
            "最大ヒステリシス",
            loop.max_hysteresis,
            output_unit,
            "同じ入力での、増加時と減少時の出力の差の最大",
        ),
        _metric(
            "hysteresis_percent", "最大ヒステリシス（FS 比）", _percent(loop.hysteresis_ratio), "%"
        ),
        _metric(
            "max_hysteresis_input",
            "最大ヒステリシスの位置（入力）",
            loop.max_hysteresis_input,
            input_unit,
        ),
        _metric(
            "linearity_error",
            "直線性誤差",
            loop.linearity_error,
            output_unit,
            "分岐の中央の曲線と回帰直線の差の最大",
        ),
        _metric("linearity_percent", "直線性誤差（FS 比）", _percent(loop.linearity_ratio), "%"),
        _metric(
            "loop_area",
            "ループ面積",
            loop.loop_area,
            _product(input_unit, output_unit),
            "増加時と減少時の分岐が囲む面積",
        ),
        _metric(
            "mean_branch_difference",
            "分岐の差の平均（増加時 - 減少時）",
            loop.mean_branch_difference,
            output_unit,
            "正なら増加時の分岐が上にある",
        ),
    ]
    counts = (
        f"サンプル数: 増加時 {loop.n_increasing}、減少時 {loop.n_decreasing}、"
        f"除外 {loop.n_excluded}"
    )
    return AnalysisResult(
        analysis_id=spec.id,
        series=[],
        headline=_hysteresis_headline(loop, input_unit, output_unit),
        summary=[*_alignment_notes(pair.record), *loop.notes],
        conditions=[*_alignment_conditions(pair.record), counts],
        metrics=metrics,
        charts=_hysteresis_charts(data, output_data, pair, loop),
    )


def _hysteresis_headline(
    loop: HysteresisResult, input_unit: str | None, output_unit: str | None
) -> str:
    """ヒステリシスの結果を、数値入りの一文にする。

    Args:
        loop: 入出力特性とヒステリシスの指標。
        input_unit: 入力の単位。
        output_unit: 出力の単位。

    Returns:
        結果の一文。
    """
    if loop.max_hysteresis is None:
        return (
            "増やすときと減らすときを比べられませんでした（入力が往復する範囲を選んでください）。"
        )
    share = (
        f"（出力の全幅の {_percent(loop.hysteresis_ratio):.3g} %）" if loop.hysteresis_ratio else ""
    )
    place = (
        f"入力 {loop.max_hysteresis_input:.3g}{_unit_text(input_unit)} の付近で"
        if loop.max_hysteresis_input is not None
        else ""
    )
    return (
        f"同じ入力でも、増やすときと減らすときで出力が違います。{place}最大 "
        f"{loop.max_hysteresis:.3g}{_unit_text(output_unit)} の差があります{share}。"
    )


def _hysteresis_charts(
    data: ChannelData, output_data: ChannelData, pair: AlignedPair, loop: HysteresisResult
) -> list[AnalysisChart]:
    """入力と出力の関係のグラフと、増やすときと減らすときの差のグラフを作る。

    差はループの線の間隔として読むには小さいことが多いので、差そのものも別のグラフで見せる。
    測定値の点は背景として薄く 1 色で描き、読み取ってほしい線を前に出す。

    Args:
        data: 入力チャンネルのデータ。
        output_data: 出力チャンネルのデータ。
        pair: 対応付けた入力と出力。
        loop: 入出力特性とヒステリシスの指標。

    Returns:
        グラフのリスト。
    """
    input_unit, output_unit = data.channel.unit, output_data.channel.unit
    x_label = _axis_label(f"入力 {data.channel.name}", input_unit)
    finite_pair = np.isfinite(pair.input_values) & np.isfinite(pair.output_values)
    x_points, y_points = _thin(pair.input_values[finite_pair], pair.output_values[finite_pair])
    series = [
        ChartSeries(
            label="測定値",
            x=x_points,
            y=y_points,
            style=ChartStyle.POINTS,
            color=_EXCLUDED_COLOR,
            show_in_legend=False,
        )
    ]
    for label, branch, color in (
        ("増やすとき", loop.increasing_branch, _INCREASING_COLOR),
        ("減らすとき", loop.decreasing_branch, _DECREASING_COLOR),
    ):
        finite = np.isfinite(branch)
        if finite.any():
            series.append(
                ChartSeries(label=label, x=loop.bin_centers[finite], y=branch[finite], color=color)
            )
    ends = np.array([loop.input_min, loop.input_max])
    series.append(
        ChartSeries(
            label="直線で近似",
            x=ends,
            y=loop.slope * ends + loop.intercept,
            style=ChartStyle.DASHED,
            color=_REFERENCE_COLOR,
        )
    )
    annotations = []
    if loop.max_hysteresis_input is not None and loop.max_hysteresis is not None:
        text = f"差が最大 {loop.max_hysteresis:.3g}{_unit_text(output_unit)}"
        annotations.append(ChartAnnotation(x=loop.max_hysteresis_input, text=text))
    charts = [
        AnalysisChart(
            title="入力と出力の関係（増やすとき・減らすとき）",
            x_label=x_label,
            y_label=_axis_label(f"出力 {output_data.channel.name}", output_unit),
            series=series,
            annotations=annotations,
        )
    ]
    difference = loop.increasing_branch - loop.decreasing_branch
    both = np.isfinite(difference)
    if both.any():
        charts.append(
            AnalysisChart(
                title="増やすときと減らすときの差",
                x_label=x_label,
                y_label=_axis_label("出力の差", output_unit),
                series=[
                    ChartSeries(
                        label="増やすとき − 減らすとき",
                        x=loop.bin_centers[both],
                        y=difference[both],
                        color=_DECREASING_COLOR,
                    )
                ],
                annotations=annotations,
            )
        )
    return charts


def _cross_correlation_result(
    spec: AnalysisSpec, pair: AlignedPair, correlation: CrossCorrelationResult
) -> AnalysisResult:
    """相互相関の結果を、表と相関の曲線のグラフにする。

    Args:
        spec: 解析。
        pair: 対応付けた入力と出力。
        correlation: 相互相関の結果。

    Returns:
        解析の結果。
    """
    metrics = [
        _metric("delay_s", "遅れ", correlation.delay_s, "s", "正なら出力が入力より遅れている"),
        _metric(
            "peak_coefficient",
            "相関係数のピーク",
            correlation.peak_coefficient,
            None,
            "負なら出力は入力と逆向きに変化している",
        ),
        _metric("sample_interval_s", "サンプル間隔", correlation.sample_interval_s, "s"),
    ]
    lag_span = float(np.max(np.abs(correlation.lags_s)))
    scale, time_unit = (1000.0, "ms") if lag_span < _MILLISECOND_LIMIT_S else (1.0, "s")
    chart = AnalysisChart(
        title="どれだけずらすと、入力と出力の波形が重なるか",
        x_label=f"ずらす時間 [{time_unit}]（正: 出力が遅れている）",
        y_label="波形の一致度（相関係数）",
        series=[
            ChartSeries(
                label="一致度",
                x=correlation.lags_s * scale,
                y=correlation.coefficients,
                color=_DECREASING_COLOR,
                show_in_legend=False,
            ),
            ChartSeries(
                label="最も重なる位置",
                x=np.array([correlation.delay_s * scale]),
                y=np.array([correlation.peak_coefficient]),
                style=ChartStyle.MARKERS,
                color=_INCREASING_COLOR,
                show_in_legend=False,
            ),
        ],
        annotations=[
            ChartAnnotation(
                x=correlation.delay_s * scale,
                text=f"最も重なる {_time_text(correlation.delay_s)}",
            )
        ],
    )
    return AnalysisResult(
        analysis_id=spec.id,
        series=[],
        headline=_correlation_headline(correlation),
        summary=[*_alignment_notes(pair.record), *correlation.notes],
        conditions=_alignment_conditions(pair.record),
        metrics=metrics,
        charts=[chart],
    )


def _correlation_headline(correlation: CrossCorrelationResult) -> str:
    """相互相関の結果を、数値入りの一文にする。

    Args:
        correlation: 相互相関の結果。

    Returns:
        結果の一文。
    """
    delay = correlation.delay_s
    relation = (
        "遅れています" if delay > 0 else "進んでいます" if delay < 0 else "同時に動いています"
    )
    timing = f"{_time_text(abs(delay))} {relation}" if delay != 0 else relation
    direction = "（逆向きに変化しています）" if correlation.peak_coefficient < 0 else ""
    return (
        f"出力は入力より {timing}{direction}。"
        f"波形の一致度は {abs(correlation.peak_coefficient):.3g} です（1 で完全に一致）。"
    )


def _alignment_notes(record: AlignmentRecord) -> list[str]:
    """時刻の対応付けのうち、結果に関わる変換を文にする（読み手に必ず見せる）。

    時刻オフセットの指定と補間は、データに手を加えた結果なので隠さない。同じ時刻のサンプルを
    そのまま使ったときは、何も変換していないので出さない。

    Args:
        record: 対応付けの記録。

    Returns:
        文のリスト。
    """
    lines = []
    if record.offset_declared:
        lines.append(
            f"時刻の対応: 出力の時刻に {record.output_time_offset_s:.6g} s を足して対応付け"
        )
    method = "線形補間" if record.method is InterpolationMethod.LINEAR else "PCHIP 補間"
    if not record.shared_timestamps:
        if record.input_interpolated and record.output_interpolated:
            lines.append(f"補間: {method}（入力と出力）")
        elif record.input_interpolated:
            lines.append(f"補間: {method}（入力）")
        else:
            lines.append(f"補間: {method}（出力）")
    if record.n_gap_samples:
        lines.append(f"欠損区間の {record.n_gap_samples} 時刻は値を求めていません")
    return lines


def _alignment_conditions(record: AlignmentRecord) -> list[str]:
    """解析した範囲など、計算の条件の記録を文にする（普段は畳んで表示する）。

    Args:
        record: 対応付けの記録。

    Returns:
        文のリスト。
    """
    lines = [f"範囲: {record.start_s:.6g}〜{record.end_s:.6g} s（{record.n_samples} サンプル）"]
    if not record.offset_declared:
        lines.append("時刻の対応: 同じ時計")
    if record.shared_timestamps:
        lines.append("補間: なし（同じ時刻のサンプルをそのまま対応付け）")
    return lines


def _metric(
    key: str, label: str, value: float | None, unit: str | None = None, note: str = ""
) -> AnalysisMetric:
    """指標を作る。

    Args:
        key: 指標の識別名。
        label: 表示名。
        value: 値。求められなければ None。
        unit: 単位。
        note: 指標の定義。

    Returns:
        指標。
    """
    return AnalysisMetric(key=key, label=label, value=value, unit=unit, note=note)


def _percent(ratio: float | None) -> float | None:
    """割合をパーセントにする。

    Args:
        ratio: 割合。None なら None を返す。

    Returns:
        パーセントの値。
    """
    return None if ratio is None else ratio * 100.0


def _ratio(numerator: str | None, denominator: str | None) -> str | None:
    """2 つの単位の比の単位を返す。

    Args:
        numerator: 分子の単位。
        denominator: 分母の単位。

    Returns:
        単位（例: "V/N"）。どちらかが不明なら None。
    """
    if numerator is None or denominator is None:
        return None
    return f"{_grouped(numerator)}/{_grouped(denominator)}"


def _product(first: str | None, second: str | None) -> str | None:
    """2 つの単位の積の単位を返す。

    Args:
        first: 1 つ目の単位。
        second: 2 つ目の単位。

    Returns:
        単位（例: "N·V"）。どちらかが不明なら None。
    """
    if first is None or second is None:
        return None
    return f"{_grouped(first)}·{_grouped(second)}"


def _axis_label(name: str, unit: str | None) -> str:
    """軸の名前を、単位があれば「名前 [単位]」の形にする。

    Args:
        name: 名前。
        unit: 単位。

    Returns:
        軸の名前。
    """
    return name if unit is None else f"{name} [{unit}]"


def _thin(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """グラフに描く点を、等間隔に間引いて MAX_CHART_POINTS 個以下にする（表示だけに使う）。

    Args:
        x: X 座標。
        y: Y 座標。

    Returns:
        間引いた (X, Y)。
    """
    stride = max(1, math.ceil(x.size / MAX_CHART_POINTS))
    return x[::stride], y[::stride]


# ---------------------------------------------------------------------------
# 計算結果のキャッシュ（derived/）
# ---------------------------------------------------------------------------


def analysis_fingerprint(spec: AnalysisSpec) -> str:
    """解析の計算結果を決める内容（入力チャンネル・種類・条件）の指紋を返す。

    名前・ID・作成日時は計算結果に影響しないので含めない。正規化データは変更しないため、
    同じ指紋のキャッシュは同じ計算結果として使える。

    Args:
        spec: 解析。

    Returns:
        指紋（SHA-256 の先頭 16 文字）。
    """
    payload = spec.model_dump_json(exclude={"id", "name", "created_at"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def result_to_frame(result: AnalysisResult, infos: list[AnalysisOutputInfo]) -> pl.DataFrame:
    """解析の結果を、キャッシュとして保存する表にする。

    列は、時刻 ``__time_s`` と、出力ごとの値（出力名）・帯（``<出力名>__lower`` / ``__upper``）・
    印（``<出力名>__marker``、印を付ける時刻なら True）。

    Args:
        result: 解析の結果。
        infos: 出力の情報（結果の系列と同じ順）。

    Returns:
        キャッシュの表。
    """
    columns: dict[str, pl.Series | FloatArray] = {DATASET_TIME_COLUMN: result.series[0].time_s}
    for info, series in zip(infos, result.series, strict=True):
        columns[info.key] = series.values
        if series.band_lower is not None and series.band_upper is not None:
            columns[info.key + _LOWER_SUFFIX] = series.band_lower
            columns[info.key + _UPPER_SUFFIX] = series.band_upper
        if series.marker_times_s is not None:
            marked = np.isin(series.time_s, series.marker_times_s)
            columns[info.key + _MARKER_SUFFIX] = pl.Series(marked, dtype=pl.Boolean)
    return pl.DataFrame(columns)


def result_from_frame(
    spec: AnalysisSpec, channel: Channel, source: Source, frame: pl.DataFrame
) -> AnalysisResult | None:
    """キャッシュの表から解析の結果を作り直す（再計算しない）。

    Args:
        spec: 解析。
        channel: 入力チャンネル（出力の名前・単位の決定に使う）。
        source: 入力チャンネルの Source（時間軸）。
        frame: キャッシュの表。

    Returns:
        解析の結果。表の列が出力と合わない場合は None（キャッシュを使わずに再計算させる）。
    """
    infos = output_infos(spec, channel)
    if DATASET_TIME_COLUMN not in frame.columns or any(
        info.key not in frame.columns for info in infos
    ):
        return None
    time_s = frame[DATASET_TIME_COLUMN].to_numpy().astype(np.float64)
    series = []
    for info in infos:
        lower = _optional_column(frame, info.key + _LOWER_SUFFIX)
        upper = _optional_column(frame, info.key + _UPPER_SUFFIX)
        marker_column = info.key + _MARKER_SUFFIX
        markers = (
            time_s[frame[marker_column].to_numpy()] if marker_column in frame.columns else None
        )
        series.append(
            PlotSeries(
                series_id=info.series_id,
                label=info.label,
                channel_name=channel.name,
                unit=info.unit,
                source=source,
                time_s=time_s,
                values=frame[info.key].to_numpy().astype(np.float64),
                band_lower=lower,
                band_upper=upper,
                marker_times_s=markers,
                analysis_id=spec.id,
            )
        )
    return AnalysisResult(analysis_id=spec.id, series=series)


def _optional_column(frame: pl.DataFrame, name: str) -> FloatArray | None:
    """表に列があれば float64 の配列で返す（なければ None）。

    Args:
        frame: 表。
        name: 列名。

    Returns:
        列の値、または None。
    """
    return frame[name].to_numpy().astype(np.float64) if name in frame.columns else None


# ---------------------------------------------------------------------------
# GUI の初期値
# ---------------------------------------------------------------------------


def estimate_noise_std(values: FloatArray) -> float:
    """信号に乗っている白色雑音の標準偏差を推定する（カルマンフィルタの初期値用）。

    隣り合う値の差 d = x[i+1] - x[i] は、ゆっくり変わる信号成分をほぼ打ち消し、
    白色雑音の分散を 2 倍にする。外れ値に強いよう中央絶対偏差（MAD）を使い、
    正規分布での換算係数 1.4826 を掛ける::

        σ ≈ 1.4826 · median(|d - median(d)|) / √2

    Args:
        values: 信号（欠損は NaN）。

    Returns:
        推定した標準偏差（推定できない場合は 1.0）。
    """
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 1.0
    steps = np.diff(finite)
    mad = float(np.median(np.abs(steps - np.median(steps))))
    sigma = 1.4826 * mad / math.sqrt(2.0)
    if sigma > 0:
        return sigma
    spread = float(np.std(finite))
    return spread if spread > 0 else 1.0


def round_significant(value: float, digits: int = DEFAULT_SIGNIFICANT_DIGITS) -> float:
    """実数を有効数字 ``digits`` 桁に丸める（GUI に入れる初期値を読みやすくするため）。

    Args:
        value: 実数。
        digits: 有効数字の桁数。

    Returns:
        丸めた値（例: 1.8412847 → 1.841、0.0137486 → 0.01375）。
    """
    return float(f"{value:.{digits}g}")


def default_kalman_settings(model_kind: KalmanModelKind, data: ChannelData) -> KalmanSettings:
    """選んだモデルの、データに合わせた初期値の条件を返す（利用者が調整する出発点）。

    観測雑音 σ はデータから推定し（estimate_noise_std）、プロセス雑音 q はフィルタの追従の速さ
    （帯域）ω がサンプリング周波数の 1/20 程度になるよう決める。等速度モデルのカルマンフィルタの
    帯域はおよそ ω ≈ (q / (σ² Δt))^(1/4) なので::

        q = ω⁴ σ² Δt,   ω = 2π / (20 Δt)

    線形モデルは、これと同じ等速度モデルを行列で表したもの、多項式の観測モデルは
    恒等写像 h(x) = x（係数 [0, 1]）から始める。
    σ・q・R・Q は有効数字 4 桁に丸める（DEFAULT_SIGNIFICANT_DIGITS）。線形モデルの Δt は、
    時刻の計算誤差による端数だけを除くよう有効数字 6 桁に丸める。

    Args:
        model_kind: モデルの種類。
        data: 入力チャンネルのデータ。

    Returns:
        カルマンフィルタの条件。
    """
    sigma = round_significant(estimate_noise_std(data.values))
    dt = round_significant(_median_interval(data.time_s), 6)
    omega = default_omega(data)
    density = round_significant(omega**4 * sigma**2 * dt)
    initial = default_initial_state(2, data, sigma=sigma, omega=omega)
    if model_kind is KalmanModelKind.KINEMATIC:
        return KalmanSettings(
            model_kind=model_kind,
            kinematic=KinematicSystemParams(
                process_noise_density=density, measurement_noise_std=sigma
            ),
            initial=initial,
        )
    if model_kind is KalmanModelKind.LINEAR:
        # 2×2 の Q は行列式に対角成分の積の 1/4 の余裕があるので、成分ごとに 4 桁へ丸めても
        # 半正定値のまま（丸めの相対誤差は 0.05 % 以下）。
        process_noise = [
            [round_significant(value) for value in row]
            for row in kinematic_process_noise(2, density, dt).tolist()
        ]
        return KalmanSettings(
            model_kind=model_kind,
            linear=LinearSystemParams(
                transition=kinematic_transition_matrix(2, dt).tolist(),
                observation=[[1.0, 0.0]],
                process_noise=process_noise,
                measurement_noise=[[round_significant(sigma**2)]],
            ),
            initial=initial,
        )
    return KalmanSettings(
        model_kind=model_kind,
        polynomial_observation=PolynomialObservationParams(
            process_noise_density=density, measurement_noise_std=sigma, coefficients=[0.0, 1.0]
        ),
        initial=initial,
    )


def default_initial_state(
    n_states: int,
    data: ChannelData,
    *,
    sigma: float,
    omega: float,
    coefficients: list[float] | None = None,
) -> KalmanInitialState:
    """カルマンフィルタの初期状態の初期値を返す。

    先頭の状態は最初の観測（多項式の観測モデルでは 1 次の項で逆算した値）、微分の状態は 0 とする。
    初期共分散は対角行列で、先頭は σ²、k 階微分は (信号の広がり × ω^k)² と大きめにして、
    最初の数サンプルでデータに合わせられるようにする。
    共分散は有効数字 4 桁に丸める。状態は丸めない（大きなオフセットのある信号で、丸めの誤差が
    σ を超えて初期の推定を誤らせないため）。

    Args:
        n_states: 状態の次元。
        data: 入力チャンネルのデータ。
        sigma: 観測雑音の標準偏差。
        omega: フィルタの帯域 [rad/s]（微分の状態の不確かさの見積もりに使う）。
        coefficients: 多項式の観測モデルの係数（逆算に使う）。None なら観測 = 状態とみなす。

    Returns:
        初期状態。
    """
    finite = data.values[np.isfinite(data.values)]
    first = float(finite[0]) if finite.size else 0.0
    spread = float(np.std(finite)) if finite.size else 1.0
    spread = spread if spread > 0 else 1.0
    if coefficients is not None and len(coefficients) >= 2 and coefficients[1] != 0:
        first = (first - coefficients[0]) / coefficients[1]
        spread /= abs(coefficients[1])
        sigma /= abs(coefficients[1])
    state = [first] + [0.0] * (n_states - 1)
    variances = [
        round_significant(variance)
        for variance in [sigma**2] + [(spread * omega**order) ** 2 for order in range(1, n_states)]
    ]
    covariance = [
        [variances[row] if row == column else 0.0 for column in range(n_states)]
        for row in range(n_states)
    ]
    return KalmanInitialState(state=state, covariance=covariance)


def default_omega(data: ChannelData) -> float:
    """初期値で使うフィルタの帯域 ω = 2π / (20 Δt) [rad/s] を返す。

    Args:
        data: 入力チャンネルのデータ。

    Returns:
        帯域 [rad/s]。
    """
    return 2.0 * math.pi / (DEFAULT_BANDWIDTH_DIVISOR * _median_interval(data.time_s))


def _median_interval(time_s: FloatArray) -> float:
    """時刻の間隔の中央値を返す（2 サンプル未満なら 1 秒）。

    Args:
        time_s: 時刻 [s]。

    Returns:
        間隔の中央値 [s]。
    """
    if time_s.size < 2:
        return 1.0
    steps = np.diff(time_s)
    positive = steps[steps > 0]
    return float(np.median(positive)) if positive.size else 1.0


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------


def _require[ValueT](value: ValueT | None, name: str) -> ValueT:
    """解析の種類に対応する条件を取り出す（モデルの検証で必ずあるが、型の上で明示する）。

    Args:
        value: 条件。
        name: 条件の項目名（エラーメッセージ用）。

    Returns:
        条件。

    Raises:
        AnalysisInputError: 条件がない場合。
    """
    if value is None:
        raise AnalysisInputError(f"解析の条件 {name} がありません")
    return value


def _per_second(unit: str | None, order: int) -> str | None:
    """単位を時間で order 回割った単位を返す（微分の結果の単位）。

    Args:
        unit: 元の単位。None なら None を返す。
        order: 時間で割る回数（0 なら元の単位のまま）。

    Returns:
        単位（例: "m" → "m/s"、"m/s" → "(m/s)/s"）。
    """
    if unit is None or order == 0:
        return unit
    suffix = "/s" if order == 1 else f"/s^{order}"
    return f"{_grouped(unit)}{suffix}"


def _times_second(unit: str | None) -> str | None:
    """単位に時間を掛けた単位を返す（積分の結果の単位）。

    Args:
        unit: 元の単位。None なら None を返す。

    Returns:
        単位（例: "m/s^2" → "(m/s^2)·s"）。
    """
    return None if unit is None else f"{_grouped(unit)}·s"


def _grouped(unit: str) -> str:
    """演算子を含む単位をかっこで囲む（"m/s" → "(m/s)"）。

    Args:
        unit: 単位。

    Returns:
        必要ならかっこで囲んだ単位。
    """
    return f"({unit})" if any(symbol in unit for symbol in "/*^· ") else unit
