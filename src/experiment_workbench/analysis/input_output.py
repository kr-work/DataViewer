"""入出力関係の解析（SPEC A-09・A-10）: 相互相関による遅れ、ステップ応答の指標、ヒステリシス。

どの解析も、同じ時刻に対応付けた入力と出力の組（AlignedPair、alignment.py）を受け取る。
入力（例: 張力）を加えたときの、出力（例: 電圧）の時間的な応答・遅れと、入出力特性を調べる。

    - CrossCorrelationAnalyzer: 相互相関のピークから遅れを推定する（連続的に変化する入力に向く）
    - StepResponseAnalyzer: ステップ状の入力に対する出力の遅れ・立ち上がり・時定数・整定・行き過ぎ
    - HysteresisAnalyzer: 回帰直線（感度・オフセット）、直線性誤差、増加時と減少時の分岐の差、
      ループ面積

欠損（NaN）は補わずにエラーにする（欠損のない範囲を選んで解析する）。
求められない指標は推測で埋めずに None にし、理由を結果の注記に書く。
このモジュールは Qt に依存しない。
"""

import math

import numpy as np
from scipy import signal

from experiment_workbench.analysis.signal_processing import (
    AnalysisInputError,
    ButterworthFilter,
    check_uniform_sampling,
    validate_time_axis,
)
from experiment_workbench.model import (
    AlignedPair,
    ButterworthKind,
    ButterworthParams,
    CrossCorrelationParams,
    CrossCorrelationResult,
    HysteresisParams,
    HysteresisResult,
    OnsetThresholdMode,
    StepDirection,
    StepResponseParams,
    StepResponseResult,
)
from experiment_workbench.validation import BoolArray, FloatArray, IntArray

SMOOTHING_ORDER = 4
"""平滑化に使うバターワースフィルタの次数（ゼロ位相で前後にかけるので、実効的には 8 次）。"""

CHANGE_NOISE_RATIO = 3.0
"""変化量が初期値の区間のばらつき（標準偏差）の何倍を超えたら「変化した」とみなすか。"""

MIN_BIN_SAMPLES = 3
"""ヒステリシスで、区間ごとの分岐の平均を求めるのに必要なサンプル数。"""

RATE_SCALE_PERCENTILE = 95.0
"""入力の変化の速さの基準にするパーセンタイル（HysteresisParams 参照）。"""

DIRECTION_FLIP_LIMIT = 0.05
"""入力の変化の向きが反転するサンプルの割合の上限（これを超えたら判定が乱れているとみなす）。

入力が往復するデータでも、向きが反転するのは折り返しの数回だけなので、割合はほぼ 0 になる。
白色雑音が支配的なら、反転の割合は約 0.5 になる。
"""

SMITH_LOW_LEVEL = 1.0 - math.exp(-1.0 / 3.0)
"""2 点法の低い方の値（約 28.3%）。一次遅れ＋むだ時間の応答は、時刻 L + τ/3 でこの値になる。"""

SMITH_HIGH_LEVEL = 1.0 - math.exp(-1.0)
"""2 点法の高い方の値（約 63.2%）。一次遅れ＋むだ時間の応答は、時刻 L + τ でこの値になる。"""

_RISE_LOW_LEVEL = 0.1
_RISE_HIGH_LEVEL = 0.9
_MIDPOINT_LEVEL = 0.5


# ---------------------------------------------------------------------------
# 相互相関（A-09）
# ---------------------------------------------------------------------------


class CrossCorrelationAnalyzer:
    """相互相関のピークから、入力に対する出力の遅れを推定するクラス（A-09）。

    遅れ k サンプルでの相関係数は、重なっている部分どうしで正規化する::

        c(k) = Σ y[n + k] x[n] / sqrt(Σ x[n]² · Σ y[n + k]²)    （和は重なっている n について）

    全体の長さで割る方法（偏りのある推定）は、遅れが大きいほど値が小さくなり、なだらかなピークの
    位置が 0 の側にずれるため使わない。

    Attributes:
        params: 相互相関の条件。
    """

    def __init__(self, params: CrossCorrelationParams) -> None:
        """相互相関の条件を受け取る。

        Args:
            params: 相互相関の条件。
        """
        self.params = params

    def analyze(self, pair: AlignedPair) -> CrossCorrelationResult:
        """遅れを推定する。

        Args:
            pair: 同じ時刻に対応付けた入力と出力（等間隔の時刻）。

        Returns:
            推定した遅れと、遅れごとの相関係数。

        Raises:
            AnalysisInputError: 欠損がある、時刻が等間隔でない、信号が変化しない、
                または探索範囲に調べられる遅れがない場合。
        """
        times, x, y = _finite_pair(pair)
        interval = 1.0 / check_uniform_sampling(times)
        if self.params.use_derivative:
            x = np.gradient(x, times)
            y = np.gradient(y, times)
        x = np.asarray(signal.detrend(x, type=self.params.detrend.value), dtype=np.float64)
        y = np.asarray(signal.detrend(y, type=self.params.detrend.value), dtype=np.float64)
        for name, values in (("入力", x), ("出力", y)):
            if not np.any(values):
                raise AnalysisInputError(
                    f"{name}が（傾向を取り除くと）変化していないため、相関を求められません"
                )
        n_samples = times.size
        lags = signal.correlation_lags(n_samples, n_samples, mode="full")
        coefficients = _overlap_normalized(signal.correlate(y, x, mode="full"), x, y, lags)
        lag_s = lags * interval
        tolerance = 1e-9 * interval  # 探索範囲の端の遅れを、丸め誤差で落とさないため
        window = (lag_s >= self.params.min_lag_s - tolerance) & (
            lag_s <= self.params.max_lag_s + tolerance
        )
        max_lag = n_samples // 2  # 重なりがサンプル数の半分以上残る遅れだけを探す
        usable = window & (np.abs(lags) <= max_lag)
        notes: list[str] = []
        if not usable.any():
            raise AnalysisInputError(
                f"遅れを探す範囲（{self.params.min_lag_s:.6g}〜{self.params.max_lag_s:.6g} s）に、"
                f"調べられる遅れがありません（サンプル間隔 {interval:.6g} s。重なりが半分以上残る"
                f"遅れは ±{max_lag * interval:.6g} s まで）"
            )
        if (window & ~usable).any():
            notes.append(
                "重なりがサンプル数の半分未満になる遅れ"
                f"（±{max_lag * interval:.6g} s を超える遅れ）は探していません。"
                "範囲を長くすると、より大きな遅れを探せます"
            )
        indices = np.flatnonzero(usable)
        local = int(np.argmax(np.abs(coefficients[indices])))
        peak = int(indices[local])
        at_edge = local in (0, indices.size - 1)
        sign = 1.0 if coefficients[peak] >= 0.0 else -1.0
        shift, peak_value = 0.0, float(coefficients[peak])
        if not at_edge:
            shift, vertex = _parabolic_peak(
                sign * float(coefficients[peak - 1]),
                sign * float(coefficients[peak]),
                sign * float(coefficients[peak + 1]),
            )
            peak_value = sign * vertex
        if sign < 0:
            notes.append("相関係数が負です。出力は入力と逆向きに変化しています")
        if at_edge:
            notes.append(
                "相関の絶対値が最大になったのは探索範囲の端です。真の遅れは範囲の外にあるかも"
                "しれないので、探索範囲を広げて確認してください"
            )
        return CrossCorrelationResult(
            delay_s=float((lags[peak] + shift) * interval),
            peak_coefficient=float(np.clip(peak_value, -1.0, 1.0)),
            sample_interval_s=interval,
            at_search_edge=at_edge,
            lags_s=np.asarray(lag_s[indices], dtype=np.float64),
            coefficients=np.asarray(coefficients[indices], dtype=np.float64),
            notes=notes,
        )


def _overlap_normalized(
    raw: FloatArray, x: FloatArray, y: FloatArray, lags: IntArray
) -> FloatArray:
    """相互相関の和を、遅れごとの重なっている部分のエネルギーで割って相関係数にする。

    遅れ k >= 0 では x[0 : n - k] と y[k : n]、k < 0 では x[-k : n] と y[0 : n + k] が重なる。
    累積和を使い、すべての遅れのエネルギーを O(n) で求める。

    Args:
        raw: scipy.signal.correlate(y, x, mode="full") の結果。
        x: 入力（傾向を除いたもの）。
        y: 出力（傾向を除いたもの）。
        lags: raw の各要素に対応する遅れ [サンプル]。

    Returns:
        遅れごとの相関係数（重なりのエネルギーが 0 の遅れは 0）。
    """
    n_samples = x.size
    cumulative_x = np.concatenate([[0.0], np.cumsum(x * x)])
    cumulative_y = np.concatenate([[0.0], np.cumsum(y * y)])
    shift = np.abs(lags)
    positive = lags >= 0
    energy_x = np.where(
        positive,
        cumulative_x[n_samples - shift] - cumulative_x[0],
        cumulative_x[n_samples] - cumulative_x[shift],
    )
    energy_y = np.where(
        positive,
        cumulative_y[n_samples] - cumulative_y[shift],
        cumulative_y[n_samples - shift] - cumulative_y[0],
    )
    denominator = np.sqrt(np.maximum(energy_x, 0.0) * np.maximum(energy_y, 0.0))
    valid = denominator > 0.0
    return np.asarray(
        np.where(valid, raw / np.where(valid, denominator, 1.0), 0.0), dtype=np.float64
    )


def _parabolic_peak(left: float, center: float, right: float) -> tuple[float, float]:
    """3 点に放物線を当てはめ、頂点の位置（中央の点からのずれ）と頂点の値を返す。

    Args:
        left: 中央の 1 つ前の値。
        center: 中央（最大）の値。
        right: 中央の 1 つ後の値。

    Returns:
        (頂点の位置のずれ [サンプル]、-0.5〜0.5, 頂点の値)。中央が極大でなければ (0, center)。
    """
    curvature = left - 2.0 * center + right
    if curvature >= 0.0:
        return 0.0, center
    shift = 0.5 * (left - right) / curvature
    return shift, center - 0.25 * (left - right) * shift


# ---------------------------------------------------------------------------
# ステップ応答（A-09）
# ---------------------------------------------------------------------------


class StepResponseAnalyzer:
    """ステップ状の入力に対する出力の応答指標を求めるクラス（A-09）。

    指標の定義は StepResponseResult を参照。雑音に強くするため、しきい値を越えた時刻は、
    応答が初めて 50% に達したサンプル（中点）を基準に探す。50% 以下の値は中点より前で最後に
    越えた時刻、50% より上の値は中点以降で最初に越えた時刻とする。中点より前の雑音による
    一瞬の越えを、変化の開始と取り違えないため。

    Attributes:
        params: ステップ応答の条件。
    """

    def __init__(self, params: StepResponseParams) -> None:
        """ステップ応答の条件を受け取る。

        Args:
            params: ステップ応答の条件。
        """
        self.params = params

    def analyze(self, pair: AlignedPair) -> StepResponseResult:
        """ステップ応答の指標を求める。

        Args:
            pair: 同じ時刻に対応付けた入力と出力（範囲に入力の変化が 1 回入っていること）。

        Returns:
            ステップ応答の指標。

        Raises:
            AnalysisInputError: 欠損がある、初期値・最終値の区間が短すぎる、入力が変化していない、
                変化の向きが指定と違う、しきい値が変化量以上、または入力の変化が初期値・最終値の
                区間の中にある場合。
        """
        params = self.params
        times, u, y = _finite_pair(pair)
        u = _smooth(times, u, params.smoothing_cutoff_hz)
        y = _smooth(times, y, params.smoothing_cutoff_hz)
        duration = float(times[-1] - times[0])
        baseline = times <= times[0] + params.baseline_fraction * duration
        final = times >= times[-1] - params.final_fraction * duration
        if np.count_nonzero(baseline) < 2 or np.count_nonzero(final) < 2:
            raise AnalysisInputError(
                "初期値・最終値を求める区間のサンプルが足りません。範囲を広げるか、区間の割合を"
                "大きくしてください"
            )
        baseline_end = int(np.flatnonzero(baseline)[-1])
        final_start = int(np.flatnonzero(final)[0])

        input_initial, input_final = float(np.mean(u[baseline])), float(np.mean(u[final]))
        input_change = input_final - input_initial
        input_spread = _difference_noise(u[baseline])
        if input_change == 0.0 or abs(input_change) <= CHANGE_NOISE_RATIO * input_spread:
            raise AnalysisInputError(
                f"入力の初期値と最終値の差（{input_change:.6g}）が、初期値の区間のばらつき"
                f"（標準偏差 {input_spread:.6g}）と区別できません。入力の変化が 1 回入る範囲を"
                "選んでください"
            )
        direction = StepDirection.RISING if input_change > 0 else StepDirection.FALLING
        if params.direction is not StepDirection.AUTO and params.direction is not direction:
            actual = "増加" if direction is StepDirection.RISING else "減少"
            raise AnalysisInputError(
                f"範囲の中で入力は{actual}していますが、変化の向きに別の向きが指定されています"
            )
        if params.onset_mode is OnsetThresholdMode.FRACTION:
            fraction = params.onset_threshold
        else:
            fraction = params.onset_threshold / abs(input_change)
            if fraction >= 1.0:
                raise AnalysisInputError(
                    f"しきい値 {params.onset_threshold:.6g} が、入力の変化量 "
                    f"{abs(input_change):.6g} 以上です"
                )
        normalized_input = (u - input_initial) / input_change
        anchor_input = _first_index_at_least(normalized_input, _MIDPOINT_LEVEL, start=0)
        if anchor_input is None or anchor_input <= baseline_end:
            raise AnalysisInputError(
                "入力の変化が、初期値を求める区間（範囲の先頭）の中にあります。範囲の開始を"
                "前にずらすか、初期値の区間の割合を小さくしてください"
            )
        if anchor_input >= final_start:
            raise AnalysisInputError(
                "入力の変化が、最終値を求める区間（範囲の末尾）の中にあります。範囲の終了を"
                "後ろにずらすか、最終値の区間の割合を小さくしてください"
            )
        onset = _level_time(times, normalized_input, fraction, anchor_input)
        if onset is None:  # 初期値の区間の平均が 0 なので、通常は起こらない
            raise AnalysisInputError("入力がしきい値を越えた時刻が見つかりません")

        output_initial, output_final = float(np.mean(y[baseline])), float(np.mean(y[final]))
        output_change = output_final - output_initial
        output_spread = _difference_noise(y[baseline])
        result = StepResponseResult(
            direction=direction,
            input_onset_s=onset,
            input_initial=input_initial,
            input_final=input_final,
            output_initial=output_initial,
            output_final=output_final,
            gain=output_change / input_change,
            response_time_s=np.asarray(times - onset, dtype=np.float64),
            normalized_input=np.asarray(normalized_input, dtype=np.float64),
            normalized_output=np.full(times.shape, np.nan),
        )
        if output_change == 0.0 or abs(output_change) <= CHANGE_NOISE_RATIO * output_spread:
            result.notes.append(
                f"出力の変化（{output_change:.6g}）が初期値の区間のばらつき（標準偏差 "
                f"{output_spread:.6g}）と区別できないため、時間の指標を求めていません"
            )
            return result
        normalized_output = np.asarray((y - output_initial) / output_change, dtype=np.float64)
        result.normalized_output = normalized_output
        anchor = _first_index_at_least(normalized_output, _MIDPOINT_LEVEL, start=baseline_end + 1)
        if anchor is None:
            result.notes.append("出力が変化量の 50% に達していないため、時間の指標を求めていません")
            return result
        self._fill_time_metrics(result, times, normalized_output, anchor, fraction, final_start)
        return result

    def _fill_time_metrics(
        self,
        result: StepResponseResult,
        times: FloatArray,
        normalized: FloatArray,
        anchor: int,
        fraction: float,
        final_start: int,
    ) -> None:
        """正規化した出力から時間の指標を求め、結果に書き込む。

        Args:
            result: 書き込む結果（input_onset_s は決まっていること）。
            times: 時刻 [s]。
            normalized: 正規化した出力 r。
            anchor: r が初めて 50% に達したサンプルの番号。
            fraction: 変化の開始を判定する割合（入力と同じ）。
            final_start: 最終値を求める区間の最初のサンプルの番号。
        """
        onset = result.input_onset_s
        levels = {
            "応答開始": fraction,
            "10%": _RISE_LOW_LEVEL,
            "28.3%": SMITH_LOW_LEVEL,
            "50%": _MIDPOINT_LEVEL,
            "63.2%": SMITH_HIGH_LEVEL,
            "90%": _RISE_HIGH_LEVEL,
        }
        crossing: dict[str, float | None] = {}
        for name, level in levels.items():
            crossing[name] = _level_time(times, normalized, level, anchor)
            if crossing[name] is None:
                result.notes.append(f"出力が変化量の {name} を越えた時刻が範囲の中に見つかりません")

        start, half = crossing["応答開始"], crossing["50%"]
        low, high = crossing["10%"], crossing["90%"]
        if start is not None:
            result.response_delay_s = start - onset
            if result.response_delay_s < 0:
                result.notes.append(
                    "出力が入力より先に変化し始めています。時刻の対応（時刻オフセット）を確認してください"
                )
        if half is not None:
            result.delay_time_s = half - onset
        if low is not None and high is not None:
            result.rise_time_s = high - low
        smith_low, smith_high = crossing["28.3%"], crossing["63.2%"]
        if smith_low is not None and smith_high is not None and smith_high > smith_low:
            result.time_constant_s = 1.5 * (smith_high - smith_low)
            result.dead_time_s = smith_high - result.time_constant_s - onset
        result.overshoot = max(0.0, float(np.max(normalized[anchor:])) - 1.0)
        result.settling_time_s = self._settling_time(result, times, normalized, anchor, final_start)

    def _settling_time(
        self,
        result: StepResponseResult,
        times: FloatArray,
        normalized: FloatArray,
        anchor: int,
        final_start: int,
    ) -> float | None:
        """整定時間（最終値 ± 幅に入ったまま出なくなるまでの時間）を求める。

        Args:
            result: 結果（注記を書き足す）。
            times: 時刻 [s]。
            normalized: 正規化した出力 r。
            anchor: r が初めて 50% に達したサンプルの番号。
            final_start: 最終値を求める区間の最初のサンプルの番号。

        Returns:
            整定時間 [s]。求められなければ None（理由は注記に書く）。
        """
        band = self.params.settling_band
        outside = np.flatnonzero(np.abs(normalized[anchor:] - 1.0) > band)
        if outside.size == 0:
            # 50% の時点で既に幅の中（幅が 50% 以上の場合）: 幅に下から入った時刻とする。
            entered = _level_time(times, normalized, 1.0 - band, anchor)
            return None if entered is None else entered - result.input_onset_s
        last = anchor + int(outside[-1])
        if last >= final_start:
            result.notes.append(
                "最終値を求める区間（範囲の末尾）でも、"
                f"最終値から ±{band * 100:.3g}% の幅を出ています。"
                "雑音が幅より大きいか、まだ整定していないため、整定時間を求めていません"
                "（平滑化するか、幅や範囲を変えてください）"
            )
            return None
        boundary = 1.0 + band if normalized[last] > 1.0 else 1.0 - band
        return _interpolate_time(times, normalized, last, last + 1, boundary) - result.input_onset_s


def _first_index_at_least(values: FloatArray, level: float, *, start: int) -> int | None:
    """``start`` 以降で、値が初めて ``level`` 以上になるサンプルの番号を返す。

    Args:
        values: 値。
        level: 基準の値。
        start: 探し始めるサンプルの番号。

    Returns:
        サンプルの番号。見つからなければ None。
    """
    found = np.flatnonzero(values[start:] >= level)
    return None if found.size == 0 else start + int(found[0])


def _level_time(
    times: FloatArray, normalized: FloatArray, level: float, anchor: int
) -> float | None:
    """正規化した応答が ``level`` を越えた時刻を、中点 ``anchor`` を基準に求める（線形補間）。

    ``level`` が 50% 以下なら、anchor 以前で level を下回っていた最後のサンプルの直後の越え、
    50% より上なら、anchor 以降で初めて level 以上になったサンプルの直前からの越えとする。

    Args:
        times: 時刻 [s]。
        normalized: 正規化した応答（0 から 1 に向かう）。
        level: 越えた時刻を求める値。
        anchor: 応答が初めて 50% に達したサンプルの番号。

    Returns:
        越えた時刻 [s]。範囲の中で見つからなければ None。
    """
    if level <= _MIDPOINT_LEVEL:
        below = np.flatnonzero(normalized[:anchor] < level)
        if below.size == 0:
            return None
        index = int(below[-1])
        return _interpolate_time(times, normalized, index, index + 1, level)
    found = _first_index_at_least(normalized, level, start=anchor)
    if found is None:
        return None
    if found == 0:
        return float(times[0])
    return _interpolate_time(times, normalized, found - 1, found, level)


def _interpolate_time(
    times: FloatArray, values: FloatArray, before: int, after: int, level: float
) -> float:
    """隣り合う 2 サンプルの間で、値が ``level`` になる時刻を線形補間で求める。

    Args:
        times: 時刻 [s]。
        values: 値。
        before: 前のサンプルの番号。
        after: 後のサンプルの番号。
        level: 求める値。

    Returns:
        時刻 [s]（2 サンプルの値が同じなら後のサンプルの時刻）。
    """
    v0, v1 = float(values[before]), float(values[after])
    t0, t1 = float(times[before]), float(times[after])
    if v1 == v0:
        return t1
    return t0 + (level - v0) * (t1 - t0) / (v1 - v0)


# ---------------------------------------------------------------------------
# ヒステリシス・入出力特性（A-10）
# ---------------------------------------------------------------------------


class HysteresisAnalyzer:
    """入出力特性とヒステリシスを求めるクラス（A-10）。

    分岐の判定と区間の分け方は HysteresisParams、指標の定義は HysteresisResult を参照。

    Attributes:
        params: ヒステリシスの条件。
    """

    def __init__(self, params: HysteresisParams) -> None:
        """ヒステリシスの条件を受け取る。

        Args:
            params: ヒステリシスの条件。
        """
        self.params = params

    def analyze(self, pair: AlignedPair) -> HysteresisResult:
        """入出力特性とヒステリシスを求める。

        Args:
            pair: 同じ時刻に対応付けた入力と出力（入力が往復する範囲であること）。

        Returns:
            入出力特性とヒステリシスの指標。

        Raises:
            AnalysisInputError: 欠損がある、入力が変化していない、または平滑化の条件が
                データに合わない（不等間隔、カットオフ周波数がナイキスト周波数以上）場合。
        """
        params = self.params
        times, u, y = _finite_pair(pair)
        rate = np.gradient(_smooth(times, u, params.smoothing_cutoff_hz), times)
        scale = float(np.percentile(np.abs(rate), RATE_SCALE_PERCENTILE))
        if not scale > 0.0:
            raise AnalysisInputError("入力が変化していないため、増加時と減少時に分けられません")
        threshold = params.min_rate_fraction * scale
        increasing = rate > threshold
        decreasing = rate < -threshold
        flips = int(np.count_nonzero(np.diff(np.sign(rate)) != 0))
        unstable = flips > DIRECTION_FLIP_LIMIT * max(rate.size - 1, 1)

        slope, intercept = (float(value) for value in np.polyfit(u, y, 1))
        residual_sum = float(np.sum((y - (slope * u + intercept)) ** 2))
        total_sum = float(np.sum((y - np.mean(y)) ** 2))
        input_min, input_max = float(np.min(u)), float(np.max(u))
        n_bins = params.n_bins
        edges = np.linspace(input_min, input_max, n_bins + 1)
        width = (input_max - input_min) / n_bins
        bins = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, n_bins - 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        increasing_branch = _values_at_centers(bins, u, y, increasing, centers)
        decreasing_branch = _values_at_centers(bins, u, y, decreasing, centers)
        pooled = _values_at_centers(bins, u, y, np.ones(u.shape, dtype=np.bool_), centers)

        full_scale = abs(slope) * (input_max - input_min)
        result = HysteresisResult(
            slope=slope,
            intercept=intercept,
            r_squared=1.0 - residual_sum / total_sum if total_sum > 0.0 else None,
            input_min=input_min,
            input_max=input_max,
            full_scale_output=full_scale if full_scale > 0.0 else None,
            linearity_error=None,
            linearity_ratio=None,
            max_hysteresis=None,
            max_hysteresis_input=None,
            hysteresis_ratio=None,
            mean_branch_difference=None,
            loop_area=None,
            n_increasing=int(np.count_nonzero(increasing)),
            n_decreasing=int(np.count_nonzero(decreasing)),
            n_excluded=int(np.count_nonzero(~(increasing | decreasing))),
            bin_centers=np.asarray(centers, dtype=np.float64),
            increasing_branch=increasing_branch,
            decreasing_branch=decreasing_branch,
            increasing_mask=np.asarray(increasing, dtype=np.bool_),
            decreasing_mask=np.asarray(decreasing, dtype=np.bool_),
            direction_unstable=unstable,
        )
        if unstable:
            result.notes.append(
                "入力の雑音で、変化方向（増加・減少）の判定が乱れています。ヒステリシスが"
                "過小評価されている可能性があります。変化方向の判定に平滑化を使ってください"
            )
        both = np.isfinite(increasing_branch) & np.isfinite(decreasing_branch)
        if both.any():
            difference = increasing_branch[both] - decreasing_branch[both]
            largest = int(np.argmax(np.abs(difference)))
            result.max_hysteresis = float(abs(difference[largest]))
            result.max_hysteresis_input = float(centers[both][largest])
            result.mean_branch_difference = float(np.mean(difference))
            result.loop_area = float(abs(np.sum(difference) * width))
            result.notes.append(
                f"{n_bins} 区間のうち {int(both.sum())} 区間で、増加時と減少時の分岐を比べました"
            )
        else:
            result.notes.append(
                "増加時と減少時の両方のサンプルがある区間がないため、ヒステリシスを求めていません"
                "（入力が往復する範囲を選んでください）"
            )
        middle = np.where(both, 0.5 * (increasing_branch + decreasing_branch), pooled)
        valid = np.isfinite(middle)
        if valid.any():
            deviation = middle[valid] - (slope * centers[valid] + intercept)
            result.linearity_error = float(np.max(np.abs(deviation)))
        if result.full_scale_output is not None:
            if result.linearity_error is not None:
                result.linearity_ratio = result.linearity_error / result.full_scale_output
            if result.max_hysteresis is not None:
                result.hysteresis_ratio = result.max_hysteresis / result.full_scale_output
        else:
            result.notes.append(
                "回帰直線の傾きが 0 のため、フルスケールに対する割合を求めていません"
            )
        if result.n_excluded:
            result.notes.append(
                f"入力の変化が遅く向きが定まらない {result.n_excluded} サンプルを、"
                "分岐から除きました（回帰直線には使っています）"
            )
        return result


def _values_at_centers(
    bins: IntArray, u: FloatArray, y: FloatArray, mask: BoolArray, centers: FloatArray
) -> FloatArray:
    """区間ごとに、条件に合うサンプルへ直線を当てはめ、区間の中央の入力での出力の値を求める。

    単純な平均ではなく直線の当てはめを使うのは、増加時と減少時でサンプルの入力の位置が
    区間の中でずれていても、同じ入力の値どうしで比べるため（平均どうしの差には、
    傾き × 入力の平均の位置のずれ が混ざり、ヒステリシスのない特性でも差が出てしまう）。
    区間の中のサンプルの入力がほとんど同じ（ばらつきが区間の幅の 10⁻⁶ 未満）なら、平均を使う。
    最小二乗の式::

        y ≈ α + β (u - c)、 α = (Σy Σd² - Σd Σdy) / (n Σd² - (Σd)²)、 d = u - c

    Args:
        bins: 各サンプルが入る区間の番号。
        u: 入力。
        y: 出力。
        mask: 使うサンプル。
        centers: 区間の中央の入力の値 c。

    Returns:
        区間ごとの値（サンプルが MIN_BIN_SAMPLES 個未満の区間は NaN）。
    """
    n_bins = centers.size
    selected = bins[mask]
    offset = u[mask] - centers[selected]
    values = y[mask]
    count = np.bincount(selected, minlength=n_bins).astype(np.float64)
    sum_d = np.bincount(selected, weights=offset, minlength=n_bins)
    sum_y = np.bincount(selected, weights=values, minlength=n_bins)
    sum_dd = np.bincount(selected, weights=offset * offset, minlength=n_bins)
    sum_dy = np.bincount(selected, weights=offset * values, minlength=n_bins)
    denominator = count * sum_dd - sum_d * sum_d
    width = float(centers[1] - centers[0]) if n_bins > 1 else 1.0
    enough = count >= MIN_BIN_SAMPLES
    spread = denominator > (count * 1e-6 * width) ** 2  # 入力が区間の中で十分ばらついている
    result = np.full(n_bins, np.nan)
    fitted = enough & spread
    result[fitted] = (
        sum_y[fitted] * sum_dd[fitted] - sum_d[fitted] * sum_dy[fitted]
    ) / denominator[fitted]
    averaged = enough & ~spread
    result[averaged] = sum_y[averaged] / count[averaged]
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------


def _difference_noise(values: FloatArray) -> float:
    """隣り合う値の差から、白色雑音の標準偏差を推定する（変化したかどうかの判定用）。

    差 d = x[i+1] - x[i] は、ゆっくりした変化を打ち消し、白色雑音の分散を 2 倍にする。
    外れ値（区間に紛れ込んだステップの 1 回の大きな差）に強いよう、中央絶対偏差を使う::

        σ ≈ 1.4826 · median(|d - median(d)|) / √2

    初期値の区間にステップが紛れ込んだ場合も、ばらつきを大きく見積もらないので、
    「変化が初期値の区間の中にある」ことを正しく知らせられる。
    雑音がなく差のほとんどが 0 の場合（中央絶対偏差が 0）は、平均絶対偏差（正規分布での換算係数
    1.2533）を使う。ステップが 1 回だけ紛れ込んでも、値は小さいまま。

    Args:
        values: 値（欠損なし）。

    Returns:
        推定した標準偏差。差が 2 つ未満なら値の標準偏差。
    """
    steps = np.diff(values)
    if steps.size < 2:
        return float(np.std(values))
    deviation = np.abs(steps - np.median(steps))
    mad = float(np.median(deviation))
    if mad > 0.0:
        return 1.4826 * mad / math.sqrt(2.0)
    return 1.2533 * float(np.mean(deviation)) / math.sqrt(2.0)


def _finite_pair(pair: AlignedPair) -> tuple[FloatArray, FloatArray, FloatArray]:
    """対応付けた組の時刻と値を取り出し、欠損がないことを確認する。

    Args:
        pair: 対応付けた入力と出力。

    Returns:
        (時刻, 入力, 出力)。

    Raises:
        AnalysisInputError: 時刻が不正、または値に欠損（NaN）がある場合。
    """
    times = validate_time_axis(pair.time_s)
    for name, values in (("入力", pair.input_values), ("出力", pair.output_values)):
        missing = ~np.isfinite(values)
        if missing.any():
            first = int(np.argmax(missing))
            raise AnalysisInputError(
                f"{name}に欠損（NaN）が {int(missing.sum())} 個あります（最初は時刻 "
                f"{times[first]:.6g} s）。欠損や欠損区間を含まない範囲を選んでください"
            )
    return times, pair.input_values, pair.output_values


def _smooth(times: FloatArray, values: FloatArray, cutoff_hz: float | None) -> FloatArray:
    """値にゼロ位相のローパスフィルタをかける（カットオフ周波数が None ならそのまま返す）。

    Args:
        times: 時刻 [s]（等間隔であること）。
        values: 値。
        cutoff_hz: カットオフ周波数 [Hz]。None なら平滑化しない。

    Returns:
        平滑化した値。

    Raises:
        AnalysisInputError: 時刻が等間隔でない、またはカットオフ周波数がナイキスト周波数以上の場合。
    """
    if cutoff_hz is None:
        return values
    lowpass = ButterworthParams(
        kind=ButterworthKind.LOWPASS, order=SMOOTHING_ORDER, cutoff_hz=cutoff_hz
    )
    return ButterworthFilter(lowpass).apply(times, values)
