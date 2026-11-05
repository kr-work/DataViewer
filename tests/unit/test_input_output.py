"""analysis/input_output.py（相互相関・ステップ応答・ヒステリシス、SPEC A-09・A-10）のテスト。

答えが解析的に分かる信号で確かめる。
    - 相互相関: 滑らかな信号を正確に遅らせた出力（遅れはサンプル間隔の途中の値にする）
    - ステップ応答: 一次遅れ＋むだ時間（遅れ・立ち上がり・時定数・整定の理論値）と、
      2 次系（オーバーシュートの理論値 exp(-ζπ/√(1-ζ²))）
    - ヒステリシス: 幅 h の遊び（バックラッシュ）の静的なループ（分岐の差は K·h）、
      2 次の非線形性（回帰直線からのずれ）、一次遅れによる速さに依存するループ
入力のステップはサンプルの間で起こるため、入力の変化の開始は 1 サンプル以内の精度で決まる。
"""

import math

import numpy as np
import pytest

from experiment_workbench.analysis import (
    AnalysisInputError,
    ChannelAligner,
    CrossCorrelationAnalyzer,
    HysteresisAnalyzer,
    StepResponseAnalyzer,
)
from experiment_workbench.model import (
    AlignedPair,
    AlignmentParams,
    CrossCorrelationParams,
    DetrendMode,
    HysteresisParams,
    HysteresisResult,
    OnsetThresholdMode,
    StepDirection,
    StepResponseParams,
    StepResponseResult,
)
from experiment_workbench.validation import FloatArray

FS = 1000.0
"""テスト用の信号のサンプリング周波数 [Hz]。"""
DT = 1.0 / FS
"""サンプル間隔 [s]。"""
STEP_TIME = 0.5
"""入力のステップの時刻 [s]（サンプルの時刻に一致する。直前のサンプルは変化前の値）。"""
INPUT_ONSET = STEP_TIME - 0.9 * DT
"""割合 0.1 のしきい値で判定される入力の変化の開始（直前のサンプルとの間の線形補間）。"""


def _times(duration_s: float) -> FloatArray:
    """0 秒から始まる等間隔の時刻を返す。

    Args:
        duration_s: 長さ [s]。

    Returns:
        時刻 [s]。
    """
    return np.arange(round(duration_s * FS), dtype=np.float64) / FS


def _pair(times: FloatArray, x: FloatArray, y: FloatArray) -> AlignedPair:
    """同じ時刻の入力と出力を、対応付けた組にする。

    Args:
        times: 時刻 [s]。
        x: 入力。
        y: 出力。

    Returns:
        対応付けた組。
    """
    return ChannelAligner(AlignmentParams()).align(times, x, times, y, time_comparable=True)


def _step(times: FloatArray, before: float, after: float) -> FloatArray:
    """STEP_TIME で before から after に変わるステップ入力を返す。

    Args:
        times: 時刻 [s]。
        before: 変化前の値。
        after: 変化後の値。

    Returns:
        入力。
    """
    return np.where(times >= STEP_TIME, after, before)


def _fopdt(times: FloatArray, delay_s: float, time_constant_s: float) -> FloatArray:
    """一次遅れ＋むだ時間の単位ステップ応答（STEP_TIME の入力に対して 0 から 1 に向かう）。

    Args:
        times: 時刻 [s]。
        delay_s: むだ時間 L [s]。
        time_constant_s: 時定数 τ [s]。

    Returns:
        応答。
    """
    elapsed = np.maximum(times - STEP_TIME - delay_s, 0.0)
    return 1.0 - np.exp(-elapsed / time_constant_s)


def _second_order(times: FloatArray, natural_hz: float, damping: float) -> FloatArray:
    """2 次系（減衰振動）の単位ステップ応答。

    Args:
        times: 時刻 [s]。
        natural_hz: 固有振動数 [Hz]。
        damping: 減衰比 ζ（0 < ζ < 1）。

    Returns:
        応答。
    """
    elapsed = np.maximum(times - STEP_TIME, 0.0)
    omega = 2 * math.pi * natural_hz
    damped = omega * math.sqrt(1 - damping**2)
    decay = np.exp(-damping * omega * elapsed)
    ratio = damping / math.sqrt(1 - damping**2)
    response = 1.0 - decay * (np.cos(damped * elapsed) + ratio * np.sin(damped * elapsed))
    return np.where(times >= STEP_TIME, response, 0.0)


def _triangle(times: FloatArray, period_s: float, low: float, high: float) -> FloatArray:
    """最小値から始まり、最大値との間を往復する三角波を返す。

    Args:
        times: 時刻 [s]。
        period_s: 周期 [s]。
        low: 最小値。
        high: 最大値。

    Returns:
        三角波。
    """
    phase = np.mod(times / period_s, 1.0)
    return low + (high - low) * (1.0 - np.abs(2.0 * phase - 1.0))


def _play(values: FloatArray, width: float) -> FloatArray:
    """幅 width の遊び（バックラッシュ）を通した値を返す。

    入力が向きを変えると、出力は入力が width だけ動くまで止まる。最初は増加してきた状態とする。

    Args:
        values: 入力。
        width: 遊びの幅。

    Returns:
        遊びを通した値。
    """
    state = float(values[0]) - width / 2
    output = np.empty_like(values)
    for index, value in enumerate(values):
        state = min(max(state, value - width / 2), value + width / 2)
        output[index] = state
    return output


def _first_order_lag(values: FloatArray, time_constant_s: float) -> FloatArray:
    """一次遅れ（時定数 τ）を、区間ごとに入力を一定とみなして厳密に離散化して通す。

    入力が一定の区間では厳密な応答になる（ランプ入力では、半サンプル分の遅れが加わる）。

    Args:
        values: 入力。
        time_constant_s: 時定数 τ [s]。

    Returns:
        出力（最初は入力と等しい定常状態）。
    """
    decay = math.exp(-DT / time_constant_s)
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, values.size):
        output[index] = decay * output[index - 1] + (1 - decay) * values[index - 1]
    return output


def _smooth_random(times: FloatArray, delay_s: float = 0.0, seed: int = 0) -> FloatArray:
    """周波数 0.3〜15 Hz の正弦波 12 成分の和を、delay_s だけ遅らせて返す（遅れの答えが厳密）。

    Args:
        times: 時刻 [s]。
        delay_s: 遅れ [s]。
        seed: 成分の乱数シード（同じシードなら同じ信号）。

    Returns:
        信号。
    """
    rng = np.random.default_rng(seed)
    total = np.zeros_like(times)
    for _ in range(12):
        frequency, amplitude = rng.uniform(0.3, 15.0), rng.uniform(0.2, 1.0)
        phase = rng.uniform(0.0, 2 * math.pi)
        total += amplitude * np.sin(2 * math.pi * frequency * (times - delay_s) + phase)
    return total


# --- 相互相関 ------------------------------------------------------------------


def _cross(
    min_lag_s: float = -0.05, max_lag_s: float = 0.05, **changes: object
) -> CrossCorrelationAnalyzer:
    """条件を指定した相互相関の解析を作る。

    Args:
        min_lag_s: 探す範囲の下限 [s]。
        max_lag_s: 探す範囲の上限 [s]。
        **changes: その他の条件。

    Returns:
        相互相関の解析。
    """
    params = CrossCorrelationParams.model_validate(
        {"min_lag_s": min_lag_s, "max_lag_s": max_lag_s} | changes
    )
    return CrossCorrelationAnalyzer(params)


@pytest.mark.parametrize("delay_s", [0.0123, -0.0207, 0.0])
def test_cross_correlation_recovers_delay_finer_than_sample(delay_s: float) -> None:
    """出力を正確に遅らせた信号から、遅れをサンプル間隔より細かく推定する（先行も負で求まる）。"""
    times = _times(4.0)
    result = _cross().analyze(_pair(times, _smooth_random(times), _smooth_random(times, delay_s)))
    assert result.delay_s == pytest.approx(delay_s, abs=0.2 * DT)
    assert result.peak_coefficient > 0.99
    assert not result.at_search_edge
    assert result.sample_interval_s == pytest.approx(DT)
    assert np.abs(result.coefficients).max() <= 1.0 + 1e-12


def test_inverted_output_gives_negative_peak() -> None:
    """出力が入力と逆向きに変化する場合は、相関係数が負のピークになり、そのことを注記する。"""
    times = _times(4.0)
    result = _cross().analyze(_pair(times, _smooth_random(times), -_smooth_random(times, 0.01)))
    assert result.delay_s == pytest.approx(0.01, abs=0.2 * DT)
    assert result.peak_coefficient < -0.99
    assert any("逆向き" in note for note in result.notes)


def test_peak_at_search_edge_is_flagged() -> None:
    """真の遅れが探索範囲の外なら、範囲の端を最大として報告し、範囲を広げるよう注記する。"""
    times = _times(4.0)
    result = _cross(max_lag_s=0.02).analyze(
        _pair(times, _smooth_random(times), _smooth_random(times, 0.04))
    )
    assert result.at_search_edge
    assert any("範囲の端" in note for note in result.notes)


def test_derivative_correlation_finds_delay_of_steps() -> None:
    """ステップ状の信号でも、変化の速さの相関を使うと遅れを正確に求められる。"""
    times = _times(3.0)
    x = np.tanh((times - 1.0) / 0.01)
    y = np.tanh((times - 1.0 - 0.0304) / 0.01)
    result = _cross(min_lag_s=0.0, max_lag_s=0.1, use_derivative=True).analyze(_pair(times, x, y))
    assert result.delay_s == pytest.approx(0.0304, abs=0.3 * DT)


def test_linear_detrend_removes_drift() -> None:
    """ゆっくりしたドリフトがあっても、直線の傾向を取り除けば遅れを求められる。"""
    times = _times(4.0)
    x = _smooth_random(times) + 3.0 * times
    y = _smooth_random(times, 0.015) - 2.0 * times
    result = _cross(detrend=DetrendMode.LINEAR).analyze(_pair(times, x, y))
    assert result.delay_s == pytest.approx(0.015, abs=0.5 * DT)


def test_lags_beyond_half_overlap_are_not_searched() -> None:
    """重なりがサンプル数の半分未満になる遅れは探さず、そのことを注記する。"""
    times = _times(1.0)
    result = _cross(min_lag_s=-0.8, max_lag_s=0.8).analyze(
        _pair(times, _smooth_random(times), _smooth_random(times, 0.01))
    )
    assert result.lags_s.max() <= 0.5 + 1e-9
    assert any("半分未満" in note for note in result.notes)


def test_cross_correlation_requires_uniform_sampling() -> None:
    """時刻が等間隔でなければ、相関を求めない（等間隔に対応付けるよう促す）。"""
    times = np.cumsum(np.random.default_rng(0).uniform(0.0005, 0.0015, 2000))
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        _cross().analyze(_pair(times, _smooth_random(times), _smooth_random(times)))


def test_missing_values_are_rejected() -> None:
    """欠損（NaN）を含む範囲では計算しない。"""
    times = _times(1.0)
    y = _smooth_random(times)
    y[100] = np.nan
    with pytest.raises(AnalysisInputError, match="欠損"):
        _cross().analyze(_pair(times, _smooth_random(times), y))


# --- ステップ応答 ----------------------------------------------------------------


def _step_result(
    times: FloatArray, u: FloatArray, y: FloatArray, **changes: object
) -> StepResponseResult:
    """ステップ応答の解析を実行する（テストの記述を短くするため）。

    Args:
        times: 時刻 [s]。
        u: 入力。
        y: 出力。
        **changes: StepResponseParams の項目と値。

    Returns:
        ステップ応答の指標。
    """
    params = StepResponseParams.model_validate(changes)
    return StepResponseAnalyzer(params).analyze(_pair(times, u, y))


def test_first_order_with_dead_time_matches_theory() -> None:
    """一次遅れ＋むだ時間の応答から、遅れ・立ち上がり・時定数・むだ時間・整定時間が理論値で求まる。"""
    delay, tau = 0.05, 0.1
    times = _times(2.0)
    result = _step_result(times, _step(times, 0.0, 10.0), 1.0 + 5.0 * _fopdt(times, delay, tau))
    start = STEP_TIME - INPUT_ONSET  # 入力の変化の開始から、真のステップの時刻まで
    assert result.direction is StepDirection.RISING
    assert result.input_onset_s == pytest.approx(INPUT_ONSET, abs=1e-9)
    assert result.gain == pytest.approx(0.5, rel=1e-4)
    assert result.response_delay_s == pytest.approx(
        start + delay + tau * math.log(1 / 0.9), abs=1e-5
    )
    assert result.delay_time_s == pytest.approx(start + delay + tau * math.log(2), abs=1e-5)
    assert result.rise_time_s == pytest.approx(tau * math.log(9), abs=1e-5)
    assert result.time_constant_s == pytest.approx(tau, abs=1e-4)
    assert result.dead_time_s == pytest.approx(start + delay, abs=1e-4)
    assert result.settling_time_s == pytest.approx(start + delay + tau * math.log(20), abs=1e-4)
    assert result.overshoot == pytest.approx(0.0, abs=1e-4)
    assert result.notes == []


def test_falling_step_gives_same_times() -> None:
    """立ち下がりでも、正規化して同じ時間の指標が求まり、向きは減少と判定される。"""
    delay, tau = 0.05, 0.1
    times = _times(2.0)
    result = _step_result(times, _step(times, 10.0, 0.0), 6.0 - 5.0 * _fopdt(times, delay, tau))
    assert result.direction is StepDirection.FALLING
    assert result.gain == pytest.approx(0.5, rel=1e-4)
    assert result.time_constant_s == pytest.approx(tau, abs=1e-4)
    assert result.rise_time_s == pytest.approx(tau * math.log(9), abs=1e-5)


def test_second_order_overshoot_matches_theory() -> None:
    """2 次系のオーバーシュートは理論値 exp(-ζπ/√(1-ζ²)) と一致する。"""
    damping = 0.3
    times = _times(3.0)
    result = _step_result(times, _step(times, 0.0, 1.0), _second_order(times, 2.0, damping))
    expected = math.exp(-damping * math.pi / math.sqrt(1 - damping**2))
    assert result.overshoot == pytest.approx(expected, abs=2e-3)
    assert result.settling_time_s is not None


def test_absolute_threshold_uses_input_units() -> None:
    """しきい値を入力の単位で指定すると、その大きさの変化で開始を判定する（出力も同じ割合）。"""
    delay, tau = 0.05, 0.1
    times = _times(2.0)
    result = _step_result(
        times,
        _step(times, 0.0, 10.0),
        5.0 * _fopdt(times, delay, tau),
        onset_mode=OnsetThresholdMode.ABSOLUTE,
        onset_threshold=2.0,
    )
    onset = STEP_TIME - 0.8 * DT  # 割合 0.2 の位置で越える
    assert result.input_onset_s == pytest.approx(onset, abs=1e-9)
    expected = STEP_TIME + delay + tau * math.log(1 / 0.8) - onset
    assert result.response_delay_s == pytest.approx(expected, abs=1e-5)


def test_noisy_output_still_gives_close_metrics() -> None:
    """出力に雑音があっても、中点を基準に越えた時刻を探すので、指標は理論値に近い。"""
    delay, tau = 0.05, 0.1
    times = _times(2.0)
    noise = np.random.default_rng(3).normal(0.0, 0.05, times.size)  # 出力の変化量の 1%
    result = _step_result(
        times, _step(times, 0.0, 10.0), 1.0 + 5.0 * _fopdt(times, delay, tau) + noise
    )
    assert result.time_constant_s == pytest.approx(tau, rel=0.1)
    assert result.rise_time_s == pytest.approx(tau * math.log(9), rel=0.1)
    assert result.delay_time_s == pytest.approx(0.0009 + delay + tau * math.log(2), abs=0.005)


def test_smoothing_is_applied_to_both_channels() -> None:
    """平滑化の条件はデータで確かめられる（ナイキスト周波数以上のカットオフはエラー）。"""
    times = _times(2.0)
    with pytest.raises(AnalysisInputError, match="ナイキスト"):
        _step_result(
            times, _step(times, 0.0, 1.0), _fopdt(times, 0.05, 0.1), smoothing_cutoff_hz=600.0
        )


@pytest.mark.parametrize(
    ("step_value", "changes", "message"),
    [
        (0.0, {}, "区別できません"),  # 入力が変化していない
        (10.0, {"direction": StepDirection.FALLING}, "向き"),
        (10.0, {"onset_mode": OnsetThresholdMode.ABSOLUTE, "onset_threshold": 20.0}, "以上"),
        (10.0, {"baseline_fraction": 0.3}, "初期値を求める区間"),
    ],
)
def test_unsuitable_step_settings_are_reported(
    step_value: float, changes: dict[str, object], message: str
) -> None:
    """入力が変化しない・向きが違う・しきい値が大きすぎる・変化が初期値の区間の中にある場合は断る。"""
    times = _times(2.0)
    with pytest.raises(AnalysisInputError, match=message):
        _step_result(times, _step(times, 0.0, step_value), _fopdt(times, 0.05, 0.1), **changes)


def test_step_inside_final_window_is_reported() -> None:
    """入力の変化が最終値を求める区間（範囲の末尾）の中にあれば断る。"""
    times = _times(2.0)
    late_step = np.where(times >= 1.9, 1.0, 0.0)
    with pytest.raises(AnalysisInputError, match="最終値を求める区間"):
        _step_result(times, late_step, late_step)


def test_unsettled_output_has_no_settling_time() -> None:
    """範囲の終わりまでに整定しない出力は、整定時間を求めず理由を注記する。"""
    times = _times(2.0)
    result = _step_result(times, _step(times, 0.0, 1.0), _fopdt(times, 0.05, 2.0))
    assert result.settling_time_s is None
    assert any("整定" in note for note in result.notes)


def test_unchanged_output_has_no_time_metrics() -> None:
    """出力が変化していなければ、時間の指標は求めず、その理由を注記する。"""
    times = _times(2.0)
    flat = 1.0 + np.random.default_rng(0).normal(0.0, 0.01, times.size)
    result = _step_result(times, _step(times, 0.0, 1.0), flat)
    assert result.delay_time_s is None
    assert result.time_constant_s is None
    assert np.isnan(result.normalized_output).all()
    assert any("区別できない" in note for note in result.notes)


# --- ヒステリシス ----------------------------------------------------------------


def _hysteresis(
    times: FloatArray, u: FloatArray, y: FloatArray, **changes: object
) -> HysteresisResult:
    """ヒステリシスの解析を実行する（テストの記述を短くするため）。

    Args:
        times: 時刻 [s]。
        u: 入力。
        y: 出力。
        **changes: HysteresisParams の項目と値。

    Returns:
        入出力特性とヒステリシスの指標。
    """
    return HysteresisAnalyzer(HysteresisParams.model_validate(changes)).analyze(_pair(times, u, y))


def _play_regression_slope(gain: float, width: float, span: float) -> float:
    """入力が 0〜span を一様に往復するときの、遊びのループへの回帰直線の傾きを数値積分で求める。

    増加時の分岐は、折り返し直後の入力 width の間は平ら（width/2）、その後は u - width/2。
    減少時の分岐は対称なので、共分散は増加時の分岐だけで求まる（一様分布の分散は span²/12）。

    Args:
        gain: 遊びの後の倍率 K。
        width: 遊びの幅 h。
        span: 入力の範囲。

    Returns:
        回帰直線の傾き。
    """
    u = np.linspace(0.0, span, 200_001)
    branch = np.where(u < width, width / 2, u - width / 2)
    covariance = float(np.trapezoid((u - span / 2) * branch, u)) / span
    return gain * covariance / (span**2 / 12)


def test_play_loop_gives_known_hysteresis() -> None:
    """幅 h の遊びのループでは、分岐の差は K·h、面積は K·h·(範囲 - h)、回帰直線は理論どおり。"""
    gain, offset, width, span = 0.5, 1.0, 1.0, 10.0
    all_times = _times(10.0)
    all_input = _triangle(all_times, 2.0, 0.0, span)
    all_output = offset + gain * _play(all_input, width)
    # 最初の 1 周期は、遊びの初期状態の影響で折り返し直後の平らな部分がないので除く。
    steady = all_times >= 2.0
    times, u = all_times[steady], all_input[steady]
    result = _hysteresis(times, u, all_output[steady])
    slope = _play_regression_slope(gain, width, span)
    assert result.max_hysteresis == pytest.approx(gain * width, rel=0.01)
    assert result.slope == pytest.approx(slope, rel=0.005)
    # ループは中心（span/2, offset + gain·span/2）について対称なので、回帰直線はそこを通る。
    assert result.intercept == pytest.approx(offset + (gain - slope) * span / 2, abs=0.01)
    assert result.hysteresis_ratio == pytest.approx(gain * width / (slope * span), rel=0.01)
    # 分岐の差は、中央の span - 2h の範囲で K·h、両端の幅 h の範囲で 0 まで直線的に減るので、
    # 面積は K·h·(span - 2h) + 2 × (K·h²/2) = K·h·(span - h)。
    assert result.loop_area == pytest.approx(gain * width * (span - width), rel=0.01)
    assert result.mean_branch_difference is not None
    assert result.mean_branch_difference < 0  # 増加時の分岐が下にある
    assert result.n_increasing > 0
    assert result.n_decreasing > 0


def test_quadratic_characteristic_gives_linearity_error() -> None:
    """2 次の非線形性は、回帰直線からのずれ（直線性誤差）として現れ、ヒステリシスはほぼ 0。"""
    times = _times(8.0)
    u = _triangle(times, 2.0, 0.0, 10.0)
    result = _hysteresis(times, u, u + 0.01 * u**2)
    # 一様に分布する u ∈ [0, 10] への u² の最小二乗直線は 10u - 50/3。
    assert result.slope == pytest.approx(1.1, rel=1e-3)
    assert result.intercept == pytest.approx(-1.0 / 6.0, abs=2e-3)
    center = 10.0 / 50 / 2  # 端の区間の中央
    expected = 0.01 * abs(center**2 - 10 * center + 50.0 / 3.0)
    assert result.linearity_error == pytest.approx(expected, rel=0.02)
    assert result.max_hysteresis is not None
    assert result.max_hysteresis < 1e-3


def test_first_order_lag_widens_loop_with_input_rate() -> None:
    """一次遅れの系では、定常なランプでの遅れ τ（＋半サンプル）の分だけ分岐が開く。"""
    tau = 0.05
    times = _times(16.0)
    u = _triangle(times, 4.0, 0.0, 10.0)
    rate = 2 * 10.0 / 4.0  # ランプの速さ [単位/s]
    result = _hysteresis(times, u, _first_order_lag(u, tau))
    assert result.max_hysteresis == pytest.approx(2 * rate * (tau + DT / 2), rel=0.02)


def test_dwell_samples_are_excluded_from_branches() -> None:
    """入力が止まっている（向きが定まらない）サンプルは、どちらの分岐にも入れない。"""
    times = _times(8.0)
    u = np.clip(_triangle(times, 2.0, -1.0, 11.0), 0.0, 10.0)  # 両端で止まる時間がある
    result = _hysteresis(times, u, u)
    assert result.n_excluded > 0.1 * times.size
    assert any("除きました" in note for note in result.notes)


def test_monotonic_input_has_no_hysteresis() -> None:
    """入力が往復しない範囲では、ヒステリシスを求めずに理由を注記する（回帰直線は求める）。"""
    times = _times(2.0)
    u = 5.0 * times
    result = _hysteresis(times, u, 2.0 * u + 1.0)
    assert result.max_hysteresis is None
    assert result.loop_area is None
    assert result.slope == pytest.approx(2.0)
    assert any("往復" in note for note in result.notes)


def test_flat_input_is_rejected() -> None:
    """入力が変化しなければ、増加時と減少時に分けられないので断る。"""
    times = _times(1.0)
    with pytest.raises(AnalysisInputError, match="変化していない"):
        _hysteresis(times, np.ones_like(times), times)


def test_smoothing_stabilizes_direction_of_noisy_input() -> None:
    """雑音の多い入力では、向きの判定にだけ平滑化を使うと、分岐の差を正しく求められる。"""
    gain, width = 0.5, 1.0
    times = _times(8.0)
    true_input = _triangle(times, 2.0, 0.0, 10.0)
    y = gain * _play(true_input, width)
    noisy = true_input + np.random.default_rng(5).normal(0.0, 0.05, times.size)
    smoothed = _hysteresis(times, noisy, y, smoothing_cutoff_hz=10.0)
    raw = _hysteresis(times, noisy, y)
    assert smoothed.max_hysteresis is not None
    assert smoothed.max_hysteresis == pytest.approx(gain * width, rel=0.1)
    assert raw.max_hysteresis is not None
    assert abs(raw.max_hysteresis - gain * width) > abs(smoothed.max_hysteresis - gain * width)
