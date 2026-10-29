"""analysis/signal_processing.py（入力検証・フィルタ・微分・積分）のテスト。

各処理は、答えが解析的に分かる信号（正弦波・多項式など）に対する精度で確認する。
"""

import numpy as np
import pytest

from experiment_workbench.analysis import (
    AnalysisInputError,
    ButterworthFilter,
    Differentiator,
    Integrator,
    MovingAverageFilter,
    SavitzkyGolayFilter,
    check_uniform_sampling,
    estimate_sampling_rate,
    validate_series,
    validate_time_axis,
)
from experiment_workbench.model import (
    ButterworthKind,
    ButterworthParams,
    DerivativeMethod,
    DerivativeParams,
    IntegralParams,
    IntegrationMethod,
    MovingAverageParams,
    SavitzkyGolayParams,
)
from experiment_workbench.validation import FloatArray

FS = 1000.0  # テスト信号のサンプリング周波数 [Hz]


def _uniform_time(duration_s: float = 4.0, fs: float = FS, start_s: float = 10.0) -> FloatArray:
    """等間隔の時刻配列を作る（原点を 0 以外にして、時刻原点に依存しないことも確かめる）。

    Args:
        duration_s: 長さ [s]。
        fs: サンプリング周波数 [Hz]。
        start_s: 先頭時刻 [s]。

    Returns:
        時刻 [s] の配列。
    """
    return start_s + np.arange(round(duration_s * fs), dtype=np.float64) / fs


def _sine(t: FloatArray, frequency_hz: float, amplitude: float = 1.0) -> FloatArray:
    """正弦波 A sin(2π f t) を返す。

    Args:
        t: 時刻 [s]。
        frequency_hz: 周波数 [Hz]。
        amplitude: 振幅。

    Returns:
        正弦波の値。
    """
    return amplitude * np.sin(2 * np.pi * frequency_hz * t)


def _middle(values: FloatArray, margin: int = 500) -> FloatArray:
    """端点の影響を除くため、前後 ``margin`` サンプルを除いた部分を返す。

    Args:
        values: 配列。
        margin: 前後から除くサンプル数。

    Returns:
        中央部分。
    """
    return values[margin:-margin]


# --- 入力検証（A-01） ---------------------------------------------------------


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 1.0, 0.5, 2.0], "逆行"),
        ([0.0, 1.0, 1.0, 2.0], "重複"),
        ([0.0, np.nan, 2.0], "NaN"),
        ([0.0, np.inf, 2.0], "NaN"),
        ([[0.0, 1.0], [2.0, 3.0]], "1 次元"),
        ([0.0], "2 以上"),
    ],
)
def test_validate_time_axis_reports_problems(
    times: list[float] | list[list[float]], message: str
) -> None:
    """時刻の逆行・重複・非有限値・次元・長さの問題は、原因が分かるエラーになる。"""
    with pytest.raises(AnalysisInputError, match=message):
        validate_time_axis(np.asarray(times, dtype=np.float64))


def test_validate_series_reports_nan_values_with_position() -> None:
    """値の NaN は、個数と最初の位置を示すエラーになる（黙って除去しない）。"""
    t = _uniform_time(0.01)
    x = np.zeros_like(t)
    x[[3, 7]] = np.nan
    with pytest.raises(AnalysisInputError, match=r"2 個.*index: 3"):
        validate_series(t, x)


def test_validate_series_rejects_length_mismatch() -> None:
    """時刻と値の長さが違う場合はエラーになる。"""
    t = _uniform_time(0.01)
    with pytest.raises(AnalysisInputError, match="一致しません"):
        validate_series(t, np.zeros(t.size - 1))


def test_check_uniform_sampling_detects_missing_sample() -> None:
    """1 サンプルの欠損（間隔が 2 倍）も等間隔でないとして検出する。"""
    t = np.delete(_uniform_time(0.1), 50)
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        check_uniform_sampling(t)


def test_check_uniform_sampling_allows_small_jitter() -> None:
    """許容範囲内の時刻ジッタは等間隔とみなし、サンプリング周波数を返す。"""
    rng = np.random.default_rng(0)
    t = _uniform_time(1.0) + rng.uniform(-1e-5, 1e-5, round(FS))
    assert check_uniform_sampling(t) == pytest.approx(FS, rel=1e-3)


def test_estimate_sampling_rate_is_robust_to_a_gap() -> None:
    """推定周波数は間隔の中央値から求めるため、欠損区間があってもずれない。"""
    t = np.delete(_uniform_time(1.0), slice(100, 300))
    assert estimate_sampling_rate(t) == pytest.approx(FS)


# --- バターワースフィルタ（A-02） -------------------------------------------------


def test_lowpass_removes_high_frequency_component() -> None:
    """2 Hz + 50 Hz の信号に 10 Hz の LPF をかけると、2 Hz 成分だけが残る。"""
    t = _uniform_time()
    low = _sine(t, 2.0)
    params = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0)
    filtered = ButterworthFilter(params).apply(t, low + _sine(t, 50.0, 0.5))
    np.testing.assert_allclose(_middle(filtered), _middle(low), atol=1e-3)


def test_highpass_removes_offset() -> None:
    """直流オフセットに 20 Hz を重ねた信号に 2 Hz の HPF をかけると、オフセットが消える。

    低いカットオフの HPF は端点の過渡応答の減衰が遅い（時定数 約 0.2 秒）ため、
    信号を長めにして端の 2 秒を除いて比較する。
    """
    t = _uniform_time(8.0)
    high = _sine(t, 20.0)
    params = ButterworthParams(kind=ButterworthKind.HIGHPASS, cutoff_hz=2.0)
    filtered = ButterworthFilter(params).apply(t, 3.0 + high)
    np.testing.assert_allclose(_middle(filtered, 2000), _middle(high, 2000), atol=1e-3)


def test_bandpass_keeps_only_the_band() -> None:
    """1 Hz・20 Hz・200 Hz の和に 10〜40 Hz の BPF をかけると、20 Hz だけが残る。"""
    t = _uniform_time()
    band = _sine(t, 20.0)
    params = ButterworthParams(kind=ButterworthKind.BANDPASS, cutoff_hz=(10.0, 40.0))
    filtered = ButterworthFilter(params).apply(t, _sine(t, 1.0) + band + _sine(t, 200.0))
    np.testing.assert_allclose(_middle(filtered), _middle(band), atol=1e-2)


def test_bandstop_removes_the_band() -> None:
    """2 Hz + 50 Hz の信号に 40〜60 Hz の BSF をかけると、50 Hz だけが消える。"""
    t = _uniform_time()
    keep = _sine(t, 2.0)
    params = ButterworthParams(kind=ButterworthKind.BANDSTOP, cutoff_hz=(40.0, 60.0))
    filtered = ButterworthFilter(params).apply(t, keep + _sine(t, 50.0))
    np.testing.assert_allclose(_middle(filtered), _middle(keep), atol=1e-2)


def test_causal_filter_does_not_respond_before_a_step() -> None:
    """因果的フィルタはステップより前に応答せず、ゼロ位相フィルタはステップの前から応答する。"""
    t = _uniform_time(2.0)
    step = (t >= t[1000]).astype(np.float64)
    causal = ButterworthFilter(
        ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0, zero_phase=False)
    ).apply(t, step)
    zero_phase = ButterworthFilter(
        ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0)
    ).apply(t, step)
    assert np.all(causal[:1000] == 0.0)
    assert abs(zero_phase[990]) > 1e-3  # 未来の値を使うため、ステップの前から立ち上がる


def test_cutoff_at_or_above_nyquist_is_rejected() -> None:
    """カットオフ周波数がナイキスト周波数以上なら、設計せずにエラーにする。"""
    t = _uniform_time(1.0, fs=100.0)
    params = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=60.0)
    with pytest.raises(AnalysisInputError, match="ナイキスト"):
        ButterworthFilter(params).apply(t, np.zeros_like(t))


def test_filter_rejects_time_gap() -> None:
    """欠損区間がある信号には、等間隔を前提とするフィルタをかけない。"""
    t = np.delete(_uniform_time(1.0), slice(400, 450))
    params = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0)
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        ButterworthFilter(params).apply(t, np.zeros_like(t))


def test_zero_phase_filter_reports_too_short_signal() -> None:
    """端点の延長に必要な長さより短い信号は、原因が分かるエラーになる。"""
    t = _uniform_time(0.01)  # 10 サンプル
    params = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0)
    with pytest.raises(AnalysisInputError, match="短すぎ"):
        ButterworthFilter(params).apply(t, np.zeros_like(t))


def test_filter_does_not_modify_input() -> None:
    """入力配列は変更されない（非破壊）。"""
    t = _uniform_time(1.0)
    x = _sine(t, 5.0)
    original = x.copy()
    ButterworthFilter(ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=10.0)).apply(t, x)
    np.testing.assert_array_equal(x, original)


# --- 移動平均・Savitzky-Golay（A-02） ------------------------------------------


def test_centered_moving_average_keeps_linear_ramp_and_marks_edges_nan() -> None:
    """中心移動平均は直線を変えず、窓が揃わない端点は NaN にする。"""
    t = _uniform_time(0.1)
    ramp = 2.0 * t + 1.0
    averaged = MovingAverageFilter(MovingAverageParams(window_samples=5)).apply(t, ramp)
    assert np.isnan(averaged[:2]).all()
    assert np.isnan(averaged[-2:]).all()
    np.testing.assert_allclose(averaged[2:-2], ramp[2:-2], atol=1e-9)


def test_trailing_moving_average_lags_by_half_window() -> None:
    """後方移動平均は (窓長 - 1) / 2 サンプル遅れる（直線なら値がその分だけ小さい）。"""
    t = _uniform_time(0.1)
    params = MovingAverageParams(window_samples=5, centered=False)
    averaged = MovingAverageFilter(params).apply(t, t.copy())
    assert np.isnan(averaged[:4]).all()
    np.testing.assert_allclose(averaged[4:], t[4:] - 2.0 / FS, atol=1e-9)


def test_moving_average_reduces_white_noise() -> None:
    """窓長 w の移動平均は白色雑音の標準偏差を約 1/√w にする。"""
    t = _uniform_time(10.0)
    noise = np.random.default_rng(1).normal(0.0, 1.0, t.size)
    averaged = MovingAverageFilter(MovingAverageParams(window_samples=25)).apply(t, noise)
    assert np.nanstd(averaged) == pytest.approx(1.0 / 5.0, rel=0.1)


def test_moving_average_rejects_window_longer_than_signal() -> None:
    """窓長がサンプル数より長い場合はエラーになる。"""
    t = _uniform_time(0.005)
    with pytest.raises(AnalysisInputError, match="窓長"):
        MovingAverageFilter(MovingAverageParams(window_samples=7)).apply(t, np.zeros_like(t))


def test_savitzky_golay_preserves_polynomial_including_edges() -> None:
    """次数以下の多項式は、端点も含めて変化しない。"""
    t = _uniform_time(0.2, start_s=0.0)
    cubic = 3.0 * t**3 - 2.0 * t + 1.0
    params = SavitzkyGolayParams(window_samples=11, polyorder=3)
    np.testing.assert_allclose(SavitzkyGolayFilter(params).apply(t, cubic), cubic, atol=1e-9)


def test_savitzky_golay_reduces_white_noise() -> None:
    """平滑化により白色雑音の標準偏差が小さくなる。"""
    t = _uniform_time(5.0)
    noise = np.random.default_rng(2).normal(0.0, 1.0, t.size)
    params = SavitzkyGolayParams(window_samples=51, polyorder=2)
    assert SavitzkyGolayFilter(params).apply(t, noise).std() < 0.3


# --- 微分（A-03） ----------------------------------------------------------------


def _non_uniform_time(n: int = 3000, seed: int = 3) -> FloatArray:
    """間隔が 1〜2 ms でばらつく不等間隔の時刻配列を作る。

    Args:
        n: サンプル数。
        seed: 乱数シード。

    Returns:
        狭義単調増加の時刻 [s] の配列。
    """
    return np.cumsum(np.random.default_rng(seed).uniform(0.001, 0.002, n))


def test_gradient_on_non_uniform_time() -> None:
    """中心差分は不等間隔の時刻でも sin の微分 cos を精度よく求める。"""
    t = _non_uniform_time()
    derivative = Differentiator(DerivativeParams()).apply(t, np.sin(t))
    np.testing.assert_allclose(derivative, np.cos(t), atol=1e-4)


def test_second_derivative() -> None:
    """2 階微分で sin の 2 階微分 -sin を求める。"""
    t = _non_uniform_time()
    second = Differentiator(DerivativeParams(order=2)).apply(t, np.sin(t))
    np.testing.assert_allclose(second[5:-5], -np.sin(t[5:-5]), atol=1e-3)


def test_savitzky_golay_derivative_is_less_noisy_than_gradient() -> None:
    """ノイズのある信号では、Savitzky-Golay 微分の誤差が中心差分よりずっと小さい。"""
    t = _uniform_time(4.0)
    noisy = _sine(t, 1.0) + np.random.default_rng(4).normal(0.0, 0.001, t.size)
    truth = 2 * np.pi * np.cos(2 * np.pi * t)
    gradient = Differentiator(DerivativeParams()).apply(t, noisy)
    savgol_params = DerivativeParams(
        method=DerivativeMethod.SAVITZKY_GOLAY,
        savitzky_golay=SavitzkyGolayParams(window_samples=101, polyorder=3),
    )
    savgol = Differentiator(savgol_params).apply(t, noisy)
    gradient_rmse = np.sqrt(np.mean(_middle(gradient - truth) ** 2))
    savgol_rmse = np.sqrt(np.mean(_middle(savgol - truth) ** 2))
    assert savgol_rmse < gradient_rmse / 5


def test_savitzky_golay_derivative_requires_uniform_sampling() -> None:
    """Savitzky-Golay 微分は不等間隔の時刻を受け付けない。"""
    t = _non_uniform_time()
    params = DerivativeParams(
        method=DerivativeMethod.SAVITZKY_GOLAY,
        savitzky_golay=SavitzkyGolayParams(window_samples=11, polyorder=2),
    )
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        Differentiator(params).apply(t, np.sin(t))


def test_gradient_requires_enough_samples() -> None:
    """端点 2 次精度の中心差分には 3 サンプル以上が必要。"""
    with pytest.raises(AnalysisInputError, match="3 サンプル以上"):
        Differentiator(DerivativeParams()).apply(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# --- 積分（A-04） ----------------------------------------------------------------


def test_trapezoid_integral_of_cos_is_sin_plus_initial_value() -> None:
    """余弦 cos を積分すると sin になり、先頭には指定した初期値が入る。"""
    t = _uniform_time(2 * np.pi, start_s=0.0)
    integral = Integrator(IntegralParams(initial_value=5.0)).apply(t, np.cos(t))
    assert integral[0] == 5.0
    np.testing.assert_allclose(integral, 5.0 + np.sin(t), atol=1e-6)


def test_trapezoid_is_exact_for_linear_integrand_on_non_uniform_time() -> None:
    """台形則は 1 次関数の積分を不等間隔でも厳密に求める（∫2t dt = t² - t0²）。"""
    t = _non_uniform_time()
    integral = Integrator(IntegralParams()).apply(t, 2.0 * t)
    np.testing.assert_allclose(integral, t**2 - t[0] ** 2, atol=1e-9)


def test_simpson_is_more_accurate_than_trapezoid_on_coarse_grid() -> None:
    """粗い格子では、シンプソン則の誤差が台形則よりずっと小さい（∫sin = 1 - cos）。"""
    t = np.linspace(0.0, np.pi, 21)
    truth = 1.0 - np.cos(t)
    trapezoid = Integrator(IntegralParams()).apply(t, np.sin(t))
    simpson = Integrator(IntegralParams(method=IntegrationMethod.SIMPSON)).apply(t, np.sin(t))
    assert np.abs(simpson - truth).max() < np.abs(trapezoid - truth).max() / 10


def test_integrator_rejects_nan() -> None:
    """NaN を含む信号は積分せずにエラーにする（欠損を勝手に埋めない）。"""
    t = _uniform_time(0.01)
    x = np.ones_like(t)
    x[4] = np.nan
    with pytest.raises(AnalysisInputError, match="NaN"):
        Integrator(IntegralParams()).apply(t, x)
