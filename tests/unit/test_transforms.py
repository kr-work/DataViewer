"""analysis/transforms.py（値の変換・時間の処理・区間統計）のテスト。

既定の条件のまま、解析的に答えが分かる信号で確かめる。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from experiment_workbench.analysis import (
    AnalysisInputError,
    Detrender,
    GapFiller,
    LinearScaler,
    Normalizer,
    OffsetRemover,
    Resampler,
    StatisticsCalculator,
    TimeCropper,
)
from experiment_workbench.model import (
    CropParams,
    DetrendParams,
    FillGapsParams,
    InterpolationMethod,
    NormalizeMode,
    NormalizeParams,
    OffsetMode,
    OffsetParams,
    ResampleParams,
    ScaleParams,
    StatisticsParams,
)

RATE = 1000.0
TIME = np.arange(4000, dtype=np.float64) / RATE
"""テスト用の時刻（1 kHz、4 秒）。"""


def _with_gap(values: np.ndarray, first: int, last: int) -> np.ndarray:
    """指定した番号の区間を欠損にしたコピーを返す。

    Args:
        values: 値。
        first: 欠損の先頭の番号。
        last: 欠損の末尾の番号（含む）。

    Returns:
        欠損を入れた値。
    """
    copied = values.copy()
    copied[first : last + 1] = np.nan
    return copied


# --- オフセット除去 -----------------------------------------------------------------


def test_offset_default_removes_whole_mean() -> None:
    """既定では全区間の平均値を差し引く。"""
    x = 7.0 + np.sin(2 * np.pi * 5.0 * TIME)
    result = OffsetRemover(OffsetParams()).apply(TIME, x)
    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result, x - np.mean(x))


def test_offset_uses_only_the_selected_section() -> None:
    """区間を指定すると、その区間の平均だけを基準にする（ゼロ点補正）。"""
    x = np.where(TIME < 1.0, 2.0, 10.0)
    remover = OffsetRemover(OffsetParams(start_s=0.0, end_s=0.9))
    assert remover.baseline(TIME, x) == pytest.approx(2.0)
    assert remover.apply(TIME, x)[-1] == pytest.approx(8.0)


def test_offset_can_use_given_value() -> None:
    """指定した値を差し引ける。"""
    params = OffsetParams(mode=OffsetMode.VALUE, value=3.0)
    np.testing.assert_allclose(OffsetRemover(params).apply(TIME, np.ones(4000)), -2.0)


def test_offset_ignores_missing_values_and_keeps_them() -> None:
    """欠損は平均に使わず、結果でも欠損のまま残す。"""
    x = _with_gap(np.full(4000, 5.0), 10, 20)
    result = OffsetRemover(OffsetParams()).apply(TIME, x)
    assert np.isnan(result[15])
    assert result[0] == pytest.approx(0.0)


def test_offset_section_without_data_is_refused() -> None:
    """平均をとる区間にデータがなければ、理由を示して断る。"""
    with pytest.raises(AnalysisInputError, match="区間"):
        OffsetRemover(OffsetParams(start_s=100.0, end_s=200.0)).apply(TIME, np.ones(4000))


# --- 線形変換 -----------------------------------------------------------------------


def test_scale_applies_gain_and_offset() -> None:
    """Y = a x + b を計算する。"""
    x = np.linspace(-1.0, 1.0, 4000)
    np.testing.assert_allclose(
        LinearScaler(ScaleParams(gain=9.8, offset=1.0)).apply(TIME, x), 9.8 * x + 1.0
    )


def test_scale_refuses_zero_gain() -> None:
    """倍率 0 は値の情報が失われるので指定できない。"""
    with pytest.raises(ValidationError):
        ScaleParams(gain=0.0)


# --- トレンド除去 -------------------------------------------------------------------


def test_detrend_default_removes_straight_line() -> None:
    """既定（1 次）で直線のトレンドを除去し、もとの波形が残る。

    整数周期の余弦波は、区間の中央について偶関数なので直線と直交し、当てはめに影響しない。
    （正弦波は直線と直交しないので、当てはめた傾きがわずかに動く）
    """
    wave = np.cos(2 * np.pi * 5.0 * (TIME + 0.5 / RATE))
    result = Detrender(DetrendParams()).apply(TIME, 3.0 + 2.0 * TIME + wave)
    np.testing.assert_allclose(result, wave, atol=1e-6)


def test_detrend_removes_quadratic_with_large_time_origin() -> None:
    """大きな時刻の値でも、2 次のトレンドを精度よく除去する。"""
    t = 1.7e9 + TIME
    x = 1.0 + 0.5 * TIME + 0.25 * TIME**2
    np.testing.assert_allclose(Detrender(DetrendParams(order=2)).apply(t, x), 0.0, atol=1e-5)


def test_detrend_skips_missing_values_in_the_fit() -> None:
    """欠損があっても残りの値で当てはめ、欠損は欠損のまま残す。"""
    x = _with_gap(2.0 * TIME, 100, 300)
    result = Detrender(DetrendParams()).apply(TIME, x)
    assert np.isnan(result[200])
    assert float(np.nanmax(np.abs(result))) == pytest.approx(0.0, abs=1e-9)


# --- 正規化 -------------------------------------------------------------------------


def test_normalize_default_is_z_score() -> None:
    """既定は Z スコア（平均 0、標準偏差 1）。"""
    x = 5.0 + 3.0 * np.sin(2 * np.pi * 5.0 * TIME)
    result = Normalizer(NormalizeParams()).apply(TIME, x)
    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(result, ddof=1)) == pytest.approx(1.0)


def test_normalize_min_max_and_peak() -> None:
    """最小-最大は 0〜1、ピークは最大の絶対値が 1 になる。"""
    x = np.linspace(-2.0, 6.0, 4000)
    scaled = Normalizer(NormalizeParams(mode=NormalizeMode.MIN_MAX)).apply(TIME, x)
    assert (float(scaled.min()), float(scaled.max())) == pytest.approx((0.0, 1.0))
    peak = Normalizer(NormalizeParams(mode=NormalizeMode.PEAK)).apply(TIME, x)
    assert float(np.max(np.abs(peak))) == pytest.approx(1.0)
    assert float(peak[0]) == pytest.approx(-2.0 / 6.0)


def test_normalize_constant_is_refused() -> None:
    """値が一定なら割れないので、理由を示して断る。"""
    with pytest.raises(AnalysisInputError, match="一定"):
        Normalizer(NormalizeParams()).apply(TIME, np.ones(4000))


# --- 切り出し -----------------------------------------------------------------------


def test_crop_keeps_original_time_axis() -> None:
    """切り出しても時刻を 0 に揃えない（他のチャンネルと比べられるように）。"""
    x = np.sin(TIME)
    t_cut, x_cut = TimeCropper(CropParams(start_s=1.0, end_s=2.0)).apply(TIME, x)
    assert float(t_cut[0]) == pytest.approx(1.0)
    assert float(t_cut[-1]) == pytest.approx(2.0)
    np.testing.assert_array_equal(x_cut, x[1000:2001])


def test_crop_outside_data_is_refused() -> None:
    """データのない範囲は、理由を示して断る。"""
    with pytest.raises(AnalysisInputError, match="サンプル"):
        TimeCropper(CropParams(start_s=100.0, end_s=200.0)).apply(TIME, np.sin(TIME))


def test_crop_refuses_reversed_range() -> None:
    """開始が終了以降の範囲は指定できない。"""
    with pytest.raises(ValidationError):
        CropParams(start_s=2.0, end_s=1.0)


# --- リサンプリング -----------------------------------------------------------------


def test_resample_default_preserves_slow_signal() -> None:
    """低い周波数の信号は、リサンプリングしても値が変わらない（端を除く）。"""
    x = np.sin(2 * np.pi * 2.0 * TIME)
    result = Resampler(ResampleParams(rate_hz=200.0)).apply(TIME, x)
    assert result.time_s.size == 800
    assert float(result.time_s[1] - result.time_s[0]) == pytest.approx(0.005)
    expected = np.sin(2 * np.pi * 2.0 * result.time_s)
    np.testing.assert_allclose(result.values[50:-50], expected[50:-50], atol=1e-3)
    assert result.anti_aliased


def test_resample_default_removes_components_that_would_alias() -> None:
    """既定ではアンチエイリアスフィルタで、折り返す成分を取り除く。"""
    x = np.sin(2 * np.pi * 270.0 * TIME)  # 200 Hz のナイキスト周波数を超え、70 Hz に折り返す
    filtered = Resampler(ResampleParams(rate_hz=200.0)).apply(TIME, x)
    raw = Resampler(ResampleParams(rate_hz=200.0, anti_alias=False)).apply(TIME, x)
    assert float(np.std(filtered.values[50:-50])) < 0.001
    assert float(np.std(raw.values)) > 0.3  # フィルタなしでは偽の波形が残る
    assert any("折り返" in note for note in raw.notes)


def test_resample_up_does_not_filter_and_says_so() -> None:
    """周波数を上げるときはフィルタをかけず、情報は増えないことを知らせる。"""
    x = np.sin(2 * np.pi * 2.0 * TIME)
    result = Resampler(ResampleParams(rate_hz=5000.0)).apply(TIME, x)
    assert not result.anti_aliased
    assert any("情報は増えません" in note for note in result.notes)


def test_resample_does_not_bridge_missing_time() -> None:
    """時刻の飛びをまたいで補間せず、外挿もしない。"""
    keep = np.ones(4000, dtype=np.bool_)
    keep[1000:2000] = False
    t, x = TIME[keep], np.sin(2 * np.pi * 2.0 * TIME[keep])
    result = Resampler(ResampleParams(rate_hz=500.0)).apply(t, x)
    inside = (result.time_s > 1.05) & (result.time_s < 1.95)
    assert np.isnan(result.values[inside]).all()
    assert float(result.time_s[-1]) <= float(t[-1])
    assert any("欠損区間" in note for note in result.notes)
    assert any("アンチエイリアス" in note for note in result.notes)  # 不等間隔ではかけられない


def test_resample_keeps_nan_samples_as_missing() -> None:
    """欠損の値を補間で埋めない。"""
    x = _with_gap(np.sin(2 * np.pi * 2.0 * TIME), 1000, 1500)
    result = Resampler(ResampleParams(rate_hz=500.0)).apply(TIME, x)
    middle = (result.time_s > 1.1) & (result.time_s < 1.4)
    assert np.isnan(result.values[middle]).all()


def test_resample_with_pchip_follows_curve_better_than_linear() -> None:
    """PCHIP 補間は、粗いデータの曲線を線形補間より精度よく補間する。"""
    t = np.arange(0.0, 2.0, 0.05)
    x = np.sin(2 * np.pi * 0.5 * t)
    linear = Resampler(ResampleParams(rate_hz=100.0)).apply(t, x)
    pchip = Resampler(ResampleParams(rate_hz=100.0, method=InterpolationMethod.PCHIP)).apply(t, x)
    truth = np.sin(2 * np.pi * 0.5 * linear.time_s)
    assert float(np.max(np.abs(pchip.values - truth))) < float(
        np.max(np.abs(linear.values - truth))
    )


# --- 欠損の補間 ---------------------------------------------------------------------


def test_fill_default_interpolates_and_records_filled_samples() -> None:
    """既定で線形補間し、埋めた位置を記録する。"""
    truth = 2.0 * TIME + 1.0
    result = GapFiller(FillGapsParams()).apply(TIME, _with_gap(truth, 100, 110))
    np.testing.assert_allclose(result.values, truth)
    assert (result.n_missing, result.n_filled) == (11, 11)
    assert result.filled_mask[105]
    assert not result.filled_mask[99]


def test_fill_does_not_extrapolate_at_the_ends() -> None:
    """先頭と末尾の欠損は外挿になるので埋めない。"""
    x = _with_gap(_with_gap(TIME.copy(), 0, 9), 3990, 3999)
    result = GapFiller(FillGapsParams()).apply(TIME, x)
    assert np.isnan(result.values[0])
    assert np.isnan(result.values[-1])
    assert result.n_filled == 0
    assert result.n_missing == 20


def test_fill_respects_maximum_length() -> None:
    """上限より長い欠損は埋めない（存在しない波形を作らないため）。"""
    x = _with_gap(_with_gap(TIME.copy(), 100, 104), 1000, 1499)
    result = GapFiller(FillGapsParams(max_gap_s=0.1)).apply(TIME, x)
    assert not np.isnan(result.values[102])
    assert np.isnan(result.values[1200])
    assert (result.n_missing, result.n_filled) == (505, 5)


def test_fill_without_missing_returns_same_values() -> None:
    """欠損がなければ値は変わらない。"""
    x = np.sin(TIME)
    result = GapFiller(FillGapsParams()).apply(TIME, x)
    np.testing.assert_array_equal(result.values, x)
    assert result.n_missing == 0


def test_fill_reports_time_jumps_that_cannot_be_filled() -> None:
    """行ごと欠けた区間は NaN にならないので、飛びの数と長さを結果で知らせる。"""
    keep = np.ones(4000, dtype=np.bool_)
    keep[1000:1500] = False
    keep[3000:3100] = False
    t, x = TIME[keep], np.sin(TIME[keep])
    result = GapFiller(FillGapsParams()).apply(t, x)
    assert result.n_missing == 0
    assert result.n_time_gaps == 2
    assert result.longest_time_gap_s == pytest.approx(501 / RATE, rel=1e-9)
    np.testing.assert_array_equal(result.values, x)  # 時刻を増やさないので値は変わらない


def test_resample_with_large_gap_factor_bridges_time_jump() -> None:
    """「欠損とみなす間隔の倍率」を大きくすると、時刻の飛びも補間できる（結果の文の案内どおり）。"""
    keep = np.ones(4000, dtype=np.bool_)
    keep[1000:1500] = False
    t, x = TIME[keep], 2.0 * TIME[keep] + 1.0
    result = Resampler(ResampleParams(rate_hz=RATE, max_gap_factor=1000.0)).apply(t, x)
    assert not np.isnan(result.values).any()
    np.testing.assert_allclose(result.values, 2.0 * result.time_s + 1.0, atol=1e-9)


def test_fill_does_not_change_input() -> None:
    """入力の配列を変更しない。"""
    x = _with_gap(TIME.copy(), 100, 110)
    GapFiller(FillGapsParams()).apply(TIME, x)
    assert np.isnan(x[105])


# --- 区間統計 -----------------------------------------------------------------------


def test_statistics_of_sine_match_theory() -> None:
    """整数周期の正弦波の平均は 0、RMS は振幅/√2、ピーク間振幅は振幅の 2 倍になる。"""
    amplitude = 3.0
    x = amplitude * np.sin(2 * np.pi * 5.0 * TIME)
    stats = StatisticsCalculator(StatisticsParams()).compute(TIME, x)
    assert stats.mean == pytest.approx(0.0, abs=1e-9)
    assert stats.rms == pytest.approx(amplitude / np.sqrt(2.0), rel=1e-6)
    assert stats.std == pytest.approx(amplitude / np.sqrt(2.0), rel=1e-3)
    assert stats.maximum == pytest.approx(amplitude, rel=1e-4)
    assert stats.minimum == pytest.approx(-amplitude, rel=1e-4)
    assert stats.maximum_time_s == pytest.approx(0.05, abs=2e-3)


def test_statistics_use_only_the_section_and_report_missing() -> None:
    """区間の中だけを集計し、欠損は除いて数を記録する。"""
    x = _with_gap(np.where(TIME < 2.0, 1.0, 100.0), 100, 109)
    stats = StatisticsCalculator(StatisticsParams(start_s=0.0, end_s=1.0)).compute(TIME, x)
    assert stats.mean == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(1.0)
    assert stats.n_missing == 10
    assert stats.n_samples == 991


def test_statistics_histogram_counts_all_samples() -> None:
    """ヒストグラムの度数の合計は、使ったサンプル数と一致する。"""
    x = np.random.default_rng(0).normal(size=4000)
    stats = StatisticsCalculator(StatisticsParams(n_bins=20)).compute(TIME, x)
    assert float(np.sum(stats.histogram_counts)) == stats.n_samples
    assert stats.histogram_centers.size == 20


def test_statistics_without_data_is_refused() -> None:
    """区間に有効な値がなければ、理由を示して断る。"""
    with pytest.raises(AnalysisInputError, match="区間"):
        StatisticsCalculator(StatisticsParams(start_s=100.0)).compute(TIME, np.ones(4000))
