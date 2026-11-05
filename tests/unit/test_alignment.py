"""analysis/alignment.py（2 チャンネルの時刻の対応付け、SPEC A-11）のテスト。

線形な信号は線形補間で誤差なく求まるので、補間した値は式の値と比べる。
外挿しないこと、欠損区間をまたがないこと、時刻の対応の明示を求めることを確かめる。
"""

import numpy as np
import pytest

from experiment_workbench.analysis import AnalysisInputError, ChannelAligner
from experiment_workbench.model import AlignmentGrid, AlignmentParams, InterpolationMethod
from experiment_workbench.validation import FloatArray


def _line(times: FloatArray, slope: float = 2.0, intercept: float = 1.0) -> FloatArray:
    """直線 slope × t + intercept の値を返す（線形補間で誤差なく再現できる信号）。

    Args:
        times: 時刻 [s]。
        slope: 傾き。
        intercept: 切片。

    Returns:
        値。
    """
    return slope * times + intercept


def _aligner(**changes: object) -> ChannelAligner:
    """条件の一部を変えた ChannelAligner を作る。

    Args:
        **changes: AlignmentParams の項目と値。

    Returns:
        ChannelAligner。
    """
    return ChannelAligner(AlignmentParams.model_validate(changes))


def test_same_source_columns_are_paired_without_interpolation() -> None:
    """時刻が同じ 2 チャンネル（同じ Source の列）は、補間せずにそのまま対応付ける。"""
    times = np.arange(0.0, 1.0, 0.01)
    x, y = np.sin(times), np.cos(times)
    pair = _aligner().align(times, x, times, y, time_comparable=True)
    np.testing.assert_array_equal(pair.time_s, times)
    np.testing.assert_array_equal(pair.input_values, x)
    np.testing.assert_array_equal(pair.output_values, y)
    record = pair.record
    assert record.shared_timestamps
    assert not record.input_interpolated
    assert not record.output_interpolated
    assert not record.offset_declared
    assert record.n_samples == times.size


def test_range_limits_the_paired_samples() -> None:
    """範囲を指定すると、その中の時刻だけを対応付ける。"""
    times = np.arange(0.0, 1.0, 0.01)
    pair = _aligner(start_s=0.2, end_s=0.5).align(times, times, times, times, time_comparable=True)
    assert pair.time_s[0] == pytest.approx(0.2)
    assert pair.time_s[-1] == pytest.approx(0.5)
    assert (pair.record.start_s, pair.record.end_s) == (pair.time_s[0], pair.time_s[-1])


def test_output_is_interpolated_onto_input_times_without_extrapolation() -> None:
    """入力の時刻に対応付けると出力だけを補間し、2 チャンネルが重なる範囲の外には出さない。"""
    t_in = np.arange(0.0, 2.0, 0.01)
    t_out = np.arange(0.5, 3.0, 1.0 / 30.0)
    pair = _aligner().align(t_in, _line(t_in), t_out, _line(t_out, 3.0, -1.0), time_comparable=True)
    assert pair.time_s[0] >= 0.5
    assert pair.time_s[-1] <= t_in[-1]
    np.testing.assert_array_equal(pair.input_values, _line(pair.time_s))
    np.testing.assert_allclose(pair.output_values, _line(pair.time_s, 3.0, -1.0), atol=1e-12)
    assert pair.record.output_interpolated
    assert not pair.record.input_interpolated


def test_output_grid_interpolates_input() -> None:
    """出力の時刻に対応付けると、入力だけを補間する。"""
    t_in = np.arange(0.0, 2.0, 0.001)
    t_out = np.arange(0.0, 2.0, 0.01)
    pair = _aligner(grid=AlignmentGrid.OUTPUT).align(
        t_in, _line(t_in), t_out, np.cos(t_out), time_comparable=True
    )
    np.testing.assert_array_equal(pair.time_s, t_out[t_out <= t_in[-1]])
    np.testing.assert_allclose(pair.input_values, _line(pair.time_s), atol=1e-12)
    assert pair.record.input_interpolated
    assert not pair.record.output_interpolated


def test_uniform_grid_interpolates_both_channels() -> None:
    """等間隔の時刻に対応付けると、両方のチャンネルを補間する。"""
    jitter = np.random.default_rng(0).uniform(-0.0005, 0.0005, 500)
    t_in = np.arange(500) * 0.004 + jitter  # 時刻のばらつきのある入力（欠損はない）
    t_out = np.arange(0.0, 2.0, 0.007)
    pair = _aligner(grid=AlignmentGrid.UNIFORM, uniform_rate_hz=50.0).align(
        t_in, _line(t_in), t_out, _line(t_out, -1.0, 0.0), time_comparable=True
    )
    np.testing.assert_allclose(np.diff(pair.time_s), 0.02, atol=1e-12)
    np.testing.assert_allclose(pair.input_values, _line(pair.time_s), atol=1e-12)
    np.testing.assert_allclose(pair.output_values, _line(pair.time_s, -1.0, 0.0), atol=1e-12)
    assert pair.record.input_interpolated
    assert pair.record.output_interpolated


def test_declared_offset_maps_output_onto_input_clock() -> None:
    """時刻オフセットを明示すると、出力の時刻に足して入力の時間軸にそろえ、記録に残す。"""
    t_in = np.arange(0.0, 3.0, 0.001)
    t_out = np.arange(-5.0, -2.0, 0.002)  # 出力の装置の時計は 5 秒遅れている
    signal_in = np.sin(2 * np.pi * t_in)
    signal_out = np.sin(2 * np.pi * (t_out + 5.0))
    pair = _aligner(output_time_offset_s=5.0).align(
        t_in, signal_in, t_out, signal_out, time_comparable=False
    )
    np.testing.assert_allclose(pair.output_values, pair.input_values, atol=1e-4)
    assert pair.record.offset_declared
    assert pair.record.output_time_offset_s == 5.0


def test_incomparable_channels_require_declared_time_correspondence() -> None:
    """時計が違うチャンネルは、時刻オフセットを明示しなければ対応付けない。"""
    times = np.arange(0.0, 1.0, 0.01)
    with pytest.raises(AnalysisInputError, match="時刻の対応"):
        _aligner().align(times, times, times, times, time_comparable=False)


def test_missing_segment_is_not_bridged() -> None:
    """欠損区間（時刻の飛び）の中の時刻は補間せずに NaN にし、その数を記録する。"""
    t_in = np.arange(0.0, 1.0, 0.001)
    t_out = np.arange(0.0, 1.0, 0.01)
    keep = (t_out < 0.4) | (t_out > 0.6)  # 0.4〜0.6 s の行が欠けている
    t_out = t_out[keep]
    pair = _aligner().align(t_in, t_in, t_out, _line(t_out), time_comparable=True)
    last_before, first_after = t_out[t_out < 0.4][-1], t_out[t_out > 0.6][0]
    inside = (pair.time_s > last_before) & (pair.time_s < first_after)
    assert np.isnan(pair.output_values[inside]).all()
    np.testing.assert_allclose(pair.output_values[~inside], _line(pair.time_s[~inside]), atol=1e-12)
    assert pair.record.n_gap_samples == int(inside.sum())


def test_gap_factor_can_be_raised_for_irregular_sampling() -> None:
    """不規則な時刻のデータでは、欠損とみなす間隔の倍率を大きくすれば、間隔の長い区間も補間する。"""
    t_out = np.sort(np.random.default_rng(1).uniform(0.0, 2.0, 300))
    t_in = np.arange(0.1, 1.9, 0.01)
    strict = _aligner().align(t_in, t_in, t_out, _line(t_out), time_comparable=True)
    assert strict.record.n_gap_samples > 0  # 既定の 1.5 倍では、長い間隔を欠損とみなす
    relaxed = _aligner(max_gap_factor=100.0).align(
        t_in, t_in, t_out, _line(t_out), time_comparable=True
    )
    assert relaxed.record.n_gap_samples == 0
    np.testing.assert_allclose(relaxed.output_values, _line(relaxed.time_s), atol=1e-12)


def test_linear_interpolation_keeps_missing_values_missing() -> None:
    """欠損（NaN）の値を埋めない（線形補間では、欠損に隣り合う区間の値も NaN になる）。"""
    t_out = np.arange(0.0, 1.0, 0.1)
    values = _line(t_out)
    values[5] = np.nan
    t_in = np.arange(0.0, 0.9, 0.01)
    pair = _aligner().align(t_in, t_in, t_out, values, time_comparable=True)
    near = (pair.time_s > t_out[4]) & (pair.time_s < t_out[6])
    assert np.isnan(pair.output_values[near]).all()
    assert np.isfinite(pair.output_values[~near]).all()


def test_pchip_does_not_overshoot_a_step() -> None:
    """PCHIP 補間は、ステップ状の変化で元のデータにない行き過ぎを作らない。"""
    t_out = np.arange(0.0, 1.0, 0.05)
    step = np.where(t_out < 0.5, 0.0, 1.0)
    t_in = np.arange(0.0, 0.95, 0.001)
    pair = _aligner(method=InterpolationMethod.PCHIP).align(
        t_in, t_in, t_out, step, time_comparable=True
    )
    assert pair.output_values.min() >= 0.0
    assert pair.output_values.max() <= 1.0
    assert pair.record.method is InterpolationMethod.PCHIP


def test_pchip_rejects_missing_values_in_range() -> None:
    """PCHIP 補間は、範囲の中に欠損があると補間しない（欠損を滑らかに埋めないため）。"""
    t_out = np.arange(0.0, 1.0, 0.1)
    values = _line(t_out)
    values[5] = np.nan
    t_in = np.arange(0.0, 0.9, 0.01)
    with pytest.raises(AnalysisInputError, match="PCHIP"):
        _aligner(method=InterpolationMethod.PCHIP).align(
            t_in, t_in, t_out, values, time_comparable=True
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output_time_offset_s": 10.0}, "重なっていません"),
        ({"start_s": 5.0, "end_s": 6.0}, "範囲"),
        ({"start_s": 0.5, "end_s": 0.505}, "2 個以上"),
    ],
)
def test_ranges_without_enough_overlap_are_reported(
    changes: dict[str, object], message: str
) -> None:
    """時刻が重ならない、範囲が重なりの外、または範囲に時刻が 2 つ未満なら、理由を示して断る。"""
    times = np.arange(0.0, 1.0, 0.01)
    with pytest.raises(AnalysisInputError, match=message):
        _aligner(**changes).align(times, times, times, times, time_comparable=True)


def test_invalid_time_axis_is_rejected() -> None:
    """時刻が逆行しているチャンネルは対応付けない（並べ替えは利用者が明示的に行う）。"""
    times = np.array([0.0, 0.2, 0.1, 0.3])
    with pytest.raises(AnalysisInputError, match="逆行"):
        _aligner().align(times, times, np.sort(times), times, time_comparable=True)
