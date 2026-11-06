"""analysis/runner.py（保存できる解析の実行・出力の情報・キャッシュ・GUI の初期値）のテスト。

解析の結果は、同じ条件で解析コアのクラスを直接呼んだ結果と一致することで確かめる
（runner は種類ごとの振り分けと表示用データへの変換だけを受け持つため）。
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from experiment_workbench.analysis import (
    AnalysisInputError,
    ButterworthFilter,
    Differentiator,
    Integrator,
    KalmanFilter,
    MovingAverageFilter,
    SavitzkyGolayFilter,
    analysis_fingerprint,
    default_initial_state,
    default_kalman_settings,
    default_omega,
    display_channel_name,
    estimate_noise_std,
    output_infos,
    output_series_id,
    overlays_input,
    result_from_frame,
    result_to_frame,
    round_significant,
    run_analysis,
    series_from_channel,
    system_from_settings,
)
from experiment_workbench.analysis.runner import MAX_CHART_POINTS
from experiment_workbench.model import (
    DATASET_TIME_COLUMN,
    TWO_CHANNEL_ANALYSIS_KINDS,
    AbsoluteParams,
    AdaptiveNoiseMethod,
    AlignmentGrid,
    AlignmentParams,
    AnalysisKind,
    AnalysisResult,
    AnalysisSpec,
    ButterworthKind,
    ButterworthParams,
    ChannelData,
    ChartStyle,
    CropParams,
    CrossCorrelationParams,
    DerivativeParams,
    DetrendParams,
    FillGapsParams,
    HysteresisParams,
    IntegralParams,
    KalmanModelKind,
    KalmanSettings,
    MovingAverageParams,
    NormalizeParams,
    OffsetParams,
    ResampleParams,
    SavitzkyGolayParams,
    ScaleParams,
    StatisticsParams,
    StepResponseParams,
)
from experiment_workbench.validation import FloatArray
from tests.conftest import Hierarchy

DT = 0.01
"""テスト用の信号のサンプリング間隔 [s]（100 Hz）。"""
TIME = np.arange(1000, dtype=np.float64) * DT
"""テスト用の時刻（0〜10 秒）。"""
TRUTH = np.sin(2 * np.pi * 0.5 * TIME)
"""テスト用の真の信号（0.5 Hz の正弦波）。"""
NOISE_STD = 0.05
"""テスト用の観測雑音の標準偏差。"""
NOISY = TRUTH + np.random.default_rng(0).normal(0.0, NOISE_STD, TIME.size)
"""真の信号に白色雑音を加えた観測。"""
NAN_ROWS = slice(200, 210)
"""欠測（NaN）にする行。"""

LOWPASS = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=5.0)
MOVING_AVERAGE = MovingAverageParams(window_samples=11)
SAVITZKY_GOLAY = SavitzkyGolayParams(window_samples=11, polyorder=2)
DERIVATIVE = DerivativeParams()
INTEGRAL = IntegralParams(initial_value=1.0)


def _data(hierarchy: Hierarchy, values: FloatArray, unit: str | None = "m") -> ChannelData:
    """テスト用の入力チャンネルのデータを作る。

    Args:
        hierarchy: テスト用のエンティティ（Source とチャンネルを使う）。
        values: チャンネルの値（時刻は TIME）。
        unit: チャンネルの単位。

    Returns:
        入力チャンネルのデータ。
    """
    _, _, _, source, channel, _ = hierarchy
    return ChannelData(
        channel=channel.model_copy(update={"unit": unit}),
        source=source,
        time_s=TIME.copy(),
        values=values,
    )


def _spec(
    data: ChannelData, kind: AnalysisKind, params: object, name: str = "解析"
) -> AnalysisSpec:
    """入力チャンネルに 1 つの処理をかける解析を作る。

    Args:
        data: 入力チャンネルのデータ。
        kind: 解析の種類。
        params: 種類に対応する条件（AnalysisSpec の同名の項目に入れる）。
        name: 解析の名前。

    Returns:
        解析。
    """
    return AnalysisSpec.model_validate(
        {"name": name, "input_channel_id": data.channel.id, "kind": kind, kind.value: params}
    )


def _with_nan(values: FloatArray) -> FloatArray:
    """NAN_ROWS の行を欠測にした配列を返す（元の配列は変えない）。

    Args:
        values: 元の値。

    Returns:
        欠測を含む値。
    """
    copied = values.copy()
    copied[NAN_ROWS] = np.nan
    return copied


def _rmse(estimate: FloatArray, truth: FloatArray) -> float:
    """二乗平均平方根誤差を返す。

    Args:
        estimate: 推定値。
        truth: 真値。

    Returns:
        RMSE。
    """
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


# --- 生データの系列・出力の情報 -------------------------------------------------


def test_series_from_channel_keeps_channel_id_and_values(hierarchy: Hierarchy) -> None:
    """生データの系列の ID はチャンネル ID、名前は「Source 名 / チャンネル名」になる。"""
    data = _data(hierarchy, NOISY)
    series = series_from_channel(data)
    assert series.series_id == data.channel.id
    assert series.label == "high_rate / accel"
    assert series.unit == "m"
    assert series.analysis_id is None
    np.testing.assert_array_equal(series.values, data.values)


@pytest.mark.parametrize(
    ("kind", "params", "unit", "expected_unit"),
    [
        (AnalysisKind.BUTTERWORTH, LOWPASS, "m", "m"),
        (AnalysisKind.MOVING_AVERAGE, MOVING_AVERAGE, None, None),
        (AnalysisKind.DERIVATIVE, DERIVATIVE, "m", "m/s"),
        (AnalysisKind.DERIVATIVE, DerivativeParams(order=2), "m", "m/s^2"),
        (AnalysisKind.DERIVATIVE, DERIVATIVE, "m/s", "(m/s)/s"),
        (AnalysisKind.INTEGRAL, INTEGRAL, "m/s^2", "(m/s^2)·s"),
    ],
)
def test_main_output_has_analysis_name_and_derived_unit(
    hierarchy: Hierarchy,
    kind: AnalysisKind,
    params: object,
    unit: str | None,
    expected_unit: str | None,
) -> None:
    """フィルタ・微分・積分の主な結果の単位は、入力の単位から決まる（微分は /s、積分は ·s）。"""
    data = _data(hierarchy, NOISY, unit)
    spec = _spec(data, kind, params, name="処理結果")
    infos = output_infos(spec, data.channel)
    assert infos[0].key == "output"
    assert infos[0].label == "処理結果"
    assert infos[0].unit == expected_unit


def test_filter_also_outputs_what_it_removed(hierarchy: Hierarchy) -> None:
    """フィルタは取り除いた成分も出力し、結果と足すと元データに戻る（何をしたかが見えるように）。"""
    data = _data(hierarchy, NOISY, "m")
    spec = _spec(data, AnalysisKind.BUTTERWORTH, LOWPASS, name="処理結果")
    infos = output_infos(spec, data.channel)
    assert [info.key for info in infos] == ["output", "removed"]
    assert infos[1].label == "処理結果: 除去成分"
    assert infos[1].unit == "m"
    kept, removed = run_analysis(spec, data).series
    np.testing.assert_allclose(kept.values + removed.values, data.values)


@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        (AnalysisKind.BUTTERWORTH, LOWPASS, True),
        (AnalysisKind.DERIVATIVE, DERIVATIVE, False),
        (AnalysisKind.INTEGRAL, INTEGRAL, False),
    ],
)
def test_only_same_quantity_is_overlaid(
    hierarchy: Hierarchy, kind: AnalysisKind, params: object, expected: bool
) -> None:
    """入力と同じ量の結果だけを重ねる（単位が違う線を重ねると元の線が読めなくなる）。"""
    data = _data(hierarchy, NOISY, "m")
    assert overlays_input(_spec(data, kind, params)) is expected


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [([0.0, 1.0], True), ([0.0, 1.0, 0.0], True), ([0.0, 2.0], False), ([1.0, 1.0], False)],
)
def test_ekf_overlays_only_when_state_is_the_observation(
    hierarchy: Hierarchy, coefficients: list[float], expected: bool
) -> None:
    """EKF の推定値は、観測の式が恒等のときだけ入力と同じ量になる。"""
    data = _data(hierarchy, NOISY, "V")
    base = default_kalman_settings(KalmanModelKind.POLYNOMIAL_OBSERVATION, data)
    polynomial = base.polynomial_observation
    assert polynomial is not None
    settings = base.model_copy(
        update={
            "polynomial_observation": polynomial.model_copy(update={"coefficients": coefficients})
        }
    )
    assert overlays_input(_spec(data, AnalysisKind.KALMAN, settings)) is expected


def test_kalman_summary_omits_zero_skipped_samples(hierarchy: Hierarchy) -> None:
    """更新しなかった時刻が 0 個なら、補足に書かない（読み取ることがないため）。"""
    data = _data(hierarchy, NOISY)
    settings = default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    result = run_analysis(_spec(data, AnalysisKind.KALMAN, settings), data)
    assert result.summary == []
    assert "イノベーションの RMS" in result.headline


def test_derivative_and_integral_have_no_removed_output(hierarchy: Hierarchy) -> None:
    """微分・積分は単位が変わるので、取り除いた成分は意味を持たず、出力しない。"""
    data = _data(hierarchy, NOISY)
    for kind, params in ((AnalysisKind.DERIVATIVE, DERIVATIVE), (AnalysisKind.INTEGRAL, INTEGRAL)):
        keys = [info.key for info in output_infos(_spec(data, kind, params), data.channel)]
        assert keys == ["output"]


def test_filter_headline_says_how_much_was_removed(hierarchy: Hierarchy) -> None:
    """フィルタの結論の文は、取り除いた変動の大きさと割合を数値で述べる。"""
    data = _data(hierarchy, NOISY, "m")
    result = run_analysis(_spec(data, AnalysisKind.BUTTERWORTH, LOWPASS), data)
    removed = result.series[1].values
    rms = float(np.sqrt(np.mean(removed**2)))
    assert f"{rms:.3g} m" in result.headline
    assert "除去成分の RMS" in result.headline
    assert "%" in result.headline


@pytest.mark.parametrize(
    ("name", "unit", "expected"),
    [
        ("張力[N]", "N", "張力"),
        ("accel (m/s^2)", "m/s^2", "accel"),
        ("電圧（V）", "V", "電圧"),
        ("force[N]", None, "force[N]"),  # 単位が読めていなければ名前を変えない
        ("force[N]", "kN", "force[N]"),
        ("[N]", "N", "[N]"),  # 除くと空になるなら除かない
        ("accel", "m", "accel"),
    ],
)
def test_display_name_drops_unit_written_in_column_name(
    hierarchy: Hierarchy, name: str, unit: str | None, expected: str
) -> None:
    """列名の単位は表示の名前から除く（見出しが "張力[N] [N]" と重ならないように）。"""
    channel = hierarchy[4].model_copy(update={"name": name, "unit": unit})
    assert display_channel_name(channel) == expected


def test_series_know_their_channel_name(hierarchy: Hierarchy) -> None:
    """系列はチャンネル名を持つ（図の見出しと、凡例の短い名前を作るため）。"""
    data = _data(hierarchy, NOISY)
    assert series_from_channel(data).channel_name == data.channel.name
    result = run_analysis(_spec(data, AnalysisKind.DERIVATIVE, DERIVATIVE), data)
    assert result.series[0].channel_name == data.channel.name


_LINEAR_3 = {
    "transition": np.eye(3).tolist(),
    "observation": [[1.0, 0.0, 0.0]],
    "process_noise": np.eye(3).tolist(),
    "measurement_noise": [[1.0]],
}
"""状態が 3 次元の線形モデル（出力の一覧のテスト用）。"""


@pytest.mark.parametrize(
    ("model", "n_states", "state_units"),
    [
        (
            {
                "model_kind": KalmanModelKind.KINEMATIC,
                "kinematic": {"process_noise_density": 1.0, "measurement_noise_std": 0.1},
            },
            2,
            ["m", "m/s"],
        ),
        (
            {
                "model_kind": KalmanModelKind.KINEMATIC,
                "kinematic": {
                    "derivative_order": 2,
                    "process_noise_density": 1.0,
                    "measurement_noise_std": 0.1,
                },
            },
            3,
            ["m", "m/s", "m/s^2"],
        ),
        (
            {
                "model_kind": KalmanModelKind.POLYNOMIAL_OBSERVATION,
                "polynomial_observation": {
                    "process_noise_density": 1.0,
                    "measurement_noise_std": 0.1,
                    "coefficients": [0.0, 2.0],
                },
            },
            2,
            [None, None],  # 推定する x は観測（センサの出力）と単位が違うので分からない
        ),
        ({"model_kind": KalmanModelKind.LINEAR, "linear": _LINEAR_3}, 3, [None, None, None]),
    ],
    ids=["kinematic_1", "kinematic_2", "polynomial", "linear_3"],
)
def test_kalman_outputs_are_states_innovation_and_noise(
    hierarchy: Hierarchy, model: dict[str, object], n_states: int, state_units: list[str | None]
) -> None:
    """カルマンフィルタの出力は、状態ごとの推定値・イノベーション・観測雑音の標準偏差。"""
    data = _data(hierarchy, NOISY)
    initial = {"state": [0.0] * n_states, "covariance": np.eye(n_states).tolist()}
    settings = KalmanSettings.model_validate(model | {"initial": initial})
    spec = _spec(data, AnalysisKind.KALMAN, settings, name="KF")
    infos = output_infos(spec, data.channel)
    expected_keys = [f"state_{index}" for index in range(n_states)] + ["innovation", "noise_std"]
    assert [info.key for info in infos] == expected_keys
    assert [info.unit for info in infos] == [*state_units, "m", "m"]
    assert all(info.label.startswith("KF: ") for info in infos)


def test_output_ids_are_stable_and_distinct(hierarchy: Hierarchy) -> None:
    """出力の ID は解析の ID と出力名だけで決まり、名前を変えても同じ、解析が違えば別になる。"""
    data = _data(hierarchy, NOISY)
    spec = _spec(
        data, AnalysisKind.KALMAN, default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    )
    infos = output_infos(spec, data.channel)
    renamed = output_infos(spec.model_copy(update={"name": "別名"}), data.channel)
    assert [info.series_id for info in renamed] == [info.series_id for info in infos]
    assert infos[0].series_id == output_series_id(spec.id, "state_0")
    assert len({info.series_id for info in infos}) == len(infos)
    other = output_infos(spec.model_copy(update={"id": uuid4()}), data.channel)
    assert not {info.series_id for info in infos} & {info.series_id for info in other}


# --- 実行 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "params", "direct"),
    [
        (AnalysisKind.BUTTERWORTH, LOWPASS, ButterworthFilter(LOWPASS).apply),
        (AnalysisKind.MOVING_AVERAGE, MOVING_AVERAGE, MovingAverageFilter(MOVING_AVERAGE).apply),
        (AnalysisKind.SAVITZKY_GOLAY, SAVITZKY_GOLAY, SavitzkyGolayFilter(SAVITZKY_GOLAY).apply),
        (AnalysisKind.DERIVATIVE, DERIVATIVE, Differentiator(DERIVATIVE).apply),
        (AnalysisKind.INTEGRAL, INTEGRAL, Integrator(INTEGRAL).apply),
    ],
    ids=["butterworth", "moving_average", "savitzky_golay", "derivative", "integral"],
)
def test_signal_processing_result_equals_direct_call(
    hierarchy: Hierarchy,
    kind: AnalysisKind,
    params: object,
    direct: Callable[[FloatArray, FloatArray], FloatArray],
) -> None:
    """フィルタ・微分・積分の結果は、解析コアのクラスを直接呼んだ結果と一致する。"""
    data = _data(hierarchy, NOISY)
    spec = _spec(data, kind, params)
    result = run_analysis(spec, data)
    assert result.analysis_id == spec.id
    assert result.summary == []
    assert result.headline
    series = result.series[0]
    np.testing.assert_array_equal(series.values, direct(TIME, NOISY))
    assert series.series_id == output_series_id(spec.id, "output")
    assert series.analysis_id == spec.id
    assert series.source == data.source  # 時間軸は入力チャンネルの Source のまま
    assert series.band_lower is None
    assert series.marker_times_s is None


def test_run_analysis_rejects_data_of_another_channel(hierarchy: Hierarchy) -> None:
    """解析の入力チャンネルと違うデータを渡すと、計算せずにエラーにする。"""
    data = _data(hierarchy, NOISY)
    spec = _spec(data, AnalysisKind.BUTTERWORTH, LOWPASS).model_copy(
        update={"input_channel_id": uuid4()}
    )
    with pytest.raises(AnalysisInputError, match="チャンネル"):
        run_analysis(spec, data)


def test_kalman_result_has_bands_markers_and_summary(hierarchy: Hierarchy) -> None:
    """カルマンフィルタの推定値には ±σ の帯、欠測で更新しなかった時刻には印が付く。

    イノベーションには ±√S の帯が付き、観測雑音の標準偏差は R が一定なら σ のまま。
    """
    data = _data(hierarchy, _with_nan(NOISY))
    settings = default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    result = run_analysis(_spec(data, AnalysisKind.KALMAN, settings), data)
    direct = KalmanFilter(system_from_settings(settings), settings.initial).run(TIME, data.values)

    estimate, velocity, innovation, noise_std = result.series
    sigma = np.sqrt(direct.covariances[:, 0, 0])
    np.testing.assert_array_equal(estimate.values, direct.states[:, 0])
    assert estimate.band_lower is not None
    assert estimate.band_upper is not None
    np.testing.assert_allclose(estimate.band_upper - estimate.values, sigma)
    np.testing.assert_allclose(estimate.values - estimate.band_lower, sigma)
    np.testing.assert_array_equal(estimate.marker_times_s, TIME[NAN_ROWS])
    np.testing.assert_array_equal(velocity.values, direct.states[:, 1])

    np.testing.assert_array_equal(innovation.values, direct.innovations[:, 0])
    assert innovation.band_upper is not None
    innovation_sigma = np.sqrt(direct.innovation_covariances[:, 0, 0])
    np.testing.assert_allclose(innovation.band_upper - innovation.values, innovation_sigma)
    assert innovation.marker_times_s is None

    assert settings.kinematic is not None
    np.testing.assert_allclose(noise_std.values, settings.kinematic.measurement_noise_std)
    assert result.summary == ["観測更新しなかった時刻: 10 個"]


def test_adaptive_kalman_summary_reports_final_noise(hierarchy: Hierarchy) -> None:
    """適応推定を使うと、要約に推定した √R の最終値と、R を正定値化した時刻の数が入る。"""
    data = _data(hierarchy, NOISY)
    base = default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    settings = KalmanSettings.model_validate(
        base.model_dump() | {"adaptive": {"method": AdaptiveNoiseMethod.SAGE_HUSA}}
    )
    result = run_analysis(_spec(data, AnalysisKind.KALMAN, settings), data)
    final = float(result.series[-1].values[-1])
    assert len(result.summary) == 2  # 更新しなかった時刻は 0 個なので書かない
    assert result.summary[0] == f"推定した観測雑音の標準偏差 √R の最終値: {final:.6g}"
    assert result.summary[1].startswith("R の正定値化を行った時刻:")


# --- 計算結果のキャッシュ -------------------------------------------------------


def test_fingerprint_ignores_identity_but_tracks_conditions(hierarchy: Hierarchy) -> None:
    """指紋は ID・名前・作成日時では変わらず、条件や入力チャンネルが変わると変わる。"""
    data = _data(hierarchy, NOISY)
    spec = _spec(data, AnalysisKind.BUTTERWORTH, LOWPASS)
    same = spec.model_copy(
        update={"id": uuid4(), "name": "別名", "created_at": datetime(2020, 1, 1, tzinfo=UTC)}
    )
    assert analysis_fingerprint(same) == analysis_fingerprint(spec)
    changed_params = spec.model_copy(
        update={"butterworth": LOWPASS.model_copy(update={"cutoff_hz": 6.0})}
    )
    changed_input = spec.model_copy(update={"input_channel_id": uuid4()})
    fingerprints = {analysis_fingerprint(s) for s in (spec, changed_params, changed_input)}
    assert len(fingerprints) == 3
    assert len(analysis_fingerprint(spec)) == 16


def test_cache_frame_round_trip_restores_series(hierarchy: Hierarchy) -> None:
    """結果をキャッシュの表にして戻すと、値・帯・印・名前・単位が同じ系列になる。"""
    data = _data(hierarchy, _with_nan(NOISY))
    spec = _spec(
        data, AnalysisKind.KALMAN, default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    )
    result = run_analysis(spec, data)
    frame = result_to_frame(result, output_infos(spec, data.channel))
    restored = result_from_frame(spec, data.channel, data.source, frame)
    assert restored is not None
    for original, copied in zip(result.series, restored.series, strict=True):
        assert (copied.series_id, copied.label, copied.unit) == (
            original.series_id,
            original.label,
            original.unit,
        )
        assert copied.analysis_id == spec.id
        np.testing.assert_array_equal(copied.time_s, original.time_s)
        np.testing.assert_array_equal(copied.values, original.values)
        for name in ("band_lower", "band_upper", "marker_times_s"):
            expected, actual = getattr(original, name), getattr(copied, name)
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)


def test_cache_frame_without_expected_columns_is_rejected(hierarchy: Hierarchy) -> None:
    """表の列が出力と合わないキャッシュは使わない（None を返して再計算させる）。"""
    data = _data(hierarchy, NOISY)
    spec = _spec(data, AnalysisKind.BUTTERWORTH, LOWPASS)
    frame = pl.DataFrame({DATASET_TIME_COLUMN: TIME, "other": NOISY})
    assert result_from_frame(spec, data.channel, data.source, frame) is None


# --- GUI の初期値 ---------------------------------------------------------------


def test_estimate_noise_std_ignores_slow_signal_outliers_and_nan() -> None:
    """雑音の推定は、ゆっくり変わる信号成分・まれな外れ値・欠測の影響をほとんど受けない。"""
    rng = np.random.default_rng(1)
    values = np.sin(2 * np.pi * 0.1 * TIME) + rng.normal(0.0, NOISE_STD, TIME.size)
    values[::97] += 5.0
    values[50:60] = np.nan
    assert estimate_noise_std(values) == pytest.approx(NOISE_STD, rel=0.15)


@pytest.mark.parametrize(
    "values", [np.array([1.0, 2.0]), np.full(10, np.nan), np.full(10, 2.0)], ids=str
)
def test_estimate_noise_std_falls_back_to_one(values: FloatArray) -> None:
    """値が少なすぎる・すべて欠測・ばらつきがない場合は、推定できないので 1.0 を返す。"""
    assert estimate_noise_std(values) == 1.0


@pytest.mark.parametrize("model_kind", list(KalmanModelKind))
def test_default_kalman_settings_smooth_noisy_data(
    hierarchy: Hierarchy, model_kind: KalmanModelKind
) -> None:
    """データから作った初期値の条件で、どのモデルでも観測より真値に近い推定値が得られる。"""
    data = _data(hierarchy, NOISY)
    settings = default_kalman_settings(model_kind, data)
    assert settings.model_kind is model_kind
    estimate = run_analysis(_spec(data, AnalysisKind.KALMAN, settings), data).series[0].values
    burn_in = 50
    assert _rmse(estimate[burn_in:], TRUTH[burn_in:]) < 0.7 * _rmse(
        NOISY[burn_in:], TRUTH[burn_in:]
    )


def test_default_kalman_settings_use_estimated_noise(hierarchy: Hierarchy) -> None:
    """初期値の観測雑音はデータから推定し、帯域 ω はサンプリング周波数の 1/20 になる。"""
    data = _data(hierarchy, NOISY)
    settings = default_kalman_settings(KalmanModelKind.KINEMATIC, data)
    assert settings.kinematic is not None
    sigma = settings.kinematic.measurement_noise_std
    assert sigma == pytest.approx(NOISE_STD, rel=0.15)
    omega = default_omega(data)
    assert omega == pytest.approx(2 * np.pi / (20 * DT))
    # 初期値は有効数字 4 桁に丸めるので、相対誤差 0.05 % 以内で一致する。
    density = settings.kinematic.process_noise_density
    assert density == pytest.approx(omega**4 * sigma**2 * DT, rel=5e-4)


def test_round_significant_keeps_requested_digits() -> None:
    """有効数字で丸めると、桁の大きさによらず指定した桁数が残る。"""
    assert round_significant(1.8412847726322092) == 1.841
    assert round_significant(0.013748675000061939) == 0.01375
    assert round_significant(428.2770970415502) == 428.3
    assert round_significant(6.136666666666667e-07) == 6.137e-07
    assert round_significant(9.999800003999921, 3) == 10.0


@pytest.mark.parametrize("model_kind", list(KalmanModelKind))
def test_default_kalman_settings_are_short_numbers(
    hierarchy: Hierarchy, model_kind: KalmanModelKind
) -> None:
    """GUI に入れる初期値（雑音・共分散）は、打ち直しやすいよう有効数字 4 桁になっている。

    初期状態の値は、オフセットの大きな信号で誤差が出ないよう丸めない。
    """
    data = _data(hierarchy, 1000.0 + NOISY)
    settings = default_kalman_settings(model_kind, data)
    numbers: list[float] = [value for row in settings.initial.covariance for value in row]
    if settings.kinematic is not None:
        numbers += [
            settings.kinematic.process_noise_density,
            settings.kinematic.measurement_noise_std,
        ]
    if settings.linear is not None:
        numbers += [value for row in settings.linear.process_noise for value in row]
        numbers += [value for row in settings.linear.measurement_noise for value in row]
    if settings.polynomial_observation is not None:
        numbers += [
            settings.polynomial_observation.process_noise_density,
            settings.polynomial_observation.measurement_noise_std,
        ]
    assert all(round_significant(value) == value for value in numbers)
    assert settings.initial.state[0] == 1000.0 + NOISY[0]


def test_default_initial_state_inverts_linear_term_of_polynomial(hierarchy: Hierarchy) -> None:
    """多項式の観測モデルでは、最初の観測を 1 次の項で逆算して x の初期値にする。"""
    values = 1.0 + 2.0 * (3.0 + NOISY)
    data = _data(hierarchy, values)
    initial = default_initial_state(3, data, sigma=0.1, omega=10.0, coefficients=[1.0, 2.0, 0.5])
    assert initial.state[0] == pytest.approx((values[0] - 1.0) / 2.0)
    assert initial.state[1:] == [0.0, 0.0]
    covariance = np.array(initial.covariance)
    assert covariance[0, 0] == pytest.approx((0.1 / 2.0) ** 2)
    np.testing.assert_array_equal(covariance, np.diag(np.diag(covariance)))  # 対角行列
    assert (np.diag(covariance) > 0).all()


# --- 2 チャンネルの解析 -----------------------------------------------------------

STEP_TIME = 3.0
"""2 チャンネルの解析のテストで、入力がステップ状に変わる時刻 [s]。"""
DELAY = 0.1
"""テスト用の系のむだ時間 [s]。"""
TIME_CONSTANT = 0.2
"""テスト用の系の時定数 [s]。"""
STEP_INPUT = np.where(TIME >= STEP_TIME, 10.0, 0.0)
"""ステップ入力。"""
STEP_OUTPUT = 1.0 + 5.0 * (1.0 - np.exp(-np.maximum(TIME - STEP_TIME - DELAY, 0.0) / TIME_CONSTANT))
"""ステップ入力に対する、一次遅れ＋むだ時間の出力。"""


def _output_channel(
    hierarchy: Hierarchy, values: FloatArray, *, same_clock: bool = True
) -> ChannelData:
    """入力とは別の出力チャンネルのデータを作る。

    Args:
        hierarchy: テスト用のエンティティ。
        values: 出力の値（時刻は TIME）。
        same_clock: False なら、別の時計の Source のチャンネルにする。

    Returns:
        出力チャンネルのデータ。
    """
    _, _, _, source, channel, _ = hierarchy
    other_source = source if same_clock else source.model_copy(update={"id": uuid4()})
    return ChannelData(
        channel=channel.model_copy(update={"id": uuid4(), "name": "voltage", "unit": "V"}),
        source=other_source,
        time_s=TIME.copy(),
        values=values,
    )


def _two_channel_spec(
    data: ChannelData,
    output: ChannelData,
    kind: AnalysisKind,
    params: object,
    alignment: AlignmentParams | None = None,
) -> AnalysisSpec:
    """2 チャンネルの解析を作る。

    Args:
        data: 入力チャンネルのデータ。
        output: 出力チャンネルのデータ。
        kind: 解析の種類。
        params: 種類に対応する条件。
        alignment: 対応付けの条件（None なら既定）。

    Returns:
        解析。
    """
    return AnalysisSpec.model_validate(
        {
            "name": "応答",
            "input_channel_id": data.channel.id,
            "output_channel_id": output.channel.id,
            "alignment": alignment or AlignmentParams(),
            "kind": kind,
            kind.value: params,
        }
    )


def _metric_values(result: AnalysisResult) -> dict[str, float | None]:
    """結果の指標を、識別名 → 値 の辞書にする。

    Args:
        result: 解析の結果。

    Returns:
        指標の辞書。
    """
    return {metric.key: metric.value for metric in result.metrics}


@pytest.mark.parametrize("kind", sorted(TWO_CHANNEL_ANALYSIS_KINDS))
def test_two_channel_kinds_have_no_time_series_outputs(
    hierarchy: Hierarchy, kind: AnalysisKind
) -> None:
    """2 チャンネルの解析は時系列を出力しない（指標とグラフで結果を見る）。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    params = {
        AnalysisKind.STEP_RESPONSE: StepResponseParams(),
        AnalysisKind.HYSTERESIS: HysteresisParams(),
        AnalysisKind.CROSS_CORRELATION: CrossCorrelationParams(min_lag_s=-1.0, max_lag_s=1.0),
    }[kind]
    assert output_infos(_two_channel_spec(data, output, kind, params), data.channel) == []


def test_step_response_result_has_metrics_chart_and_markers(hierarchy: Hierarchy) -> None:
    """ステップ応答の結果には、指標の表・正規化した応答のグラフ・時刻の印が入る。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    result = run_analysis(spec, data, output_data=output)
    assert result.series == []
    values = _metric_values(result)
    assert values["time_constant_s"] == pytest.approx(TIME_CONSTANT, abs=DT)
    assert values["dead_time_s"] == pytest.approx(DELAY, abs=1.5 * DT)
    assert values["gain"] == pytest.approx(0.5, rel=1e-3)
    assert next(m for m in result.metrics if m.key == "gain").unit == "V/N"
    labels = [marker.label for marker in result.time_markers]
    assert labels == ["①", "②", "③", "④"]  # 近い時刻でも重ならないよう、図の印は番号だけ
    assert result.time_markers[0].time_s == pytest.approx(STEP_TIME, abs=DT)
    chart = result.charts[0]
    assert [series.label for series in chart.series if series.show_in_legend] == ["入力", "出力"]
    notes = [annotation.text for annotation in chart.annotations]
    assert [note[:1] for note in notes] == labels  # 番号の意味はグラフの注記で分かる
    assert "補間: なし（同じ時刻のサンプルをそのまま対応付け）" in result.conditions
    assert not any(line.startswith("補間") for line in result.summary)  # 変換がないなら見せない


def test_step_headline_states_the_timing_in_plain_words(hierarchy: Hierarchy) -> None:
    """ステップ応答の結論の文は、動き始めと落ち着くまでの時間を述べる。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    result = run_analysis(spec, data, output_data=output)
    values = _metric_values(result)
    assert "応答開始遅れは" in result.headline
    assert "整定時間 Ts は" in result.headline
    delay = values["response_delay_s"]
    assert delay is not None
    expected = f"{delay * 1000:.3g} ms" if abs(delay) < 1.0 else f"{delay:.3g} s"
    assert expected in result.headline


def test_step_notes_stay_off_the_output_curve(hierarchy: Hierarchy) -> None:
    """ステップ応答の注記は、曲線のない所に置く（曲線を隠さないように）。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    chart = run_analysis(spec, data, output_data=output).charts[0]
    x, y = chart.series[1].x, chart.series[1].y
    for note in chart.annotations:
        if note.on_left:
            assert float(np.nanmax(y[x <= note.x])) < 100.0 * note.height
        else:  # 右に置く文字は、曲線がその高さより上にある所に置く
            assert float(np.interp(note.x, x, y)) > 100.0 * note.height


def test_step_chart_uses_percent_and_readable_time_unit(hierarchy: Hierarchy) -> None:
    """ステップ応答のグラフは、変化前 0 %・変化後 100 % に揃える。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    chart = run_analysis(spec, data, output_data=output).charts[0]
    assert chart.y_label == "正規化した応答 [%]"
    assert float(np.nanmax(chart.series[1].y)) == pytest.approx(100.0, abs=5.0)


def test_hysteresis_result_has_branches_and_regression(hierarchy: Hierarchy) -> None:
    """ヒステリシスの結果には、分岐を色分けした X-Y のグラフと回帰直線が入る。"""
    x = 5.0 + 5.0 * np.sin(2 * np.pi * 0.5 * TIME)
    y = 2.0 * np.interp(TIME - 0.05, TIME, x)  # 遅れがあるので、ループが開く
    data = _data(hierarchy, x, "N")
    output = _output_channel(hierarchy, y)
    spec = _two_channel_spec(data, output, AnalysisKind.HYSTERESIS, HysteresisParams())
    result = run_analysis(spec, data, output_data=output)
    values = _metric_values(result)
    assert values["slope"] == pytest.approx(2.0, rel=0.1)
    assert values["max_hysteresis"] is not None
    assert values["max_hysteresis"] > 0
    assert next(m for m in result.metrics if m.key == "loop_area").unit == "N·V"
    chart = result.charts[0]
    assert chart.x_label == "入力 accel [N]"
    styles = {series.label: series.style for series in chart.series}
    assert styles["測定値"] is ChartStyle.POINTS
    assert styles["増加時"] is ChartStyle.LINE
    assert styles["減少時"] is ChartStyle.LINE
    assert styles["回帰直線"] is ChartStyle.DASHED
    legend = [series.label for series in chart.series if series.show_in_legend]
    assert legend == ["増加時", "減少時", "回帰直線"]  # 読み取る線だけを凡例に出す
    assert any(line.startswith("サンプル数") for line in result.conditions)
    # 差はループの間隔としては小さいので、差そのものの図も作り、最大の位置に注記を付ける。
    assert result.charts[1].title == "分岐の差（増加時 − 減少時）"
    maximum = values["max_hysteresis"]
    assert f"{maximum:.3g}" in chart.annotations[0].text
    assert "最大" in result.headline
    assert f"{maximum:.3g} V" in result.headline


def test_headline_warns_when_direction_is_unstable(hierarchy: Hierarchy) -> None:
    """向きの判定が乱れているときは、数値を結論として読まれないよう、結論の文で注意する。"""
    clean = 5.0 + 5.0 * np.sin(2 * np.pi * 0.5 * TIME)
    noisy = clean + np.random.default_rng(0).normal(0.0, 0.2, clean.size)
    data = _data(hierarchy, noisy, "N")
    output = _output_channel(hierarchy, 2.0 * clean)
    plain = _two_channel_spec(data, output, AnalysisKind.HYSTERESIS, HysteresisParams())
    assert "正しく分離できていません" in run_analysis(plain, data, output_data=output).headline
    smoothed = _two_channel_spec(
        data, output, AnalysisKind.HYSTERESIS, HysteresisParams(smoothing_cutoff_hz=5.0)
    )
    assert (
        "正しく分離できていません" not in run_analysis(smoothed, data, output_data=output).headline
    )


def test_chart_points_are_thinned_but_metrics_use_all_samples(hierarchy: Hierarchy) -> None:
    """グラフの点は表示用に間引き、指標は全サンプルで計算する。"""
    times = np.arange(100_000, dtype=np.float64) * 1e-3
    x = 5.0 + 5.0 * np.sin(2 * np.pi * 0.05 * times)
    _, _, _, source, channel, _ = hierarchy
    data = ChannelData(channel=channel, source=source, time_s=times, values=x)
    output = ChannelData(
        channel=channel.model_copy(update={"id": uuid4()}),
        source=source,
        time_s=times,
        values=2.0 * x,
    )
    spec = _two_channel_spec(data, output, AnalysisKind.HYSTERESIS, HysteresisParams())
    result = run_analysis(spec, data, output_data=output)
    points = [s for s in result.charts[0].series if s.style is ChartStyle.POINTS]
    assert all(series.x.size <= MAX_CHART_POINTS for series in points)
    assert any(line.startswith("サンプル数") and "増加時 4" in line for line in result.conditions)


def test_cross_correlation_result_reports_delay_and_curve(hierarchy: Hierarchy) -> None:
    """相互相関の結果には、遅れと相関の曲線が入り、推定した遅れに印が付く。"""
    delay = 0.123
    x = np.sin(2 * np.pi * 0.7 * TIME) + 0.5 * np.sin(2 * np.pi * 1.9 * TIME)
    y = np.sin(2 * np.pi * 0.7 * (TIME - delay)) + 0.5 * np.sin(2 * np.pi * 1.9 * (TIME - delay))
    data = _data(hierarchy, x)
    output = _output_channel(hierarchy, y)
    params = CrossCorrelationParams(min_lag_s=-0.4, max_lag_s=0.4)
    spec = _two_channel_spec(data, output, AnalysisKind.CROSS_CORRELATION, params)
    result = run_analysis(spec, data, output_data=output)
    assert _metric_values(result)["delay_s"] == pytest.approx(delay, abs=0.2 * DT)
    chart = result.charts[0]
    marker = next(s for s in chart.series if s.style is ChartStyle.MARKERS)
    assert "[ms]" in chart.x_label  # 短い範囲は ms で表す
    assert float(marker.x[0]) / 1000.0 == pytest.approx(delay, abs=0.2 * DT)
    assert "遅れています" in result.headline
    assert "123 ms" in result.headline


def test_cross_correlation_headline_says_when_output_leads(hierarchy: Hierarchy) -> None:
    """出力が先に動くデータでは、結論の文は「進んでいます」になる。"""
    x = np.sin(2 * np.pi * 0.7 * (TIME - 0.1))
    y = np.sin(2 * np.pi * 0.7 * TIME)
    data = _data(hierarchy, x)
    output = _output_channel(hierarchy, y)
    params = CrossCorrelationParams(min_lag_s=-0.4, max_lag_s=0.4)
    spec = _two_channel_spec(data, output, AnalysisKind.CROSS_CORRELATION, params)
    assert "進んでいます" in run_analysis(spec, data, output_data=output).headline


def test_two_channel_analysis_needs_output_data(hierarchy: Hierarchy) -> None:
    """2 チャンネルの解析は、出力チャンネルのデータがなければ計算しない。"""
    data = _data(hierarchy, STEP_INPUT)
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    with pytest.raises(AnalysisInputError, match="出力チャンネル"):
        run_analysis(spec, data)


def test_different_clocks_need_declared_offset(hierarchy: Hierarchy) -> None:
    """時計が違うチャンネルは、時刻オフセットを指定すると解析でき、要約に記録される。"""
    data = _data(hierarchy, STEP_INPUT)
    output = _output_channel(hierarchy, STEP_OUTPUT, same_clock=False)
    plain = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    with pytest.raises(AnalysisInputError, match="時刻の対応"):
        run_analysis(plain, data, output_data=output)
    declared = _two_channel_spec(
        data,
        output,
        AnalysisKind.STEP_RESPONSE,
        StepResponseParams(),
        AlignmentParams(output_time_offset_s=0.25, end_s=9.0),
    )
    result = run_analysis(declared, data, output_data=output)
    assert any("0.25 s を足して" in line for line in result.summary)
    assert _metric_values(result)["dead_time_s"] == pytest.approx(DELAY + 0.25, abs=1.5 * DT)


def test_interpolation_is_recorded_in_summary(hierarchy: Hierarchy) -> None:
    """等間隔の時刻に対応付けると、両方を補間したことが要約に残る。"""
    data = _data(hierarchy, STEP_INPUT)
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(
        data,
        output,
        AnalysisKind.STEP_RESPONSE,
        StepResponseParams(),
        AlignmentParams(grid=AlignmentGrid.UNIFORM, uniform_rate_hz=50.0),
    )
    result = run_analysis(spec, data, output_data=output)
    assert "補間: 線形補間（入力と出力）" in result.summary


# --- 値の変換・時間の処理・統計 -------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "params", "unit", "expected_unit", "keys"),
    [
        (AnalysisKind.OFFSET, OffsetParams(), "N", "N", ["output"]),
        (AnalysisKind.SCALE, ScaleParams(gain=2.0, unit="kN"), "N", "kN", ["output"]),
        (AnalysisKind.SCALE, ScaleParams(gain=2.0), "N", None, ["output"]),
        (AnalysisKind.SCALE, ScaleParams(), "N", "N", ["output"]),
        (AnalysisKind.DETREND, DetrendParams(), "N", "N", ["output", "removed"]),
        (AnalysisKind.ABSOLUTE, AbsoluteParams(), "N", "N", ["output"]),
        (AnalysisKind.NORMALIZE, NormalizeParams(), "N", None, ["output"]),
        (AnalysisKind.CROP, CropParams(start_s=0.5, end_s=1.0), "N", "N", ["output"]),
        (AnalysisKind.RESAMPLE, ResampleParams(rate_hz=50.0), "N", "N", ["output"]),
        (AnalysisKind.FILL_GAPS, FillGapsParams(), "N", "N", ["output"]),
    ],
)
def test_processing_outputs_and_units(
    hierarchy: Hierarchy,
    kind: AnalysisKind,
    params: object,
    unit: str,
    expected_unit: str | None,
    keys: list[str],
) -> None:
    """値が変わる処理の単位を正しく扱う（倍率を掛けた値に元の単位を付けない）。"""
    data = _data(hierarchy, NOISY, unit)
    spec = _spec(data, kind, params)
    infos = output_infos(spec, data.channel)
    assert [info.key for info in infos] == keys
    assert infos[0].unit == expected_unit
    result = run_analysis(spec, data)
    assert result.headline
    assert [series.series_id for series in result.series] == [info.series_id for info in infos]


@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        (AnalysisKind.OFFSET, OffsetParams(), False),
        (AnalysisKind.SCALE, ScaleParams(gain=2.0), False),
        (AnalysisKind.DETREND, DetrendParams(), False),
        (AnalysisKind.NORMALIZE, NormalizeParams(), False),
        (AnalysisKind.ABSOLUTE, AbsoluteParams(), True),
        (AnalysisKind.CROP, CropParams(start_s=0.5, end_s=1.0), True),
        (AnalysisKind.RESAMPLE, ResampleParams(rate_hz=50.0), True),
        (AnalysisKind.FILL_GAPS, FillGapsParams(), True),
    ],
)
def test_overlay_rule_for_new_processing(
    hierarchy: Hierarchy, kind: AnalysisKind, params: object, expected: bool
) -> None:
    """大きさが変わる処理は重ねず、波形が同じ大きさの処理だけを重ねる。"""
    data = _data(hierarchy, NOISY)
    assert overlays_input(_spec(data, kind, params)) is expected


def test_offset_headline_states_the_baseline(hierarchy: Hierarchy) -> None:
    """オフセット除去の結論の文は、差し引いた基準値を述べる。"""
    data = _data(hierarchy, np.full(TIME.shape, 7.0) + NOISY - np.mean(NOISY), "N")
    result = run_analysis(_spec(data, AnalysisKind.OFFSET, OffsetParams()), data)
    assert "基準値 7 N" in result.headline
    assert float(np.mean(result.series[0].values)) == pytest.approx(0.0, abs=1e-9)


def test_resample_result_has_its_own_time_axis_and_can_be_cached(hierarchy: Hierarchy) -> None:
    """リサンプリングの結果は別の長さの時刻を持ち、キャッシュから同じ結果に戻せる。"""
    data = _data(hierarchy, NOISY, "N")
    spec = _spec(data, AnalysisKind.RESAMPLE, ResampleParams(rate_hz=50.0))
    result = run_analysis(spec, data)
    infos = output_infos(spec, data.channel)
    series = result.series[0]
    assert series.time_s.size < TIME.size
    assert series.channel_name == data.channel.name
    restored = result_from_frame(spec, data.channel, data.source, result_to_frame(result, infos))
    assert restored is not None
    np.testing.assert_array_equal(restored.series[0].time_s, series.time_s)
    np.testing.assert_array_equal(restored.series[0].values, series.values)


def test_filled_samples_get_markers_that_survive_the_cache(hierarchy: Hierarchy) -> None:
    """補間した点には印を付け、キャッシュから戻しても印が残る（埋めた点を隠さない）。"""
    values = NOISY.copy()
    values[10:15] = np.nan
    data = _data(hierarchy, values)
    spec = _spec(data, AnalysisKind.FILL_GAPS, FillGapsParams())
    result = run_analysis(spec, data)
    series = result.series[0]
    assert series.marker_times_s is not None
    np.testing.assert_array_equal(series.marker_times_s, TIME[10:15])
    assert "欠損 5 個のうち 5 個を補間しました" in result.headline
    infos = output_infos(spec, data.channel)
    restored = result_from_frame(spec, data.channel, data.source, result_to_frame(result, infos))
    assert restored is not None
    assert restored.series[0].marker_times_s is not None
    np.testing.assert_array_equal(restored.series[0].marker_times_s, TIME[10:15])


def test_statistics_result_has_metrics_chart_and_markers(hierarchy: Hierarchy) -> None:
    """区間統計は時系列を出さず、指標の表・分布・最大と最小の時刻の印を返す。"""
    data = _data(hierarchy, NOISY, "N")
    spec = _spec(data, AnalysisKind.STATISTICS, StatisticsParams())
    assert output_infos(spec, data.channel) == []
    result = run_analysis(spec, data)
    values = _metric_values(result)
    assert result.series == []
    assert values["mean"] == pytest.approx(float(np.mean(NOISY)))
    assert values["std"] == pytest.approx(float(np.std(NOISY, ddof=1)))
    assert values["rms"] == pytest.approx(float(np.sqrt(np.mean(NOISY**2))))
    assert values["peak_to_peak"] == pytest.approx(float(np.ptp(NOISY)))
    assert result.charts[0].title == "値の分布（ヒストグラム）"
    assert [marker.label for marker in result.time_markers] == ["最大", "最小"]
    assert "平均" in result.headline


def test_two_channel_labels_use_standard_terms(hierarchy: Hierarchy) -> None:
    """図と指標の用語は標準の用語を使う。"""
    data = _data(hierarchy, STEP_INPUT, "N")
    output = _output_channel(hierarchy, STEP_OUTPUT)
    spec = _two_channel_spec(data, output, AnalysisKind.STEP_RESPONSE, StepResponseParams())
    result = run_analysis(spec, data, output_data=output)
    labels = {metric.label for metric in result.metrics}
    assert {"遅れ時間 Td", "立ち上がり時間 Tr", "時定数 τ", "むだ時間 L", "整定時間 Ts"} <= labels
    assert {"オーバーシュート", "定常ゲイン"} <= labels
