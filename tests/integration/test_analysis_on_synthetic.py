"""合成データに解析コアを適用する結合テスト。

CSV 読み込み → 時刻の明示的な換算 → 解析 → 真値と比較、の流れを確かめる。
実データと同じく、CSV の観測列だけを解析の入力に使い、真値列（true_*）は評価にだけ使う。
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiment_workbench.analysis import (
    AnalysisInputError,
    ButterworthFilter,
    Differentiator,
    Integrator,
    KalmanFilter,
    KinematicSystem,
)
from experiment_workbench.model import (
    ButterworthKind,
    ButterworthParams,
    DerivativeParams,
    IntegralParams,
    KalmanInitialState,
    KinematicSystemParams,
    SyntheticConfig,
    SyntheticManifest,
    SyntheticSourceRecord,
)
from experiment_workbench.synthetic import SyntheticDataGenerator
from experiment_workbench.validation import FloatArray

Dataset = tuple[Path, SyntheticManifest]
"""(出力ディレクトリ, 目録) の組。"""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """10 秒 × 3 試技の合成データを 1 回だけ生成する。

    Args:
        tmp_path_factory: pytest の一時ディレクトリ生成器。

    Returns:
        (出力ディレクトリ, 目録) の組。
    """
    output = tmp_path_factory.mktemp("analysis") / "data"
    config = SyntheticConfig(duration_s=10.0, n_trials=3)
    return output, SyntheticDataGenerator(config).generate(output)


def _load(output: Path, source: SyntheticSourceRecord) -> tuple[FloatArray, pl.DataFrame]:
    """CSV を読み込み、時刻列を秒に換算する（単位の換算は呼び出し側で明示的に行う方針）。

    Args:
        output: 合成データの出力ディレクトリ。
        source: 読み込むファイルの記録。

    Returns:
        (時刻 [s], 読み込んだ表)。
    """
    frame = pl.read_csv(output / source.relpath)
    raw = frame[source.time_column].to_numpy().astype(np.float64)
    scale = {"s": 1.0, "ms": 1e-3}[source.time_unit]
    return raw * scale, frame


def _column(frame: pl.DataFrame, name: str) -> FloatArray:
    """表の列を float64 の配列として取り出す。

    Args:
        frame: 表。
        name: 列名。

    Returns:
        float64 の配列。
    """
    return frame[name].to_numpy().astype(np.float64)


def _rmse(estimate: FloatArray, truth: FloatArray) -> float:
    """二乗平均平方根誤差を返す。

    Args:
        estimate: 推定値。
        truth: 真値。

    Returns:
        RMSE。
    """
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def test_lowpass_removes_vibration_from_acceleration(dataset: Dataset) -> None:
    """加速度の観測に LPF をかけると、高周波振動とノイズが除かれ、真値 + バイアスに近づく。"""
    output, manifest = dataset
    high = manifest.trials[0].sources[0]
    t, frame = _load(output, high)
    params = ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=5.0)
    filtered = ButterworthFilter(params).apply(t, _column(frame, "accel"))
    target = _column(frame, "true_accel") + high.noise_parameters["bias"]
    edge = 1000  # 端点の過渡応答（1 秒分）を除く
    filtered_rmse = _rmse(filtered[edge:-edge], target[edge:-edge])
    raw_rmse = _rmse(_column(frame, "accel")[edge:-edge], target[edge:-edge])
    assert filtered_rmse < 0.1 * raw_rmse


def test_integrating_true_acceleration_reproduces_velocity(dataset: Dataset) -> None:
    """真の加速度を、真の初速度を初期値として積分すると真の速度になる。"""
    output, manifest = dataset
    t, frame = _load(output, manifest.trials[0].sources[0])
    velocity = _column(frame, "true_velocity")
    params = IntegralParams(initial_value=float(velocity[0]))
    integrated = Integrator(params).apply(t, _column(frame, "true_accel"))
    np.testing.assert_allclose(integrated, velocity, atol=1e-3)


def test_differentiating_true_position_reproduces_velocity(dataset: Dataset) -> None:
    """低レート側（ミリ秒の時刻を秒に換算）の真の位置を微分すると、理論誤差の範囲で真の速度になる。

    中心差分の打ち切り誤差は内部で h²|x'''|/6、端点（2 次精度の片側差分）で h²|x'''|/3 以下。
    |x'''| の上限は運動パラメータから Σ A (2πf)³ で求め、CSV の小数 6 桁への丸めによる誤差
    （位置 5e-7 m を h で割った程度）を足して許容値にする。
    """
    output, manifest = dataset
    trial = manifest.trials[0]
    t, frame = _load(output, trial.sources[1])
    derivative = Differentiator(DerivativeParams()).apply(t, _column(frame, "true_position"))
    velocity = _column(frame, "true_velocity")

    h = float(np.median(np.diff(t)))
    jerk_bound = sum(
        abs(c.amplitude) * (2 * np.pi * c.frequency_hz) ** 3
        for c in trial.motion.position_components
    )
    rounding = 2e-6 / h + 1e-6
    np.testing.assert_allclose(
        derivative[1:-1], velocity[1:-1], atol=h**2 * jerk_bound / 6 + rounding
    )
    np.testing.assert_allclose(
        derivative[[0, -1]], velocity[[0, -1]], atol=h**2 * jerk_bound / 3 + rounding
    )


def test_kalman_filter_handles_nan_measurements_in_synthetic_data(dataset: Dataset) -> None:
    """NaN を含む位置の観測（3 試技目）でも、KF は欠測区間を予測でつないで推定する。

    観測を更新に使った時刻は NaN でない時刻と一致し、位置の誤差は観測ノイズより小さくなる。
    """
    output, manifest = dataset
    low = manifest.trials[2].sources[1]
    t, frame = _load(output, low)
    measured = _column(frame, "position")
    noise_std = low.noise_parameters["noise_std"]
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=10.0, measurement_noise_std=noise_std)
    )
    initial = KalmanInitialState(
        state=[float(measured[0]), 0.0], covariance=[[noise_std**2, 0.0], [0.0, 10.0]]
    )
    result = KalmanFilter(system, initial).run(t, measured)

    np.testing.assert_array_equal(result.updated, ~np.isnan(measured))
    assert (~result.updated).sum() > 0  # 異常を仕込んだ試技であることの確認
    position = _column(frame, "true_position")
    burn_in = 100
    assert _rmse(result.states[burn_in:, 0], position[burn_in:]) < noise_std


def test_gap_in_high_rate_data_is_reported_by_filter_but_not_by_integrator(
    dataset: Dataset,
) -> None:
    """欠損区間のある 2 試技目の扱いを確かめる。

    等間隔を前提とするフィルタはエラーにし、不等間隔に対応する積分は計算できる。
    """
    output, manifest = dataset
    t, frame = _load(output, manifest.trials[1].sources[0])
    accel = _column(frame, "accel")
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        ButterworthFilter(ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=5.0)).apply(
            t, accel
        )
    assert np.isfinite(Integrator(IntegralParams()).apply(t, accel)).all()
