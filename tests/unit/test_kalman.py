"""analysis/kalman.py（線形 KF・EKF・適応 KF）のテスト。

答えが分かる問題（一定値の推定 = 標本平均、既存ライブラリ filterpy の結果）との一致と、
真値が分かる信号に対する推定精度で確認する。閾値は、観測ノイズのみの誤差や
単純な差分による速度推定の誤差と比べて、フィルタが明確に改善していることを表す。
"""

import numpy as np
import pytest
from filterpy.kalman import KalmanFilter as FilterpyKalmanFilter

from experiment_workbench.analysis import (
    AnalysisCancelledError,
    AnalysisInputError,
    KalmanFilter,
    KinematicSystem,
    LinearSystem,
    NonlinearSystem,
    PolynomialObservationSystem,
    kinematic_process_noise,
    kinematic_transition_matrix,
    system_from_settings,
)
from experiment_workbench.model import (
    AdaptiveNoiseMethod,
    AdaptiveNoiseParams,
    KalmanInitialState,
    KalmanModelKind,
    KalmanResult,
    KalmanSettings,
    KinematicSystemParams,
    LinearSystemParams,
    PolynomialObservationParams,
)
from experiment_workbench.validation import FloatArray

DT = 0.01  # 等間隔テストの時間間隔 [s]
BEACON_HEIGHT = 1.0  # EKF テストで距離を測る基準点の高さ [m]
POLYNOMIAL = (0.5, 2.0, 0.3)  # 多項式の観測モデルのテストで使う係数 z = c0 + c1 x + c2 x²

INITIAL_CV = KalmanInitialState(state=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 10.0]])
"""等速度モデル用の初期状態（位置 0・速度 0、不確かさは大きめ）。"""


def _rmse(estimate: FloatArray, truth: FloatArray) -> float:
    """二乗平均平方根誤差を返す。

    Args:
        estimate: 推定値。
        truth: 真値。

    Returns:
        RMSE。
    """
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def _cv_linear_params(q: float, r: float) -> LinearSystemParams:
    """時間間隔 DT の等速度モデルを定数行列で表したパラメータを作る。

    Args:
        q: プロセス雑音のパワースペクトル密度。
        r: 観測雑音の標準偏差。

    Returns:
        線形状態空間モデルのパラメータ。
    """
    return LinearSystemParams(
        transition=kinematic_transition_matrix(2, DT).tolist(),
        observation=[[1.0, 0.0]],
        process_noise=kinematic_process_noise(2, q, DT).tolist(),
        measurement_noise=[[r**2]],
    )


def _noisy_sine_on_non_uniform_time(
    noise_std: float, seed: int = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """不等間隔の時刻で、正弦波の位置にノイズを加えた観測を作る。

    Args:
        noise_std: 観測ノイズの標準偏差。
        seed: 乱数シード。

    Returns:
        (時刻, 観測, 真の位置, 真の速度)。位置 = sin(πt)、速度 = π cos(πt)。
    """
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.005, 0.015, 1000))
    position = np.sin(np.pi * t)
    velocity = np.pi * np.cos(np.pi * t)
    return t, position + rng.normal(0.0, noise_std, t.size), position, velocity


# --- EKF テスト用のモデル関数（関数内関数を使わないためモジュールに置く） ---


def _cv_transition(state: FloatArray, dt: float) -> FloatArray:
    """等速度モデルの状態遷移 f(x, Δt) = F(Δt) x。

    Args:
        state: 状態 [位置, 速度]。
        dt: 時間幅 [s]。

    Returns:
        予測状態。
    """
    return kinematic_transition_matrix(2, dt) @ state


def _cv_transition_jacobian(state: FloatArray, dt: float) -> FloatArray:
    """等速度モデルの状態遷移のヤコビアン（線形なので F(Δt) そのもの）。

    Args:
        state: 状態（使わない）。
        dt: 時間幅 [s]。

    Returns:
        F(Δt)。
    """
    return kinematic_transition_matrix(2, dt)


def _position_observation(state: FloatArray) -> FloatArray:
    """位置を観測する線形の観測関数 h(x) = x[0]。

    Args:
        state: 状態 [位置, 速度]。

    Returns:
        予測観測（長さ 1）。
    """
    return state[:1].copy()


def _position_observation_jacobian(state: FloatArray) -> FloatArray:
    """位置観測のヤコビアン [[1, 0]]。

    Args:
        state: 状態（使わない）。

    Returns:
        1×2 行列。
    """
    return np.array([[1.0, 0.0]])


def _range_observation(state: FloatArray) -> FloatArray:
    """高さ BEACON_HEIGHT の基準点までの距離 h(x) = √(x² + d²)（非線形な観測）。

    Args:
        state: 状態 [位置, 速度]。

    Returns:
        予測観測（長さ 1）。
    """
    return np.array([np.hypot(state[0], BEACON_HEIGHT)])


def _range_observation_jacobian(state: FloatArray) -> FloatArray:
    """距離観測のヤコビアン [[x / √(x² + d²), 0]]。

    Args:
        state: 状態 [位置, 速度]。

    Returns:
        1×2 行列。
    """
    return np.array([[state[0] / np.hypot(state[0], BEACON_HEIGHT), 0.0]])


def _wrong_size_observation(state: FloatArray) -> FloatArray:
    """観測次元（1）と合わない長さ 2 の配列を返す、誤った観測関数。

    Args:
        state: 状態。

    Returns:
        長さ 2 の配列。
    """
    return state.copy()


def _polynomial_observation(state: FloatArray) -> FloatArray:
    """係数 POLYNOMIAL の 2 次式による観測 h(x) = c0 + c1 x + c2 x²。

    Args:
        state: 状態 [x, x の微分]。

    Returns:
        予測観測（長さ 1）。
    """
    c0, c1, c2 = POLYNOMIAL
    x = state[0]
    return np.array([c0 + c1 * x + c2 * x**2])


def _polynomial_observation_jacobian(state: FloatArray) -> FloatArray:
    """2 次式の観測のヤコビアン [[c1 + 2 c2 x, 0]]。

    Args:
        state: 状態 [x, x の微分]。

    Returns:
        1×2 行列。
    """
    _, c1, c2 = POLYNOMIAL
    return np.array([[c1 + 2.0 * c2 * state[0], 0.0]])


class _ProgressRecorder:
    """KalmanFilter.run に渡す進捗の関数。受け取った進捗を記録し、指定回数で中止を指示する。

    関数内関数を使わずに状態を持たせるため、呼び出し可能なクラスにしている。

    Attributes:
        fractions: 受け取った進捗（0〜1）。
    """

    def __init__(self, stop_at_call: int | None = None) -> None:
        """中止を指示する呼び出し回数を受け取る。

        Args:
            stop_at_call: この回数目の呼び出しで False（中止）を返す。None なら中止しない。
        """
        self._stop_at_call = stop_at_call
        self.fractions: list[float] = []

    def __call__(self, fraction: float) -> bool:
        """進捗を記録し、続けるかどうかを返す。

        Args:
            fraction: 進捗（0〜1）。

        Returns:
            続けるなら True。
        """
        self.fractions.append(fraction)
        return self._stop_at_call is None or len(self.fractions) < self._stop_at_call


# --- 運動学モデルの行列 -------------------------------------------------------


def test_kinematic_matrices_match_closed_forms() -> None:
    """等速度・等加速度モデルの F・Q が、教科書の閉じた式と一致する。"""
    dt, q = 0.1, 2.0
    np.testing.assert_allclose(
        kinematic_transition_matrix(2, dt), np.array([[1.0, dt], [0.0, 1.0]])
    )
    np.testing.assert_allclose(
        kinematic_process_noise(2, q, dt), q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    )
    np.testing.assert_allclose(
        kinematic_transition_matrix(3, dt),
        np.array([[1.0, dt, dt**2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]),
    )
    expected_q3 = q * np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    np.testing.assert_allclose(kinematic_process_noise(3, q, dt), expected_q3)


@pytest.mark.parametrize("n_states", [2, 3])
def test_kinematic_process_noise_is_positive_semidefinite(n_states: int) -> None:
    """プロセス雑音共分散は対称・半正定値になる。"""
    noise = kinematic_process_noise(n_states, 5.0, 0.02)
    np.testing.assert_allclose(noise, noise.T)
    assert np.linalg.eigvalsh(noise).min() > -1e-15


# --- 線形 KF ------------------------------------------------------------------


def test_constant_estimation_equals_running_mean() -> None:
    """一定値を Q=0・十分大きな初期共分散で推定すると、推定値は標本平均、分散は R/(k+1) になる。"""
    rng = np.random.default_rng(0)
    r = 0.2
    z = 5.0 + rng.normal(0.0, r, 50)
    t = np.arange(z.size, dtype=np.float64) * DT
    system = LinearSystem(
        LinearSystemParams(
            transition=[[1.0]],
            observation=[[1.0]],
            process_noise=[[0.0]],
            measurement_noise=[[r**2]],
        )
    )
    initial = KalmanInitialState(state=[0.0], covariance=[[1e12]])
    result = KalmanFilter(system, initial).run(t, z)
    counts = np.arange(1, z.size + 1)
    np.testing.assert_allclose(result.states[:, 0], np.cumsum(z) / counts, rtol=1e-9)
    np.testing.assert_allclose(result.covariances[:, 0, 0], r**2 / counts, rtol=1e-6)


def test_linear_kf_matches_filterpy() -> None:
    """同じ等速度モデルで、状態・共分散が既存ライブラリ filterpy の結果と一致する。"""
    rng = np.random.default_rng(1)
    t = np.arange(500, dtype=np.float64) * DT
    z = np.sin(t) + rng.normal(0.0, 0.05, t.size)
    params = _cv_linear_params(q=5.0, r=0.05)
    result = KalmanFilter(LinearSystem(params), INITIAL_CV).run(t, z)

    reference = FilterpyKalmanFilter(dim_x=2, dim_z=1)
    reference.x = np.array([[0.0], [0.0]])
    reference.P = np.array(INITIAL_CV.covariance)
    reference.F = np.array(params.transition)
    reference.H = np.array(params.observation)
    reference.Q = np.array(params.process_noise)
    reference.R = np.array(params.measurement_noise)
    states: list[FloatArray] = []
    covariances: list[FloatArray] = []
    for k, measurement in enumerate(z):
        if k > 0:  # 先頭時刻は初期状態を事前推定として更新のみ行う（本実装と同じ手順）
            reference.predict()
        reference.update(measurement)
        states.append(reference.x.ravel().copy())
        covariances.append(reference.P.copy())

    np.testing.assert_allclose(result.states, np.array(states), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.covariances, np.array(covariances), rtol=1e-9, atol=1e-12)


def test_two_dimensional_observation_matches_filterpy() -> None:
    """位置と速度を同時に観測する（観測が 2 次元の）場合も、filterpy の結果と一致する。

    観測が 1 次元のときはゲインを割り算で、多次元のときは連立方程式で求めるため、
    多次元の計算経路も確認する。
    """
    rng = np.random.default_rng(4)
    t = np.arange(300, dtype=np.float64) * DT
    z = np.column_stack(
        [np.sin(t) + rng.normal(0.0, 0.05, t.size), np.cos(t) + rng.normal(0.0, 0.2, t.size)]
    )
    params = LinearSystemParams(
        transition=kinematic_transition_matrix(2, DT).tolist(),
        observation=[[1.0, 0.0], [0.0, 1.0]],
        process_noise=kinematic_process_noise(2, 5.0, DT).tolist(),
        measurement_noise=[[0.05**2, 0.0], [0.0, 0.2**2]],
    )
    result = KalmanFilter(LinearSystem(params), INITIAL_CV).run(t, z)

    reference = FilterpyKalmanFilter(dim_x=2, dim_z=2)
    reference.x = np.array([[0.0], [0.0]])
    reference.P = np.array(INITIAL_CV.covariance)
    reference.F = np.array(params.transition)
    reference.H = np.array(params.observation)
    reference.Q = np.array(params.process_noise)
    reference.R = np.array(params.measurement_noise)
    states: list[FloatArray] = []
    for k, measurement in enumerate(z):
        if k > 0:
            reference.predict()
        reference.update(measurement.reshape(2, 1))
        states.append(reference.x.ravel().copy())
    np.testing.assert_allclose(result.states, np.array(states), rtol=1e-9, atol=1e-12)


def test_progress_is_reported_until_the_end() -> None:
    """進捗の関数は一定のステップごとに 0〜1 の増える値で呼ばれ、結果は渡さない場合と同じ。"""
    t = np.arange(5000, dtype=np.float64) * DT
    z = np.sin(t)
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=1.0, measurement_noise_std=0.1)
    )
    recorder = _ProgressRecorder()
    with_progress = KalmanFilter(system, INITIAL_CV).run(t, z, recorder)
    without_progress = KalmanFilter(system, INITIAL_CV).run(t, z)
    assert recorder.fractions[0] == 0.0
    assert len(recorder.fractions) >= 2
    assert all(0.0 <= value < 1.0 for value in recorder.fractions)
    assert recorder.fractions == sorted(recorder.fractions)
    np.testing.assert_array_equal(with_progress.states, without_progress.states)


def test_progress_returning_false_cancels_filter() -> None:
    """進捗の関数が False を返すと、計算を途中でやめて AnalysisCancelledError を送出する。"""
    t = np.arange(5000, dtype=np.float64) * DT
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=1.0, measurement_noise_std=0.1)
    )
    recorder = _ProgressRecorder(stop_at_call=2)
    with pytest.raises(AnalysisCancelledError):
        KalmanFilter(system, INITIAL_CV).run(t, np.sin(t), recorder)
    assert len(recorder.fractions) == 2  # 中止を指示した後は計算を続けない


def test_kinematic_matrices_are_cached_read_only_and_stay_correct() -> None:
    """運動学モデルの F・Q は時間幅ごとに覚えて使い回し、書き換えられないようにする。

    覚えておく数の上限を超えるほど多くの時間幅を使っても、正しい行列を返す。
    """
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=2.0, measurement_noise_std=0.1)
    )
    _, first = system.predict(np.zeros(2), DT)
    _, again = system.predict(np.zeros(2), DT)
    assert again is first  # 同じ時間幅では計算し直さない
    assert not first.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first[0, 1] = 0.0
    for step in range(1, 301):
        dt = 0.001 * step
        _, transition = system.predict(np.zeros(2), dt)
        np.testing.assert_array_equal(transition, kinematic_transition_matrix(2, dt))
        np.testing.assert_array_equal(system.process_noise(dt), kinematic_process_noise(2, 2.0, dt))


def test_kinematic_filter_on_non_uniform_time_beats_raw_measurement() -> None:
    """不等間隔の観測でも、位置の誤差は観測ノイズより小さく、速度は単純な差分よりずっと正確になる。"""
    t, z, position, velocity = _noisy_sine_on_non_uniform_time(noise_std=0.05)
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=10.0, measurement_noise_std=0.05)
    )
    result = KalmanFilter(system, INITIAL_CV).run(t, z)
    burn_in = 100  # 初期値の影響が残る先頭は評価から除く
    position_rmse = _rmse(result.states[burn_in:, 0], position[burn_in:])
    assert position_rmse < 0.6 * _rmse(z[burn_in:], position[burn_in:])
    naive_velocity = np.gradient(z, t)
    velocity_rmse = _rmse(result.states[burn_in:, 1], velocity[burn_in:])
    assert velocity_rmse < _rmse(naive_velocity[burn_in:], velocity[burn_in:]) / 4
    assert result.updated.all()
    assert not result.noise_floor_applied.any()


def test_nan_measurements_are_skipped_and_uncertainty_grows() -> None:
    """NaN の観測は更新せず（予測のみ）、その間は共分散が増え続ける。"""
    t, z, _, _ = _noisy_sine_on_non_uniform_time(noise_std=0.05)
    z = z.copy()
    z[300:320] = np.nan
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=10.0, measurement_noise_std=0.05)
    )
    result = KalmanFilter(system, INITIAL_CV).run(t, z)
    assert not result.updated[300:320].any()
    assert result.updated[:300].all()
    assert result.updated[320:].all()
    assert np.isnan(result.innovations[300:320]).all()
    traces = np.trace(result.covariances, axis1=1, axis2=2)
    assert np.all(np.diff(traces[299:320]) > 0)
    assert np.isfinite(result.states).all()


def test_linear_system_requires_uniform_sampling() -> None:
    """定数行列のモデルに不等間隔の時刻を渡すとエラーになる。

    F・Q が 1 サンプル分の時間を前提とするため。
    """
    t, z, _, _ = _noisy_sine_on_non_uniform_time(noise_std=0.05)
    system = LinearSystem(_cv_linear_params(q=5.0, r=0.05))
    with pytest.raises(AnalysisInputError, match="一定ではありません"):
        KalmanFilter(system, INITIAL_CV).run(t, z)


def test_initial_state_dimension_must_match_model() -> None:
    """初期状態の次元がモデルと合わない場合はエラーになる。"""
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=1.0, measurement_noise_std=0.1)
    )
    initial = KalmanInitialState(state=[0.0, 0.0, 0.0], covariance=np.eye(3).tolist())
    with pytest.raises(AnalysisInputError, match="次元"):
        KalmanFilter(system, initial)


@pytest.mark.parametrize(
    ("measurements", "message"),
    [(np.zeros((10, 2)), "形"), (np.array([0.0] * 9 + [np.inf]), "無限大")],
)
def test_invalid_measurements_are_rejected(measurements: FloatArray, message: str) -> None:
    """観測の形が合わない、または無限大を含む場合はエラーになる。"""
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=1.0, measurement_noise_std=0.1)
    )
    t = np.arange(10, dtype=np.float64) * DT
    with pytest.raises(AnalysisInputError, match=message):
        KalmanFilter(system, INITIAL_CV).run(t, measurements)


# --- EKF ----------------------------------------------------------------------


def test_ekf_with_linear_functions_equals_linear_kf() -> None:
    """線形な関数で定義した EKF は、同じ行列の線形 KF と同じ結果になる。"""
    rng = np.random.default_rng(2)
    t = np.arange(300, dtype=np.float64) * DT
    z = np.cos(t) + rng.normal(0.0, 0.05, t.size)
    params = _cv_linear_params(q=5.0, r=0.05)
    linear = KalmanFilter(LinearSystem(params), INITIAL_CV).run(t, z)
    nonlinear_system = NonlinearSystem(
        _cv_transition,
        _cv_transition_jacobian,
        _position_observation,
        _position_observation_jacobian,
        process_noise=np.array(params.process_noise),
        measurement_noise=np.array(params.measurement_noise),
    )
    ekf = KalmanFilter(nonlinear_system, INITIAL_CV).run(t, z)
    np.testing.assert_allclose(ekf.states, linear.states, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ekf.covariances, linear.covariances, rtol=1e-12, atol=1e-14)


def test_ekf_tracks_position_from_nonlinear_range_measurement() -> None:
    """基準点までの距離（非線形な観測）から、EKF で位置を推定できる。

    比較対象は、距離を逆算しただけの位置 √(z² - d²)。EKF はこれより誤差が小さくなる。
    """
    rng = np.random.default_rng(3)
    t = np.arange(1000, dtype=np.float64) * DT
    position = 3.0 + np.sin(np.pi * t)
    z = np.hypot(position, BEACON_HEIGHT) + rng.normal(0.0, 0.02, t.size)
    system = NonlinearSystem(
        _cv_transition,
        _cv_transition_jacobian,
        _range_observation,
        _range_observation_jacobian,
        process_noise=kinematic_process_noise(2, 10.0, DT),
        measurement_noise=np.array([[0.02**2]]),
    )
    initial = KalmanInitialState(state=[3.0, 0.0], covariance=[[1.0, 0.0], [0.0, 10.0]])
    result = KalmanFilter(system, initial).run(t, z)
    burn_in = 100
    inverted = np.sqrt(z**2 - BEACON_HEIGHT**2)
    ekf_rmse = _rmse(result.states[burn_in:, 0], position[burn_in:])
    assert ekf_rmse < 0.7 * _rmse(inverted[burn_in:], position[burn_in:])


def test_nonlinear_system_reports_wrong_function_output_shape() -> None:
    """ユーザー定義関数が観測次元と合わない形を返したら、推定を続けずにエラーにする。"""
    system = NonlinearSystem(
        _cv_transition,
        _cv_transition_jacobian,
        _wrong_size_observation,
        _position_observation_jacobian,
        process_noise=kinematic_process_noise(2, 1.0, DT),
        measurement_noise=np.array([[0.01]]),
    )
    t = np.arange(10, dtype=np.float64) * DT
    with pytest.raises(AnalysisInputError, match=r"h\(x\)"):
        KalmanFilter(system, INITIAL_CV).run(t, np.zeros(10))


def test_nonlinear_system_rejects_invalid_noise_covariance() -> None:
    """観測雑音共分散が正定値でなければ、モデルを作れない。"""
    with pytest.raises(ValueError, match="正定値"):
        NonlinearSystem(
            _cv_transition,
            _cv_transition_jacobian,
            _position_observation,
            _position_observation_jacobian,
            process_noise=np.zeros((2, 2)),
            measurement_noise=np.array([[0.0]]),
        )


def _polynomial_params(noise_std: float) -> PolynomialObservationParams:
    """係数 POLYNOMIAL の多項式の観測モデルの条件を作る。

    Args:
        noise_std: 観測雑音の標準偏差。

    Returns:
        多項式の観測モデルの条件。
    """
    return PolynomialObservationParams(
        process_noise_density=10.0, measurement_noise_std=noise_std, coefficients=list(POLYNOMIAL)
    )


def test_polynomial_observation_system_equals_generic_ekf() -> None:
    """多項式の観測モデルは、同じ関数を Python で与えた汎用の EKF と同じ結果になる。"""
    rng = np.random.default_rng(5)
    t = np.arange(400, dtype=np.float64) * DT
    x = 2.0 + np.sin(np.pi * t)
    z = POLYNOMIAL[0] + POLYNOMIAL[1] * x + POLYNOMIAL[2] * x**2 + rng.normal(0.0, 0.05, t.size)
    initial = KalmanInitialState(state=[2.0, 0.0], covariance=[[1.0, 0.0], [0.0, 10.0]])
    builtin = KalmanFilter(PolynomialObservationSystem(_polynomial_params(0.05)), initial).run(t, z)
    generic_system = NonlinearSystem(
        _cv_transition,
        _cv_transition_jacobian,
        _polynomial_observation,
        _polynomial_observation_jacobian,
        process_noise=kinematic_process_noise(2, 10.0, DT),
        measurement_noise=np.array([[0.05**2]]),
    )
    generic = KalmanFilter(generic_system, initial).run(t, z)
    np.testing.assert_allclose(builtin.states, generic.states, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(builtin.covariances, generic.covariances, rtol=1e-10, atol=1e-12)


def test_polynomial_observation_ekf_recovers_hidden_quantity() -> None:
    """センサの出力 z が物理量 x の 2 次式のとき、EKF で x とその変化率を推定できる。

    比較対象は、各時刻の z から 2 次方程式を解いて逆算しただけの x。EKF はこれより誤差が
    小さく、変化率も逆算値の差分より正確になる（電圧から張力を推定する場面を想定）。
    """
    rng = np.random.default_rng(6)
    t = np.arange(1000, dtype=np.float64) * DT
    x = 2.0 + np.sin(np.pi * t)
    rate = np.pi * np.cos(np.pi * t)
    c0, c1, c2 = POLYNOMIAL
    z = c0 + c1 * x + c2 * x**2 + rng.normal(0.0, 0.1, t.size)
    initial = KalmanInitialState(state=[2.0, 0.0], covariance=[[1.0, 0.0], [0.0, 10.0]])
    result = KalmanFilter(PolynomialObservationSystem(_polynomial_params(0.1)), initial).run(t, z)
    inverted = (-c1 + np.sqrt(c1**2 - 4.0 * c2 * (c0 - z))) / (2.0 * c2)
    burn_in = 100
    assert _rmse(result.states[burn_in:, 0], x[burn_in:]) < 0.7 * _rmse(
        inverted[burn_in:], x[burn_in:]
    )
    naive_rate = np.gradient(inverted, t)
    assert _rmse(result.states[burn_in:, 1], rate[burn_in:]) < 0.25 * _rmse(
        naive_rate[burn_in:], rate[burn_in:]
    )


_INITIAL_2D = {"state": [0.0, 0.0], "covariance": [[1.0, 0.0], [0.0, 1.0]]}
"""system_from_settings のテストで使う 2 次元の初期状態。"""


@pytest.mark.parametrize(
    ("settings", "expected_type"),
    [
        (
            {
                "model_kind": KalmanModelKind.KINEMATIC,
                "kinematic": {"process_noise_density": 1.0, "measurement_noise_std": 0.1},
            },
            KinematicSystem,
        ),
        (
            {
                "model_kind": KalmanModelKind.LINEAR,
                "linear": {
                    "transition": [[1.0, DT], [0.0, 1.0]],
                    "observation": [[1.0, 0.0]],
                    "process_noise": [[1e-6, 0.0], [0.0, 1e-4]],
                    "measurement_noise": [[0.01]],
                },
            },
            LinearSystem,
        ),
        (
            {
                "model_kind": KalmanModelKind.POLYNOMIAL_OBSERVATION,
                "polynomial_observation": {
                    "process_noise_density": 1.0,
                    "measurement_noise_std": 0.1,
                    "coefficients": [0.0, 1.0],
                },
            },
            PolynomialObservationSystem,
        ),
    ],
    ids=["kinematic", "linear", "polynomial_observation"],
)
def test_system_from_settings_builds_selected_model(
    settings: dict[str, object], expected_type: type
) -> None:
    """保存できる条件から、選んだ種類の状態空間モデルを作る。"""
    system = system_from_settings(
        KalmanSettings.model_validate(settings | {"initial": _INITIAL_2D})
    )
    assert isinstance(system, expected_type)
    assert system.state_dim == 2
    assert system.measurement_dim == 1


# --- 適応 KF ------------------------------------------------------------------


def _noise_change_scenario() -> tuple[FloatArray, FloatArray, FloatArray]:
    """途中で観測ノイズが 0.01 から 0.05 に増える信号を作る（100 Hz、40 秒）。

    Returns:
        (時刻, 観測, 真の位置)。20 秒で標準偏差が切り替わる。
    """
    rng = np.random.default_rng(11)
    t = np.arange(4000, dtype=np.float64) / 100.0
    position = np.sin(2 * np.pi * 0.2 * t)
    noise_std = np.where(t < 20.0, 0.01, 0.05)
    return t, position + rng.normal(0.0, 1.0, t.size) * noise_std, position


def _run_noise_change(adaptive: AdaptiveNoiseParams | None) -> KalmanResult:
    """ノイズが切り替わる信号に、R の初期値を前半のノイズ（0.01）としてフィルタをかける。

    Args:
        adaptive: 適応推定の条件。None なら R は一定。

    Returns:
        推定結果。
    """
    t, z, _ = _noise_change_scenario()
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=1.0, measurement_noise_std=0.01)
    )
    return KalmanFilter(system, INITIAL_CV, adaptive=adaptive).run(t, z)


@pytest.mark.parametrize(
    "adaptive",
    [
        AdaptiveNoiseParams(method=AdaptiveNoiseMethod.INNOVATION_WINDOW, window=400),
        AdaptiveNoiseParams(method=AdaptiveNoiseMethod.SAGE_HUSA, forgetting_factor=0.995),
    ],
    ids=["innovation_window", "sage_husa"],
)
def test_adaptive_kf_tracks_measurement_noise_change(adaptive: AdaptiveNoiseParams) -> None:
    """適応 KF は観測ノイズの変化に追従して R を推定し、固定 R の KF より位置の誤差が小さくなる。"""
    t, _, position = _noise_change_scenario()
    akf = _run_noise_change(adaptive)
    fixed = _run_noise_change(None)
    estimated_std = np.sqrt(akf.measurement_noise[:, 0, 0])
    assert estimated_std[1990] == pytest.approx(0.01, rel=0.15)  # 切り替え直前（19.9 秒）
    assert estimated_std[-1] == pytest.approx(0.05, rel=0.15)  # 終了時（40 秒）
    second_half = t >= 25.0
    akf_rmse = _rmse(akf.states[second_half, 0], position[second_half])
    fixed_rmse = _rmse(fixed.states[second_half, 0], position[second_half])
    assert akf_rmse < 0.85 * fixed_rmse
    np.testing.assert_array_equal(fixed.measurement_noise[:, 0, 0], 0.01**2)


def test_innovation_window_uses_initial_noise_until_window_is_full() -> None:
    """窓付き推定は、イノベーションが窓長だけたまるまでは R の初期値を使う。"""
    window = 400
    akf = _run_noise_change(
        AdaptiveNoiseParams(method=AdaptiveNoiseMethod.INNOVATION_WINDOW, window=window)
    )
    np.testing.assert_array_equal(akf.measurement_noise[: window - 1, 0, 0], 0.01**2)
    assert akf.measurement_noise[window - 1, 0, 0] != 0.01**2


def test_noise_floor_is_applied_and_recorded() -> None:
    """推定した R が正定値でなくなる場合は固有値を下限で抑え、その時刻を結果に記録する。

    ノイズのない一定値を観測すると、イノベーションがほぼ 0 になり R = C - H P⁻ Hᵀ が負になる。
    """
    t = np.arange(50, dtype=np.float64) * DT
    z = np.full(t.size, 5.0)
    system = LinearSystem(
        LinearSystemParams(
            transition=[[1.0]],
            observation=[[1.0]],
            process_noise=[[0.0]],
            measurement_noise=[[1.0]],
        )
    )
    adaptive = AdaptiveNoiseParams(
        method=AdaptiveNoiseMethod.INNOVATION_WINDOW, window=5, min_variance=1e-6
    )
    initial = KalmanInitialState(state=[0.0], covariance=[[1.0]])
    result = KalmanFilter(system, initial, adaptive=adaptive).run(t, z)
    assert result.noise_floor_applied.any()
    assert result.measurement_noise[result.noise_floor_applied, 0, 0] == pytest.approx(1e-6)
    assert np.isfinite(result.states).all()
