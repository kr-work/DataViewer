"""カルマンフィルタ系（SPEC §7 A-05〜A-08）: 線形 KF・拡張 KF（EKF）・適応 KF（AKF）。

状態空間モデル::

    x_k = f(x_{k-1}, Δt_k) + w_k,   w_k ~ N(0, Q(Δt_k))
    z_k = h(x_k) + v_k,             v_k ~ N(0, R)

1 ステップの計算（EKF の形。線形モデルでは f・h が行列積になり、通常の KF と一致する）::

    予測: x⁻ = f(x, Δt),   F = ∂f/∂x（x で評価）,   P⁻ = F P Fᵀ + Q(Δt)
    更新: ν = z - h(x⁻),   H = ∂h/∂x（x⁻ で評価）,  S = H P⁻ Hᵀ + R
          K = P⁻ Hᵀ S⁻¹,   x = x⁻ + K ν
          P = (I - K H) P⁻ (I - K H)ᵀ + K R Kᵀ

共分散の更新に Joseph 形式を使うのは、丸め誤差があっても P が対称・半正定値を保ちやすいため。
先頭時刻では予測を行わず、初期状態（KalmanInitialState）を事前推定として観測更新だけを行う。
観測に NaN を含む時刻は欠測とみなし、予測のみを行う（補間などはしない）。

適応 KF（AKF）は、観測更新のたびにイノベーション ν から観測雑音共分散 R を推定し直す
（AdaptiveNoiseParams 参照）。

使うモデルは ``StateSpaceSystem`` の実装で選ぶ:
    - ``LinearSystem``: 定数行列（等間隔の時刻が必要）
    - ``KinematicSystem``: 値とその時間微分を状態とする運動学モデル（不等間隔に対応）
    - ``NonlinearSystem``: 任意の非線形関数とヤコビアン（EKF）。Python の関数で与えるため、
      条件をモデル（JSON）として保存することはできない（docs/DECISIONS.md 参照）。

このモジュールは Qt に依存しない。
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

import numpy as np

from experiment_workbench.analysis.signal_processing import (
    DEFAULT_UNIFORM_TOLERANCE,
    AnalysisInputError,
    check_uniform_sampling,
    validate_time_axis,
)
from experiment_workbench.model import (
    AdaptiveNoiseMethod,
    AdaptiveNoiseParams,
    KalmanInitialState,
    KalmanResult,
    KinematicSystemParams,
    LinearSystemParams,
)
from experiment_workbench.validation import FloatArray, check_covariance

TransitionFunction = Callable[[FloatArray, float], FloatArray]
"""状態遷移関数 f(x, Δt) → 予測状態 x⁻（長さ n の配列）。"""

TransitionJacobian = Callable[[FloatArray, float], FloatArray]
"""状態遷移関数のヤコビアン ∂f/∂x(x, Δt) → n×n 行列。"""

ObservationFunction = Callable[[FloatArray], FloatArray]
"""観測関数 h(x) → 予測観測（長さ m の配列）。"""

ObservationJacobian = Callable[[FloatArray], FloatArray]
"""観測関数のヤコビアン ∂h/∂x(x) → m×n 行列。"""


# ---------------------------------------------------------------------------
# 状態空間モデル
# ---------------------------------------------------------------------------


class StateSpaceSystem(ABC):
    """カルマンフィルタに渡す状態空間モデルの共通インターフェース。

    実装クラスは、予測（状態遷移）と観測の計算、および雑音共分散を提供する。
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """状態の次元 n。"""

    @property
    @abstractmethod
    def measurement_dim(self) -> int:
        """観測の次元 m。"""

    @property
    @abstractmethod
    def requires_uniform_sampling(self) -> bool:
        """等間隔の時刻を前提とするモデルか（True なら KalmanFilter が等間隔を確認する）。"""

    @abstractmethod
    def predict(self, state: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
        """状態を Δt だけ先へ予測する。

        Args:
            state: 現在の状態推定 x（長さ n）。
            dt: 予測する時間幅 Δt [s]（正の値）。

        Returns:
            (予測状態 x⁻（長さ n）, 状態遷移のヤコビアン F（n×n）)。
        """

    @abstractmethod
    def process_noise(self, dt: float) -> FloatArray:
        """時間幅 Δt のプロセス雑音共分散 Q（n×n）を返す。

        Args:
            dt: 予測する時間幅 Δt [s]。

        Returns:
            プロセス雑音共分散 Q。
        """

    @abstractmethod
    def measure(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """状態から観測を予測する。

        Args:
            state: 予測状態 x⁻（長さ n）。

        Returns:
            (予測観測 h(x⁻)（長さ m）, 観測のヤコビアン H（m×n）)。
        """

    @abstractmethod
    def measurement_noise(self) -> FloatArray:
        """観測雑音共分散 R（m×m）を返す。適応 KF では推定の初期値になる。

        Returns:
            観測雑音共分散 R。
        """


class LinearSystem(StateSpaceSystem):
    """定数行列の線形状態空間モデル（LinearSystemParams）。

    F と Q は 1 サンプル分の時間に対する値なので、時刻は等間隔でなければならない。

    Attributes:
        params: モデルの条件。
    """

    def __init__(self, params: LinearSystemParams) -> None:
        """モデルの条件を受け取り、行列を NumPy 配列にしておく。

        Args:
            params: モデルの条件（形・性質はモデル側で検証済み）。
        """
        self.params = params
        self._transition = np.asarray(params.transition, dtype=np.float64)
        self._observation = np.asarray(params.observation, dtype=np.float64)
        self._process_noise = np.asarray(params.process_noise, dtype=np.float64)
        self._measurement_noise = np.asarray(params.measurement_noise, dtype=np.float64)

    @property
    def state_dim(self) -> int:
        """状態の次元 n。"""
        return int(self._transition.shape[0])

    @property
    def measurement_dim(self) -> int:
        """観測の次元 m。"""
        return int(self._observation.shape[0])

    @property
    def requires_uniform_sampling(self) -> bool:
        """定数行列は 1 サンプル分の時間を前提とするため、常に True。"""
        return True

    def predict(self, state: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
        """x⁻ = F x を計算する（Δt は使わない。等間隔の確認は KalmanFilter が行う）。

        Args:
            state: 現在の状態推定 x。
            dt: 予測する時間幅 Δt [s]（このモデルでは使わない）。

        Returns:
            (予測状態, F)。
        """
        return self._transition @ state, self._transition

    def process_noise(self, dt: float) -> FloatArray:
        """定数のプロセス雑音共分散 Q を返す。

        Args:
            dt: 予測する時間幅 Δt [s]（このモデルでは使わない）。

        Returns:
            Q のコピー。
        """
        return self._process_noise.copy()

    def measure(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """予測観測 H x⁻ を計算する。

        Args:
            state: 予測状態 x⁻。

        Returns:
            (予測観測, H)。
        """
        return self._observation @ state, self._observation

    def measurement_noise(self) -> FloatArray:
        """定数の観測雑音共分散 R を返す。

        Returns:
            R のコピー。
        """
        return self._measurement_noise.copy()


class KinematicSystem(StateSpaceSystem):
    """値とその時間微分を状態とする運動学モデル（KinematicSystemParams）。

    状態は [値, 1 階微分, ..., d 階微分]（n = d + 1）、観測は値のみ（H = [1, 0, ..., 0]）。
    F・Q を時間幅 Δt から毎回計算するため、不等間隔の時刻に対応する。

    Attributes:
        params: モデルの条件。
    """

    def __init__(self, params: KinematicSystemParams) -> None:
        """モデルの条件を受け取って初期化する。

        Args:
            params: モデルの条件。
        """
        self.params = params
        self._n_states = params.derivative_order + 1
        self._observation = np.zeros((1, self._n_states), dtype=np.float64)
        self._observation[0, 0] = 1.0

    @property
    def state_dim(self) -> int:
        """状態の次元 n（= 微分の最高階数 + 1）。"""
        return self._n_states

    @property
    def measurement_dim(self) -> int:
        """観測の次元（値のみなので 1）。"""
        return 1

    @property
    def requires_uniform_sampling(self) -> bool:
        """Δt ごとに F・Q を計算するため、等間隔は不要（False）。"""
        return False

    def predict(self, state: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
        """テイラー展開に基づく状態遷移 x⁻ = F(Δt) x を計算する。

        Args:
            state: 現在の状態推定 x。
            dt: 予測する時間幅 Δt [s]。

        Returns:
            (予測状態, F(Δt))。
        """
        transition = kinematic_transition_matrix(self._n_states, dt)
        return transition @ state, transition

    def process_noise(self, dt: float) -> FloatArray:
        """時間幅 Δt のプロセス雑音共分散 Q(Δt) を返す。

        Args:
            dt: 予測する時間幅 Δt [s]。

        Returns:
            Q(Δt)。
        """
        return kinematic_process_noise(self._n_states, self.params.process_noise_density, dt)

    def measure(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """予測観測（状態の先頭要素 = 値）を返す。

        Args:
            state: 予測状態 x⁻。

        Returns:
            (予測観測, H)。
        """
        return self._observation @ state, self._observation

    def measurement_noise(self) -> FloatArray:
        """観測雑音共分散 R = σ² を返す。

        Returns:
            1×1 の R。
        """
        return np.array([[self.params.measurement_noise_std**2]], dtype=np.float64)


class NonlinearSystem(StateSpaceSystem):
    """任意の非線形関数とそのヤコビアンで定義する状態空間モデル（EKF 用）。

    関数は Python で与える。返り値の形がモデルの次元と合わない場合は、
    推定を続けずに AnalysisInputError を送出する。

    Attributes:
        transition: 状態遷移関数 f(x, Δt)。
        transition_jacobian: ∂f/∂x(x, Δt)。
        observation: 観測関数 h(x)。
        observation_jacobian: ∂h/∂x(x)。
    """

    def __init__(
        self,
        transition: TransitionFunction,
        transition_jacobian: TransitionJacobian,
        observation: ObservationFunction,
        observation_jacobian: ObservationJacobian,
        process_noise: FloatArray,
        measurement_noise: FloatArray,
        *,
        requires_uniform_sampling: bool = False,
    ) -> None:
        """関数と雑音共分散を受け取り、共分散の形・性質を確認する。

        Args:
            transition: 状態遷移関数 f(x, Δt)。
            transition_jacobian: 状態遷移関数のヤコビアン。
            observation: 観測関数 h(x)。
            observation_jacobian: 観測関数のヤコビアン。
            process_noise: プロセス雑音共分散 Q（n×n、半正定値）。Δt によらず一定とする。
            measurement_noise: 観測雑音共分散 R（m×m、正定値）。
            requires_uniform_sampling: f・Q が 1 サンプル分の時間を前提とするなら True。

        Raises:
            ValueError: Q・R が正方でない、または対称・（半）正定値でない場合。
        """
        self.transition = transition
        self.transition_jacobian = transition_jacobian
        self.observation = observation
        self.observation_jacobian = observation_jacobian
        self._process_noise = np.asarray(process_noise, dtype=np.float64)
        self._measurement_noise = np.asarray(measurement_noise, dtype=np.float64)
        self._requires_uniform_sampling = requires_uniform_sampling
        for name, matrix in (("Q", self._process_noise), ("R", self._measurement_noise)):
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{name} は正方行列にしてください（実際の形: {matrix.shape}）")
        check_covariance("Q", self._process_noise.tolist(), strict=False)
        check_covariance("R", self._measurement_noise.tolist(), strict=True)

    @property
    def state_dim(self) -> int:
        """状態の次元 n（Q の大きさ）。"""
        return int(self._process_noise.shape[0])

    @property
    def measurement_dim(self) -> int:
        """観測の次元 m（R の大きさ）。"""
        return int(self._measurement_noise.shape[0])

    @property
    def requires_uniform_sampling(self) -> bool:
        """生成時に指定した、等間隔を前提とするかどうか。"""
        return self._requires_uniform_sampling

    def predict(self, state: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
        """f(x, Δt) と、x で評価したヤコビアン F を計算する。

        Args:
            state: 現在の状態推定 x。
            dt: 予測する時間幅 Δt [s]。

        Returns:
            (予測状態, F)。

        Raises:
            AnalysisInputError: 関数の返り値の形がモデルの次元と合わない場合。
        """
        n = self.state_dim
        predicted = _as_shape("f(x, dt)", self.transition(state, dt), (n,))
        jacobian = _as_shape("∂f/∂x", self.transition_jacobian(state, dt), (n, n))
        return predicted, jacobian

    def process_noise(self, dt: float) -> FloatArray:
        """一定のプロセス雑音共分散 Q を返す。

        Args:
            dt: 予測する時間幅 Δt [s]（このモデルでは使わない）。

        Returns:
            Q のコピー。
        """
        return self._process_noise.copy()

    def measure(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        """h(x⁻) と、x⁻ で評価したヤコビアン H を計算する。

        Args:
            state: 予測状態 x⁻。

        Returns:
            (予測観測, H)。

        Raises:
            AnalysisInputError: 関数の返り値の形がモデルの次元と合わない場合。
        """
        m, n = self.measurement_dim, self.state_dim
        predicted = _as_shape("h(x)", self.observation(state), (m,))
        jacobian = _as_shape("∂h/∂x", self.observation_jacobian(state), (m, n))
        return predicted, jacobian

    def measurement_noise(self) -> FloatArray:
        """観測雑音共分散 R を返す。

        Returns:
            R のコピー。
        """
        return self._measurement_noise.copy()


# ---------------------------------------------------------------------------
# 運動学モデルの行列
# ---------------------------------------------------------------------------


def kinematic_transition_matrix(n_states: int, dt: float) -> FloatArray:
    """運動学モデルの状態遷移行列 F(Δt) を返す。

    状態 [値, 1 階微分, ..., (n-1) 階微分] をテイラー展開で Δt 進める::

        F[i, j] = Δt^(j - i) / (j - i)!   （j >= i）、それ以外は 0

    例: n=2 なら [[1, Δt], [0, 1]]。

    Args:
        n_states: 状態の次元 n。
        dt: 時間幅 Δt [s]。

    Returns:
        n×n の状態遷移行列。
    """
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(i, n_states):
            transition[i, j] = dt ** (j - i) / math.factorial(j - i)
    return transition


def kinematic_process_noise(n_states: int, density: float, dt: float) -> FloatArray:
    """運動学モデルのプロセス雑音共分散 Q(Δt) を返す。

    最高階の微分が強度 q の連続時間白色雑音で駆動されるとして、Δt の間に蓄積する
    状態の誤差の共分散を厳密に離散化したもの::

        Q[i, j] = q Δt^(2n-1-i-j) / ((2n-1-i-j) (n-1-i)! (n-1-j)!)

    例: n=2（等速度モデル）なら q [[Δt³/3, Δt²/2], [Δt²/2, Δt]]。

    Args:
        n_states: 状態の次元 n。
        density: 白色雑音のパワースペクトル密度 q。
        dt: 時間幅 Δt [s]。

    Returns:
        n×n のプロセス雑音共分散。
    """
    noise = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            power = 2 * n_states - 1 - i - j
            denominator = (
                power * math.factorial(n_states - 1 - i) * math.factorial(n_states - 1 - j)
            )
            noise[i, j] = density * dt**power / denominator
    return noise


# ---------------------------------------------------------------------------
# カルマンフィルタ本体
# ---------------------------------------------------------------------------


class KalmanFilter:
    """線形 KF・EKF・適応 KF を共通の手順で実行するクラス。

    ``system`` が線形モデルなら通常のカルマンフィルタ、``NonlinearSystem`` なら EKF になる。
    ``adaptive`` を指定すると、観測雑音共分散 R を逐次推定する適応カルマンフィルタになる。

    使用例::

        system = KinematicSystem(KinematicSystemParams(process_noise_density=1.0,
                                                       measurement_noise_std=0.01))
        initial = KalmanInitialState(state=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 1.0]])
        result = KalmanFilter(system, initial).run(t, z)

    Attributes:
        system: 状態空間モデル。
        initial: 初期状態（先頭時刻の事前推定）。
        adaptive: 適応推定の条件。None なら R は一定。
        uniform_tolerance: 等間隔を前提とするモデルで、等間隔とみなす間隔のばらつきの許容値。
    """

    def __init__(
        self,
        system: StateSpaceSystem,
        initial: KalmanInitialState,
        adaptive: AdaptiveNoiseParams | None = None,
        uniform_tolerance: float = DEFAULT_UNIFORM_TOLERANCE,
    ) -> None:
        """モデル・初期状態・適応推定の条件を受け取り、次元の整合を確認する。

        Args:
            system: 状態空間モデル。
            initial: 初期状態（先頭時刻の事前推定）。
            adaptive: 適応推定の条件。None なら R は一定。
            uniform_tolerance: 等間隔とみなす間隔のばらつきの許容値。

        Raises:
            AnalysisInputError: 初期状態の次元がモデルの状態次元と合わない場合。
        """
        if len(initial.state) != system.state_dim:
            raise AnalysisInputError(
                f"初期状態の次元 {len(initial.state)} がモデルの状態次元 {system.state_dim} と"
                "一致しません"
            )
        self.system = system
        self.initial = initial
        self.adaptive = adaptive
        self.uniform_tolerance = uniform_tolerance

    def run(self, t: FloatArray, z: FloatArray) -> KalmanResult:
        """時系列の観測に対してフィルタを実行する。

        Args:
            t: 時刻 [s] の配列（狭義単調増加）。
            z: 観測の配列。観測が 1 次元なら (n,) または (n, 1)、m 次元なら (n, m)。
                NaN を含む行は欠測として観測更新をしない。無限大は不可。

        Returns:
            時刻ごとの推定結果。

        Raises:
            AnalysisInputError: 時刻・観測が不正な場合、等間隔が必要なモデルで不等間隔の場合、
                またはモデルの関数が不正な形の値を返した場合。
        """
        times = validate_time_axis(t, min_samples=1)
        measurements = self._validate_measurements(z, times.size)
        if self.system.requires_uniform_sampling and times.size >= 2:
            check_uniform_sampling(times, self.uniform_tolerance)

        n, nx, nz = times.size, self.system.state_dim, self.system.measurement_dim
        states = np.empty((n, nx), dtype=np.float64)
        covariances = np.empty((n, nx, nx), dtype=np.float64)
        innovations = np.full((n, nz), np.nan, dtype=np.float64)
        innovation_covariances = np.full((n, nz, nz), np.nan, dtype=np.float64)
        noise_history = np.empty((n, nz, nz), dtype=np.float64)
        updated = np.zeros(n, dtype=np.bool_)
        noise_floor_applied = np.zeros(n, dtype=np.bool_)

        identity = np.eye(nx, dtype=np.float64)
        state: FloatArray = np.asarray(self.initial.state, dtype=np.float64)
        covariance: FloatArray = np.asarray(self.initial.covariance, dtype=np.float64)
        noise = self.system.measurement_noise()
        estimator = None if self.adaptive is None else _AdaptiveNoiseEstimator(self.adaptive, noise)

        for k in range(n):
            # --- 予測（先頭時刻は初期状態をそのまま事前推定とする） ---
            if k > 0:
                dt = float(times[k] - times[k - 1])
                state, transition = self.system.predict(state, dt)
                covariance = transition @ covariance @ transition.T + self.system.process_noise(dt)

            # --- 観測更新（NaN を含む観測は欠測として飛ばす） ---
            observed = measurements[k]
            if not np.isnan(observed).any():
                predicted_measurement, observation = self.system.measure(state)
                innovation = observed - predicted_measurement
                projected = observation @ covariance @ observation.T  # H P⁻ Hᵀ
                if estimator is not None:
                    noise, noise_floor_applied[k] = estimator.estimate(innovation, projected)
                innovation_covariance = projected + noise
                # K = P⁻ Hᵀ S⁻¹ を、逆行列を作らずに S Kᵀ = H P⁻ を解いて求める（S, P⁻ は対称）。
                gain = np.linalg.solve(innovation_covariance, observation @ covariance).T
                state = state + gain @ innovation
                correction = identity - gain @ observation
                covariance = correction @ covariance @ correction.T + gain @ noise @ gain.T
                covariance = 0.5 * (covariance + covariance.T)  # 丸め誤差による非対称を除く
                innovations[k] = innovation
                innovation_covariances[k] = innovation_covariance
                updated[k] = True

            states[k] = state
            covariances[k] = covariance
            noise_history[k] = noise

        return KalmanResult(
            time_s=times.copy(),
            states=states,
            covariances=covariances,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            measurement_noise=noise_history,
            updated=updated,
            noise_floor_applied=noise_floor_applied,
        )

    def _validate_measurements(self, z: FloatArray, n_samples: int) -> FloatArray:
        """観測配列の形を (n, m) にそろえ、無限大を含まないことを確認する。

        Args:
            z: 観測の配列。
            n_samples: 時刻の数 n。

        Returns:
            (n, m) の float64 配列。

        Raises:
            AnalysisInputError: 形がモデルの観測次元と合わない、または無限大を含む場合。
        """
        measurements: FloatArray = np.asarray(z, dtype=np.float64)
        nz = self.system.measurement_dim
        if measurements.ndim == 1 and nz == 1:
            measurements = measurements.reshape(-1, 1)
        if measurements.shape != (n_samples, nz):
            raise AnalysisInputError(
                f"観測の形 {np.shape(z)} が (時刻数 {n_samples}, 観測次元 {nz}) と一致しません"
            )
        if np.isinf(measurements).any():
            raise AnalysisInputError("観測に無限大が含まれています")
        return measurements


# ---------------------------------------------------------------------------
# 適応推定・補助関数（モジュール内部用）
# ---------------------------------------------------------------------------


class _AdaptiveNoiseEstimator:
    """観測雑音共分散 R の逐次推定（KalmanFilter の内部で使う）。

    推定式（ν: イノベーション、H P⁻ Hᵀ: 予測状態の不確かさを観測空間へ写したもの）:
        - 窓付きイノベーション推定: C = (1/N) Σ_{直近 N 個} ν νᵀ,  R = C - H P⁻ Hᵀ
          （イノベーション共分散の理論値 S = H P⁻ Hᵀ + R を、標本共分散 C と一致させる考え方）
        - Sage-Husa 推定: d_k = (1 - b) / (1 - b^(k+1)),
          R_k = (1 - d_k) R_{k-1} + d_k (ν νᵀ - H P⁻ Hᵀ)
          （k は 0 始まりの更新回数。k=0 では d=1 となり、初期値は最初の推定で置き換わる）

    推定した R は対称化したうえで、固有値を下限 ``min_variance`` で抑えて正定値にする。
    """

    def __init__(self, params: AdaptiveNoiseParams, initial_noise: FloatArray) -> None:
        """推定の条件と R の初期値を受け取って初期化する。

        Args:
            params: 適応推定の条件。
            initial_noise: R の初期値（窓がたまるまではこれを使う）。
        """
        self.params = params
        self.current = initial_noise.copy()
        self._recent: deque[FloatArray] = deque()
        self._recent_sum = np.zeros_like(initial_noise)
        self._n_updates = 0

    def estimate(self, innovation: FloatArray, projected: FloatArray) -> tuple[FloatArray, bool]:
        """今回のイノベーションを取り込んで R を推定する。

        Args:
            innovation: イノベーション ν（長さ m）。
            projected: H P⁻ Hᵀ（m×m）。

        Returns:
            (今回の更新で使う R, 正定値化のために固有値を下限で抑えたか)。
        """
        outer = np.outer(innovation, innovation)
        if self.params.method is AdaptiveNoiseMethod.INNOVATION_WINDOW:
            # 直近 N 個の ν νᵀ の和を保持し、窓から外れたものを引く。
            self._recent.append(outer)
            self._recent_sum = self._recent_sum + outer
            if len(self._recent) > self.params.window:
                self._recent_sum = self._recent_sum - self._recent.popleft()
            if len(self._recent) < self.params.window:
                return self.current.copy(), False  # 窓がたまるまでは初期値を使う
            raw = self._recent_sum / self.params.window - projected
        else:
            b = self.params.forgetting_factor
            weight = (1.0 - b) / (1.0 - b ** (self._n_updates + 1))
            raw = (1.0 - weight) * self.current + weight * (outer - projected)
            self._n_updates += 1
        self.current, floored = _floor_eigenvalues(raw, self.params.min_variance)
        return self.current.copy(), floored


def _floor_eigenvalues(matrix: FloatArray, minimum: float) -> tuple[FloatArray, bool]:
    """対称化した行列の固有値を下限で抑え、正定値行列にする。

    Args:
        matrix: 推定した共分散行列（丸め誤差で非対称でもよい）。
        minimum: 固有値の下限（正の値）。

    Returns:
        (正定値化した行列, 下限で抑えた固有値があったか)。
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floored = bool((eigenvalues < minimum).any())
    if not floored:
        return symmetric, False
    clipped = np.maximum(eigenvalues, minimum)
    return (eigenvectors * clipped) @ eigenvectors.T, True


def _as_shape(name: str, value: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """ユーザー定義関数の返り値を float64 配列にし、期待する形であることを確認する。

    Args:
        name: エラーメッセージに含める関数名（例: "h(x)"）。
        value: 関数の返り値。
        shape: 期待する形。

    Returns:
        float64 の配列。

    Raises:
        AnalysisInputError: 形が異なる、または非有限値を含む場合。
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise AnalysisInputError(f"{name} の返り値の形が {array.shape} です（期待: {shape}）")
    if not np.isfinite(array).all():
        raise AnalysisInputError(f"{name} の返り値に NaN・無限大が含まれています")
    return array
