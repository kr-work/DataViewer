"""解析コア（SPEC §7 A-01〜A-08）: フィルタ・微分・積分・カルマンフィルタ系。

GUI に依存せず、時刻配列と値の配列を受け取って新しい配列（または推定結果）を返す。
処理の条件は ``experiment_workbench.model`` の pydantic モデルで指定する。

利用側は ``from experiment_workbench.analysis import ButterworthFilter`` のように、
このパッケージから公開 API を import する。このパッケージは Qt に依存しない。
"""

from experiment_workbench.analysis.kalman import (
    KalmanFilter,
    KinematicSystem,
    LinearSystem,
    NonlinearSystem,
    StateSpaceSystem,
    kinematic_process_noise,
    kinematic_transition_matrix,
)
from experiment_workbench.analysis.signal_processing import (
    DEFAULT_UNIFORM_TOLERANCE,
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

__all__ = [
    "DEFAULT_UNIFORM_TOLERANCE",
    "AnalysisInputError",
    "ButterworthFilter",
    "Differentiator",
    "Integrator",
    "KalmanFilter",
    "KinematicSystem",
    "LinearSystem",
    "MovingAverageFilter",
    "NonlinearSystem",
    "SavitzkyGolayFilter",
    "StateSpaceSystem",
    "check_uniform_sampling",
    "estimate_sampling_rate",
    "kinematic_process_noise",
    "kinematic_transition_matrix",
    "validate_series",
    "validate_time_axis",
]
