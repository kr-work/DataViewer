"""解析コア（SPEC §7 A-01〜A-11）: フィルタ・微分・積分・カルマンフィルタ系、入出力関係の解析。

GUI に依存せず、時刻配列と値の配列を受け取って新しい配列（または推定結果）を返す。
処理の条件は ``experiment_workbench.model`` の pydantic モデルで指定する。
保存できる 1 段の解析（AnalysisSpec）は ``run_analysis()`` で実行する。

利用側は ``from experiment_workbench.analysis import ButterworthFilter`` のように、
このパッケージから公開 API を import する。このパッケージは Qt に依存しない。
"""

from experiment_workbench.analysis.alignment import ChannelAligner
from experiment_workbench.analysis.input_output import (
    CrossCorrelationAnalyzer,
    HysteresisAnalyzer,
    StepResponseAnalyzer,
)
from experiment_workbench.analysis.kalman import (
    KalmanFilter,
    KinematicSystem,
    LinearSystem,
    NonlinearSystem,
    PolynomialObservationSystem,
    ProgressCallback,
    StateSpaceSystem,
    kinematic_process_noise,
    kinematic_transition_matrix,
    system_from_settings,
)
from experiment_workbench.analysis.runner import (
    analysis_fingerprint,
    default_initial_state,
    default_kalman_settings,
    default_omega,
    estimate_noise_std,
    output_infos,
    output_series_id,
    overlays_input,
    result_from_frame,
    result_to_frame,
    round_significant,
    run_analysis,
    series_from_channel,
)
from experiment_workbench.analysis.signal_processing import (
    DEFAULT_UNIFORM_TOLERANCE,
    AnalysisCancelledError,
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
    "AnalysisCancelledError",
    "AnalysisInputError",
    "ButterworthFilter",
    "ChannelAligner",
    "CrossCorrelationAnalyzer",
    "Differentiator",
    "HysteresisAnalyzer",
    "Integrator",
    "KalmanFilter",
    "KinematicSystem",
    "LinearSystem",
    "MovingAverageFilter",
    "NonlinearSystem",
    "PolynomialObservationSystem",
    "ProgressCallback",
    "SavitzkyGolayFilter",
    "StateSpaceSystem",
    "StepResponseAnalyzer",
    "analysis_fingerprint",
    "check_uniform_sampling",
    "default_initial_state",
    "default_kalman_settings",
    "default_omega",
    "estimate_noise_std",
    "estimate_sampling_rate",
    "kinematic_process_noise",
    "kinematic_transition_matrix",
    "output_infos",
    "output_series_id",
    "overlays_input",
    "result_from_frame",
    "result_to_frame",
    "round_significant",
    "run_analysis",
    "series_from_channel",
    "system_from_settings",
    "validate_series",
    "validate_time_axis",
]
