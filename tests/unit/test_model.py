"""model.py（pydantic ドメインモデル）のテスト。"""

import math
from uuid import uuid4

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from experiment_workbench.model import (
    AdaptiveNoiseMethod,
    AdaptiveNoiseParams,
    ButterworthKind,
    ButterworthParams,
    DerivativeMethod,
    DerivativeParams,
    Event,
    Experiment,
    IntegralParams,
    KalmanInitialState,
    KalmanResult,
    KinematicSystemParams,
    LinearSystemParams,
    MovingAverageParams,
    Project,
    SavitzkyGolayParams,
    Source,
    SyntheticConfig,
    SyntheticMotion,
    TimeAxisKind,
    TimeBase,
    Trial,
)
from tests.conftest import Hierarchy


def test_entities_get_unique_ids_and_aware_timestamps() -> None:
    """ID は自動で一意に採番され、作成日時はタイムゾーン付きになる。"""
    first, second = Experiment(name="a"), Experiment(name="b")
    assert first.id != second.id
    assert first.created_at.tzinfo is not None


def test_unknown_fields_are_rejected() -> None:
    """未知のフィールドは黙って捨てずにエラーにする。"""
    with pytest.raises(ValidationError, match="extra"):
        Experiment.model_validate({"name": "a", "unknown_field": 1})


def test_assignment_is_validated() -> None:
    """生成後の代入も検証される。"""
    experiment = Experiment(name="a")
    with pytest.raises(ValidationError):
        experiment.name = "   "


def test_trial_rejects_duplicate_tags() -> None:
    """Trial のタグの重複はエラーになる。"""
    with pytest.raises(ValidationError, match="重複"):
        Trial(session_id=uuid4(), name="t", tags=["success", "success"])


def test_project_records_format_and_app_version() -> None:
    """Project には保存形式バージョンとアプリのバージョンが記録される。"""
    project = Project(name="p")
    assert project.format_version >= 1
    assert project.app_version


@pytest.mark.parametrize(
    "kind",
    [TimeAxisKind.SAMPLE_INDEX, TimeAxisKind.RELATIVE_SECONDS, TimeAxisKind.ABSOLUTE_EPOCH_NS],
)
def test_time_base_accepts_raw_time_axis_kinds(kind: TimeAxisKind) -> None:
    """生データの時間軸として使える 3 種類は受け付ける。"""
    assert TimeBase(kind=kind).kind is kind


def test_time_base_rejects_event_relative_axis() -> None:
    """イベント相対時刻は整列後の軸なので、生データの時間軸の種類には含めない。"""
    with pytest.raises(ValidationError):
        TimeBase.model_validate({"kind": "event_relative_seconds"})


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_names_must_not_be_blank(name: str) -> None:
    """名前に空文字・空白のみは指定できない。"""
    with pytest.raises(ValidationError):
        Experiment(name=name)


def test_names_keep_surrounding_spaces() -> None:
    """名前の前後の空白は勝手に除去しない。"""
    assert Experiment(name=" 実験 ").name == " 実験 "


def test_trial_rejects_blank_tags() -> None:
    """空白のみのタグはエラーになる。"""
    with pytest.raises(ValidationError, match="空"):
        Trial(session_id=uuid4(), name="t", tags=["ok", " "])


def test_source_without_time_column_must_use_sample_index(hierarchy: Hierarchy) -> None:
    """時刻列がないのに相対秒を名乗る Source はエラーになる。"""
    source = hierarchy[3]
    data = source.model_dump() | {"time_column": None}
    with pytest.raises(ValidationError, match="sample_index"):
        Source.model_validate(data)
    data["time_base"] = TimeBase(kind=TimeAxisKind.SAMPLE_INDEX).model_dump()
    assert Source.model_validate(data).time_column is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_sha256", "A" * 64),  # 大文字は不可
        ("original_sha256", "a" * 63),  # 長さ不足
        ("original_relpath", "../outside.csv"),
        ("dataset_relpath", "C:/data.parquet"),
        ("original_size_bytes", -1),
    ],
)
def test_source_rejects_invalid_file_metadata(
    hierarchy: Hierarchy, field: str, value: object
) -> None:
    """ハッシュ・相対パス・サイズの不正値は拒否される。"""
    data = hierarchy[3].model_dump() | {field: value}
    with pytest.raises(ValidationError):
        Source.model_validate(data)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_event_time_must_be_finite(hierarchy: Hierarchy, value: float) -> None:
    """イベント時刻に NaN・無限大は指定できない。"""
    data = hierarchy[5].model_dump() | {"time_s": value}
    with pytest.raises(ValidationError):
        Event.model_validate(data)


def test_all_hierarchy_models_round_trip_through_json(hierarchy: Hierarchy) -> None:
    """全モデルが JSON を経由しても同じ値に戻る（保存形式の可逆性）。"""
    entity: BaseModel
    for entity in (*hierarchy, Project(name="p")):
        restored = type(entity).model_validate_json(entity.model_dump_json())
        assert restored == entity


def test_synthetic_config_limits_low_rate_to_millisecond_resolution() -> None:
    """ミリ秒整数の時刻列で重複が出ないよう、低レート側は 1000 Hz を超えられない。"""
    with pytest.raises(ValidationError):
        SyntheticConfig(low_rate_hz=1001.0)


def test_synthetic_config_requires_minimum_duration() -> None:
    """欠損区間などを配置するため、試技長は 1 秒以上が必要。"""
    with pytest.raises(ValidationError):
        SyntheticConfig(duration_s=0.5)


def test_synthetic_motion_requires_at_least_one_component() -> None:
    """真の運動には少なくとも 1 つの正弦波成分が必要。"""
    with pytest.raises(ValidationError):
        SyntheticMotion(position_components=[])


# --- 解析コアのパラメータ ---------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"kind": ButterworthKind.LOWPASS, "cutoff_hz": (1.0, 2.0)},  # LPF に 2 つ
        {"kind": ButterworthKind.BANDPASS, "cutoff_hz": 5.0},  # BPF に 1 つ
        {"kind": ButterworthKind.BANDPASS, "cutoff_hz": (20.0, 10.0)},  # 下限 >= 上限
        {"kind": ButterworthKind.LOWPASS, "cutoff_hz": 10.0, "order": 0},
        {"kind": ButterworthKind.LOWPASS, "cutoff_hz": 10.0, "order": 11},
        {"kind": ButterworthKind.LOWPASS, "cutoff_hz": -1.0},
    ],
)
def test_butterworth_params_reject_inconsistent_settings(params: dict[str, object]) -> None:
    """カットオフ周波数の個数・大小、次数の範囲が不正な条件は拒否される。"""
    with pytest.raises(ValidationError):
        ButterworthParams.model_validate(params)


def test_centered_moving_average_requires_odd_window() -> None:
    """中心移動平均の窓長は奇数に限る。後方移動平均は偶数でもよい。"""
    with pytest.raises(ValidationError, match="奇数"):
        MovingAverageParams(window_samples=4)
    assert MovingAverageParams(window_samples=4, centered=False).window_samples == 4


@pytest.mark.parametrize(("window", "polyorder"), [(10, 2), (5, 5)])
def test_savitzky_golay_params_reject_invalid_window(window: int, polyorder: int) -> None:
    """窓長が偶数、または多項式の次数が窓長以上の条件は拒否される。"""
    with pytest.raises(ValidationError):
        SavitzkyGolayParams(window_samples=window, polyorder=polyorder)


@pytest.mark.parametrize(
    "params",
    [
        {"method": DerivativeMethod.SAVITZKY_GOLAY},  # 条件がない
        {"savitzky_golay": {"window_samples": 5, "polyorder": 2}},  # 中心差分なのに条件がある
        {
            "method": DerivativeMethod.SAVITZKY_GOLAY,
            "order": 2,
            "savitzky_golay": {"window_samples": 5, "polyorder": 1},  # 次数 < 階数
        },
        {"order": 3},
        {"edge_order": 0},
    ],
)
def test_derivative_params_reject_inconsistent_settings(params: dict[str, object]) -> None:
    """微分の方法と条件が整合しない、または階数・精度が範囲外の条件は拒否される。"""
    with pytest.raises(ValidationError):
        DerivativeParams.model_validate(params)


@pytest.mark.parametrize(
    "params",
    [
        {"observation": [[1.0, 0.0, 0.0]]},  # H の列数が状態次元と合わない
        {"process_noise": [[1.0, 2.0], [0.0, 1.0]]},  # Q が非対称
        {"process_noise": [[-1.0, 0.0], [0.0, 1.0]]},  # Q が半正定値でない
        {"measurement_noise": [[0.0]]},  # R が正定値でない
        {"transition": [[1.0, 0.1], [0.0]]},  # F の行の長さが不揃い
    ],
)
def test_linear_system_params_reject_inconsistent_matrices(params: dict[str, object]) -> None:
    """行列の形や共分散の性質が整合しない線形モデルは拒否される。"""
    base: dict[str, object] = {
        "transition": [[1.0, 0.1], [0.0, 1.0]],
        "observation": [[1.0, 0.0]],
        "process_noise": [[0.0, 0.0], [0.0, 0.01]],
        "measurement_noise": [[0.04]],
    }
    with pytest.raises(ValidationError):
        LinearSystemParams.model_validate(base | params)


@pytest.mark.parametrize(
    ("state", "covariance"),
    [([], []), ([0.0, 0.0], [[1.0]]), ([0.0], [[0.0]])],
)
def test_kalman_initial_state_rejects_invalid_shapes(
    state: list[float], covariance: list[list[float]]
) -> None:
    """空の状態、形の合わない共分散、正定値でない共分散は拒否される。"""
    with pytest.raises(ValidationError):
        KalmanInitialState(state=state, covariance=covariance)


@pytest.mark.parametrize(
    "params",
    [{"forgetting_factor": 1.0}, {"forgetting_factor": 0.0}, {"window": 1}, {"min_variance": 0.0}],
)
def test_adaptive_noise_params_reject_out_of_range(params: dict[str, object]) -> None:
    """忘却係数・窓長・固有値の下限が範囲外の条件は拒否される。"""
    with pytest.raises(ValidationError):
        AdaptiveNoiseParams.model_validate({"method": AdaptiveNoiseMethod.SAGE_HUSA} | params)


def test_analysis_params_round_trip_through_json() -> None:
    """解析条件のモデルは JSON を経由しても同じ値に戻る（処理条件の保存・再現のため）。"""
    params: list[BaseModel] = [
        ButterworthParams(kind=ButterworthKind.BANDPASS, cutoff_hz=(1.0, 10.0), padtype=None),
        MovingAverageParams(window_samples=9),
        DerivativeParams(
            method=DerivativeMethod.SAVITZKY_GOLAY,
            savitzky_golay=SavitzkyGolayParams(window_samples=11, polyorder=3),
        ),
        IntegralParams(initial_value=1.5),
        LinearSystemParams(
            transition=[[1.0]],
            observation=[[1.0]],
            process_noise=[[0.0]],
            measurement_noise=[[1.0]],
        ),
        KinematicSystemParams(
            derivative_order=2, process_noise_density=3.0, measurement_noise_std=0.1
        ),
        KalmanInitialState(state=[0.0, 1.0], covariance=[[1.0, 0.0], [0.0, 2.0]]),
        AdaptiveNoiseParams(method=AdaptiveNoiseMethod.INNOVATION_WINDOW, window=20),
    ]
    for item in params:
        assert type(item).model_validate_json(item.model_dump_json()) == item


def _kalman_result_arrays(
    n: int = 4,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """形の整合した KalmanResult 用の配列一式を作る（状態 2 次元・観測 1 次元）。

    Args:
        n: 時刻の数。

    Returns:
        フィールド名 → 配列 の辞書。
    """
    return {
        "time_s": np.arange(n, dtype=np.float64),
        "states": np.zeros((n, 2)),
        "covariances": np.zeros((n, 2, 2)),
        "innovations": np.zeros((n, 1)),
        "innovation_covariances": np.zeros((n, 1, 1)),
        "measurement_noise": np.zeros((n, 1, 1)),
        "updated": np.ones(n, dtype=np.bool_),
        "noise_floor_applied": np.zeros(n, dtype=np.bool_),
    }


def test_kalman_result_accepts_consistent_arrays() -> None:
    """配列の dtype と形が整合していれば、結果モデルを作れる。"""
    assert KalmanResult.model_validate(_kalman_result_arrays()).states.shape == (4, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("states", np.zeros((3, 2))),  # 時刻数と合わない
        ("covariances", np.zeros((4, 2, 3))),  # 状態次元と合わない
        ("states", np.zeros((4, 2), dtype=np.float32)),  # 精度が落ちている
        ("updated", np.ones(4)),  # bool でない
    ],
)
def test_kalman_result_rejects_inconsistent_arrays(
    field: str, value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
) -> None:
    """配列の dtype や形が整合しなければ拒否される（変換して受け入れない）。"""
    arrays = _kalman_result_arrays() | {field: value}
    with pytest.raises(ValidationError):
        KalmanResult.model_validate(arrays)
