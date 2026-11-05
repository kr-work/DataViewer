"""取り込んだ合成データから、入出力関係の解析で既知の値が求まることを確かめる（SPEC A-09〜A-11）。

io_pair.csv の系は、むだ時間・時定数・遊びの幅が目録に記録されている。
high_rate.csv と low_rate.csv は時計が違うので、目録の時刻の原点から時刻オフセットを計算できる。
"""

import pytest

from experiment_workbench.analysis import run_analysis
from experiment_workbench.importer import DataImporter
from experiment_workbench.model import (
    AlignmentGrid,
    AlignmentParams,
    AnalysisKind,
    AnalysisResult,
    AnalysisSpec,
    Channel,
    ChannelData,
    ChannelImportSpec,
    CrossCorrelationParams,
    HysteresisParams,
    ImportRequest,
    Source,
    SourceFormat,
    StepResponseParams,
    TimeUnit,
)
from experiment_workbench.storage import ProjectStore
from tests.conftest import SyntheticDataset, high_rate_request, low_rate_request


def _import_io_pair(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> tuple[ChannelData, ChannelData]:
    """io_pair.csv を取り込み、入力と出力のチャンネルのデータを返す。

    Args:
        project: 取り込み先のプロジェクト。
        synthetic_dataset: 合成データ。

    Returns:
        (入力, 出力) のチャンネルのデータ。
    """
    data, _ = synthetic_dataset
    request = ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="time_s",
        time_unit=TimeUnit.SECOND,
        channels=[
            ChannelImportSpec(column="input", unit="N"),
            ChannelImportSpec(column="output", unit="V"),
        ],
        experiment_name="合成",
        session_name="s1",
        trial_name="trial_001",
        source_name="io_pair",
    )
    prepared = DataImporter(project).import_file(data / "trial_001/io_pair.csv", request)
    first, second = prepared.channels
    return (
        project.read_channel_data(prepared.source, first),
        project.read_channel_data(prepared.source, second),
    )


def _run(
    data: ChannelData,
    output: ChannelData,
    kind: AnalysisKind,
    params: object,
    alignment: AlignmentParams,
) -> dict[str, float | None]:
    """2 チャンネルの解析を実行し、指標を識別名 → 値 の辞書で返す。

    Args:
        data: 入力チャンネルのデータ。
        output: 出力チャンネルのデータ。
        kind: 解析の種類。
        params: 種類に対応する条件。
        alignment: 対応付けの条件。

    Returns:
        指標の辞書。
    """
    spec = AnalysisSpec.model_validate(
        {
            "name": "応答",
            "input_channel_id": data.channel.id,
            "output_channel_id": output.channel.id,
            "alignment": alignment,
            "kind": kind,
            kind.value: params,
        }
    )
    result: AnalysisResult = run_analysis(spec, data, output_data=output)
    return {metric.key: metric.value for metric in result.metrics}


def test_step_response_recovers_known_delay_and_time_constant(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """最初のステップの前後を範囲に選ぶと、系のむだ時間・時定数・ゲインが求まる。"""
    _, manifest = synthetic_dataset
    trial = manifest.trials[0]
    system = trial.io_system
    data, output = _import_io_pair(project, synthetic_dataset)
    step_time = trial.io_pair.clock_offset_s + system.step_times_s[0]
    interval = 1.0 / trial.io_pair.nominal_rate_hz
    values = _run(
        data,
        output,
        AnalysisKind.STEP_RESPONSE,
        StepResponseParams(),
        AlignmentParams(start_s=step_time - 0.1, end_s=step_time + 0.2),
    )
    assert values["time_constant_s"] == pytest.approx(system.time_constant_s, rel=0.15)
    assert values["dead_time_s"] == pytest.approx(system.delay_s, abs=4 * interval)
    assert values["gain"] == pytest.approx(system.gain, rel=0.03)


def test_hysteresis_recovers_known_branch_gap(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """三角波の区間を範囲に選ぶと、遊びと遅れから決まる分岐の差と、系の感度が求まる。

    定常なランプでの分岐の差は K × (h + 2 r (L + τ + Δt/2))。h は遊びの幅、r は入力の速さ、
    L はむだ時間、τ は時定数で、Δt/2 は一次遅れの離散化による半サンプルの遅れ。
    """
    _, manifest = synthetic_dataset
    trial = manifest.trials[0]
    system = trial.io_system
    data, output = _import_io_pair(project, synthetic_dataset)
    start = trial.io_pair.clock_offset_s + system.triangle_start_s
    values = _run(
        data,
        output,
        AnalysisKind.HYSTERESIS,
        HysteresisParams(n_bins=30, smoothing_cutoff_hz=20.0),
        AlignmentParams(start_s=start),
    )
    interval = 1.0 / trial.io_pair.nominal_rate_hz
    rate = 2 * (system.triangle_high - system.triangle_low) / system.triangle_period_s
    lag = system.delay_s + system.time_constant_s + interval / 2
    expected = system.gain * (system.play_width + 2 * rate * lag)
    assert values["max_hysteresis"] == pytest.approx(expected, rel=0.1)
    assert values["slope"] == pytest.approx(system.gain, rel=0.1)


def test_cross_correlation_between_clocks_with_declared_offset(
    project: ProjectStore, synthetic_dataset: SyntheticDataset
) -> None:
    """時計が違う 2 つのファイルも、時刻オフセットを明示すると対応付けて遅れを測れる。

    真の速度は同じ運動を 2 つのファイルで記録したものなので、正しいオフセットなら遅れは 0、
    オフセットを 20 ms ずらすと、その分だけ遅れが負に現れる。
    """
    data_dir, manifest = synthetic_dataset
    high, low = manifest.trials[0].sources
    importer = DataImporter(project)
    high_request = high_rate_request()
    low_request = low_rate_request().model_copy(
        update={"channels": [ChannelImportSpec(column="true_velocity", unit="m/s")]}
    )
    importer.import_file(data_dir / "trial_001/high_rate.csv", high_request)
    importer.import_file(data_dir / "trial_001/low_rate.csv", low_request)
    sources = {source.name: source for source in project.database.list_entities(Source)}
    channels = {
        name: {
            channel.name: channel
            for channel in project.database.list_entities(Channel, source_id=source.id)
        }
        for name, source in sources.items()
    }
    data = project.read_channel_data(sources["high_rate"], channels["high_rate"]["true_velocity"])
    output = project.read_channel_data(sources["low_rate"], channels["low_rate"]["true_velocity"])
    offset = high.clock_offset_s - low.clock_offset_s
    params = CrossCorrelationParams(min_lag_s=-0.2, max_lag_s=0.2)
    for shift, expected in ((0.0, 0.0), (-0.02, -0.02)):
        values = _run(
            data,
            output,
            AnalysisKind.CROSS_CORRELATION,
            params,
            AlignmentParams(
                output_time_offset_s=offset + shift,
                grid=AlignmentGrid.UNIFORM,
                uniform_rate_hz=200.0,
            ),
        )
        assert values["delay_s"] == pytest.approx(expected, abs=0.002)
        assert values["peak_coefficient"] == pytest.approx(1.0, abs=0.01)
