"""テスト共通のフィクスチャ。"""

import pytest

from experiment_workbench.model import (
    Channel,
    Event,
    Experiment,
    SampleRate,
    Session,
    Source,
    SourceFormat,
    TimeAxisKind,
    TimeBase,
    Trial,
)

Hierarchy = tuple[Experiment, Session, Trial, Source, Channel, Event]
"""親から子の順に並べた、1 系列分のエンティティ（DB に insert_many でそのまま渡せる）。"""


@pytest.fixture
def hierarchy() -> Hierarchy:
    """Experiment → Session → Trial → Source → Channel / Event の 1 系列を作る。

    Returns:
        親から子の順に並べたエンティティのタプル。
    """
    experiment = Experiment(name="実験A", description="テスト用")
    session = Session(experiment_id=experiment.id, name="session_01")
    trial = Trial(session_id=session.id, name="trial_001", tags=["success"])
    source = Source(
        trial_id=trial.id,
        name="high_rate",
        format=SourceFormat.CSV,
        original_filename="high_rate.csv",
        original_relpath="originals/trial_001/high_rate.csv",
        original_sha256="a" * 64,
        original_size_bytes=1234,
        dataset_relpath="datasets/trial_001/high_rate.parquet",
        time_column="time_s",
        time_base=TimeBase(
            kind=TimeAxisKind.RELATIVE_SECONDS, original_unit="s", origin="装置の電源投入"
        ),
    )
    channel = Channel(
        source_id=source.id,
        name="accel",
        original_name="accel",
        unit="m/s^2",
        original_unit="m/s^2",
        sample_rate=SampleRate(nominal_hz=1000.0, estimated_hz=999.98),
    )
    event = Event(
        trial_id=trial.id,
        source_id=source.id,
        label="start",
        time_s=12.5,
        detection_params={"threshold": 0.5, "note": "手動確認済み"},
    )
    return experiment, session, trial, source, channel, event
