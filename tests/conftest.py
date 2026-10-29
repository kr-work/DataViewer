"""テスト共通のフィクスチャ。"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from experiment_workbench.model import (
    Channel,
    ChannelImportSpec,
    Event,
    Experiment,
    ImportRequest,
    SampleRate,
    Session,
    Source,
    SourceFormat,
    SyntheticConfig,
    SyntheticManifest,
    TimeAxisKind,
    TimeBase,
    TimeUnit,
    Trial,
)
from experiment_workbench.storage import ProjectStore
from experiment_workbench.synthetic import SyntheticDataGenerator

# GUI のテスト（pytest-qt）で画面を出さない。QApplication が作られる前に設定する必要がある。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

Hierarchy = tuple[Experiment, Session, Trial, Source, Channel, Event]
"""親から子の順に並べた、1 系列分のエンティティ（DB に insert_many でそのまま渡せる）。"""

SyntheticDataset = tuple[Path, SyntheticManifest]
"""(合成データの出力ディレクトリ, 目録) の組。"""


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
            kind=TimeAxisKind.RELATIVE_SECONDS,
            original_unit=TimeUnit.SECOND,
            origin="装置の電源投入",
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


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """取り込み・表示のテストで共有する小さな合成データ（3 秒 × 3 試技）を 1 回だけ生成する。

    テストからは読み取りだけを行い、変更しないこと（元ファイルを変えない確認にも使うため）。

    Args:
        tmp_path_factory: pytest の一時ディレクトリ生成器。

    Returns:
        (出力ディレクトリ, 目録) の組。
    """
    output = tmp_path_factory.mktemp("shared_synthetic") / "data"
    manifest = SyntheticDataGenerator(SyntheticConfig(duration_s=3.0, n_trials=3)).generate(output)
    return output, manifest


@pytest.fixture
def project(tmp_path: Path) -> Iterator[ProjectStore]:
    """空のプロジェクトを作って開き、テストの後に閉じる。

    Args:
        tmp_path: pytest の一時ディレクトリ。

    Yields:
        開いたプロジェクト。
    """
    store = ProjectStore.create(tmp_path / "project", name="テスト")
    yield store
    store.close()


def high_rate_request(
    trial_name: str = "trial_001", *, shares_trial_clock: bool = False
) -> ImportRequest:
    """合成データの high_rate.csv（時刻は秒）を取り込む指定を作る。

    Args:
        trial_name: 登録先の試技名。
        shares_trial_clock: 同期済みとして取り込むなら True。

    Returns:
        取り込みの指定。
    """
    return ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="time_s",
        time_unit=TimeUnit.SECOND,
        channels=[
            ChannelImportSpec(column="accel", unit="m/s^2"),
            ChannelImportSpec(column="true_velocity", unit="m/s"),
        ],
        experiment_name="合成",
        session_name="s1",
        trial_name=trial_name,
        source_name="high_rate",
        shares_trial_clock=shares_trial_clock,
    )


def low_rate_request(
    trial_name: str = "trial_001", *, shares_trial_clock: bool = False
) -> ImportRequest:
    """合成データの low_rate.csv（時刻はミリ秒）を取り込む指定を作る。

    Args:
        trial_name: 登録先の試技名。
        shares_trial_clock: 同期済みとして取り込むなら True。

    Returns:
        取り込みの指定。
    """
    return ImportRequest(
        file_format=SourceFormat.CSV,
        time_column="time_ms",
        time_unit=TimeUnit.MILLISECOND,
        channels=[ChannelImportSpec(column="position", unit="m")],
        experiment_name="合成",
        session_name="s1",
        trial_name=trial_name,
        source_name="low_rate",
        shares_trial_clock=shares_trial_clock,
    )
