"""取り込み → 保存 → 再オープン → 解析の結合テスト。

Phase 1 の解析コアと、Phase 2 の取り込みがつながることを確かめる。
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from experiment_workbench.analysis import ButterworthFilter, KalmanFilter, KinematicSystem
from experiment_workbench.importer import DataImporter
from experiment_workbench.model import (
    ButterworthKind,
    ButterworthParams,
    Channel,
    KalmanInitialState,
    KinematicSystemParams,
    Source,
)
from experiment_workbench.storage import ProjectStore
from tests.conftest import SyntheticDataset, high_rate_request, low_rate_request


def _channel(store: ProjectStore, source_name: str, channel_name: str) -> tuple[Source, Channel]:
    """名前で Source とチャンネルを探す。

    Args:
        store: プロジェクト。
        source_name: Source 名。
        channel_name: チャンネル名。

    Returns:
        (Source, チャンネル)。
    """
    source = next(s for s in store.database.list_entities(Source) if s.name == source_name)
    channel = next(
        c
        for c in store.database.list_entities(Channel, source_id=source.id)
        if c.name == channel_name
    )
    return source, channel


def test_imported_data_survives_reopen_and_feeds_analysis(
    tmp_path: Path, synthetic_dataset: SyntheticDataset
) -> None:
    """取り込んだデータは閉じて開き直しても元ファイルと同じ値で読め、解析コアにそのまま渡せる。"""
    data, manifest = synthetic_dataset
    high_path = data / "trial_001/high_rate.csv"
    low_path = data / "trial_001/low_rate.csv"
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (high_path, low_path)}
    root = tmp_path / "project"
    with ProjectStore.create(root, name="結合テスト") as store:
        importer = DataImporter(store)
        importer.import_file(high_path, high_rate_request())
        importer.import_file(low_path, low_rate_request())

    with ProjectStore.open(root) as store:
        accel_source, accel_channel = _channel(store, "high_rate", "accel")
        accel = store.read_channel_data(accel_source, accel_channel)
        position_source, position_channel = _channel(store, "low_rate", "position")
        position = store.read_channel_data(position_source, position_channel)

    # 元ファイルは変わっておらず、読み戻した値は元ファイルと一致する。
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes} == hashes
    original = pl.read_csv(high_path)
    np.testing.assert_array_equal(accel.values, original["accel"].to_numpy())

    # 解析コア: 加速度の高周波振動をローパスフィルタで除く（真値 + バイアスに近づく）。
    high_record = manifest.trials[0].sources[0]
    lowpass = ButterworthFilter(ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=5.0))
    filtered = lowpass.apply(accel.time_s, accel.values)
    target = original["true_accel"].to_numpy() + high_record.noise_parameters["bias"]
    edge = 500
    assert np.sqrt(np.mean((filtered[edge:-edge] - target[edge:-edge]) ** 2)) < 0.05

    # 解析コア: 位置の観測からカルマンフィルタで推定する（観測ノイズより誤差が小さくなる）。
    low_record = manifest.trials[0].sources[1]
    noise_std = low_record.noise_parameters["noise_std"]
    system = KinematicSystem(
        KinematicSystemParams(process_noise_density=10.0, measurement_noise_std=noise_std)
    )
    initial = KalmanInitialState(
        state=[float(position.values[0]), 0.0], covariance=[[noise_std**2, 0.0], [0.0, 10.0]]
    )
    result = KalmanFilter(system, initial).run(position.time_s, position.values)
    truth = pl.read_csv(low_path)["true_position"].to_numpy()
    assert np.sqrt(np.mean((result.states[50:, 0] - truth[50:]) ** 2)) < noise_std
