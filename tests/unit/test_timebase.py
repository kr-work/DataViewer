"""timebase.py（時刻の単位換算と、時刻を比較できるかの判定）のテスト。"""

from uuid import UUID, uuid4

import numpy as np
import pytest

from experiment_workbench.model import Source, SourceFormat, TimeAxisKind, TimeBase, TimeUnit
from experiment_workbench.timebase import are_time_comparable, time_domain_key, to_seconds


def _source(trial_id: UUID, *, shares_trial_clock: bool) -> Source:
    """時刻の判定に必要な項目だけを持つ Source を作る。

    Args:
        trial_id: 所属する試技の ID。
        shares_trial_clock: 同期済みなら True。

    Returns:
        Source。
    """
    return Source(
        trial_id=trial_id,
        name="source",
        format=SourceFormat.CSV,
        original_filename="data.csv",
        original_relpath="originals/x/data.csv",
        original_sha256="0" * 64,
        original_size_bytes=1,
        dataset_relpath="datasets/x.parquet",
        time_column="t",
        time_base=TimeBase(
            kind=TimeAxisKind.RELATIVE_SECONDS,
            original_unit=TimeUnit.SECOND,
            shares_trial_clock=shares_trial_clock,
        ),
    )


@pytest.mark.parametrize(
    ("unit", "factor"),
    [
        (TimeUnit.SECOND, 1.0),
        (TimeUnit.MILLISECOND, 1e-3),
        (TimeUnit.MICROSECOND, 1e-6),
        (TimeUnit.NANOSECOND, 1e-9),
    ],
)
def test_to_seconds_converts_each_unit(unit: TimeUnit, factor: float) -> None:
    """各単位の時刻が秒に換算され、入力は変更されない。"""
    raw = np.array([0.0, 10.0, 2500.0])
    original = raw.copy()
    np.testing.assert_allclose(to_seconds(raw, unit), original * factor)
    np.testing.assert_array_equal(raw, original)


def test_source_is_comparable_with_itself() -> None:
    """同じ Source の列どうしは、同期済みでなくても比較できる。"""
    source = _source(uuid4(), shares_trial_clock=False)
    assert are_time_comparable(source, source)


def test_synchronized_sources_in_same_trial_are_comparable() -> None:
    """同じ試技で、どちらも同期済みと明示した Source どうしは比較できる。"""
    trial_id = uuid4()
    first = _source(trial_id, shares_trial_clock=True)
    second = _source(trial_id, shares_trial_clock=True)
    assert are_time_comparable(first, second)
    assert time_domain_key(first) == f"trial:{trial_id}"


@pytest.mark.parametrize(
    ("first_shares", "second_shares"), [(False, False), (True, False), (False, True)]
)
def test_unsynchronized_sources_are_not_comparable(
    *, first_shares: bool, second_shares: bool
) -> None:
    """どちらかが同期済みでなければ、同じ試技でも比較しない（勝手に揃えない）。"""
    trial_id = uuid4()
    first = _source(trial_id, shares_trial_clock=first_shares)
    second = _source(trial_id, shares_trial_clock=second_shares)
    assert not are_time_comparable(first, second)


def test_synchronized_sources_in_different_trials_are_not_comparable() -> None:
    """別の試技の Source は、同期済みでも比較しない（試技どうしの比較はイベント整列で行う）。"""
    first = _source(uuid4(), shares_trial_clock=True)
    second = _source(uuid4(), shares_trial_clock=True)
    assert not are_time_comparable(first, second)
