"""時間軸に関する処理: 時刻の単位換算と、Source どうしの時刻を比較できるかの判定。

SPEC の原則「異なるデータソースの時刻原点・単位・サンプリング周波数を勝手に揃えない」に従い、
時刻を比較できるのは次の場合だけとする。
    - 同じ Source の列どうし（同じ時刻列を共有している）
    - 同じ試技の中で、どちらも「同期済み」（TimeBase.shares_trial_clock）と明示された Source どうし

時刻オフセットによる対応付けや補間（SPEC A-11 の残り）は Phase 4 で追加する。
このモジュールは Qt に依存しない。
"""

import numpy as np

from experiment_workbench.model import Source, TimeUnit
from experiment_workbench.validation import FloatArray

SECONDS_PER_UNIT: dict[TimeUnit, float] = {
    TimeUnit.SECOND: 1.0,
    TimeUnit.MILLISECOND: 1e-3,
    TimeUnit.MICROSECOND: 1e-6,
    TimeUnit.NANOSECOND: 1e-9,
}
"""時刻の単位 → 秒への換算係数。"""


def to_seconds(raw: FloatArray, unit: TimeUnit) -> FloatArray:
    """時刻の値を秒に換算した新しい配列を返す（入力は変更しない）。

    Args:
        raw: 元の単位の時刻。
        unit: 元の単位。

    Returns:
        秒に換算した時刻。
    """
    return np.asarray(raw * SECONDS_PER_UNIT[unit], dtype=np.float64)


def time_domain_key(source: Source) -> str:
    """時刻を直接比較できる Source の組を表す識別子を返す。

    同じ識別子を持つ Source どうしだけが、同じ時間軸に並べて比較できる。
        - 同期済みの Source: ``"trial:<試技 ID>"``（同じ試技の同期済み Source と共通）
        - それ以外: ``"source:<Source ID>"``（自分自身とだけ比較できる）

    Args:
        source: 判定する Source。

    Returns:
        時間軸の識別子。
    """
    if source.time_base.shares_trial_clock:
        return f"trial:{source.trial_id}"
    return f"source:{source.id}"


def are_time_comparable(first: Source, second: Source) -> bool:
    """2 つの Source の時刻を、変換なしで直接比較できるかを返す。

    Args:
        first: 1 つ目の Source。
        second: 2 つ目の Source。

    Returns:
        同じ Source、または同じ試技でどちらも同期済みなら True。
    """
    return time_domain_key(first) == time_domain_key(second)
