"""値の変換・時間の処理・区間統計（SPEC A-15〜A-17）。

    - 値の変換: オフセット除去、線形変換（倍率・単位換算）、トレンド除去、正規化
    - 時間の処理: 区間の切り出し、リサンプリング、欠損の補間
    - 区間統計: 平均・標準偏差・RMS・最大・最小など

値の変換は欠損（NaN）をそのまま残し、計算に使う統計量は有限の値だけから求める。
欠損を埋めるのは ``GapFiller`` だけで、埋めた時刻は結果に記録する（黙って修正しない）。
時間の処理は外挿せず、欠損区間をまたいで補間しない（A-11 と同じ規則）。
入力の配列は変更しない。このモジュールは Qt に依存しない。
"""

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from experiment_workbench.analysis.alignment import interpolate_on_grid
from experiment_workbench.analysis.signal_processing import (
    AnalysisInputError,
    ButterworthFilter,
    estimate_sampling_rate,
    validate_time_axis,
)
from experiment_workbench.model import (
    DEFAULT_MAX_GAP_FACTOR,
    ButterworthKind,
    ButterworthParams,
    CropParams,
    DetrendParams,
    FillGapsParams,
    GapFillResult,
    InterpolationMethod,
    NormalizeMode,
    NormalizeParams,
    OffsetMode,
    OffsetParams,
    ResampleParams,
    ResampleResult,
    ScaleParams,
    SectionStatistics,
    StatisticsParams,
)
from experiment_workbench.validation import BoolArray, FloatArray

ANTI_ALIAS_CUTOFF_RATIO = 0.4
"""アンチエイリアスフィルタのカットオフ周波数の、変換後のサンプリング周波数に対する比。

変換後のナイキスト周波数（比 0.5）より少し低くして、遷移域の成分が折り返さないようにする。
"""

ANTI_ALIAS_ORDER = 8
"""アンチエイリアスフィルタの次数（ゼロ位相なので実効的な次数は 2 倍）。

変換後のナイキスト周波数での減衰が約 30 dB になる（|H|² = 1 / (1 + 1.25¹⁶) を 2 回かける）。
"""

_GRID_EPSILON = 1e-9
"""等間隔の時刻の数を決めるときに、浮動小数点の誤差で終端が落ちないよう足す余裕（サンプル数）。"""


def _checked_values(times: FloatArray, values: FloatArray) -> FloatArray:
    """値の配列の形が時刻と合い、無限大を含まないことを確認する（欠損の NaN は許す）。

    Args:
        times: 時刻 [s]。
        values: 値。

    Returns:
        float64 に変換した値。

    Raises:
        AnalysisInputError: 形が合わない、または無限大を含む場合。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != np.shape(times):
        raise AnalysisInputError(
            f"値の形 {array.shape} が時刻の形 {np.shape(times)} と一致しません"
        )
    if np.isinf(array).any():
        raise AnalysisInputError("値に無限大が含まれています")
    return array


def _in_range(times: FloatArray, start_s: float | None, end_s: float | None) -> BoolArray:
    """指定した時刻の範囲に入るサンプルを返す。

    Args:
        times: 時刻 [s]。
        start_s: 範囲の開始 [s]。None なら先頭から。
        end_s: 範囲の終了 [s]。None なら末尾まで。

    Returns:
        範囲に入れば True の配列。
    """
    lower = -math.inf if start_s is None else start_s
    upper = math.inf if end_s is None else end_s
    return np.asarray((times >= lower) & (times <= upper), dtype=np.bool_)


class OffsetRemover:
    """オフセット（基準値）を差し引くクラス。

    基準値は、指定した区間（なければ全区間）の平均値か、指定した値。

    Attributes:
        params: オフセット除去の条件。
    """

    def __init__(self, params: OffsetParams) -> None:
        """オフセット除去の条件を受け取る。

        Args:
            params: オフセット除去の条件。
        """
        self.params = params

    def baseline(self, t: FloatArray, x: FloatArray) -> float:
        """差し引く基準値を返す。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            基準値。

        Raises:
            AnalysisInputError: 平均をとる区間に有限の値がない場合。
        """
        params = self.params
        if params.mode is OffsetMode.VALUE:
            return params.value
        values = _checked_values(t, x)
        selected = values[_in_range(np.asarray(t), params.start_s, params.end_s)]
        finite = selected[np.isfinite(selected)]
        if finite.size == 0:
            raise AnalysisInputError(
                "平均をとる区間に有効な値がありません。区間を選び直してください"
            )
        return float(np.mean(finite))

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """基準値を差し引く。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            基準値を差し引いた値。
        """
        return _checked_values(t, x) - self.baseline(t, x)


class LinearScaler:
    """線形変換 y = a x + b を行うクラス（倍率の補正・単位換算）。

    Attributes:
        params: 線形変換の条件。
    """

    def __init__(self, params: ScaleParams) -> None:
        """線形変換の条件を受け取る。

        Args:
            params: 線形変換の条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """線形変換を行う。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            変換後の値。
        """
        return self.params.gain * _checked_values(t, x) + self.params.offset


class Detrender:
    """最小二乗法で当てはめた多項式（トレンド）を差し引くクラス。

    Attributes:
        params: トレンド除去の条件。
    """

    def __init__(self, params: DetrendParams) -> None:
        """トレンド除去の条件を受け取る。

        Args:
            params: トレンド除去の条件。
        """
        self.params = params

    def trend(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """当てはめたトレンドを返す。

        大きな時刻の値で係数の精度が落ちないよう、``Polynomial.fit`` の定義域の変換を使う。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN。当てはめには使わない）。

        Returns:
            各時刻のトレンドの値。

        Raises:
            AnalysisInputError: 次数に対して有効な値が足りない場合。
        """
        times = validate_time_axis(t)
        values = _checked_values(times, x)
        finite = np.isfinite(values)
        if int(finite.sum()) <= self.params.order:
            raise AnalysisInputError(
                f"{self.params.order} 次のトレンドを求めるには、有効な値が足りません"
                f"（{int(finite.sum())} 個）"
            )
        fitted = np.polynomial.Polynomial.fit(times[finite], values[finite], self.params.order)
        return np.asarray(fitted(times), dtype=np.float64)

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """トレンドを差し引く。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            トレンドを除去した値。
        """
        return _checked_values(t, x) - self.trend(t, x)


class Normalizer:
    """値を基準の大きさで割って正規化するクラス。

    Attributes:
        params: 正規化の条件。
    """

    def __init__(self, params: NormalizeParams) -> None:
        """正規化の条件を受け取る。

        Args:
            params: 正規化の条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """値を正規化する。

            Z スコア: (x - 平均) / 標準偏差
            最小-最大: (x - 最小) / (最大 - 最小)
            ピーク: x / max|x|

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            正規化した値。

        Raises:
            AnalysisInputError: 有効な値がない、または値が一定で割れない場合。
        """
        values = _checked_values(t, x)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise AnalysisInputError("有効な値がないため、正規化できません")
        if self.params.mode is NormalizeMode.Z_SCORE:
            center, scale = float(np.mean(finite)), float(np.std(finite, ddof=1))
        elif self.params.mode is NormalizeMode.MIN_MAX:
            center, scale = float(np.min(finite)), float(np.ptp(finite))
        else:
            center, scale = 0.0, float(np.max(np.abs(finite)))
        if not scale > 0.0:
            raise AnalysisInputError("値が一定のため、正規化できません")
        return (values - center) / scale


class TimeCropper:
    """指定した時間の区間を切り出すクラス。

    Attributes:
        params: 切り出しの条件。
    """

    def __init__(self, params: CropParams) -> None:
        """切り出しの条件を受け取る。

        Args:
            params: 切り出しの条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """区間を切り出す（時刻はそのまま保つ。0 に揃えない）。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            (切り出した時刻, 切り出した値)。

        Raises:
            AnalysisInputError: 区間の中にサンプルが 2 個未満の場合。
        """
        times = np.asarray(t, dtype=np.float64)
        values = _checked_values(times, x)
        inside = _in_range(times, self.params.start_s, self.params.end_s)
        if int(inside.sum()) < 2:
            raise AnalysisInputError(
                f"{self.params.start_s:.6g}〜{self.params.end_s:.6g} s の中にサンプルが "
                f"{int(inside.sum())} 個しかありません。区間を選び直してください"
            )
        return times[inside], values[inside]


class Resampler:
    """等間隔の時刻にリサンプリングするクラス。

    時刻は先頭のサンプルの時刻から始める。外挿せず、欠損区間をまたいで補間しない。
    サンプリング周波数を下げるときは、高い周波数の成分が低い周波数に折り返す（エイリアシング）
    ので、既定でアンチエイリアスフィルタ（ゼロ位相のローパスフィルタ）を先にかける。

    Attributes:
        params: リサンプリングの条件。
    """

    def __init__(self, params: ResampleParams) -> None:
        """リサンプリングの条件を受け取る。

        Args:
            params: リサンプリングの条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> ResampleResult:
        """等間隔の時刻にリサンプリングする。

        Args:
            t: 時刻 [s]（狭義単調増加）。
            x: 値（欠損は NaN）。

        Returns:
            リサンプリングの結果。

        Raises:
            AnalysisInputError: 時刻が不正、または変換後のサンプルが 2 個未満の場合。
        """
        params = self.params
        times = validate_time_axis(t)
        values = _checked_values(times, x)
        count = math.floor((times[-1] - times[0]) * params.rate_hz + _GRID_EPSILON) + 1
        if count < 2:
            raise AnalysisInputError(
                f"サンプリング周波数 {params.rate_hz:.6g} Hz では、データの長さに対して"
                "サンプルが 2 個未満になります"
            )
        grid = times[0] + np.arange(count, dtype=np.float64) / params.rate_hz
        source_rate = estimate_sampling_rate(times)
        notes: list[str] = []
        prefiltered = values
        anti_aliased = False
        if params.rate_hz >= source_rate:
            notes.append("元のサンプリング周波数以上に上げても、情報は増えません")
        elif params.anti_alias:
            prefiltered, anti_aliased = self._anti_alias(times, values, notes)
        else:
            notes.append(
                "アンチエイリアスフィルタを使っていないので、変換後のナイキスト周波数より高い"
                "成分が折り返している可能性があります"
            )
        resampled, gap = interpolate_on_grid(
            times, prefiltered, grid, params.method, params.max_gap_factor, "入力"
        )
        if gap.any():
            notes.append(f"欠損区間の {int(gap.sum())} 時刻は値を求めていません")
        return ResampleResult(
            time_s=grid,
            values=resampled,
            source_rate_hz=source_rate,
            anti_aliased=anti_aliased,
            notes=notes,
        )

    def _anti_alias(
        self, times: FloatArray, values: FloatArray, notes: list[str]
    ) -> tuple[FloatArray, bool]:
        """アンチエイリアスフィルタをかける（かけられないときは理由を注記に残す）。

        Args:
            times: 時刻 [s]。
            values: 値。
            notes: 注記の追加先。

        Returns:
            (フィルタ後の値, フィルタをかけられたら True)。
        """
        cutoff = ANTI_ALIAS_CUTOFF_RATIO * self.params.rate_hz
        anti_alias = ButterworthFilter(
            ButterworthParams(
                kind=ButterworthKind.LOWPASS, order=ANTI_ALIAS_ORDER, cutoff_hz=cutoff
            )
        )
        try:
            return anti_alias.apply(times, values), True
        except AnalysisInputError as exc:
            # 欠損や不等間隔があるとかけられない。折り返しの可能性を隠さず知らせる。
            notes.append(f"アンチエイリアスフィルタをかけられませんでした: {exc}")
            return values, False


class GapFiller:
    """欠損（NaN）を前後の値から補間して埋めるクラス。

    外挿しないので、先頭と末尾の欠損は埋めない。``max_gap_s`` より長い欠損も埋めない
    （長い欠損を埋めると、存在しない波形を作ってしまうため）。

    時刻そのものが飛んでいる区間（行ごと欠けた区間）には値を入れる時刻がないので埋まらない。
    NaN がなくても「欠損なし」と読めないように、飛びの数と長さを結果で知らせる。

    Attributes:
        params: 欠損の補間の条件。
    """

    def __init__(self, params: FillGapsParams) -> None:
        """欠損の補間の条件を受け取る。

        Args:
            params: 欠損の補間の条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> GapFillResult:
        """欠損を補間する。

        Args:
            t: 時刻 [s]（狭義単調増加）。
            x: 値（欠損は NaN）。

        Returns:
            補間の結果（埋めたサンプルの位置を含む）。

        Raises:
            AnalysisInputError: 時刻が不正、または有効な値が 2 個未満の場合。
        """
        times = validate_time_axis(t)
        values = _checked_values(times, x)
        missing = np.isnan(values)
        filled = np.zeros(values.shape, dtype=np.bool_)
        jumps = _time_jumps(times)
        n_jumps = int(jumps.size)
        longest = float(jumps.max()) if n_jumps else None
        if not missing.any():
            return GapFillResult(
                values=values.copy(),
                filled_mask=filled,
                n_missing=0,
                n_filled=0,
                n_time_gaps=n_jumps,
                longest_time_gap_s=longest,
            )
        known = ~missing
        if int(known.sum()) < 2:
            raise AnalysisInputError("有効な値が 2 個未満のため、補間できません")
        for first, last in _missing_runs(missing):
            filled[first : last + 1] = self._can_fill(times, first, last)
        if self.params.method is InterpolationMethod.LINEAR:
            estimate = np.interp(times, times[known], values[known])
        else:
            estimate = PchipInterpolator(times[known], values[known], extrapolate=False)(times)
        return GapFillResult(
            values=np.where(filled, estimate, values),
            filled_mask=filled,
            n_missing=int(missing.sum()),
            n_filled=int(filled.sum()),
            n_time_gaps=n_jumps,
            longest_time_gap_s=longest,
        )

    def _can_fill(self, times: FloatArray, first: int, last: int) -> bool:
        """連続した欠損を埋めてよいかを返す。

        Args:
            times: 時刻 [s]。
            first: 欠損の先頭の番号。
            last: 欠損の末尾の番号。

        Returns:
            前後に有効な値があり、長さが上限以下なら True。
        """
        if first == 0 or last == times.size - 1:
            return False  # 外挿になるので埋めない
        limit = self.params.max_gap_s
        return limit is None or float(times[last + 1] - times[first - 1]) <= limit


def _time_jumps(times: FloatArray) -> FloatArray:
    """時刻が飛んでいる箇所の間隔を返す（読み込みの警告と同じ基準）。

    Args:
        times: 時刻 [s]（狭義単調増加）。

    Returns:
        間隔の中央値の ``DEFAULT_MAX_GAP_FACTOR`` 倍を超える間隔 [s] の配列。
    """
    steps = np.diff(times)
    if steps.size == 0:
        return steps
    return steps[steps > DEFAULT_MAX_GAP_FACTOR * float(np.median(steps))]


def _missing_runs(missing: BoolArray) -> list[tuple[int, int]]:
    """連続した欠損の区間を返す。

    Args:
        missing: 欠損なら True の配列。

    Returns:
        (先頭の番号, 末尾の番号) のリスト。
    """
    edges = np.flatnonzero(np.diff(np.concatenate(([False], missing, [False])).astype(np.int8)))
    return [
        (int(first), int(last) - 1) for first, last in zip(edges[::2], edges[1::2], strict=True)
    ]


class StatisticsCalculator:
    """区間の統計量を求めるクラス。

    Attributes:
        params: 区間統計の条件。
    """

    def __init__(self, params: StatisticsParams) -> None:
        """区間統計の条件を受け取る。

        Args:
            params: 区間統計の条件。
        """
        self.params = params

    def compute(self, t: FloatArray, x: FloatArray) -> SectionStatistics:
        """区間の統計量を求める（欠損は除いて計算し、その数を記録する）。

        標準偏差は不偏分散の平方根（自由度 n - 1）とする。

        Args:
            t: 時刻 [s]。
            x: 値（欠損は NaN）。

        Returns:
            区間の統計量。

        Raises:
            AnalysisInputError: 区間の中に有効な値がない場合。
        """
        times = np.asarray(t, dtype=np.float64)
        values = _checked_values(times, x)
        inside = _in_range(times, self.params.start_s, self.params.end_s)
        section_t, section_x = times[inside], values[inside]
        finite = np.isfinite(section_x)
        if not finite.any():
            raise AnalysisInputError("区間の中に有効な値がありません。区間を選び直してください")
        valid_t, valid_x = section_t[finite], section_x[finite]
        counts, edges = np.histogram(valid_x, bins=self.params.n_bins)
        largest, smallest = int(np.argmax(valid_x)), int(np.argmin(valid_x))
        return SectionStatistics(
            start_s=float(section_t[0]),
            end_s=float(section_t[-1]),
            n_samples=int(valid_x.size),
            n_missing=int(section_x.size - valid_x.size),
            mean=float(np.mean(valid_x)),
            std=float(np.std(valid_x, ddof=1)) if valid_x.size > 1 else 0.0,
            rms=float(np.sqrt(np.mean(valid_x**2))),
            minimum=float(valid_x[smallest]),
            maximum=float(valid_x[largest]),
            median=float(np.median(valid_x)),
            minimum_time_s=float(valid_t[smallest]),
            maximum_time_s=float(valid_t[largest]),
            histogram_centers=np.asarray(0.5 * (edges[:-1] + edges[1:]), dtype=np.float64),
            histogram_counts=np.asarray(counts, dtype=np.float64),
        )
