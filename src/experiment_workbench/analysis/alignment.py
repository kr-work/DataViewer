"""2 チャンネルの時刻の対応付け（SPEC A-11）。

入力と出力の 2 チャンネルを、同じ時刻に対応付けた組（AlignedPair）にする。入出力関係の解析
（応答・遅れ・ヒステリシス、Phase 5 の周波数応答）は、この組を使って計算する。

SPEC の原則「時刻原点・単位・サンプリング周波数を勝手に揃えない」に従い、次のように扱う。
    - 時刻をそのまま比較できない 2 チャンネル（時計が違う）は、時刻オフセットが明示されて
      いなければ対応付けない（エラーにする）。明示したオフセットは解析の条件として保存される。
    - 補間は、2 つのチャンネルの時刻が重なる範囲の中だけで行い、外挿しない。
    - 欠損区間（時刻の間隔が中央値の数倍を超える区間）はまたいで補間しない（その時刻は NaN）。
    - 時刻がまったく同じ 2 チャンネル（同じ Source の列）は、補間せずにそのまま対応付ける。
    - 行ったこと（オフセット、補間の有無と方法、範囲、欠損で値を求めなかった時刻の数）は
      AlignmentRecord として結果に残す。

このモジュールは Qt に依存しない。
"""

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from experiment_workbench.analysis.signal_processing import AnalysisInputError, validate_time_axis
from experiment_workbench.model import (
    AlignedPair,
    AlignmentGrid,
    AlignmentParams,
    AlignmentRecord,
    InterpolationMethod,
)
from experiment_workbench.validation import BoolArray, FloatArray

_GRID_EPSILON = 1e-9
"""等間隔の時刻の数を決めるときに、浮動小数点の誤差で終端が落ちないよう足す余裕（サンプル数）。"""


class ChannelAligner:
    """入力と出力の 2 チャンネルを同じ時刻に対応付けるクラス。

    使用例::

        aligner = ChannelAligner(AlignmentParams(grid=AlignmentGrid.INPUT))
        pair = aligner.align(t_in, x_in, t_out, x_out, time_comparable=True)

    Attributes:
        params: 対応付けの条件。
    """

    def __init__(self, params: AlignmentParams) -> None:
        """対応付けの条件を受け取る。

        Args:
            params: 対応付けの条件。
        """
        self.params = params

    def align(
        self,
        input_time_s: FloatArray,
        input_values: FloatArray,
        output_time_s: FloatArray,
        output_values: FloatArray,
        *,
        time_comparable: bool,
    ) -> AlignedPair:
        """2 つのチャンネルを同じ時刻に対応付ける。

        Args:
            input_time_s: 入力チャンネルの時刻 [s]（狭義単調増加）。
            input_values: 入力チャンネルの値（欠損は NaN）。
            output_time_s: 出力チャンネルの時刻 [s]（出力チャンネル自身の時間軸）。
            output_values: 出力チャンネルの値（欠損は NaN）。
            time_comparable: 2 つのチャンネルの時刻をそのまま比較できる（同じ Source、
                または同じ試技で同期済みの Source）なら True。

        Returns:
            対応付けた組（時刻は入力の時間軸）。

        Raises:
            AnalysisInputError: 時刻の対応が明示されていない、時刻や値が不正、重なる範囲がない、
                範囲の中に時刻が 2 つ未満、または PCHIP 補間で範囲の中に欠損がある場合。
        """
        params = self.params
        if params.output_time_offset_s is None and not time_comparable:
            raise AnalysisInputError(
                "入力と出力のチャンネルは時計が違うため、時刻をそのまま比較できません。"
                "同じ時計で記録したなら取り込み時に「時刻が同期済み」とするか、"
                "時刻オフセット（出力の時刻に足す値）を指定して時刻の対応を明示してください"
            )
        t_in = validate_time_axis(input_time_s)
        t_out_own = validate_time_axis(output_time_s)
        x_in = _checked_values(input_values, t_in, "入力")
        x_out = _checked_values(output_values, t_out_own, "出力")
        offset = 0.0 if params.output_time_offset_s is None else params.output_time_offset_s
        t_out = t_out_own + offset

        start, end = self._range(t_in, t_out)
        in_range_in = (t_in >= start) & (t_in <= end)
        in_range_out = (t_out >= start) & (t_out <= end)
        shared = (
            params.grid is not AlignmentGrid.UNIFORM
            and t_in.shape == t_out.shape
            and bool(np.array_equal(t_in, t_out))
        )
        # 同じ時刻列（同じ Source の列）なら、どちらの時刻を選んでも同じ並びになる。
        grid = (
            t_in[in_range_in]
            if shared
            else self._grid(t_in, t_out, in_range_in, in_range_out, start, end)
        )
        if grid.size < 2:
            raise AnalysisInputError(
                f"範囲 {start:.6g}〜{end:.6g} s の中に、対応付ける時刻が {grid.size} 個しか"
                "ありません（2 個以上必要です）。範囲を広げてください"
            )
        no_gap = np.zeros(grid.shape, dtype=np.bool_)
        if shared:
            # 補間せず、範囲の中のサンプルをそのまま使う。
            aligned_in, aligned_out, gap = x_in[in_range_in], x_out[in_range_out], no_gap
        else:
            if params.grid is AlignmentGrid.INPUT:
                aligned_in, gap_in = x_in[in_range_in], no_gap
            else:
                aligned_in, gap_in = self._resample(t_in, x_in, grid, "入力")
            if params.grid is AlignmentGrid.OUTPUT:
                aligned_out, gap_out = x_out[in_range_out], no_gap
            else:
                aligned_out, gap_out = self._resample(t_out, x_out, grid, "出力")
            gap = gap_in | gap_out
        record = AlignmentRecord(
            output_time_offset_s=offset,
            offset_declared=params.output_time_offset_s is not None,
            shared_timestamps=shared,
            input_interpolated=not shared and params.grid is not AlignmentGrid.INPUT,
            output_interpolated=not shared and params.grid is not AlignmentGrid.OUTPUT,
            method=params.method,
            start_s=float(grid[0]),
            end_s=float(grid[-1]),
            n_samples=int(grid.size),
            n_gap_samples=int(gap.sum()),
        )
        return AlignedPair(
            time_s=np.asarray(grid, dtype=np.float64),
            input_values=np.asarray(aligned_in, dtype=np.float64),
            output_values=np.asarray(aligned_out, dtype=np.float64),
            record=record,
        )

    def _range(self, t_in: FloatArray, t_out: FloatArray) -> tuple[float, float]:
        """2 つのチャンネルの時刻が重なる範囲と、指定した範囲の共通部分を返す。

        Args:
            t_in: 入力の時刻 [s]。
            t_out: 入力の時間軸に移した出力の時刻 [s]。

        Returns:
            (開始, 終了) [s]。

        Raises:
            AnalysisInputError: 時刻が重ならない、または指定した範囲が重なりの外の場合。
        """
        overlap_start = max(float(t_in[0]), float(t_out[0]))
        overlap_end = min(float(t_in[-1]), float(t_out[-1]))
        if overlap_start >= overlap_end:
            raise AnalysisInputError(
                f"入力（{t_in[0]:.6g}〜{t_in[-1]:.6g} s）と出力（入力の時間軸で "
                f"{t_out[0]:.6g}〜{t_out[-1]:.6g} s）の時刻が重なっていません。"
                "時刻の対応（同期の有無・時刻オフセット）を確認してください"
            )
        start = (
            overlap_start
            if self.params.start_s is None
            else max(overlap_start, self.params.start_s)
        )
        end = overlap_end if self.params.end_s is None else min(overlap_end, self.params.end_s)
        if start >= end:
            raise AnalysisInputError(
                f"指定した範囲（{self.params.start_s}〜{self.params.end_s} s）が、"
                f"2 つのチャンネルの時刻が重なる範囲（{overlap_start:.6g}〜{overlap_end:.6g} s）"
                "の外です"
            )
        return start, end

    def _grid(
        self,
        t_in: FloatArray,
        t_out: FloatArray,
        in_range_in: BoolArray,
        in_range_out: BoolArray,
        start: float,
        end: float,
    ) -> FloatArray:
        """対応付ける時刻の並びを作る。

        Args:
            t_in: 入力の時刻 [s]。
            t_out: 入力の時間軸に移した出力の時刻 [s]。
            in_range_in: 入力の時刻のうち範囲の中にあるもの。
            in_range_out: 出力の時刻のうち範囲の中にあるもの。
            start: 範囲の開始 [s]。
            end: 範囲の終了 [s]。

        Returns:
            時刻の並び [s]。
        """
        if self.params.grid is AlignmentGrid.INPUT:
            return t_in[in_range_in]
        if self.params.grid is AlignmentGrid.OUTPUT:
            return t_out[in_range_out]
        rate = self.params.uniform_rate_hz
        if rate is None:  # モデルの検証で起こらないが、型の上で明示する
            raise AnalysisInputError("等間隔の時刻のサンプリング周波数が指定されていません")
        n_samples = math.floor((end - start) * rate + _GRID_EPSILON) + 1
        return start + np.arange(n_samples, dtype=np.float64) / rate

    def _resample(
        self, t_src: FloatArray, v_src: FloatArray, grid: FloatArray, name: str
    ) -> tuple[FloatArray, BoolArray]:
        """チャンネルの値を、指定した時刻での値に補間する（外挿せず、欠損区間をまたがない）。

        時刻の並びは、元のチャンネルの時刻の範囲の中にある（重なりの範囲から作るため外挿にならない）。

        Args:
            t_src: 元のチャンネルの時刻 [s]。
            v_src: 元のチャンネルの値（欠損は NaN）。
            grid: 値を求める時刻 [s]。
            name: エラーメッセージに使うチャンネルの呼び名（"入力" など）。

        Returns:
            (補間した値, 欠損区間に入ったため NaN にした時刻)。

        Raises:
            AnalysisInputError: PCHIP 補間で、使う区間に欠損（NaN）がある場合。
        """
        # 時刻の並びを囲む区間だけを取り出す（PCHIP の計算を軽くし、欠損の確認を範囲内に限るため）。
        first = max(int(np.searchsorted(t_src, grid[0], side="right")) - 1, 0)
        last = min(int(np.searchsorted(t_src, grid[-1], side="left")) + 1, t_src.size)
        t_win, v_win = t_src[first:last], v_src[first:last]
        if self.params.method is InterpolationMethod.LINEAR:
            # 線形補間では、欠損（NaN）に隣り合う区間の値は NaN になる（欠損を埋めない）。
            values = np.interp(grid, t_win, v_win)
        else:
            missing = np.isnan(v_win)
            if missing.any():
                raise AnalysisInputError(
                    f"{name}チャンネルの範囲内に欠損（NaN）が {int(missing.sum())} 個あるため、"
                    "PCHIP 補間はできません。線形補間を選ぶか、欠損のない範囲を選んでください"
                )
            values = PchipInterpolator(t_win, v_win, extrapolate=False)(grid)
        gap = _in_gap(t_src, t_win, grid, self.params.max_gap_factor)
        values = np.where(gap, np.nan, values)
        return np.asarray(values, dtype=np.float64), gap


def _checked_values(values: FloatArray, times: FloatArray, name: str) -> FloatArray:
    """値の配列の形が時刻と合い、無限大を含まないことを確認する（欠損の NaN は許す）。

    Args:
        values: 値の配列。
        times: 検証済みの時刻 [s]。
        name: エラーメッセージに使うチャンネルの呼び名。

    Returns:
        float64 に変換した値。

    Raises:
        AnalysisInputError: 形が合わない、または無限大を含む場合。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != times.shape:
        raise AnalysisInputError(
            f"{name}チャンネルの値の形 {array.shape} が時刻の形 {times.shape} と一致しません"
        )
    if np.isinf(array).any():
        raise AnalysisInputError(f"{name}チャンネルの値に無限大が含まれています")
    return array


def _in_gap(
    t_src: FloatArray, t_win: FloatArray, grid: FloatArray, max_gap_factor: float
) -> BoolArray:
    """値を求める時刻のうち、元のチャンネルの欠損区間の中にあるものを返す。

    欠損区間は、時刻の間隔が「チャンネル全体の間隔の中央値 × max_gap_factor」を超える区間。
    区間の両端（実際のサンプルの時刻）は欠損に含めない。

    Args:
        t_src: 元のチャンネルの時刻（全体。間隔の中央値を求めるのに使う）。
        t_win: 値を求める時刻を囲む区間の時刻。
        grid: 値を求める時刻。
        max_gap_factor: 欠損とみなす間隔の、中央値に対する倍率。

    Returns:
        欠損区間の中にあれば True の配列（grid と同じ長さ）。
    """
    steps = np.diff(t_win)
    if steps.size == 0:
        return np.zeros(grid.shape, dtype=np.bool_)
    median = float(np.median(np.diff(t_src)))
    gap_intervals = steps > max_gap_factor * median
    if not gap_intervals.any():
        return np.zeros(grid.shape, dtype=np.bool_)
    # grid の各時刻が入る区間の番号 j（t_win[j] <= t < t_win[j + 1]）。
    interval = np.clip(np.searchsorted(t_win, grid, side="right") - 1, 0, steps.size - 1)
    inside = (grid > t_win[interval]) & (grid < t_win[interval + 1])
    return np.asarray(gap_intervals[interval] & inside, dtype=np.bool_)
