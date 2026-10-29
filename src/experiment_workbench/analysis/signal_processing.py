"""信号処理（SPEC §7 A-01〜A-04）: 入力検証、フィルタ、微分、積分。

処理クラスはどれも ``apply(t, x)`` で、時刻 t [s] と値 x（同じ長さの 1 次元配列）を受け取り、
x と同じ長さの新しい配列を返す。入力配列は変更しない（非破壊）。
時刻を秒以外の単位（ミリ秒など）で持っている場合は、呼び出し側で明示的に秒へ換算してから渡す。

入力が処理の前提を満たさない場合は、補間・除去などで勝手に直さず ``AnalysisInputError`` を送出する。
    - 時刻: 有限値で、狭義単調増加（逆行・重複なし）
    - 値: NaN・無限大を含まない
    - 等間隔を前提とする処理（バターワース、移動平均、Savitzky-Golay）:
      サンプリング間隔のばらつきが許容範囲内（欠損による時刻の飛びは必ず検出される）

計算そのものは SciPy / NumPy の既存関数に任せ、このモジュールは前提の確認と条件の受け渡しを担う。
このモジュールは Qt に依存しない。
"""

import numpy as np
from scipy import integrate, signal

from experiment_workbench.model import (
    ButterworthParams,
    DerivativeMethod,
    DerivativeParams,
    FilterPadType,
    IntegralParams,
    IntegrationMethod,
    MovingAverageParams,
    SavitzkyGolayParams,
)
from experiment_workbench.validation import FloatArray

DEFAULT_UNIFORM_TOLERANCE = 0.05
"""等間隔とみなすサンプリング間隔のばらつきの既定の許容値（中央値に対する相対値）。

5% にしているのは、ロガーの時刻ジッタ程度は許しつつ、1 サンプル以上の欠損（間隔が 2 倍以上）を
確実に検出するため。より厳しく・緩くしたい場合は各処理クラスの ``uniform_tolerance`` で指定する。
"""


class AnalysisInputError(ValueError):
    """解析処理に渡された時系列が、処理の前提を満たさない。"""


# ---------------------------------------------------------------------------
# 入力検証（A-01）
# ---------------------------------------------------------------------------


def validate_time_axis(t: FloatArray, *, min_samples: int = 2) -> FloatArray:
    """時刻配列が 1 次元・有限値・狭義単調増加であることを確認する。

    Args:
        t: 時刻 [s] の配列。
        min_samples: 必要な最小サンプル数。

    Returns:
        float64 に変換した時刻配列（入力が float64 ならコピーしない。変更はしないこと）。

    Raises:
        AnalysisInputError: 次元・長さが不正、非有限値を含む、または逆行・重複がある場合。
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1:
        raise AnalysisInputError(f"時刻は 1 次元の配列にしてください（実際: {times.ndim} 次元）")
    if times.size < min_samples:
        raise AnalysisInputError(
            f"サンプル数が {times.size} です（この処理には {min_samples} 以上必要）"
        )
    not_finite = ~np.isfinite(times)
    if not_finite.any():
        first = int(np.argmax(not_finite))
        raise AnalysisInputError(
            f"時刻に NaN・無限大が {int(not_finite.sum())} 個あります（最初の index: {first}）"
        )
    steps = np.diff(times)
    if (steps <= 0).any():
        index = int(np.argmax(steps <= 0))
        before, after = times[index], times[index + 1]
        kind = "重複" if steps[index] == 0 else "逆行"
        raise AnalysisInputError(
            f"時刻が{kind}しています（index {index} → {index + 1}: {before} → {after} s）。"
            "並べ替えや重複の除去は、内容を確認したうえで明示的に行ってください"
        )
    return times


def validate_series(t: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """時刻と値の組を検証する（値に NaN・無限大を許さない処理用）。

    Args:
        t: 時刻 [s] の配列。
        x: 値の配列（t と同じ長さの 1 次元配列）。

    Returns:
        float64 に変換した (時刻, 値)。

    Raises:
        AnalysisInputError: 時刻が不正、値の形が合わない、または値に NaN・無限大がある場合。
    """
    times = validate_time_axis(t)
    values = np.asarray(x, dtype=np.float64)
    if values.shape != times.shape:
        raise AnalysisInputError(f"値の形 {values.shape} が時刻の形 {times.shape} と一致しません")
    not_finite = ~np.isfinite(values)
    if not_finite.any():
        first = int(np.argmax(not_finite))
        raise AnalysisInputError(
            f"値に NaN・無限大が {int(not_finite.sum())} 個あります（最初の index: {first}）。"
            "欠損の扱い（区間の分割・補間など）は明示的に行ってください"
        )
    return times, values


def estimate_sampling_rate(t: FloatArray) -> float:
    """時刻配列からサンプリング周波数を推定する。

    外れ値（欠損による飛びなど）の影響を受けにくいよう、間隔の中央値の逆数を使う。

    Args:
        t: 検証済みの時刻 [s] の配列（狭義単調増加、2 サンプル以上）。

    Returns:
        推定したサンプリング周波数 [Hz]。
    """
    return float(1.0 / np.median(np.diff(t)))


def check_uniform_sampling(t: FloatArray, tolerance: float = DEFAULT_UNIFORM_TOLERANCE) -> float:
    """サンプリング間隔が一定とみなせることを確認し、サンプリング周波数を返す。

    各間隔 Δt_i と中央値 Δt_med について ``|Δt_i - Δt_med| / Δt_med <= tolerance`` を要求する。

    Args:
        t: 検証済みの時刻 [s] の配列。
        tolerance: 許容する相対的なばらつき。

    Returns:
        サンプリング周波数 [Hz]（間隔の中央値の逆数）。

    Raises:
        AnalysisInputError: ばらつきが許容値を超える場合。最も外れた間隔の位置と大きさを示す。
    """
    steps = np.diff(t)
    median = float(np.median(steps))
    deviation = np.abs(steps - median) / median
    worst = int(np.argmax(deviation))
    if deviation[worst] > tolerance:
        raise AnalysisInputError(
            f"サンプリング間隔が一定ではありません（index {worst} → {worst + 1} の間隔 "
            f"{steps[worst]:.6g} s は中央値 {median:.6g} s の {steps[worst] / median:.3g} 倍）。"
            "欠損による時刻の飛びの可能性があります。区間を分けるか、明示的にリサンプリングしてください"
        )
    return 1.0 / median


# ---------------------------------------------------------------------------
# フィルタ（A-02）
# ---------------------------------------------------------------------------


class ButterworthFilter:
    """バターワースフィルタ（低域通過・高域通過・帯域通過・帯域阻止）。

    数値的に安定な 2 次セクション（SOS）形式で設計し、ゼロ位相なら ``sosfiltfilt``、
    因果的なら ``sosfilt`` でかける。等間隔の時刻が必要。

    Attributes:
        params: フィルタの条件。
        uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
    """

    def __init__(
        self, params: ButterworthParams, uniform_tolerance: float = DEFAULT_UNIFORM_TOLERANCE
    ) -> None:
        """フィルタの条件を受け取って初期化する。

        Args:
            params: フィルタの条件。
            uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
        """
        self.params = params
        self.uniform_tolerance = uniform_tolerance

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """信号にフィルタをかける。

        Args:
            t: 時刻 [s]（等間隔）。
            x: 値。

        Returns:
            フィルタ後の値（x と同じ長さ）。

        Raises:
            AnalysisInputError: 入力が不正、不等間隔、カットオフ周波数がナイキスト周波数以上、
                またはゼロ位相フィルタの端点延長に対して信号が短すぎる場合。
        """
        times, values = validate_series(t, x)
        fs = check_uniform_sampling(times, self.uniform_tolerance)
        cutoffs = (
            self.params.cutoff_hz
            if isinstance(self.params.cutoff_hz, tuple)
            else (self.params.cutoff_hz,)
        )
        nyquist = fs / 2.0
        if max(cutoffs) >= nyquist:
            raise AnalysisInputError(
                f"カットオフ周波数 {max(cutoffs)} Hz がナイキスト周波数 {nyquist:.6g} Hz 以上です"
                f"（サンプリング周波数 {fs:.6g} Hz）"
            )
        sos = signal.butter(
            self.params.order,
            list(cutoffs) if len(cutoffs) == 2 else cutoffs[0],
            btype=self.params.kind.value,
            output="sos",
            fs=fs,
        )
        if not self.params.zero_phase:
            return np.asarray(signal.sosfilt(sos, values), dtype=np.float64)
        try:
            filtered = _sosfiltfilt(sos, values, self.params.padtype)
        except ValueError as exc:
            # 端点を延長する長さ（padlen）より信号が短いと SciPy が ValueError を送出する。
            raise AnalysisInputError(
                f"ゼロ位相フィルタをかけるには信号が短すぎます（{values.size} サンプル）: {exc}"
            ) from exc
        return np.asarray(filtered, dtype=np.float64)


class MovingAverageFilter:
    """単純移動平均（窓内の値の算術平均）。

    窓内のサンプルが揃わない端点は NaN にする（端点を折り返しなどで補うと、
    存在しない値を使った平均になるため）。等間隔の時刻が必要。

    Attributes:
        params: 移動平均の条件。
        uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
    """

    def __init__(
        self, params: MovingAverageParams, uniform_tolerance: float = DEFAULT_UNIFORM_TOLERANCE
    ) -> None:
        """移動平均の条件を受け取って初期化する。

        Args:
            params: 移動平均の条件。
            uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
        """
        self.params = params
        self.uniform_tolerance = uniform_tolerance

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """信号の移動平均を計算する。

        中心窓（窓長 w = 2h + 1）では y_i = mean(x_{i-h} … x_{i+h})、
        後方窓では y_i = mean(x_{i-w+1} … x_i) とする。

        Args:
            t: 時刻 [s]（等間隔）。
            x: 値。

        Returns:
            移動平均（x と同じ長さ。窓が揃わない端点は NaN）。

        Raises:
            AnalysisInputError: 入力が不正、不等間隔、または窓長がサンプル数より長い場合。
        """
        times, values = validate_series(t, x)
        check_uniform_sampling(times, self.uniform_tolerance)
        window = self.params.window_samples
        if window > values.size:
            raise AnalysisInputError(
                f"窓長 {window} がサンプル数 {values.size} より長いため移動平均を計算できません"
            )
        # mode="valid" は窓が完全に信号内に収まる位置だけを計算する（長さ n - w + 1）。
        averaged = np.convolve(values, np.full(window, 1.0 / window), mode="valid")
        result = np.full(values.size, np.nan, dtype=np.float64)
        start = (window - 1) // 2 if self.params.centered else window - 1
        stop = start + averaged.size
        result[start:stop] = averaged
        return result


class SavitzkyGolayFilter:
    """Savitzky-Golay 平滑化（窓内で多項式を最小二乗当てはめした値を使う）。

    移動平均よりピークの高さや形を保ちやすい。等間隔の時刻が必要。

    Attributes:
        params: Savitzky-Golay フィルタの条件。
        uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
    """

    def __init__(
        self, params: SavitzkyGolayParams, uniform_tolerance: float = DEFAULT_UNIFORM_TOLERANCE
    ) -> None:
        """フィルタの条件を受け取って初期化する。

        Args:
            params: Savitzky-Golay フィルタの条件。
            uniform_tolerance: 等間隔とみなすサンプリング間隔のばらつきの許容値。
        """
        self.params = params
        self.uniform_tolerance = uniform_tolerance

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """信号を平滑化する。

        Args:
            t: 時刻 [s]（等間隔）。
            x: 値。

        Returns:
            平滑化した値（x と同じ長さ）。

        Raises:
            AnalysisInputError: 入力が不正、不等間隔、または窓長がサンプル数より長い場合。
        """
        times, values = validate_series(t, x)
        check_uniform_sampling(times, self.uniform_tolerance)
        _require_window_fits(self.params.window_samples, values.size)
        smoothed = signal.savgol_filter(
            values,
            self.params.window_samples,
            self.params.polyorder,
            mode=self.params.mode.value,
        )
        return np.asarray(smoothed, dtype=np.float64)


# ---------------------------------------------------------------------------
# 微分（A-03）
# ---------------------------------------------------------------------------


class Differentiator:
    """数値微分（中心差分または Savitzky-Golay 微分）。

    中心差分は不等間隔の時刻に対応するが、ノイズを増幅する。ノイズの多い信号では
    Savitzky-Golay 微分（等間隔が必要）か、先にフィルタをかけることを検討する。

    Attributes:
        params: 微分の条件。
        uniform_tolerance: Savitzky-Golay 微分で等間隔とみなす間隔のばらつきの許容値。
    """

    def __init__(
        self, params: DerivativeParams, uniform_tolerance: float = DEFAULT_UNIFORM_TOLERANCE
    ) -> None:
        """微分の条件を受け取って初期化する。

        Args:
            params: 微分の条件。
            uniform_tolerance: Savitzky-Golay 微分で等間隔とみなす間隔のばらつきの許容値。
        """
        self.params = params
        self.uniform_tolerance = uniform_tolerance

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """信号を時間で微分する。

        中心差分では ``numpy.gradient`` を ``order`` 回適用する。内部の点は 2 次精度、
        端点は ``edge_order`` 次精度の片側差分になる。Savitzky-Golay 微分では、窓内で当てはめた
        多項式の ``order`` 階微分を、サンプリング間隔を考慮して評価する。

        Args:
            t: 時刻 [s]。
            x: 値。

        Returns:
            微分値（x と同じ長さ。単位は 値の単位 / s^order）。

        Raises:
            AnalysisInputError: 入力が不正、サンプル数が足りない、
                または Savitzky-Golay 微分で不等間隔・窓長が長すぎる場合。
        """
        times, values = validate_series(t, x)
        if self.params.method is DerivativeMethod.SAVITZKY_GOLAY:
            return self._savitzky_golay(times, values)
        # numpy.gradient は edge_order + 1 個以上のサンプルを必要とする。
        if values.size < self.params.edge_order + 1:
            raise AnalysisInputError(
                f"中心差分（edge_order={self.params.edge_order}）には "
                f"{self.params.edge_order + 1} サンプル以上が必要です（実際: {values.size}）"
            )
        result = values
        for _ in range(self.params.order):
            result = _gradient(result, times, self.params.edge_order)
        return result

    def _savitzky_golay(self, times: FloatArray, values: FloatArray) -> FloatArray:
        """Savitzky-Golay 微分を計算する。

        Args:
            times: 検証済みの時刻 [s]。
            values: 検証済みの値。

        Returns:
            微分値。

        Raises:
            AnalysisInputError: 不等間隔、または窓長がサンプル数より長い場合。
        """
        params = self.params.savitzky_golay
        if params is None:
            # DerivativeParams の検証で必ず指定されるため通常は起こらないが、型の上で明示する。
            raise AnalysisInputError("Savitzky-Golay 微分の条件（savitzky_golay）がありません")
        fs = check_uniform_sampling(times, self.uniform_tolerance)
        _require_window_fits(params.window_samples, values.size)
        derivative = signal.savgol_filter(
            values,
            params.window_samples,
            params.polyorder,
            deriv=self.params.order,
            delta=1.0 / fs,
            mode=params.mode.value,
        )
        return np.asarray(derivative, dtype=np.float64)


# ---------------------------------------------------------------------------
# 積分（A-04）
# ---------------------------------------------------------------------------


class Integrator:
    """累積数値積分（台形則またはシンプソン則）。不等間隔の時刻に対応する。

    結果の先頭は ``initial_value``（積分定数）とし、以降は先頭時刻からの積分値を足す。
    バイアスを含む信号を積分すると誤差が時間に比例して増える（ドリフト）点に注意する。

    Attributes:
        params: 積分の条件。
    """

    def __init__(self, params: IntegralParams) -> None:
        """積分の条件を受け取って初期化する。

        Args:
            params: 積分の条件。
        """
        self.params = params

    def apply(self, t: FloatArray, x: FloatArray) -> FloatArray:
        """信号を時間で累積積分する。

        台形則: y_i = y_0 + Σ_{k<i} (x_k + x_{k+1}) / 2 · (t_{k+1} - t_k)

        Args:
            t: 時刻 [s]。
            x: 値。

        Returns:
            積分値（x と同じ長さ。単位は 値の単位 × s）。

        Raises:
            AnalysisInputError: 入力が不正な場合。
        """
        times, values = validate_series(t, x)
        if self.params.method is IntegrationMethod.SIMPSON:
            integral = integrate.cumulative_simpson(values, x=times, initial=0.0)
        else:
            integral = integrate.cumulative_trapezoid(values, x=times, initial=0)
        return np.asarray(integral, dtype=np.float64) + self.params.initial_value


# ---------------------------------------------------------------------------
# 補助関数（モジュール内部用）
# ---------------------------------------------------------------------------


def _sosfiltfilt(sos: FloatArray, values: FloatArray, padtype: FilterPadType | None) -> FloatArray:
    """scipy.signal.sosfiltfilt でゼロ位相フィルタをかける。

    SciPy の型定義は padtype に文字列リテラルを要求するため、
    typing.Literal を使わずに済むよう列挙値ごとに分岐して直接渡す。

    Args:
        sos: 2 次セクション形式のフィルタ係数。
        values: 値。
        padtype: 端点の延長方法。None なら延長しない。

    Returns:
        フィルタ後の値。

    Raises:
        ValueError: 信号が端点の延長長さより短い場合（SciPy が送出する）。
    """
    match padtype:
        case FilterPadType.ODD:
            filtered = signal.sosfiltfilt(sos, values, padtype="odd")
        case FilterPadType.EVEN:
            filtered = signal.sosfiltfilt(sos, values, padtype="even")
        case FilterPadType.CONSTANT:
            filtered = signal.sosfiltfilt(sos, values, padtype="constant")
        case None:
            filtered = signal.sosfiltfilt(sos, values, padtype=None)
    return np.asarray(filtered, dtype=np.float64)


def _gradient(values: FloatArray, times: FloatArray, edge_order: int) -> FloatArray:
    """numpy.gradient で 1 階微分を計算する。

    numpy の型定義は edge_order に 1 か 2 のリテラルを要求するため、
    typing.Literal を使わずに済むよう分岐して直接渡す。

    Args:
        values: 値。
        times: 時刻 [s]（不等間隔でよい）。
        edge_order: 端点の差分の精度（1 または 2。DerivativeParams で検証済み）。

    Returns:
        1 階微分値。
    """
    if edge_order == 1:
        derivative = np.gradient(values, times, edge_order=1)
    else:
        derivative = np.gradient(values, times, edge_order=2)
    return np.asarray(derivative, dtype=np.float64)


def _require_window_fits(window_samples: int, n_samples: int) -> None:
    """窓長がサンプル数以下であることを確認する（Savitzky-Golay 用）。

    Args:
        window_samples: 窓の長さ [サンプル]。
        n_samples: 信号のサンプル数。

    Raises:
        AnalysisInputError: 窓長がサンプル数より長い場合。
    """
    if window_samples > n_samples:
        raise AnalysisInputError(
            f"窓長 {window_samples} がサンプル数 {n_samples} より長いため計算できません"
        )
