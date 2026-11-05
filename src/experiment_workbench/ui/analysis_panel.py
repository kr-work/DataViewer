"""解析パネル（SPEC F-11）: 処理を選び、見ながら条件を調整する。

メイン画面の右側のドックに置く。画面は 2 つで、迷わないように一度に片方だけを出す。
    1. 処理の一覧: 左の一覧で選んだチャンネルに対して行う処理のボタンが並ぶ。
       押すとすぐに計算して、結果がグラフに重なる。
    2. 条件の調整: 選んだ処理の条件を編集する。主要な数値にはスライダーを付けている。

実行や保存のボタンはない。条件を変えると、最後の変更から少し待って実行を求め
（``run_requested``）、計算が終わった結果は自動で保存される（AnalysisController が行う）。
数値を打っている途中で毎回計算し直さないよう、少し待っている。
実行・表示・保存はこのパネルの外で行い、このパネルは条件の編集と指示だけを受け持つ。
2 チャンネルの解析（ステップ応答・ヒステリシス・相互相関）では、出力チャンネル・解析する範囲・
時刻の対応付けの条件も編集する（AlignmentPage）。範囲はグラフの上でも選べる。
不正な値は pydantic の検証エラーとしてパネル内に表示し、実行しない。
"""

import math
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from experiment_workbench.analysis import (
    default_initial_state,
    default_kalman_settings,
    default_omega,
    estimate_noise_std,
    estimate_sampling_rate,
    round_significant,
)
from experiment_workbench.model import (
    TWO_CHANNEL_ANALYSIS_KINDS,
    AdaptiveNoiseMethod,
    AdaptiveNoiseParams,
    AlignmentGrid,
    AlignmentParams,
    AnalysisKind,
    AnalysisSpec,
    ButterworthKind,
    ButterworthParams,
    ChannelData,
    CrossCorrelationParams,
    DerivativeMethod,
    DerivativeParams,
    DetrendMode,
    FilterPadType,
    HysteresisParams,
    IntegralParams,
    IntegrationMethod,
    InterpolationMethod,
    KalmanInitialState,
    KalmanModelKind,
    KalmanSettings,
    KinematicSystemParams,
    LinearSystemParams,
    MovingAverageParams,
    OnsetThresholdMode,
    PolynomialObservationParams,
    SavitzkyGolayMode,
    SavitzkyGolayParams,
    StepDirection,
    StepResponseParams,
)
from experiment_workbench.ui.theme import MUTED

AUTO_RUN_DELAY_MS = 200
"""条件を変えてから再計算するまでの待ち時間 [ms]（スライダーの動きに追従して見える長さ）。"""

BUSY_DISPLAY_DELAY_MS = 300
"""計算中の表示を出すまでの待ち時間 [ms]（すぐ終わる計算で表示がちらつかないように）。"""

KIND_LABELS = {
    AnalysisKind.BUTTERWORTH: "バターワースフィルタ",
    AnalysisKind.MOVING_AVERAGE: "移動平均",
    AnalysisKind.SAVITZKY_GOLAY: "Savitzky-Golay 平滑化",
    AnalysisKind.DERIVATIVE: "微分",
    AnalysisKind.INTEGRAL: "積分",
    AnalysisKind.KALMAN: "カルマンフィルタ",
    AnalysisKind.STEP_RESPONSE: "ステップ応答（遅れ・時定数）",
    AnalysisKind.HYSTERESIS: "ヒステリシス・入出力特性",
    AnalysisKind.CROSS_CORRELATION: "相互相関（遅れ）",
}
"""解析の種類の表示名。"""

PRESET_NONE = ""
"""処理の一覧のボタンの、追加の設定なし。"""
PRESET_KF = "kf"
"""カルマンフィルタ（運動学モデル、R は一定）。"""
PRESET_EKF = "ekf"
"""拡張カルマンフィルタ（非線形の観測モデル）。"""
PRESET_AKF = "akf"
"""適応カルマンフィルタ（観測雑音 R を逐次推定）。"""

METHOD_GROUPS = (
    (
        "フィルタ",
        "ノイズなどの不要な変動を取り除きます",
        (
            (
                "バターワース",
                AnalysisKind.BUTTERWORTH,
                PRESET_NONE,
                "周波数を指定して成分を取り除きます。迷ったらこれ",
            ),
            (
                "移動平均",
                AnalysisKind.MOVING_AVERAGE,
                PRESET_NONE,
                "前後のサンプルの平均をとります",
            ),
            (
                "Savitzky-Golay",
                AnalysisKind.SAVITZKY_GOLAY,
                PRESET_NONE,
                "ピークの形を保ちながら平滑化します",
            ),
        ),
    ),
    (
        "微分・積分",
        "変化の速さや、積み上げた量を求めます",
        (
            ("微分", AnalysisKind.DERIVATIVE, PRESET_NONE, "時間で微分します"),
            ("積分", AnalysisKind.INTEGRAL, PRESET_NONE, "時間で積分します"),
        ),
    ),
    (
        "カルマンフィルタ",
        "ノイズを除きながら、値と変化の速さを推定します",
        (
            (
                "KF",
                AnalysisKind.KALMAN,
                PRESET_KF,
                "線形カルマンフィルタ。値とその微分を推定します",
            ),
            (
                "EKF",
                AnalysisKind.KALMAN,
                PRESET_EKF,
                "拡張カルマンフィルタ。非線形な観測に使います",
            ),
            (
                "AKF",
                AnalysisKind.KALMAN,
                PRESET_AKF,
                "適応カルマンフィルタ。観測雑音を逐次推定します",
            ),
        ),
    ),
    (
        "2 チャンネルの関係",
        "入力に対して、出力がどう動くかを調べます",
        (
            (
                "ステップ応答",
                AnalysisKind.STEP_RESPONSE,
                PRESET_NONE,
                "入力の変化に対する出力の遅れ・時定数を求めます",
            ),
            (
                "ヒステリシス",
                AnalysisKind.HYSTERESIS,
                PRESET_NONE,
                "増加時と減少時の入出力特性の差を求めます",
            ),
            (
                "相互相関",
                AnalysisKind.CROSS_CORRELATION,
                PRESET_NONE,
                "2 つのチャンネルの間の時間の遅れを求めます",
            ),
        ),
    ),
)
"""処理の一覧: (見出し, 説明, ((ボタンの名前, 解析の種類, 追加の設定, 説明), ...)) の並び。"""

_REMOVED_HINT = "取り除いた成分は、左の一覧のチェックで見られます。"
_METHOD_DESCRIPTIONS = {
    AnalysisKind.MOVING_AVERAGE: (
        "前後のサンプルの平均をとって滑らかにします。窓長を大きくするほど滑らかになりますが、"
        f"急な変化も鈍ります。{_REMOVED_HINT}"
    ),
    AnalysisKind.SAVITZKY_GOLAY: (
        "短い区間ごとに多項式を当てはめて滑らかにします。ピークの高さや形が崩れにくい方法です。"
        f"窓長を大きくするほど滑らかになります。{_REMOVED_HINT}"
    ),
    AnalysisKind.DERIVATIVE: (
        "値がどれだけ速く変化しているか（傾き）を求めます。ノイズは強調されるので、"
        "線が荒いときは Savitzky-Golay 微分を選びます。"
    ),
    AnalysisKind.INTEGRAL: (
        "値を時間で積み上げます（加速度 → 速度 など）。わずかなずれも時間とともに積み重なるので、"
        "線が傾いていくことがあります。"
    ),
    AnalysisKind.STEP_RESPONSE: (
        "入力が急に変わったとき、出力がどれだけ遅れて動き始め、どれくらいで追いつくかを測ります。"
        "青い帯の両端を動かして、入力の変化が 1 回だけ入る範囲を選びます。"
    ),
    AnalysisKind.HYSTERESIS: (
        "入力を増やすときと減らすときで、同じ入力でも出力が違うかを調べます。"
        "入力が行って戻る範囲を選ぶと、行きと帰りの差が分かります。"
    ),
    AnalysisKind.CROSS_CORRELATION: (
        "2 つの波形を少しずつずらして重ね、最もよく重なるずれを遅れとして求めます。"
        "連続的に変化している区間に向いています。"
    ),
}
"""処理の説明（何をするか、条件を動かすとどうなるか）。条件で変わるものはページが返す。"""

_BUTTERWORTH_DESCRIPTIONS = {
    ButterworthKind.LOWPASS: (
        "カットオフより速い変動を取り除き、ゆっくりした変化を残します。"
        f"カットオフを下げるほど滑らかになりますが、速い動きも消えます。{_REMOVED_HINT}"
    ),
    ButterworthKind.HIGHPASS: (
        "カットオフより遅い変動（ゆっくりしたずれ）を取り除き、速い変動を残します。"
        f"カットオフを上げるほど、より多くの変動を取り除きます。{_REMOVED_HINT}"
    ),
    ButterworthKind.BANDPASS: (f"カットオフと上限の間の速さの変動だけを残します。{_REMOVED_HINT}"),
    ButterworthKind.BANDSTOP: (
        f"カットオフと上限の間の速さの変動だけを取り除きます（電源ノイズなど）。{_REMOVED_HINT}"
    ),
}
"""バターワースフィルタの種類ごとの説明。"""

_KALMAN_DESCRIPTION = (
    "雑音のある観測から、値とその変化の速さを推定します。"
    "プロセス雑音 q を大きくすると観測に素早く追従し、観測雑音 σ を大きくすると滑らかになります。"
)
_EKF_DESCRIPTION = (
    "観測が推定したい量の非線形な式で表せるとき（電圧から張力を求めるなど）に使います。"
    "係数で観測の式を指定します。q を大きくすると素早く追従し、σ を大きくすると滑らかになります。"
)
_LINEAR_DESCRIPTION = "状態遷移 F・観測 H・雑音 Q・R を行列で指定して推定します。"
_ADAPTIVE_DESCRIPTION = (
    "推定の途中で、観測雑音の大きさ R も観測から推定します。"
    "雑音の大きさが時間とともに変わるデータに使います。"
)

_KIND_SHORT_NAMES = {
    AnalysisKind.BUTTERWORTH: "バターワース",
    AnalysisKind.MOVING_AVERAGE: "移動平均",
    AnalysisKind.SAVITZKY_GOLAY: "SG 平滑化",
    AnalysisKind.DERIVATIVE: "微分",
    AnalysisKind.INTEGRAL: "積分",
    AnalysisKind.KALMAN: "カルマン",
    AnalysisKind.STEP_RESPONSE: "ステップ応答",
    AnalysisKind.HYSTERESIS: "ヒステリシス",
    AnalysisKind.CROSS_CORRELATION: "相互相関",
}
"""解析の名前の既定値に使う短い名前（凡例や一覧で読みやすいように）。"""

_BUTTERWORTH_LABELS = {
    ButterworthKind.LOWPASS: "低域通過（LPF）",
    ButterworthKind.HIGHPASS: "高域通過（HPF）",
    ButterworthKind.BANDPASS: "帯域通過（BPF）",
    ButterworthKind.BANDSTOP: "帯域阻止（BSF）",
}
_PADTYPE_LABELS = {
    FilterPadType.ODD.value: "点対称に折り返す（odd）",
    FilterPadType.EVEN.value: "線対称に折り返す（even）",
    FilterPadType.CONSTANT.value: "端の値で延長（constant）",
    "": "延長しない",
}
_SAVGOL_MODE_LABELS = {
    SavitzkyGolayMode.INTERP: "端の窓の多項式で評価（interp）",
    SavitzkyGolayMode.MIRROR: "折り返し（mirror）",
    SavitzkyGolayMode.NEAREST: "端の値で延長（nearest）",
    SavitzkyGolayMode.CONSTANT: "0 で延長（constant）",
    SavitzkyGolayMode.WRAP: "循環（wrap）",
}
_MODEL_LABELS = {
    KalmanModelKind.KINEMATIC: "運動学モデル（値とその微分）",
    KalmanModelKind.LINEAR: "線形モデル（行列 F・H・Q・R を指定）",
    KalmanModelKind.POLYNOMIAL_OBSERVATION: "EKF: 運動学モデル + 多項式の観測",
}
_ADAPTIVE_LABELS = {
    "": "使わない（R は一定）",
    AdaptiveNoiseMethod.INNOVATION_WINDOW.value: "窓付きイノベーション推定",
    AdaptiveNoiseMethod.SAGE_HUSA.value: "Sage-Husa 推定",
}
_MAX_LINEAR_STATES = 6
_MAX_POLYNOMIAL_DEGREE = 5
_METHOD_LABELS = {
    InterpolationMethod.LINEAR.value: "線形補間",
    InterpolationMethod.PCHIP.value: "PCHIP（形を保つ 3 次補間）",
}
_GRID_LABELS = {
    AlignmentGrid.INPUT.value: "入力の時刻（出力を補間）",
    AlignmentGrid.OUTPUT.value: "出力の時刻（入力を補間）",
    AlignmentGrid.UNIFORM.value: "等間隔の時刻（両方を補間）",
}
_DIRECTION_LABELS = {
    StepDirection.AUTO.value: "自動（初期値と最終値から判定）",
    StepDirection.RISING.value: "増加",
    StepDirection.FALLING.value: "減少",
}
_ONSET_LABELS = {
    OnsetThresholdMode.FRACTION.value: "変化量に対する割合",
    OnsetThresholdMode.ABSOLUTE.value: "入力の単位の大きさ",
}
_DETREND_LABELS = {
    DetrendMode.CONSTANT.value: "平均を引く",
    DetrendMode.LINEAR.value: "直線の傾向を引く",
}
_BUTTERWORTH_SHORT_NAMES = {
    ButterworthKind.LOWPASS: "LPF",
    ButterworthKind.HIGHPASS: "HPF",
    ButterworthKind.BANDPASS: "BPF",
    ButterworthKind.BANDSTOP: "BSF",
}
_DEFAULT_MAX_LAG_S = 1.0
"""相互相関で遅れを探す範囲の既定の上限 [s]。"""
_DEFAULT_LAG_FRACTION = 0.1
"""相互相関で遅れを探す範囲の既定の上限の、データの長さに対する割合。"""
_SLIDER_STEPS = 1000
"""スライダーの刻みの数。"""
_SLIDER_MIN_WIDTH = 100
"""スライダーの最小の幅 [px]（項目名が長くても、つまみを動かせる幅を残す）。"""
_SLIDER_SIGNIFICANT_DIGITS = 3
"""スライダーで決めた実数の有効桁数（端数のない見やすい値にする）。"""
_MIN_WINDOW_LIMIT = 11
"""窓長のスライダーの上限の下限 [サンプル]（データが短くても動かせる幅を残す）。"""
_WINDOW_LIMIT_FRACTION = 0.1
"""窓長のスライダーの上限の、データ長に対する割合。"""
_MIN_CUTOFF_FRACTION = 1.0 / 2000.0
"""カットオフ周波数のスライダーの下限の、サンプリング周波数に対する割合。"""
_MAX_CUTOFF_FRACTION = 0.49
"""カットオフ周波数のスライダーの上限の、サンプリング周波数に対する割合（ナイキスト未満）。"""
_NOISE_SLIDER_SPAN = 1000.0
"""カルマンフィルタの雑音のスライダーの幅（現在の値の 1/この値 〜 この値の倍）。"""


# ---------------------------------------------------------------------------
# 入力部品
# ---------------------------------------------------------------------------


class FloatEdit(QLineEdit):
    """実数を入力する欄（"1e-3" のような指数表記も受け付ける）。

    Attributes:
        label: エラーメッセージに使う項目名。
    """

    def __init__(self, label: str, value: float, parent: QWidget | None = None) -> None:
        """項目名と初期値を受け取る。

        Args:
            label: 項目名。
            value: 初期値。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.label = label
        self.set_value(value)

    def value(self) -> float:
        """入力された実数を返す。

        Returns:
            実数。

        Raises:
            ValueError: 実数として読めない場合（項目名を含むメッセージ）。
        """
        try:
            return float(self.text().strip())
        except ValueError:
            raise ValueError(f"「{self.label}」には数値を入力してください") from None

    def set_value(self, value: float) -> None:
        """値を、読み戻すと同じ値になる最短の表記で表示する。

        保存した解析を読み込んで再計算したとき、表示の丸めで条件が変わらないようにするため
        （有効数字で丸めると、読み込んだだけで条件と計算結果のキャッシュが別物になる）。

        Args:
            value: 表示する値。
        """
        self.setText(_exact_text(value))


class MatrixEditor(QTableWidget):
    """実数の行列を表で編集する部品（カルマンフィルタの F・H・Q・R・初期状態など）。

    Attributes:
        label: エラーメッセージに使う行列名。
    """

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        """行列名を受け取る。

        Args:
            label: 行列名（例: "F"）。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.label = label
        self.horizontalHeader().setDefaultSectionSize(90)
        self.verticalHeader().setDefaultSectionSize(24)

    def set_values(self, values: list[list[float]]) -> None:
        """行列の値を表示する（表の大きさも合わせる。表記は FloatEdit と同じく丸めない）。

        Args:
            values: 行のリスト。
        """
        self.blockSignals(True)
        rows = len(values)
        cols = len(values[0]) if values else 0
        self.setRowCount(rows)
        self.setColumnCount(cols)
        for row, row_values in enumerate(values):
            for col, value in enumerate(row_values):
                self.setItem(row, col, QTableWidgetItem(_exact_text(value)))
        self.blockSignals(False)
        self._fit_height()

    def values(self) -> list[list[float]]:
        """表の値を行列として返す。

        Returns:
            行のリスト。

        Raises:
            ValueError: 実数として読めないセルがある場合（行列名と位置を含むメッセージ）。
        """
        matrix: list[list[float]] = []
        for row in range(self.rowCount()):
            row_values: list[float] = []
            for col in range(self.columnCount()):
                item = self.item(row, col)
                text = item.text().strip() if item is not None else ""
                try:
                    row_values.append(float(text))
                except ValueError:
                    raise ValueError(
                        f"行列 {self.label} の {row + 1} 行 {col + 1} 列が数値ではありません"
                    ) from None
            matrix.append(row_values)
        return matrix

    def resize_matrix(self, rows: int, cols: int, diagonal: float | None = None) -> None:
        """行列の大きさを変える。既存の値は残し、新しいセルは 0（対角は ``diagonal``）にする。

        Args:
            rows: 行数。
            cols: 列数。
            diagonal: 新しい対角要素の値。None なら 0。
        """
        try:
            current = self.values()
        except ValueError:
            current = []
        resized = [
            [
                current[row][col]
                if row < len(current) and col < len(current[row])
                else (diagonal if diagonal is not None and row == col else 0.0)
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        self.set_values(resized)

    def _fit_height(self) -> None:
        """行数に合わせて表の高さを決める（スクロールしなくても全体が見えるように）。"""
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(row) for row in range(self.rowCount()))
        self.setFixedHeight(min(height + 4, 220))


def _exact_text(value: float) -> str:
    """実数を、float() で読み戻すと同じ値になる最短の文字列にする（"0.1"、"1e-12" など）。

    Args:
        value: 実数。

    Returns:
        文字列。
    """
    return repr(float(value))


class SliderField(QWidget):
    """数値の欄の横にスライダーを付けた部品（結果を見ながら値を動かすため）。

    数値の欄は値の持ち主のままで、スライダーは欄の値を動かす手段にすぎない。正確な値は欄に打てる。
    周波数や雑音の強さは桁で効くので、スライダーは対数の目盛りにしている。

    Attributes:
        slider: スライダー。
    """

    value_changed = Signal()
    """スライダーの操作で値が変わった。"""

    def __init__(
        self,
        target: FloatEdit | QSpinBox,
        low: float,
        high: float,
        *,
        odd: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """数値の欄とスライダーの範囲を受け取る。

        Args:
            target: 値を持つ数値の欄。
            low: スライダーの左端の値（正の値）。
            high: スライダーの右端の値。
            odd: 整数の欄で奇数だけを選べるようにするなら True（中心窓の窓長など）。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._target = target
        self._odd = odd
        self._low = low
        self._high = high
        self._dragging = False
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.slider.setMinimumWidth(_SLIDER_MIN_WIDTH)
        self.slider.valueChanged.connect(self._on_slider_moved)
        target.setMaximumWidth(84)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(target)
        if isinstance(target, FloatEdit):
            target.textChanged.connect(self._on_target_changed)
        else:
            target.valueChanged.connect(self._on_target_changed)
        self.sync()

    def set_range(self, low: float, high: float) -> None:
        """スライダーの範囲を変える（値は変えない）。

        Args:
            low: 左端の値（正の値）。
            high: 右端の値。
        """
        if not (0 < low < high):
            return
        self._low, self._high = low, high
        self.sync()

    def sync(self) -> None:
        """スライダーの位置を、数値の欄の値に合わせる（範囲の外なら端に置く）。"""
        value = self._target_value()
        if value is None or value <= 0:
            return
        ratio = math.log(value / self._low) / math.log(self._high / self._low)
        self.slider.blockSignals(True)
        self.slider.setValue(round(min(max(ratio, 0.0), 1.0) * _SLIDER_STEPS))
        self.slider.blockSignals(False)

    def _target_value(self) -> float | None:
        """数値の欄の値を返す。

        Returns:
            値。読めなければ None。
        """
        if isinstance(self._target, QSpinBox):
            return float(self._target.value())
        try:
            return self._target.value()
        except ValueError:
            return None

    def _on_slider_moved(self, position: int) -> None:
        """スライダーの位置を値に直して、数値の欄に入れる。

        Args:
            position: スライダーの位置。
        """
        value = self._low * (self._high / self._low) ** (position / _SLIDER_STEPS)
        self._dragging = True
        try:
            if isinstance(self._target, QSpinBox):
                number = round(value)
                self._target.setValue(number | 1 if self._odd else number)
            else:
                self._target.set_value(round_significant(value, _SLIDER_SIGNIFICANT_DIGITS))
        finally:
            self._dragging = False
        self.value_changed.emit()

    def _on_target_changed(self, *args: object) -> None:
        """数値の欄が直接編集されたら、スライダーを合わせる。

        Args:
            *args: 部品のシグナルの引数（使わない）。
        """
        if not self._dragging:
            self.sync()


def _form_layout(parent: QWidget) -> QFormLayout:
    """条件の入力欄を並べるレイアウトを作る。

    項目名が長いときは入力欄を次の行に回す（パネルの幅が狭くても入力欄が潰れないように）。

    Args:
        parent: レイアウトを置く部品。

    Returns:
        レイアウト。
    """
    form = QFormLayout(parent)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


class _PageStack(QWidget):
    """ページを 1 つだけ表示する入れ物（表示中のページの高さだけを使う）。

    QStackedWidget は、表示していないページの高さも全体の高さに含めるため、小さいページの下に
    大きな空白ができる（項目を折り返すフォームでは、サイズポリシーでも無視させられない）。
    ここでは表示しないページを本当に隠して、レイアウトの計算から外す。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """空の入れ物を作る。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._pages: list[QWidget] = []

    def add_page(self, page: QWidget) -> None:
        """ページを加える（隠した状態で加える）。

        Args:
            page: ページ。
        """
        self._pages.append(page)
        self._layout.addWidget(page)
        page.setVisible(False)

    def show_page(self, shown: QWidget) -> None:
        """指定したページだけを表示する。

        Args:
            shown: 表示するページ。
        """
        for page in self._pages:
            page.setVisible(page is shown)


class CollapsibleSection(QWidget):
    """見出しをクリックすると中身を開閉できる部品（使う頻度の低い条件を畳んでおくため）。

    Attributes:
        toggle_button: 見出しのボタン。
        body: 中身の部品。
    """

    def __init__(
        self, title: str, body: QWidget, *, expanded: bool = False, parent: QWidget | None = None
    ) -> None:
        """見出しと中身を受け取る。

        Args:
            title: 見出し。
            body: 中身の部品。
            expanded: 最初から開いておくなら True。
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.body = body
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setStyleSheet("font-weight: 600; border: none;")
        self.toggle_button.toggled.connect(self.set_expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(body)
        self.toggle_button.setChecked(expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        """中身の表示を切り替える。

        Args:
            expanded: 開くなら True。
        """
        self.body.setVisible(expanded)
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self.toggle_button.setArrowType(arrow)


def _combo(labels: dict[str, str]) -> QComboBox:
    """表示名と値の対応から選択欄を作る（値は文字列で持たせる）。

    Args:
        labels: 値 → 表示名。

    Returns:
        選択欄。
    """
    combo = QComboBox()
    for value, label in labels.items():
        combo.addItem(label, str(value))
    return combo


def _select(combo: QComboBox, value: str) -> None:
    """選択欄で値に対応する項目を選ぶ。

    Args:
        combo: 選択欄。
        value: 選ぶ値（文字列）。
    """
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    """整数の入力欄を作る。

    Args:
        minimum: 最小値。
        maximum: 最大値。
        value: 初期値。

    Returns:
        整数の入力欄。
    """
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


# ---------------------------------------------------------------------------
# 解析の種類ごとの条件のページ
# ---------------------------------------------------------------------------


class _ParamsPage(QWidget):
    """条件のページの共通部分（変更の通知）。"""

    changed = Signal()
    """条件が変更された。"""

    def short_name(self) -> str:
        """結果の名前に使う短い説明を返す（凡例で条件の違いが分かるように）。

        Returns:
            短い説明。空なら解析の種類の短い名前を使う。
        """
        return ""

    def description(self) -> str:
        """条件で変わる処理の説明を返す。

        Returns:
            説明。空なら解析の種類ごとの説明を使う。
        """
        return ""

    def _emit_changed(self, *args: object) -> None:
        """部品の変更シグナルを受けて、条件の変更を知らせる。

        Args:
            *args: 部品のシグナルの引数（使わない）。
        """
        self.changed.emit()

    def _watch(self, *widgets: QWidget) -> None:
        """部品の変更を ``changed`` につなぐ。

        Args:
            *widgets: 監視する部品。
        """
        for widget in widgets:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._emit_changed)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._emit_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._emit_changed)
            elif isinstance(widget, QLineEdit):
                widget.textEdited.connect(self._emit_changed)
            elif isinstance(widget, QTableWidget):
                widget.itemChanged.connect(self._emit_changed)
            elif isinstance(widget, SliderField):
                widget.value_changed.connect(self._emit_changed)


class ButterworthPage(_ParamsPage):
    """バターワースフィルタの条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.kind_combo = _combo({kind.value: label for kind, label in _BUTTERWORTH_LABELS.items()})
        self.order_spin = _spin(1, 10, 4)
        self.low_edit = FloatEdit("カットオフ周波数（下限）", 10.0)
        self.high_edit = FloatEdit("カットオフ周波数（上限）", 20.0)
        self.zero_phase_check = QCheckBox("ゼロ位相（前後にかけて位相遅れをなくす）")
        self.zero_phase_check.setChecked(True)
        self.padtype_combo = _combo(_PADTYPE_LABELS)
        self.low_slider = SliderField(self.low_edit, 0.1, 1000.0)
        self.high_slider = SliderField(self.high_edit, 0.1, 1000.0)
        details = QWidget()
        details_form = _form_layout(details)
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.addRow("端点の延長", self.padtype_combo)
        main = QWidget()
        self._form = _form_layout(main)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.addRow("種類", self.kind_combo)
        self._form.addRow("カットオフ [Hz]", self.low_slider)
        self._form.addRow("上限 [Hz]", self.high_slider)
        self._form.addRow("次数", self.order_spin)
        self._form.addRow("", self.zero_phase_check)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.addWidget(main)
        layout.addWidget(CollapsibleSection("詳細", details))
        self.kind_combo.currentIndexChanged.connect(self._update_enabled)
        self._watch(
            self.kind_combo,
            self.order_spin,
            self.low_edit,
            self.high_edit,
            self.low_slider,
            self.high_slider,
            self.zero_phase_check,
            self.padtype_combo,
        )
        self._update_enabled()

    def params(self) -> ButterworthParams:
        """入力された条件を返す。

        Returns:
            バターワースフィルタの条件。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合。
        """
        kind = ButterworthKind(str(self.kind_combo.currentData()))
        band = kind in (ButterworthKind.BANDPASS, ButterworthKind.BANDSTOP)
        cutoff: float | tuple[float, float] = (
            (self.low_edit.value(), self.high_edit.value()) if band else self.low_edit.value()
        )
        padtype = str(self.padtype_combo.currentData())
        return ButterworthParams(
            kind=kind,
            order=self.order_spin.value(),
            cutoff_hz=cutoff,
            zero_phase=self.zero_phase_check.isChecked(),
            padtype=FilterPadType(padtype) if padtype else None,
        )

    def load(self, params: ButterworthParams) -> None:
        """条件を表示する。

        Args:
            params: バターワースフィルタの条件。
        """
        _select(self.kind_combo, params.kind.value)
        self.order_spin.setValue(params.order)
        if isinstance(params.cutoff_hz, tuple):
            self.low_edit.set_value(params.cutoff_hz[0])
            self.high_edit.set_value(params.cutoff_hz[1])
        else:
            self.low_edit.set_value(params.cutoff_hz)
        self.zero_phase_check.setChecked(params.zero_phase)
        _select(self.padtype_combo, params.padtype.value if params.padtype is not None else "")

    def set_default_cutoff(self, sampling_rate_hz: float) -> None:
        """入力の周波数に合わせてカットオフ周波数の初期値を決める（ナイキスト周波数未満）。

        下限は 10 Hz とサンプリング周波数の 1/10 の小さい方、上限はその 2 倍とナイキスト周波数の
        0.8 倍の小さい方。推定したサンプリング周波数の端数を見せないよう、有効数字 3 桁に丸める
        （丸めても 1/10・0.8 倍の余裕があるので、ナイキスト周波数を超えない）。

        Args:
            sampling_rate_hz: 入力のサンプリング周波数 [Hz]。
        """
        low = round_significant(min(10.0, sampling_rate_hz / 10.0), 3)
        self.low_edit.set_value(low)
        self.high_edit.set_value(round_significant(min(2.0 * low, 0.4 * sampling_rate_hz), 3))
        self.set_sampling_rate(sampling_rate_hz)

    def description(self) -> str:
        """選んでいるフィルタの種類の説明を返す。

        Returns:
            説明。
        """
        return _BUTTERWORTH_DESCRIPTIONS[ButterworthKind(str(self.kind_combo.currentData()))]

    def set_sampling_rate(self, sampling_rate_hz: float) -> None:
        """スライダーの範囲を、入力のサンプリング周波数に合わせる（ナイキスト周波数未満）。

        Args:
            sampling_rate_hz: 入力のサンプリング周波数 [Hz]。
        """
        low = sampling_rate_hz * _MIN_CUTOFF_FRACTION
        high = sampling_rate_hz * _MAX_CUTOFF_FRACTION
        self.low_slider.set_range(low, high)
        self.high_slider.set_range(low, high)

    def short_name(self) -> str:
        """結果の名前に使う短い説明を返す（"LPF 10 Hz" など）。

        Returns:
            短い説明。読めない値があれば空。
        """
        kind = ButterworthKind(str(self.kind_combo.currentData()))
        try:
            low, high = self.low_edit.value(), self.high_edit.value()
        except ValueError:
            return ""
        band = kind in (ButterworthKind.BANDPASS, ButterworthKind.BANDSTOP)
        cutoff = f"{low:.4g}-{high:.4g}" if band else f"{low:.4g}"
        return f"{_BUTTERWORTH_SHORT_NAMES[kind]} {cutoff} Hz"

    def _update_enabled(self) -> None:
        """帯域通過・帯域阻止のときだけ上限の欄を出す。"""
        kind = ButterworthKind(str(self.kind_combo.currentData()))
        band = kind in (ButterworthKind.BANDPASS, ButterworthKind.BANDSTOP)
        self.high_edit.setEnabled(band)
        self._form.setRowVisible(self.high_slider, band)


class MovingAveragePage(_ParamsPage):
    """移動平均の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.window_spin = _spin(1, 1_000_000, 11)
        self.centered_check = QCheckBox("中心窓（位相遅れなし。窓長は奇数）")
        self.centered_check.setChecked(True)
        self.window_slider = SliderField(self.window_spin, 1, _MIN_WINDOW_LIMIT, odd=True)
        form = _form_layout(self)
        form.addRow("窓長 [サンプル]", self.window_slider)
        form.addRow("", self.centered_check)
        self._watch(self.window_spin, self.window_slider, self.centered_check)

    def params(self) -> MovingAverageParams:
        """入力された条件を返す。

        Returns:
            移動平均の条件。

        Raises:
            pydantic.ValidationError: 中心窓で窓長が偶数の場合など。
        """
        return MovingAverageParams(
            window_samples=self.window_spin.value(), centered=self.centered_check.isChecked()
        )

    def load(self, params: MovingAverageParams) -> None:
        """条件を表示する。

        Args:
            params: 移動平均の条件。
        """
        self.window_spin.setValue(params.window_samples)
        self.centered_check.setChecked(params.centered)

    def set_data_size(self, samples: int) -> None:
        """スライダーの範囲を、データの長さに合わせる。

        Args:
            samples: データのサンプル数。
        """
        limit = max(_MIN_WINDOW_LIMIT, round(samples * _WINDOW_LIMIT_FRACTION))
        self.window_slider.set_range(1, limit)

    def short_name(self) -> str:
        """結果の名前に使う短い説明を返す。

        Returns:
            短い説明。
        """
        return f"移動平均 {self.window_spin.value()}"


class SavitzkyGolayPage(_ParamsPage):
    """Savitzky-Golay 平滑化の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.window_spin = _spin(1, 1_000_001, 11)
        self.polyorder_spin = _spin(0, 10, 2)
        self.mode_combo = _combo({mode.value: label for mode, label in _SAVGOL_MODE_LABELS.items()})
        self.window_slider = SliderField(self.window_spin, 3, _MIN_WINDOW_LIMIT, odd=True)
        details = QWidget()
        details_form = _form_layout(details)
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.addRow("端点の扱い", self.mode_combo)
        main = QWidget()
        form = _form_layout(main)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("窓長 [サンプル]", self.window_slider)
        form.addRow("多項式の次数", self.polyorder_spin)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.addWidget(main)
        layout.addWidget(CollapsibleSection("詳細", details))
        self._watch(self.window_spin, self.window_slider, self.polyorder_spin, self.mode_combo)

    def params(self) -> SavitzkyGolayParams:
        """入力された条件を返す。

        Returns:
            Savitzky-Golay 平滑化の条件。

        Raises:
            pydantic.ValidationError: 窓長が偶数、または次数が窓長以上の場合。
        """
        return SavitzkyGolayParams(
            window_samples=self.window_spin.value(),
            polyorder=self.polyorder_spin.value(),
            mode=SavitzkyGolayMode(str(self.mode_combo.currentData())),
        )

    def load(self, params: SavitzkyGolayParams) -> None:
        """条件を表示する。

        Args:
            params: Savitzky-Golay 平滑化の条件。
        """
        self.window_spin.setValue(params.window_samples)
        self.polyorder_spin.setValue(params.polyorder)
        _select(self.mode_combo, params.mode.value)

    def set_data_size(self, samples: int) -> None:
        """スライダーの範囲を、データの長さに合わせる。

        Args:
            samples: データのサンプル数。
        """
        limit = max(_MIN_WINDOW_LIMIT, round(samples * _WINDOW_LIMIT_FRACTION))
        self.window_slider.set_range(3, limit)

    def short_name(self) -> str:
        """結果の名前に使う短い説明を返す。

        Returns:
            短い説明。
        """
        return f"SG {self.window_spin.value()}"


class DerivativePage(_ParamsPage):
    """微分の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.method_combo = _combo(
            {
                DerivativeMethod.GRADIENT.value: "中心差分（不等間隔に対応）",
                DerivativeMethod.SAVITZKY_GOLAY.value: "Savitzky-Golay 微分（平滑化付き）",
            }
        )
        self.order_spin = _spin(1, 2, 1)
        self.edge_order_spin = _spin(1, 2, 2)
        self.window_spin = _spin(3, 1_000_001, 21)
        self.polyorder_spin = _spin(1, 10, 3)
        form = _form_layout(self)
        form.addRow("方法", self.method_combo)
        form.addRow("階数", self.order_spin)
        form.addRow("端点の差分の精度（中心差分）", self.edge_order_spin)
        form.addRow("窓長 [サンプル]（SG）", self.window_spin)
        form.addRow("多項式の次数（SG）", self.polyorder_spin)
        self.method_combo.currentIndexChanged.connect(self._update_enabled)
        self._watch(
            self.method_combo,
            self.order_spin,
            self.edge_order_spin,
            self.window_spin,
            self.polyorder_spin,
        )
        self._update_enabled()

    def params(self) -> DerivativeParams:
        """入力された条件を返す。

        Returns:
            微分の条件。

        Raises:
            pydantic.ValidationError: 条件が不正な場合。
        """
        method = DerivativeMethod(str(self.method_combo.currentData()))
        savgol = (
            SavitzkyGolayParams(
                window_samples=self.window_spin.value(), polyorder=self.polyorder_spin.value()
            )
            if method is DerivativeMethod.SAVITZKY_GOLAY
            else None
        )
        return DerivativeParams(
            method=method,
            order=self.order_spin.value(),
            edge_order=self.edge_order_spin.value(),
            savitzky_golay=savgol,
        )

    def load(self, params: DerivativeParams) -> None:
        """条件を表示する。

        Args:
            params: 微分の条件。
        """
        _select(self.method_combo, params.method.value)
        self.order_spin.setValue(params.order)
        self.edge_order_spin.setValue(params.edge_order)
        if params.savitzky_golay is not None:
            self.window_spin.setValue(params.savitzky_golay.window_samples)
            self.polyorder_spin.setValue(params.savitzky_golay.polyorder)

    def _update_enabled(self) -> None:
        """方法に合わせて、使う欄だけを入力できるようにする。"""
        savgol = (
            DerivativeMethod(str(self.method_combo.currentData()))
            is DerivativeMethod.SAVITZKY_GOLAY
        )
        self.edge_order_spin.setEnabled(not savgol)
        self.window_spin.setEnabled(savgol)
        self.polyorder_spin.setEnabled(savgol)


class IntegralPage(_ParamsPage):
    """積分の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.method_combo = _combo(
            {
                IntegrationMethod.TRAPEZOID.value: "累積台形則",
                IntegrationMethod.SIMPSON.value: "累積シンプソン則",
            }
        )
        self.initial_edit = FloatEdit("初期値（積分定数）", 0.0)
        form = _form_layout(self)
        form.addRow("方法", self.method_combo)
        form.addRow("初期値（積分定数）", self.initial_edit)
        self._watch(self.method_combo, self.initial_edit)

    def params(self) -> IntegralParams:
        """入力された条件を返す。

        Returns:
            積分の条件。

        Raises:
            ValueError: 初期値が数値として読めない場合。
        """
        return IntegralParams(
            method=IntegrationMethod(str(self.method_combo.currentData())),
            initial_value=self.initial_edit.value(),
        )

    def load(self, params: IntegralParams) -> None:
        """条件を表示する。

        Args:
            params: 積分の条件。
        """
        _select(self.method_combo, params.method.value)
        self.initial_edit.set_value(params.initial_value)


class AlignmentPage(_ParamsPage):
    """2 チャンネルの解析に共通の条件（出力チャンネル・解析する範囲・時刻の対応）。

    時刻の対応と補間は、同じファイルの列どうしなら意識しなくてよいので、下にまとめて置く。
    """

    range_toggled = Signal(bool)
    """範囲の指定のオン・オフが変わった（引数: オンなら True）。"""
    fit_range_requested = Signal()
    """範囲を、縦並べの表示中の範囲に合わせることが求められた。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._channel_sources: dict[UUID, UUID] = {}
        self.output_combo = QComboBox()
        self.range_check = QCheckBox("範囲を指定（グラフの青い帯をドラッグ）")
        self.start_edit = FloatEdit("範囲の開始", 0.0)
        self.end_edit = FloatEdit("範囲の終了", 1.0)
        fit_button = QPushButton("表示中の範囲に合わせる")
        fit_button.clicked.connect(self._on_fit_clicked)
        self.fit_range_button = fit_button
        self.offset_check = QCheckBox("時刻オフセットを指定")
        self.offset_check.setToolTip(
            "時計が違うファイルのチャンネルを対応付けるときに、時刻の対応を明示します"
        )
        self.offset_edit = FloatEdit("時刻オフセット", 0.0)
        self.offset_edit.setToolTip("出力チャンネルの時刻に足して、入力の時刻に揃える値 [s]")
        self.method_combo = _combo(_METHOD_LABELS)
        self.grid_combo = _combo(_GRID_LABELS)
        self.rate_edit = FloatEdit("サンプリング周波数", 1000.0)

        main = QWidget()
        self._form = _form_layout(main)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.addRow("出力チャンネル", self.output_combo)
        self._form.addRow(self.range_check)
        self._form.addRow("開始 [s]", self.start_edit)
        self._form.addRow("終了 [s]", self.end_edit)
        self._form.addRow(self.fit_range_button)
        details = QWidget()
        details_form = _form_layout(details)
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.addRow(self.offset_check)
        details_form.addRow("時刻オフセット [s]", self.offset_edit)
        details_form.addRow("補間の方法", self.method_combo)
        details_form.addRow("対応付ける時刻", self.grid_combo)
        details_form.addRow("周波数 [Hz]（等間隔）", self.rate_edit)
        # 同じファイルの列どうしなら触らなくてよい条件なので、畳んでおく。
        self.details_section = CollapsibleSection("時刻の対応と補間（別のファイルのとき）", details)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main)
        layout.addWidget(self.details_section)
        self.range_check.toggled.connect(self._on_range_toggled)
        self.offset_check.toggled.connect(self._update_enabled)
        self.grid_combo.currentIndexChanged.connect(self._update_enabled)
        self._watch(
            self.output_combo,
            self.range_check,
            self.start_edit,
            self.end_edit,
            self.offset_check,
            self.offset_edit,
            self.method_combo,
            self.grid_combo,
            self.rate_edit,
        )
        self._update_enabled()

    def set_channel_choices(self, choices: list[tuple[UUID, str, UUID]]) -> None:
        """出力チャンネルの選択肢を入れ替える（選んでいたチャンネルは残す）。

        Args:
            choices: (チャンネル ID, 表示名, Source ID) のリスト（一覧の順）。
        """
        current = self.output_combo.currentData()
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        self._channel_sources = {channel_id: source_id for channel_id, _, source_id in choices}
        for channel_id, label, _ in choices:
            self.output_combo.addItem(label, str(channel_id))
        _select(self.output_combo, str(current))
        self.output_combo.blockSignals(False)

    def select_default_output(self, input_channel_id: UUID, source_id: UUID) -> None:
        """入力に合う出力チャンネルを選ぶ。

        入力と同じファイルの別の列があればそれを選ぶ（時刻の対応がそのまま取れるため）。
        同じファイルの列を既に選んでいれば変えない。別のファイルにしかなければ、
        入力と同じチャンネルを選んでいるときだけ、次のチャンネルに変える。

        Args:
            input_channel_id: 入力チャンネルの ID。
            source_id: 入力チャンネルの Source の ID。
        """
        same_file = [
            channel_id
            for channel_id, channel_source in self._channel_sources.items()
            if channel_source == source_id and channel_id != input_channel_id
        ]
        current = self.output_channel_id()
        if current in same_file:
            return
        if same_file:
            _select(self.output_combo, str(same_file[0]))
        elif self.output_combo.count() >= 2 and current in (None, input_channel_id):
            index = max(self.output_combo.findData(str(input_channel_id)), 0)
            self.output_combo.setCurrentIndex((index + 1) % self.output_combo.count())

    def output_channel_id(self) -> UUID | None:
        """選んでいる出力チャンネルの ID を返す。

        Returns:
            チャンネルの ID。選べるチャンネルがなければ None。
        """
        data = self.output_combo.currentData()
        return UUID(data) if isinstance(data, str) else None

    def params(self) -> AlignmentParams:
        """入力された条件を返す。

        Returns:
            対応付けの条件。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合（範囲の開始が終了以降など）。
        """
        grid = AlignmentGrid(str(self.grid_combo.currentData()))
        ranged = self.range_check.isChecked()
        return AlignmentParams(
            output_time_offset_s=self.offset_edit.value()
            if self.offset_check.isChecked()
            else None,
            method=InterpolationMethod(str(self.method_combo.currentData())),
            grid=grid,
            uniform_rate_hz=self.rate_edit.value() if grid is AlignmentGrid.UNIFORM else None,
            start_s=self.start_edit.value() if ranged else None,
            end_s=self.end_edit.value() if ranged else None,
        )

    def load(self, params: AlignmentParams, output_channel_id: UUID | None) -> None:
        """条件を表示する。

        Args:
            params: 対応付けの条件。
            output_channel_id: 出力チャンネルの ID。
        """
        _select(self.output_combo, str(output_channel_id))
        offset = params.output_time_offset_s
        self.offset_check.setChecked(offset is not None)
        if offset is not None:
            self.offset_edit.set_value(offset)
        _select(self.method_combo, params.method.value)
        _select(self.grid_combo, params.grid.value)
        if params.uniform_rate_hz is not None:
            self.rate_edit.set_value(params.uniform_rate_hz)
        ranged = params.start_s is not None and params.end_s is not None
        if params.start_s is not None and params.end_s is not None:
            self.set_range(params.start_s, params.end_s)
        self.range_check.setChecked(ranged)
        self._update_enabled()

    def set_range(self, start_s: float, end_s: float) -> None:
        """解析する範囲の欄を更新する（有効数字 7 桁に丸めて見やすくする）。

        Args:
            start_s: 開始 [s]。
            end_s: 終了 [s]。
        """
        self.start_edit.set_value(round_significant(start_s, 7))
        self.end_edit.set_value(round_significant(end_s, 7))

    def range_values(self) -> tuple[float, float] | None:
        """範囲の欄の値を返す。

        Returns:
            (開始, 終了) [s]。読めない、または開始が終了以降なら None。
        """
        try:
            start, end = self.start_edit.value(), self.end_edit.value()
        except ValueError:
            return None
        return (start, end) if start < end else None

    def _on_range_toggled(self, enabled: bool) -> None:
        """範囲の指定のオン・オフを知らせる。

        Args:
            enabled: オンなら True。
        """
        self._update_enabled()
        self.range_toggled.emit(enabled)

    def _on_fit_clicked(self) -> None:
        """範囲を表示中の範囲に合わせることを求める。"""
        self.fit_range_requested.emit()

    def _update_enabled(self) -> None:
        """チェックや選択に合わせて、使う欄だけを入力できるようにする。"""
        ranged = self.range_check.isChecked()
        for widget in (self.start_edit, self.end_edit, self.fit_range_button):
            widget.setEnabled(ranged)
            self._form.setRowVisible(widget, ranged)
        self.offset_edit.setEnabled(self.offset_check.isChecked())
        uniform = str(self.grid_combo.currentData()) == AlignmentGrid.UNIFORM.value
        self.rate_edit.setEnabled(uniform)


class StepResponsePage(_ParamsPage):
    """ステップ応答の指標の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        hint = QLabel(
            "範囲に入力の変化が 1 回だけ入るようにします。範囲の先頭と末尾の区間の平均を、"
            "変化前の値（初期値）と変化後の値（最終値）にします。"
        )
        hint.setWordWrap(True)
        self.direction_combo = _combo(_DIRECTION_LABELS)
        self.onset_mode_combo = _combo(_ONSET_LABELS)
        self.threshold_edit = FloatEdit("変化の開始のしきい値", 0.1)
        self.baseline_edit = FloatEdit("初期値の区間", 0.1)
        self.final_edit = FloatEdit("最終値の区間", 0.1)
        self.band_edit = FloatEdit("整定の幅", 0.05)
        self.smoothing_check = QCheckBox("平滑化してから求める")
        self.smoothing_check.setToolTip(
            "雑音が多いときに、入力と出力にローパスをかけてから求めます"
        )
        self.smoothing_edit = FloatEdit("平滑化のカットオフ周波数", 100.0)
        form = _form_layout(self)
        form.addRow(hint)
        form.addRow("入力の変化の向き", self.direction_combo)
        form.addRow("変化の開始のしきい値", self.onset_mode_combo)
        form.addRow("しきい値", self.threshold_edit)
        form.addRow("初期値の区間（範囲に対する割合）", self.baseline_edit)
        form.addRow("最終値の区間（範囲に対する割合）", self.final_edit)
        form.addRow("整定の幅（変化量に対する割合）", self.band_edit)
        form.addRow(self.smoothing_check)
        form.addRow("カットオフ [Hz]", self.smoothing_edit)
        self.smoothing_check.toggled.connect(self.smoothing_edit.setEnabled)
        self.smoothing_edit.setEnabled(False)
        self._watch(
            self.direction_combo,
            self.onset_mode_combo,
            self.threshold_edit,
            self.baseline_edit,
            self.final_edit,
            self.band_edit,
            self.smoothing_check,
            self.smoothing_edit,
        )

    def params(self) -> StepResponseParams:
        """入力された条件を返す。

        Returns:
            ステップ応答の条件。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合。
        """
        smoothed = self.smoothing_check.isChecked()
        return StepResponseParams(
            direction=StepDirection(str(self.direction_combo.currentData())),
            onset_mode=OnsetThresholdMode(str(self.onset_mode_combo.currentData())),
            onset_threshold=self.threshold_edit.value(),
            baseline_fraction=self.baseline_edit.value(),
            final_fraction=self.final_edit.value(),
            settling_band=self.band_edit.value(),
            smoothing_cutoff_hz=self.smoothing_edit.value() if smoothed else None,
        )

    def load(self, params: StepResponseParams) -> None:
        """条件を表示する。

        Args:
            params: ステップ応答の条件。
        """
        _select(self.direction_combo, params.direction.value)
        _select(self.onset_mode_combo, params.onset_mode.value)
        self.threshold_edit.set_value(params.onset_threshold)
        self.baseline_edit.set_value(params.baseline_fraction)
        self.final_edit.set_value(params.final_fraction)
        self.band_edit.set_value(params.settling_band)
        self.smoothing_check.setChecked(params.smoothing_cutoff_hz is not None)
        if params.smoothing_cutoff_hz is not None:
            self.smoothing_edit.set_value(params.smoothing_cutoff_hz)


class HysteresisPage(_ParamsPage):
    """入出力特性とヒステリシスの条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        hint = QLabel(
            "入力が往復する範囲を選びます。入力の変化の向きで増加時と減少時に分けて、"
            "同じ入力の値での出力の差を求めます。"
        )
        hint.setWordWrap(True)
        self.bins_spin = _spin(2, 10_000, 50)
        self.rate_edit = FloatEdit("除外する遅い変化の割合", 0.05)
        self.smoothing_check = QCheckBox("向きの判定に平滑化を使う")
        self.smoothing_check.setToolTip(
            "入力の雑音が多いときに使います。X-Y の値と指標には、平滑化しない値を使います"
        )
        self.smoothing_edit = FloatEdit("平滑化のカットオフ周波数", 10.0)
        form = _form_layout(self)
        form.addRow(hint)
        form.addRow("入力の範囲の区間数", self.bins_spin)
        form.addRow("除外する遅い変化の割合", self.rate_edit)
        form.addRow(self.smoothing_check)
        form.addRow("カットオフ [Hz]", self.smoothing_edit)
        self.smoothing_check.toggled.connect(self.smoothing_edit.setEnabled)
        self.smoothing_edit.setEnabled(False)
        self._watch(self.bins_spin, self.rate_edit, self.smoothing_check, self.smoothing_edit)

    def params(self) -> HysteresisParams:
        """入力された条件を返す。

        Returns:
            ヒステリシスの条件。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合。
        """
        smoothed = self.smoothing_check.isChecked()
        return HysteresisParams(
            n_bins=self.bins_spin.value(),
            min_rate_fraction=self.rate_edit.value(),
            smoothing_cutoff_hz=self.smoothing_edit.value() if smoothed else None,
        )

    def load(self, params: HysteresisParams) -> None:
        """条件を表示する。

        Args:
            params: ヒステリシスの条件。
        """
        self.bins_spin.setValue(params.n_bins)
        self.rate_edit.set_value(params.min_rate_fraction)
        self.smoothing_check.setChecked(params.smoothing_cutoff_hz is not None)
        if params.smoothing_cutoff_hz is not None:
            self.smoothing_edit.set_value(params.smoothing_cutoff_hz)


class CrossCorrelationPage(_ParamsPage):
    """相互相関の条件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        hint = QLabel(
            "連続的に変化する信号の遅れを求めます。等間隔の時刻が必要です"
            "（2 つのファイルの場合は「等間隔の時刻」に対応付けてください）。"
        )
        hint.setWordWrap(True)
        self.min_lag_edit = FloatEdit("遅れを探す範囲の下限", -_DEFAULT_MAX_LAG_S)
        self.max_lag_edit = FloatEdit("遅れを探す範囲の上限", _DEFAULT_MAX_LAG_S)
        self.detrend_combo = _combo(_DETREND_LABELS)
        self.derivative_check = QCheckBox("変化の速さの相関を取る")
        self.derivative_check.setToolTip("ステップ状の信号では、相関のピークが鋭くなります")
        form = _form_layout(self)
        form.addRow(hint)
        form.addRow("遅れの下限 [s]", self.min_lag_edit)
        form.addRow("遅れの上限 [s]", self.max_lag_edit)
        form.addRow("傾向の除去", self.detrend_combo)
        form.addRow(self.derivative_check)
        self._watch(self.min_lag_edit, self.max_lag_edit, self.detrend_combo, self.derivative_check)

    def params(self) -> CrossCorrelationParams:
        """入力された条件を返す。

        Returns:
            相互相関の条件。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 下限が上限以上の場合。
        """
        return CrossCorrelationParams(
            min_lag_s=self.min_lag_edit.value(),
            max_lag_s=self.max_lag_edit.value(),
            detrend=DetrendMode(str(self.detrend_combo.currentData())),
            use_derivative=self.derivative_check.isChecked(),
        )

    def load(self, params: CrossCorrelationParams) -> None:
        """条件を表示する。

        Args:
            params: 相互相関の条件。
        """
        self.min_lag_edit.set_value(params.min_lag_s)
        self.max_lag_edit.set_value(params.max_lag_s)
        _select(self.detrend_combo, params.detrend.value)
        self.derivative_check.setChecked(params.use_derivative)

    def set_default_lags(self, duration_s: float) -> None:
        """データの長さに合わせて、遅れを探す範囲の初期値を決める。

        Args:
            duration_s: データの長さ [s]。
        """
        limit = round_significant(min(_DEFAULT_MAX_LAG_S, _DEFAULT_LAG_FRACTION * duration_s), 3)
        self.min_lag_edit.set_value(-limit)
        self.max_lag_edit.set_value(limit)


class KalmanPage(_ParamsPage):
    """カルマンフィルタ系の条件（モデル・行列・初期状態・適応推定）。"""

    defaults_requested = Signal()
    """「データから初期値を作る」が押された（初期値の計算には入力データが要るため、パネルに任せる）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.model_combo = _combo({kind.value: label for kind, label in _MODEL_LABELS.items()})

        # 運動学モデル
        self.kinematic_order_spin = _spin(1, 2, 1)
        self.kinematic_q_edit = FloatEdit("プロセス雑音の強度 q", 1.0)
        self.kinematic_sigma_edit = FloatEdit("観測雑音の標準偏差 σ", 1.0)
        self.kinematic_q_slider = SliderField(self.kinematic_q_edit, 1e-3, 1e3)
        self.kinematic_sigma_slider = SliderField(self.kinematic_sigma_edit, 1e-3, 1e3)
        kinematic = QWidget()
        kinematic_form = _form_layout(kinematic)
        self.kinematic_q_slider.setToolTip("プロセス雑音の強度。大きいほど観測に素早く追従します")
        self.kinematic_sigma_slider.setToolTip("観測雑音の標準偏差。大きいほど滑らかになります")
        self.kinematic_order_spin.setToolTip("微分の最高階数（1: 等速度モデル、2: 等加速度モデル）")
        kinematic_form.addRow("プロセス雑音 q", self.kinematic_q_slider)
        kinematic_form.addRow("観測雑音 σ", self.kinematic_sigma_slider)
        kinematic_form.addRow("微分の階数", self.kinematic_order_spin)

        # 線形モデル（観測は入力チャンネル 1 つなので、H は 1×n、R は 1×1）
        self.linear_dim_spin = _spin(1, _MAX_LINEAR_STATES, 2)
        self.linear_f = MatrixEditor("F")
        self.linear_h = MatrixEditor("H")
        self.linear_q = MatrixEditor("Q")
        self.linear_r = MatrixEditor("R")
        linear = QWidget()
        linear_form = _form_layout(linear)
        linear_form.addRow("状態の次元 n", self.linear_dim_spin)
        linear_form.addRow("状態遷移 F（n×n）", self.linear_f)
        linear_form.addRow("観測 H（1×n）", self.linear_h)
        linear_form.addRow("プロセス雑音 Q（n×n）", self.linear_q)
        linear_form.addRow("観測雑音 R（1×1）", self.linear_r)

        # EKF の組み込みモデル（運動学モデル + 多項式の観測）
        self.polynomial_order_spin = _spin(1, 2, 1)
        self.polynomial_q_edit = FloatEdit("プロセス雑音の強度 q", 1.0)
        self.polynomial_sigma_edit = FloatEdit("観測雑音の標準偏差 σ", 1.0)
        self.polynomial_degree_spin = _spin(1, _MAX_POLYNOMIAL_DEGREE, 1)
        self.polynomial_coefficients = MatrixEditor("係数")
        self.polynomial_q_slider = SliderField(self.polynomial_q_edit, 1e-3, 1e3)
        self.polynomial_sigma_slider = SliderField(self.polynomial_sigma_edit, 1e-3, 1e3)
        polynomial = QWidget()
        polynomial_form = _form_layout(polynomial)
        polynomial_form.addRow("プロセス雑音 q", self.polynomial_q_slider)
        polynomial_form.addRow("観測雑音 σ", self.polynomial_sigma_slider)
        polynomial_form.addRow("x の微分の階数", self.polynomial_order_spin)
        polynomial_form.addRow("多項式の次数", self.polynomial_degree_spin)
        polynomial_form.addRow("係数 c0, c1, …（z = c0 + c1·x + …）", self.polynomial_coefficients)

        self.model_stack = _PageStack()
        self.model_stack.add_page(kinematic)
        self.model_stack.add_page(linear)
        self.model_stack.add_page(polynomial)
        self._model_pages = {
            KalmanModelKind.KINEMATIC: kinematic,
            KalmanModelKind.LINEAR: linear,
            KalmanModelKind.POLYNOMIAL_OBSERVATION: polynomial,
        }

        # 初期状態
        self.x0_editor = MatrixEditor("x0")
        self.p0_editor = MatrixEditor("P0")
        defaults_button = QPushButton("データから初期値を作る")
        defaults_button.clicked.connect(self._on_defaults_clicked)
        initial_group = QWidget()
        initial_form = _form_layout(initial_group)
        initial_form.setContentsMargins(0, 0, 0, 0)
        initial_form.addRow("x0（1×n）", self.x0_editor)
        initial_form.addRow("P0（n×n）", self.p0_editor)
        initial_form.addRow("", defaults_button)

        # 適応推定
        self.adaptive_combo = _combo(_ADAPTIVE_LABELS)
        self.window_spin = _spin(2, 1_000_000, 50)
        self.forgetting_edit = FloatEdit("忘却係数 b", 0.98)
        self.min_variance_edit = FloatEdit("R の固有値の下限", 1e-12)
        adaptive_group = QWidget()
        adaptive_form = _form_layout(adaptive_group)
        adaptive_form.setContentsMargins(0, 0, 0, 0)
        adaptive_form.addRow("方法", self.adaptive_combo)
        adaptive_form.addRow("窓長 N（窓付き）", self.window_spin)
        adaptive_form.addRow("忘却係数 b（Sage-Husa）", self.forgetting_edit)
        adaptive_form.addRow("R の固有値の下限", self.min_variance_edit)

        model_form = QFormLayout()
        model_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        model_form.addRow("モデル", self.model_combo)
        layout = QVBoxLayout(self)
        layout.addLayout(model_form)
        layout.addWidget(self.model_stack)
        # まず調整するのはモデルの条件なので、それ以外は畳んでおく。
        self.adaptive_section = CollapsibleSection("適応推定（R の逐次推定）", adaptive_group)
        layout.addWidget(CollapsibleSection("初期状態 x0・P0", initial_group))
        layout.addWidget(self.adaptive_section)

        self.model_combo.currentIndexChanged.connect(self._on_structure_changed)
        self.kinematic_order_spin.valueChanged.connect(self._on_structure_changed)
        self.linear_dim_spin.valueChanged.connect(self._on_structure_changed)
        self.polynomial_order_spin.valueChanged.connect(self._on_structure_changed)
        self.polynomial_degree_spin.valueChanged.connect(self._on_degree_changed)
        self.adaptive_combo.currentIndexChanged.connect(self._update_enabled)
        self._watch(
            self.model_combo,
            self.kinematic_order_spin,
            self.kinematic_q_edit,
            self.kinematic_sigma_edit,
            self.linear_dim_spin,
            self.kinematic_q_slider,
            self.kinematic_sigma_slider,
            self.linear_f,
            self.linear_h,
            self.linear_q,
            self.linear_r,
            self.polynomial_order_spin,
            self.polynomial_q_edit,
            self.polynomial_sigma_edit,
            self.polynomial_q_slider,
            self.polynomial_sigma_slider,
            self.polynomial_degree_spin,
            self.polynomial_coefficients,
            self.x0_editor,
            self.p0_editor,
            self.adaptive_combo,
            self.window_spin,
            self.forgetting_edit,
            self.min_variance_edit,
        )
        self._reset_matrices()
        self._update_enabled()
        self.model_stack.show_page(kinematic)

    # --- 値の取り出しと表示 --------------------------------------------------

    def model_kind(self) -> KalmanModelKind:
        """選んでいるモデルの種類を返す。

        Returns:
            モデルの種類。
        """
        return KalmanModelKind(str(self.model_combo.currentData()))

    def state_dim(self) -> int:
        """選んでいるモデルの状態の次元を返す。

        Returns:
            状態の次元 n。
        """
        kind = self.model_kind()
        if kind is KalmanModelKind.KINEMATIC:
            return self.kinematic_order_spin.value() + 1
        if kind is KalmanModelKind.LINEAR:
            return self.linear_dim_spin.value()
        return self.polynomial_order_spin.value() + 1

    def settings(self) -> KalmanSettings:
        """入力された条件を返す。

        Returns:
            カルマンフィルタの条件。

        Raises:
            ValueError: 数値として読めない欄・セルがある場合。
            pydantic.ValidationError: 条件が不正な場合（共分散が正定値でないなど）。
        """
        kind = self.model_kind()
        initial = KalmanInitialState(
            state=self.x0_editor.values()[0], covariance=self.p0_editor.values()
        )
        adaptive = self.adaptive_params()
        if kind is KalmanModelKind.KINEMATIC:
            return KalmanSettings(
                model_kind=kind,
                kinematic=KinematicSystemParams(
                    derivative_order=self.kinematic_order_spin.value(),
                    process_noise_density=self.kinematic_q_edit.value(),
                    measurement_noise_std=self.kinematic_sigma_edit.value(),
                ),
                initial=initial,
                adaptive=adaptive,
            )
        if kind is KalmanModelKind.LINEAR:
            return KalmanSettings(
                model_kind=kind,
                linear=LinearSystemParams(
                    transition=self.linear_f.values(),
                    observation=self.linear_h.values(),
                    process_noise=self.linear_q.values(),
                    measurement_noise=self.linear_r.values(),
                ),
                initial=initial,
                adaptive=adaptive,
            )
        return KalmanSettings(
            model_kind=kind,
            polynomial_observation=PolynomialObservationParams(
                derivative_order=self.polynomial_order_spin.value(),
                process_noise_density=self.polynomial_q_edit.value(),
                measurement_noise_std=self.polynomial_sigma_edit.value(),
                coefficients=self.polynomial_coefficients.values()[0],
            ),
            initial=initial,
            adaptive=adaptive,
        )

    def adaptive_params(self) -> AdaptiveNoiseParams | None:
        """入力された適応推定の条件を返す。

        Returns:
            適応推定の条件。「使わない」を選んでいれば None。

        Raises:
            ValueError: 数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合（忘却係数が 0〜1 の外など）。
        """
        method = str(self.adaptive_combo.currentData())
        if not method:
            return None
        return AdaptiveNoiseParams(
            method=AdaptiveNoiseMethod(method),
            window=self.window_spin.value(),
            forgetting_factor=self.forgetting_edit.value(),
            min_variance=self.min_variance_edit.value(),
        )

    def load(self, settings: KalmanSettings) -> None:
        """条件を表示する。

        Args:
            settings: カルマンフィルタの条件。
        """
        _select(self.model_combo, settings.model_kind.value)
        if settings.kinematic is not None:
            self.kinematic_order_spin.setValue(settings.kinematic.derivative_order)
            self.kinematic_q_edit.set_value(settings.kinematic.process_noise_density)
            self.kinematic_sigma_edit.set_value(settings.kinematic.measurement_noise_std)
        if settings.linear is not None:
            self.linear_dim_spin.setValue(len(settings.linear.transition))
            self.linear_f.set_values(settings.linear.transition)
            self.linear_h.set_values(settings.linear.observation)
            self.linear_q.set_values(settings.linear.process_noise)
            self.linear_r.set_values(settings.linear.measurement_noise)
        if settings.polynomial_observation is not None:
            polynomial = settings.polynomial_observation
            self.polynomial_order_spin.setValue(polynomial.derivative_order)
            self.polynomial_q_edit.set_value(polynomial.process_noise_density)
            self.polynomial_sigma_edit.set_value(polynomial.measurement_noise_std)
            self.polynomial_degree_spin.setValue(len(polynomial.coefficients) - 1)
            self.polynomial_coefficients.set_values([list(polynomial.coefficients)])
        self.x0_editor.set_values([list(settings.initial.state)])
        self.p0_editor.set_values([list(row) for row in settings.initial.covariance])
        adaptive = settings.adaptive
        _select(self.adaptive_combo, adaptive.method.value if adaptive is not None else "")
        if adaptive is not None:
            self.window_spin.setValue(adaptive.window)
            self.forgetting_edit.set_value(adaptive.forgetting_factor)
            self.min_variance_edit.set_value(adaptive.min_variance)
        self._update_enabled()

    def apply_preset(self, preset: str) -> None:
        """処理の一覧のボタンに合わせて、モデルと適応推定を選ぶ。

        Args:
            preset: PRESET_KF・PRESET_EKF・PRESET_AKF のいずれか（他の値では何もしない）。
        """
        if preset == PRESET_KF:
            _select(self.model_combo, KalmanModelKind.KINEMATIC.value)
            _select(self.adaptive_combo, "")
        elif preset == PRESET_EKF:
            _select(self.model_combo, KalmanModelKind.POLYNOMIAL_OBSERVATION.value)
            _select(self.adaptive_combo, "")
        elif preset == PRESET_AKF:
            _select(self.model_combo, KalmanModelKind.KINEMATIC.value)
            _select(self.adaptive_combo, AdaptiveNoiseMethod.INNOVATION_WINDOW.value)
            self.adaptive_section.toggle_button.setChecked(True)

    def description(self) -> str:
        """選んでいるモデルの説明を返す。

        Returns:
            説明。
        """
        kind = self.model_kind()
        if kind is KalmanModelKind.LINEAR:
            text = _LINEAR_DESCRIPTION
        elif kind is KalmanModelKind.POLYNOMIAL_OBSERVATION:
            text = _EKF_DESCRIPTION
        else:
            text = _KALMAN_DESCRIPTION
        return f"{text}{_ADAPTIVE_DESCRIPTION}" if self.adaptive_combo.currentData() else text

    def fit_sliders(self) -> None:
        """雑音のスライダーの範囲を、今の値を中心に上下へ広げる（データの大きさに合わせるため）。"""
        for slider, edit in (
            (self.kinematic_q_slider, self.kinematic_q_edit),
            (self.kinematic_sigma_slider, self.kinematic_sigma_edit),
            (self.polynomial_q_slider, self.polynomial_q_edit),
            (self.polynomial_sigma_slider, self.polynomial_sigma_edit),
        ):
            try:
                value = edit.value()
            except ValueError:
                continue
            if value > 0:
                slider.set_range(value / _NOISE_SLIDER_SPAN, value * _NOISE_SLIDER_SPAN)

    def short_name(self) -> str:
        """結果の名前に使う短い説明を返す。

        Returns:
            短い説明（"KF"、"EKF"、"AKF"）。
        """
        if str(self.adaptive_combo.currentData()):
            return "AKF"
        if self.model_kind() is KalmanModelKind.POLYNOMIAL_OBSERVATION:
            return "EKF"
        return "KF"

    def load_initial(self, initial: KalmanInitialState) -> None:
        """初期状態だけを表示する（「データから初期値を作る」の結果）。

        Args:
            initial: 初期状態。
        """
        self.x0_editor.set_values([list(initial.state)])
        self.p0_editor.set_values([list(row) for row in initial.covariance])

    def coefficients(self) -> list[float] | None:
        """多項式の観測モデルの係数を返す（他のモデル、または読めない場合は None）。

        Returns:
            係数、または None。
        """
        if self.model_kind() is not KalmanModelKind.POLYNOMIAL_OBSERVATION:
            return None
        try:
            return self.polynomial_coefficients.values()[0]
        except (ValueError, IndexError):
            return None

    # --- 内部処理 ------------------------------------------------------------

    def _reset_matrices(self) -> None:
        """行列の欄を、選んでいる次元に合わせた初期値にする。"""
        n = self.linear_dim_spin.value()
        self.linear_f.set_values([[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)])
        self.linear_h.set_values([[1.0 if c == 0 else 0.0 for c in range(n)]])
        self.linear_q.set_values([[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)])
        self.linear_r.set_values([[1.0]])
        self.polynomial_coefficients.set_values([[0.0, 1.0]])
        dim = self.state_dim()
        self.x0_editor.set_values([[0.0] * dim])
        self.p0_editor.set_values(
            [[1.0 if r == c else 0.0 for c in range(dim)] for r in range(dim)]
        )

    def _on_structure_changed(self) -> None:
        """モデル・次数・次元が変わったら、画面と行列の大きさを合わせる（既存の値は残す）。"""
        self.model_stack.show_page(self._model_pages[self.model_kind()])
        n = self.linear_dim_spin.value()
        self.linear_f.resize_matrix(n, n, diagonal=1.0)
        self.linear_h.resize_matrix(1, n)
        self.linear_q.resize_matrix(n, n, diagonal=1.0)
        dim = self.state_dim()
        self.x0_editor.resize_matrix(1, dim)
        self.p0_editor.resize_matrix(dim, dim, diagonal=1.0)

    def _on_degree_changed(self) -> None:
        """多項式の次数が変わったら、係数の欄の数を合わせる（既存の係数は残す）。"""
        self.polynomial_coefficients.resize_matrix(1, self.polynomial_degree_spin.value() + 1)

    def _update_enabled(self) -> None:
        """適応推定の方法に合わせて、使う欄だけを入力できるようにする。"""
        method = str(self.adaptive_combo.currentData())
        self.window_spin.setEnabled(method == AdaptiveNoiseMethod.INNOVATION_WINDOW.value)
        self.forgetting_edit.setEnabled(method == AdaptiveNoiseMethod.SAGE_HUSA.value)
        self.min_variance_edit.setEnabled(bool(method))

    def _on_defaults_clicked(self) -> None:
        """「データから初期値を作る」が押されたことを知らせる。"""
        self.defaults_requested.emit()


# ---------------------------------------------------------------------------
# 解析パネル
# ---------------------------------------------------------------------------


class _MethodList(QWidget):
    """処理の一覧（押すと、その処理を始めるボタンが並ぶ）。

    Attributes:
        buttons: ボタンの名前 → ボタン。
    """

    chosen = Signal(str, str)
    """処理が選ばれた（引数: 解析の種類の値, 追加の設定）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self.buttons: dict[str, QPushButton] = {}
        self._two_channel: list[QPushButton] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for title, description, methods in METHOD_GROUPS:
            heading = QLabel(title)
            heading.setStyleSheet("font-weight: 600; margin-top: 8px;")
            note = QLabel(description)
            note.setStyleSheet(f"color: {MUTED};")
            note.setWordWrap(True)
            layout.addWidget(heading)
            layout.addWidget(note)
            grid = QGridLayout()
            grid.setSpacing(4)
            for index, (label, kind, preset, description) in enumerate(methods):
                button = QPushButton(label)
                button.setToolTip(description)
                button.clicked.connect(partial(self._emit_chosen, kind.value, preset))
                grid.addWidget(button, index // 2, index % 2)
                self.buttons[label] = button
                if kind in TWO_CHANNEL_ANALYSIS_KINDS:
                    self._two_channel.append(button)
            layout.addLayout(grid)
        layout.addStretch(1)

    def set_two_channel_enabled(self, *, enabled: bool) -> None:
        """2 チャンネルの解析のボタンを使えるかどうかを切り替える。

        Args:
            enabled: チャンネルが 2 つ以上あれば True。
        """
        for button in self._two_channel:
            button.setEnabled(enabled)

    def _emit_chosen(self, kind: str, preset: str, checked: bool = False) -> None:
        """選ばれた処理を知らせる。

        Args:
            kind: 解析の種類の値。
            preset: 追加の設定。
            checked: ボタンのシグナルの引数（使わない）。
        """
        self.chosen.emit(kind, preset)


class AnalysisPanel(QWidget):
    """処理を選び、条件を調整するパネル。

    Attributes:
        input_label: 処理の対象のチャンネルの表示。
        method_list: 処理の一覧。
        title_label: 調整中の処理の名前。
        description_label: 調整中の処理の説明（何をするか、動かすとどうなるか）。
        name_edit: 解析の名前の欄。
        pages: 解析の種類 → 条件のページ。
        back_button: 処理の一覧に戻るボタン。
        duplicate_button: 同じ条件の処理を増やすボタン（条件を変えて見比べるため）。
        delete_button: 調整中の処理を削除するボタン。
        cancel_button: 計算を中止するボタン。
    """

    run_requested = Signal(object)
    """実行が求められた（引数: AnalysisSpec）。"""
    cancel_requested = Signal()
    """実行中の計算の中止が求められた。"""
    delete_requested = Signal(object)
    """調整中の処理の削除が求められた（引数: 解析の ID）。"""
    methods_shown = Signal()
    """処理の一覧に戻った（調整中の処理の時刻の印などを片付けるために使う）。"""
    range_selection_changed = Signal(bool)
    """解析する範囲の選択を表示するかどうかが変わった（引数: 表示するなら True）。"""
    fit_range_requested = Signal()
    """解析する範囲を、縦並べの表示中の範囲に合わせることが求められた。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """部品を組み立てる（入力はまだ選ばれていない状態）。

        Args:
            parent: 親ウィジェット。
        """
        super().__init__(parent)
        self._input: ChannelData | None = None
        self._analysis_id: UUID = uuid4()
        # 作成日時は処理の並び順に使うので、条件を変えて保存し直しても変えない。
        self._created_at = datetime.now(UTC)
        self._kind = AnalysisKind.BUTTERWORTH
        self._name_edited = False
        self._loading = False
        self._editing = False

        self.input_label = QLabel()
        self.input_label.setWordWrap(True)
        self.input_label.setStyleSheet("font-weight: 600;")
        self.method_list = _MethodList()
        self.method_list.chosen.connect(self._on_method_chosen)

        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_edited)

        self.butterworth_page = ButterworthPage()
        self.moving_average_page = MovingAveragePage()
        self.savitzky_golay_page = SavitzkyGolayPage()
        self.derivative_page = DerivativePage()
        self.integral_page = IntegralPage()
        self.kalman_page = KalmanPage()
        self.step_response_page = StepResponsePage()
        self.hysteresis_page = HysteresisPage()
        self.cross_correlation_page = CrossCorrelationPage()
        self.alignment_page = AlignmentPage()
        self.pages: dict[AnalysisKind, _ParamsPage] = {
            AnalysisKind.BUTTERWORTH: self.butterworth_page,
            AnalysisKind.MOVING_AVERAGE: self.moving_average_page,
            AnalysisKind.SAVITZKY_GOLAY: self.savitzky_golay_page,
            AnalysisKind.DERIVATIVE: self.derivative_page,
            AnalysisKind.INTEGRAL: self.integral_page,
            AnalysisKind.KALMAN: self.kalman_page,
            AnalysisKind.STEP_RESPONSE: self.step_response_page,
            AnalysisKind.HYSTERESIS: self.hysteresis_page,
            AnalysisKind.CROSS_CORRELATION: self.cross_correlation_page,
        }
        # 解析の種類 → そのページから条件を取り出す関数（AnalysisSpec の同名の項目に入れる）。
        self._params_getters: dict[AnalysisKind, Callable[[], object]] = {
            AnalysisKind.BUTTERWORTH: self.butterworth_page.params,
            AnalysisKind.MOVING_AVERAGE: self.moving_average_page.params,
            AnalysisKind.SAVITZKY_GOLAY: self.savitzky_golay_page.params,
            AnalysisKind.DERIVATIVE: self.derivative_page.params,
            AnalysisKind.INTEGRAL: self.integral_page.params,
            AnalysisKind.KALMAN: self.kalman_page.settings,
            AnalysisKind.STEP_RESPONSE: self.step_response_page.params,
            AnalysisKind.HYSTERESIS: self.hysteresis_page.params,
            AnalysisKind.CROSS_CORRELATION: self.cross_correlation_page.params,
        }
        self.page_stack = _PageStack()
        for page in self.pages.values():
            self.page_stack.add_page(page)
            page.changed.connect(self._on_changed)
        self.kalman_page.defaults_requested.connect(self._on_kalman_defaults)
        self.alignment_page.changed.connect(self._on_changed)
        self.alignment_page.range_toggled.connect(self._emit_range_selection)
        self.alignment_page.fit_range_requested.connect(self.fit_range_requested)

        name_box = QWidget()
        name_layout = QVBoxLayout(name_box)
        name_layout.setContentsMargins(9, 0, 9, 4)
        name_layout.addWidget(self.name_edit)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.alignment_page)
        scroll_layout.addWidget(self.page_stack)
        scroll_layout.addWidget(CollapsibleSection("名前を変える", name_box))
        scroll_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(scroll_content)

        self.back_button = QToolButton()
        self.back_button.setText("← 処理の一覧")
        self.back_button.setAutoRaise(True)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 600;")
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f"color: {MUTED};")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.cancel_button = QPushButton("中止")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.busy_row = QWidget()
        busy_layout = QHBoxLayout(self.busy_row)
        busy_layout.setContentsMargins(0, 0, 0, 0)
        busy_layout.addWidget(self.progress_bar, stretch=1)
        busy_layout.addWidget(self.cancel_button)
        self.busy_row.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: 600;")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.duplicate_button = QPushButton("複製して比べる")
        self.duplicate_button.setToolTip(
            "同じ条件の処理を増やします。条件を変えて結果を見比べられます"
        )
        self.duplicate_button.clicked.connect(self._on_duplicate_clicked)
        self.delete_button = QPushButton("削除")
        self.delete_button.setToolTip("この処理と結果を削除します（元のデータは消えません）")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        action_row = QHBoxLayout()
        action_row.addWidget(self.duplicate_button)
        action_row.addWidget(self.delete_button)
        title_row = QHBoxLayout()
        title_row.addWidget(self.back_button)
        title_row.addWidget(self.title_label, stretch=1)

        self._editor_page = QWidget()
        editor_layout = QVBoxLayout(self._editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addLayout(title_row)
        editor_layout.addWidget(self.description_label)
        editor_layout.addWidget(scroll, stretch=1)
        editor_layout.addWidget(self.busy_row)
        editor_layout.addWidget(self.status_label)
        editor_layout.addLayout(action_row)
        self._method_page = QWidget()
        method_layout = QVBoxLayout(self._method_page)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.addWidget(self.method_list)
        self._screens = _PageStack()
        self._screens.add_page(self._method_page)
        self._screens.add_page(self._editor_page)
        self._screens.show_page(self._method_page)
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_label)
        layout.addWidget(self._screens, stretch=1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(AUTO_RUN_DELAY_MS)
        self._debounce.timeout.connect(self.request_run)
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(BUSY_DISPLAY_DELAY_MS)
        self._busy_timer.timeout.connect(self._show_busy)
        self._show_page(self._kind)
        self._update_input_label()
        self._update_buttons()

    # --- 状態の参照 ----------------------------------------------------------

    @property
    def input_data(self) -> ChannelData | None:
        """入力チャンネルのデータ（選ばれていなければ None）。"""
        return self._input

    @property
    def analysis_id(self) -> UUID:
        """調整中の解析の ID。"""
        return self._analysis_id

    @property
    def is_editing(self) -> bool:
        """条件の調整の画面を出していれば True。"""
        return self._editing

    def kind(self) -> AnalysisKind:
        """調整中の解析の種類を返す。

        Returns:
            解析の種類。
        """
        return self._kind

    def range_selection_active(self) -> bool:
        """解析する範囲の選択を表示する状態かを返す。

        Returns:
            2 チャンネルの解析を調整中で、範囲を指定していれば True。
        """
        return (
            self._input is not None
            and self._editing
            and self.kind() in TWO_CHANNEL_ANALYSIS_KINDS
            and self.alignment_page.range_check.isChecked()
        )

    # --- 公開操作 ------------------------------------------------------------

    def set_input(self, data: ChannelData | None) -> None:
        """処理の対象のチャンネルを設定し、処理の一覧を出す。

        フィルタなどの条件はそのまま残し（別のチャンネルに同じ条件をかけやすいように）、
        データの大きさに依存するカルマンフィルタの条件（すべてのモデル。適応推定の条件は残す）と
        カットオフ周波数の初期値だけを入れ直す。

        Args:
            data: 入力チャンネルのデータ。None なら入力なし。
        """
        self._flush_pending()
        self._input = data
        self._analysis_id = uuid4()
        self._name_edited = False
        if data is not None:
            self._loading = True
            try:
                sampling_rate = self._fit_to_data(data)
                if sampling_rate is not None:
                    self.butterworth_page.set_default_cutoff(sampling_rate)
                    self.alignment_page.rate_edit.set_value(round_significant(sampling_rate, 4))
                if data.time_s.size >= 2:
                    duration = float(data.time_s[-1] - data.time_s[0])
                    if math.isfinite(duration) and duration > 0:
                        self.cross_correlation_page.set_default_lags(duration)
                self._load_kalman_defaults(data)
                self.kalman_page.fit_sliders()
                self.alignment_page.select_default_output(data.channel.id, data.source.id)
            finally:
                self._loading = False
        self._update_default_name()
        self._update_input_label()
        self._update_buttons()
        self.show_methods()

    def show_methods(self) -> None:
        """処理の一覧を出す（反映していない条件の変更は先に計算する）。"""
        self._flush_pending()
        self._editing = False
        self.status_label.clear()
        self._screens.show_page(self._method_page)
        self._emit_range_selection()
        self.methods_shown.emit()

    def start_method(self, kind: AnalysisKind, preset: str = PRESET_NONE) -> None:
        """新しい処理を始める（条件の調整の画面を出して、すぐに計算を求める）。

        Args:
            kind: 解析の種類。
            preset: 追加の設定（KF・EKF・AKF の区別）。
        """
        if self._input is None:
            return
        self._flush_pending()
        self._loading = True
        try:
            self._analysis_id = uuid4()
            self._created_at = datetime.now(UTC)
            self._name_edited = False
            self._kind = kind
            self.kalman_page.apply_preset(preset)
            # ステップ応答は入力の変化を 1 回含む範囲が必要なので、範囲の指定を始めておく。
            self.alignment_page.range_check.setChecked(kind is AnalysisKind.STEP_RESPONSE)
        finally:
            self._loading = False
        self._show_page(kind)
        self._open_editor()
        self._update_default_name()
        self._emit_range_selection()
        self.request_run()

    def load_spec(self, spec: AnalysisSpec, data: ChannelData) -> None:
        """保存した処理を、調整のために読み込む（ID と名前も引き継ぐ）。

        Args:
            spec: 保存した処理。
            data: 入力チャンネルのデータ。
        """
        self._flush_pending()
        self._loading = True
        try:
            self._input = data
            self._analysis_id = spec.id
            self._created_at = spec.created_at
            # チャンネル名で始まる名前は自動で付けたものとみなし、条件に合わせて更新する。
            self._name_edited = not spec.name.startswith(f"{data.channel.name} ")
            self.name_edit.setText(spec.name)
            self._kind = spec.kind
            self._show_page(spec.kind)
            self._fit_to_data(data)
            if spec.butterworth is not None:
                self.butterworth_page.load(spec.butterworth)
            if spec.moving_average is not None:
                self.moving_average_page.load(spec.moving_average)
            if spec.savitzky_golay is not None:
                self.savitzky_golay_page.load(spec.savitzky_golay)
            if spec.derivative is not None:
                self.derivative_page.load(spec.derivative)
            if spec.integral is not None:
                self.integral_page.load(spec.integral)
            if spec.kalman is not None:
                self.kalman_page.load(spec.kalman)
                self.kalman_page.fit_sliders()
            if spec.step_response is not None:
                self.step_response_page.load(spec.step_response)
            if spec.hysteresis is not None:
                self.hysteresis_page.load(spec.hysteresis)
            if spec.cross_correlation is not None:
                self.cross_correlation_page.load(spec.cross_correlation)
            if spec.alignment is not None:
                self.alignment_page.load(spec.alignment, spec.output_channel_id)
        finally:
            self._loading = False
        self._update_input_label()
        self._update_buttons()
        self._open_editor()
        self.status_label.clear()
        self._emit_range_selection()

    def build_spec(self) -> AnalysisSpec:
        """パネルの内容から解析の条件を作る。

        Returns:
            解析の条件。

        Raises:
            ValueError: 入力が選ばれていない、または数値として読めない欄がある場合。
            pydantic.ValidationError: 条件が不正な場合。
        """
        if self._input is None:
            raise ValueError("左の一覧でチャンネルを選んでください")
        kind = self.kind()
        params = self._params_getters[kind]()
        name = self.name_edit.text().strip() or self._default_name()
        fields: dict[str, object] = {
            "id": self._analysis_id,
            "created_at": self._created_at,
            "name": name,
            "input_channel_id": self._input.channel.id,
            "kind": kind,
            kind.value: params,
        }
        if kind in TWO_CHANNEL_ANALYSIS_KINDS:
            output_channel_id = self.alignment_page.output_channel_id()
            if output_channel_id is None:
                raise ValueError("出力チャンネルを選んでください")
            fields["output_channel_id"] = output_channel_id
            fields["alignment"] = self.alignment_page.params()
        return AnalysisSpec.model_validate(fields)

    def set_channel_choices(self, choices: list[tuple[UUID, str, UUID]]) -> None:
        """2 チャンネルの解析の、出力チャンネルの選択肢を入れ替える。

        Args:
            choices: (チャンネル ID, 表示名, Source ID) のリスト（一覧の順）。
        """
        self.alignment_page.set_channel_choices(choices)
        self.method_list.set_two_channel_enabled(enabled=len(choices) >= 2)
        if self._input is not None:
            self.alignment_page.select_default_output(self._input.channel.id, self._input.source.id)

    def range_values(self) -> tuple[float, float] | None:
        """範囲の欄の値を返す。

        Returns:
            (開始, 終了) [s]。読めない、または開始が終了以降なら None。
        """
        return self.alignment_page.range_values()

    def set_range(self, start_s: float, end_s: float) -> None:
        """解析する範囲を更新する（条件の変更として扱い、再計算の対象にする）。

        Args:
            start_s: 開始 [s]。
            end_s: 終了 [s]。
        """
        self.alignment_page.set_range(start_s, end_s)
        self._on_changed()

    def request_run(self) -> None:
        """パネルの内容で実行を求める（条件が不正なら理由を表示して求めない）。"""
        self._debounce.stop()
        spec = self._spec_or_error()
        if spec is not None:
            self.run_requested.emit(spec)

    def set_running(self, *, running: bool) -> None:
        """計算中の表示（進捗と中止ボタン）を切り替える。

        すぐ終わる計算で表示がちらつかないよう、少し長くかかるときだけ表示する。

        Args:
            running: 計算中なら True。
        """
        self.progress_bar.setValue(0)
        self.cancel_button.setEnabled(running)
        if running:
            self._busy_timer.start()
        else:
            self._busy_timer.stop()
            self.busy_row.setVisible(False)

    def set_progress(self, fraction: float) -> None:
        """計算の進捗を表示する。

        Args:
            fraction: 進捗（0〜1）。
        """
        self.progress_bar.setValue(round(fraction * 100))

    def show_summary(self, lines: list[str]) -> None:
        """計算結果の要約を表示する。

        Args:
            lines: 要約の行。
        """
        self.status_label.setStyleSheet("font-weight: 600;")
        self.status_label.setText("\n".join(line for line in lines if line))

    def show_error(self, text: str) -> None:
        """エラーを赤字で表示する。

        Args:
            text: エラーの説明。
        """
        self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
        self.status_label.setText(text)

    def show_status(self, text: str) -> None:
        """状態の説明を表示する（中止した など）。

        Args:
            text: 説明。
        """
        self.status_label.setStyleSheet("font-weight: 600;")
        self.status_label.setText(text)

    # --- 内部処理 ------------------------------------------------------------

    def _spec_or_error(self) -> AnalysisSpec | None:
        """解析の条件を作る。不正ならエラーを表示して None を返す。

        Returns:
            解析の条件、または None。
        """
        try:
            return self.build_spec()
        except ValidationError as exc:
            details = [str(detail["msg"]) for detail in exc.errors()]
            self.show_error("条件が不正です: " + " / ".join(details))
        except ValueError as exc:
            self.show_error(str(exc))
        return None

    def _flush_pending(self) -> None:
        """待っている条件の変更があれば、画面を切り替える前に計算を求める。

        切り替えた後で待ち時間が終わると、別の入力の条件として実行してしまうため。
        """
        if self._debounce.isActive():
            self.request_run()

    def _fit_to_data(self, data: ChannelData) -> float | None:
        """スライダーの範囲を、データの長さとサンプリング周波数に合わせる。

        Args:
            data: 入力チャンネルのデータ。

        Returns:
            サンプリング周波数 [Hz]。時刻の逆行や重複で求められなければ None。
        """
        self.moving_average_page.set_data_size(int(data.time_s.size))
        self.savitzky_golay_page.set_data_size(int(data.time_s.size))
        if data.time_s.size < 2:
            return None
        sampling_rate = estimate_sampling_rate(data.time_s)
        if not math.isfinite(sampling_rate) or sampling_rate <= 0:
            return None
        self.butterworth_page.set_sampling_rate(sampling_rate)
        return float(sampling_rate)

    def _open_editor(self) -> None:
        """条件の調整の画面を出す。"""
        self._editing = True
        self.title_label.setText(KIND_LABELS[self._kind])
        self._update_description()
        self._screens.show_page(self._editor_page)

    def _update_description(self) -> None:
        """調整中の処理の説明を表示する。"""
        text = self.pages[self._kind].description() or _METHOD_DESCRIPTIONS.get(self._kind, "")
        self.description_label.setText(text)

    def _load_kalman_defaults(self, data: ChannelData) -> None:
        """すべてのカルマンフィルタのモデルの欄に、データから作った初期値を入れる。

        選んでいないモデルの欄にも入れるのは、モデルを切り替えたときにも、データに合った雑音の
        強さから調整を始められるようにするため。最後に選んでいるモデルの初期値を入れて選択を
        元に戻す。適応推定の条件はデータの大きさによらないので、利用者が選んだものを残す
        （読めない値なら使わない設定にする）。

        Args:
            data: 入力チャンネルのデータ。
        """
        try:
            adaptive = self.kalman_page.adaptive_params()
        except (ValueError, ValidationError):
            adaptive = None
        current = self.kalman_page.model_kind()
        others = [kind for kind in KalmanModelKind if kind is not current]
        for kind in [*others, current]:
            defaults = default_kalman_settings(kind, data)
            self.kalman_page.load(defaults.model_copy(update={"adaptive": adaptive}))

    def _on_changed(self) -> None:
        """条件が変わったら、名前の既定値を更新し、再計算の待ち時間を始める。"""
        if self._loading:
            return
        self._update_default_name()
        self._update_description()
        if self._editing and self._input is not None:
            self._debounce.start()

    def _on_method_chosen(self, kind: str, preset: str) -> None:
        """処理の一覧で選ばれた処理を始める。

        Args:
            kind: 解析の種類の値。
            preset: 追加の設定。
        """
        self.start_method(AnalysisKind(kind), preset)

    def _show_busy(self) -> None:
        """計算が長くかかっているので、進捗と中止ボタンを出す。"""
        self.busy_row.setVisible(True)

    def _show_page(self, kind: AnalysisKind) -> None:
        """解析の種類の条件のページと、2 チャンネルの解析の共通の条件の表示を切り替える。

        Args:
            kind: 解析の種類。
        """
        self.page_stack.show_page(self.pages[kind])
        self.alignment_page.setVisible(kind in TWO_CHANNEL_ANALYSIS_KINDS)

    def _emit_range_selection(self, *args: object) -> None:
        """解析する範囲の選択を表示するかどうかを知らせる。

        準備の途中の状態を知らせないよう、処理を始める・読み込む間は黙っている（終わりに知らせる）。

        Args:
            *args: シグナルの引数（使わない）。
        """
        if not self._loading:
            self.range_selection_changed.emit(self.range_selection_active())

    def _on_back_clicked(self) -> None:
        """処理の一覧に戻る。"""
        self.show_methods()

    def _on_cancel_clicked(self) -> None:
        """実行中の計算の中止を求める。"""
        self.cancel_requested.emit()

    def _on_duplicate_clicked(self) -> None:
        """同じ条件の処理を増やして、そちらの調整に移る（元の処理は残る）。"""
        self._flush_pending()
        self._analysis_id = uuid4()
        self._created_at = datetime.now(UTC)
        self.request_run()

    def _on_delete_clicked(self) -> None:
        """調整中の処理の削除を求める。"""
        self.delete_requested.emit(self._analysis_id)

    def _on_name_edited(self, text: str) -> None:
        """名前が手で編集されたら、以後は名前を自動で変えず、変更を保存する。

        Args:
            text: 編集後の名前。
        """
        self._name_edited = bool(text.strip())
        self._on_changed()

    def _on_kalman_defaults(self) -> None:
        """入力データから、カルマンフィルタの初期状態の初期値を作って表示する。"""
        if self._input is None:
            self.show_error("左の一覧でチャンネルを選んでください")
            return
        coefficients = self.kalman_page.coefficients()
        initial = default_initial_state(
            self.kalman_page.state_dim(),
            self._input,
            sigma=estimate_noise_std(self._input.values),
            omega=default_omega(self._input),
            coefficients=coefficients,
        )
        self.kalman_page.load_initial(initial)
        self._on_changed()

    def _default_name(self) -> str:
        """名前の既定値（"チャンネル名 条件の短い説明"）を返す。

        Returns:
            名前の既定値。
        """
        channel = self._input.channel.name if self._input is not None else "入力"
        short = self.pages[self.kind()].short_name() or _KIND_SHORT_NAMES[self.kind()]
        return f"{channel} {short}"

    def _update_default_name(self) -> None:
        """名前が手で編集されていなければ、既定値にする。"""
        if not self._name_edited:
            self.name_edit.setText(self._default_name())

    def _update_input_label(self) -> None:
        """入力チャンネルの表示を更新する。"""
        if self._input is None:
            self.input_label.setText("左の一覧でチャンネルを選ぶと、処理を追加できます")
        else:
            unit = f" [{self._input.channel.unit}]" if self._input.channel.unit else ""
            self.input_label.setText(f"対象: {self._input.channel.name}{unit}")

    def _update_buttons(self) -> None:
        """入力の有無で、処理の一覧を使えるかどうかを切り替える。"""
        self.method_list.setEnabled(self._input is not None)
