"""モデルのフィールド検証に使う関数・定数と、NumPy 配列の型の別名。

``model.py`` は pydantic モデルの定義だけに限定する方針のため、
フィールドの検証処理や既定値の生成処理はこのモジュールに置き、モデル側からは呼び出すだけにする。
このモジュールは Qt にも他のサブパッケージにも依存しない（model.py・storage・analysis から使う）。
``typing`` モジュールは使わない方針（docs/DECISIONS.md「コーディング規約」）。
"""

import codecs
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

# ---------------------------------------------------------------------------
# 型の別名・定数
# ---------------------------------------------------------------------------

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
"""float64 の NumPy 配列（形は問わない）。"""

BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
"""bool の NumPy 配列（形は問わない）。"""

IntArray = np.ndarray[tuple[int, ...], np.dtype[np.intp]]
"""整数（配列の番号）の NumPy 配列（形は問わない）。"""

NON_BLANK_PATTERN = r"\S"
"""空白以外の文字を 1 つ以上含むことを表す正規表現（pydantic の ``Field(pattern=...)`` 用）。

pydantic の pattern は部分一致で判定されるため、空文字・空白のみの文字列だけが拒否される。
前後の空白は除去せず、入力をそのまま保持する（隠して修正しない方針）。
"""

SHA256_PATTERN = r"^[0-9a-f]{64}$"
"""SHA-256 ダイジェストの 16 進小文字表現（64 文字）。"""

VIEW_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
"""表示設定の名前（views/<名前>.json のファイル名に使うため、英数字・"_"・"-" に限る）。"""

COLOR_PATTERN = r"^#[0-9A-Fa-f]{6}$"
"""色の指定（"#rrggbb" の 16 進表記）。"""

# 行列の対称性・正定値性の判定に使う許容誤差。保存した JSON の丸め誤差程度は許す。
_SYMMETRY_RTOL = 1e-9
_SYMMETRY_ATOL = 1e-12


# ---------------------------------------------------------------------------
# 一般的な検証
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """現在時刻を UTC のタイムゾーン付き datetime で返す。

    タイムゾーンなし datetime は OS や保存先によって解釈がぶれるため、
    メタデータの作成日時は常に UTC で記録する。モデルの ``default_factory`` に使う。

    Returns:
        タイムゾーン情報（UTC）付きの現在時刻。
    """
    return datetime.now(UTC)


def require_non_blank(value: str) -> str:
    """空文字・空白のみの文字列を拒否する。

    名前やラベルが空だとツリー表示や凡例で区別できなくなるため、明示的にエラーにする。
    前後の空白は勝手に除去せず、入力をそのまま保持する（隠して修正しない方針）。

    Args:
        value: 検証する文字列。

    Returns:
        入力と同じ文字列。

    Raises:
        ValueError: 空文字または空白のみの場合。
    """
    if not value.strip():
        raise ValueError("空文字や空白のみの文字列は指定できません")
    return value


def reject_duplicates(values: list[str]) -> list[str]:
    """リスト内の重複要素を拒否する。

    重複を黙って除去すると入力との対応が分からなくなるため、エラーとして通知する。

    Args:
        values: 検証する文字列のリスト。

    Returns:
        入力と同じリスト（順序も保持する）。

    Raises:
        ValueError: 同じ要素が 2 回以上含まれる場合。エラーメッセージに重複要素を列挙する。
    """
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"重複した要素があります: {duplicates}")
    return values


def validate_tags(tags: list[str]) -> list[str]:
    """タグ一覧を検証する（各タグが空白のみでなく、重複がないこと）。

    Args:
        tags: 検証するタグ一覧。

    Returns:
        入力と同じタグ一覧。

    Raises:
        ValueError: 空白のみのタグ、または重複したタグがある場合。
    """
    for tag in tags:
        require_non_blank(tag)
    return reject_duplicates(tags)


def validate_relative_posix_path(value: str) -> str:
    """スラッシュ区切りの正規化済み相対パス文字列を検証する。

    プロジェクトフォルダを移動したり、Windows と Ubuntu の間で受け渡したりしても
    パスが壊れないよう、メタデータには POSIX 形式の相対パスだけを保存する。
    また外部ファイル取り込み時のパストラバーサルを防ぐため、".." を禁止する。

    許可しないもの:
        - 空文字
        - バックスラッシュ（Windows 形式の区切り。保存前に "/" へ変換しておくこと）
        - 絶対パス（"/a"、"C:/a"、"C:a" などドライブ指定を含む）
        - 空要素・"."・".." を含むパス（"a//b"、"./a"、"a/../b"、末尾 "/" など）

    Args:
        value: 検証する相対パス（例: ``"datasets/trial_001/imu.parquet"``）。

    Returns:
        入力と同じ文字列。

    Raises:
        ValueError: 上記のいずれかに該当する場合。
    """
    if not value:
        raise ValueError("相対パスが空です")
    if "\\" in value:
        raise ValueError(f"相対パスは '/' 区切りで指定してください: {value!r}")
    # PureWindowsPath の anchor で "C:" のようなドライブ指定も検出する。
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor:
        raise ValueError(f"絶対パスは指定できません: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"空要素・'.'・'..' を含むパスは指定できません: {value!r}")
    return value


def check_encoding(value: str) -> str:
    """文字コード名が Python で使えるコーデック名であることを確認する。

    取り込みの途中ではなく、条件を指定した時点で誤りに気付けるようにするため。

    Args:
        value: 文字コード名（例: "utf-8"、"cp932"）。

    Returns:
        入力と同じ文字列。

    Raises:
        ValueError: 未知の文字コード名の場合。
    """
    try:
        codecs.lookup(value)
    except LookupError:
        raise ValueError(f"文字コード {value!r} は使えません（例: utf-8、cp932）") from None
    return value


def check_import_columns(
    *,
    time_column: str | None,
    sample_interval_s: float | None,
    channel_columns: list[str],
    reserved_column: str,
) -> None:
    """取り込み指定の時刻列・サンプル間隔・チャンネル列の組み合わせを確認する。

    時刻の決め方（時刻列か、サンプル番号 × 間隔か）が必ずどちらか一方に定まるようにする。

    Args:
        time_column: 時刻列の名前。時刻列がない場合は None。
        sample_interval_s: 時刻列がない場合のサンプル間隔 [s]。
        channel_columns: 取り込むチャンネルの列名。
        reserved_column: 正規化データの時刻列として予約されている名前。

    Raises:
        ValueError: 時刻の決め方が定まらない、時刻列をチャンネルにも指定した、
            チャンネル列が重複している、または予約名をチャンネルに指定した場合。
    """
    if time_column is None and sample_interval_s is None:
        raise ValueError("時刻列を選ぶか、時刻列がない場合はサンプル間隔を指定してください")
    if time_column is not None and sample_interval_s is not None:
        raise ValueError(
            "時刻列を使う場合はサンプル間隔を指定しないでください（どちらで時刻を決めたか曖昧になるため）"
        )
    if time_column is not None and time_column in channel_columns:
        raise ValueError(f"時刻列 {time_column!r} はチャンネルとして取り込めません")
    reject_duplicates(channel_columns)
    if reserved_column in channel_columns:
        raise ValueError(
            f"列名 {reserved_column!r} は正規化データの時刻列として予約されているため、"
            "チャンネルとして取り込めません"
        )


def check_series_arrays(time_s: FloatArray, values: FloatArray) -> None:
    """時刻と値の配列が float64 の 1 次元配列で、同じ長さであることを確認する。

    Args:
        time_s: 時刻 [s] の配列。
        values: 値の配列。

    Raises:
        ValueError: dtype・次元・長さが条件を満たさない場合。
    """
    for name, array in (("time_s", time_s), ("values", values)):
        if array.dtype != np.float64 or array.ndim != 1:
            raise ValueError(
                f"{name} は float64 の 1 次元配列にしてください"
                f"（実際: {array.dtype}、{array.ndim} 次元）"
            )
    if time_s.shape != values.shape:
        raise ValueError(f"時刻の長さ {time_s.shape[0]} と値の長さ {values.shape[0]} が違います")


def check_selected_settings(*, selected: str, provided: list[str]) -> None:
    """種類を選んで条件を指定するモデルで、選んだ種類の条件だけが指定されていることを確認する。

    使わない条件を受け付けると、保存された内容から実際に使った条件を読み違えるため拒否する。

    Args:
        selected: 選んだ種類（条件の項目名と同じ文字列）。
        provided: 指定されていた（None でない）条件の項目名。

    Raises:
        ValueError: 選んだ種類の条件がない、または他の種類の条件も指定されている場合。
    """
    if provided != [selected]:
        raise ValueError(
            f"種類 {selected!r} の条件だけを指定してください（指定されていた条件: {provided}）"
        )


def check_plot_series_arrays(
    time_s: FloatArray,
    values: FloatArray,
    band_lower: FloatArray | None,
    band_upper: FloatArray | None,
    marker_times_s: FloatArray | None,
) -> None:
    """表示用の系列の配列（時刻・値・帯・印）の dtype と形が整合していることを確認する。

    Args:
        time_s: 時刻 [s]。
        values: 値。
        band_lower: 帯の下端（なければ None）。
        band_upper: 帯の上端（なければ None）。
        marker_times_s: 印を付ける時刻（なければ None）。

    Raises:
        ValueError: 時刻と値が不整合、帯の片方だけが指定されている、帯の形が値と違う、
            または印の時刻が float64 の 1 次元配列でない場合。
    """
    check_series_arrays(time_s, values)
    if (band_lower is None) != (band_upper is None):
        raise ValueError("帯の下端と上端は組で指定してください")
    for name, band in (("band_lower", band_lower), ("band_upper", band_upper)):
        if band is not None and (band.dtype != np.float64 or band.shape != values.shape):
            raise ValueError(f"{name} は値と同じ形の float64 の配列にしてください")
    if marker_times_s is not None and (
        marker_times_s.dtype != np.float64 or marker_times_s.ndim != 1
    ):
        raise ValueError("marker_times_s は float64 の 1 次元配列にしてください")


def check_time_column_consistency(*, has_time_column: bool, is_sample_index: bool) -> None:
    """Source の時刻列の有無と時間軸の種類が整合していることを確認する。

    時刻列がないのに相対秒・絶対時刻を名乗ると、存在しない時刻情報を捏造することになる。
    逆に、時刻列があってもフレーム番号列の場合はサンプル番号軸として扱えるため許可する。

    Args:
        has_time_column: 時刻列があれば True。
        is_sample_index: 時間軸の種類がサンプル番号なら True。

    Raises:
        ValueError: 時刻列がないのに時間軸がサンプル番号以外の場合。
    """
    if not has_time_column and not is_sample_index:
        raise ValueError("時刻列がない Source の時間軸はサンプル番号（sample_index）にしてください")


# ---------------------------------------------------------------------------
# 信号処理パラメータの検証
# ---------------------------------------------------------------------------


def check_cutoff_frequencies(cutoff_hz: float | tuple[float, float], *, band: bool) -> None:
    """バターワースフィルタのカットオフ周波数の個数と大小関係を確認する。

    低域通過・高域通過は 1 つ、帯域通過・帯域阻止は (下限, 上限) の 2 つで指定する。
    ナイキスト周波数との比較は、サンプリング周波数が分かる適用時に行う。

    Args:
        cutoff_hz: カットオフ周波数 [Hz]。1 つ、または (下限, 上限)。
        band: 帯域通過・帯域阻止フィルタなら True。

    Raises:
        ValueError: 個数がフィルタの種類と合わない場合、または下限 >= 上限の場合。
    """
    if band:
        if not isinstance(cutoff_hz, tuple):
            raise ValueError(
                "帯域通過・帯域阻止フィルタのカットオフ周波数は "
                "(下限, 上限) の 2 つで指定してください"
            )
        low, high = cutoff_hz
        if low >= high:
            raise ValueError(f"カットオフ周波数の下限 {low} Hz が上限 {high} Hz 以上です")
    elif isinstance(cutoff_hz, tuple):
        raise ValueError("低域通過・高域通過フィルタのカットオフ周波数は 1 つで指定してください")


def check_odd_window(window_samples: int, *, purpose: str) -> None:
    """窓の長さが奇数であることを確認する。

    中心窓は現在のサンプルの前後に同じ数のサンプルを使う必要があるため、奇数に限る。
    偶数を許すと半サンプルの時間ずれが生じ、それが結果から見えなくなる。

    Args:
        window_samples: 窓の長さ [サンプル]。
        purpose: エラーメッセージに含める用途の説明（例: "中心移動平均"）。

    Raises:
        ValueError: 偶数の場合。
    """
    if window_samples % 2 == 0:
        raise ValueError(f"{purpose}の窓長は奇数にしてください（指定値: {window_samples}）")


def check_polynomial_order(polyorder: int, window_samples: int) -> None:
    """Savitzky-Golay フィルタの多項式次数が窓長より小さいことを確認する。

    窓内のサンプル数が多項式の係数の数（次数 + 1）以下だと、当てはめが一意に決まらない。

    Args:
        polyorder: 多項式の次数。
        window_samples: 窓の長さ [サンプル]。

    Raises:
        ValueError: 次数が窓長以上の場合。
    """
    if polyorder >= window_samples:
        raise ValueError(f"多項式の次数 {polyorder} は窓長 {window_samples} より小さくしてください")


def check_derivative_settings(
    *, uses_savitzky_golay: bool, savitzky_golay_polyorder: int | None, order: int
) -> None:
    """数値微分の方法と、Savitzky-Golay の条件の有無・次数が整合していることを確認する。

    使わない条件を受け付けると、保存された条件から実際の計算方法を読み違えるため拒否する。

    Args:
        uses_savitzky_golay: 微分の方法が Savitzky-Golay なら True。
        savitzky_golay_polyorder: Savitzky-Golay の多項式の次数。条件が指定されていなければ None。
        order: 微分の階数。

    Raises:
        ValueError: 方法と条件が整合しない場合、または多項式の次数が微分の階数未満の場合。
    """
    if uses_savitzky_golay and savitzky_golay_polyorder is None:
        raise ValueError("Savitzky-Golay 微分には savitzky_golay の条件が必要です")
    if not uses_savitzky_golay and savitzky_golay_polyorder is not None:
        raise ValueError("savitzky_golay の条件は Savitzky-Golay 微分のときだけ指定できます")
    if savitzky_golay_polyorder is not None and savitzky_golay_polyorder < order:
        raise ValueError("Savitzky-Golay の多項式の次数は微分の階数以上にしてください")


# ---------------------------------------------------------------------------
# カルマンフィルタのパラメータ・結果の検証
# ---------------------------------------------------------------------------


def check_matrix_shape(
    name: str,
    matrix: Sequence[Sequence[float]],
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    """入れ子のリストで表した行列が長方形で、指定の形であることを確認する。

    Args:
        name: エラーメッセージに含める行列名（例: "F"）。
        matrix: 行のリスト。
        rows: 期待する行数。None なら確認しない。
        cols: 期待する列数。None なら確認しない。

    Returns:
        (行数, 列数)。

    Raises:
        ValueError: 空、各行の長さが不揃い、または指定の形と異なる場合。
    """
    if not matrix or not matrix[0]:
        raise ValueError(f"行列 {name} が空です")
    n_cols = len(matrix[0])
    if any(len(row) != n_cols for row in matrix):
        raise ValueError(f"行列 {name} の各行の長さが揃っていません")
    shape = (len(matrix), n_cols)
    if (rows is not None and shape[0] != rows) or (cols is not None and shape[1] != cols):
        expected = f"{rows if rows is not None else '*'}×{cols if cols is not None else '*'}"
        raise ValueError(f"行列 {name} の形が {shape[0]}×{shape[1]} です（期待: {expected}）")
    return shape


def check_covariance(name: str, matrix: Sequence[Sequence[float]], *, strict: bool) -> None:
    """共分散行列が対称で、正定値（または半正定値）であることを確認する。

    Args:
        name: エラーメッセージに含める行列名（例: "R"）。
        matrix: 正方行列。
        strict: True なら正定値、False なら半正定値を要求する。
            観測雑音 R や初期共分散は逆行列計算に関わるため正定値、
            プロセス雑音 Q は一部の状態に雑音がないモデルもあるため半正定値でよい。

    Raises:
        ValueError: 非対称、または固有値が条件を満たさない場合。
    """
    array = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(array, array.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
        raise ValueError(f"共分散行列 {name} が対称ではありません")
    min_eigenvalue = float(np.linalg.eigvalsh(array).min())
    # 数値誤差による微小な負の固有値は、行列の大きさに応じて許容する。
    tolerance = _SYMMETRY_ATOL + _SYMMETRY_RTOL * float(np.abs(array).max())
    if strict and min_eigenvalue <= 0.0:
        raise ValueError(f"共分散行列 {name} が正定値ではありません（最小固有値 {min_eigenvalue}）")
    if not strict and min_eigenvalue < -tolerance:
        raise ValueError(
            f"共分散行列 {name} が半正定値ではありません（最小固有値 {min_eigenvalue}）"
        )


def check_linear_system(
    transition: Sequence[Sequence[float]],
    observation: Sequence[Sequence[float]],
    process_noise: Sequence[Sequence[float]],
    measurement_noise: Sequence[Sequence[float]],
) -> None:
    """線形状態空間モデルの行列 F・H・Q・R の形と性質が整合していることを確認する。

    状態の次元を n、観測の次元を m として、F は n×n、H は m×n、Q は n×n の半正定値、
    R は m×m の正定値でなければならない。

    Args:
        transition: 状態遷移行列 F。
        observation: 観測行列 H。
        process_noise: プロセス雑音共分散 Q。
        measurement_noise: 観測雑音共分散 R。

    Raises:
        ValueError: 形や性質が整合しない場合。
    """
    n_state, _ = check_matrix_shape("F", transition, cols=len(transition))
    n_measurement, _ = check_matrix_shape("H", observation, cols=n_state)
    check_matrix_shape("Q", process_noise, rows=n_state, cols=n_state)
    check_matrix_shape("R", measurement_noise, rows=n_measurement, cols=n_measurement)
    check_covariance("Q", process_noise, strict=False)
    check_covariance("R", measurement_noise, strict=True)


def check_initial_state(state: Sequence[float], covariance: Sequence[Sequence[float]]) -> None:
    """カルマンフィルタの初期状態と初期共分散の形・性質を確認する。

    Args:
        state: 初期状態ベクトル（長さ n）。
        covariance: 初期共分散（n×n の正定値行列）。

    Raises:
        ValueError: 状態が空、共分散の形が合わない、または正定値でない場合。
    """
    if not state:
        raise ValueError("初期状態が空です")
    check_matrix_shape("P0", covariance, rows=len(state), cols=len(state))
    check_covariance("P0", covariance, strict=True)


def check_kalman_result_arrays(
    time_s: FloatArray,
    states: FloatArray,
    covariances: FloatArray,
    innovations: FloatArray,
    innovation_covariances: FloatArray,
    measurement_noise: FloatArray,
    updated: BoolArray,
    noise_floor_applied: BoolArray,
) -> None:
    """カルマンフィルタの結果の各配列の dtype と形が整合していることを確認する。

    float の配列は float64 に限る（精度が暗黙に落ちていないことを保証するため、変換はしない）。
    形は時刻数 n・状態次元・観測次元と整合していなければならない。

    Args:
        time_s: 時刻 (n,)。
        states: 状態 (n, 状態次元)。
        covariances: 状態の共分散 (n, 状態次元, 状態次元)。
        innovations: イノベーション (n, 観測次元)。
        innovation_covariances: イノベーション共分散 (n, 観測次元, 観測次元)。
        measurement_noise: 使用した観測雑音共分散 (n, 観測次元, 観測次元)。
        updated: 観測更新の有無 (n,)。
        noise_floor_applied: 適応推定で正定値化を行ったか (n,)。

    Raises:
        ValueError: dtype または形が整合しない場合。
    """
    float_arrays = {
        "time_s": time_s,
        "states": states,
        "covariances": covariances,
        "innovations": innovations,
        "innovation_covariances": innovation_covariances,
        "measurement_noise": measurement_noise,
    }
    for name, array in float_arrays.items():
        if array.dtype != np.float64:
            raise ValueError(f"{name} は float64 の配列にしてください（実際: {array.dtype}）")
    for name, flags in {"updated": updated, "noise_floor_applied": noise_floor_applied}.items():
        if flags.dtype != np.bool_:
            raise ValueError(f"{name} は bool の配列にしてください（実際: {flags.dtype}）")

    if time_s.ndim != 1 or states.ndim != 2 or innovations.ndim != 2:
        raise ValueError("time_s は 1 次元、states・innovations は 2 次元の配列にしてください")
    n = time_s.shape[0]
    n_state = states.shape[1]
    n_measurement = innovations.shape[1]
    expected_shapes = [
        ("states", states.shape, (n, n_state)),
        ("covariances", covariances.shape, (n, n_state, n_state)),
        ("innovations", innovations.shape, (n, n_measurement)),
        (
            "innovation_covariances",
            innovation_covariances.shape,
            (n, n_measurement, n_measurement),
        ),
        ("measurement_noise", measurement_noise.shape, (n, n_measurement, n_measurement)),
        ("updated", updated.shape, (n,)),
        ("noise_floor_applied", noise_floor_applied.shape, (n,)),
    ]
    for name, actual, expected in expected_shapes:
        if actual != expected:
            raise ValueError(f"{name} の形が {actual} です（期待: {expected}）")


# ---------------------------------------------------------------------------
# 入出力関係の解析と、解析結果の表示の検証（Phase 4）
# ---------------------------------------------------------------------------


def check_alignment_settings(
    *, uniform: bool, uniform_rate_hz: float | None, start_s: float | None, end_s: float | None
) -> None:
    """2 チャンネルの対応付けの条件の組み合わせを確認する。

    Args:
        uniform: 等間隔の時刻に対応付けるなら True。
        uniform_rate_hz: 等間隔の時刻のサンプリング周波数 [Hz]（等間隔でなければ None）。
        start_s: 解析する範囲の開始時刻 [s]（指定しなければ None）。
        end_s: 解析する範囲の終了時刻 [s]（指定しなければ None）。

    Raises:
        ValueError: 等間隔なのに周波数がない、等間隔でないのに周波数がある
            （どちらの時刻を使ったか曖昧になるため）、または範囲の開始が終了以降の場合。
    """
    if uniform and uniform_rate_hz is None:
        raise ValueError("等間隔の時刻に対応付けるときは、サンプリング周波数を指定してください")
    if not uniform and uniform_rate_hz is not None:
        raise ValueError(
            "サンプリング周波数は、等間隔の時刻に対応付けるときだけ指定してください"
            "（入力・出力の時刻を使う場合は不要です）"
        )
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError(f"範囲の開始 {start_s} s が終了 {end_s} s 以降になっています")


def check_time_range(start_s: float | None, end_s: float | None) -> None:
    """時間の範囲の開始が終了より前であることを確認する。

    Args:
        start_s: 開始 [s]（指定しなければ None）。
        end_s: 終了 [s]（指定しなければ None）。

    Raises:
        ValueError: 開始が終了以降の場合。
    """
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError(f"範囲の開始 {start_s} s が終了 {end_s} s 以降になっています")


def check_onset_threshold(*, fraction_mode: bool, threshold: float) -> None:
    """ステップ応答の、変化の開始を判定するしきい値を確認する。

    Args:
        fraction_mode: しきい値を変化量に対する割合で指定するなら True。
        threshold: しきい値（正の値であることはモデルの型で確認済み）。

    Raises:
        ValueError: 割合で指定しているのに 1 以上の場合（変化が終わるまで判定できないため）。
    """
    if fraction_mode and threshold >= 1.0:
        raise ValueError(f"割合のしきい値は 1 未満にしてください（実際: {threshold}）")


def check_lag_range(min_lag_s: float, max_lag_s: float) -> None:
    """相互相関で遅れを探す範囲を確認する。

    Args:
        min_lag_s: 探す範囲の下限 [s]。
        max_lag_s: 探す範囲の上限 [s]。

    Raises:
        ValueError: 下限が上限以上の場合。
    """
    if min_lag_s >= max_lag_s:
        raise ValueError(
            f"遅れを探す範囲の下限 {min_lag_s} s が、上限 {max_lag_s} s 以上になっています"
        )


def check_two_channel_settings(
    *, two_channel: bool, has_output_channel: bool, has_alignment: bool, same_channel: bool
) -> None:
    """解析の種類と、出力チャンネル・対応付けの条件の指定が合っていることを確認する。

    Args:
        two_channel: 入力と出力の 2 チャンネルを使う種類なら True。
        has_output_channel: 出力チャンネルが指定されていれば True。
        has_alignment: 対応付けの条件が指定されていれば True。
        same_channel: 入力と出力に同じチャンネルが指定されていれば True。

    Raises:
        ValueError: 2 チャンネルの解析で出力チャンネルか対応付けの条件がない、
            入力と出力が同じチャンネル、または 1 チャンネルの解析で指定されている場合
            （使わない条件を残すと、保存した内容を読み違えるため）。
    """
    if two_channel:
        if not has_output_channel or not has_alignment:
            raise ValueError(
                "入力と出力の 2 チャンネルを使う解析には、出力チャンネルと"
                "時刻の対応付けの条件が必要です"
            )
        if same_channel:
            raise ValueError("入力と出力に同じチャンネルは指定できません")
    elif has_output_channel or has_alignment:
        raise ValueError(
            "出力チャンネルと時刻の対応付けの条件は、2 チャンネルを使う解析にだけ指定してください"
        )


def check_xy_arrays(x: FloatArray, y: FloatArray) -> None:
    """グラフの系列の X と Y が、同じ長さの float64 の 1 次元配列であることを確認する。

    Args:
        x: X 座標。
        y: Y 座標。

    Raises:
        ValueError: dtype・次元・長さが条件を満たさない場合。
    """
    for name, array in (("x", x), ("y", y)):
        if array.dtype != np.float64 or array.ndim != 1:
            raise ValueError(
                f"{name} は float64 の 1 次元配列にしてください"
                f"（実際: {array.dtype}、{array.ndim} 次元）"
            )
    if x.shape != y.shape:
        raise ValueError(f"x の長さ {x.shape[0]} と y の長さ {y.shape[0]} が違います")
