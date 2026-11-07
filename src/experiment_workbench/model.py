"""Experiment Workbench のドメインモデル定義（pydantic モデルのみ）。

このモジュールには pydantic モデルと、そのフィールド型として使う列挙型・定数だけを置く。
検証処理や既定値の生成処理は ``validation.py`` に置き、
モデルの検証メソッドからは呼び出すだけにする。
保存・読み込みは ``storage``、解析処理は ``analysis`` サブパッケージに置く。Qt にも依存しない。
``typing`` モジュールは使わない方針（docs/DECISIONS.md「コーディング規約」）。

階層構造（SPEC §5.1）::

    Project ─ Experiment ─ Session ─ Trial ─┬─ Source ─ Channel
                                             └─ Event（いずれかの Source の時間軸で定義）

時刻の扱い（SPEC §5.2）:
    - 各 Source は独立した時間軸（``TimeBase``）を持つ。Source 間で時刻原点・単位・
      サンプリング周波数を勝手に揃えない。揃える場合は将来の「明示的な変換」として記録する。
    - 内部の時刻は秒単位の float64 を基本とし、絶対時刻が必要なら epoch ナノ秒を併用する。

解析コア（SPEC §7 A-01〜A-08）の処理条件と結果、データの取り込み、表示状態のモデルも
このモジュールに置く。Pipeline モデルは v0.2 で追加する。
"""

from datetime import date
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from experiment_workbench import __version__
from experiment_workbench.validation import (
    COLOR_PATTERN,
    NON_BLANK_PATTERN,
    SHA256_PATTERN,
    VIEW_NAME_PATTERN,
    BoolArray,
    FloatArray,
    check_alignment_settings,
    check_cutoff_frequencies,
    check_derivative_settings,
    check_encoding,
    check_import_columns,
    check_initial_state,
    check_kalman_result_arrays,
    check_lag_range,
    check_linear_system,
    check_odd_window,
    check_onset_threshold,
    check_plot_series_arrays,
    check_polynomial_order,
    check_selected_settings,
    check_series_arrays,
    check_time_column_consistency,
    check_time_range,
    check_two_channel_settings,
    check_xy_arrays,
    utc_now,
    validate_relative_posix_path,
    validate_tags,
)

# project.json の保存形式バージョン。
# metadata.sqlite のスキーマバージョン（PRAGMA user_version）とは独立に管理する。
# 詳細は docs/DECISIONS.md「メタデータの保存先とバージョン管理」を参照。
PROJECT_FORMAT_VERSION = 1

# 正規化データ（datasets/ の Parquet）で、秒に換算した時刻を入れる列の名前。
# チャンネル名には使えない（予約名）。"time_s" のような実データでよく使う列名と衝突しないよう、
# 先頭に "__" を付けている（元ファイルの time_s 列をチャンネルとして取り込めるようにするため）。
DATASET_TIME_COLUMN = "__time_s"

# 表示設定（views/*.json）の保存形式バージョン。
# v2（Phase 3）: パネルの channel_ids を series_ids に改名（解析の出力も置けるようにしたため）。
VIEW_FORMAT_VERSION = 2

# 保存した解析（pipelines/*.json）の保存形式バージョン。
# v2（Phase 4）: 2 チャンネルの解析（output_channel_id・alignment・step_response・hysteresis・
# cross_correlation）を追加。v1 のファイルは、追加した項目が None としてそのまま読める。
ANALYSIS_FORMAT_VERSION = 3

DEFAULT_MAX_GAP_FACTOR = 1.5
"""時刻の間隔が、間隔の中央値のこの倍数を超えたら、行ごと欠けた区間とみなす。

1 サンプル欠ければ間隔は 2 倍になるため確実に検出でき、時刻のジッタ（数 % 程度）では
反応しない値。読み込みの警告・時刻の対応付け・リサンプリング・欠損の補間で同じ値を使う。
"""


# ---------------------------------------------------------------------------
# 基底クラス
# ---------------------------------------------------------------------------


class WorkbenchModel(BaseModel):
    """全モデル共通の基底クラス。

    設定の意図:
        - ``extra="forbid"``: 未知のフィールド（タイプミスや新しい保存形式のフィールド）を
          黙って捨てずにエラーにする。古いアプリで新しいデータを壊さないため。
        - ``validate_assignment=True``: 生成後の属性代入でも検証を行う。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Entity(WorkbenchModel):
    """ID と作成日時を持つ永続化対象モデルの基底クラス。

    Attributes:
        id: エンティティを一意に識別する UUID。プロジェクト間でコピーしても衝突しにくいよう
            連番ではなく UUID4 を使う。
        created_at: 作成日時（UTC）。
    """

    id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# プロジェクト階層
# ---------------------------------------------------------------------------


class Project(Entity):
    """研究またはデータ解析プロジェクト（project.json に保存する）。

    プロジェクト自体の情報は metadata.sqlite ではなく project.json に保存する。
    フォルダを開いたときに、DB を開く前に形式バージョンを確認できるようにするため。

    Attributes:
        name: プロジェクト名（空白のみは不可）。
        description: 任意の説明文。
        format_version: project.json の保存形式バージョン。
        app_version: プロジェクトを作成したアプリのバージョン（再現性の記録用）。
    """

    name: str = Field(pattern=NON_BLANK_PATTERN)
    description: str = ""
    format_version: PositiveInt = PROJECT_FORMAT_VERSION
    app_version: str = __version__


class Experiment(Entity):
    """目的・実験条件を共有する試験群。

    Attributes:
        name: 実験名（空白のみは不可）。プロジェクト内で一意（DB の UNIQUE 制約で保証）。
        description: 実験の目的や条件の説明。
    """

    name: str = Field(pattern=NON_BLANK_PATTERN)
    description: str = ""


class Session(Entity):
    """収録日や装置設定など、共通条件を持つ試技のまとまり。

    Attributes:
        experiment_id: 所属する Experiment の ID。
        name: セッション名（空白のみは不可）。同じ Experiment 内で一意。
        recorded_on: 収録日（不明なら None）。
        notes: 装置設定などのメモ。
    """

    experiment_id: UUID
    name: str = Field(pattern=NON_BLANK_PATTERN)
    recorded_on: date | None = None
    notes: str = ""


class Trial(Entity):
    """1 回の試行。

    Attributes:
        session_id: 所属する Session の ID。
        name: 試技名（空白のみは不可）。同じ Session 内で一意。
        tags: 「success」「failure」などの任意タグ。
            空白のみのタグと重複は、黙って除去せずエラーにする。
        notes: 任意のメモ。
    """

    session_id: UUID
    name: str = Field(pattern=NON_BLANK_PATTERN)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, tags: list[str]) -> list[str]:
        """タグ一覧を検証する（validation.validate_tags に委ねる）。

        Args:
            tags: 検証するタグ一覧。

        Returns:
            入力と同じタグ一覧。
        """
        return validate_tags(tags)


# ---------------------------------------------------------------------------
# 入力データの異常
# ---------------------------------------------------------------------------


class DataIssueKind(StrEnum):
    """入力データの異常の種類（SPEC §9「データ整合性」、§10「失敗データ」）。

    合成データに仕込んだ異常の記録と、取り込み時の検証結果の報告に使う。
    """

    EMPTY_FILE = "empty_file"
    """ファイルが 0 バイト（またはヘッダー行がない）。"""
    HEADER_ONLY = "header_only"
    """ヘッダー行のみでデータ行がない。"""
    DUPLICATE_COLUMN = "duplicate_column"
    """同じ列名が複数ある。"""
    NON_UTF8_ENCODING = "non_utf8_encoding"
    """指定した文字コードで読めない（Shift_JIS のファイルを UTF-8 として読んだ場合など）。"""
    NAN_VALUES = "nan_values"
    """NaN や空欄の値を含む。"""
    NON_MONOTONIC_TIME = "non_monotonic_time"
    """時刻が逆行している。"""
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    """同じ時刻のサンプルが複数ある。"""
    MISSING_SEGMENT = "missing_segment"
    """一定区間のサンプルが欠落している（時刻の飛び）。"""
    NON_NUMERIC_VALUES = "non_numeric_values"
    """数値として読めない値がある。"""
    INVALID_COLUMN = "invalid_column"
    """列名が空、予約名と重なる、または指定した列がない。"""
    READ_ERROR = "read_error"
    """ファイルを表として読み込めない（形式や区切り文字の誤りなど）。"""


class IssueSeverity(StrEnum):
    """入力データの異常の重大度。"""

    ERROR = "error"
    """取り込みを中止する（データとして意味が定まらない）。"""
    WARNING = "warning"
    """取り込みは行い、記録と表示で知らせる（データはそのまま保持する）。"""


class DataIssue(WorkbenchModel):
    """入力データの異常 1 件。

    Attributes:
        kind: 異常の種類。
        description: 人が読める説明。
        column: 異常のある列名。ファイル全体に関わる場合は None。
        severity: 重大度。エラーがあると取り込まない。
        row_index: 最初に該当したデータ行の番号（0 始まり、ヘッダーを除く）。
            行に関係しない場合は None。
    """

    kind: DataIssueKind
    description: str
    column: str | None = None
    severity: IssueSeverity = IssueSeverity.WARNING
    row_index: NonNegativeInt | None = None


# ---------------------------------------------------------------------------
# 時間軸・Source・Channel
# ---------------------------------------------------------------------------


class TimeAxisKind(StrEnum):
    """Source の生データの時間軸の種類（SPEC §3.2「明示的な時刻定義」）。

    イベント時刻を t=0 とする「イベント相対時刻」は、整列という変換の結果であり
    生データの時間軸ではないため、ここには含めない（整列機能の Phase で別に定義する）。
    """

    SAMPLE_INDEX = "sample_index"
    """サンプル番号（時刻列がない、またはフレーム番号のみのデータ）。"""
    RELATIVE_SECONDS = "relative_seconds"
    """装置固有の原点からの相対秒。"""
    ABSOLUTE_EPOCH_NS = "absolute_epoch_ns"
    """UNIX エポックからの時刻（絶対時刻）。単位は ``TimeBase.original_unit`` で表す。"""


class TimeUnit(StrEnum):
    """時刻列の単位。秒への換算係数は timebase.py にある。"""

    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


class TimeBase(WorkbenchModel):
    """Source の時間軸の定義。

    時刻の値そのものは Parquet 側に保持し、ここには「その時刻が何を意味するか」を記録する。

    Attributes:
        kind: 時刻の種類。
        original_unit: 元ファイルの時刻列の単位。時刻列がない場合は None。
            正規化時に秒へ換算しても、元の単位をここに残す。
        origin: 時刻原点の説明（例: "装置の電源投入時"、"収録開始"）。不明なら "unknown"。
        clock: 時刻を刻んだ装置クロックの由来（例: "ロガー内部クロック"）。不明なら "unknown"。
        resolution_s: 時刻の分解能 [s]。不明なら None。
        epoch_origin_ns: 相対秒 0 に対応する絶対時刻（エポックナノ秒）。既知の場合のみ。
        sample_interval_s: 時刻列がない場合に、サンプル番号から時刻 [s] を作るときの間隔。
            ユーザーが取り込み時に指定する。時刻列がある場合は None。
        shares_trial_clock: True なら、この Source の時刻（秒に換算後）は、同じ試技で同じく True の
            Source と同じ時計・同じ原点の値として直接比較できる（ユーザーが取り込み時に明示する）。
            False なら、他の Source とは時刻を比較できないものとして扱う（勝手に揃えない）。
    """

    kind: TimeAxisKind
    original_unit: TimeUnit | None = None
    origin: str = "unknown"
    clock: str = "unknown"
    resolution_s: PositiveFloat | None = None
    epoch_origin_ns: int | None = None
    sample_interval_s: PositiveFloat | None = None
    shares_trial_clock: bool = False


class SampleRate(WorkbenchModel):
    """サンプリング周波数のメタデータ。

    公称値と、実時刻から推定した値を区別して保持する。異なる周波数のデータを
    無断でリサンプリングしないため、ここに値があっても自動では揃えない。

    Attributes:
        nominal_hz: 装置設定などによる公称値 [Hz]。不明なら None。
        estimated_hz: 時刻列から推定した値 [Hz]。未推定なら None。
    """

    nominal_hz: PositiveFloat | None = None
    estimated_hz: PositiveFloat | None = None


class SourceFormat(StrEnum):
    """取り込み元ファイルの形式（v0.1 の対応形式）。"""

    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"


class ReadOptions(WorkbenchModel):
    """元ファイルを表として読み込むときの条件（同じ条件で読み直せるよう記録する）。

    Attributes:
        delimiter: 区切り文字（1 文字）。CSV はカンマ、TSV はタブなど。Parquet では使わない。
        encoding: 文字コード（Python のコーデック名。例: "utf-8"、"cp932"）。Parquet では使わない。
            UTF-8 の BOM は "utf-8" のままで読める（BOM はデータではないため取り除いて読む）。
        skip_rows: ヘッダー行の前に読み飛ばす行数（装置が書き出す前置きの行など）。
    """

    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: str = "utf-8"
    skip_rows: NonNegativeInt = 0

    @field_validator("encoding")
    @classmethod
    def _check_encoding(cls, value: str) -> str:
        """文字コード名を検証する（validation.check_encoding に委ねる）。

        Args:
            value: 文字コード名。

        Returns:
            入力と同じ文字列。
        """
        return check_encoding(value)


class Source(Entity):
    """1 つの入力ファイル由来のデータ群。

    元ファイルは originals/ にコピーして SHA-256 を記録し、正規化データは datasets/ の
    Parquet に保存する（SPEC §5.3）。パスはすべてプロジェクトルートからの相対パス。

    Attributes:
        trial_id: 所属する Trial の ID。
        name: Source 名（空白のみは不可）。同じ Trial 内で一意。
        format: 元ファイルの形式。
        original_filename: 取り込み時の元ファイル名（ディレクトリを含まない）。
        original_relpath: originals/ にコピーした元ファイルの相対パス。
        original_sha256: 元ファイルの SHA-256（16 進小文字）。改変検知と再現性の記録に使う。
        original_size_bytes: 元ファイルのサイズ [byte]。
        dataset_relpath: 正規化データ（Parquet）の相対パス。
        time_column: 元ファイルで時刻として使った列名。時刻列がない場合は None。
        time_base: この Source の時間軸の定義。
        read_options: CSV / TSV の読み込み条件。Parquet と、スキーマ v1 で登録した Source は None。
        issues: 取り込み時に検出した異常（警告）。
            エラーがあるファイルは取り込まないため、警告のみが入る。
    """

    trial_id: UUID
    name: str = Field(pattern=NON_BLANK_PATTERN)
    format: SourceFormat
    original_filename: str = Field(pattern=NON_BLANK_PATTERN)
    original_relpath: str
    original_sha256: str = Field(pattern=SHA256_PATTERN)
    original_size_bytes: NonNegativeInt
    dataset_relpath: str
    time_column: str | None
    time_base: TimeBase
    read_options: ReadOptions | None = None
    issues: list[DataIssue] = Field(default_factory=list)

    @field_validator("original_relpath", "dataset_relpath")
    @classmethod
    def _check_relpath(cls, value: str) -> str:
        """相対パスを検証する（validation.validate_relative_posix_path に委ねる）。

        Args:
            value: 検証する相対パス。

        Returns:
            入力と同じ文字列。
        """
        return validate_relative_posix_path(value)

    @model_validator(mode="after")
    def _check_time_column(self) -> "Source":
        """時刻列の有無と時間軸の種類の整合性を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_time_column_consistency(
            has_time_column=self.time_column is not None,
            is_sample_index=self.time_base.kind is TimeAxisKind.SAMPLE_INDEX,
        )
        return self


class Channel(Entity):
    """個別の時系列（1 列分）のメタデータ。

    値そのものは Source の Parquet に保持し、ここには列名・単位・周波数を記録する。

    Attributes:
        source_id: 所属する Source の ID。
        name: 正規化データ（Parquet）上の列名（空白のみは不可）。同じ Source 内で一意。
        original_name: 元ファイルでの列名。空文字や重複していた場合の追跡のため、そのまま残す。
        unit: 正規化後の単位（原則 SI）。単位不明は None、無次元は "1" とする。
        original_unit: 元ファイルでの単位。不明なら None。
        sample_rate: サンプリング周波数（公称値・推定値）。
        description: 任意の説明。
    """

    source_id: UUID
    name: str = Field(pattern=NON_BLANK_PATTERN)
    original_name: str
    unit: str | None = None
    original_unit: str | None = None
    sample_rate: SampleRate = Field(default_factory=SampleRate)
    description: str = ""


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class EventDetectionMethod(StrEnum):
    """イベント時刻の決め方（SPEC §9「イベント時刻の定義と算出方法を保存する」）。"""

    MANUAL = "manual"
    """ユーザーが手動で指定した。"""
    IMPORTED = "imported"
    """外部ファイル（装置のトリガ記録など）から取り込んだ。"""
    AUTOMATIC = "automatic"
    """検出アルゴリズムで求めた（v0.2 以降）。"""


class Event(Entity):
    """試技内の任意の時刻に付けるイベント。

    Source ごとに時間軸が異なるため、イベント時刻は必ず「どの Source の時間軸での時刻か」
    （``source_id``）とセットで保持する。Trial 共通の時間軸は存在しない前提とする。

    Attributes:
        trial_id: 所属する Trial の ID。
        source_id: ``time_s`` の基準となる時間軸を持つ Source の ID。
            同じ Trial に属する Source であること（DB の複合外部キーで保証）。
        label: イベント名（例: "start"。空白のみは不可）。
        time_s: ``source_id`` の時間軸での時刻 [s]。NaN・無限大は不可。
        detection_method: イベント時刻の決め方。
        detection_params: 自動検出の閾値など、時刻の算出条件（JSON で表せる値）。
        note: 任意のメモ。
    """

    trial_id: UUID
    source_id: UUID
    label: str = Field(pattern=NON_BLANK_PATTERN)
    time_s: FiniteFloat
    detection_method: EventDetectionMethod = EventDetectionMethod.MANUAL
    detection_params: dict[str, JsonValue] = Field(default_factory=dict)
    note: str = ""


# ---------------------------------------------------------------------------
# 合成データ（SPEC §10）
# ---------------------------------------------------------------------------


class SyntheticConfig(WorkbenchModel):
    """合成データ生成の設定。

    特定のセンサは想定しない。「真値が解析的に分かる 1 次元の運動」を、サンプリング周波数と
    時刻原点が異なる 2 つのファイルで観測したデータを作る。既定値は SPEC §10 の規模
    （180 秒、1000 Hz と 100 Hz）に合わせる。テストでは ``duration_s`` を短くして使う。

    Attributes:
        seed: 乱数シード。同じ設定なら同じファイルが生成される。
        duration_s: 1 試技の長さ（真の時間軸）[s]。欠損区間などを配置するため 1 秒以上。
        n_trials: 生成する試技数。
        high_rate_hz: 加速度を観測する高レートファイルのサンプリング周波数 [Hz]。
        low_rate_hz: 位置を観測する低レートファイルのサンプリング周波数 [Hz]。
            時刻列をミリ秒の整数で書き出すため、1000 Hz を超えると時刻が重複する。
            それを防ぐため上限を 1000 Hz とする。
    """

    seed: int = 20261029
    duration_s: float = Field(default=180.0, ge=1.0)
    n_trials: PositiveInt = 3
    high_rate_hz: PositiveFloat = 1000.0
    low_rate_hz: float = Field(default=100.0, gt=0.0, le=1000.0)


class SineComponent(WorkbenchModel):
    """正弦波 1 成分 ``A sin(2π f t + φ)`` のパラメータ。

    Attributes:
        amplitude: 振幅 A。
        frequency_hz: 周波数 f [Hz]。
        phase_rad: 初期位相 φ [rad]。
    """

    amplitude: float
    frequency_hz: PositiveFloat
    phase_rad: float


class SyntheticMotion(WorkbenchModel):
    """合成データの真値となる 1 次元の運動。

    位置 x(t) [m] は正弦波の和で、速度・加速度はそれを解析的に微分した値とする::

        x(t) =  Σ_k A_k sin(2π f_k t + φ_k)
        v(t) =  Σ_k (2π f_k) A_k cos(2π f_k t + φ_k)
        a(t) = -Σ_k (2π f_k)^2 A_k sin(2π f_k t + φ_k)

    数値積分・フィルタ・カルマンフィルタの結果を、離散化誤差のない真値と比較するため
    解析解を使う。

    Attributes:
        position_components: 位置 x(t) を構成する正弦波成分（1 つ以上。振幅の単位は m）。
    """

    position_components: list[SineComponent] = Field(min_length=1)


class SyntheticSourceRecord(WorkbenchModel):
    """合成した 1 ファイル（Source）の生成条件と真値情報。

    ファイルの時刻と真の時刻 ``t_true`` [s] の関係::

        source_time_s = clock_offset_s + t_true

    時刻列は ``time_unit`` の単位で書かれる（"ms" の場合は ``source_time_s × 1000`` を
    整数に丸めた値）。

    Attributes:
        relpath: 出力ディレクトリからの相対パス。
        time_column: 時刻列名。
        time_unit: 時刻列の単位（"s" または "ms"）。
        nominal_rate_hz: 公称サンプリング周波数 [Hz]。
        n_rows: データ行数（欠損区間を除いた後）。
        column_units: 列名 → 単位 の対応（時刻列を含む）。
        measured_columns: 観測値の列。ノイズなどを含み、解析の入力に使う。
        truth_columns: 真値の列（``true_`` 接頭辞）。解析結果の評価に使う。
        clock_offset_s: 真の時刻 0 におけるファイル時刻の値 [s]。
        recording_start_true_s: 収録開始の真の時刻 [s]。
        measurement_model: 観測値の作り方（式）の説明。
        noise_parameters: 観測ノイズなどのパラメータ（名前 → 値）。
            カルマンフィルタの雑音共分散を設定するときの参考値になる。
        injected_issues: 意図的に仕込んだ異常。
    """

    relpath: str
    time_column: str
    time_unit: TimeUnit
    nominal_rate_hz: PositiveFloat
    n_rows: NonNegativeInt
    column_units: dict[str, str]
    measured_columns: list[str]
    truth_columns: list[str]
    clock_offset_s: float
    recording_start_true_s: NonNegativeFloat
    measurement_model: str
    noise_parameters: dict[str, float]
    injected_issues: list[DataIssue] = Field(default_factory=list)

    @field_validator("relpath")
    @classmethod
    def _check_relpath(cls, value: str) -> str:
        """相対パスを検証する（validation.validate_relative_posix_path に委ねる）。

        Args:
            value: 検証する相対パス。

        Returns:
            入力と同じ文字列。
        """
        return validate_relative_posix_path(value)


class SyntheticIoSystem(WorkbenchModel):
    """入出力の合成データ（io_pair.csv）を作った系の真値（入出力関係の解析の検証用）。

    出力のモデル::

        p(t) = play_h(u(t))                                   # 幅 h の遊び（バックラッシュ）
        y(t) = offset + gain × LAG_τ[p](t - delay) + N(0, σ_y²)  # 一次遅れ（時定数 τ）とむだ時間

    遊びは、入力が向きを変えると、入力が h だけ動くまで p を止める（最初は増加してきた状態）。
    一次遅れは区間ごとに入力を一定とみなして厳密に離散化する（ステップには厳密、ランプには
    半サンプル分の遅れが加わる）。入力の観測値は u + N(0, σ_u²)。

    入力 u(t) の作り方（真の時刻）:
        - 前半: ``step_times_s`` の各時刻に ``step_levels`` の値へ変わるステップ
          （最初の値は ``initial_level``）。ステップの時刻はサンプルの時刻に合わせる。
        - 後半: ``triangle_start_s`` から、``triangle_low`` と ``triangle_high`` の間を
          周期 ``triangle_period_s`` で往復する三角波（low から上がり始める）。

    Attributes:
        gain: 遊びの後の倍率（出力の単位 / 入力の単位）。
        offset: 出力のオフセット。
        delay_s: むだ時間 [s]（サンプル間隔の整数倍）。
        time_constant_s: 一次遅れの時定数 [s]。
        play_width: 遊びの幅（入力の単位）。
        input_noise_std: 入力の観測雑音の標準偏差。
        output_noise_std: 出力の観測雑音の標準偏差。
        initial_level: 入力の最初の値。
        step_times_s: ステップの真の時刻 [s]。
        step_levels: 各ステップの後の入力の値。
        triangle_start_s: 三角波を始める真の時刻 [s]。
        triangle_period_s: 三角波の周期 [s]。
        triangle_low: 三角波の最小値。
        triangle_high: 三角波の最大値。
    """

    gain: float
    offset: float
    delay_s: NonNegativeFloat
    time_constant_s: PositiveFloat
    play_width: NonNegativeFloat
    input_noise_std: NonNegativeFloat
    output_noise_std: NonNegativeFloat
    initial_level: float
    step_times_s: list[float]
    step_levels: list[float]
    triangle_start_s: float
    triangle_period_s: PositiveFloat
    triangle_low: float
    triangle_high: float


class SyntheticTrialRecord(WorkbenchModel):
    """合成した 1 試技の真値情報。

    Attributes:
        name: 試技名（ディレクトリ名と同じ）。
        motion: この試技の真の運動。
        sources: この試技の運動の観測ファイル（高レート・低レート）。
        io_pair: 入出力の合成データのファイル（io_pair.csv）。
        io_system: io_pair.csv を作った系の真値。
    """

    name: str
    motion: SyntheticMotion
    sources: list[SyntheticSourceRecord]
    io_pair: SyntheticSourceRecord
    io_system: SyntheticIoSystem


class SyntheticBrokenFileRecord(WorkbenchModel):
    """取り込みエラー検証用の失敗データ 1 件。

    Attributes:
        relpath: 出力ディレクトリからの相対パス。
        encoding: ファイルの文字コード。
        description: どのように壊れているかの説明。
        expected_issues: 取り込み時に検出されるべき異常。
    """

    relpath: str
    encoding: str
    description: str
    expected_issues: list[DataIssue]

    @field_validator("relpath")
    @classmethod
    def _check_relpath(cls, value: str) -> str:
        """相対パスを検証する（validation.validate_relative_posix_path に委ねる）。

        Args:
            value: 検証する相対パス。

        Returns:
            入力と同じ文字列。
        """
        return validate_relative_posix_path(value)


class SyntheticManifest(WorkbenchModel):
    """合成データ一式の目録（manifest.json）。

    テストや解析機能（フィルタ・積分・カルマンフィルタ・時刻同期）の検証で「正解」として
    参照できるよう、生成条件と真値を保存する。

    Attributes:
        generator_version: 生成に使ったパッケージのバージョン。
        created_at: 生成日時（UTC）。ファイル内容の決定性には影響しない。
        config: 生成設定。
        trials: 試技ごとの真値情報。
        broken_files: 失敗データの一覧。
    """

    generator_version: str
    created_at: AwareDatetime = Field(default_factory=utc_now)
    config: SyntheticConfig
    trials: list[SyntheticTrialRecord]
    broken_files: list[SyntheticBrokenFileRecord]


# ---------------------------------------------------------------------------
# 解析コア: 信号処理のパラメータ（SPEC §7 A-02〜A-04）
#
# 処理の条件を後から再現できるよう、パラメータはすべてモデルとして保存できる形にする。
# 複数フィールドにまたがる検証は validation.py の関数に委ね、ここでは呼び出すだけにする。
# ---------------------------------------------------------------------------


class ButterworthKind(StrEnum):
    """バターワースフィルタの種類（値は scipy.signal.butter の btype と同じ）。"""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class FilterPadType(StrEnum):
    """ゼロ位相フィルタで信号の端を延長する方法（scipy.signal.sosfiltfilt の padtype）。"""

    ODD = "odd"
    """端点を中心に点対称に折り返す（端の傾きを保ちやすい。scipy の既定）。"""
    EVEN = "even"
    """端点を軸に線対称に折り返す。"""
    CONSTANT = "constant"
    """端点の値で延長する。"""


class ButterworthParams(WorkbenchModel):
    """バターワースフィルタの条件。

    Attributes:
        kind: フィルタの種類。
        order: フィルタの次数（1〜10）。
            ゼロ位相の場合、前後 2 回かけるため実効的な次数は 2 倍になる。
        cutoff_hz: カットオフ周波数 [Hz]。低域通過・高域通過は 1 つ、
            帯域通過・帯域阻止は (下限, 上限)。ナイキスト周波数未満であること（適用時に確認）。
        zero_phase: True なら前向き・後ろ向きにかけて位相遅れをなくす（scipy.signal.sosfiltfilt）。
            False なら因果的にかける（scipy.signal.sosfilt）。
            因果的な場合、フィルタの初期状態は 0 とし、先頭に過渡応答が現れる。
        padtype: ゼロ位相の場合の端点の延長方法。None なら延長しない。因果的な場合は使わない。
    """

    kind: ButterworthKind
    order: int = Field(default=4, ge=1, le=10)
    cutoff_hz: PositiveFloat | tuple[PositiveFloat, PositiveFloat]
    zero_phase: bool = True
    padtype: FilterPadType | None = FilterPadType.ODD

    @model_validator(mode="after")
    def _check_cutoff(self) -> "ButterworthParams":
        """カットオフ周波数の個数がフィルタの種類と合っているかを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        band = self.kind in (ButterworthKind.BANDPASS, ButterworthKind.BANDSTOP)
        check_cutoff_frequencies(self.cutoff_hz, band=band)
        return self


class MovingAverageParams(WorkbenchModel):
    """移動平均の条件。

    窓内のサンプルが揃わない端点は、値を捏造しないよう NaN にする。

    Attributes:
        window_samples: 窓の長さ [サンプル]。
        centered: True なら中心窓（位相遅れなし、窓長は奇数）。
            False なら後方窓（現在までのサンプルだけを使う因果的な平均）。
            後方窓は (窓長 - 1) / 2 サンプル遅れる。
    """

    window_samples: PositiveInt
    centered: bool = True

    @model_validator(mode="after")
    def _check_window(self) -> "MovingAverageParams":
        """中心窓の窓長が奇数かを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        if self.centered:
            check_odd_window(self.window_samples, purpose="中心移動平均")
        return self


class SavitzkyGolayMode(StrEnum):
    """Savitzky-Golay フィルタの端点の扱い（scipy.signal.savgol_filter の mode）。"""

    INTERP = "interp"
    """端の窓に当てはめた多項式で端点を評価する（外挿や折り返しをしない。scipy の既定）。"""
    MIRROR = "mirror"
    """端点を軸に折り返して延長する。"""
    NEAREST = "nearest"
    """端点の値で延長する。"""
    CONSTANT = "constant"
    """0 で延長する。"""
    WRAP = "wrap"
    """反対側の端から循環させて延長する。"""


class SavitzkyGolayParams(WorkbenchModel):
    """Savitzky-Golay フィルタ（窓内の多項式当てはめによる平滑化・微分）の条件。

    Attributes:
        window_samples: 窓の長さ [サンプル]（奇数）。
        polyorder: 当てはめる多項式の次数（窓長未満）。
        mode: 端点の扱い。
    """

    window_samples: PositiveInt
    polyorder: NonNegativeInt
    mode: SavitzkyGolayMode = SavitzkyGolayMode.INTERP

    @model_validator(mode="after")
    def _check_window(self) -> "SavitzkyGolayParams":
        """窓長が奇数で、多項式の次数が窓長未満かを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_odd_window(self.window_samples, purpose="Savitzky-Golay フィルタ")
        check_polynomial_order(self.polyorder, self.window_samples)
        return self


class DerivativeMethod(StrEnum):
    """数値微分の方法。"""

    GRADIENT = "gradient"
    """中心差分（numpy.gradient）。不等間隔の時刻にも対応する。ノイズは増幅される。"""
    SAVITZKY_GOLAY = "savitzky_golay"
    """Savitzky-Golay 微分。平滑化と微分を同時に行う。等間隔の時刻が必要。"""


class DerivativeParams(WorkbenchModel):
    """数値微分の条件。

    Attributes:
        method: 微分の方法。
        order: 微分の階数（1 または 2）。
        edge_order: 中心差分の端点での差分の精度（numpy.gradient の edge_order。1 または 2）。
        savitzky_golay: Savitzky-Golay 微分の条件。``method`` が SAVITZKY_GOLAY のときだけ指定する。
            多項式の次数は微分の階数以上であること。
    """

    method: DerivativeMethod = DerivativeMethod.GRADIENT
    order: int = Field(default=1, ge=1, le=2)
    edge_order: int = Field(default=2, ge=1, le=2)
    savitzky_golay: SavitzkyGolayParams | None = None

    @model_validator(mode="after")
    def _check_method_params(self) -> "DerivativeParams":
        """微分の方法と、Savitzky-Golay の条件の有無・次数が整合しているかを確認する。

        Returns:
            検証済みの自身。
        """
        check_derivative_settings(
            uses_savitzky_golay=self.method is DerivativeMethod.SAVITZKY_GOLAY,
            savitzky_golay_polyorder=(
                None if self.savitzky_golay is None else self.savitzky_golay.polyorder
            ),
            order=self.order,
        )
        return self


class IntegrationMethod(StrEnum):
    """数値積分の方法（いずれも不等間隔の時刻に対応する）。"""

    TRAPEZOID = "trapezoid"
    """累積台形則（scipy.integrate.cumulative_trapezoid）。"""
    SIMPSON = "simpson"
    """累積シンプソン則（scipy.integrate.cumulative_simpson）。滑らかな信号では台形則より高精度。"""


class IntegralParams(WorkbenchModel):
    """数値積分の条件。

    Attributes:
        method: 積分の方法。
        initial_value: 先頭時刻での積分値（積分定数）。暗黙に 0 とせず、条件として記録する。
    """

    method: IntegrationMethod = IntegrationMethod.TRAPEZOID
    initial_value: FiniteFloat = 0.0


# ---------------------------------------------------------------------------
# 解析コア: カルマンフィルタ系のパラメータと結果（SPEC §7 A-05〜A-08）
# ---------------------------------------------------------------------------


class LinearSystemParams(WorkbenchModel):
    """定数行列の線形状態空間モデル。

    モデル::

        x_k = F x_{k-1} + w_k,  w_k ~ N(0, Q)
        z_k = H x_k + v_k,      v_k ~ N(0, R)

    F と Q は 1 サンプル分の時間に対する値なので、等間隔の時刻を前提とする。

    Attributes:
        transition: 状態遷移行列 F（n×n）。
        observation: 観測行列 H（m×n）。
        process_noise: プロセス雑音共分散 Q（n×n、半正定値）。
        measurement_noise: 観測雑音共分散 R（m×m、正定値）。
    """

    transition: list[list[FiniteFloat]]
    observation: list[list[FiniteFloat]]
    process_noise: list[list[FiniteFloat]]
    measurement_noise: list[list[FiniteFloat]]

    @model_validator(mode="after")
    def _check_shapes(self) -> "LinearSystemParams":
        """行列の形と共分散の性質が整合しているかを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_linear_system(
            self.transition, self.observation, self.process_noise, self.measurement_noise
        )
        return self


class KinematicSystemParams(WorkbenchModel):
    """値とその時間微分を状態とする運動学モデル（時刻の間隔に応じて F・Q を計算する）。

    ``derivative_order`` が d のとき、状態は [値, 1 階微分, ..., d 階微分]、観測は値のみとする。
    d 階微分が強度 q の連続時間白色雑音で駆動されると仮定し、時間間隔 Δt ごとに離散化する
    （d=1 は等速度モデル、d=2 は等加速度モデル）。不等間隔の時刻に対応する。

    Attributes:
        derivative_order: 状態に含める微分の最高階数（1 または 2）。
        process_noise_density: 最高階の微分を駆動する白色雑音のパワースペクトル密度 q。
            単位は (値の単位)² / s^(2d-1)。大きいほど観測に素早く追従し、平滑化は弱くなる。
        measurement_noise_std: 観測雑音の標準偏差（値の単位）。R = σ²。
    """

    derivative_order: int = Field(default=1, ge=1, le=2)
    process_noise_density: PositiveFloat
    measurement_noise_std: PositiveFloat


class KalmanInitialState(WorkbenchModel):
    """カルマンフィルタの初期状態（先頭時刻における事前推定）。

    初期値は推定結果に影響するため、暗黙に決めずに明示して記録する。

    Attributes:
        state: 初期状態ベクトル x0（長さ n）。
        covariance: 初期共分散 P0（n×n、正定値）。不確かなほど大きくする。
    """

    state: list[FiniteFloat]
    covariance: list[list[FiniteFloat]]

    @model_validator(mode="after")
    def _check_shapes(self) -> "KalmanInitialState":
        """状態と共分散の形・性質を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_initial_state(self.state, self.covariance)
        return self


class AdaptiveNoiseMethod(StrEnum):
    """適応カルマンフィルタで観測雑音共分散 R を推定する方法。"""

    INNOVATION_WINDOW = "innovation_window"
    """直近 N 個のイノベーションの標本共分散 C から R = C - H P⁻ Hᵀ と推定する
    （Mohamed & Schwarz, 1999 の innovation-based adaptive estimation）。"""
    SAGE_HUSA = "sage_husa"
    """忘却係数 b で過去を指数的に忘れながら R を逐次更新する（Sage-Husa 推定）。"""


class AdaptiveNoiseParams(WorkbenchModel):
    """適応カルマンフィルタ（観測雑音共分散 R の逐次推定）の条件。

    推定した R が正定値でない場合は、固有値を ``min_variance`` で下から抑えて正定値にする。
    この補正を行った時刻は結果（``KalmanResult.noise_floor_applied``）に記録する。

    Attributes:
        method: 推定方法。
        window: INNOVATION_WINDOW で使うイノベーションの個数 N（2 以上）。
            N 個たまるまでは初期の R を使う。
        forgetting_factor: SAGE_HUSA の忘却係数 b（0 < b < 1）。1 に近いほど長い過去を使う。
        min_variance: 推定した R の固有値の下限。
    """

    method: AdaptiveNoiseMethod
    window: int = Field(default=50, ge=2)
    forgetting_factor: float = Field(default=0.98, gt=0.0, lt=1.0)
    min_variance: PositiveFloat = 1e-12


class KalmanResult(WorkbenchModel):
    """カルマンフィルタ系の推定結果（時刻ごとの記録）。

    n は時刻の数、nx は状態の次元、nz は観測の次元。観測が NaN で更新しなかった時刻は、
    状態・共分散に予測値が入り、イノベーション関連の値は NaN になる。

    Attributes:
        time_s: 時刻 [s] (n,)。入力と同じ値。
        states: 推定した状態（観測更新後の事後推定）(n, nx)。
        covariances: 状態の共分散 (n, nx, nx)。
        innovations: イノベーション（観測 - 予測観測）(n, nz)。
        innovation_covariances: イノベーション共分散 S (n, nz, nz)。
        measurement_noise: その時刻で使った観測雑音共分散 R (n, nz, nz)。
            適応推定では時刻ごとに変わる。
        updated: 観測更新を行ったか (n,)。観測に NaN を含む時刻は False。
        noise_floor_applied: 適応推定で R の正定値化（固有値の下限処理）を行ったか (n,)。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time_s: FloatArray
    states: FloatArray
    covariances: FloatArray
    innovations: FloatArray
    innovation_covariances: FloatArray
    measurement_noise: FloatArray
    updated: BoolArray
    noise_floor_applied: BoolArray

    @model_validator(mode="after")
    def _check_arrays(self) -> "KalmanResult":
        """各配列の dtype と形が整合しているかを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_kalman_result_arrays(
            self.time_s,
            self.states,
            self.covariances,
            self.innovations,
            self.innovation_covariances,
            self.measurement_noise,
            self.updated,
            self.noise_floor_applied,
        )
        return self


# ---------------------------------------------------------------------------
# データの取り込み（SPEC F-02・F-03）
# ---------------------------------------------------------------------------


class ChannelImportSpec(WorkbenchModel):
    """取り込むチャンネル（列）1 つの指定。

    Attributes:
        column: 元ファイルの列名（空白のみは不可）。正規化データでも同じ名前の列になる。
        unit: 単位（任意。例: "V"、"N"）。単位の換算はしない（換算は v0.2 の単位変換で扱う）。
    """

    column: str = Field(pattern=NON_BLANK_PATTERN)
    unit: str | None = None


class ImportRequest(WorkbenchModel):
    """1 つのファイルを取り込むときの指定（取り込みダイアログでユーザーが選ぶ内容）。

    Attributes:
        file_format: ファイル形式。
        read_options: 読み込み条件（CSV / TSV で使う）。
        time_column: 時刻として使う列。
            時刻列がない場合は None とし、``sample_interval_s`` を指定する。
        time_unit: 時刻列の単位。秒に換算して正規化データに保存し、元の単位は TimeBase に残す。
        time_is_absolute: 時刻列が UNIX エポックからの値（絶対時刻）なら True。
        sample_interval_s: 時刻列がない場合のサンプル間隔 [s]。時刻列がある場合は指定しない。
        channels: 取り込むチャンネル（1 つ以上。列名の重複は不可）。
        experiment_name: 登録先の実験名。同じ名前があればそれを使い、なければ作る。
        session_name: 登録先のセッション名（実験の中で同じ扱い）。
        trial_name: 登録先の試技名（セッションの中で同じ扱い）。
        source_name: Source 名。同じ試技の中で一意でなければならない。
        shares_trial_clock: この試技の他の同期済み Source と同じ時計・同じ原点で記録した
            （時刻を直接比較できる）なら True。TimeBase.shares_trial_clock として記録する。
    """

    file_format: SourceFormat
    read_options: ReadOptions = Field(default_factory=ReadOptions)
    time_column: str | None
    time_unit: TimeUnit = TimeUnit.SECOND
    time_is_absolute: bool = False
    sample_interval_s: PositiveFloat | None = None
    channels: list[ChannelImportSpec] = Field(min_length=1)
    experiment_name: str = Field(pattern=NON_BLANK_PATTERN)
    session_name: str = Field(pattern=NON_BLANK_PATTERN)
    trial_name: str = Field(pattern=NON_BLANK_PATTERN)
    source_name: str = Field(pattern=NON_BLANK_PATTERN)
    shares_trial_clock: bool = False

    @model_validator(mode="after")
    def _check_columns(self) -> "ImportRequest":
        """時刻列・サンプル間隔・チャンネル列の組み合わせを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_import_columns(
            time_column=self.time_column,
            sample_interval_s=self.sample_interval_s,
            channel_columns=[channel.column for channel in self.channels],
            reserved_column=DATASET_TIME_COLUMN,
        )
        return self


class ColumnPreview(WorkbenchModel):
    """プレビューで分かった列の情報。

    Attributes:
        name: 列名。
        dtype: 先頭行から推定した型（表示用。Polars の型名）。
        is_numeric: 数値の列なら True。数値でない列はチャンネルとして取り込めない。
    """

    name: str
    dtype: str
    is_numeric: bool


class FilePreview(WorkbenchModel):
    """取り込み前のプレビュー（ファイルの先頭部分と、読み込み前に分かる異常）。

    Attributes:
        columns: 列の情報（ファイル内の順）。
        rows: 先頭数行の値（表示用の文字列。欠損は空文字）。
        issues: ヘッダー・文字コード・データ行の有無など、ファイル全体を読む前に分かる異常。
    """

    columns: list[ColumnPreview]
    rows: list[list[str]]
    issues: list[DataIssue]


class ImportSuggestion(WorkbenchModel):
    """ファイルの中身から自動で決めた取り込みの指定（ファイルを開くだけで始めるため）。

    自信を持って決められないときは ``request`` を None にして、理由を ``review_reasons`` に入れる。
    呼び出し側は取り込みダイアログで確認を求める（推測で取り込まない）。

    Attributes:
        file_format: 判定したファイル形式。
        read_options: 判定した読み込み条件（確認を求めるときのダイアログの初期値にも使う）。
        request: 取り込みの指定。確認が必要なら None。
        review_reasons: 利用者の確認が必要な理由。
        assumptions: 確認までは求めないが、利用者に知らせておく仮定（時刻の単位など）。
    """

    file_format: SourceFormat
    read_options: ReadOptions = Field(default_factory=ReadOptions)
    request: ImportRequest | None = None
    review_reasons: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ImportTargets(WorkbenchModel):
    """取り込み先の実験・セッション・試技（名前から解決した結果）。

    Attributes:
        experiment: 登録先の実験。
        session: 登録先のセッション。
        trial: 登録先の試技。
        new_entities: まだ DB にない（取り込みの確定時に登録する）エンティティ。親から子の順。
    """

    experiment: Experiment
    session: Session
    trial: Trial
    new_entities: list[Experiment | Session | Trial] = Field(default_factory=list)


class PreparedImport(WorkbenchModel):
    """ファイルの検証・コピー・正規化が終わり、DB への登録を待っている取り込み結果。

    Attributes:
        source: 登録する Source（警告は ``source.issues`` に入っている）。
        channels: 登録するチャンネル。
        written_relpaths: 書き込んだファイルの相対パス（登録に失敗したら削除する）。
    """

    source: Source
    channels: list[Channel]
    written_relpaths: list[str]


# ---------------------------------------------------------------------------
# 表示（SPEC F-07・F-10）
# ---------------------------------------------------------------------------


class ChannelData(WorkbenchModel):
    """表示や解析に渡す 1 チャンネル分のデータ（正規化データから読み込んだもの）。

    Attributes:
        channel: チャンネルのメタデータ。
        source: チャンネルが属する Source（時間軸の判定に使う）。
        time_s: 時刻 [s]（Source の時間軸のまま。他の Source と勝手に揃えない）。
        values: 値。欠損（空欄・NaN）は NaN。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    channel: Channel
    source: Source
    time_s: FloatArray
    values: FloatArray

    @model_validator(mode="after")
    def _check_arrays(self) -> "ChannelData":
        """時刻と値の配列の dtype と形を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_series_arrays(self.time_s, self.values)
        return self


class PlotPanelState(WorkbenchModel):
    """縦並べ表示の 1 パネルの状態。

    Attributes:
        series_ids: パネルに表示する系列の ID（表示順）。チャンネル ID、または解析の出力の ID。
        y_range: 縦軸の範囲 (最小, 最大)。None なら自動調整。
    """

    series_ids: list[UUID] = Field(default_factory=list)
    y_range: tuple[FiniteFloat, FiniteFloat] | None = None


class PlotView(WorkbenchModel):
    """縦並べ表示の状態（views/<name>.json に保存する）。

    Attributes:
        format_version: 保存形式のバージョン。
        name: 表示設定の名前（ファイル名に使うため英数字・"_"・"-" のみ）。
        panels: 上から順のパネル。
        x_range: 基準パネル（最初にデータを置いたパネル）の時間軸の範囲 [s]。None なら自動調整。
        cursor_time_s: 基準パネルのカーソル位置 [s]。None なら未設定。
    """

    format_version: PositiveInt = VIEW_FORMAT_VERSION
    name: str = Field(default="main", pattern=VIEW_NAME_PATTERN)
    panels: list[PlotPanelState] = Field(default_factory=list)
    x_range: tuple[FiniteFloat, FiniteFloat] | None = None
    cursor_time_s: FiniteFloat | None = None


# ---------------------------------------------------------------------------
# 入出力関係の解析（SPEC A-09〜A-11・Phase 4）
#
# 入力と出力の 2 チャンネルの関係を調べる解析の条件と結果。
# 2 チャンネルは、時刻の対応（同じ時計、または明示した時刻オフセット）を確かめたうえで
# 同じ時刻に対応付けて（AlignedPair）から解析する。
# ---------------------------------------------------------------------------


class InterpolationMethod(StrEnum):
    """時刻の対応付け（A-11）で、サンプルのない時刻の値を求める補間の方法。"""

    LINEAR = "linear"
    """線形補間（numpy.interp）。前後のサンプルを直線で結ぶ。"""
    PCHIP = "pchip"
    """形状を保つ区分 3 次エルミート補間（scipy.interpolate.PchipInterpolator）。

    滑らかで、元のデータにない行き過ぎ（オーバーシュート）を作らない。欠損（NaN）を含むデータには使えない。
    """


class AlignmentGrid(StrEnum):
    """2 チャンネルを対応付ける時刻の並び。"""

    INPUT = "input"
    """入力チャンネルの時刻（出力チャンネルを補間する）。"""
    OUTPUT = "output"
    """出力チャンネルの時刻（入力チャンネルを補間する）。"""
    UNIFORM = "uniform"
    """指定した周波数の等間隔の時刻（両方を補間する）。相互相関など、等間隔が必要な解析用。"""


class AlignmentParams(WorkbenchModel):
    """入力・出力の 2 チャンネルを同じ時刻に対応付ける条件（A-11）。

    時刻は入力チャンネルの時間軸で表す。出力チャンネルの時刻 t_out は次の式で入力の時間軸に移す::

        t = t_out + output_time_offset_s

    補間は 2 つのチャンネルの時刻が重なる範囲の中だけで行い（外挿しない）、時刻の間隔が
    中央値の ``max_gap_factor`` 倍を超える区間（欠損）はまたがない（その時刻の値は NaN にする）。
    2 つのチャンネルの時刻がまったく同じなら（同じ Source の列）、補間せずにそのまま対応付ける。

    Attributes:
        output_time_offset_s: 出力チャンネルの時刻に足す値 [s]。None なら、2 つのチャンネルの時刻が
            そのまま比較できる（同じ Source、または同じ試技で同期済みとして取り込んだ Source）ことを
            要求する。時計が違うチャンネルを対応付けるときは、ここに時刻の対応を明示する。
        method: 補間の方法。
        grid: 対応付ける時刻の並び。
        uniform_rate_hz: ``grid`` が UNIFORM のときのサンプリング周波数 [Hz]。それ以外は None。
        start_s: 解析する範囲の開始時刻（入力の時間軸）[s]。None なら重なりの先頭から。
        end_s: 解析する範囲の終了時刻 [s]。None なら重なりの末尾まで。
        max_gap_factor: 欠損とみなす時刻の間隔の、中央値に対する倍率（1 より大きい）。
    """

    output_time_offset_s: FiniteFloat | None = None
    method: InterpolationMethod = InterpolationMethod.LINEAR
    grid: AlignmentGrid = AlignmentGrid.INPUT
    uniform_rate_hz: PositiveFloat | None = None
    start_s: FiniteFloat | None = None
    end_s: FiniteFloat | None = None
    max_gap_factor: float = Field(default=DEFAULT_MAX_GAP_FACTOR, gt=1.0)

    @model_validator(mode="after")
    def _check_grid_and_range(self) -> "AlignmentParams":
        """等間隔の周波数の有無と、範囲の向きを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_alignment_settings(
            uniform=self.grid is AlignmentGrid.UNIFORM,
            uniform_rate_hz=self.uniform_rate_hz,
            start_s=self.start_s,
            end_s=self.end_s,
        )
        return self


class AlignmentRecord(WorkbenchModel):
    """2 チャンネルの対応付けで実際に行ったことの記録（結果と一緒に表示する）。

    Attributes:
        output_time_offset_s: 出力の時刻に足した値 [s]（時刻をそのまま比較できる場合は 0）。
        offset_declared: 時刻オフセットを指定して対応付けたなら True。
        shared_timestamps: 2 つのチャンネルの時刻が同じで、どちらも補間しなかったなら True。
        input_interpolated: 入力チャンネルを補間したなら True。
        output_interpolated: 出力チャンネルを補間したなら True。
        method: 補間の方法。
        start_s: 対応付けた範囲の開始時刻（入力の時間軸）[s]。
        end_s: 対応付けた範囲の終了時刻 [s]。
        n_samples: 対応付けた時刻の数。
        n_gap_samples: 欠損区間に入ったため値を求めなかった（NaN にした）時刻の数。
    """

    output_time_offset_s: float
    offset_declared: bool
    shared_timestamps: bool
    input_interpolated: bool
    output_interpolated: bool
    method: InterpolationMethod
    start_s: float
    end_s: float
    n_samples: NonNegativeInt
    n_gap_samples: NonNegativeInt


class AlignedPair(WorkbenchModel):
    """同じ時刻に対応付けた入力・出力の 2 チャンネル。

    Attributes:
        time_s: 時刻（入力の時間軸）[s]。
        input_values: 入力チャンネルの値。
        output_values: 出力チャンネルの値。
        record: 対応付けの記録。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time_s: FloatArray
    input_values: FloatArray
    output_values: FloatArray
    record: AlignmentRecord

    @model_validator(mode="after")
    def _check_arrays(self) -> "AlignedPair":
        """時刻と 2 つの値の配列の dtype と形を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_series_arrays(self.time_s, self.input_values)
        check_series_arrays(self.time_s, self.output_values)
        return self


class StepDirection(StrEnum):
    """ステップ状の入力の変化の向き。"""

    AUTO = "auto"
    """範囲の先頭（初期値）と末尾（最終値）の大小から決める。"""
    RISING = "rising"
    """増加（立ち上がり）。"""
    FALLING = "falling"
    """減少（立ち下がり）。"""


class OnsetThresholdMode(StrEnum):
    """入力の変化の開始を判定するしきい値の決め方。"""

    FRACTION = "fraction"
    """入力の変化量（最終値 - 初期値）に対する割合。"""
    ABSOLUTE = "absolute"
    """入力の初期値からの変化の大きさ（入力の単位）。"""


class StepResponseParams(WorkbenchModel):
    """ステップ状の入力に対する出力の応答指標の条件（A-09）。

    解析する範囲（AlignmentParams の start_s〜end_s）には、入力の変化が 1 回だけ入るようにする。
    範囲の長さのうち、先頭の ``baseline_fraction`` を変化前の値（初期値）、末尾の
    ``final_fraction`` を変化後の値（最終値）を平均で求める区間とする。入力の変化の開始は、
    初期値の区間より後で、入力がしきい値を最初に越えた時刻とする（サンプルの間は線形補間）。
    出力の変化の開始は、入力と同じ割合（入力のしきい値 ÷ 入力の変化量）で判定する。

    Attributes:
        direction: 入力の変化の向き。
        onset_mode: 入力の変化の開始を判定するしきい値の決め方。
        onset_threshold: しきい値。FRACTION なら 0〜1 の割合、ABSOLUTE なら入力の単位の大きさ。
        baseline_fraction: 範囲の長さのうち、先頭から初期値を求める区間の割合（0〜0.5）。
        final_fraction: 範囲の長さのうち、末尾から最終値を求める区間の割合（0〜0.5）。
        settling_band: 整定とみなす最終値からのずれの幅（出力の変化量に対する割合）。
        smoothing_cutoff_hz: 指標を求める前に入力と出力にかけるゼロ位相ローパスフィルタ
            （4 次のバターワース）のカットオフ周波数 [Hz]。None なら平滑化しない。
    """

    direction: StepDirection = StepDirection.AUTO
    onset_mode: OnsetThresholdMode = OnsetThresholdMode.FRACTION
    onset_threshold: PositiveFloat = 0.1
    baseline_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    final_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    settling_band: float = Field(default=0.05, gt=0.0, lt=1.0)
    smoothing_cutoff_hz: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_threshold(self) -> "StepResponseParams":
        """割合のしきい値が 1 未満であることを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_onset_threshold(
            fraction_mode=self.onset_mode is OnsetThresholdMode.FRACTION,
            threshold=self.onset_threshold,
        )
        return self


class StepResponseResult(WorkbenchModel):
    """ステップ応答の指標（A-09）。時間の指標は、入力の変化の開始からの時間 [s]。

    出力を正規化した応答 r(t) = (y(t) - y0) / (y1 - y0)（初期値 y0 から最終値 y1 へ、
    0 から 1 に向かう）について::

        応答開始の遅れ    : r が入力と同じ割合のしきい値を最初に越えるまで
        遅れ時間 Td       : r = 0.5 に達するまで
        立ち上がり時間 Tr : r = 0.1 から 0.9 に達するまで
        時定数 τ          : 1.5 × (t(0.632) - t(0.283))  （一次遅れ＋むだ時間で近似する 2 点法）
        むだ時間 L        : t(0.632) - τ
        整定時間 Ts       : |r - 1| <= settling_band を満たし続けるようになるまで
        オーバーシュート  : max(r) - 1（負なら 0）

    2 点法は、一次遅れ＋むだ時間の応答なら τ と L を厳密に与える（Smith の方法）。
    求められなかった指標は None にし、理由を ``notes`` に書く。

    Attributes:
        direction: 判定した入力の変化の向き（RISING または FALLING）。
        input_onset_s: 入力の変化の開始時刻（入力の時間軸）[s]。
        input_initial: 入力の初期値。
        input_final: 入力の最終値。
        output_initial: 出力の初期値。
        output_final: 出力の最終値。
        gain: 出力の変化量 ÷ 入力の変化量。
        response_delay_s: 応答開始の遅れ [s]。
        delay_time_s: 遅れ時間 Td [s]。
        rise_time_s: 立ち上がり時間 Tr [s]。
        time_constant_s: 時定数 τ [s]。
        dead_time_s: むだ時間 L [s]。
        settling_time_s: 整定時間 Ts [s]。
        overshoot: オーバーシュート（出力の変化量に対する割合）。
        response_time_s: 各サンプルの、入力の変化の開始からの時間 [s]（表示用）。
        normalized_input: 正規化した入力（表示用）。
        normalized_output: 正規化した出力 r（表示用）。出力が変化していない場合は NaN。
        notes: 求められなかった指標の理由などの注記。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: StepDirection
    input_onset_s: float
    input_initial: float
    input_final: float
    output_initial: float
    output_final: float
    gain: float
    response_delay_s: float | None = None
    delay_time_s: float | None = None
    rise_time_s: float | None = None
    time_constant_s: float | None = None
    dead_time_s: float | None = None
    settling_time_s: float | None = None
    overshoot: float | None = None
    response_time_s: FloatArray
    normalized_input: FloatArray
    normalized_output: FloatArray
    notes: list[str] = Field(default_factory=list)


class HysteresisParams(WorkbenchModel):
    """入出力特性とヒステリシスの条件（A-10）。

    入力の変化の速さ du/dt の符号で、各サンプルを「増加時」と「減少時」に分ける。
    次の条件を満たす、向きが定まらないサンプル（折り返しや静止）は、どちらの分岐にも入れない::

        |du/dt| < min_rate_fraction × (|du/dt| の 95 パーセンタイル)

    基準に最大値ではなく 95 パーセンタイルを使うのは、雑音による一瞬の大きな値に左右されないため。
    入力の範囲を ``n_bins`` 個の区間に分け、区間ごとに、増加時・減少時それぞれのサンプルに直線を
    当てはめて、区間の中央の入力での出力の値（分岐の曲線）を求める（3 サンプル以上ある区間だけ）。
    平均ではなく直線の当てはめを使うのは、2 つの分岐を同じ入力の値どうしで比べるため。

    Attributes:
        n_bins: 入力の範囲を分ける区間の数。
        min_rate_fraction: 向きが定まらないとして分岐から除く、変化の速さの割合。
        smoothing_cutoff_hz: 変化の向きの判定にだけ使う、入力のゼロ位相ローパスフィルタ
            （4 次のバターワース）のカットオフ周波数 [Hz]。None なら平滑化しない。
            X-Y の値と指標の計算には、平滑化しない値を使う。
            微分は雑音を強調するので、雑音のある入力で平滑化しないと向きの判定が乱れて、
            ヒステリシスが小さく見積もられる（HysteresisResult.direction_unstable で分かる）。
    """

    n_bins: int = Field(default=50, ge=2, le=10_000)
    min_rate_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    smoothing_cutoff_hz: PositiveFloat | None = None


class HysteresisResult(WorkbenchModel):
    """入出力特性とヒステリシスの指標（A-10）。

    出力のフルスケール FS は、回帰直線の上で入力が最小値から最大値まで動いたときの出力の変化
    |a| × (u_max - u_min) とする（a は回帰直線の傾き）::

        直線性誤差   : 分岐の中央の曲線と回帰直線の差の最大値。中央の曲線は、区間ごとの増加時と
                       減少時の分岐の値の中点（片方の分岐しかない区間は、全サンプルから求めた値）
        ヒステリシス : 同じ入力の値での分岐の差 d = y_inc - y_dec。最大値は |d| の最大
        ループ面積   : |Σ d Δu|（両方の分岐がある区間の和。分岐が囲む面積）

    Attributes:
        slope: 回帰直線の傾き（感度）。
        intercept: 回帰直線の切片（オフセット）。
        r_squared: 決定係数。出力がすべて同じ値なら None。
        input_min: 入力の最小値。
        input_max: 入力の最大値。
        full_scale_output: FS。傾きが 0 なら None。
        linearity_error: 直線性誤差（出力の単位）。求められなければ None。
        linearity_ratio: 直線性誤差 ÷ FS。
        max_hysteresis: 最大ヒステリシス（出力の単位）。両方の分岐がある区間がなければ None。
        max_hysteresis_input: 最大ヒステリシスの区間の中央の入力の値。
        hysteresis_ratio: 最大ヒステリシス ÷ FS。
        mean_branch_difference: d の平均（正なら増加時の分岐が上にある）。
        loop_area: ループ面積（入力の単位 × 出力の単位）。
        n_increasing: 増加時のサンプル数。
        n_decreasing: 減少時のサンプル数。
        n_excluded: 向きが定まらず除いたサンプル数。
        bin_centers: 区間の中央の入力の値（表示用）。
        increasing_branch: 区間の中央の入力での、増加時の出力の値（サンプルが足りない区間は NaN）。
        decreasing_branch: 区間の中央の入力での、減少時の出力の値（同上）。
        increasing_mask: 各サンプルが増加時なら True（表示用）。
        decreasing_mask: 各サンプルが減少時なら True（表示用）。
        direction_unstable: 入力の変化の向きの判定が雑音で乱れていれば True。このとき増加時と
            減少時のサンプルが混ざるので、ヒステリシスは小さく見積もられる。
        notes: 注記。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    slope: float
    intercept: float
    r_squared: float | None
    input_min: float
    input_max: float
    full_scale_output: float | None
    linearity_error: float | None
    linearity_ratio: float | None
    max_hysteresis: float | None
    max_hysteresis_input: float | None
    hysteresis_ratio: float | None
    mean_branch_difference: float | None
    loop_area: float | None
    n_increasing: NonNegativeInt
    n_decreasing: NonNegativeInt
    n_excluded: NonNegativeInt
    bin_centers: FloatArray
    increasing_branch: FloatArray
    decreasing_branch: FloatArray
    increasing_mask: BoolArray
    decreasing_mask: BoolArray
    direction_unstable: bool = False
    notes: list[str] = Field(default_factory=list)


class DetrendMode(StrEnum):
    """相互相関を求める前に取り除く傾向（scipy.signal.detrend の type）。"""

    CONSTANT = "constant"
    """平均を引く。"""
    LINEAR = "linear"
    """最小二乗で当てはめた直線を引く（ゆっくりしたドリフトが相関を支配しないように）。"""


class CrossCorrelationParams(WorkbenchModel):
    """相互相関による遅れの推定の条件（A-09）。

    出力が入力より遅れる向きを正とする（出力 y(t) ≈ 入力 x(t - τ) のとき遅れ τ > 0）。
    遅れごとに、重なっている部分どうしの相関係数を求め、探索範囲の中で絶対値が最大になる遅れを
    選ぶ。前後の 3 点に放物線を当てはめて、サンプル間隔より細かく求める。
    重なりがサンプル数の半分未満になる遅れは、信頼できないので探さない。
    等間隔の時刻が必要（AlignmentParams の grid を UNIFORM にするか、等間隔のチャンネルを使う）。

    Attributes:
        min_lag_s: 遅れを探す範囲の下限 [s]（負にすると、出力が先行する場合も探す）。
        max_lag_s: 遅れを探す範囲の上限 [s]。
        detrend: 相関を求める前に取り除く傾向。
        use_derivative: True なら、値の代わりに変化の速さ（時間微分）の相関を取る。
            ステップ状の信号では、値の相関のピークはなだらかだが、変化の速さの相関のピークは鋭くなる。
    """

    min_lag_s: FiniteFloat
    max_lag_s: FiniteFloat
    detrend: DetrendMode = DetrendMode.CONSTANT
    use_derivative: bool = False

    @model_validator(mode="after")
    def _check_lags(self) -> "CrossCorrelationParams":
        """探索範囲の下限が上限より小さいことを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_lag_range(self.min_lag_s, self.max_lag_s)
        return self


class CrossCorrelationResult(WorkbenchModel):
    """相互相関による遅れの推定結果（A-09）。

    Attributes:
        delay_s: 推定した遅れ [s]（正なら出力が遅れる）。
        peak_coefficient: その遅れでの相関係数（-1〜1。負なら出力は入力と逆向きに変化する）。
        sample_interval_s: サンプル間隔 [s]（遅れの細かさの目安）。
        at_search_edge: 相関の絶対値が最大になったのが探索範囲の端なら True
            （真の遅れは探索範囲の外にあるかもしれない）。
        lags_s: 探した遅れ [s]（表示用）。
        coefficients: 遅れごとの相関係数（表示用）。
        notes: 注記。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delay_s: float
    peak_coefficient: float
    sample_interval_s: PositiveFloat
    at_search_edge: bool
    lags_s: FloatArray
    coefficients: FloatArray
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 保存できる解析と、その表示（SPEC F-11・Phase 3）
# ---------------------------------------------------------------------------


class PolynomialObservationParams(WorkbenchModel):
    """EKF の組み込みモデル「運動学モデル + 多項式の観測」の条件。

    状態は運動学モデル（KinematicSystemParams と同じ。状態 x とその時間微分）で、
    観測は状態の先頭要素 x の多項式とする::

        z = c0 + c1 x + c2 x² + ... + v,   v ~ N(0, σ²)

    センサの出力が、物理量の非線形な関数として分かっている場合に、
    物理量とその変化率を推定するのに使う。観測が非線形なので EKF になる。

    Attributes:
        derivative_order: 状態に含める x の微分の最高階数（1 または 2）。
        process_noise_density: x の最高階の微分を駆動する白色雑音のパワースペクトル密度 q。
        measurement_noise_std: 観測（センサの出力）の雑音の標準偏差。
        coefficients: 多項式の係数 [c0, c1, c2, ...]（昇べきの順、2 つ以上）。
    """

    derivative_order: int = Field(default=1, ge=1, le=2)
    process_noise_density: PositiveFloat
    measurement_noise_std: PositiveFloat
    coefficients: list[FiniteFloat] = Field(min_length=2)


class KalmanModelKind(StrEnum):
    """GUI などから選べるカルマンフィルタの状態空間モデル（値は KalmanSettings の項目名と同じ）。"""

    KINEMATIC = "kinematic"
    """運動学モデル（値とその時間微分。線形 KF）。"""
    LINEAR = "linear"
    """定数行列の線形モデル（線形 KF）。"""
    POLYNOMIAL_OBSERVATION = "polynomial_observation"
    """運動学モデル + 多項式の観測（EKF の組み込みモデル）。"""


class KalmanSettings(WorkbenchModel):
    """保存できるカルマンフィルタの条件（モデル・初期状態・適応推定）。

    ``model_kind`` で選んだモデルの条件だけを指定する（使わない条件は None）。

    Attributes:
        model_kind: 状態空間モデルの種類。
        kinematic: 運動学モデルの条件。
        linear: 線形モデルの条件。
        polynomial_observation: EKF の組み込みモデルの条件。
        initial: 初期状態（モデルの状態の次元と合っていること。実行時に確認する）。
        adaptive: 適応推定（観測雑音 R の逐次推定）の条件。None なら R は一定。
    """

    model_kind: KalmanModelKind
    kinematic: KinematicSystemParams | None = None
    linear: LinearSystemParams | None = None
    polynomial_observation: PolynomialObservationParams | None = None
    initial: KalmanInitialState
    adaptive: AdaptiveNoiseParams | None = None

    @model_validator(mode="after")
    def _check_model(self) -> "KalmanSettings":
        """選んだモデルの条件だけが指定されているかを確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        provided = {
            KalmanModelKind.KINEMATIC: self.kinematic,
            KalmanModelKind.LINEAR: self.linear,
            KalmanModelKind.POLYNOMIAL_OBSERVATION: self.polynomial_observation,
        }
        check_selected_settings(
            selected=self.model_kind.value,
            provided=[kind.value for kind, value in provided.items() if value is not None],
        )
        return self


class OffsetMode(StrEnum):
    """オフセット除去で差し引く基準値の決め方。"""

    MEAN = "mean"
    """区間の平均値（区間を指定しなければ全区間）。"""
    VALUE = "value"
    """指定した値。"""


class OffsetParams(WorkbenchModel):
    """オフセット除去の条件（A-15）。

    Attributes:
        mode: 基準値の決め方。
        value: 指定した基準値（``mode`` が VALUE のとき）。
        start_s: 平均をとる区間の開始 [s]。None なら先頭から。
        end_s: 平均をとる区間の終了 [s]。None なら末尾まで。
    """

    mode: OffsetMode = OffsetMode.MEAN
    value: FiniteFloat = 0.0
    start_s: FiniteFloat | None = None
    end_s: FiniteFloat | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "OffsetParams":
        """区間の開始が終了より前であることを確認する。

        Returns:
            検証済みの自身。
        """
        check_time_range(self.start_s, self.end_s)
        return self


class ScaleParams(WorkbenchModel):
    """線形変換 y = gain · x + offset の条件（A-15。倍率の補正・単位換算）。

    Attributes:
        gain: 倍率（0 にすると値の情報が失われるので指定できない）。
        offset: 切片。
        unit: 変換後の単位。None なら、恒等変換のときだけ入力の単位を引き継ぐ。
    """

    gain: FiniteFloat = 1.0
    offset: FiniteFloat = 0.0
    unit: str | None = None

    @field_validator("gain")
    @classmethod
    def _check_gain(cls, value: float) -> float:
        """倍率が 0 でないことを確認する。

        Args:
            value: 倍率。

        Returns:
            入力と同じ値。

        Raises:
            ValueError: 倍率が 0 の場合。
        """
        if value == 0.0:
            raise ValueError("倍率に 0 は指定できません")
        return value


class DetrendParams(WorkbenchModel):
    """トレンド除去の条件（A-15）。

    Attributes:
        order: 最小二乗法で当てはめる多項式の次数（0: 平均値、1: 直線）。
    """

    order: int = Field(default=1, ge=0, le=10)


class AbsoluteParams(WorkbenchModel):
    """絶対値の条件（A-15。条件はないが、他の処理と同じ形で保存する）。"""


class NormalizeMode(StrEnum):
    """正規化の方法。"""

    Z_SCORE = "z_score"
    """平均を引いて標準偏差で割る（平均 0、標準偏差 1）。"""
    MIN_MAX = "min_max"
    """最小値を 0、最大値を 1 にする。"""
    PEAK = "peak"
    """絶対値の最大で割る（ピークを ±1 にする）。"""


class NormalizeParams(WorkbenchModel):
    """正規化の条件（A-15）。

    Attributes:
        mode: 正規化の方法。
    """

    mode: NormalizeMode = NormalizeMode.Z_SCORE


class CropParams(WorkbenchModel):
    """区間の切り出しの条件（A-16）。

    Attributes:
        start_s: 開始 [s]。
        end_s: 終了 [s]。
    """

    start_s: FiniteFloat
    end_s: FiniteFloat

    @model_validator(mode="after")
    def _check_range(self) -> "CropParams":
        """区間の開始が終了より前であることを確認する。

        Returns:
            検証済みの自身。
        """
        check_time_range(self.start_s, self.end_s)
        return self


class ResampleParams(WorkbenchModel):
    """リサンプリングの条件（A-16）。

    Attributes:
        rate_hz: 変換後のサンプリング周波数 [Hz]。
        method: 補間方法。
        anti_alias: 周波数を下げるときにアンチエイリアスフィルタをかけるなら True。
        max_gap_factor: 欠損区間とみなす時刻の間隔の、間隔の中央値に対する倍率。
    """

    rate_hz: PositiveFloat
    method: InterpolationMethod = InterpolationMethod.LINEAR
    anti_alias: bool = True
    max_gap_factor: float = Field(default=DEFAULT_MAX_GAP_FACTOR, gt=1.0)


class ResampleResult(WorkbenchModel):
    """リサンプリングの結果。

    Attributes:
        time_s: 等間隔の時刻 [s]。
        values: リサンプリングした値（欠損区間は NaN）。
        source_rate_hz: 元のサンプリング周波数 [Hz]。
        anti_aliased: アンチエイリアスフィルタをかけたなら True。
        notes: 注記。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time_s: FloatArray
    values: FloatArray
    source_rate_hz: float
    anti_aliased: bool
    notes: list[str] = Field(default_factory=list)


class FillGapsParams(WorkbenchModel):
    """欠損の補間の条件（A-16）。

    Attributes:
        method: 補間方法。
        max_gap_s: 補間する欠損の長さの上限 [s]（前後の有効な値の時刻の差）。
            None なら上限なし。
    """

    method: InterpolationMethod = InterpolationMethod.LINEAR
    max_gap_s: PositiveFloat | None = None


class GapFillResult(WorkbenchModel):
    """欠損の補間の結果。

    Attributes:
        values: 補間後の値（埋めなかった欠損は NaN のまま）。
        filled_mask: 補間で埋めたサンプルなら True。
        n_missing: 元の欠損（NaN）の数。
        n_filled: 埋めた欠損の数。
        n_time_gaps: 時刻が飛んでいる箇所の数（行ごと欠けた区間。NaN とは別に数える）。
        longest_time_gap_s: 最も長い時刻の飛び [s]。飛びがなければ None。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: FloatArray
    filled_mask: BoolArray
    n_missing: NonNegativeInt
    n_filled: NonNegativeInt
    n_time_gaps: NonNegativeInt = 0
    longest_time_gap_s: FiniteFloat | None = None


class StatisticsParams(WorkbenchModel):
    """区間統計の条件（A-17）。

    Attributes:
        start_s: 区間の開始 [s]。None なら先頭から。
        end_s: 区間の終了 [s]。None なら末尾まで。
        n_bins: ヒストグラムの階級数。
    """

    start_s: FiniteFloat | None = None
    end_s: FiniteFloat | None = None
    n_bins: int = Field(default=50, ge=2, le=10_000)

    @model_validator(mode="after")
    def _check_range(self) -> "StatisticsParams":
        """区間の開始が終了より前であることを確認する。

        Returns:
            検証済みの自身。
        """
        check_time_range(self.start_s, self.end_s)
        return self


class SectionStatistics(WorkbenchModel):
    """区間の統計量。

    Attributes:
        start_s: 区間の最初のサンプルの時刻 [s]。
        end_s: 区間の最後のサンプルの時刻 [s]。
        n_samples: 計算に使ったサンプル数（欠損を除く）。
        n_missing: 区間の中の欠損の数。
        mean: 平均値。
        std: 標準偏差（自由度 n - 1）。
        rms: 実効値（二乗平均平方根）。
        minimum: 最小値。
        maximum: 最大値。
        median: 中央値。
        minimum_time_s: 最小値の時刻 [s]。
        maximum_time_s: 最大値の時刻 [s]。
        histogram_centers: ヒストグラムの階級の中央の値。
        histogram_counts: ヒストグラムの度数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_s: float
    end_s: float
    n_samples: NonNegativeInt
    n_missing: NonNegativeInt
    mean: float
    std: float
    rms: float
    minimum: float
    maximum: float
    median: float
    minimum_time_s: float
    maximum_time_s: float
    histogram_centers: FloatArray
    histogram_counts: FloatArray


class AnalysisKind(StrEnum):
    """保存できる解析の種類（値は AnalysisSpec の項目名と同じ）。"""

    BUTTERWORTH = "butterworth"
    MOVING_AVERAGE = "moving_average"
    SAVITZKY_GOLAY = "savitzky_golay"
    DERIVATIVE = "derivative"
    INTEGRAL = "integral"
    KALMAN = "kalman"
    STEP_RESPONSE = "step_response"
    """ステップ応答の指標（入力と出力の 2 チャンネル。Phase 4）。"""
    HYSTERESIS = "hysteresis"
    """入出力特性とヒステリシス（2 チャンネル。Phase 4）。"""
    CROSS_CORRELATION = "cross_correlation"
    """相互相関による遅れの推定（2 チャンネル。Phase 4）。"""
    OFFSET = "offset"
    """オフセット除去。"""
    SCALE = "scale"
    """線形変換（倍率の補正・単位換算）。"""
    DETREND = "detrend"
    """トレンド除去。"""
    ABSOLUTE = "absolute"
    """絶対値。"""
    NORMALIZE = "normalize"
    """正規化。"""
    CROP = "crop"
    """区間の切り出し。"""
    RESAMPLE = "resample"
    """等間隔へのリサンプリング。"""
    FILL_GAPS = "fill_gaps"
    """欠損の補間。"""
    STATISTICS = "statistics"
    """区間統計。"""


TWO_CHANNEL_ANALYSIS_KINDS = frozenset(
    {AnalysisKind.STEP_RESPONSE, AnalysisKind.HYSTERESIS, AnalysisKind.CROSS_CORRELATION}
)
"""入力と出力の 2 チャンネルを使う解析の種類（output_channel_id と alignment が要る）。"""

METRIC_ANALYSIS_KINDS = TWO_CHANNEL_ANALYSIS_KINDS | {AnalysisKind.STATISTICS}
"""時系列を出力せず、指標とグラフを返す解析の種類（解析結果の欄に表示する）。"""

RANGE_ANALYSIS_KINDS = METRIC_ANALYSIS_KINDS | {AnalysisKind.CROP, AnalysisKind.OFFSET}
"""グラフの上で時間の範囲を選べる解析の種類。"""


class AnalysisSpec(Entity):
    """保存できる 1 段の解析。

    1 つの入力チャンネルに 1 つの処理をかける解析（フィルタ・カルマンフィルタなど）と、
    入力と出力の 2 チャンネルの関係を調べる解析（ステップ応答・ヒステリシス・相互相関。
    TWO_CHANNEL_ANALYSIS_KINDS）がある。後者だけが ``output_channel_id`` と ``alignment`` を持つ。
    pipelines/<ID>.json に保存し、同じ条件でいつでも再計算できるようにする。
    ``kind`` で選んだ種類の条件だけを指定する（使わない条件は None）。
    複数段の処理（パイプライン）と複数試技への一括適用は v0.2 で扱う。

    Attributes:
        format_version: 保存形式のバージョン。
        name: 解析の名前（表示用）。
        input_channel_id: 入力チャンネルの ID。
        kind: 解析の種類。
        output_channel_id: 2 チャンネルの解析の出力チャンネルの ID（それ以外は None）。
        alignment: 2 チャンネルを同じ時刻に対応付ける条件（2 チャンネルの解析だけ）。
        butterworth: バターワースフィルタの条件。
        moving_average: 移動平均の条件。
        savitzky_golay: Savitzky-Golay 平滑化の条件。
        derivative: 微分の条件。
        integral: 積分の条件。
        kalman: カルマンフィルタ系の条件。
        step_response: ステップ応答の条件。
        hysteresis: 入出力特性とヒステリシスの条件。
        cross_correlation: 相互相関の条件。
        offset: オフセット除去の条件。
        scale: 線形変換の条件。
        detrend: トレンド除去の条件。
        absolute: 絶対値の条件。
        normalize: 正規化の条件。
        crop: 区間の切り出しの条件。
        resample: リサンプリングの条件。
        fill_gaps: 欠損の補間の条件。
        statistics: 区間統計の条件。
    """

    format_version: PositiveInt = ANALYSIS_FORMAT_VERSION
    name: str = Field(pattern=NON_BLANK_PATTERN)
    input_channel_id: UUID
    kind: AnalysisKind
    output_channel_id: UUID | None = None
    alignment: AlignmentParams | None = None
    butterworth: ButterworthParams | None = None
    moving_average: MovingAverageParams | None = None
    savitzky_golay: SavitzkyGolayParams | None = None
    derivative: DerivativeParams | None = None
    integral: IntegralParams | None = None
    kalman: KalmanSettings | None = None
    step_response: StepResponseParams | None = None
    hysteresis: HysteresisParams | None = None
    cross_correlation: CrossCorrelationParams | None = None
    offset: OffsetParams | None = None
    scale: ScaleParams | None = None
    detrend: DetrendParams | None = None
    absolute: AbsoluteParams | None = None
    normalize: NormalizeParams | None = None
    crop: CropParams | None = None
    resample: ResampleParams | None = None
    fill_gaps: FillGapsParams | None = None
    statistics: StatisticsParams | None = None

    @model_validator(mode="after")
    def _check_params(self) -> "AnalysisSpec":
        """選んだ種類の条件だけが指定され、2 チャンネルの指定が種類と合うかを確認する。

        Returns:
            検証済みの自身。
        """
        provided = {
            AnalysisKind.BUTTERWORTH: self.butterworth,
            AnalysisKind.MOVING_AVERAGE: self.moving_average,
            AnalysisKind.SAVITZKY_GOLAY: self.savitzky_golay,
            AnalysisKind.DERIVATIVE: self.derivative,
            AnalysisKind.INTEGRAL: self.integral,
            AnalysisKind.KALMAN: self.kalman,
            AnalysisKind.STEP_RESPONSE: self.step_response,
            AnalysisKind.HYSTERESIS: self.hysteresis,
            AnalysisKind.CROSS_CORRELATION: self.cross_correlation,
            AnalysisKind.OFFSET: self.offset,
            AnalysisKind.SCALE: self.scale,
            AnalysisKind.DETREND: self.detrend,
            AnalysisKind.ABSOLUTE: self.absolute,
            AnalysisKind.NORMALIZE: self.normalize,
            AnalysisKind.CROP: self.crop,
            AnalysisKind.RESAMPLE: self.resample,
            AnalysisKind.FILL_GAPS: self.fill_gaps,
            AnalysisKind.STATISTICS: self.statistics,
        }
        check_selected_settings(
            selected=self.kind.value,
            provided=[kind.value for kind, value in provided.items() if value is not None],
        )
        check_two_channel_settings(
            two_channel=self.kind in TWO_CHANNEL_ANALYSIS_KINDS,
            has_output_channel=self.output_channel_id is not None,
            has_alignment=self.alignment is not None,
            same_channel=self.output_channel_id == self.input_channel_id,
        )
        return self


class AnalysisOutputInfo(WorkbenchModel):
    """解析の出力 1 つの情報（計算しなくても分かる、ID・名前・単位）。

    Attributes:
        series_id: 出力の ID（解析の ID と ``key`` から決まり、再計算しても変わらない）。
        key: 解析の中での出力の名前（例: "output"、"state_0"、"innovation"）。
        label: 表示名。
        unit: 単位。分からなければ None。
    """

    series_id: UUID
    key: str
    label: str
    unit: str | None = None


class PlotSeries(WorkbenchModel):
    """縦並べ表示に置く 1 系列（生データのチャンネル、または解析の出力）。

    Attributes:
        series_id: 系列の ID（チャンネル ID、または解析の出力の ID）。
        label: 凡例やカーソル表示に使う名前。
        channel_name: チャンネル名（解析の出力は入力チャンネルの名前）。図の見出しと、
            凡例の短い名前（"生データ"、"LPF 10 Hz" など）を作るのに使う。
        unit: 単位。分からなければ None。
        source: 時間軸を決める Source（解析の出力は入力チャンネルの Source）。
        time_s: 時刻 [s]。
        values: 値。欠損は NaN。
        band_lower: 帯の下端（カルマンフィルタの推定値 - σ など）。帯がなければ None。
        band_upper: 帯の上端。``band_lower`` と組で指定する。
        marker_times_s: 印を付ける時刻（観測更新しなかった時刻など）。なければ None。
        analysis_id: 解析の出力なら解析の ID。生データなら None。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    series_id: UUID
    label: str
    channel_name: str = ""
    unit: str | None = None
    source: Source
    time_s: FloatArray
    values: FloatArray
    band_lower: FloatArray | None = None
    band_upper: FloatArray | None = None
    marker_times_s: FloatArray | None = None
    analysis_id: UUID | None = None

    @model_validator(mode="after")
    def _check_arrays(self) -> "PlotSeries":
        """配列の dtype・形の整合を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_plot_series_arrays(
            self.time_s, self.values, self.band_lower, self.band_upper, self.marker_times_s
        )
        return self


class MetricDisplay(StrEnum):
    """解析結果の表で、指標の値をどう書くか。"""

    QUANTITY = "quantity"
    """物理量（有効数字で書き、1 秒未満の時間はミリ秒で書く）。"""
    INSTANT = "instant"
    """時間軸の上の時刻（長さではないのでミリ秒にせず、秒のまま小数で書く）。"""
    COUNT = "count"
    """個数（整数で書く）。"""


class AnalysisMetric(WorkbenchModel):
    """解析の結果の指標 1 つ（解析結果の表に表示する）。

    Attributes:
        key: 指標の識別名（例: "delay_time_s"）。テストや出力で指標を探すのに使う。
        label: 表示名。
        value: 値。求められなかった場合は None。
        unit: 単位。単位がない、または分からない場合は None。
        note: 指標の定義や、求められなかった理由。
        display: 値の書き方。
    """

    key: str
    label: str
    value: FiniteFloat | None = None
    unit: str | None = None
    note: str = ""
    display: MetricDisplay = MetricDisplay.QUANTITY


class ChartStyle(StrEnum):
    """解析結果のグラフで、系列をどう描くか。"""

    LINE = "line"
    """実線。"""
    DASHED = "dashed"
    """破線（回帰直線や基準の線）。"""
    POINTS = "points"
    """小さな点（散布図）。"""
    MARKERS = "markers"
    """目立つ印（ピークなど、少数の点）。"""
    BARS = "bars"
    """棒（ヒストグラム。X は等間隔の階級の中心で、棒の幅は階級の幅）。"""


class ChartSeries(WorkbenchModel):
    """解析結果のグラフの 1 系列（X と Y の組）。

    Attributes:
        label: 凡例に出す名前。
        x: X 座標（float64 の 1 次元配列）。
        y: Y 座標（x と同じ長さ）。
        style: 描き方。
        color: 色（"#rrggbb"）。None なら表示側が順番に割り当てる。
            増加時・減少時の分岐のように、意味で色を決めたいときに指定する。
        show_in_legend: 凡例に出すなら True。基準線や背景の点は出さない
            （凡例の項目が多いと、読み取ってほしい線が分からなくなるため）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    x: FloatArray
    y: FloatArray
    style: ChartStyle = ChartStyle.LINE
    color: str | None = Field(default=None, pattern=COLOR_PATTERN)
    show_in_legend: bool = True

    @model_validator(mode="after")
    def _check_arrays(self) -> "ChartSeries":
        """X と Y の配列の dtype と形を確認する（validation に委ねる）。

        Returns:
            検証済みの自身。
        """
        check_xy_arrays(self.x, self.y)
        return self


class ChartAnnotation(WorkbenchModel):
    """解析結果のグラフに直接書き込む注記（X の位置の縦線と文字）。

    凡例と別の表を見比べなくても、図の上で「どこが何か」が分かるようにするため。

    文字が読み取ってほしい線に重ならないよう、置く場所はグラフごとに指定する。

    Attributes:
        x: 縦線の X 座標。
        text: 添える文字。
        height: 文字を置く高さ（0 が図の下端、1 が上端）。
        on_left: 文字を縦線の左に置くなら True。
    """

    x: FiniteFloat
    text: str
    height: float = Field(default=0.92, ge=0.0, le=1.0)
    on_left: bool = False


class AnalysisChart(WorkbenchModel):
    """解析結果のグラフ 1 つ（X-Y 表示や相関の曲線。時系列の縦並べとは別に表示する）。

    Attributes:
        title: グラフの題名（何の図かが分かる、普通の言葉で書く）。
        x_label: X 軸の名前（単位を含む）。
        y_label: Y 軸の名前（単位を含む）。
        series: 系列。
        annotations: 図に書き込む注記。
        log_x: X 軸を対数にするなら True（周波数解析で使う）。
        log_y: Y 軸を対数にするなら True。
    """

    title: str
    x_label: str
    y_label: str
    series: list[ChartSeries]
    annotations: list[ChartAnnotation] = Field(default_factory=list)
    log_x: bool = False
    log_y: bool = False


class TimeMarker(WorkbenchModel):
    """縦並べ表示に引く時刻の印（ステップ応答の、入力の変化の開始など）。

    Attributes:
        time_s: 時刻（解析の入力チャンネルの時間軸）[s]。
        label: 印に添える名前。
    """

    time_s: FiniteFloat
    label: str


class AnalysisResult(WorkbenchModel):
    """解析を実行した結果（表示用の系列・指標・グラフ・時刻の印と、結果の要約）。

    1 チャンネルの処理（フィルタ・カルマンフィルタなど）は ``series`` に時系列を返す。
    2 チャンネルの解析（ステップ応答など）は時系列を返さず、``metrics``・``charts``・
    ``time_markers`` を返す。

    Attributes:
        analysis_id: 解析の ID。
        series: 出力の系列（AnalysisOutputInfo と同じ順）。
        headline: 結果を数値入りの一文で述べたもの（図を見て何が起きたかが分かるように）。
        summary: 結果の補足。読み手に必ず見せる内容（推定した R の最終値、補間や時刻オフセット
            など結果に関わる変換、注意点）。
        conditions: 計算の条件の記録（解析した範囲、サンプル数など）。普段は畳んで表示する。
        metrics: 指標（解析結果の表に表示する）。
        charts: グラフ（X-Y 表示など）。
        time_markers: 縦並べ表示に引く時刻の印（入力チャンネルの時間軸）。
    """

    analysis_id: UUID
    series: list[PlotSeries]
    headline: str = ""
    summary: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    metrics: list[AnalysisMetric] = Field(default_factory=list)
    charts: list[AnalysisChart] = Field(default_factory=list)
    time_markers: list[TimeMarker] = Field(default_factory=list)
