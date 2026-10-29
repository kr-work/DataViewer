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

解析コア（SPEC §7 A-01〜A-08）の処理条件と結果のモデルも、このモジュールの末尾に置く。
Pipeline / View モデルは、それぞれ必要になる Phase（View: Phase 3、Pipeline: v0.2）で追加する。
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
    NON_BLANK_PATTERN,
    SHA256_PATTERN,
    BoolArray,
    FloatArray,
    check_cutoff_frequencies,
    check_derivative_settings,
    check_initial_state,
    check_kalman_result_arrays,
    check_linear_system,
    check_odd_window,
    check_polynomial_order,
    check_time_column_consistency,
    utc_now,
    validate_relative_posix_path,
    validate_tags,
)

# project.json の保存形式バージョン。
# metadata.sqlite のスキーマバージョン（PRAGMA user_version）とは独立に管理する。
# 詳細は docs/DECISIONS.md「メタデータの保存先とバージョン管理」を参照。
PROJECT_FORMAT_VERSION = 1


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
    """UNIX エポックからのナノ秒（絶対時刻）。"""


class TimeBase(WorkbenchModel):
    """Source の時間軸の定義。

    時刻の値そのものは Parquet 側に保持し、ここには「その時刻が何を意味するか」を記録する。

    Attributes:
        kind: 時刻の種類。
        original_unit: 元ファイルの時刻列の単位（例: "s"、"ms"、"ns"）。時刻列がない場合は None。
            正規化時に秒へ換算しても、元の単位をここに残す。
        origin: 時刻原点の説明（例: "装置の電源投入時"、"収録開始"）。不明なら "unknown"。
        clock: 時刻を刻んだ装置クロックの由来（例: "ロガー内部クロック"）。不明なら "unknown"。
        resolution_s: 時刻の分解能 [s]。不明なら None。
        epoch_origin_ns: 相対秒 0 に対応する絶対時刻（エポックナノ秒）。既知の場合のみ。
    """

    kind: TimeAxisKind
    original_unit: str | None = None
    origin: str = "unknown"
    clock: str = "unknown"
    resolution_s: PositiveFloat | None = None
    epoch_origin_ns: int | None = None


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
# 入力データの異常
# ---------------------------------------------------------------------------


class DataIssueKind(StrEnum):
    """入力データの異常の種類（SPEC §9「データ整合性」、§10「失敗データ」）。

    合成データに仕込んだ異常の記録と、取り込み時の検証結果の報告に使う。
    """

    EMPTY_FILE = "empty_file"
    """ファイルが 0 バイト。"""
    HEADER_ONLY = "header_only"
    """ヘッダー行のみでデータ行がない。"""
    DUPLICATE_COLUMN = "duplicate_column"
    """同じ列名が複数ある。"""
    NON_UTF8_ENCODING = "non_utf8_encoding"
    """UTF-8 以外の文字コード（Shift_JIS など）。"""
    NAN_VALUES = "nan_values"
    """NaN や空欄の値を含む。"""
    NON_MONOTONIC_TIME = "non_monotonic_time"
    """時刻が逆行している。"""
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    """同じ時刻のサンプルが複数ある。"""
    MISSING_SEGMENT = "missing_segment"
    """一定区間のサンプルが欠落している（時刻の飛び）。"""


class DataIssue(WorkbenchModel):
    """入力データの異常 1 件。

    Attributes:
        kind: 異常の種類。
        description: 人が読める説明。
        column: 異常のある列名。ファイル全体に関わる場合は None。
    """

    kind: DataIssueKind
    description: str
    column: str | None = None


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
    time_unit: str
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


class SyntheticTrialRecord(WorkbenchModel):
    """合成した 1 試技の真値情報。

    Attributes:
        name: 試技名（ディレクトリ名と同じ）。
        motion: この試技の真の運動。
        sources: この試技のファイル一覧。
    """

    name: str
    motion: SyntheticMotion
    sources: list[SyntheticSourceRecord]


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
