"""検証用の合成データ生成（SPEC §10）。

特定のセンサは想定しない。真値が解析的に分かる 1 次元の運動を観測した、汎用の時系列データを作る。
解析機能（フィルタ・積分・カルマンフィルタ）の結果を真値と比べたり、取り込み時の
異常検出を確かめたりするために使う。

生成するもの::

    <output>/
    ├── manifest.json        # 生成条件と真値（SyntheticManifest）
    ├── trial_001/
    │   ├── high_rate.csv    # 高レート（既定 1000 Hz）: 加速度の観測値 + 真値列
    │   ├── low_rate.csv     # 低レート（既定 100 Hz）: 位置の観測値 + 真値列
    │   └── io_pair.csv      # 入出力: むだ時間・一次遅れ・遊びが既知の系の入力と出力
    ├── trial_002/ ...
    └── broken/              # 取り込みエラー検証用の失敗データ

各ファイルの列:
    - high_rate.csv: ``time_s, accel, true_accel, true_velocity, true_position``
    - low_rate.csv: ``time_ms, position, true_position, true_velocity``
    - io_pair.csv: ``time_s, input, output, true_input, true_output``

使い方の例:
    - ``accel`` にはバイアス・白色ノイズ・高周波の振動が乗っている。
      ローパスフィルタや、積分によるドリフトの確認に使う。
    - ``position`` には白色ノイズが乗っている。カルマンフィルタで位置・速度を推定する確認に使う。
    - high_rate と low_rate は周波数・時刻原点・時刻の単位が異なる。明示的な時刻同期の確認に使う。
    - io_pair の前半はステップ入力（遅れ・時定数の確認）、後半は三角波の入力（ヒステリシスの確認）。

意図的に仕込む異常（試技 index は 0 始まり）:
    - index % 3 == 1: high_rate.csv の一部区間の行を削除する（時刻の飛び）。
    - index % 3 == 2: low_rate.csv の ``position`` の一部区間を NaN にする。

実行方法::

    uv run ewb-synthetic --output data/synthetic
    uv run python -m experiment_workbench.synthetic --output data/synthetic --duration 10
"""

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import ValidationError
from scipy import signal

from experiment_workbench import __version__
from experiment_workbench.model import (
    DataIssue,
    DataIssueKind,
    SineComponent,
    SyntheticBrokenFileRecord,
    SyntheticConfig,
    SyntheticIoSystem,
    SyntheticManifest,
    SyntheticMotion,
    SyntheticSourceRecord,
    SyntheticTrialRecord,
    TimeUnit,
)
from experiment_workbench.validation import FloatArray

MANIFEST_FILE_NAME = "manifest.json"
HIGH_RATE_FILE_NAME = "high_rate.csv"
LOW_RATE_FILE_NAME = "low_rate.csv"
IO_PAIR_FILE_NAME = "io_pair.csv"

# CSV に書き出す浮動小数点の小数桁数。1 kHz の時刻（1 ms）を µs の分解能で表せ、
# ファイルサイズが大きくなりすぎない桁数にしている。
CSV_FLOAT_PRECISION = 6


class SyntheticDataGenerator:
    """合成データ一式（試技ごとの CSV、失敗データ、manifest.json）を生成するクラス。

    同じ ``SyntheticConfig`` からは常に同じファイル内容が生成される（manifest.json の
    ``created_at`` を除く）。試技ごとに ``[seed, 試技 index]`` から乱数生成器を作るため、
    ``n_trials`` を変えても既存の試技の内容は変わらない。

    Attributes:
        config: 生成設定。
    """

    def __init__(self, config: SyntheticConfig) -> None:
        """生成設定を受け取って初期化する。

        Args:
            config: 生成設定。
        """
        self.config = config

    def generate(self, output_dir: Path) -> SyntheticManifest:
        """合成データ一式を ``output_dir`` に書き出す。

        既存ファイルを上書きしないよう、空でないディレクトリには書き出さない。

        Args:
            output_dir: 出力先ディレクトリ。存在しないか、空であること。

        Returns:
            書き出した内容の目録（manifest.json と同じ内容）。

        Raises:
            FileExistsError: ``output_dir`` がファイル、または空でないディレクトリの場合。
        """
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError(
                f"出力先が空のディレクトリではありません（上書き防止のため中止）: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        trials = [self._generate_trial(output_dir, index) for index in range(self.config.n_trials)]
        manifest = SyntheticManifest(
            generator_version=__version__,
            config=self.config,
            trials=trials,
            broken_files=self._write_broken_files(output_dir),
        )
        (output_dir / MANIFEST_FILE_NAME).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    # --- 試技 ----------------------------------------------------------------

    def _generate_trial(self, output_dir: Path, trial_index: int) -> SyntheticTrialRecord:
        """1 試技分の CSV（高レート・低レート）を書き出す。

        Args:
            output_dir: 出力先のルートディレクトリ。
            trial_index: 0 始まりの試技番号。乱数の種と、仕込む異常の割り当てに使う。

        Returns:
            試技の真値情報。
        """
        rng = np.random.default_rng([self.config.seed, trial_index])
        name = f"trial_{trial_index + 1:03d}"
        trial_dir = output_dir / name
        trial_dir.mkdir()

        motion = _random_motion(rng)
        # 試技ごとの個体差として、ノイズの大きさを 0.8〜1.5 倍の範囲で変える。
        noise_gain = float(rng.uniform(0.8, 1.5))
        high = self._write_high_rate(
            trial_dir, rng, motion, noise_gain, drop_segment=trial_index % 3 == 1
        )
        low = self._write_low_rate(
            trial_dir, rng, motion, noise_gain, inject_nan=trial_index % 3 == 2
        )
        # 入出力のデータは、既存のファイルの乱数を変えないよう、最後に乱数を使って作る。
        io_pair, io_system = self._write_io_pair(trial_dir, rng, high)
        return SyntheticTrialRecord(
            name=name, motion=motion, sources=[high, low], io_pair=io_pair, io_system=io_system
        )

    def _max_start_delay_s(self) -> float:
        """収録開始の遅れ（真の時刻）の上限を返す。

        短い試技でも、収録開始が欠損区間より前になるよう試技長の 10% までに制限する。

        Returns:
            収録開始の遅れの上限 [s]。
        """
        return min(2.0, 0.1 * self.config.duration_s)

    def _write_high_rate(
        self,
        trial_dir: Path,
        rng: np.random.Generator,
        motion: SyntheticMotion,
        noise_gain: float,
        *,
        drop_segment: bool,
    ) -> SyntheticSourceRecord:
        """高レートの加速度ファイル（high_rate.csv）を書き出す。

        観測値のモデル::

            accel = a(t) + bias + A_vib sin(2π f_vib t) + N(0, σ²)

        振動の周波数 f_vib はサンプリング周波数の 5%（既定 1000 Hz なら 50 Hz）とし、
        運動の成分（1 Hz 以下）と十分に離してローパスフィルタで分離できるようにする。
        時刻列 ``time_s`` の原点は真の時刻から 10〜60 秒ずらす（装置固有の時刻原点を模擬）。

        Args:
            trial_dir: 試技のディレクトリ。
            rng: 乱数生成器。
            motion: 真の運動。
            noise_gain: ノイズの倍率（試技ごとの個体差）。
            drop_segment: True なら一部区間の行を削除し、時刻の飛びを作る。

        Returns:
            書き出したファイルの生成条件と真値情報。
        """
        rate = self.config.high_rate_hz
        start_true = float(rng.uniform(0.0, self._max_start_delay_s()))
        clock_offset = float(rng.uniform(10.0, 60.0))
        t = _true_sample_times(start_true, self.config.duration_s, rate)
        position, velocity, accel = evaluate_motion(motion, t)

        bias = float(rng.normal(0.0, 0.05))
        noise_std = 0.05 * noise_gain
        vibration_amplitude = 0.5
        vibration_frequency = 0.05 * rate
        measured = (
            accel
            + bias
            + vibration_amplitude * np.sin(2 * np.pi * vibration_frequency * t)
            + rng.normal(0.0, noise_std, t.size)
        )
        columns: dict[str, FloatArray] = {
            "time_s": clock_offset + t,
            "accel": measured,
            "true_accel": accel,
            "true_velocity": velocity,
            "true_position": position,
        }

        issues: list[DataIssue] = []
        if drop_segment:
            # 試技長の 25% の位置から、最大 0.5 秒（試技長の 5%）の行を削除する。
            gap_start = 0.25 * self.config.duration_s
            gap_end = gap_start + min(0.5, 0.05 * self.config.duration_s)
            keep = ~((t >= gap_start) & (t < gap_end))
            columns = {name: values[keep] for name, values in columns.items()}
            issues.append(
                DataIssue(
                    kind=DataIssueKind.MISSING_SEGMENT,
                    description=(
                        f"真の時刻 {gap_start:.3f}〜{gap_end:.3f} s の "
                        f"{int((~keep).sum())} 行を削除"
                    ),
                )
            )

        frame = pl.DataFrame(columns)
        frame.write_csv(trial_dir / HIGH_RATE_FILE_NAME, float_precision=CSV_FLOAT_PRECISION)
        return SyntheticSourceRecord(
            relpath=f"{trial_dir.name}/{HIGH_RATE_FILE_NAME}",
            time_column="time_s",
            time_unit=TimeUnit.SECOND,
            nominal_rate_hz=rate,
            n_rows=frame.height,
            column_units={
                "time_s": "s",
                "accel": "m/s^2",
                "true_accel": "m/s^2",
                "true_velocity": "m/s",
                "true_position": "m",
            },
            measured_columns=["accel"],
            truth_columns=["true_accel", "true_velocity", "true_position"],
            clock_offset_s=clock_offset,
            recording_start_true_s=start_true,
            measurement_model="accel = true_accel + bias + A_vib*sin(2*pi*f_vib*t) + N(0, sigma^2)",
            noise_parameters={
                "bias": bias,
                "noise_std": noise_std,
                "vibration_amplitude": vibration_amplitude,
                "vibration_frequency_hz": vibration_frequency,
            },
            injected_issues=issues,
        )

    def _write_low_rate(
        self,
        trial_dir: Path,
        rng: np.random.Generator,
        motion: SyntheticMotion,
        noise_gain: float,
        *,
        inject_nan: bool,
    ) -> SyntheticSourceRecord:
        """低レートの位置ファイル（low_rate.csv）を書き出す。

        観測値のモデル::

            position = x(t) + N(0, σ²)

        時刻列 ``time_ms`` は収録開始を 0 とするミリ秒の整数とする（時刻の単位が秒でない
        データの確認用）。ミリ秒の格子からずれないよう、収録開始は 1 ms 単位に丸める。

        Args:
            trial_dir: 試技のディレクトリ。
            rng: 乱数生成器。
            motion: 真の運動。
            noise_gain: ノイズの倍率（試技ごとの個体差）。
            inject_nan: True なら ``position`` の一部区間を NaN にする。

        Returns:
            書き出したファイルの生成条件と真値情報。
        """
        rate = self.config.low_rate_hz
        start_true = round(float(rng.uniform(0.0, self._max_start_delay_s())), 3)
        clock_offset = -start_true
        t = _true_sample_times(start_true, self.config.duration_s, rate)
        position, velocity, _ = evaluate_motion(motion, t)

        noise_std = 0.01 * noise_gain
        measured = position + rng.normal(0.0, noise_std, t.size)

        issues: list[DataIssue] = []
        if inject_nan:
            # 試技長の 80% の位置から、最大 0.2 秒（試技長の 5%）を NaN にする。
            nan_start = 0.8 * self.config.duration_s
            nan_end = nan_start + min(0.2, 0.05 * self.config.duration_s)
            mask = (t >= nan_start) & (t < nan_end)
            measured = np.where(mask, np.nan, measured)
            issues.append(
                DataIssue(
                    kind=DataIssueKind.NAN_VALUES,
                    description=(
                        f"真の時刻 {nan_start:.3f}〜{nan_end:.3f} s の "
                        f"{int(mask.sum())} 行を NaN に置換"
                    ),
                    column="position",
                )
            )

        frame = pl.DataFrame(
            {
                "time_ms": np.round((clock_offset + t) * 1000.0).astype(np.int64),
                "position": measured,
                "true_position": position,
                "true_velocity": velocity,
            }
        )
        frame.write_csv(trial_dir / LOW_RATE_FILE_NAME, float_precision=CSV_FLOAT_PRECISION)
        return SyntheticSourceRecord(
            relpath=f"{trial_dir.name}/{LOW_RATE_FILE_NAME}",
            time_column="time_ms",
            time_unit=TimeUnit.MILLISECOND,
            nominal_rate_hz=rate,
            n_rows=frame.height,
            column_units={
                "time_ms": "ms",
                "position": "m",
                "true_position": "m",
                "true_velocity": "m/s",
            },
            measured_columns=["position"],
            truth_columns=["true_position", "true_velocity"],
            clock_offset_s=clock_offset,
            recording_start_true_s=start_true,
            measurement_model="position = true_position + N(0, sigma^2)",
            noise_parameters={"noise_std": noise_std},
            injected_issues=issues,
        )

    def _write_io_pair(
        self, trial_dir: Path, rng: np.random.Generator, high: SyntheticSourceRecord
    ) -> tuple[SyntheticSourceRecord, SyntheticIoSystem]:
        """入出力の合成データ（io_pair.csv）を書き出す。

        入力に対する応答（むだ時間・一次遅れ）と遊び（ヒステリシス）が既知の系の、入力と出力を
        同じファイルに記録したデータ。モデルは SyntheticIoSystem を参照。前半はステップ応答、
        後半はヒステリシスの解析の確認に使う。高レートファイルと同じ装置で記録した想定で、
        時刻の原点・周波数・収録開始は高レートファイルと同じにする。

        Args:
            trial_dir: 試技のディレクトリ。
            rng: 乱数生成器。
            high: 高レートファイルの記録（時刻の原点と収録開始に使う）。

        Returns:
            (書き出したファイルの記録, 系の真値)。
        """
        rate = high.nominal_rate_hz
        t = _true_sample_times(high.recording_start_true_s, self.config.duration_s, rate)
        span = float(t[-1] - t[0])
        system = _random_io_system(rng, t, rate)
        true_input = _io_input(system, t)
        true_output = evaluate_io_output(system, true_input, rate)
        measured_input = true_input + rng.normal(0.0, system.input_noise_std, t.size)
        measured_output = true_output + rng.normal(0.0, system.output_noise_std, t.size)
        frame = pl.DataFrame(
            {
                "time_s": high.clock_offset_s + t,
                "input": measured_input,
                "output": measured_output,
                "true_input": true_input,
                "true_output": true_output,
            }
        )
        frame.write_csv(trial_dir / IO_PAIR_FILE_NAME, float_precision=CSV_FLOAT_PRECISION)
        record = SyntheticSourceRecord(
            relpath=f"{trial_dir.name}/{IO_PAIR_FILE_NAME}",
            time_column="time_s",
            time_unit=TimeUnit.SECOND,
            nominal_rate_hz=rate,
            n_rows=frame.height,
            column_units={
                "time_s": "s",
                "input": "a.u.",
                "output": "a.u.",
                "true_input": "a.u.",
                "true_output": "a.u.",
            },
            measured_columns=["input", "output"],
            truth_columns=["true_input", "true_output"],
            clock_offset_s=high.clock_offset_s,
            recording_start_true_s=high.recording_start_true_s,
            measurement_model=(
                "output = offset + gain * LAG[play(true_input)](t - delay) + N(0, sigma_y^2), "
                f"input = true_input + N(0, sigma_u^2); span {span:.3f} s"
            ),
            noise_parameters={
                "input_noise_std": system.input_noise_std,
                "output_noise_std": system.output_noise_std,
            },
        )
        return record, system

    # --- 失敗データ ----------------------------------------------------------

    def _write_broken_files(self, output_dir: Path) -> list[SyntheticBrokenFileRecord]:
        """取り込みエラー検証用の失敗データを ``broken/`` に書き出す。

        Polars では列名の重複などを書き出せないため、標準ライブラリの csv で 1 行ずつ書く。
        元になる正常データは 100 Hz・50 行の ``time_s, value_a, value_b``。

        Args:
            output_dir: 出力先のルートディレクトリ。

        Returns:
            失敗データの一覧。
        """
        broken_dir = output_dir / "broken"
        broken_dir.mkdir()
        header = ["time_s", "value_a", "value_b"]
        rows = [
            [f"{k * 0.01:.2f}", f"{math.sin(k * 0.1):.6f}", f"{math.cos(k * 0.1):.6f}"]
            for k in range(50)
        ]
        records: list[SyntheticBrokenFileRecord] = []

        # 1. 空ファイル（0 バイト）
        (broken_dir / "empty.csv").write_bytes(b"")
        records.append(
            _broken_record(
                "empty.csv", "0 バイトのファイル", DataIssueKind.EMPTY_FILE, "ファイルが空"
            )
        )

        # 2. ヘッダーのみ
        _write_csv_rows(broken_dir / "header_only.csv", header, [])
        records.append(
            _broken_record(
                "header_only.csv",
                "ヘッダー行のみでデータ行がない",
                DataIssueKind.HEADER_ONLY,
                "データ行が 0 行",
            )
        )

        # 3. 列名の重複
        _write_csv_rows(broken_dir / "duplicate_columns.csv", ["time_s", "value", "value"], rows)
        records.append(
            _broken_record(
                "duplicate_columns.csv",
                "列名 value が 2 回現れる",
                DataIssueKind.DUPLICATE_COLUMN,
                "列名 value が重複",
                column="value",
            )
        )

        # 4. 文字コード違い（日本語ヘッダーを Windows の Shift_JIS = cp932 で保存）
        _write_csv_rows(
            broken_dir / "shift_jis_header.csv", ["時刻[s]", "値A", "値B"], rows, encoding="cp932"
        )
        records.append(
            _broken_record(
                "shift_jis_header.csv",
                "日本語ヘッダーを cp932（Shift_JIS）で保存。UTF-8 としては読めない",
                DataIssueKind.NON_UTF8_ENCODING,
                "文字コードが cp932",
                encoding="cp932",
            )
        )

        # 5. NaN と空欄
        nan_rows = [row.copy() for row in rows]
        for k in range(10, 15):
            nan_rows[k][1] = "NaN"
        for k in range(20, 25):
            nan_rows[k][2] = ""
        _write_csv_rows(broken_dir / "nan_values.csv", header, nan_rows)
        records.append(
            SyntheticBrokenFileRecord(
                relpath="broken/nan_values.csv",
                encoding="utf-8",
                description=(
                    "value_a の 10〜14 行目が 'NaN'、value_b の 20〜24 行目が空欄（0 始まり）"
                ),
                expected_issues=[
                    DataIssue(
                        kind=DataIssueKind.NAN_VALUES, description="'NaN' 文字列", column="value_a"
                    ),
                    DataIssue(kind=DataIssueKind.NAN_VALUES, description="空欄", column="value_b"),
                ],
            )
        )

        # 6. 時刻の逆転（20〜29 行目の行順を逆にする）
        reversed_rows = rows[:20] + rows[20:30][::-1] + rows[30:]
        _write_csv_rows(broken_dir / "timestamp_reversed.csv", header, reversed_rows)
        records.append(
            _broken_record(
                "timestamp_reversed.csv",
                "20〜29 行目（0 始まり）の行順が逆で、時刻が減少する",
                DataIssueKind.NON_MONOTONIC_TIME,
                "time_s が逆行",
                column="time_s",
            )
        )

        # 7. 時刻の重複（31・32 行目の時刻を 30 行目と同じにする）
        duplicate_rows = [row.copy() for row in rows]
        for k in (31, 32):
            duplicate_rows[k][0] = rows[30][0]
        _write_csv_rows(broken_dir / "duplicate_timestamps.csv", header, duplicate_rows)
        records.append(
            _broken_record(
                "duplicate_timestamps.csv",
                "30〜32 行目（0 始まり）が同じ時刻を持つ",
                DataIssueKind.DUPLICATE_TIMESTAMP,
                "time_s が重複",
                column="time_s",
            )
        )
        return records


# ---------------------------------------------------------------------------
# 真値の計算
# ---------------------------------------------------------------------------


def evaluate_motion(
    motion: SyntheticMotion, t: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """真の運動の位置・速度・加速度を、指定時刻で解析的に計算する。

    式は ``SyntheticMotion`` の docstring を参照。各成分の角周波数を ω = 2πf として、
    位置 A sin(ωt+φ) を 1 回・2 回微分した Aω cos(ωt+φ)、-Aω² sin(ωt+φ) を足し合わせる。

    Args:
        motion: 真の運動。
        t: 真の時刻 [s] の配列。

    Returns:
        (位置 [m], 速度 [m/s], 加速度 [m/s^2]) のタプル。各配列の形は ``t`` と同じ。
    """
    position = np.zeros_like(t)
    velocity = np.zeros_like(t)
    accel = np.zeros_like(t)
    for component in motion.position_components:
        omega = 2.0 * np.pi * component.frequency_hz
        phase = omega * t + component.phase_rad
        position += component.amplitude * np.sin(phase)
        velocity += component.amplitude * omega * np.cos(phase)
        accel -= component.amplitude * omega**2 * np.sin(phase)
    return position, velocity, accel


def _random_motion(rng: np.random.Generator) -> SyntheticMotion:
    """試技ごとの真の運動を乱数で決める。

    ゆっくりした大きな動き（0.1〜0.3 Hz、振幅 0.5〜1.5 m）と、やや速い小さな動き
    （0.5〜1.0 Hz、振幅 0.1〜0.3 m）の 2 成分にする。どちらも高レートファイルの
    振動成分（既定 50 Hz）より十分に低い周波数にして、フィルタで分離できるようにする。

    Args:
        rng: 乱数生成器。

    Returns:
        真の運動。
    """
    return SyntheticMotion(
        position_components=[
            SineComponent(
                amplitude=float(rng.uniform(0.5, 1.5)),
                frequency_hz=float(rng.uniform(0.1, 0.3)),
                phase_rad=float(rng.uniform(0.0, 2 * np.pi)),
            ),
            SineComponent(
                amplitude=float(rng.uniform(0.1, 0.3)),
                frequency_hz=float(rng.uniform(0.5, 1.0)),
                phase_rad=float(rng.uniform(0.0, 2 * np.pi)),
            ),
        ]
    )


def _random_io_system(rng: np.random.Generator, t: FloatArray, rate_hz: float) -> SyntheticIoSystem:
    """入出力の系のパラメータと、入力の作り方を決める。

    時定数とむだ時間は、短い試技でも各ステップの間に整定するよう、収録の長さで上限を決める。
    ステップの時刻は、ステップ応答の答えが厳密になるようサンプルの時刻に合わせる。

    Args:
        rng: 乱数生成器。
        t: サンプルの真の時刻 [s]。
        rate_hz: サンプリング周波数 [Hz]。

    Returns:
        系の真値。
    """
    span = float(t[-1] - t[0])
    low, high = 2.0, 12.0
    step_indices = [round(fraction * (t.size - 1)) for fraction in (0.1, 0.2, 0.3, 0.4)]
    max_time_constant = min(0.03, 0.01 * span)
    delay_samples = max(1, round(float(rng.uniform(0.005, 0.015)) * rate_hz))
    return SyntheticIoSystem(
        gain=float(rng.uniform(0.4, 0.6)),
        offset=float(rng.uniform(0.5, 1.5)),
        delay_s=delay_samples / rate_hz,
        time_constant_s=float(rng.uniform(0.5, 1.0)) * max_time_constant,
        play_width=0.5,
        input_noise_std=0.02,
        output_noise_std=0.01,
        initial_level=low,
        step_times_s=[float(t[index]) for index in step_indices],
        step_levels=[high, low, high, low],
        triangle_start_s=float(t[round(0.5 * (t.size - 1))]),
        triangle_period_s=0.25 * span,
        triangle_low=low,
        triangle_high=high,
    )


def _io_input(system: SyntheticIoSystem, t: FloatArray) -> FloatArray:
    """入出力の系の入力（前半はステップ、後半は三角波）を計算する。

    Args:
        system: 系の真値。
        t: サンプルの真の時刻 [s]。

    Returns:
        入力の真値。
    """
    values = np.full(t.shape, system.initial_level, dtype=np.float64)
    for step_time, level in zip(system.step_times_s, system.step_levels, strict=True):
        values[t >= step_time] = level
    triangle = t >= system.triangle_start_s
    phase = np.mod((t[triangle] - system.triangle_start_s) / system.triangle_period_s, 1.0)
    span = system.triangle_high - system.triangle_low
    values[triangle] = system.triangle_low + span * (1.0 - np.abs(2.0 * phase - 1.0))
    return values


def evaluate_io_output(
    system: SyntheticIoSystem, true_input: FloatArray, rate_hz: float
) -> FloatArray:
    """入出力の系の、雑音のない出力を計算する（式は SyntheticIoSystem を参照）。

    Args:
        system: 系の真値。
        true_input: 入力の真値（等間隔のサンプル）。
        rate_hz: サンプリング周波数 [Hz]。

    Returns:
        出力の真値。
    """
    half_width = system.play_width / 2.0
    played = np.empty_like(true_input)
    state = float(true_input[0]) - half_width  # 最初は増加してきた状態とする
    for index, value in enumerate(true_input):
        state = min(max(state, float(value) - half_width), float(value) + half_width)
        played[index] = state
    delay_samples = round(system.delay_s * rate_hz)
    delayed = np.concatenate(
        [np.full(delay_samples, played[0]), played[: true_input.size - delay_samples]]
    )
    # 一次遅れの厳密な離散化: s[k] = a s[k-1] + (1 - a) q[k-1]、a = exp(-Δt/τ)。
    decay = math.exp(-1.0 / (rate_hz * system.time_constant_s))
    numerator, denominator = [0.0, 1.0 - decay], [1.0, -decay]
    initial = signal.lfilter_zi(numerator, denominator) * delayed[0]  # 最初は定常状態
    lagged, _ = signal.lfilter(numerator, denominator, delayed, zi=initial)
    return np.asarray(system.offset + system.gain * lagged, dtype=np.float64)


def _true_sample_times(start_true_s: float, end_true_s: float, rate_hz: float) -> FloatArray:
    """収録開始から終了まで、等間隔のサンプル時刻（真の時間軸）を返す。

    Args:
        start_true_s: 収録開始の真の時刻 [s]。
        end_true_s: 収録終了の真の時刻 [s]（この時刻を超えないサンプルまで含める）。
        rate_hz: サンプリング周波数 [Hz]。

    Returns:
        真の時刻 [s] の配列。
    """
    # 浮動小数点の誤差で終端のサンプルが落ちないよう、わずかな余裕を足して切り捨てる。
    n_samples = math.floor((end_true_s - start_true_s) * rate_hz + 1e-9) + 1
    return start_true_s + np.arange(n_samples, dtype=np.float64) / rate_hz


# ---------------------------------------------------------------------------
# 失敗データ用の補助関数
# ---------------------------------------------------------------------------


def _write_csv_rows(
    path: Path, header: list[str], rows: list[list[str]], encoding: str = "utf-8"
) -> None:
    """ヘッダーと行を CSV として書き出す。

    OS によって改行コードが変わらないよう、改行は常に LF にする（Polars の出力と揃える）。

    Args:
        path: 出力先ファイル。
        header: ヘッダー行（列名の重複も許す）。
        rows: データ行（文字列のまま書き出す）。
        encoding: 文字コード。
    """
    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _broken_record(
    filename: str,
    description: str,
    kind: DataIssueKind,
    issue_description: str,
    *,
    column: str | None = None,
    encoding: str = "utf-8",
) -> SyntheticBrokenFileRecord:
    """異常が 1 種類だけの失敗データの記録を作る。

    Args:
        filename: ``broken/`` 内のファイル名。
        description: ファイル全体の説明。
        kind: 取り込み時に検出されるべき異常の種類。
        issue_description: 異常の説明。
        column: 異常のある列名（ファイル全体の異常なら None）。
        encoding: ファイルの文字コード。

    Returns:
        失敗データの記録。
    """
    return SyntheticBrokenFileRecord(
        relpath=f"broken/{filename}",
        encoding=encoding,
        description=description,
        expected_issues=[DataIssue(kind=kind, description=issue_description, column=column)],
    )


# ---------------------------------------------------------------------------
# コマンドライン
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """合成データ生成のコマンドラインエントリポイント。

    Args:
        argv: コマンドライン引数（プログラム名を除く）。None なら ``sys.argv[1:]``。

    Returns:
        終了コード。0: 成功、1: 出力先が空でないなどの実行時エラー、2: 設定の誤り。
    """
    defaults = SyntheticConfig()
    parser = argparse.ArgumentParser(
        prog="ewb-synthetic",
        description="検証用の合成データ（汎用の時系列と失敗データ）を生成する。",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="出力先（空のディレクトリ）"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed, help="乱数シード")
    parser.add_argument(
        "--duration", type=float, default=defaults.duration_s, help="1 試技の長さ [s]"
    )
    parser.add_argument("--trials", type=int, default=defaults.n_trials, help="試技数")
    args = parser.parse_args(argv)

    try:
        config = SyntheticConfig(seed=args.seed, duration_s=args.duration, n_trials=args.trials)
    except ValidationError as exc:
        print(f"設定が不正です:\n{exc}", file=sys.stderr)
        return 2

    try:
        manifest = SyntheticDataGenerator(config).generate(args.output)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    n_files = sum(len(trial.sources) + 1 for trial in manifest.trials)  # + 1 は io_pair.csv
    print(
        f"{args.output} に合成データを生成しました: 試技 {len(manifest.trials)} 件、"
        f"データファイル {n_files} 件、失敗データ {len(manifest.broken_files)} 件"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
