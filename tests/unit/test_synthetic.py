"""synthetic.py（合成データ生成）のテスト。"""

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiment_workbench.model import (
    DataIssueKind,
    SyntheticConfig,
    SyntheticManifest,
    SyntheticSourceRecord,
)
from experiment_workbench.synthetic import (
    MANIFEST_FILE_NAME,
    SyntheticDataGenerator,
    evaluate_io_output,
    evaluate_motion,
    main,
)
from experiment_workbench.validation import FloatArray

# 3 試技で、異常なし・欠損区間あり・NaN ありの全パターンが揃う。
SMALL_CONFIG = SyntheticConfig(duration_s=3.0, n_trials=3)

Generated = tuple[Path, SyntheticManifest]
"""(出力ディレクトリ, 目録) の組。"""


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Generated:
    """小さな設定で合成データを 1 回だけ生成し、モジュール内のテストで共有する。

    Args:
        tmp_path_factory: pytest の一時ディレクトリ生成器。

    Returns:
        (出力ディレクトリ, 目録) の組。
    """
    output = tmp_path_factory.mktemp("synthetic") / "data"
    return output, SyntheticDataGenerator(SMALL_CONFIG).generate(output)


def _sources(manifest: SyntheticManifest) -> list[SyntheticSourceRecord]:
    """全試技の Source 記録を 1 つのリストにまとめる。

    Args:
        manifest: 合成データの目録。

    Returns:
        Source 記録のリスト（試技順）。
    """
    return [source for trial in manifest.trials for source in trial.sources]


def _issue_kinds(source: SyntheticSourceRecord) -> set[DataIssueKind]:
    """Source に仕込まれた異常の種類の集合を返す。

    Args:
        source: Source 記録。

    Returns:
        異常の種類の集合。
    """
    return {issue.kind for issue in source.injected_issues}


def _true_times(frame: pl.DataFrame, source: SyntheticSourceRecord) -> FloatArray:
    """ファイルの時刻列を、真の時刻 [s] に換算する。

    ``source_time_s = clock_offset_s + t_true`` の関係（SyntheticSourceRecord 参照）を逆に使う。

    Args:
        frame: 読み込んだ CSV。
        source: その CSV の Source 記録。

    Returns:
        真の時刻 [s] の配列。
    """
    raw = frame[source.time_column].to_numpy().astype(np.float64)
    seconds = raw / 1000.0 if source.time_unit == "ms" else raw
    return seconds - source.clock_offset_s


# --- 出力ファイルの構成 --------------------------------------------------------


def test_manifest_file_matches_returned_manifest(generated: Generated) -> None:
    """manifest.json は generate() の戻り値と同じ内容になる。"""
    output, manifest = generated
    saved = SyntheticManifest.model_validate_json(
        (output / MANIFEST_FILE_NAME).read_text(encoding="utf-8")
    )
    assert saved == manifest


def test_files_have_declared_columns_and_row_counts(generated: Generated) -> None:
    """各ファイルの列・行数が目録と一致し、観測列と真値列が区別されている。"""
    output, manifest = generated
    assert len(manifest.trials) == SMALL_CONFIG.n_trials
    for source in _sources(manifest):
        frame = pl.read_csv(output / source.relpath)
        assert frame.columns == list(source.column_units)
        assert frame.height == source.n_rows
        assert set(source.measured_columns).isdisjoint(source.truth_columns)
        assert all(name.startswith("true_") for name in source.truth_columns)
        assert {*source.measured_columns, *source.truth_columns} <= set(frame.columns)


def test_files_use_different_rates_and_time_units(generated: Generated) -> None:
    """1 試技内の 2 ファイルは、周波数・時刻の単位が異なる（同期機能の検証用）。"""
    _, manifest = generated
    for trial in manifest.trials:
        high, low = trial.sources
        assert high.nominal_rate_hz != low.nominal_rate_hz
        assert (high.time_unit, low.time_unit) == ("s", "ms")


# --- 時刻 ----------------------------------------------------------------------


def test_time_columns_are_strictly_increasing(generated: Generated) -> None:
    """正常データの時刻列には逆行・重複がない。"""
    output, manifest = generated
    for source in _sources(manifest):
        times = pl.read_csv(output / source.relpath)[source.time_column].to_numpy()
        assert np.all(np.diff(times) > 0), source.relpath


def test_sampling_interval_matches_nominal_rate(generated: Generated) -> None:
    """サンプル間隔は公称周波数どおりで、欠損区間を仕込んだファイルだけ飛びがある。"""
    output, manifest = generated
    for source in _sources(manifest):
        frame = pl.read_csv(output / source.relpath)
        intervals = np.diff(_true_times(frame, source))
        expected = 1.0 / source.nominal_rate_hz
        if DataIssueKind.MISSING_SEGMENT in _issue_kinds(source):
            gap = min(0.5, 0.05 * SMALL_CONFIG.duration_s)
            assert intervals.max() == pytest.approx(gap + expected, abs=2 * expected)
            intervals = intervals[intervals < 2 * expected]
        # CSV の小数 6 桁への丸めによる誤差だけを許容する。
        np.testing.assert_allclose(intervals, expected, atol=2e-6)


def test_time_origin_follows_manifest(generated: Generated) -> None:
    """先頭サンプルの時刻は、目録のオフセットと収録開始時刻から計算した値になる。"""
    output, manifest = generated
    for source in _sources(manifest):
        frame = pl.read_csv(output / source.relpath)
        first_true = float(_true_times(frame, source)[0])
        assert first_true == pytest.approx(source.recording_start_true_s, abs=1e-5)
    for trial in manifest.trials:
        low = trial.sources[1]
        first_ms = pl.read_csv(output / low.relpath)[low.time_column][0]
        assert first_ms == 0  # 低レート側は収録開始を 0 ms とする


# --- 値 ------------------------------------------------------------------------


def test_truth_columns_follow_motion_model(generated: Generated) -> None:
    """真値列は、目録の運動パラメータから解析的に計算した値と一致する。"""
    output, manifest = generated
    for trial in manifest.trials:
        for source in trial.sources:
            frame = pl.read_csv(output / source.relpath)
            position, velocity, accel = evaluate_motion(trial.motion, _true_times(frame, source))
            expected = {
                "true_position": position,
                "true_velocity": velocity,
                "true_accel": accel,
            }
            for column in source.truth_columns:
                # 時刻・値の小数 6 桁への丸めによる誤差を許容する。
                np.testing.assert_allclose(
                    frame[column].to_numpy(), expected[column], atol=1e-4, err_msg=column
                )


def test_measured_values_follow_noise_model(generated: Generated) -> None:
    """観測値から既知の成分を引いた残差は、目録のノイズ標準偏差に従う。"""
    output, manifest = generated
    for trial in manifest.trials:
        high, low = trial.sources
        frame = pl.read_csv(output / high.relpath)
        t = _true_times(frame, high)
        params = high.noise_parameters
        vibration = params["vibration_amplitude"] * np.sin(
            2 * np.pi * params["vibration_frequency_hz"] * t
        )
        residual = (
            frame["accel"].to_numpy() - frame["true_accel"].to_numpy() - params["bias"] - vibration
        )
        assert residual.std() == pytest.approx(params["noise_std"], rel=0.15)
        assert abs(residual.mean()) < 4 * params["noise_std"] / np.sqrt(residual.size)

        frame = pl.read_csv(output / low.relpath)
        residual = frame["position"].to_numpy() - frame["true_position"].to_numpy()
        residual = residual[~np.isnan(residual)]
        assert residual.std() == pytest.approx(low.noise_parameters["noise_std"], rel=0.2)


def test_nan_appears_only_where_declared(generated: Generated) -> None:
    """NaN は異常を仕込んだ観測列にだけ現れ、真値列には現れない。"""
    output, manifest = generated
    for source in _sources(manifest):
        frame = pl.read_csv(output / source.relpath)
        declared = {
            issue.column
            for issue in source.injected_issues
            if issue.kind is DataIssueKind.NAN_VALUES
        }
        for column in [*source.measured_columns, *source.truth_columns]:
            has_nan = bool(frame[column].is_nan().any())
            assert has_nan == (column in declared), f"{source.relpath}:{column}"


def test_injected_issues_cover_each_pattern(generated: Generated) -> None:
    """3 試技で「異常なし」「欠損区間」「NaN」の各パターンが揃う。"""
    _, manifest = generated
    kinds_per_trial = [
        set().union(*(_issue_kinds(source) for source in trial.sources))
        for trial in manifest.trials
    ]
    assert kinds_per_trial == [
        set(),
        {DataIssueKind.MISSING_SEGMENT},
        {DataIssueKind.NAN_VALUES},
    ]


def test_truth_is_consistent_under_numerical_integration(generated: Generated) -> None:
    """真の加速度を台形則で積分すると真の速度に一致する（積分機能の検証に使える前提の確認）。"""
    output, manifest = generated
    high = manifest.trials[0].sources[0]  # 欠損区間のない高レートファイル
    frame = pl.read_csv(output / high.relpath)
    t = frame["time_s"].to_numpy()
    accel = frame["true_accel"].to_numpy()
    velocity = frame["true_velocity"].to_numpy()
    increments = 0.5 * (accel[1:] + accel[:-1]) * np.diff(t)
    integrated = velocity[0] + np.concatenate([[0.0], np.cumsum(increments)])
    np.testing.assert_allclose(integrated, velocity, atol=1e-3)


# --- 失敗データ ----------------------------------------------------------------


def _read_rows(path: Path, encoding: str = "utf-8") -> list[list[str]]:
    """CSV をヘッダー込みで文字列の行リストとして読む。

    Args:
        path: CSV ファイル。
        encoding: 文字コード。

    Returns:
        行のリスト（先頭がヘッダー）。
    """
    with path.open(encoding=encoding, newline="") as file:
        return list(csv.reader(file))


def test_broken_files_contain_declared_defects(generated: Generated) -> None:
    """失敗データには、目録に書かれたとおりの異常が入っている。"""
    output, manifest = generated
    broken = {Path(record.relpath).name: record for record in manifest.broken_files}
    directory = output / "broken"
    assert set(broken) == {path.name for path in directory.iterdir()}

    assert (directory / "empty.csv").stat().st_size == 0
    assert len(_read_rows(directory / "header_only.csv")) == 1

    header = _read_rows(directory / "duplicate_columns.csv")[0]
    assert header.count("value") == 2

    raw = (directory / "shift_jis_header.csv").read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert raw.decode("cp932").startswith("時刻[s]")
    assert broken["shift_jis_header.csv"].encoding == "cp932"

    rows = _read_rows(directory / "nan_values.csv")[1:]
    assert any(row[1] == "NaN" for row in rows)
    assert any(row[2] == "" for row in rows)

    times = [float(row[0]) for row in _read_rows(directory / "timestamp_reversed.csv")[1:]]
    assert np.any(np.diff(times) < 0)

    times = [float(row[0]) for row in _read_rows(directory / "duplicate_timestamps.csv")[1:]]
    assert np.any(np.diff(times) == 0)


# --- 決定性と上書き防止 ----------------------------------------------------------


def _csv_contents(output: Path) -> dict[str, bytes]:
    """出力ディレクトリ内の全 CSV の中身を、相対パスをキーにして読む。

    Args:
        output: 合成データの出力ディレクトリ。

    Returns:
        相対パス（"/" 区切り）→ ファイル内容 の辞書。
    """
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(output.rglob("*.csv"))
    }


def test_same_config_produces_identical_files(tmp_path: Path) -> None:
    """同じ設定なら、何度生成しても同じファイル内容になる。"""
    config = SyntheticConfig(duration_s=1.0, n_trials=2)
    SyntheticDataGenerator(config).generate(tmp_path / "a")
    SyntheticDataGenerator(config).generate(tmp_path / "b")
    assert _csv_contents(tmp_path / "a") == _csv_contents(tmp_path / "b")


def test_different_seed_changes_data(tmp_path: Path) -> None:
    """シードを変えると試技のデータが変わる。"""
    SyntheticDataGenerator(SyntheticConfig(duration_s=1.0, n_trials=1, seed=1)).generate(
        tmp_path / "a"
    )
    SyntheticDataGenerator(SyntheticConfig(duration_s=1.0, n_trials=1, seed=2)).generate(
        tmp_path / "b"
    )
    first, second = _csv_contents(tmp_path / "a"), _csv_contents(tmp_path / "b")
    assert first["trial_001/high_rate.csv"] != second["trial_001/high_rate.csv"]


def test_existing_trials_do_not_depend_on_trial_count(tmp_path: Path) -> None:
    """試技数を増やしても、既存の試技の内容は変わらない。"""
    SyntheticDataGenerator(SyntheticConfig(duration_s=1.0, n_trials=1)).generate(tmp_path / "a")
    SyntheticDataGenerator(SyntheticConfig(duration_s=1.0, n_trials=2)).generate(tmp_path / "b")
    first, second = _csv_contents(tmp_path / "a"), _csv_contents(tmp_path / "b")
    for relpath in ("trial_001/high_rate.csv", "trial_001/low_rate.csv"):
        assert first[relpath] == second[relpath]


def test_generate_refuses_non_empty_directory(tmp_path: Path) -> None:
    """既存ファイルを上書きしないよう、空でないディレクトリには生成しない。"""
    (tmp_path / "keep.txt").write_text("既存データ", encoding="utf-8")
    with pytest.raises(FileExistsError):
        SyntheticDataGenerator(SMALL_CONFIG).generate(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["keep.txt"]


# --- コマンドライン ------------------------------------------------------------


def test_cli_generates_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI で生成でき、終了コード 0 と件数の要約が出る。"""
    output = tmp_path / "cli"
    assert main(["--output", str(output), "--duration", "1", "--trials", "1"]) == 0
    manifest = json.loads((output / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert manifest["config"]["n_trials"] == 1
    assert "試技 1 件" in capsys.readouterr().out


def test_cli_rejects_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """不正な設定は終了コード 2 で、何も生成しない。"""
    output = tmp_path / "cli"
    assert main(["--output", str(output), "--duration", "0.1"]) == 2
    assert not output.exists()
    assert "設定が不正" in capsys.readouterr().err


def test_cli_refuses_non_empty_directory(tmp_path: Path) -> None:
    """空でない出力先には生成せず、終了コード 1 を返す。"""
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    assert main(["--output", str(tmp_path)]) == 1


# --- 入出力の合成データ ----------------------------------------------------------


def test_io_pair_matches_manifest(generated: Generated) -> None:
    """io_pair.csv の列・行数・時刻の原点が目録と一致し、高レートファイルと同じ時計で記録される。"""
    output, manifest = generated
    for trial in manifest.trials:
        record = trial.io_pair
        frame = pl.read_csv(output / record.relpath)
        assert frame.columns == list(record.column_units)
        assert frame.height == record.n_rows
        high = trial.sources[0]
        assert record.clock_offset_s == high.clock_offset_s
        assert record.recording_start_true_s == high.recording_start_true_s


def test_io_output_follows_system_model(generated: Generated) -> None:
    """真の出力は、遊び・むだ時間・一次遅れを 1 サンプルずつ計算した値と一致する。"""
    output, manifest = generated
    for trial in manifest.trials:
        system = trial.io_system
        rate = trial.io_pair.nominal_rate_hz
        frame = pl.read_csv(output / trial.io_pair.relpath)
        true_input = frame["true_input"].to_numpy().astype(np.float64)
        half = system.play_width / 2
        played = np.empty_like(true_input)
        state = true_input[0] - half
        for index, value in enumerate(true_input):
            state = min(max(state, value - half), value + half)
            played[index] = state
        delay = round(system.delay_s * rate)
        decay = np.exp(-1.0 / (rate * system.time_constant_s))
        lagged = np.empty_like(played)
        lagged[0] = played[0]
        for index in range(1, played.size):
            held = played[max(index - 1 - delay, 0)]
            lagged[index] = decay * lagged[index - 1] + (1 - decay) * held
        expected = system.offset + system.gain * lagged
        # 入力・出力の小数 6 桁への丸めによる誤差を許容する。
        np.testing.assert_allclose(frame["true_output"].to_numpy(), expected, atol=1e-5)
        assert evaluate_io_output(system, true_input, rate) == pytest.approx(expected, abs=1e-5)


def test_io_measured_values_follow_noise_model(generated: Generated) -> None:
    """入力・出力の観測値と真値の差は、目録の雑音の標準偏差に従う。"""
    output, manifest = generated
    for trial in manifest.trials:
        frame = pl.read_csv(output / trial.io_pair.relpath)
        system = trial.io_system
        input_noise = frame["input"].to_numpy() - frame["true_input"].to_numpy()
        output_noise = frame["output"].to_numpy() - frame["true_output"].to_numpy()
        assert input_noise.std() == pytest.approx(system.input_noise_std, rel=0.15)
        assert output_noise.std() == pytest.approx(system.output_noise_std, rel=0.15)


def test_io_steps_settle_before_next_step(generated: Generated) -> None:
    """どの試技でも、ステップの間隔はむだ時間 + 時定数の 5 倍より長い（ステップ応答の前提）。"""
    _, manifest = generated
    for trial in manifest.trials:
        system = trial.io_system
        spacing = np.diff(system.step_times_s)
        assert spacing.min() > system.delay_s + 5 * system.time_constant_s
