"""ui/analysis_panel.py（解析条件の数値編集）のテスト。

処理の一覧 → 条件の調整の画面の流れ、入力部品から AnalysisSpec（pydantic モデル）が正しく
作られること、再計算の待ち時間、不正な値の扱いを確かめる。実行・表示はメイン画面側
（test_analysis_controller.py）で確かめる。
"""

from uuid import UUID, uuid4

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QPushButton
from pytestqt.qtbot import QtBot

from experiment_workbench.analysis import (
    default_kalman_settings,
    estimate_noise_std,
    round_significant,
)
from experiment_workbench.model import (
    TWO_CHANNEL_ANALYSIS_KINDS,
    AbsoluteParams,
    AdaptiveNoiseMethod,
    AlignmentGrid,
    AlignmentParams,
    AnalysisKind,
    AnalysisSpec,
    ButterworthKind,
    ChannelData,
    CropParams,
    DerivativeMethod,
    DetrendParams,
    FillGapsParams,
    KalmanModelKind,
    KalmanSettings,
    NormalizeMode,
    NormalizeParams,
    OffsetMode,
    OffsetParams,
    ResampleParams,
    SavitzkyGolayParams,
    ScaleParams,
    StatisticsParams,
    StepResponseParams,
)
from experiment_workbench.ui.analysis_panel import (
    AUTO_RUN_DELAY_MS,
    BUSY_DISPLAY_DELAY_MS,
    METHOD_GROUPS,
    PRESET_AKF,
    PRESET_EKF,
    AnalysisPanel,
    CollapsibleSection,
    MatrixEditor,
)
from tests.conftest import Hierarchy

NO_EMISSION_WAIT_MS = AUTO_RUN_DELAY_MS + 300
"""自動再計算が起きないことを確かめるときに待つ時間 [ms]（待ち時間より長くする）。"""


@pytest.fixture
def panel(qtbot: QtBot) -> AnalysisPanel:
    """入力のない解析パネルを作る（表示はしない）。

    Args:
        qtbot: pytest-qt のヘルパー。

    Returns:
        解析パネル。
    """
    widget = AnalysisPanel()
    qtbot.addWidget(widget)
    return widget


def _input(hierarchy: Hierarchy) -> ChannelData:
    """1 kHz・2 秒の、雑音を含む正弦波の入力チャンネルを作る（欠測なし）。

    Args:
        hierarchy: テスト用のエンティティ（Source とチャンネル accel を使う）。

    Returns:
        入力チャンネルのデータ。
    """
    _, _, _, source, channel, _ = hierarchy
    time_s = np.arange(2000, dtype=np.float64) / 1000.0
    rng = np.random.default_rng(0)
    values = 3.0 + np.sin(2 * np.pi * 2.0 * time_s) + rng.normal(0.0, 0.05, time_s.size)
    return ChannelData(channel=channel, source=source, time_s=time_s, values=values)


def _set_choices(panel: AnalysisPanel, data: ChannelData) -> UUID:
    """入力と同じファイルの列と、別のファイルの列を、出力チャンネルの選択肢にする。

    Args:
        panel: 解析パネル。
        data: 入力チャンネルのデータ。

    Returns:
        入力と同じファイルの別の列の ID。
    """
    same_file = uuid4()
    panel.set_channel_choices(
        [
            (uuid4(), "other / a", uuid4()),
            (data.channel.id, "high_rate / accel", data.source.id),
            (same_file, "high_rate / force", data.source.id),
        ]
    )
    return same_file


def _select_data(combo: QComboBox, value: str) -> None:
    """選択欄で、値（文字列）に対応する項目を選ぶ（利用者が選ぶのと同じくシグナルが出る）。

    Args:
        combo: 選択欄。
        value: 選ぶ項目の値。
    """
    index = combo.findData(value)
    assert index >= 0, f"選択肢 {value!r} がありません"
    combo.setCurrentIndex(index)


def _choose_kind(panel: AnalysisPanel, kind: AnalysisKind) -> None:
    """処理の一覧で処理を選ぶ（すぐに計算が求められる）。

    Args:
        panel: 解析パネル。
        kind: 解析の種類。
    """
    panel.start_method(kind)


# --- 入力と初期値 ---------------------------------------------------------------


def test_panel_without_input_cannot_run(panel: AnalysisPanel) -> None:
    """チャンネルを選ぶまでは処理を選べず、何をすればよいかを案内する。"""
    assert not panel.method_list.isEnabled()
    assert "左の一覧でチャンネルを選ぶ" in panel.input_label.text()
    with pytest.raises(ValueError, match="チャンネルを選んで"):
        panel.build_spec()
    panel.start_method(AnalysisKind.BUTTERWORTH)
    assert not panel.is_editing


def test_set_input_fills_defaults_from_data(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """入力を選ぶと、名前・カットオフ周波数・カルマンフィルタの観測雑音がデータから決まる。"""
    data = _input(hierarchy)
    panel.set_input(data)
    assert panel.input_label.text() == "対象: accel [m/s^2]"
    assert panel.name_edit.text() == "accel LPF 10 Hz"
    assert panel.method_list.isEnabled()
    spec = panel.build_spec()
    assert spec.input_channel_id == data.channel.id
    assert spec.butterworth is not None
    assert spec.butterworth.cutoff_hz == 10.0  # min(10 Hz, 1000 Hz / 10)
    settings = panel.kalman_page.settings()
    assert settings.kinematic is not None
    sigma = settings.kinematic.measurement_noise_std
    assert sigma == round_significant(estimate_noise_std(data.values))
    assert panel.kalman_page.kinematic_sigma_edit.text() == repr(sigma)  # 短い表記のまま見える


def test_every_kalman_model_starts_from_data_based_defaults(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """入力を選んだ後にモデルを切り替えても、データから推定した雑音の強さから始まる。"""
    data = _input(hierarchy)
    panel.set_input(data)
    sigma = round_significant(estimate_noise_std(data.values))
    page = panel.kalman_page
    _select_data(page.model_combo, KalmanModelKind.POLYNOMIAL_OBSERVATION.value)
    polynomial = page.settings().polynomial_observation
    assert polynomial is not None
    assert polynomial.measurement_noise_std == sigma
    _select_data(page.model_combo, KalmanModelKind.LINEAR.value)
    linear = page.settings().linear
    assert linear is not None
    assert linear.measurement_noise == [[round_significant(sigma**2)]]
    assert page.settings().initial.state[0] == data.values[0]


def test_adaptive_choice_is_kept_for_new_input(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """適応推定の条件はデータの大きさによらないので、入力を選び直しても残る。"""
    data = _input(hierarchy)
    panel.set_input(data)
    page = panel.kalman_page
    _select_data(page.adaptive_combo, AdaptiveNoiseMethod.SAGE_HUSA.value)
    page.forgetting_edit.set_value(0.9)
    panel.set_input(data)
    adaptive = page.settings().adaptive
    assert adaptive is not None
    assert (adaptive.method, adaptive.forgetting_factor) == (AdaptiveNoiseMethod.SAGE_HUSA, 0.9)


def test_choosing_input_again_starts_new_analysis(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """入力を選び直すと新しい解析として扱う（前の解析の出力を上書きしない）。"""
    data = _input(hierarchy)
    panel.set_input(data)
    first = panel.analysis_id
    panel.set_input(data)
    assert panel.analysis_id != first


@pytest.mark.parametrize("kind", list(AnalysisKind))
def test_build_spec_uses_page_of_selected_kind(
    panel: AnalysisPanel, hierarchy: Hierarchy, kind: AnalysisKind
) -> None:
    """選んだ種類のページの条件だけが、解析の条件に入る。"""
    data = _input(hierarchy)
    _set_choices(panel, data)
    panel.set_input(data)
    _choose_kind(panel, kind)
    spec = panel.build_spec()
    assert spec.kind is kind
    assert getattr(spec, kind.value) is not None
    assert spec.name.startswith("accel ")
    assert (spec.output_channel_id is not None) == (kind in TWO_CHANNEL_ANALYSIS_KINDS)


def test_each_method_starts_a_new_analysis_but_loading_keeps_id(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """処理を選ぶと新しい処理になり、保存した処理を読み込むと ID・名前・条件を引き継ぐ。"""
    data = _input(hierarchy)
    panel.set_input(data)
    panel.start_method(AnalysisKind.BUTTERWORTH)
    first = panel.analysis_id
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    assert panel.analysis_id != first
    saved = AnalysisSpec(
        name="保存した平滑化",
        input_channel_id=data.channel.id,
        kind=AnalysisKind.SAVITZKY_GOLAY,
        savitzky_golay=SavitzkyGolayParams(window_samples=21, polyorder=3),
    )
    panel.load_spec(saved, data)
    assert panel.analysis_id == saved.id
    assert panel.kind() is AnalysisKind.SAVITZKY_GOLAY
    assert panel.is_editing
    rebuilt = panel.build_spec()
    assert rebuilt.model_dump(exclude={"created_at"}) == saved.model_dump(exclude={"created_at"})


def test_rebuilding_keeps_created_at(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """条件を変えて保存し直しても作成日時は変わらない（一覧の並びが動かないように）。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    first = panel.build_spec()
    panel.butterworth_page.order_spin.setValue(2)
    assert panel.build_spec().created_at == first.created_at


# --- 処理の一覧と調整の画面 ----------------------------------------------------------


def test_every_analysis_kind_has_a_button(panel: AnalysisPanel) -> None:
    """どの処理も一覧のボタンから始められ、用途の説明が付いている。"""
    kinds = {method[1] for _, _, methods in METHOD_GROUPS for method in methods}
    assert kinds == set(AnalysisKind)
    assert all(description for _, description, _ in METHOD_GROUPS)
    for button in panel.method_list.buttons.values():
        assert button.toolTip()


def test_method_button_opens_editor_and_runs(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """処理のボタンを押すと、調整の画面に移ってすぐに計算を求める。"""
    panel.set_input(_input(hierarchy))
    assert not panel.is_editing
    with qtbot.waitSignal(panel.run_requested, timeout=1000) as blocker:
        panel.method_list.buttons["移動平均"].click()
    assert panel.is_editing
    assert panel.title_label.text() == "移動平均"
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.kind is AnalysisKind.MOVING_AVERAGE


@pytest.mark.parametrize("kind", list(AnalysisKind))
def test_every_method_explains_what_it_does(
    panel: AnalysisPanel, hierarchy: Hierarchy, kind: AnalysisKind
) -> None:
    """どの処理にも、何をする処理かの説明が出る。"""
    data = _input(hierarchy)
    _set_choices(panel, data)
    panel.set_input(data)
    panel.start_method(kind)
    assert len(panel.description_label.text()) > 15


def test_description_follows_the_chosen_filter(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """フィルタの種類を変えると、説明もその種類のものに変わる。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    assert "高い周波数成分を除去" in panel.description_label.text()
    _select_data(panel.butterworth_page.kind_combo, ButterworthKind.HIGHPASS.value)
    assert "低い周波数成分" in panel.description_label.text()


def test_summary_shows_headline_without_blank_lines(panel: AnalysisPanel) -> None:
    """結論の文がない結果でも、空の行を出さない。"""
    panel.show_summary(["", "補足"])
    assert panel.status_label.text() == "補足"


def test_back_button_returns_to_method_list(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """「処理の一覧」に戻ると、調整の画面を閉じたことを知らせる。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    with qtbot.waitSignal(panel.methods_shown, timeout=1000):
        panel.back_button.click()
    assert not panel.is_editing


def test_pending_change_is_applied_before_leaving(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """待ち時間の途中で画面を移っても、最後の変更を捨てずに計算を求める。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    panel.butterworth_page.order_spin.setValue(2)
    with qtbot.waitSignal(panel.run_requested, timeout=100) as blocker:
        panel.back_button.click()
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.butterworth is not None
    assert spec.butterworth.order == 2


def test_duplicate_makes_a_new_analysis_with_same_conditions(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """「複製して比べる」は、同じ条件で新しい ID の処理を求める（元の処理は残る）。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    panel.butterworth_page.order_spin.setValue(2)
    panel.request_run()  # 待っている変更を先に反映しておく
    first = panel.analysis_id
    with qtbot.waitSignal(panel.run_requested, timeout=1000) as blocker:
        panel.duplicate_button.click()
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.id == panel.analysis_id
    assert spec.id != first
    assert spec.butterworth is not None
    assert spec.butterworth.order == 2


def test_delete_button_requests_deletion(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """「削除」は、調整中の処理の削除を求める。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    with qtbot.waitSignal(panel.delete_requested, timeout=1000) as blocker:
        panel.delete_button.click()
    assert blocker.args == [panel.analysis_id]


def test_name_shows_conditions_and_manual_name_is_kept(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """既定の名前は条件を表し（凡例で見分けられる）、手で付けた名前は変えない。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    page = panel.butterworth_page
    page.low_edit.setText("25")
    page.low_edit.textEdited.emit("25")
    assert panel.name_edit.text() == "accel LPF 25 Hz"
    _select_data(page.kind_combo, ButterworthKind.BANDPASS.value)
    page.high_edit.setText("40")
    page.high_edit.textEdited.emit("40")
    assert panel.name_edit.text() == "accel BPF 25-40 Hz"
    panel.name_edit.setText("張力の低域")
    panel.name_edit.textEdited.emit("張力の低域")
    page.low_edit.setText("30")
    page.low_edit.textEdited.emit("30")
    assert panel.name_edit.text() == "張力の低域"


@pytest.mark.parametrize(
    ("label", "name", "model", "adaptive"),
    [
        ("KF", "accel KF", KalmanModelKind.KINEMATIC, False),
        ("EKF", "accel EKF", KalmanModelKind.POLYNOMIAL_OBSERVATION, False),
        ("AKF", "accel AKF", KalmanModelKind.KINEMATIC, True),
    ],
)
def test_kalman_buttons_select_the_variant(
    panel: AnalysisPanel,
    hierarchy: Hierarchy,
    label: str,
    name: str,
    model: KalmanModelKind,
    adaptive: bool,
) -> None:
    """KF・EKF・AKF のボタンで、それぞれのモデルと適応推定が選ばれる。"""
    panel.set_input(_input(hierarchy))
    panel.method_list.buttons[label].click()
    spec = panel.build_spec()
    assert spec.name == name
    assert spec.kalman is not None
    assert spec.kalman.model_kind is model
    assert (spec.kalman.adaptive is not None) == adaptive


def test_switching_from_akf_to_kf_clears_adaptive(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """適応推定の選択が次の KF に残らない（ボタンの名前と中身を一致させる）。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.KALMAN, PRESET_AKF)
    panel.method_list.buttons["KF"].click()
    kalman = panel.build_spec().kalman
    assert kalman is not None
    assert kalman.adaptive is None
    panel.start_method(AnalysisKind.KALMAN, PRESET_EKF)
    assert panel.name_edit.text() == "accel EKF"


def test_hysteresis_smooths_direction_by_default(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """ヒステリシスは既定で向きの判定を平滑化する（なしでは差が小さく見積もられるため）。"""
    data = _input(hierarchy)  # 1 kHz
    _set_choices(panel, data)
    panel.set_input(data)
    panel.start_method(AnalysisKind.HYSTERESIS)
    params = panel.build_spec().hysteresis
    assert params is not None
    assert params.smoothing_cutoff_hz == 5.0


def test_two_channel_buttons_need_two_channels(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """チャンネルが 1 つしかなければ、2 チャンネルの処理は押せない。"""
    data = _input(hierarchy)
    panel.set_channel_choices([(data.channel.id, "high_rate / accel", data.source.id)])
    panel.set_input(data)
    assert not panel.method_list.buttons["ヒステリシス"].isEnabled()
    assert panel.method_list.buttons["バターワース"].isEnabled()
    _set_choices(panel, data)
    assert panel.method_list.buttons["ヒステリシス"].isEnabled()


def test_busy_row_appears_only_for_long_runs(qtbot: QtBot, panel: AnalysisPanel) -> None:
    """すぐ終わる計算では中止ボタンを出さない（スライダーの操作中にちらつかないように）。"""
    panel.set_running(running=True)
    panel.set_running(running=False)
    qtbot.wait(BUSY_DISPLAY_DELAY_MS + 100)
    assert panel.busy_row.isHidden()
    panel.set_running(running=True)
    qtbot.wait(BUSY_DISPLAY_DELAY_MS + 100)
    assert not panel.busy_row.isHidden()
    panel.set_running(running=False)
    assert panel.busy_row.isHidden()


# --- スライダー -------------------------------------------------------------------


def test_slider_changes_value_and_requests_run(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """スライダーを動かすと数値の欄が変わり、再計算を求める。値は見やすい桁に丸める。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    page = panel.butterworth_page
    before = page.low_edit.value()
    with qtbot.waitSignal(panel.run_requested, timeout=3000) as blocker:
        page.low_slider.slider.setValue(page.low_slider.slider.value() + 100)
    assert page.low_edit.value() > before
    assert len(page.low_edit.text().rstrip("0").replace(".", "")) <= 4
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.butterworth is not None
    assert spec.butterworth.cutoff_hz == page.low_edit.value()


def test_cutoff_slider_stays_below_nyquist(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """カットオフ周波数のスライダーの右端は、ナイキスト周波数を超えない。"""
    panel.set_input(_input(hierarchy))  # 1 kHz
    page = panel.butterworth_page
    page.low_slider.slider.setValue(page.low_slider.slider.maximum())
    assert page.low_edit.value() < 500.0
    page.low_slider.slider.setValue(0)
    assert page.low_edit.value() > 0.0


def test_typed_value_moves_slider(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """数値の欄に打った値にスライダーが合い、範囲の外の値も欄では保つ。"""
    panel.set_input(_input(hierarchy))
    page = panel.butterworth_page
    slider = page.low_slider.slider
    page.low_edit.setText("100")
    middle = slider.value()
    assert 0 < middle < slider.maximum()
    page.low_edit.setText("1e6")
    assert slider.value() == slider.maximum()
    assert page.low_edit.value() == 1e6


def test_window_slider_keeps_odd_for_centered_window(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """窓長のスライダーは奇数だけを選ぶ（中心窓は奇数でないと計算できないため）。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    page = panel.moving_average_page
    for position in (100, 333, 500, 777):
        page.window_slider.slider.setValue(position)
        assert page.window_spin.value() % 2 == 1
    assert panel.build_spec().moving_average is not None


def test_sliders_keep_usable_width_in_every_kalman_model(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """パネルの既定の幅で、どのモデルのスライダーも動かせる幅を保つ（項目名に潰されない）。"""
    panel.resize(370, 800)
    panel.show()
    qtbot.waitExposed(panel)
    panel.set_input(_input(hierarchy))
    page = panel.kalman_page
    panel.start_method(AnalysisKind.KALMAN, PRESET_EKF)
    qtbot.wait(20)
    assert page.polynomial_q_slider.slider.width() >= 100
    panel.start_method(AnalysisKind.KALMAN)
    _select_data(page.model_combo, KalmanModelKind.KINEMATIC.value)
    qtbot.wait(20)
    assert page.kinematic_q_slider.slider.width() >= 100
    assert page.kinematic_sigma_slider.slider.width() >= 100


def test_noise_sliders_center_on_data_based_value(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """カルマンフィルタの雑音のスライダーは、データから決めた値が中央に来る。"""
    panel.set_input(_input(hierarchy))
    slider = panel.kalman_page.kinematic_sigma_slider.slider
    assert slider.value() == pytest.approx(slider.maximum() / 2, abs=2)


# --- 種類ごとの条件 -------------------------------------------------------------


def test_butterworth_band_uses_both_cutoffs(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """帯域通過を選ぶと上限の欄が使えるようになり、(下限, 上限) が条件に入る。"""
    panel.set_input(_input(hierarchy))
    page = panel.butterworth_page
    assert not page.high_edit.isEnabled()
    _select_data(page.kind_combo, ButterworthKind.BANDPASS.value)
    assert page.high_edit.isEnabled()
    page.low_edit.set_value(5.0)
    page.high_edit.set_value(50.0)
    page.zero_phase_check.setChecked(False)
    params = panel.build_spec().butterworth
    assert params is not None
    assert (params.kind, params.cutoff_hz, params.zero_phase) == (
        ButterworthKind.BANDPASS,
        (5.0, 50.0),
        False,
    )


def test_derivative_with_savitzky_golay(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """Savitzky-Golay 微分を選ぶと窓長・次数の欄が使えるようになり、条件に入る。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.DERIVATIVE)
    page = panel.derivative_page
    _select_data(page.method_combo, DerivativeMethod.SAVITZKY_GOLAY.value)
    assert not page.edge_order_spin.isEnabled()
    page.window_spin.setValue(31)
    page.order_spin.setValue(2)
    params = panel.build_spec().derivative
    assert params is not None
    assert params.method is DerivativeMethod.SAVITZKY_GOLAY
    assert params.order == 2
    assert params.savitzky_golay == SavitzkyGolayParams(window_samples=31, polyorder=3)


def test_matrix_editor_round_trip_and_resize(qtbot: QtBot) -> None:
    """行列の表は値を丸めずに読み戻せ、大きさを変えると既存の値を残して新しい対角を埋める。"""
    editor = MatrixEditor("F")
    qtbot.addWidget(editor)
    editor.set_values([[1.0, 0.1], [1e-12, 4.0]])
    assert editor.values() == [[1.0, 0.1], [1e-12, 4.0]]
    editor.resize_matrix(3, 3, diagonal=9.0)
    assert editor.values() == [[1.0, 0.1, 0.0], [1e-12, 4.0, 0.0], [0.0, 0.0, 9.0]]
    editor.resize_matrix(1, 2)
    assert editor.values() == [[1.0, 0.1]]


def test_matrix_editor_reports_non_numeric_cell(qtbot: QtBot) -> None:
    """数値でないセルは、行列名と位置を示すエラーになる。"""
    editor = MatrixEditor("F")
    qtbot.addWidget(editor)
    editor.set_values([[1.0, 2.0]])
    item = editor.item(0, 1)
    assert item is not None
    item.setText("x")
    with pytest.raises(ValueError, match="行列 F の 1 行 2 列"):
        editor.values()


def test_linear_model_matrices_follow_state_dimension(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """線形モデルの次元を変えると F・H・Q・x0・P0 の大きさが合い、編集した値が条件に入る。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.KALMAN)
    page = panel.kalman_page
    _select_data(page.model_combo, KalmanModelKind.LINEAR.value)
    page.linear_dim_spin.setValue(3)
    assert np.shape(page.linear_f.values()) == (3, 3)
    assert page.linear_f.values()[2][2] == 1.0
    assert page.linear_h.values() == [[1.0, 0.0, 0.0]]
    assert np.shape(page.x0_editor.values()) == (1, 3)
    assert np.shape(page.p0_editor.values()) == (3, 3)
    item = page.linear_f.item(0, 1)
    assert item is not None
    item.setText("0.001")
    settings = panel.build_spec().kalman
    assert settings is not None
    assert settings.model_kind is KalmanModelKind.LINEAR
    assert settings.linear is not None
    assert settings.linear.transition[0][1] == 0.001
    assert len(settings.initial.state) == 3


def test_polynomial_degree_resizes_coefficients(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """多項式の次数を変えると係数の欄の数が合い、係数が条件に入る。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.KALMAN)
    page = panel.kalman_page
    _select_data(page.model_combo, KalmanModelKind.POLYNOMIAL_OBSERVATION.value)
    page.polynomial_degree_spin.setValue(3)
    assert page.polynomial_coefficients.values() == [[0.0, 1.0, 0.0, 0.0]]
    settings = panel.build_spec().kalman
    assert settings is not None
    assert settings.polynomial_observation is not None
    assert settings.polynomial_observation.coefficients == [0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("model_kind", list(KalmanModelKind))
def test_kalman_settings_round_trip_through_page(
    panel: AnalysisPanel, hierarchy: Hierarchy, model_kind: KalmanModelKind
) -> None:
    """カルマンフィルタの条件をページに表示して読み戻すと、値を丸めずに同じ条件になる。"""
    data = _input(hierarchy)
    base = default_kalman_settings(model_kind, data)
    settings = KalmanSettings.model_validate(
        base.model_dump()
        | {"adaptive": {"method": AdaptiveNoiseMethod.SAGE_HUSA, "forgetting_factor": 0.97}}
    )
    panel.kalman_page.load(settings)
    assert panel.kalman_page.settings() == settings


def test_defaults_button_builds_initial_state_from_data(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """「データから初期値を作る」で、多項式の観測を逆算した初期状態が入る。"""
    data = _input(hierarchy)
    panel.set_input(data)
    page = panel.kalman_page
    _select_data(page.model_combo, KalmanModelKind.POLYNOMIAL_OBSERVATION.value)
    page.polynomial_coefficients.set_values([[1.0, 2.0]])
    page.x0_editor.set_values([[0.0, 0.0]])
    (button,) = [b for b in page.findChildren(QPushButton) if b.text() == "データから初期値を作る"]
    button.click()
    state = page.x0_editor.values()[0]
    assert state[0] == pytest.approx((data.values[0] - 1.0) / 2.0)
    assert state[1] == 0.0


# --- 自動再計算と不正な値 --------------------------------------------------------


def test_parameter_changes_run_once_after_delay(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """条件を続けて変えても、最後の変更から少し待って 1 回だけ実行を求める。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    page = panel.butterworth_page
    with qtbot.waitSignal(panel.run_requested, timeout=3000) as blocker:
        page.order_spin.setValue(2)
        page.order_spin.setValue(3)
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.butterworth is not None
    assert spec.butterworth.order == 3
    with qtbot.assertNotEmitted(panel.run_requested, wait=NO_EMISSION_WAIT_MS):
        pass


def test_matrix_edit_triggers_auto_run(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """行列のセルを書き換えても自動で再計算を求める（カルマンフィルタの調整用）。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.KALMAN)
    item = panel.kalman_page.p0_editor.item(0, 0)
    assert item is not None
    with qtbot.waitSignal(panel.run_requested, timeout=3000) as blocker:
        item.setText("2.5")
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.kalman is not None
    assert spec.kalman.initial.covariance[0][0] == 2.5


def test_no_run_is_requested_on_method_list(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """処理の一覧を出している間は、条件を変えても実行を求めない（何も調整していないため）。"""
    panel.set_input(_input(hierarchy))
    with qtbot.assertNotEmitted(panel.run_requested, wait=NO_EMISSION_WAIT_MS):
        panel.butterworth_page.order_spin.setValue(2)


def test_non_numeric_value_shows_error_and_does_not_run(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """数値でない入力は、項目名を含むエラーを表示して実行を求めない。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.BUTTERWORTH)
    panel.butterworth_page.low_edit.setText("abc")
    with qtbot.assertNotEmitted(panel.run_requested):
        panel.request_run()
    assert "「カットオフ周波数」には数値を入力してください" in panel.status_label.text()


def test_invalid_combination_shows_validation_error(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """条件の組み合わせが不正（中心移動平均の窓長が偶数）なら、検証エラーを表示して実行しない。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.MOVING_AVERAGE)
    panel.moving_average_page.window_spin.setValue(10)
    with qtbot.assertNotEmitted(panel.run_requested):
        panel.request_run()
    assert panel.status_label.text().startswith("条件が不正です")


# --- 2 チャンネルの解析 -----------------------------------------------------------


def test_two_channel_options_show_only_for_two_channel_kinds(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """出力チャンネルや範囲の欄は、2 チャンネルの解析のときだけ表示する。"""
    panel.set_input(_input(hierarchy))
    assert not panel.alignment_page.isVisibleTo(panel)
    _choose_kind(panel, AnalysisKind.HYSTERESIS)
    assert panel.alignment_page.isVisibleTo(panel)
    _choose_kind(panel, AnalysisKind.INTEGRAL)
    assert not panel.alignment_page.isVisibleTo(panel)


def test_output_defaults_to_another_column_of_same_file(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """出力チャンネルの既定は、入力と同じファイルの別の列にする（時刻の対応がそのまま取れる）。"""
    data = _input(hierarchy)
    same_file = _set_choices(panel, data)
    panel.set_input(data)
    assert panel.alignment_page.output_channel_id() == same_file


def test_alignment_settings_go_into_spec(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """時刻オフセット・等間隔の時刻・範囲の指定が、解析の条件に入る。"""
    data = _input(hierarchy)
    _set_choices(panel, data)
    panel.set_input(data)
    _choose_kind(panel, AnalysisKind.CROSS_CORRELATION)
    page = panel.alignment_page
    page.offset_check.setChecked(True)
    page.offset_edit.set_value(1.5)
    _select_data(page.grid_combo, AlignmentGrid.UNIFORM.value)
    page.rate_edit.set_value(200.0)
    page.range_check.setChecked(True)
    page.set_range(0.5, 1.5)
    alignment = panel.build_spec().alignment
    assert alignment is not None
    assert alignment.output_time_offset_s == 1.5
    assert alignment.uniform_rate_hz == 200.0
    assert (alignment.start_s, alignment.end_s) == (0.5, 1.5)


def test_without_output_choice_run_is_refused(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """出力チャンネルを選べないときは、理由を表示して実行しない。"""
    panel.set_input(_input(hierarchy))
    with qtbot.assertNotEmitted(panel.run_requested):
        _choose_kind(panel, AnalysisKind.HYSTERESIS)
    assert "出力チャンネルを選んでください" in panel.status_label.text()


def test_step_response_starts_range_selection(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """ステップ応答を選ぶと、入力の変化を 1 回含む範囲を選べるよう、範囲の指定を始める。"""
    data = _input(hierarchy)
    _set_choices(panel, data)
    panel.set_input(data)
    with qtbot.waitSignal(panel.range_selection_changed, timeout=1000) as blocker:
        _choose_kind(panel, AnalysisKind.STEP_RESPONSE)
    assert blocker.args == [True]
    assert panel.range_selection_active()
    with qtbot.waitSignal(panel.range_selection_changed, timeout=1000) as blocker:
        _choose_kind(panel, AnalysisKind.INTEGRAL)
    assert blocker.args == [False]


def test_saved_two_channel_analysis_is_loaded_back(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """保存した 2 チャンネルの解析を読み込むと、出力・範囲・条件が戻る。"""
    data = _input(hierarchy)
    same_file = _set_choices(panel, data)
    saved = AnalysisSpec(
        name="応答",
        input_channel_id=data.channel.id,
        output_channel_id=same_file,
        kind=AnalysisKind.STEP_RESPONSE,
        alignment=AlignmentParams(start_s=0.25, end_s=0.75),
        step_response=StepResponseParams(settling_band=0.02, smoothing_cutoff_hz=50.0),
    )
    panel.load_spec(saved, data)
    rebuilt = panel.build_spec()
    assert rebuilt.model_dump(exclude={"created_at"}) == saved.model_dump(exclude={"created_at"})


def test_setting_range_counts_as_a_change(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """グラフで範囲を動かすと、条件の変更として自動で再計算を求める。"""
    data = _input(hierarchy)
    _set_choices(panel, data)
    panel.set_input(data)
    _choose_kind(panel, AnalysisKind.HYSTERESIS)
    panel.alignment_page.range_check.setChecked(True)
    with qtbot.waitSignal(panel.run_requested, timeout=3000) as blocker:
        panel.set_range(0.2, 0.8)
    spec = blocker.args[0] if blocker.args else None
    assert isinstance(spec, AnalysisSpec)
    assert spec.alignment is not None
    assert (spec.alignment.start_s, spec.alignment.end_s) == (0.2, 0.8)


# --- 画面の使いやすさ -------------------------------------------------------------


def test_small_pages_do_not_reserve_height_of_large_pages(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """小さいページの下に、大きいページの分の空白を作らない（隠したページの高さを含めない）。"""
    panel.set_input(_input(hierarchy))
    _choose_kind(panel, AnalysisKind.MOVING_AVERAGE)
    small = panel.page_stack.sizeHint().height()
    _choose_kind(panel, AnalysisKind.DERIVATIVE)
    assert panel.page_stack.sizeHint().height() > small
    _choose_kind(panel, AnalysisKind.MOVING_AVERAGE)
    assert panel.page_stack.sizeHint().height() == small
    page = panel.kalman_page
    kinematic = page.model_stack.sizeHint().height()
    _select_data(page.model_combo, KalmanModelKind.LINEAR.value)
    assert page.model_stack.sizeHint().height() > kinematic


def test_collapsible_section_shows_body_when_opened(qtbot: QtBot) -> None:
    """畳んだ部分は見出しのクリックで開く。"""
    body = QPushButton("中身")
    section = CollapsibleSection("見出し", body)
    qtbot.addWidget(section)
    assert not body.isVisibleTo(section)
    section.toggle_button.click()
    assert body.isVisibleTo(section)


# --- 用語と入力欄 -----------------------------------------------------------------


def test_band_filter_names_lower_and_upper_cutoff(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """帯域の特性では、下側と上側のカットオフ周波数として表示する。"""
    panel.set_input(_input(hierarchy))
    page = panel.butterworth_page
    assert page.low_edit.label == "カットオフ周波数"
    _select_data(page.kind_combo, ButterworthKind.BANDPASS.value)
    assert page.low_edit.label == "下側カットオフ周波数"


def test_nyquist_frequency_is_shown(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """カットオフ周波数の上限の目安として、ナイキスト周波数を表示する。"""
    panel.set_input(_input(hierarchy))  # 1 kHz
    assert panel.butterworth_page.nyquist_label.text() == "ナイキスト周波数 500 Hz"


def test_padding_option_applies_only_to_zero_phase(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """端点処理はゼロ位相フィルタでだけ使うので、それ以外では選べない。"""
    panel.set_input(_input(hierarchy))
    page = panel.butterworth_page
    page.zero_phase_check.setChecked(False)
    assert not page.padtype_combo.isEnabled()
    page.zero_phase_check.setChecked(True)
    assert page.padtype_combo.isEnabled()


def test_window_length_is_also_shown_as_time(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """窓長は、サンプル数に加えて時間に換算した値も表示する。"""
    panel.set_input(_input(hierarchy))  # 1 kHz
    panel.start_method(AnalysisKind.MOVING_AVERAGE)
    assert panel.moving_average_page.window_slider.note_label.text() == "11 ms に相当"
    panel.moving_average_page.window_spin.setValue(2001)
    assert panel.moving_average_page.window_slider.note_label.text() == "2.001 s に相当"


def test_kalman_noise_labels_show_units_of_input(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """雑音の項目名に、入力の単位から決まる単位を表示する（q の単位は階数で変わる）。"""
    panel.set_input(_input(hierarchy))  # 単位は m/s^2
    page = panel.kalman_page
    assert page._kinematic_sigma_label.text() == "観測雑音の標準偏差 σ [m/s^2]"
    assert page._kinematic_q_label.text() == "プロセス雑音 PSD q [(m/s^2)²/s^3]"
    page.kinematic_order_spin.setValue(2)
    assert page._kinematic_q_label.text() == "プロセス雑音 PSD q [(m/s^2)²/s^5]"


def test_derivative_shows_only_fields_of_chosen_method(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """微分の計算方法に関係しない欄は出さない。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.DERIVATIVE)
    page = panel.derivative_page
    assert page.edge_order_spin.isVisibleTo(page)
    assert not page.window_spin.isVisibleTo(page)
    _select_data(page.method_combo, DerivativeMethod.SAVITZKY_GOLAY.value)
    assert not page.edge_order_spin.isVisibleTo(page)
    assert page.window_spin.isVisibleTo(page)


# --- 値の変換・時間の処理の条件 -------------------------------------------------------


def test_crop_always_uses_range(qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """切り出しは範囲が必須なので、範囲の指定のチェックを出さずに帯を出す。"""
    panel.set_input(_input(hierarchy))
    panel.alignment_page.set_range(0.5, 1.0)
    with qtbot.waitSignal(panel.range_selection_changed, timeout=1000) as blocker:
        panel.start_method(AnalysisKind.CROP)
    assert blocker.args == [True]
    page = panel.alignment_page
    assert page.isVisibleTo(panel)
    assert not page.range_check.isVisibleTo(page)
    assert not page.output_combo.isVisibleTo(page)
    assert not page.details_section.isVisibleTo(page)
    crop = panel.build_spec().crop
    assert crop is not None
    assert (crop.start_s, crop.end_s) == (0.5, 1.0)


def test_offset_range_is_optional_and_hidden_for_given_value(
    panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """平均値を差し引くときは区間を選べ、指定した値を差し引くときは区間の欄を出さない。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.OFFSET)
    page = panel.offset_page
    assert panel.alignment_page.isVisibleTo(panel)
    assert not page.value_edit.isVisibleTo(page)
    offset = panel.build_spec().offset
    assert offset is not None
    assert offset.start_s is None
    _select_data(page.mode_combo, OffsetMode.VALUE.value)
    assert not panel.alignment_page.isVisibleTo(panel)
    assert page.value_edit.isVisibleTo(page)
    assert not panel.range_selection_active()


def test_resample_starts_at_half_of_input_rate(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """リサンプリングの周波数の初期値は入力の半分（1 kHz なら 500 Hz）。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.RESAMPLE)
    resample = panel.build_spec().resample
    assert resample is not None
    assert resample.rate_hz == 500.0
    assert resample.anti_alias
    assert panel.name_edit.text() == "accel 500 Hz"


def test_scale_refuses_zero_gain_in_panel(
    qtbot: QtBot, panel: AnalysisPanel, hierarchy: Hierarchy
) -> None:
    """倍率 0 は理由を表示して実行しない。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.SCALE)
    panel.scale_page.gain_edit.set_value(0.0)
    with qtbot.assertNotEmitted(panel.run_requested):
        panel.request_run()
    assert "倍率" in panel.status_label.text()


def test_fill_limit_field_appears_when_enabled(panel: AnalysisPanel, hierarchy: Hierarchy) -> None:
    """欠損の長さの上限は、チェックしたときだけ条件に入る。"""
    panel.set_input(_input(hierarchy))
    panel.start_method(AnalysisKind.FILL_GAPS)
    page = panel.fill_gaps_page
    fill = panel.build_spec().fill_gaps
    assert fill is not None
    assert fill.max_gap_s is None
    page.limit_check.setChecked(True)
    page.limit_edit.set_value(0.5)
    fill = panel.build_spec().fill_gaps
    assert fill is not None
    assert fill.max_gap_s == 0.5


@pytest.mark.parametrize(
    ("kind", "field", "params"),
    [
        (AnalysisKind.OFFSET, "offset", OffsetParams(start_s=0.25, end_s=0.75)),
        (AnalysisKind.OFFSET, "offset", OffsetParams(mode=OffsetMode.VALUE, value=2.5)),
        (AnalysisKind.SCALE, "scale", ScaleParams(gain=9.8, offset=1.0, unit="N")),
        (AnalysisKind.DETREND, "detrend", DetrendParams(order=3)),
        (AnalysisKind.ABSOLUTE, "absolute", AbsoluteParams()),
        (AnalysisKind.NORMALIZE, "normalize", NormalizeParams(mode=NormalizeMode.PEAK)),
        (AnalysisKind.CROP, "crop", CropParams(start_s=0.25, end_s=0.75)),
        (AnalysisKind.RESAMPLE, "resample", ResampleParams(rate_hz=250.0, anti_alias=False)),
        (AnalysisKind.FILL_GAPS, "fill_gaps", FillGapsParams(max_gap_s=0.5)),
        (
            AnalysisKind.STATISTICS,
            "statistics",
            StatisticsParams(start_s=0.5, end_s=1.5, n_bins=20),
        ),
    ],
    ids=str,
)
def test_saved_processing_is_loaded_back(
    panel: AnalysisPanel,
    hierarchy: Hierarchy,
    kind: AnalysisKind,
    field: str,
    params: object,
) -> None:
    """保存した処理を読み込むと、すべての条件が同じ値で戻る。"""
    data = _input(hierarchy)
    saved = AnalysisSpec.model_validate(
        {"name": "保存", "input_channel_id": data.channel.id, "kind": kind, field: params}
    )
    panel.load_spec(saved, data)
    rebuilt = panel.build_spec()
    assert rebuilt.model_dump(exclude={"created_at"}) == saved.model_dump(exclude={"created_at"})
