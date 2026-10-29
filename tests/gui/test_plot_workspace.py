"""ui/plot_workspace.py（縦並べ比較）のテスト。"""

from uuid import UUID, uuid4

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from experiment_workbench.model import (
    Channel,
    ChannelData,
    Source,
    SourceFormat,
    TimeAxisKind,
    TimeBase,
    TimeUnit,
)
from experiment_workbench.ui.plot_workspace import StackedPlotWorkspace

TIME = np.arange(0.0, 10.0, 0.01)
"""テスト用の時刻（0〜10 秒、100 Hz）。"""


def _source(trial_id: UUID, name: str, *, shares_trial_clock: bool = False) -> Source:
    """表示のテストに必要な項目だけを持つ Source を作る。

    Args:
        trial_id: 所属する試技の ID。
        name: Source 名。
        shares_trial_clock: 同期済みなら True。

    Returns:
        Source。
    """
    return Source(
        trial_id=trial_id,
        name=name,
        format=SourceFormat.CSV,
        original_filename=f"{name}.csv",
        original_relpath=f"originals/x/{name}.csv",
        original_sha256="0" * 64,
        original_size_bytes=1,
        dataset_relpath=f"datasets/{name}.parquet",
        time_column="t",
        time_base=TimeBase(
            kind=TimeAxisKind.RELATIVE_SECONDS,
            original_unit=TimeUnit.SECOND,
            shares_trial_clock=shares_trial_clock,
        ),
    )


def _data(source: Source, name: str, frequency_hz: float = 1.0) -> ChannelData:
    """正弦波のチャンネルデータを作る。

    Args:
        source: チャンネルが属する Source。
        name: チャンネル名。
        frequency_hz: 正弦波の周波数 [Hz]。

    Returns:
        チャンネルのデータ。
    """
    channel = Channel(source_id=source.id, name=name, original_name=name, unit="V")
    values = np.sin(2 * np.pi * frequency_hz * TIME)
    return ChannelData(channel=channel, source=source, time_s=TIME.copy(), values=values)


@pytest.fixture
def workspace(qtbot: QtBot) -> StackedPlotWorkspace:
    """空の縦並べ表示を作る。

    Args:
        qtbot: pytest-qt のヘルパー。

    Returns:
        縦並べ表示。
    """
    widget = StackedPlotWorkspace()
    qtbot.addWidget(widget)
    widget.resize(800, 600)
    return widget


def test_panels_of_same_source_share_time_axis(workspace: StackedPlotWorkspace) -> None:
    """同じ Source のチャンネルを別のパネルに置くと、時間軸のズームが連動する。"""
    source = _source(uuid4(), "logger")
    assert workspace.add_channel(_data(source, "voltage"))
    second = workspace.add_panel()
    assert workspace.add_channel(_data(source, "tension"), second)
    first = workspace.panels[0]
    first.set_x_range((2.0, 3.0))
    assert second.x_range() == pytest.approx((2.0, 3.0), abs=0.05)
    assert not first.warning_label.isVisibleTo(workspace)
    assert not second.warning_label.isVisibleTo(workspace)


def test_cursor_moves_together_and_reads_nearest_sample(workspace: StackedPlotWorkspace) -> None:
    """カーソルを動かすと連動するパネルのカーソルも動き、最も近いサンプルの値を表示する。"""
    source = _source(uuid4(), "logger")
    workspace.add_channel(_data(source, "voltage"))
    second = workspace.add_panel()
    workspace.add_channel(_data(source, "tension", frequency_hz=0.5), second)
    first = workspace.panels[0]
    first.cursor_line.setValue(2.503)  # ドラッグと同じく位置変更のシグナルが出る
    assert second.cursor_time() == pytest.approx(2.503)
    expected = np.sin(2 * np.pi * 0.5 * 2.50)  # 最も近いサンプルは t = 2.50 s
    assert f"{expected:.6g} V" in second.readout_label.text()
    assert "t = 2.5 s" in second.readout_label.text()


def test_unsynchronized_source_is_not_linked_and_warned(workspace: StackedPlotWorkspace) -> None:
    """同期していない別の Source のパネルは時間軸を連動させず、警告を表示する。"""
    trial_id = uuid4()
    workspace.add_channel(_data(_source(trial_id, "a"), "voltage"))
    other = workspace.add_panel()
    assert workspace.add_channel(_data(_source(trial_id, "b"), "tension"), other)
    assert other.warning_label.isVisibleTo(workspace)
    before = other.x_range()
    workspace.panels[0].set_x_range((2.0, 3.0))
    assert other.x_range() == pytest.approx(before)


def test_incomparable_channel_is_refused_in_same_panel(workspace: StackedPlotWorkspace) -> None:
    """時刻を比較できないチャンネルは同じパネルに置けず、理由を知らせる。"""
    trial_id = uuid4()
    messages: list[str] = []
    workspace.message.connect(messages.append)
    workspace.add_channel(_data(_source(trial_id, "a"), "voltage"))
    assert not workspace.add_channel(_data(_source(trial_id, "b"), "tension"))
    assert workspace.panels[0].channel_ids() != []
    assert len(workspace.panels[0].channel_ids()) == 1
    assert "時刻を比較できない" in messages[-1]


def test_synchronized_sources_are_linked(workspace: StackedPlotWorkspace) -> None:
    """同じ試技で同期済みと明示した Source どうしは、同じパネルにも別のパネルにも並べられる。"""
    trial_id = uuid4()
    first_source = _source(trial_id, "a", shares_trial_clock=True)
    second_source = _source(trial_id, "b", shares_trial_clock=True)
    workspace.add_channel(_data(first_source, "voltage"))
    assert workspace.add_channel(_data(second_source, "tension"))
    other = workspace.add_panel()
    assert workspace.add_channel(_data(second_source, "temperature"), other)
    workspace.panels[0].set_x_range((4.0, 5.0))
    assert other.x_range() == pytest.approx((4.0, 5.0), abs=0.05)
    assert not other.warning_label.isVisibleTo(workspace)


def test_duplicate_channel_in_panel_is_refused(workspace: StackedPlotWorkspace) -> None:
    """同じチャンネルを同じパネルに 2 回は置けない。"""
    data = _data(_source(uuid4(), "a"), "voltage")
    assert workspace.add_channel(data)
    assert not workspace.add_channel(data)


def test_view_state_round_trip(qtbot: QtBot, workspace: StackedPlotWorkspace) -> None:
    """表示状態を保存用のモデルにして別の画面で復元すると、パネル・時間軸・カーソルが戻る。"""
    source = _source(uuid4(), "logger")
    voltage, tension = _data(source, "voltage"), _data(source, "tension")
    workspace.add_channel(voltage)
    workspace.add_channel(tension, workspace.add_panel())
    workspace.panels[0].set_x_range((1.0, 4.0))
    workspace.panels[0].set_cursor(2.0)
    view = workspace.view_state()

    restored = StackedPlotWorkspace()
    qtbot.addWidget(restored)
    by_id = {voltage.channel.id: voltage, tension.channel.id: tension}
    messages = restored.restore(view, by_id.get)
    assert messages == []
    assert [panel.channel_ids() for panel in restored.panels] == [
        [voltage.channel.id],
        [tension.channel.id],
    ]
    assert restored.panels[0].x_range() == pytest.approx((1.0, 4.0), abs=0.05)
    assert restored.panels[1].cursor_time() == pytest.approx(2.0)


def test_restore_reports_missing_channel(qtbot: QtBot) -> None:
    """保存されていたチャンネルが見つからない場合は、表示から外して理由を返す。"""
    source = _source(uuid4(), "logger")
    voltage = _data(source, "voltage")
    workspace = StackedPlotWorkspace()
    qtbot.addWidget(workspace)
    workspace.add_channel(voltage)
    view = workspace.view_state()
    restored = StackedPlotWorkspace()
    qtbot.addWidget(restored)
    nothing: dict[UUID, ChannelData] = {}
    messages = restored.restore(view, nothing.get)
    assert len(messages) == 1
    assert str(voltage.channel.id) in messages[0]
    assert restored.panels[0].channel_ids() == []


def test_removing_last_panel_keeps_one_empty_panel(workspace: StackedPlotWorkspace) -> None:
    """チャンネルを外すと表示が消え、最後のパネルを削除しても空のパネルが 1 つ残る。"""
    data = _data(_source(uuid4(), "a"), "voltage")
    workspace.add_channel(data)
    panel = workspace.panels[0]
    workspace.remove_channel(panel, data.channel.id)
    assert panel.channel_ids() == []
    assert panel.readout_label.text() == ""
    workspace.remove_panel(panel)
    assert len(workspace.panels) == 1
    assert workspace.panels[0].channel_ids() == []
