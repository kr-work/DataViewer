"""実行ファイル（PyInstaller）の入口。

普段は GUI を起動する（``ewb`` コマンドと同じ。2 つ目の引数にファイルを渡すと開く）。

``--smoke-test <ログのパス>`` を付けると、画面を出さずに主な処理を一通り動かして
終了する。ビルドした実行ファイルに必要なライブラリ（Qt・NumPy・SciPy・Polars など）が
そろっているかを、配布する前に確かめるため。設定とデータは一時フォルダに作り、
利用者の設定（Windows のレジストリなど）には書き込まない。
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from experiment_workbench.model import SyntheticConfig
from experiment_workbench.synthetic import SyntheticDataGenerator
from experiment_workbench.ui.main_window import (
    WORKSPACE_ROOT_KEY,
    MainWindow,
    application_settings,
    main,
)
from experiment_workbench.ui.theme import apply_theme

SMOKE_TEST_OPTION = "--smoke-test"
"""起動の確認だけを行う指定。"""

WAIT_LIMIT_S = 120.0
"""取り込みや計算の終了を待つ最大時間 [s]。"""

SMOKE_METHODS = ("バターワース", "KF", "区間統計")
"""起動の確認で実行する処理（フィルタ・カルマンフィルタ・結果の図をそれぞれ 1 つ）。"""


def wait_until_idle(application: QApplication, window: MainWindow) -> bool:
    """取り込みと計算が終わるまで、イベントを処理しながら待つ。

    Args:
        application: Qt のアプリケーション。
        window: メイン画面。

    Returns:
        時間内に終われば True。
    """
    deadline = time.monotonic() + WAIT_LIMIT_S
    while window.is_importing or window.analysis.is_running:
        if time.monotonic() > deadline:
            return False
        application.processEvents()
        time.sleep(0.01)
    # 結果を画面に反映する処理（自動保存・再描画）を流しきる。
    settle = time.monotonic() + 0.3
    while time.monotonic() < settle:
        application.processEvents()
        time.sleep(0.01)
    return True


def check_method(application: QApplication, window: MainWindow, label: str, log: list[str]) -> bool:
    """処理の一覧のボタンを押して計算し、結果が表示されたかを確かめる。

    Args:
        application: Qt のアプリケーション。
        window: メイン画面。
        label: 処理のボタンの名前。
        log: 結果を書き足すログ。

    Returns:
        結果が表示されたら True。
    """
    panel = window.analysis_panel
    if panel.is_editing:
        panel.back_button.click()
        application.processEvents()
    panel.method_list.buttons[label].click()
    if not wait_until_idle(application, window):
        log.append(f"NG {label}: 計算が {WAIT_LIMIT_S:.0f} 秒で終わらない")
        return False
    shown = len(window.workspace.panels[0].series_ids()) >= 2
    reported = window.result_view.analysis_id == panel.analysis_id
    status = panel.status_label.text()
    ok = shown or reported
    log.append(f"{'OK' if ok else 'NG'} {label}: 表示={shown} 解析結果の欄={reported} {status}")
    return ok


def run_smoke_test(log_path: Path) -> int:
    """画面を出さずに、ファイルを開いて処理を実行し、結果をログに書く。

    Args:
        log_path: ログを書くファイル（画面のある実行ファイルとして作るので、標準出力が
            使えない場合がある）。

    Returns:
        終了コード（すべて動けば 0、どれかが失敗すれば 1）。
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    log: list[str] = [f"Python {sys.version.split()[0]} / {sys.platform}"]
    ok = False
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(base / "s"))
        try:
            application = QApplication([sys.argv[0]])
            apply_theme(application)
            application_settings().setValue(WORKSPACE_ROOT_KEY, str(base / "workspaces"))
            SyntheticDataGenerator(SyntheticConfig(duration_s=10.0, n_trials=1)).generate(
                base / "data"
            )
            window = MainWindow()
            window.show()
            window.open_files([base / "data" / "trial_001" / "high_rate.csv"])
            opened = wait_until_idle(application, window) and bool(window.explorer_channel_ids())
            log.append(f"{'OK' if opened else 'NG'} ファイルを開く")
            results = [check_method(application, window, label, log) for label in SMOKE_METHODS]
            ok = opened and all(results)
            window.close()
            application.processEvents()
        except Exception:
            log.append(traceback.format_exc())
    log.append("成功" if ok else "失敗")
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0 if ok else 1


def entry() -> int:
    """実行ファイルの入口。引数で起動の確認と通常の起動を切り替える。

    Returns:
        終了コード。
    """
    if len(sys.argv) >= 3 and sys.argv[1] == SMOKE_TEST_OPTION:
        return run_smoke_test(Path(sys.argv[2]))
    return main()


if __name__ == "__main__":
    raise SystemExit(entry())
