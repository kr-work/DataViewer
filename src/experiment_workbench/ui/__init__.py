"""GUI（PySide6 + PyQtGraph）。

計算・保存・時間軸の判定はコア（model / importer / storage / timebase / analysis）に任せ、
このパッケージには画面の組み立てと操作の受け渡しだけを置く（SPEC §8）。

起動: ``uv run ewb``（``experiment_workbench.ui.main_window.main``）
"""
