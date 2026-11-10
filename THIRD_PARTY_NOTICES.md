# 第三者ソフトウェアのライセンス

Experiment Workbench が使っているライブラリと、そのライセンスの確認結果です（2026-11-06 時点、`uv.lock` のバージョン）。
確認の記録であり、法的な助言ではありません。

## 結論

- 実行に必要な 13 パッケージは、すべて問題なく使えます。GPL でしか使えないものはありません。
- このリポジトリには、ライブラリのコードも二進ファイルも含めていません。`uv sync` が配布元から利用者の環境へ直接入れるので、
  公開と利用に追加の作業は要りません。
- ライブラリは `import` して使うだけなので、本体のライセンスはライブラリの影響を受けません。
- ライブラリを同梱した配布物（Release の実行ファイル）には、[配布物を作るとき](#配布物を作るとき)の作業が必要です。
  `packaging/build_release.py` が自動で行います。

## 実行に必要なライブラリ

| パッケージ | バージョン | ライセンス | 著作権者 |
|---|---|---|---|
| PySide6-Essentials | 6.12.0 | LGPL-3.0-only / GPL-2.0-only / GPL-3.0-only のいずれか（LGPL-3.0 を選択） | Qt for Python Team |
| shiboken6 | 6.12.0 | 同上 | Qt for Python Team |
| pyqtgraph | 0.14.0 | MIT（同梱のカラーマップは下記） | University of North Carolina at Chapel Hill |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | NumPy Developers |
| scipy | 1.18.1 | BSD-3-Clause | Enthought, Inc.、SciPy Developers |
| polars | 2.1.0 | MIT | Ritchie Vink |
| polars-runtime-32 | 2.1.0 | MIT | Ritchie Vink |
| pydantic | 2.14.1 | MIT | Pydantic Services Inc. and individual contributors |
| pydantic-core | 2.50.1 | MIT | Samuel Colvin |
| annotated-types | 0.8.0 | MIT | the contributors |
| typing-inspection | 0.4.4 | MIT | Pydantic Services Inc. |
| typing-extensions | 4.16.0 | PSF-2.0 | Python Software Foundation |
| colorama | 0.4.6 | BSD-3-Clause | Jonathan Hartley |

開発時だけ使う 26 パッケージ（pytest、mypy、Ruff、filterpy、その依存の matplotlib など）は、配布物に入れません。ライセンスの
表記に GPL のものはありません。弱いコピーレフトは `pathspec`（MPL-2.0）だけです。

## 注意が必要なもの

### PySide6（Qt）

LGPL-3.0 を選んで使います。実行時に読み込むのは QtCore、QtGui、QtWidgets、QtOpenGL、QtOpenGLWidgets、QtSvg、QtTest です
（pyqtgraph が読み込むものを含みます）。

依存は `pyside6-essentials` だけにしています。Qt のモジュールには、Qt Charts のように GPL-3.0 か商用でしか提供されないものが
あります。`PySide6` や `pyside6-addons` を入れるときと、新しい Qt のモジュールを使うときは、先に
[そのモジュールのライセンス](https://doc.qt.io/qt-6/qtmodules.html)を確認してください。

### numpy・scipy の同梱ライブラリ

OpenBLAS（BSD-3-Clause）、LAPACK（BSD-3-Clause-Open-MPI）、GCC ランタイム（GPL-3.0-or-later WITH GCC-exception-3.1）を
同梱しています。GCC ランタイムは例外条項により、使う側のプログラムのライセンスに影響しません。

### pyqtgraph のカラーマップ

`pyqtgraph/colors/maps` の CET カラーマップは CC-BY 4.0、viridis などは CC0 です。今のコードはどちらも使っていません。
周波数解析の図で CET カラーマップを使って配布するときは、クレジットの表示が必要です。

## 配布物を作るとき

ライブラリを同梱した実行ファイルには、次の作業が必要です。ソースのまま `uv` で使う間は要りません。Release の実行ファイル
（Windows・Ubuntu）は `packaging/build_release.py` が作り、1〜5 を自動で行います（GitHub Actions の `release.yml` から実行）。
手元で配布物を作るときも、このスクリプトを使ってください。

1. 同梱した全ファイルと、実行ファイル内のアーカイブに入れた全モジュールの出どころを照らし合わせ、出どころのパッケージの
   ライセンス文書を `licenses/` に入れる。出どころの分からないものがあれば止める。
2. Python 本体の `LICENSE.txt` と、Python に組み込まれた OpenSSL・libffi・zlib・expat・libmpdec などの文書を入れる。
   後者は Python の配布物に入っていないことがあるので、同じ版の CPython の `Doc/license.rst` を取得して入れる。
3. PySide6 について、[LGPL-3.0 第 4 条](https://www.gnu.org/licenses/lgpl-3.0.html)に従う。
   - Qt を LGPL-3.0 で使っていることを、`licenses/README.txt` と Release の説明に書く。
   - LGPL-3.0 と GPL-3.0 の全文（`packaging/licenses/`）を入れる。**PySide6 の wheel には含まれていない。**
   - 利用者が Qt を差し替えられるよう、Qt を別ファイルの DLL のまま配る（1 ファイルにまとめないフォルダ形式）。
   - 同じ版の Qt と PySide6 のソースの入手先を書く（実在を確かめる）。Qt の DLL を同梱すると Qt そのものの配布にもなるため
     （[GPL-3.0 第 6 条](https://www.gnu.org/licenses/gpl-3.0.html)）。入手先が消えた場合に備えるなら、ソースを Release に
     一緒に置く。
   - Qt の改変やそのデバッグのためのリバースエンジニアリングを禁じない（本体は MIT なので禁じていない）。
   - 画面に著作権表示を出すなら、Qt の著作権表示と、ライセンス文書の場所も一緒に出す（今の画面は出していない）。
4. OS のライブラリ（Windows の UCRT、Linux の `/usr/lib` のものなど）は同梱せず、利用者の OS のものを使う。
5. 実行に使わないもの（Qt のツール、開発用の道具、GNU readline と結び付く `readline` モジュール）は入れない。
6. CET カラーマップを使うようになったら、クレジットを表示する（今は使っていない）。

PyInstaller が実行ファイルに組み込むブートローダーは例外付きの GPL で、配布物のライセンスには影響しません。

## 依存を足すとき

実行に必要なライブラリを足すときは、ライセンスを確認して上の表に加えてください。GPL でしか使えないライブラリは、本体の
ライセンスに影響するので使う前に検討が必要です。一覧は次のコマンドで確認できます。

```bash
uv export --no-dev --no-hashes --no-emit-project
```
