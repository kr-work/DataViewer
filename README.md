# Experiment Workbench

研究実験で得た任意の時系列データを、読み込む・確認する・同期する・処理する・比較する・再現するための
汎用デスクトップツールです。特定のセンサには依存しません。
フィルタ（バターワース・移動平均など）、微分・積分、カルマンフィルタ系（KF / EKF / 適応 KF）などの
解析を、保存・再適用できる形でかけられるようにすることを目指しています。

製品要件は [docs/SPEC.md](docs/SPEC.md)、設計判断の記録は [docs/DECISIONS.md](docs/DECISIONS.md) を参照してください。

## 現在の状態

**Phase 2（取り込みから縦並べ表示まで）まで実装済み。**
GUI でデータを取り込み、時間軸を連動させた縦並べ表示で比較し、表示を保存・復元できます。
解析コアは Python から使えます（GUI からの解析条件の編集は次の Phase 3）。

| 実装済み | 内容 |
|---|---|
| GUI: 取り込み | CSV / TSV / Parquet。プレビュー、時刻列・単位・チャンネル・登録先の選択、異常の表示 |
| GUI: 縦並べ表示 | 複数パネル、時間軸のズーム・パン・カーソルの連動、カーソル位置の値、表示の保存と復元 |
| 取り込みの検証 | 空・列名重複・文字コード違い・数値でない値はエラー、NaN・時刻の逆行/重複/飛びは警告 |
| 正規化データ | 元ファイルのコピーと SHA-256、秒に換算した時刻と値の Parquet（元ファイルは変更しない） |
| 解析: 入力検証 | 時刻の逆行・重複、NaN、不等間隔（欠損による時刻の飛び）を説明付きのエラーにする |
| 解析: フィルタ | バターワース（LPF / HPF / BPF / BSF、ゼロ位相・因果的）、移動平均、Savitzky-Golay 平滑化 |
| 解析: 微分・積分 | 中心差分（不等間隔対応）・Savitzky-Golay 微分（1・2 階）、累積台形則・シンプソン則 |
| 解析: カルマンフィルタ | 線形 KF、EKF（非線形関数とヤコビアンを指定）、適応 KF（観測雑音 R の逐次推定） |
| ドメインモデル | Project / Experiment / Session / Trial / Source / Channel / Event、解析条件（pydantic） |
| プロジェクトフォルダ | 作成・オープン、`project.json` の形式バージョン確認、相対パスの安全な解決 |
| SQLite メタデータ | スキーマ初期化、バージョン管理とマイグレーション、整合性制約 |
| 合成データ生成 | 真値付きの汎用時系列（2 種類の周波数・時刻原点）と、取り込みエラー検証用の失敗データ |
| 品質 | pytest / Ruff / mypy（strict）、GitHub Actions（Windows・Ubuntu） |

## 必要なもの

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/)

## セットアップ

Windows（PowerShell）・Ubuntu（bash）とも同じコマンドです。

```bash
git clone <このリポジトリ>
cd DataViewer
uv sync
```

`uv sync` で `.venv` に依存パッケージと開発ツールがインストールされます。

Ubuntu で GUI が起動しない場合は、Qt が使うシステムライブラリを入れてください。

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libxcb-cursor0 libfontconfig1 libdbus-1-3
```

## GUI の起動と基本操作

```bash
uv run ewb                       # 起動
uv run ewb path/to/my-experiment # プロジェクトを開いて起動
```

1. 「ファイル → 新規プロジェクト」でプロジェクトを作る（既存なら「プロジェクトを開く」）。
2. 「データ → ファイルを取り込む」でファイルを選び、ダイアログで時刻列・単位・チャンネル・登録先を選ぶ。
   エラーのあるファイルは取り込めず、理由がダイアログとメッセージ欄に表示されます。
3. 左のツリーでチャンネルをダブルクリックすると、選択中（枠が青い）のパネルに表示されます。
   右クリックで新しいパネルにも追加できます。パネルは「表示 → パネルを追加」で増やせます。
4. ズーム・パンとカーソル（赤い縦線、ドラッグで移動）は、時刻を比較できるパネルどうしで連動します。
5. 表示は閉じるとき（または「ファイル → 表示を保存」）に保存され、次に開いたときに復元されます。

時刻を比較できるのは、同じファイルの列どうしと、取り込み時に「時刻が同期済み」とチェックした
同じ試技のファイルどうしだけです。それ以外のファイルを並べると、そのパネルは時間軸を連動させず、
警告を表示します（異なる装置の時刻を勝手に揃えないため）。

## 開発コマンド

```bash
uv run pytest                 # テスト
uv run ruff check .           # Lint
uv run ruff format .          # フォーマット（確認だけなら --check）
uv run mypy                   # 型チェック（src と tests）
```

## 解析コアの使い方

時刻 [s] と値の NumPy 配列を渡します。処理の条件は pydantic モデルで指定します。
時刻が秒以外の単位なら、渡す前に秒へ換算してください。

```python
import numpy as np
import polars as pl

from experiment_workbench.analysis import (
    ButterworthFilter,
    Differentiator,
    Integrator,
    KalmanFilter,
    KinematicSystem,
)
from experiment_workbench.model import (
    ButterworthKind,
    ButterworthParams,
    DerivativeParams,
    IntegralParams,
    KalmanInitialState,
    KinematicSystemParams,
)

frame = pl.read_csv("data/synthetic/trial_001/high_rate.csv")
t = frame["time_s"].to_numpy().astype(np.float64)
accel = frame["accel"].to_numpy().astype(np.float64)

# 5 Hz のゼロ位相ローパスフィルタ → 積分
lowpass = ButterworthFilter(ButterworthParams(kind=ButterworthKind.LOWPASS, cutoff_hz=5.0))
smoothed = lowpass.apply(t, accel)
velocity = Integrator(IntegralParams(initial_value=0.0)).apply(t, smoothed)
jerk = Differentiator(DerivativeParams()).apply(t, smoothed)

# ミリ秒の時刻を秒に換算し、位置の観測からカルマンフィルタで位置・速度を推定
low = pl.read_csv("data/synthetic/trial_001/low_rate.csv")
t_low = low["time_ms"].to_numpy().astype(np.float64) / 1000.0
position = low["position"].to_numpy().astype(np.float64)
system = KinematicSystem(
    KinematicSystemParams(process_noise_density=10.0, measurement_noise_std=0.01)
)
initial = KalmanInitialState(state=[position[0], 0.0], covariance=[[1e-4, 0.0], [0.0, 10.0]])
result = KalmanFilter(system, initial).run(t_low, position)
estimated_position, estimated_velocity = result.states[:, 0], result.states[:, 1]
```

- 適応 KF にするには `KalmanFilter(..., adaptive=AdaptiveNoiseParams(method=...))` を渡します。
- EKF にするには `NonlinearSystem` に状態遷移・観測の関数とそのヤコビアンを渡します。
- 入力が前提を満たさない場合は `AnalysisInputError` になります。例えば NaN を含む信号にフィルタをかけた場合や、
  欠損区間のある信号に等間隔前提の処理をかけた場合です。値を勝手に補間・除去することはありません。
  カルマンフィルタだけは、NaN の観測を欠測として扱い、予測でつなぎます。

## 合成データの生成

```bash
uv run ewb-synthetic --output data/synthetic                 # 既定: 180 秒 × 3 試技
uv run ewb-synthetic --output data/synthetic-short --duration 10 --trials 3 --seed 1
```

出力先は空のディレクトリを指定してください（既存ファイルを上書きしないよう、空でなければ中止します）。
`data/` は `.gitignore` 済みです。

```text
data/synthetic/
├── manifest.json        # 生成条件と真値（運動のパラメータ、ノイズ、時刻オフセット、仕込んだ異常）
├── trial_001/
│   ├── high_rate.csv    # time_s, accel, true_accel, true_velocity, true_position（既定 1000 Hz）
│   └── low_rate.csv     # time_ms, position, true_position, true_velocity（既定 100 Hz）
├── trial_002/ ...       # 一部区間の行が欠けている（時刻の飛び）
├── trial_003/ ...       # position の一部区間が NaN
└── broken/              # 空・ヘッダーのみ・列名重複・cp932・NaN/空欄・時刻逆転・時刻重複
```

- 観測列（`accel`・`position`）にはノイズなどが乗っています。解析の入力に使います。
- 真値列（`true_` で始まる列）は解析的に計算した正解です。フィルタ・積分・カルマンフィルタの結果の評価に使います。
- 2 つのファイルは周波数・時刻原点・時刻の単位が異なります。時刻を揃える処理は明示的に行う前提です。

## プロジェクトフォルダの構成

```text
my-experiment/
├── project.json       # プロジェクト情報と保存形式バージョン
├── metadata.sqlite    # 実験 / 試技 / Source / Channel / Event
├── originals/         # 取り込んだ元データのコピー（Source ごとのフォルダ）
├── datasets/          # 正規化データ（Parquet。秒に換算した時刻 __time_s と各チャンネル）
├── derived/           # 処理結果のキャッシュ
├── pipelines/         # 処理定義（JSON）
├── views/             # 表示設定（JSON）
└── exports/           # CSV、画像等
```

メタデータ内のパスはすべてプロジェクトルートからの相対パスなので、フォルダを移動しても開き直せます。

## ソースコードの構成

```text
src/experiment_workbench/
├── model.py               # pydantic モデルのみ（データ受け渡し用の型・解析条件・解析結果・表示状態）
├── validation.py          # モデルの検証関数・定数、NumPy 配列の型の別名
├── timebase.py            # 時刻の単位換算と、Source どうしの時刻を比較できるかの判定
├── importer.py            # データファイルの取り込み（プレビュー・異常の検出・正規化・登録）
├── analysis/
│   ├── signal_processing.py  # 入力検証・フィルタ・微分・積分
│   └── kalman.py             # 線形 KF・EKF・適応 KF
├── storage/
│   ├── database.py        # SQLite メタデータ（スキーマ・マイグレーション・保存/取得）
│   └── project_store.py   # プロジェクトフォルダ、元データ・正規化データ・表示設定の読み書き
├── ui/
│   ├── main_window.py     # メイン画面と起動（ewb コマンド）
│   ├── plot_workspace.py  # 縦並べ表示
│   └── import_dialog.py   # 取り込みダイアログ
└── synthetic.py           # 合成データ生成（ewb-synthetic コマンド）
tests/
├── unit/                  # 単体テスト
├── integration/           # 結合テスト（保存と再オープン、取り込みから解析まで）
└── gui/                   # GUI のテスト（pytest-qt。画面は出さない）
```

計算・保存・時間軸の判定のコードは Qt に依存しません。GUI は `ui/` に分けています。
コーディング規約（`model.py` はモデルのみ、`typing` は使わない など）は `docs/DECISIONS.md` にあります。
