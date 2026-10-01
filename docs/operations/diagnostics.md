# 環境別の調査スクリプトと判断ツール

**目的**: 「学習PCでしか分からないこと」「VPSでしか分からないこと」を、専用スクリプトで測ってレポート(JSON)にし、
開発PC（このリポジトリの作業環境）で読み込んで設計上の判断を出す。

```
学習PC:  diagnose_training_pc ──▶ reports/training_pc.json, dynamic_feature_labels.csv ─┐
VPS:     diagnose_vps         ──▶ reports/vps.json                                      ├─▶ 開発PC: decide ──▶ reports/decision.md
                                                                                        ┘
```

レポートに**秘密値は含まれない**（環境変数は設定の有無のみ）。各スクリプトは読み取りのみで、ストアやモデルを変更しない。
調査項目のどれかが失敗（接続不可など）しても他の項目は続行し、失敗は `error`/`skip` として記録される。

## 1. 学習PCで実行

```bash
# リポジトリのルートで（特徴量ストアとモデルがあるPC）
python -m src.scripts.diagnose.diagnose_training_pc --out reports/training_pc.json \
    --model-dir models/ensemble --sample-races 3 --feature-sample 200
```

| 調査項目 | 内容 | 判断に使う先 |
|---|---|---|
| environment | メモリ・CPU・ライブラリ版 | VPSとの版一致 |
| feature_store | 列数・年・合計サイズ・ブロック別列数 | 配置・容量 |
| parquet_layout | 行グループ数・race_idの統計の有無 | 行フィルタ読み込みが効くか |
| feature_row_read | **実データで** `load_rows_for_keys` と `load_columns` の時間・メモリを比較、1000列換算 | 推論時の特徴量の読み方 |
| model | モデル容量・ロード時間・推論ピークRSS（別プロセスで実測） | モデルの配布・メモリ |
| builder | 本ビルダーが実装済みか（疑似か） | 次の実装作業 |
| dynamic_label_template | 特徴量ごとの CSV（`guess` は名前からの推定） | 動的特徴量の割合 |

**動的特徴量のラベル付け**: `reports/dynamic_feature_labels.csv` の `label` 列に、T-45時点で初めて決まる/変わる特徴量
（オッズ・馬体重・馬場状態・取消・騎手変更など）は `dynamic`、それ以外は `static` を記入する（空欄は `guess` を暫定値として使う）。
既存のCSVは上書きしない。

## 2. VPSで実行

```bash
# サービス稼働中の状態で（空きメモリは「稼働中の余裕」を見るため）
KEIBA_ENV=prod python -m src.scripts.diagnose.diagnose_vps --out reports/vps.json \
    --model-dir /path/to/model --netkeiba-trial
```

| 調査項目 | 内容 | 判断に使う先 |
|---|---|---|
| system | 空きメモリ・スワップ・CPU・ディスク・ライブラリ版 | 推論の実行場所、版一致 |
| processes / redis | 常駐プロセスのメモリ上位、Redis使用量 | 余裕の内訳 |
| gcs_latency / cloud_sql_latency | GCS 1オブジェクト取得、`SELECT 1` の遅延 | ページ表示設計 |
| netkeiba | **`--netkeiba-trial` 指定時のみ**、VPSのIPから1リクエスト | スクレイピングをVPSで動かせるか |
| inference | 推論のピークRSS・ロード・予測時間（`--model-dir` 無しなら1000特徴量の疑似モデルを生成して測定） | 推論の実行場所 |

実データの出馬表での計測は従来どおり `measure_inference_memory --race-id <ID>`。開催日の繁忙時間帯にもう一度測るとよい。

## 3. 開発PCで判断

```bash
python -m src.scripts.diagnose.decide --training reports/training_pc.json --vps reports/vps.json \
    --labels reports/dynamic_feature_labels.csv --out reports/decision.md
```

片方しか無い場合は、その分だけ判断し、残りは「未判定（どの環境でどのコマンドを実行するか）」を表示する。

| 判断 | 入力 | 主なルール（しきい値は `decide.THRESHOLDS`） |
|---|---|---|
| T-45推論の実行場所 | VPS system/inference | 余裕（空き−ピーク）300MB以上→VPS、100〜300→上限付きVPS、100未満→GCP |
| スクレイピングをVPSで | VPS netkeiba | 403/429/503→案Bを再検討、取得可→案A確定 |
| ライブラリ版の一致 | 学習PC・VPSのlibraries | major.minor不一致→requirements固定・再ビルド |
| 推論時の特徴量の読み方 | 学習PC feature_row_read/parquet_layout | 1000列換算3秒/レース以内→`load_rows_for_keys`、超→レース単位スナップショット |
| ページ表示の遅延 | VPS gcs/cloud_sql | GCS中央値×5件が500ms超→集約オブジェクト＋先読み、DB中央値50ms超→キャッシュ |
| モデルの容量 | 学習PC model | 500MB超→容量削減を検討。配布の送信費も表示 |
| 動的特徴量の割合 | ラベルCSV | 10%以下→静的部分を前日に事前計算、超→二段構え |
| 特徴量ビルダー | 学習PC builder | 疑似のまま→本実装が必要 |

判断結果（`decision.md`）の「次の作業」を、`docs/git_management/todo/` の該当項目に反映する。
