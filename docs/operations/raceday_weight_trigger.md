# 馬体重の検知による予測の起動（本番ランナーの追加分。2026-10-14 準備）

> **最新の設計は `docs/html/operations/raceday_prediction_workflow.html`（毎レースの予測出力までのワークフロー）と `docs/html/operations/cron_schedule.html`（cron 登録内容）を参照。このメモは当時（2026-10-14）の記録で、以降の変更（馬体重ランナーの cron 登録、v4 の影モード、確定結果の橋渡し、サービング状態の事前計算など）は反映していない。**

**状態（2026-10-19 更新）: ホストの cron に登録済み（23:00 UTC = 08:00 JST、開催日のみ動く）。以下の「未起動」は当時の記述。**既存の `raceday-runner`（T-60 オッズ・T-15 バンドル）は変更していない。
方針（ユーザー指示 2026-10-14）: 予測は馬体重の発表後（基本は発走の約 1 時間前）にだけ回し、前日に準備できる部分は先に進めて処理時間を短くする。

## 追加したもの

| 項目 | 場所 | 内容 |
|---|---|---|
| 馬体重の検知ランナー | `auto_scrape --task raceday-weight-runner`（`auto_scrape_queue.task_raceday_weight_runner`、純粋ロジックは `src/scraper/raceday_weight_watch.py`） | 各R **T-65〜T-30** に出馬表（`race_shutuba`）を 90 秒間隔で取り直し、9 割以上の馬に馬体重が入ったら起動。期限（T-30）までに入らなければ馬体重なしで起動 |
| 推論モード | `src/pipeline/features/horse_history/inference.py` | 出馬表の行を結果なしの走として履歴に足し、学習と同じ `build_from_runs` で 127 特徴量を作る |
| 起動処理 | `src/pipeline/inference/weight_triggered_predict.py` | 履歴は起動時に 1 回だけ読み込み、馬体重が揃ったレースをまとめて計算。出力は `data/local/inference/features/<日付>/<race_id>.parquet`、記録は `data/local/inference/events/<日付>.jsonl` |
| tier 表 | `config/modeling/feature_tiers.json` | 特徴量ごとの確定時点（0 前日 110 個 / 1 当日朝 7 個 / 2 T-60 馬体重 10 個）。摂動法の実測 |
| 履歴テーブルの定期エクスポート | `scripts/cron/export_tables_daily.sh` | 当年（1 月は前年も）の `race_result,race_shutuba,race_result_lap,race_result_on_time` を増分エクスポート。二重起動はロックで防ぐ |
| 確認ツール | `python -m src.scripts.data.verify_inference_features --day 2024-07-06` | 推論モードの特徴量を学習時の特徴量と突き合わせ、所要時間を出す |

馬体重の検知は `race_shutuba` の取得だけで、**ログイン不要**（rescrape のセッションとは衝突しない）。

## 起動・停止（ホスト PC。root で、作業ディレクトリ `/home/jovyan/work/keiba-vpn`）

```bash
# 開催日の朝に 1 回起動（常駐して全レースの起動が済んだら終了する）
nohup bash scripts/cron/run_auto_scrape_logged.sh "$PWD" raceday-weight-runner logs/raceday_weight_runner.log >/dev/null 2>&1 &
# 状態: tail -f logs/raceday_weight_runner.log ; tail data/local/inference/events/$(date +%Y%m%d).jsonl
# 停止: pkill -f 'auto_scrape.*--task.*raceday-weight-runner'
```

自動起動にする場合のホストの crontab 行（UTC。`raceday-runner` と同じ 22:30 UTC = 07:30 JST）:

```
30 22 * * * cd /home/jovyan/work/keiba-vpn && TZ=Asia/Tokyo bash scripts/cron/run_auto_scrape_logged.sh /home/jovyan/work/keiba-vpn raceday-weight-runner logs/raceday_weight_runner.log # KEIBA-VPN-ALL
30 21 * * * cd /home/jovyan/work/keiba-vpn && TZ=Asia/Tokyo bash scripts/cron/export_tables_daily.sh /home/jovyan/work/keiba-vpn # KEIBA-VPN-ALL
```

（`export_tables_daily` は、結果の取り込みが済む時間帯に 1 日 1 回。上の 21:30 UTC = 06:30 JST は例。）crontab の変更は承認後に行う。

## 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `KEIBA_WEIGHT_TRIGGER_MODE` | `features` | `features`: 特徴量の作成まで。`predict`: 加えて既存の予測（`race_day_workflow.predict_race`、保存先は従来どおり）も起動 |
| `KEIBA_WEIGHT_WATCH_START_MIN` / `_DEADLINE_MIN` | 65 / 30 | 窓の開始／期限（発走の何分前） |
| `KEIBA_WEIGHT_WATCH_POLL_SEC` | 90 | 取得の間隔（下限 30） |
| `KEIBA_WEIGHT_WATCH_MIN_RATIO` | 0.9 | 馬体重が入った馬の割合の閾値 |
| `KEIBA_WEIGHT_WATCH_DATE` / `_REPLAY` | 未設定 | リハーサル。`DATE=YYYYMMDD REPLAY=1` で、取得も待機もせず保存済みの出馬表だけで一巡する |
| `KEIBA_INFERENCE_DIR` | `data/local/inference` | 出力先 |

## リハーサル（取得なし）

```bash
KEIBA_ENV=stg KEIBA_WEIGHT_WATCH_DATE=20260613 KEIBA_WEIGHT_WATCH_REPLAY=1 /opt/conda/bin/python3 -m src.scraper.auto_scrape --task raceday-weight-runner
```

## 注意点・未解決

- **最終モデルは未学習**（tp-04 以降）。現状は特徴量の作成までが本番相当で、`predict` モードは従来の疑似特徴量ビルダーによる予測を呼ぶだけ。最終モデルができたら `weight_triggered_predict.run_for_races` に推論を足す。
- **履歴に 2026 年を含める**（推論は最新の履歴が要るため。学習には使わない）。2026 年のテーブルは取得が不完全で、`export_tables_daily` を回さないと古いまま。起動時に `history_last_date` をログ・イベントに出すので、前日までの結果が入っているか確認する。
- **計算時間（実測、2024-07-06 の 36 レース・422 頭、ホスト PC）**: 履歴の読み込み 3.0 秒（起動時に 1 回）＋特徴量の計算 約 16 秒（レースをまとめて 1 回）。目標 30 秒・上限 120 秒に収まるため、tier 0 の前日キャッシュは作っていない（必要になったら追加）。
- 摂動法の tier 分類は 1 開催日（2024-07-06）の実測。`tests/pipeline/test_weight_triggered_predict.py` が、馬体重の有無で tier 0 の特徴量が変わらないことを確認している。
- 同日の先行レースの時計（`sr_var_today` / `sr_var_n`）は、発走 90 分以上前のレースだけを使う設計だが、推論モードでは当日の結果がテーブルに入っていないため、前の開催日の値で代替される（学習時より情報が少ない側に倒れる）。**以前は代替も効かず全行が欠損になっていた**（学習と本番のずれ）ため、2026-10-14 に直した。

## 実データでの突き合わせ結果（2024-07-06、`verify_inference_features`）

出馬表の行から推論モードで作った特徴量を、同じ日の学習時の特徴量（結果込み）と比べた。馬体重を入れた状態で、出走取消・競走中止の馬を除くと、不一致は次の 8 個だけ。

| 区分 | 不一致 | 原因 |
|---|---|---|
| tier 2（馬体重） | 0 個 | 一致 |
| tier 1 | `sr_var_today`（405 行）、`f1_sim_res`・`f1_sim_z`・`f1_sw`（3 行） | 同日の先行レースの結果が無い（想定どおり）。馬場状態は出馬表の取得時点の値で、確定値と異なるレースがある |
| tier 0 | `sr_var_n`（304 行）、`b_aff_class`・`b_n_class`（7 行）、`c_cls_step`（9 行） | `sr_var_n` は同日の先行レースの件数（想定どおり）。クラス系は、結果表の `grade` が `race_class` と食い違う少数のレース（1 勝クラスなのに 2 勝・3 勝と書かれる）で、学習ラベル側の表記ゆれ |

この突き合わせで見つかり、**修正した**もの（`normalize_card`）:

- 出馬表の `grade` は、1〜3 勝クラスが空欄、重賞（G1〜G3・L）は一律 `OP`。そのままだとクラスの数値が 1.0 に落ちる → `race_class` から導き、重賞は履歴のレース名から格を引く（新設の重賞は `OP` のまま）。
- 出馬表の馬場状態は略称（稍・不）。そのままだと良に落ちる → 稍重・不良に直す。
- 馬体重が揃ったレースで体重の無い馬は、出走取消・除外とみなして外す（イベントの `dropped_no_weight`）。**出馬表のパーサーは取消馬を扱っていない**ため、取消馬が出馬表にどう載るかは 10/17〜18 の実運用で確認する。

**学習と推論のずれ（未対応。tp-03 で両案を比較して決める）**: 学習時の `load_runs` は着順が付かなかった馬（競走中止・除外など。この日は 3 頭 / 422 頭）を履歴から落とすため、レース内で相対化する特徴量（`c_fs_step`・`c_opp_step`・`f2_*`・`f3_*` の 29 行）は「完走馬だけの出走馬」で計算されている。推論では全出走馬になる。差は小さい（この日は 7% の行）が、学習側で非完走馬を出走馬に含める案（着順が必要な集計からは除く）を、tp-03 で「含める／含めない」の両方で比較する。

## ホストでのリハーサル結果（2026-10-14、`KEIBA_WEIGHT_WATCH_DATE=20260613 KEIBA_WEIGHT_WATCH_REPLAY=1`）

36 レースを 1 回のまとめ計算で処理し、特徴量の計算 約 20 秒、ランナー全体で 32 秒（履歴の読み込み 3 秒を含む）。全レース `ok`、体重の無い 1 頭を除外（`dropped_no_weight`）。出力は `features/<日付>/<race_id>.parquet` と `events/<日付>.jsonl`。

リハーサルで見つかり、修正したこと:

- **終了後に取得した出馬表はレース情報が空**（面・距離・日付・格・馬場状態）。2026 年 7 月以降に保存された出馬表の多くがこの状態（例: 10/11 開催分）。出走前の出馬表（10/17 東京5R で実際に取得して確認）は、面・距離・日付・格をパーサーが正しく取れる。発走時刻・天候・馬場状態は、当日になってページに載る。→ 日付と発走時刻は発走時刻表で補い、面・距離が空のレースは `error` イベントにして除外する（推論モードに入れると Elo の日付ループが止まらなくなるため、`build_inference_features` でも弾く）。
- 空の列（全部 null）で正規化が落ちる不具合。
- 履歴の最終日（`history_last_date`）が 2026-10-04 のまま。**`export_tables_daily` を回していないため**で、起動前に増分エクスポートを 1 回実行しておくこと。

実運用の初回（10/17〜18）に見るもの: `events/<日付>.jsonl` の `weights_known`（検知が T-65〜T-30 の窓に収まったか）、`dropped_no_weight`（取消馬が出馬表にどう載るか）、`going_known`、`build_sec`。

## 小さなメモリのマシンで動かす（サービング用バンドル、2026-10-14）

- `KEIBA_INFERENCE_ENGINE`: `auto`（既定。バンドルがあればバンドル、無ければ全履歴）／`bundle`（無ければエラー）／`full`（全履歴）。
- バンドルは `feature-accumulate`（毎日 07:40 JST）が `data/local/modeling/feature_store/serving_bundle/`（約 140MB）に作る。`KEIBA_SERVING_BUNDLE_DIR` で場所を変えられる。
- バンドル方式の実測（ホスト）: 特徴量の計算 約 340MB（固有 約 160MB）・0.6 秒。全履歴方式は 2.4GB・約 20 秒。結果は全 127 特徴量で一致する。設計・検証: `docs/html/modeling/chapters/feature_serving_low_memory.html`。
- 起動時のログとイベントに `engine`・`history_last_date` が出る。履歴が 10 日より古いと警告する。
- プロセス全体では約 430MB（GCS クライアント等の import を含む）。400MB の余裕に収めるには、取得と計算のプロセスを分ける必要がある（未実施）。
- 大きなマシンから小さなマシンへのバンドルの配布は未整備。
