# 追走難度 v2 の前夜の保存と、実測との週次の突き合わせ

追走難度 v2（tp-54、`src/pipeline/models/tracking_difficulty_v2.py`）を、毎週の開催前夜に定期実行で算出して保存し、当日は保存した固定値を返す。
あわせて、保存した予測を実測（上がり）と突き合わせる集計を週次で出し、精度の推移を追えるようにする。

## 仕組み

| 項目 | 内容 |
|---|---|
| 有効化 | `KEIBA_TRACKING_V2=1`（ホストの `.env`）。無いときは何もしない（既存の追走難度 v1 は別系統で、変わらない） |
| 前夜の保存 | raceday-eve（毎日 18:00 JST、翌日が開催日のとき）で、出馬表・馬柱の取得の後、v1 の事前計算の直後に `precompute_tracking_v2_for_race_ids`（`src/pipeline/inference/tracking_v2_service.py`）。キュー版（`auto_scrape_queue._eve_precomputes`）と旧版（`auto_scrape.run_raceday_eve_for_date`）の両方に入っている。**失敗しても eve は成功扱い**（例外は警告ログ、1 レースの失敗は他に波及しない）。結果は eve の戻り値の `tracking_precompute.tracking_v2`（ok/skip/fail/skip_reasons） |
| 保存先 | `data/local/inference/tracking_v2/<開催日>/<race_id>.json`（`KEIBA_TRACKING_V2_DIR` で変更可）。前夜の値。取消で作り直した値は `<race_id>.recomputed.json`（前夜の保存は消さない） |
| 保存の中身 | version、race_id、date、コース、馬数、`horse_numbers`（署名）、`computed_at`（JST）、`source`（raceday_eve / recomputed / backfill）、入力の取得時刻（`card_fetched_at` / `past_fetched_at`）、モデルと設定のハッシュ、位置取り側のペース、整合層のペース、各馬の `tracking_v2`（ease_pct・label・drain_sec・demand_sec・capacity_sec）・想定位置・入力（w_adj など） |
| 作れないレース | 位置取り側のペースが unknown（`pace_unknown`）、モデル・設定が無い、出馬表が無い。保存せずスキップ（eve の `skip_reasons` に理由） |
| API | `GET /api/race/{race_id}/predictions` の各馬 `tracking_v2`。保存があって**いまの出走馬の馬番の集合が保存時と同じ**なら保存の値（`tracking_v2_meta.source="saved"`）。応答の形は従来と同じ（`tracking_v2_meta` に `source`・`computed_at` が足されるだけ）。保存が無ければ従来どおりその場で計算（`source="live"`） |
| 当日の変動の扱い | 取消・除外・追加で馬番の集合が変わったレースだけ、その場で計算し直して `.recomputed.json` に別保存し、それを返す（`source="recomputed"`、2 回目以降は `recomputed_saved`）。馬体重の発表では**動かさない**（固定値）。`KEIBA_TRACKING_V2_RECOMPUTE=none` なら取消でも前夜の値のまま（`saved_stale`）、`always` なら毎回その場計算 |

## 週次の突き合わせ

実測のラベル Y = （実測の上がり3F − そのレースの中央値）− その馬の過去 5 走の「位置取りの効果を除いた上がりの力」w_adj（保存時の入力）。Y が大きい＝終いが落ちた。
保存した drain_sec（予測 Y）・ease_pct と Y を突き合わせる（tp-54 の学習のラベルと同じ定義）。

| 項目 | 内容 |
|---|---|
| コード | `src/pipeline/inference/tracking_v2_eval.py`。手動: `python -m src.scripts.data.tracking_v2_eval --date YYYYMMDD` / `--from A --to B` / `--week YYYYMMDD`（`--dry-run` は保存せず表示） |
| 指標 | 予測 vs 実測の相関（Pearson・Spearman）、ease_pct と Y の相関（負が理想）、十分位別の実測 Y と単調性（十分位平均の Spearman、上位−下位の差）、ease_pct の分布（60 以上・30 未満の割合）、面・距離帯別、保存の被覆（保存したレース数/開催レース数）、v2 の付与率（馬単位）、作り直したレース数 |
| 保存先 | `data/local/inference/tracking_v2_eval/{YYYYMMDD}.json`（日次。pairs 付き）、`weekly/{週の月曜}.json`（週次）、`_trend.json`（週ごとの要約の推移）。`KEIBA_TRACKING_V2_EVAL_DIR` で変更可 |
| 自動実行 | `KEIBA_TRACKING_V2_EVAL=1` のとき、金曜の weekly-update の後（`finalize_weekly_update`）に直近の対象日を突き合わせ（`out["tracking_v2_eval"]`）。**失敗しても本体は成功**。既定オフ |
| cron | `scripts/cron/tracking_v2_weekly.sh`（直近 10 日を突き合わせ。環境変数に依らず実行）。登録は `bash scripts/cron/install_tracking_v2_cron.sh`（月曜 08:30 JST、タグ `KEIBA-TRACKING-V2`、登録前に crontab を `data/local/meta/cron_backups/` に退避）。**このリポジトリでは登録しない**（ホストで root が実行） |

## 過去分のバックフィル

`python -m src.scripts.data.tracking_v2_backfill --from 20260101 --to 20261018`（保存済みは飛ばす。`--overwrite` で作り直し、`--dry-run` で数だけ）。
保存済みの出馬表・馬柱から作る（as-of は馬柱の対象日より前の走）。ホストでは `/tmp/heavy_run.sh <ラベル> …` か `nice -n 19` で直列に実行する。
バックフィルで作った保存は `source="backfill"`。前夜の本番の保存とは区別して集計できる。

## 反映・ロールバック

1. コードをホストへ反映（`tracking_v2_service.py`・`tracking_v2_eval.py`・`tracking_difficulty_v2.py`・`app.py`・`auto_scrape_queue.py`・`auto_scrape.py`）→ API 再起動。
2. 過去分のバックフィル（任意）→ 週次の突き合わせを手動で一度回して数値を確認。
3. `.env` に `KEIBA_TRACKING_V2_EVAL=1`（自動の突き合わせ）を追記。cron を使う場合は `install_tracking_v2_cron.sh`。
4. ロールバック: `.env` の `KEIBA_TRACKING_V2_EVAL` を消す（突き合わせが止まる）。`KEIBA_TRACKING_V2` を外せば保存も API の v2 も止まる（保存ファイルは残るだけで無害）。

## 注意

- 馬体重（`weight`）は出馬表にあれば入力に使われるが、`weight_change` は常に欠損。前夜は `weight` が無いことが多く、当日の発表後に値が変わりうるが、固定値の方針としてそのままにしている（影響の大きさは週次の突き合わせで見る）。
- 障害レースは平地の走りで学習した値のため低く出すぎる（画面側で非表示）。集計は面・距離帯別で障害を分けて見る。
- 突き合わせの Y は結果が確定してから作れる。暫定結果の日は上がりが無いことがあり、`no_results` になる（確定後に再実行）。
