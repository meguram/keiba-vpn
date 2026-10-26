# 予測成績（予測結果の集計）の運用

その日のレースが終わって結果が揃ったときに、予測と実際の着順を突き合わせて集計し、ホームの「予測ニュース」と
専用ページ「予測成績」（`/prediction-results`）に出す。

## 仕組み

| 項目 | 内容 |
|---|---|
| データ層 | `src/pipeline/inference/prediction_results.py`（純粋関数）。日次の集計を `data/local/prediction_results/<YYYYMMDD>.json` に保存（アトミック書き込み・冪等）。**日次が正本**で、週（月曜始まり）・月・シーズンは日次の合算 |
| 暫定/確定 | 速報結果（`race_result_on_time`）だけの日は `provisional`（暫定）。**全レースで確定結果＋公式の払戻がそろったら** `final`（確定）。確定済みの日を暫定で上書きしない |
| 対象の予測 | GCS の `race_predictions` → 無ければローカルの shadow / バックフィル（`src/api/local_predictions.py`。画面と同じ経路） |
| 指標 | ①◎（勝率の最大＋印）・予測1位の1着/連対/複勝圏 ②◎・○・★ の着順分布（1着/2着/3着/4着以下）③推奨馬券の仮想購入（既定 1R 上限 2,000 円、券種別）④注目レース（高オッズの的中・大きな払戻・大外れ）⑤勝ち馬の対数損失 |
| 印の定義 | ◎=`honmei`（1着優位・各レース1頭まで）、○=`pair`（2連系の相手）、★=`star`（10〜30倍帯で期待値が高い中穴・各レース1頭まで）。`assign_mece_marks` と同じ |
| 仮想購入 | `BettingOptimizer`（API の推奨と同じ。券種=単勝・複勝・馬連・ワイド・3連複）を、その時点で保存済みの最新（過去のレースは確定）オッズで解く。払戻は公式払戻（`race_result.payoff`）、無ければオッズからの概算（画面に「概算」と出る） |
| 欠損 | 予測なし・結果なしのレースは集計から除外し、件数を画面に出す |

## 自動実行（既定オフ）

環境変数 `KEIBA_PREDICTION_RESULTS=1`（ホストの `.env`）のときだけ動く。失敗しても本体のジョブは成功扱い（警告ログのみ）。

| 契機 | 場所 | 集計 |
|---|---|---|
| 開催日の最終レース後（速報結果が揃ったとき） | `raceday-result-runner` の終了処理（`auto_scrape_queue.py`） | 暫定 |
| 確定結果が取れたとき | `recent-retry`（`recent_retry.py`）。確定結果が保存できた日付だけ | 確定（払戻がそろった日） |
| 金曜の週次更新後 | `finalize_weekly_update`（`auto_scrape.py`） | 取り直し |

手動: `python -m src.scripts.data.prediction_results --date YYYYMMDD`（範囲は `--from/--to`、保存せず確認は `--dry-run`）。
ホストでは `nice -n 19` で直列に実行する。

## API

- `GET /api/prediction-results?scope=day|week|month|season&date=YYYYMMDD`（日別推移・KPI・レース別）
- `GET /api/prediction-results/news?limit=N`（ホストのカード用。`subtitle_public` は回収率を除いた文言）
- `GET /api/prediction-results/days`

**推奨度が標準以上の集計（2026-10-26）**: 回収率・購入・的中の集計は、**画面に出している推奨馬券（既定方策 target＝3連複・目標払戻、`delegates._optimize_target` と同じ計算）**を既定上限で購入したとみなした仮想購入だけを使い、**推奨度が標準以上（見送りを除く）のレースだけ**で出す。レース記録の `bets_t`（status/level/total_bet/total_return/by_type）がその記録で、`level` は画面のレース推奨度（0=見送り〜4）。従来の最適化（単勝・複勝・馬連など）の `bets` は参考として残すが、画面の集計には使わない。見送りを含む回収率は、どの画面・APIの集計でも出さない（ニュース・レース一覧・予測成績・注目レースの払戻・ホームの推奨度別）。会員・開発者の応答に `rated`（標準以上のレースの `kpi`・`summary`、件数 `counts`＝settled/rated/skipped/unrated、`legend`）と `series[].honmei_win_rated`・`roi_rated` が付く（匿名には tier.py の許可リストで出ない）。`bets_t` が無い（未記録・計算できなかった）レースは見送りと区別できないため標準以上に入れず `unrated` に数える。

**既存の日に `bets_t` を足す（初回と、モデル・推奨ロジックを変えたとき）**: 新しい日は集計時に自動で付く。保存済みの日（確定済みを含む）は次の CLI で追記する（印・着順・既存の数値は変えない。冪等。`nice` をつけて直列で）。追記後、GCS に置く運用なら `KEIBA_PREDICTION_RESULTS_GCS=1` で書き戻され、`--sync-gcs` で一括でも置ける。
```
nice -n 19 python -m src.scripts.data.prediction_results --from 20260101 --to 20261025 --add-target-bets
```

予測 API と同じ公開の GET。回収率（仮想購入）の表示は画面側で会員向けにしている。dev では、集計ファイルが無い日を dev ストアから都度作る。

## 反映・有効化・ロールバック

1. コードをホストへ反映（`config` や DB の変更は無い）→ API 再起動 → Next の再ビルド・切替。
2. 過去分のバックフィル（任意）: `nice -n 19 python -m src.scripts.data.prediction_results --from 20260101 --to 20261018`（開催日だけ保存される）。
3. `.env` に `KEIBA_PREDICTION_RESULTS=1` を追記して有効化（以降の日次が自動で集計される）。
4. ロールバック: `.env` の行を消す。保存ファイルは `data/local/prediction_results/` を消せば画面は「集計なし」になる。ホーム・ページ自体は `frontend/lib/features.ts` の `prediction-results` を `enabled:false` にすると隠せる。

## 注意

- 確定結果・確定払戻は、数日遅れることがある（金曜の weekly-update ごろ）。それまでは「暫定」。
- 仮想購入は「その日の最新オッズで選んだ買い目」の精算であり、レース前に実際に買えたとは限らない（購入時点のオッズは確定と違う）。参考値。
- バックフィル／shadow の予測を含む日は、画面にその旨を出す。
