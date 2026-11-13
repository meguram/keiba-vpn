# 削除した Cloud Scheduler: 停止中の keiba-g-* 13 件（2026-11-13）

## 理由
- GCP の並走 Jobs（keiba-dispatch-s/m/l/xl）を起こす Scheduler。費用の歯止め（`src/ops/cost_guard.py`）と 11/12 の判断で、すべて PAUSED になっていた。
- Cloud Scheduler は**停止中のジョブにも 1 件 約 15 円/月**かかる（課金エクスポートの `Cloud Scheduler Jobs` の数量が、停止後も 17 件のまま）。13 件で月 約 195 円。
  月 2,000 円の上限（CLAUDE.md）を守るため、使っていない停止中のジョブを消す（費用の台帳 `docs/operations/cost_ledger.md` の #3）。
- ユーザーの承認: 2026-11-13 14:1x JST（統括役のセッション経由）。
- 残したもの: `keiba-g-0730-shadow`（ENABLED。費用の日報と影モードの日次集計。cost_guard の `KEEP`）。
- 消した後は、cost_guard が止める対象（名前が `keiba-g-` で始まり KEEP 以外）が 0 件になる。並走 Jobs を再開するときは、下の手順で作り直す。

## 消したジョブ（場所 asia-northeast1、プロジェクト horse-racing-466415、タイムゾーン Asia/Tokyo。消す前はすべて PAUSED）
| 名前 | スケジュール | 対象 | 本文（base64 を戻したもの） | 説明（description） |
|---|---|---|---|---|
| keiba-g-0530 | `30 5 * * *` | `jobs/keiba-dispatch-l:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0530"]}]}}` | 騎手・調教師統計 |
| keiba-g-0700 | `0 7 * * *` | `jobs/keiba-dispatch-l:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0700"]}]}}` | 履歴テーブル(当年) → 履歴特徴量とサービング用バンドル → テーブルの drift 検査（直列。serving-state-* は day-prep に置き換えて無効） |
| keiba-g-0730 | `30 7 * * *` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0730"]}]}}` | ラップ予想（当日）+ 朝サマリー |
| keiba-g-0730-dayprep | `30 7 * * *` | `jobs/keiba-dispatch-xl:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0730-dayprep"]}]}}` | 開催日の朝に、その日の全レースの日単位キャッシュ(v4_day・px_day)だけを作る（予測は作らない）。ピーク約12GBなので 16GiB |
| keiba-g-0800 | `0 8 * * *` | `jobs/keiba-dispatch-l:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0800"]}]}}` | ページ品質・並走検証。月曜は網羅状況と追走難度 v2 の突き合わせも |
| keiba-g-1000-mon | `0 10 * * 1` | `jobs/keiba-dispatch-l:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-1000-mon"]}]}}` | ドリフト監視・凍結追跡（月曜） |
| keiba-g-1910-eve | `10,40 19 * * *` | `jobs/keiba-dispatch-l:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-1910-eve"]}]}}` | 前夜の計算・翌日のレース予約・ラップ予想（翌日）。2 回走らせて取得の遅れを吸収（冪等）。 |
| keiba-g-2130 | `30 21 * * *` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-2130"]}]}}` | 夜のスクレイピング状況サマリー。金曜は馬名インデックスと成長曲線の週次更新も（約 45 分。前夜のグループに置くと直列で 5400 秒を超えうるため、ここに置く） |
| keiba-g-5min | `*/15 * * * *` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-5min"]}]}}` | VPS ハートビート監視。15 分おきにスクレイピング異常の通知（2026-10-29: 起動を 5 分→15 分おきに。動くジョブが 15 分おきだけのため） |
| keiba-g-hourly | `5 * * * *` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-hourly"]}]}}` | VPS 実行役の定期タスクを 6 時間先まで予約 |
| keiba-g-pump | `* 8-16 * * 0,1,6` | `jobs/keiba-dispatch-s:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-pump"]}]}}` | 開催日（土日月）の日中、イベントを毎分処理して予測タスクを積む |
| keiba-g-results | `*/20 12-21 * * 0,1,2,6` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-results"]}]}}` | 予測成績の集計（土日は暫定、月・火は確定）と通知 |
| keiba-g-sun-2200 | `0 22 * * 0` | `jobs/keiba-dispatch-m:run` | `{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-sun-2200"]}]}}` | オッズ乖離の週次集計（日曜） |

共通: HTTP POST、ヘッダ `Content-Type: application/json`、OAuth のサービスアカウント `keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com`
（scope `https://www.googleapis.com/auth/cloud-platform`）、`attemptDeadline` 180s、再試行 `maxRetryDuration` 0s・`minBackoff` 5s・`maxBackoff` 3600s・`maxDoublings` 5。

## 作り直すとき
正は `config/gcp_job_map.json`（`scheduler_groups`）。基本は、`bash scripts/gcp/deploy_cloud_run_jobs.sh show scheduler` で生成されるコマンドを確かめてから作る（`src/ops/scheduler_gen.py`）。
**作り直すと、1 件 約 15 円/月の固定費と Jobs の実行の費用（並走時 1 日 60〜200 円）が戻る**。費用の台帳で見積もってから行う。
消す前の設定のとおりに戻すときは、次のコマンドを 1 件ずつ実行する。

```
gcloud scheduler jobs create http keiba-g-0530 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "30 5 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-l:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0530"]}]}}' \
  --description '騎手・調教師統計' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-0530 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-0700 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "0 7 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-l:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0700"]}]}}' \
  --description '履歴テーブル(当年) → 履歴特徴量とサービング用バンドル → テーブルの drift 検査（直列。serving-state-* は day-prep に置き換えて無効）' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-0700 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-0730 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "30 7 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0730"]}]}}' \
  --description 'ラップ予想（当日）+ 朝サマリー' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-0730 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-0730-dayprep \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "30 7 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-xl:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0730-dayprep"]}]}}' \
  --description '開催日の朝に、その日の全レースの日単位キャッシュ(v4_day・px_day)だけを作る（予測は作らない）。ピーク約12GBなので 16GiB' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-0730-dayprep --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-0800 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "0 8 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-l:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-0800"]}]}}' \
  --description 'ページ品質・並走検証。月曜は網羅状況と追走難度 v2 の突き合わせも' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-0800 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-1000-mon \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "0 10 * * 1" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-l:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-1000-mon"]}]}}' \
  --description 'ドリフト監視・凍結追跡（月曜）' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-1000-mon --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-1910-eve \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "10,40 19 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-l:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-1910-eve"]}]}}' \
  --description '前夜の計算・翌日のレース予約・ラップ予想（翌日）。2 回走らせて取得の遅れを吸収（冪等）。' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-1910-eve --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-2130 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "30 21 * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-2130"]}]}}' \
  --description '夜のスクレイピング状況サマリー。金曜は馬名インデックスと成長曲線の週次更新も（約 45 分。前夜のグループに置くと直列で 5400 秒を超えうるため、ここに置く）' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-2130 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-5min \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "*/15 * * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-5min"]}]}}' \
  --description 'VPS ハートビート監視。15 分おきにスクレイピング異常の通知（2026-10-29: 起動を 5 分→15 分おきに。動くジョブが 15 分おきだけのため）' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-5min --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-hourly \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "5 * * * *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-hourly"]}]}}' \
  --description 'VPS 実行役の定期タスクを 6 時間先まで予約' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-hourly --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-pump \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "* 8-16 * * 0,1,6" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-s:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-pump"]}]}}' \
  --description '開催日（土日月）の日中、イベントを毎分処理して予測タスクを積む' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-pump --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-results \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "*/20 12-21 * * 0,1,2,6" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-results"]}]}}' \
  --description '予測成績の集計（土日は暫定、月・火は確定）と通知' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-results --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき

gcloud scheduler jobs create http keiba-g-sun-2200 \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "0 22 * * 0" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/keiba-dispatch-m:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"args":["-m","src.ops.job_dispatch","group","g-sun-2200"]}]}}' \
  --description 'オッズ乖離の週次集計（日曜）' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
  --min-backoff 5s --max-backoff 3600s --max-doublings 5 --max-retry-duration 0s
gcloud scheduler jobs pause keiba-g-sun-2200 --project horse-racing-466415 --location asia-northeast1   # 作ってすぐ動かしたくないとき
```

## 昇格のたびに自動で作り直される経路と、止め方（2026-11-13 夜に追記）
- 経路: リポジトリの変数 `ENABLE_JOBS_DEPLOY=true` のため、master への昇格のたびに次が動く。
  1. `.github/workflows/promote.yml`（121 行付近）
  2. `scripts/ci/deploy_gcp.sh jobs`（172 行付近）
  3. `scripts/gcp/deploy_cloud_run_jobs.sh apply scheduler`
  4. `src/ops/scheduler_gen.py` が、`config/gcp_job_map.json` で ready なグループの Scheduler を「あれば update、無ければ create」する。
  - 消した 13 件は、job_map では ready のままだった。そのため、次の昇格で ENABLED で復活し、並走の Jobs が動き出すところだった（月 約 +1,700 円＋GCS の操作。開催日の keiba-g-pump は毎分）。
- 止め方（実施済み）:
  - `config/gcp_job_map.json` の `scheduler_groups` の 13 グループに `retired`（理由）を付けた。
  - `scheduler_gen` は、retired のグループのコマンドを出さない（`--include-unready` でも出さない）。
  - `tests/ops/test_scheduler_gen_retired.py` で、次の 2 点を確かめる。
    - 実際の job_map で出る create/update が retired でないグループ（今は `keiba-g-0730-shadow` だけ）であること。
    - `config/scheduled_jobs.json` で retired・paused の Scheduler が、job_map でも retired であること。
- 再開するとき: そのグループの `retired` を消し、費用の台帳で見積もってから昇格する（この文書の「作り直すとき」）。
