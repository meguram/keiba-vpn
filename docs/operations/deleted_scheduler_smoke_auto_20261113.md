# 削除した Cloud Scheduler: smoke-auto-*（2026-11-13）

## 理由
- 2026-10-21 に作った、取得の疎通確認（Cloud Run Job `netkeiba-smoke`）を 1 回ずつ動かすための Scheduler 3 件。
- スケジュールが 10 月の日付を指定した一度きり（`30 13 22 10 *` など）で、2026-10 に実行済み。次は 2027-10 まで動かない。
- Cloud Scheduler はジョブ 1 件ごとに月 0.1 ドル（約 15 円）かかる（無料枠は請求先アカウントあたり 3 件）。月 2,000 円の上限（CLAUDE.md）を守るため、使わないジョブを消す。
- ユーザーの承認: 2026-11-13（Claude のセッションで直接確認）。

## 消したジョブ（場所 asia-northeast1、プロジェクト horse-racing-466415、タイムゾーン Asia/Tokyo）
| 名前 | スケジュール | 対象 | 本文（base64 を戻したもの） |
|---|---|---|---|
| smoke-auto-thu-noon | `30 13 22 10 *` | `jobs/netkeiba-smoke:run` | `{"overrides":{"containerOverrides":[{"env":[{"name":"SMOKE_SLOT","value":"weekday_noon"}]}]}}` |
| smoke-auto-sat-morning | `30 7 24 10 *` | `jobs/netkeiba-smoke:run` | `{"overrides":{"containerOverrides":[{"env":[{"name":"SMOKE_SLOT","value":"raceday_morning"}]}]}}` |
| smoke-auto-sun-night | `50 23 25 10 *` | `jobs/netkeiba-smoke:run` | `{"overrides":{"containerOverrides":[{"env":[{"name":"SMOKE_SLOT","value":"night"}]}]}}` |

共通: HTTP POST `https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/netkeiba-smoke:run`、
`Content-Type: application/json`、OAuth のサービスアカウント `keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com`
（scope `https://www.googleapis.com/auth/cloud-platform`）、`attemptDeadline` 180s、再試行なし（`maxRetryDuration` 0s）。

## 作り直すとき（例: smoke-auto-thu-noon。日付は実行したい日に変える）
```
gcloud scheduler jobs create http smoke-auto-thu-noon \
  --project horse-racing-466415 --location asia-northeast1 --time-zone Asia/Tokyo \
  --schedule "30 13 22 10 *" --attempt-deadline 180s \
  --uri "https://run.googleapis.com/v2/projects/horse-racing-466415/locations/asia-northeast1/jobs/netkeiba-smoke:run" \
  --http-method POST --headers Content-Type=application/json \
  --message-body '{"overrides":{"containerOverrides":[{"env":[{"name":"SMOKE_SLOT","value":"weekday_noon"}]}]}}' \
  --oauth-service-account-email keiba-scheduler@horse-racing-466415.iam.gserviceaccount.com \
  --oauth-token-scope https://www.googleapis.com/auth/cloud-platform
```
疎通確認を 1 回だけ動かすなら、Scheduler を作らずに `gcloud run jobs execute netkeiba-smoke --region asia-northeast1 --update-env-vars SMOKE_SLOT=weekday_noon` でよい（ジョブの費用は数円）。
