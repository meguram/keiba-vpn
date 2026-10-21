#!/usr/bin/env bash
# 到達試験の 3 回を Cloud Scheduler で自動実行する（Cloud Shell で 1 回だけ設定）。結果は show_results.sh で見る。
#   bash schedule_auto.sh create    # 3 つの予約を作る（何度流しても同じ状態）
#   bash schedule_auto.sh status    # 予約の一覧
#   bash schedule_auto.sh delete    # 3 回終わったら必ず削除（日付固定の cron なので放置すると来年の同日にも走る）
# 前提: netkeiba-smoke のジョブが作成済み（bash run.sh <枠> を 1 回流してある）。keiba-scheduler に run.invoker 付与済み。
# 実行時刻（JST）: 10/22(木) 13:30 昼 / 10/24(土) 07:30 開催日の朝 / 10/25(日) 23:50 夜間。
set -euo pipefail
REGION="${REGION:-asia-northeast1}"
JOB="${JOB:-netkeiba-smoke}"
PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定" >&2; exit 1; }
SA="keiba-scheduler@$PROJECT.iam.gserviceaccount.com"
URI="https://run.googleapis.com/v2/projects/$PROJECT/locations/$REGION/jobs/$JOB:run"
# 名前|cron(分 時 日 月 *)|枠
PLANS=(
  "smoke-auto-thu-noon|30 13 22 10 *|weekday_noon"
  "smoke-auto-sat-morning|30 7 24 10 *|raceday_morning"
  "smoke-auto-sun-night|50 23 25 10 *|night"
)
case "${1:-status}" in
  create)
    gcloud run jobs describe "$JOB" --region "$REGION" --format='value(name)' >/dev/null || { echo "ジョブ $JOB がありません。先に bash run.sh <枠> を 1 回流してください" >&2; exit 1; }
    for p in "${PLANS[@]}"; do
      IFS='|' read -r name cron slot <<<"$p"
      body="{\"overrides\":{\"containerOverrides\":[{\"env\":[{\"name\":\"SMOKE_SLOT\",\"value\":\"$slot\"}]}]}}"
      args=(--location "$REGION" --schedule "$cron" --time-zone Asia/Tokyo --uri "$URI" --http-method POST
            --oauth-service-account-email "$SA" --headers Content-Type=application/json --message-body "$body" --attempt-deadline 180s)
      if gcloud scheduler jobs describe "$name" --location "$REGION" >/dev/null 2>&1; then
        gcloud scheduler jobs update http "$name" "${args[@]}" >/dev/null
      else
        gcloud scheduler jobs create http "$name" "${args[@]}" >/dev/null
      fi
      echo "予約: $name  $cron (Asia/Tokyo)  枠=$slot"
    done ;;
  delete)
    for p in "${PLANS[@]}"; do
      IFS='|' read -r name _ _ <<<"$p"
      gcloud scheduler jobs delete "$name" --location "$REGION" --quiet 2>/dev/null && echo "削除: $name" || echo "なし: $name"
    done ;;
  status)
    gcloud scheduler jobs list --location "$REGION" --filter="name~smoke-auto" --format="table(name.basename(),schedule,state,lastAttemptTime)" ;;
  *) echo "使い方: bash schedule_auto.sh create|status|delete" >&2; exit 2 ;;
esac
