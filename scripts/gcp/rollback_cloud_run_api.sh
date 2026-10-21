#!/usr/bin/env bash
# Cloud Run の API を『1 つ前のリビジョン』にトラフィックを戻す（または --to <リビジョン名>）。既定は表示のみ、--apply で実行。
# 数秒で終わる。VPS 側で nginx の上流をホストに戻す場合は scripts/ops/vps/switch_api_upstream.sh host（docs/operations/cloud_run_api.md）。
#   bash scripts/gcp/rollback_cloud_run_api.sh [--to <rev>] [--apply]
set -euo pipefail
PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"; REGION="${REGION:-asia-northeast1}"; SERVICE="${SERVICE:-keiba-api}"
APPLY=0; TO=""
while [ $# -gt 0 ]; do case "$1" in --apply) APPLY=1;; --to) TO="${2:?--to にリビジョン名}"; shift;; *) echo "不明な引数: $1" >&2; exit 2;; esac; shift; done
run() { printf '+'; printf ' %q' "$@"; printf '\n'; if [ "$APPLY" = 1 ]; then "$@"; fi; }
if [ -z "$TO" ]; then
  if [ "$APPLY" = 1 ]; then
    # 作成が新しい順に、今トラフィックを持つ最新リビジョンの次を選ぶ
    mapfile -t REVS < <(gcloud run revisions list --project "$PROJECT_ID" --region "$REGION" --service "$SERVICE" --sort-by='~metadata.creationTimestamp' --format='value(metadata.name)' --limit 5)
    CUR="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.traffic[0].revisionName)')"
    TO=""; for r in "${REVS[@]}"; do if [ "$r" != "$CUR" ]; then TO="$r"; break; fi; done
    [ -n "$TO" ] || { echo "戻し先のリビジョンが見つかりません" >&2; exit 1; }
    echo "# 現在: $CUR → 戻し先: $TO"
  else
    TO="<1 つ前のリビジョン名（gcloud run revisions list で確認）>"
  fi
fi
run gcloud run services update-traffic "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --to-revisions "${TO}=100"
