#!/usr/bin/env bash
# 到達試験の結果を一覧で見る（自動実行・手動実行のどちらも）。
#   bash show_results.sh        # 直近 10 回
#   N=3 bash show_results.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${REGION:-asia-northeast1}"
JOB="${JOB:-netkeiba-smoke}"
N="${N:-10}"
names="$(gcloud run jobs executions list --job "$JOB" --region "$REGION" --limit "$N" --format='value(name)')"
[ -n "$names" ] || { echo "実行の記録がありません"; exit 0; }
for name in $(echo "$names" | tac); do
  started="$(gcloud run jobs executions describe "$name" --region "$REGION" --format='value(status.startTime)' 2>/dev/null || true)"
  echo "---- $name  開始(UTC)=$started"
  gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB\" AND labels.\"run.googleapis.com/execution_name\"=\"$name\"" \
    --limit 400 --freshness 14d --order asc --format=json 2>/dev/null | python3 "$HERE/summarize.py" | grep -E '^(RESULT:|送信元|リクエスト数|  エラー|記録用)' || echo "  (ログが読めません)"
done
echo
echo "合格条件: 3 回とも成功率 93% 以上・403/429 が 0 件。503 が全件で CloudFront の『ページが表示できません』なら netkeiba 側の不調の可能性があり、判定に使わない(VPS からの 1 件確認と照らす)。"
