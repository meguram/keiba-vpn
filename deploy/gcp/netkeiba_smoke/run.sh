#!/usr/bin/env bash
# Cloud Shell で 1 本実行する: ビルド → ジョブ作成/更新 → 実行 → 結果の要約と合否。
#
# 使い方（Cloud Shell。リポジトリの取得は docs/operations/gcp_setup_runbook.md の 0 章）:
#   cd keiba-vpn/deploy/gcp/netkeiba_smoke
#   bash run.sh weekday_noon        # 枠: weekday_noon | raceday_morning | night
#   YES=1 bash run.sh night         # 確認の質問を飛ばす
#   CLEANUP=1 bash run.sh night     # 実行後にジョブを削除（3 回終わったら）
#
# 公開ページのみ・ログインなし・30 リクエスト（約 3 分）。403/429 が出たらその場で中止して FAIL。
# 費用は無料枠内（Cloud Build 2,500 分/月・Artifact Registry 0.5GB・Cloud Run Jobs 数十 vCPU 秒）。
set -euo pipefail

SLOT="${1:-}"
case "$SLOT" in weekday_noon|raceday_morning|night) ;; *) echo "使い方: bash run.sh weekday_noon|raceday_morning|night" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${REGION:-asia-northeast1}"
AR_REPO="${AR_REPO:-keiba}"
JOB="${JOB:-netkeiba-smoke}"
PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクトが未設定です: gcloud config set project <ID>" >&2; exit 1; }
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$AR_REPO/$JOB:latest"

JST_NOW="$(TZ=Asia/Tokyo date '+%F %a %H:%M')"
echo "== netkeiba 到達試験  project=$PROJECT region=$REGION 枠=$SLOT 現在(JST)=$JST_NOW"
if [ "${YES:-}" != "1" ]; then
  echo "確認: (1) ホストの取得ジョブ(cron/ランナー)は今止まっていますか？ netkeiba へ同時にアクセスしてはいけません（DEC-023）。"
  echo "      ホストでの確認: ssh keiba-remote \"pgrep -af '[a]uto_scrape' || echo 動いていない\""
  read -r -p "止まっていれば y: " ans
  [ "$ans" = "y" ] || { echo "中止しました"; exit 3; }
fi

echo "== 1/5 API と Artifact Registry"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location "$REGION" --description="keiba containers"

echo "== 2/5 イメージのビルド（Cloud Build）"
gcloud builds submit "$HERE" --tag "$IMAGE" --region "$REGION" --quiet

echo "== 3/5 ジョブの作成または更新"
COMMON=(--image "$IMAGE" --region "$REGION" --max-retries 0 --task-timeout 600 --tasks 1 --cpu 1 --memory 256Mi)
if gcloud run jobs describe "$JOB" --region "$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "$JOB" "${COMMON[@]}"
else
  gcloud run jobs create "$JOB" "${COMMON[@]}"
fi

echo "== 4/5 実行（約 3 分。終わるまで待ちます）"
set +e
gcloud run jobs execute "$JOB" --region "$REGION" --update-env-vars "SMOKE_SLOT=$SLOT" --wait
EXEC_RC=$?
set -e
EXEC_NAME="$(gcloud run jobs executions list --job "$JOB" --region "$REGION" --limit 1 --format='value(name)')"
echo "実行: $EXEC_NAME  (終了コード $EXEC_RC)"

echo "== 5/5 結果の要約"
OUT="$HERE/last_result_${SLOT}.txt"
RESULT=""
for i in 1 2 3 4 5 6; do
  gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB\" AND labels.\"run.googleapis.com/execution_name\"=\"$EXEC_NAME\"" \
    --limit 400 --freshness 2h --order asc --format=json > "$HERE/last_logs.json" || true
  set +e
  python3 "$HERE/summarize.py" < "$HERE/last_logs.json" > "$OUT"
  RC=$?
  set -e
  RESULT="$(grep '^RESULT:' "$OUT" || true)"
  case "$RESULT" in "RESULT: UNKNOWN"*) echo "  ログの反映待ち($i/6)…"; sleep 10 ;; *) break ;; esac
done
cat "$OUT"
echo
echo "（コンテナの終了コード=$EXEC_RC: 0 が合格、非 0 が不合格。ログが読めないときはこちらを正とする）"
[ "${CLEANUP:-}" = "1" ] && gcloud run jobs delete "$JOB" --region "$REGION" --quiet && echo "ジョブを削除しました"
exit "$EXEC_RC"
