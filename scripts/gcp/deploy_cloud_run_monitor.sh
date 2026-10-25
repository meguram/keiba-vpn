#!/usr/bin/env bash
# =============================================================================
# 監視ポータル（src/monitor/cloud_app.py。GCS の ops/ を読むだけ）を Cloud Run の別サービス keiba-monitor として配備する。設計: docs/operations/monitor_cloudrun.md
# イメージは API と同じ（deploy/docker/Dockerfile.cloudrun）。起動コマンドだけ違う。最小インスタンス 0・最大 1 なので、見ていない間は 0 円。
#
# 既定は『コマンドを表示するだけ』。実行は --apply のときだけ（Cloud Shell で）。
#   IMAGE=<API と同じイメージ。CI が push した keiba-api:<8桁>> PROJECT_ID=… GCS_BUCKET=… bash scripts/gcp/deploy_cloud_run_monitor.sh [--apply]
# 事前: 読み取り専用の SA keiba-monitor（ops/ だけ読める）と、秘密 KEIBA_PROXY_KEY（API と同じ値でよい）。手順は monitor_cloudrun.md の 2 章。
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
APPLY=0
for a in "$@"; do case "$a" in --apply) APPLY=1;; --dry-run) APPLY=0;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done

PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"; REGION="${REGION:-asia-northeast1}"; SERVICE="${SERVICE:-keiba-monitor}"
GCS_BUCKET="${GCS_BUCKET:-YOUR_GCS_BUCKET}"; AR_REPO="${AR_REPO:-keiba-vpn}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/keiba-api:TAG}"
RUN_SA="${RUN_SA:-keiba-monitor@${PROJECT_ID}.iam.gserviceaccount.com}"
if [ "$APPLY" = 1 ]; then
  case "$PROJECT_ID$GCS_BUCKET$IMAGE" in *YOUR_*|*:TAG) echo "PROJECT_ID / GCS_BUCKET / IMAGE を実値にしてください" >&2; exit 2;; esac
  [ "${KEIBA_ENV:-}" != "dev" ] || { echo "dev PC からは実行できません（Cloud Shell で）" >&2; exit 2; }
fi
run() { printf '+'; printf ' %q' "$@"; printf '\n'; if [ "$APPLY" = 1 ]; then "$@"; fi; }

# 平文の環境変数（秘密は含めない）。KEIBA_RUN_LOCATION=gcp は通知・表示で GCP 側と見分けるため
ENV_VARS="KEIBA_ENV=prod,KEIBA_RUN_LOCATION=gcp,GCS_BUCKET=${GCS_BUCKET},GCS_PROJECT_ID=${PROJECT_ID},KEIBA_PROXY_KEY_REQUIRED=1,TZ=Asia/Tokyo"
SECRETS="KEIBA_PROXY_KEY=KEIBA_PROXY_KEY:latest"

# メモリ 512Mi・CPU 1・最大 1 台: GCS の小さな JSON を読んで表を作るだけ。料金は無料枠の範囲の見込み（リクエストが少ない）
run gcloud run deploy "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --service-account "$RUN_SA" \
  --memory 512Mi --cpu 1 --min-instances 0 --max-instances 1 --concurrency 10 --timeout 30 \
  --port 8080 --ingress all --allow-unauthenticated \
  --command python --args=-m,uvicorn,--factory,src.monitor.cloud_app:asgi_app,--host,0.0.0.0,--port,8080 \
  --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS"
echo "# 確認: curl -s \$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')/api/health   （鍵なしで 200）"
echo "# 画面: ヘッダ X-Keiba-Proxy-Key 付きで / を開く（前段の VPS nginx / Cloudflare Access が付ける）"
