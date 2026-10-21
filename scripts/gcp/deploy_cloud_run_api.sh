#!/usr/bin/env bash
# =============================================================================
# Cloud Run の API（Dockerfile.cloudrun）のビルド・デプロイ。設計: docs/operations/cloud_run_api.md
#
# 既定は『コマンドを表示するだけ』（何も実行しない）。実際に実行するのは --apply を付けたときだけ。
# dev PC は GCP に接続できない（gcp_guard）。実行はユーザーの Cloud Shell / ホスト PC から。
#
# Usage:
#   bash scripts/gcp/deploy_cloud_run_api.sh build   [--dry-run|--apply]   # Cloud Build でイメージを作って Artifact Registry へ
#   bash scripts/gcp/deploy_cloud_run_api.sh deploy  [--dry-run|--apply]   # 直前に build したイメージ（TAG）を Cloud Run へ。新リビジョンは --no-traffic で作り、確認後に切替
#   bash scripts/gcp/deploy_cloud_run_api.sh promote [--dry-run|--apply]   # 最新リビジョンに 100% 切替（確認後）
#   bash scripts/gcp/deploy_cloud_run_api.sh all     [--dry-run|--apply]   # build → deploy(--no-traffic)。切替は promote
#
# 必須の環境変数: PROJECT_ID, GCS_BUCKET。任意: REGION(asia-northeast1) SERVICE(keiba-api) TAG(日時) AR_REPO(keiba-vpn)
#   MEMORY(2Gi) CPU(1) MIN_INSTANCES(0) MAX_INSTANCES(3) CONCURRENCY(30) TIMEOUT(60) GCS_PREFIX WITH_NEXT_KEY(1 で KEIBA_PROXY_KEY_NEXT も載せる)
# 事前に scripts/gcp/setup_cloud_run_api.sh で サービスアカウント・Secret Manager・Artifact Registry を用意する。
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

ACTION="${1:-}"; shift || true
APPLY=0
for a in "$@"; do case "$a" in --apply) APPLY=1;; --dry-run) APPLY=0;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done
[ -n "$ACTION" ] || { sed -n '2,20p' "$0"; exit 2; }

PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
REGION="${REGION:-asia-northeast1}"
SERVICE="${SERVICE:-keiba-api}"
AR_REPO="${AR_REPO:-keiba-vpn}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE}:${TAG}}"
RUN_SA="${RUN_SA:-keiba-api-run@${PROJECT_ID}.iam.gserviceaccount.com}"
MEMORY="${MEMORY:-2Gi}"; CPU="${CPU:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"; MAX_INSTANCES="${MAX_INSTANCES:-3}"
CONCURRENCY="${CONCURRENCY:-30}"; TIMEOUT="${TIMEOUT:-60}"
GCS_BUCKET="${GCS_BUCKET:-YOUR_GCS_BUCKET}"
GCS_PREFIX="${GCS_PREFIX:-}"

run() {  # 表示して、--apply のときだけ実行
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$APPLY" = 1 ]; then "$@"; fi
}

if [ "$APPLY" = 1 ]; then
  case "$PROJECT_ID$GCS_BUCKET" in *YOUR_*) echo "PROJECT_ID / GCS_BUCKET を実値にしてください" >&2; exit 2;; esac
  [ "${KEIBA_ENV:-}" != "dev" ] || { echo "dev PC からは実行できません（Cloud Shell かホスト PC で）" >&2; exit 2; }
fi

# 平文の環境変数（秘密を含めない）。KEIBA_ENV=prod / KEIBA_RUN_LOCATION=gcp は必須（通知に PROD・GCP と出る）
ENV_VARS="KEIBA_ENV=prod,KEIBA_RUN_LOCATION=gcp,GCS_BUCKET=${GCS_BUCKET},GCS_PROJECT_ID=${PROJECT_ID}"
ENV_VARS+=",KEIBA_PROXY_KEY_REQUIRED=1,KEIBA_SERVING_STATE_PULL=1,KEIBA_ODDS_TRIFECTA=0"
[ -z "$GCS_PREFIX" ] || ENV_VARS+=",GCS_PREFIX=${GCS_PREFIX}"
# Secret Manager（<環境変数名>=<シークレット名>:<バージョン>）。値はここに書かない
SECRETS="KEIBA_PROXY_KEY=keiba-proxy-key:latest,KEIBA_HEARTBEAT_SECRET=keiba-heartbeat-secret:latest"
SECRETS+=",SLACK_WEBHOOK_URL=keiba-slack-webhook:latest,DEV_SECRET_KEY=keiba-dev-secret-key:latest,DEV_PASSWORD=keiba-dev-password:latest"
[ "${WITH_NEXT_KEY:-0}" != 1 ] || SECRETS+=",KEIBA_PROXY_KEY_NEXT=keiba-proxy-key-next:latest"

# Dockerfile.cloudrun を指定するため、cloudbuild 設定を一時生成する
build() {
  local cfg; cfg="$(mktemp --suffix=.yaml)"
  cat > "$cfg" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    env: ['DOCKER_BUILDKIT=1']
    args: ['build', '-f', 'Dockerfile.cloudrun', '-t', '${IMAGE}', '.']
images: ['${IMAGE}']
timeout: 1200s
YAML
  echo "# cloudbuild 設定: $cfg"; cat "$cfg"
  run gcloud builds submit --project "$PROJECT_ID" --region "$REGION" --config "$cfg" --ignore-file Dockerfile.cloudrun.gcloudignore .
  rm -f "$cfg"
}

deploy() {
  # 初回（サービスがまだ無い）は --no-traffic が使えない。2 回目以降は新リビジョンをトラフィック 0% で作る
  local traffic=(--no-traffic --tag "r${TAG//[^a-zA-Z0-9]/}")
  if [ "$APPLY" = 1 ] && ! gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" >/dev/null 2>&1; then
    traffic=(); echo "# 初回デプロイ: サービスを新規作成（トラフィックは最初のリビジョンに 100%）"
  fi
  run gcloud run deploy "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
    --service-account "$RUN_SA" \
    --memory "$MEMORY" --cpu "$CPU" --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" \
    --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
    --cpu-throttling --execution-environment gen2 \
    --port 8080 --ingress all --allow-unauthenticated \
    --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS" \
    "${traffic[@]}"
  echo "# 新リビジョンはトラフィック 0%（タグ URL で確認）。確認後: bash $0 promote --apply"
  echo "# 戻す: bash scripts/gcp/rollback_cloud_run_api.sh --apply"
}

promote() {
  run gcloud run services update-traffic "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --to-latest
}

case "$ACTION" in
  build) build ;;
  deploy) deploy ;;
  promote) promote ;;
  all) build; deploy ;;
  *) echo "不明なアクション: $ACTION" >&2; exit 2 ;;
esac
