#!/usr/bin/env bash
# Cloud Run の API に必要な土台（API 有効化・Artifact Registry・サービスアカウント・Secret Manager の器）を作る。
# 既定は表示のみ。--apply で実行（冪等。既にあるものはスキップするよう、エラーは無視して進める）。
# 秘密の『値』はここでは入れない: 値は標準入力から入れる手順を最後に表示する（チャットやリポジトリに値を書かない）。
#   PROJECT_ID=... GCS_BUCKET=... bash scripts/gcp/setup_cloud_run_api.sh [--apply]
set -uo pipefail
PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"; REGION="${REGION:-asia-northeast1}"; AR_REPO="${AR_REPO:-keiba-vpn}"
GCS_BUCKET="${GCS_BUCKET:-YOUR_GCS_BUCKET}"
# 【旧】2026-10-24: 実行サービスアカウントと秘密の初期設定は scripts/gcp/setup_project.sh と put_secret.sh（docs/operations/gcp_setup_runbook.md 2・5 章）に一本化した。
# このスクリプトの keiba-api-run と小文字の秘密名（keiba-proxy-key など）は deploy_cloud_run_api.sh から参照されない。使わないこと。
RUN_SA_ID="keiba-api-run"; RUN_SA="${RUN_SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
run() { printf '+'; printf ' %q' "$@"; printf '\n'; if [ "$APPLY" = 1 ]; then "$@" || echo "  (失敗/既存のため続行)"; fi; }
if [ "$APPLY" = 1 ]; then case "$PROJECT_ID$GCS_BUCKET" in *YOUR_*) echo "PROJECT_ID / GCS_BUCKET を実値にしてください" >&2; exit 2;; esac; fi

echo "== 1. API の有効化"
run gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project "$PROJECT_ID"
echo "== 2. Artifact Registry（Docker）。無料枠 0.5GB を超えないよう、古いイメージは掃除する（下の cleanup）"
run gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location "$REGION" --project "$PROJECT_ID"
run gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location "$REGION" --project "$PROJECT_ID" --policy=<(printf '[{"name":"keep-recent","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":3}},{"name":"delete-old","action":{"type":"Delete"},"condition":{"olderThan":"604800s"}}]') --no-dry-run
echo "== 3. 実行用サービスアカウント（最小権限）: GCS はバケット単位の objectAdmin（ハートビートと serving_state の読み書きのため）、Secret は使うものだけ"
run gcloud iam service-accounts create "$RUN_SA_ID" --display-name "keiba-api (Cloud Run)" --project "$PROJECT_ID"
run gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" --member "serviceAccount:${RUN_SA}" --role roles/storage.objectAdmin
echo "== 4. Secret Manager の器（値は後で標準入力から）"
for s in keiba-proxy-key keiba-heartbeat-secret keiba-slack-webhook keiba-dev-secret-key keiba-dev-password; do
  run gcloud secrets create "$s" --replication-policy=automatic --project "$PROJECT_ID"
  run gcloud secrets add-iam-policy-binding "$s" --member "serviceAccount:${RUN_SA}" --role roles/secretmanager.secretAccessor --project "$PROJECT_ID"
done
cat <<'TXT'

== 5. 秘密の値を入れる（ユーザーが手元で。値はチャット・リポジトリ・シェル履歴に残さない）
  # 共有ヘッダ鍵とハートビート署名鍵は新しく作る（同じ値を VPS の /etc/keiba-ui/ にも入れる: docs/operations/cloud_run_api.md §4）
  openssl rand -hex 32 | gcloud secrets versions add keiba-proxy-key --data-file=- --project <PROJECT_ID>
  openssl rand -hex 32 | gcloud secrets versions add keiba-heartbeat-secret --data-file=- --project <PROJECT_ID>
  # Slack・ログイン用の鍵はホストの .env の値を使う（画面に出さず、読み込んで直接渡す）
  #   例: grep '^SLACK_WEBHOOK_URL=' .env | cut -d= -f2- | tr -d '\n' | gcloud secrets versions add keiba-slack-webhook --data-file=- --project <PROJECT_ID>
TXT
