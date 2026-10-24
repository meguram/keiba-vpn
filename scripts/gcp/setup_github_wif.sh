#!/usr/bin/env bash
# GitHub Actions（promote.yml）が GCP に『鍵ファイルなし』で入るための Workload Identity Federation を用意する。
# Cloud Shell で実行する。既定は『コマンドを表示するだけ』（何も変更しない）。実行は --apply。冪等（すでにあれば飛ばす）。
#
#   bash scripts/gcp/setup_github_wif.sh                 # 実行内容を表示（dry-run が既定）
#   bash scripts/gcp/setup_github_wif.sh --apply         # 実行する
#
# 必須: GCS_BUCKET（公開先のバケット名）。任意: PROJECT_ID（既定は gcloud の現在のプロジェクト） GITHUB_REPO（既定 meguram/keiba-vpn）
#       POOL（github-pool） PROVIDER（github-provider） DEPLOY_SA（keiba-deployer）
#
# 作るもの
#  1. Workload Identity Pool と OIDC Provider。入れる条件: リポジトリが GITHUB_REPO、かつ『master + Environment production（承認あり）』または『stg + Environment stg』
#     （＝承認の関門を通った master の job と、stg ブランチの job だけが、この身元を使える。フォークや他のブランチ、承認前の job は入れない）
#  2. デプロイ用 SA（keiba-deployer）に roles/iam.workloadIdentityUser（上の Pool のこのリポジトリだけ）
#  3. デプロイ用 SA にバケットの ops/releases/ 接頭辞だけへの roles/storage.objectUser（IAM Conditions）
# 作らないもの: サービスアカウントの鍵（JSON）。鍵は一切作らない。
set -euo pipefail

APPLY=0
for a in "$@"; do case "$a" in --apply) APPLY=1;; --dry-run) APPLY=0;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] || { echo "PROJECT_ID が未設定です（gcloud config set project <ID>）" >&2; exit 1; }
BUCKET="${GCS_BUCKET:-}"
[ -n "$BUCKET" ] || { echo "GCS_BUCKET（公開先のバケット名）を指定してください" >&2; exit 1; }
REPO="${GITHUB_REPO:-meguram/keiba-vpn}"
POOL="${POOL:-github-pool}"
PROVIDER="${PROVIDER:-github-provider}"
DEPLOY_SA="${DEPLOY_SA:-keiba-deployer}"
SA_EMAIL="$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com"
PREFIX="ops/releases/"

run() { if [ "$APPLY" = 1 ]; then "$@"; else printf '[dry] '; printf '%q ' "$@"; echo; fi; }
exists() { [ "$APPLY" = 1 ] && "$@" >/dev/null 2>&1; }

echo "== project=$PROJECT_ID repo=$REPO bucket=$BUCKET sa=$SA_EMAIL  ($([ "$APPLY" = 1 ] && echo 実行 || echo 表示のみ))"

PROJECT_NUMBER="$( [ "$APPLY" = 1 ] && gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' || echo '<PROJECT_NUMBER>' )"

echo "== 1/3 Workload Identity Pool と Provider"
if ! exists gcloud iam workload-identity-pools describe "$POOL" --location=global --project "$PROJECT_ID"; then
  run gcloud iam workload-identity-pools create "$POOL" --location=global --project "$PROJECT_ID" \
    --display-name="GitHub Actions"
fi
# master ブランチ + Environment production（承認あり）／stg ブランチ + Environment stg だけが、この身元を使える
COND="assertion.repository=='$REPO' && ((assertion.ref=='refs/heads/master' && assertion.environment=='production') || (assertion.ref=='refs/heads/stg' && assertion.environment=='stg'))"
if ! exists gcloud iam workload-identity-pools providers describe "$PROVIDER" --workload-identity-pool="$POOL" --location=global --project "$PROJECT_ID"; then
  run gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --workload-identity-pool="$POOL" --location=global \
    --project "$PROJECT_ID" --display-name="GitHub" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref,attribute.environment=assertion.environment" \
    --attribute-condition="$COND"
else
  # 既にある Provider も条件を最新にする（以前は main + production のみだった）
  run gcloud iam workload-identity-pools providers update-oidc "$PROVIDER" --workload-identity-pool="$POOL" --location=global \
    --project "$PROJECT_ID" --attribute-condition="$COND"
fi

echo "== 2/3 デプロイ用 SA をこのリポジトリの身元から使えるようにする（鍵は作らない）"
MEMBER="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO"
run gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --project "$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" --member="$MEMBER" --condition=None --quiet

echo "== 3/3 バケットの $PREFIX だけに書き込める権限（IAM Conditions）"
COND_EXPR="resource.name.startsWith(\"projects/_/buckets/$BUCKET/objects/$PREFIX\")"
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectUser" \
  --condition="expression=$COND_EXPR,title=ops-releases-only,description=GitHub Actions は ops/releases/ だけに書ける"

echo
echo "リポジトリの Variables（Settings → Secrets and variables → Actions → Variables）に入れる値:"
echo "  WIF_PROVIDER         = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
echo "  WIF_SERVICE_ACCOUNT  = $SA_EMAIL"
echo "  GCS_BUCKET           = $BUCKET"
echo "（これらは秘密ではなく Variables でよい。鍵ファイルは作っていないので、Secrets に入れるものはありません）"
if [ "$APPLY" != 1 ]; then echo "→ 内容を確認して、実行するときは --apply を付けてください"; fi
