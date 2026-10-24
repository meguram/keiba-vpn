#!/usr/bin/env bash
# GCP の初期設定を Cloud Shell から 1 本で行う（冪等。何度流しても同じ状態になる）。
# 手順書: docs/operations/gcp_setup_runbook.md
#
#   DRY=1 bash scripts/gcp/setup_project.sh      # 何を実行するか表示するだけ
#   BUCKET=<既存の GCS バケット名> bash scripts/gcp/setup_project.sh
#
# やること: API の有効化 / Artifact Registry / サービスアカウント 4 つ（api・jobs・scheduler・deployer）と最小権限 /
#           BUCKET を渡したときはそのバケットへの権限。秘密の値は扱わない（scripts/gcp/put_secret.sh）。
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1   # 未有効の API を確認する読み取りコマンドが「有効にしますか?」で止まらないようにする

REGION="${REGION:-asia-northeast1}"
AR_REPO="${AR_REPO:-keiba-vpn}"     # deploy_cloud_run_{api,jobs}.sh と同じ名前（以前は keiba で、push 先が見つからず失敗した）
BUCKET="${BUCKET:-}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定: gcloud config set project <ID>" >&2; exit 1; }

run() { if [ "${DRY:-}" = "1" ]; then echo "[dry] $*" >&2; else "$@"; fi; }

APIS=(run.googleapis.com cloudscheduler.googleapis.com cloudtasks.googleapis.com secretmanager.googleapis.com
      artifactregistry.googleapis.com cloudbuild.googleapis.com billingbudgets.googleapis.com bigquery.googleapis.com
      iam.googleapis.com logging.googleapis.com)
echo "== project=$PROJECT region=$REGION bucket=${BUCKET:-(未指定)}"
echo "== 1. API の有効化"
run gcloud services enable "${APIS[@]}" --project "$PROJECT"

echo "== 2. Artifact Registry（コンテナ置き場）"
if ! gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  run gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location "$REGION" \
    --description="keiba containers" --project "$PROJECT"
fi
# 古いイメージの自動掃除（パッケージごとに新しい 3 版は残し、それ以外で 7 日より古いものを削除）。Jobs のイメージは 1 版 数 GB（torch 入り）
run gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location "$REGION" --project "$PROJECT" --no-dry-run \
  --policy=<(printf '[{"name":"keep-recent","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":3}},{"name":"delete-old","action":{"type":"Delete"},"condition":{"olderThan":"604800s"}}]')

echo "== 3. サービスアカウント"
declare -A SA_DESC=(
  [keiba-api]="Cloud Run の API（読み取り中心。GCS のキャッシュ書き込みのみ）"
  [keiba-jobs]="Cloud Run Jobs（予測生成・集計・通知）"
  [keiba-scheduler]="Cloud Scheduler / Cloud Tasks から Jobs・API を起動するだけ"
  [keiba-deployer]="ビルドとデプロイ（人間が Cloud Shell で行う間は不要。自動化するときに使う）"
)
for sa in keiba-api keiba-jobs keiba-scheduler keiba-deployer; do
  email="$sa@$PROJECT.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$email" --project "$PROJECT" >/dev/null 2>&1; then
    run gcloud iam service-accounts create "$sa" --display-name "$sa" --description "${SA_DESC[$sa]}" --project "$PROJECT"
  fi
done

bind_project() { # SA ロール（roles/ が無ければ補う）
  local role="$2"; case "$role" in roles/*) ;; *) role="roles/$role" ;; esac
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$1@$PROJECT.iam.gserviceaccount.com" \
    --role "$role" --condition=None --quiet >/dev/null
}
bind_bucket() { # SA ロール
  run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$1@$PROJECT.iam.gserviceaccount.com" \
    --role "$2" --condition=None --quiet >/dev/null     # 条件つきの権限（WIF の ops/releases/ 限定）が入った後は、条件なしの明示が要る
}

echo "== 4. 最小権限（プロジェクト）"
bind_project keiba-jobs bigquery.jobUser          # 課金エクスポートの集計クエリ
bind_project keiba-jobs cloudtasks.enqueuer       # 開催日の予約実行を積む
bind_project keiba-scheduler run.invoker          # Jobs の起動・API の呼び出し（run.jobs.run を含むか runbook 4.3 で確認）
bind_project keiba-deployer run.admin
bind_project keiba-deployer artifactregistry.writer
bind_project keiba-deployer cloudbuild.builds.editor
bind_project keiba-deployer cloudscheduler.admin     # CI(master)が Jobs の Scheduler を更新する（deploy_gcp.sh jobs）
bind_project keiba-deployer cloudtasks.admin         # 同: 予測のキュー(keiba-predict)
bind_project keiba-deployer serviceusage.serviceUsageConsumer   # gcloud が API を使うために必要
for target in keiba-api keiba-jobs keiba-scheduler; do   # deployer が各 SA でサービス/ジョブを動かせるように
  run gcloud iam service-accounts add-iam-policy-binding "$target@$PROJECT.iam.gserviceaccount.com" \
    --member "serviceAccount:keiba-deployer@$PROJECT.iam.gserviceaccount.com" --role roles/iam.serviceAccountUser \
    --project "$PROJECT" --quiet >/dev/null
done

echo "== 5. 最小権限（GCS バケット）"
if [ -n "$BUCKET" ]; then
  bind_bucket keiba-api roles/storage.objectUser    # オブジェクトの読み書きのみ（バケットの削除・設定変更は不可）
  bind_bucket keiba-jobs roles/storage.objectUser
else
  echo "  BUCKET 未指定のためスキップ（BUCKET=<名前> で再実行）"
fi

echo "== 6. 確認"
if [ "${DRY:-}" != "1" ]; then
  echo "-- 有効な API（必要分）"
  gcloud services list --enabled --project "$PROJECT" --format="value(config.name)" | grep -E "$(IFS='|'; echo "${APIS[*]}")" | sort
  echo "-- サービスアカウント"
  gcloud iam service-accounts list --project "$PROJECT" --format="table(email,displayName)" --filter="email~^keiba-"
  echo "-- Artifact Registry"
  gcloud artifacts repositories list --location "$REGION" --project "$PROJECT" --format="table(name,format,location)"
fi
echo "完了。次: 秘密の登録 → bash scripts/gcp/put_secret.sh <NAME>（runbook 5 章）"
