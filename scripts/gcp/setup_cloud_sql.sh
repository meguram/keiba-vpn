#!/usr/bin/env bash
# Cloud SQL（PostgreSQL）を「固定費最小」で作るための設計図スクリプト。
#
# 既定は show（コマンドを表示するだけで何も実行しない）。実行する場合のみ --apply を付ける。
#   bash scripts/gcp/setup_cloud_sql.sh                 # コマンド一覧を表示（既定）
#   PROJECT_ID=your-gcp-project bash scripts/gcp/setup_cloud_sql.sh --apply
#
# コスト最小化の方針（docs/operations/deployment-vps-vs-gcp.md「固定費を抑える設計」参照）:
#   - stg と prod で 1 インスタンスを共用し、DB 名（keiba_db / keiba_db_stg）とユーザーで分離する
#   - 最小ティア(db-f1-micro)・HDD・単一ゾーン(HA なし)・バックアップ世代を絞る
#   - ストレージ自動拡張には上限を付け、想定外の増加による課金を防ぐ
# 注意: 本スクリプトは設計図で、実 GCP 環境では未検証。実行前に `gcloud sql instances create --help`
#       でフラグを確認すること。db-f1-micro は RAM 約0.6GB・SLA なし。クエリが重くなったら
#       db-g1-small へ変更する（`gcloud sql instances patch --tier=db-g1-small`、再起動あり）。

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-asia-northeast1}"
INSTANCE="${INSTANCE:-keiba-db}"
TIER="${TIER:-db-f1-micro}"
STORAGE_GB="${STORAGE_GB:-10}"
STORAGE_LIMIT_GB="${STORAGE_LIMIT_GB:-50}"
# サービスアカウント（.env の GCS_CLIENT_EMAIL）。Cloud SQL Connector 経由の接続に cloudsql.client が必要
SA_EMAIL="${SA_EMAIL:-service-account@${PROJECT_ID}.iam.gserviceaccount.com}"
# 以下のパスワードは環境変数で渡す（コマンド履歴・リポジトリに実値を残さない）
PROD_DB_PASSWORD="${PROD_DB_PASSWORD:-CHANGE_ME_PROD}"
STG_DB_PASSWORD="${STG_DB_PASSWORD:-CHANGE_ME_STG}"

commands() {
  cat <<EOF
# 1) インスタンス（stg/prod 共用）
gcloud sql instances create ${INSTANCE} \\
  --project=${PROJECT_ID} --region=${REGION} \\
  --database-version=POSTGRES_15 --edition=ENTERPRISE \\
  --tier=${TIER} \\
  --storage-type=HDD --storage-size=${STORAGE_GB}GB \\
  --storage-auto-increase --storage-auto-increase-limit=${STORAGE_LIMIT_GB} \\
  --availability-type=zonal \\
  --backup-start-time=18:00 --retained-backups-count=3

# 2) DB を環境別に分離
gcloud sql databases create keiba_db     --instance=${INSTANCE} --project=${PROJECT_ID}
gcloud sql databases create keiba_db_stg --instance=${INSTANCE} --project=${PROJECT_ID}

# 3) ユーザーも環境別に分離（.env.prod / .env.stg の CLOUD_SQL_DB_USER / CLOUD_SQL_DB_PASSWORD と一致させる）
gcloud sql users create keiba_user     --instance=${INSTANCE} --project=${PROJECT_ID} --password="\${PROD_DB_PASSWORD}"
gcloud sql users create keiba_stg_user --instance=${INSTANCE} --project=${PROJECT_ID} --password="\${STG_DB_PASSWORD}"

# 4) アプリのサービスアカウントに接続権限を付与
gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
  --member="serviceAccount:${SA_EMAIL}" --role="roles/cloudsql.client"

# 5) （任意）stg を使わない期間のコスト削減: 停止中は CPU/メモリ課金が止まる（ストレージ・バックアップは継続）。
#    共用インスタンスは prod も使うため通常は停止しない。stg 専用に分けた場合のみ有効。
# gcloud sql instances patch ${INSTANCE} --activation-policy=NEVER     # 停止
# gcloud sql instances patch ${INSTANCE} --activation-policy=ALWAYS    # 再開
EOF
}

case "${1:-show}" in
  show|list)
    commands
    ;;
  --apply|apply)
    if [[ "$PROJECT_ID" == "your-gcp-project" || "$PROD_DB_PASSWORD" == CHANGE_ME_* || "$STG_DB_PASSWORD" == CHANGE_ME_* ]]; then
      echo "[setup_cloud_sql] エラー: PROJECT_ID / PROD_DB_PASSWORD / STG_DB_PASSWORD を実値で指定してください（プレースホルダのままでは実行しません）" >&2
      exit 1
    fi
    export PROD_DB_PASSWORD STG_DB_PASSWORD
    echo "[setup_cloud_sql] 以下を実行します（PROJECT_ID=${PROJECT_ID}）" >&2
    bash -c "$(commands | grep -v '^#' | grep -v '^$')"
    ;;
  -h|--help)
    sed -n '2,19p' "$0"
    ;;
  *)
    echo "usage: $0 [show|--apply]" >&2
    exit 1
    ;;
esac
