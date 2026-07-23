#!/usr/bin/env bash
# GCS → PostgreSQL 増分同期（日次 cron / 手動実行用）
#
# Usage:
#   bash scripts/cron/sync_pg_from_gcs.sh
#   KEIBA_ENV=prod bash scripts/cron/sync_pg_from_gcs.sh
#   KEIBA_PG_SYNC_ENVS="stg prod" bash scripts/cron/sync_pg_from_gcs.sh
#
# 環境変数:
#   KEIBA_PG_SYNC_RECENT_DAYS  直近 N 日（デフォルト 7）
#   KEIBA_PG_SYNC_ENVS         同期する環境（スペース区切り、デフォルト stg）
#   KEIBA_PG_SYNC_AFTER_SCRAPE スクレイプ後フック用（本スクリプトでは未使用）
set -euo pipefail
export TZ="${TZ:-Asia/Tokyo}"

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$PROJECT_DIR"

if [ -x "/opt/conda/bin/python3" ]; then
    PYTHON="/opt/conda/bin/python3"
else
    PYTHON="$(command -v python3 2>/dev/null || echo python3)"
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

RECENT_DAYS="${KEIBA_PG_SYNC_RECENT_DAYS:-7}"
ENVS="${KEIBA_PG_SYNC_ENVS:-stg}"
LOG_FILE="${LOG_DIR}/sync_pg_from_gcs.log"
TS="$(date -Is 2>/dev/null || date)"

{
  echo "===== ${TS} sync_pg_from_gcs START recent_days=${RECENT_DAYS} envs=${ENVS} ====="
} >>"$LOG_FILE"

OVERALL_EC=0
for ENV_NAME in $ENVS; do
  {
    echo "--- KEIBA_ENV=${ENV_NAME} ---"
  } >>"$LOG_FILE"

  set +e
  KEIBA_ENV="$ENV_NAME" TZ=Asia/Tokyo "$PYTHON" -m src.scripts.data.etl_stg_db \
    --recent-days "$RECENT_DAYS" \
    --skip-if-pg-complete \
    --batch-size 30 \
    >>"$LOG_FILE" 2>&1
  EC=$?
  set -e

  if [ "$EC" -ne 0 ]; then
    OVERALL_EC=$EC
    echo "ETL failed for ${ENV_NAME} (exit ${EC})" >>"$LOG_FILE"
    continue
  fi

  if [ "$ENV_NAME" = "prod" ]; then
    set +e
    KEIBA_ENV=prod TZ=Asia/Tokyo "$PYTHON" -m src.scripts.data.build_db_coverage_cache \
      --recent-days "$RECENT_DAYS" \
      >>"$LOG_FILE" 2>&1
    CACHE_EC=$?
    set -e
    if [ "$CACHE_EC" -ne 0 ]; then
      OVERALL_EC=$CACHE_EC
      echo "db_coverage cache failed for prod (exit ${CACHE_EC})" >>"$LOG_FILE"
    fi
  fi
done

TS_END="$(date -Is 2>/dev/null || date)"
{
  echo "===== ${TS_END} sync_pg_from_gcs END exit=${OVERALL_EC} ====="
  echo ""
} >>"$LOG_FILE"

exit "$OVERALL_EC"
