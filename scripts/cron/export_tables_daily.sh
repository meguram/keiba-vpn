#!/usr/bin/env bash
# 推論モードの履歴（race_result_flat など）を最新に保つため、当年（1 月は前年も）のテーブルを増分エクスポートする。
# --refresh: 各カテゴリが自分の flat に無いレースを GCS から追記する（これが無いと race_result に新しいレースが入らない）。
# Usage: bash scripts/cron/export_tables_daily.sh [PROJECT_DIR] [categories]
set -u
export TZ="${TZ:-Asia/Tokyo}"
PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CATS="${2:-race_result,race_shutuba,race_result_lap,race_result_on_time}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
LOG="logs/export_tables_daily.log"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
YEAR="$(date +%Y)"
YEARS="$YEAR"
[ "$(date +%m)" = "01" ] && YEARS="$((YEAR - 1)),$YEAR"
exec 9>"/tmp/keiba_export_tables_daily.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] export_tables_daily already running, skip" >>"$LOG"
  exit 0
fi
echo "===== $(date -Is) START years=$YEARS categories=$CATS =====" >>"$LOG"
"$PYTHON" -m src.scraper.export_tables --years "$YEARS" --categories "$CATS" --refresh >>"$LOG" 2>&1
EC=$?
echo "===== $(date -Is) END exit=$EC =====" >>"$LOG"
exit "$EC"
