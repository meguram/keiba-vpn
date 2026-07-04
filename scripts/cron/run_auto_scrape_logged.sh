#!/usr/bin/env bash
# cron から auto_scrape を実行し、ログに開始・終了境界を必ず書き出す。
# Usage: run_auto_scrape_logged.sh <PROJECT_DIR> <task> <log_file_rel_to_project>
# 例: bash scripts/cron/run_auto_scrape_logged.sh /path/to/keiba-vpn raceday-eve logs/raceday_eve.log
set -u
# 境界行の date と Python ログの asctime を日本時間に揃える（stg/Docker UTC 対策）
export TZ="${TZ:-Asia/Tokyo}"
PROJECT_DIR="${1:?project dir}"
TASK="${2:?task}"
LOG_REL="${3:?log path relative to project}"
cd "$PROJECT_DIR" || exit 1
LOG_FILE="${PROJECT_DIR}/${LOG_REL}"
mkdir -p "$(dirname "$LOG_FILE")"
# conda 優先で Python を解決（cron の PATH が限定的な環境向け）
if [ -z "${PYTHON:-}" ]; then
    if [ -x "/opt/conda/bin/python3" ]; then
        PYTHON="/opt/conda/bin/python3"
    else
        PYTHON="$(command -v python3 2>/dev/null || echo python3)"
    fi
fi
TS_START="$(date -Is 2>/dev/null || date)"
{
  echo "===== ${TS_START} cron START task=${TASK} pid=$$ cwd=$(pwd) ====="
} >>"$LOG_FILE"
set +e
"$PYTHON" -m src.scraper.auto_scrape --task "$TASK" >>"$LOG_FILE" 2>&1
EC=$?
set -e
TS_END="$(date -Is 2>/dev/null || date)"
{
  echo "===== ${TS_END} cron END   task=${TASK} exit=${EC} ====="
  echo ""
} >>"$LOG_FILE"
exit "$EC"
