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
# KEIBA_ENV が dev に解決される環境で cron を回すと、取得したデータが GCS ではなくローカルの dev ストア(data/dev_mock)に
# 書かれてしまう（2026-10-09〜10 に stg ホストで発生）。誤設定を検知したら、取得せずに終了する。
# 開発 PC でわざと動かすときは KEIBA_ALLOW_DEV_CRON=1。
if [ "${KEIBA_ALLOW_DEV_CRON:-0}" != "1" ]; then
  "$PYTHON" -c "import os,sys; from src.utils.project_env import load_project_dotenv; load_project_dotenv(); sys.exit(3 if os.environ.get('KEIBA_ENV','').strip().lower()=='dev' else 0)" >/dev/null 2>&1
  if [ "$?" -eq 3 ]; then
    echo "KEIBA_ENV が dev に解決されたため、取得せずに終了する（GCS に保存されないため）。.env の KEIBA_ENV を確認すること" >>"$LOG_FILE"
    echo "===== $(date -Is 2>/dev/null || date) cron END   task=${TASK} exit=3 (dev 環境のため中止) =====" >>"$LOG_FILE"
    exit 3
  fi
fi
"$PYTHON" -m src.scraper.auto_scrape --task "$TASK" >>"$LOG_FILE" 2>&1
EC=$?
set -e
TS_END="$(date -Is 2>/dev/null || date)"
{
  echo "===== ${TS_END} cron END   task=${TASK} exit=${EC} ====="
  echo ""
} >>"$LOG_FILE"
exit "$EC"
