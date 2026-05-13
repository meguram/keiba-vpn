#!/usr/bin/env bash
# =============================================================================
# 騎手・調教師成績統計（jockey_trainer_stats）の定期再生成
#
#   data/features/jockey_trainer_stats/*.parquet を
#   data/local/tables の race_result 全年から再計算して上書きする。
#
# Usage:
#   bash scripts/cron/update_jockey_trainer_stats.sh
#   bash scripts/cron/update_jockey_trainer_stats.sh --nar   # 地方開催も含める
#
# cron 例は scripts/cron/setup_jockey_trainer_stats_cron.sh を参照。
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/jockey_trainer_stats.log"
LOCK_FILE="${LOG_DIR}/jockey_trainer_stats.lock"

PYTHON="${PYTHON:-python3}"
EXTRA=()
if [ "${1:-}" = "--nar" ]; then
  EXTRA=(--nar)
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) update_jockey_trainer_stats: skip (lock held)" | tee -a "$LOG_FILE"
  exit 0
fi

{
  echo "======== $(date -Is) update_jockey_trainer_stats start ========"
  if ! "$PYTHON" -m src.pipeline.build_jockey_trainer_stats "${EXTRA[@]}"; then
    ec=$?
    echo "======== $(date -Is) update_jockey_trainer_stats end FAILED exit=${ec} ========"
    exit "$ec"
  fi
  echo "======== $(date -Is) update_jockey_trainer_stats end ok ========"
} >>"$LOG_FILE" 2>&1
