#!/usr/bin/env bash
# 過去分（2016〜2019 年）の補填を、夜間の時間窓（既定 02:30-05:45 JST）に、平日の朝だけ進める。
# 取得の強さの制限（夜ごとの件数の上限・段階的な引き上げ・悪化の検知による中断・土日は補填しない・他の取得が動いていれば待つ）は
# src/scraper/backfill_guard.py と backfill.py の --phase past が行う。仕様・運用: docs/operations/backfill_2016.md
# Usage: bash scripts/cron/backfill_past_night.sh [PROJECT_DIR] [追加の引数...]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
[ $# -gt 0 ] && shift
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
LOG="logs/backfill_past.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
exec 9>"/tmp/keiba_backfill_past.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>"$LOG"; exit 0; fi
echo "===== $(date -Is) START =====" >>"$LOG"
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scraper.backfill --phase past "$@" >>"$LOG" 2>&1
RC=$?
echo "===== $(date -Is) END rc=${RC} =====" >>"$LOG"
exit "$RC"
