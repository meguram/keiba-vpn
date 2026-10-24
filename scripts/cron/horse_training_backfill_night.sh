#!/usr/bin/env bash
# 調教ページ（horse_training）の穴埋めを、夜間に 1 晩 N 頭ずつ進める。対象が無くなれば 0 頭で終わる（害は無い）。
# Usage: bash scripts/cron/horse_training_backfill_night.sh [PROJECT_DIR]
# 環境変数: TRAINING_BACKFILL_FROM（既定 2026-04-01）、TRAINING_BACKFILL_LIMIT（既定 2200 頭 ≒ 約 3.5 時間）
# 他の取得（開催日ランナー・定期取得・取得系ジョブ）が動いている間は待ち（backfill_horse_training_recent の wait_ok）、
# 待ちきれなければ中断する。再実行すると続きから（進捗: data/local/meta/horse_training_backfill_progress.json）。
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs
LOG="logs/horse_training_backfill.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
exec 9>"/tmp/keiba_horse_training_backfill.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>"$LOG"; exit 0; fi
echo "===== $(date -Is) START =====" >>"$LOG"
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scripts.scraping.backfill_horse_training_recent \
  --from "${TRAINING_BACKFILL_FROM:-2026-04-01}" --limit "${TRAINING_BACKFILL_LIMIT:-2200}" >>"$LOG" 2>&1
echo "===== $(date -Is) END rc=$? =====" >>"$LOG"
