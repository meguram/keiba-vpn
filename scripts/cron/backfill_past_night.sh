#!/usr/bin/env bash
# 過去分（2016〜2019 年）の補填を、時間帯・曜日で分けず一定ペース（約 2 秒おき。上限なし・エラー 1 件で即中断し、HALT 印を人が消すまで再開しない）で進める（毎時起動、実行中なら何もしない）。
# 取得の強さの制限（1 回の件数の上限・段階的な引き上げ・悪化の検知による中断・日の総件数の上限・他の取得が動いていれば待つ）は
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
# 固まり検知（src/scraper/stall_watchdog.py）が強制終了した場合（rc=75）は、次の毎時起動を待たず 60 秒置いて最大 5 回まで再起動する。
RC=0
for attempt in 1 2 3 4 5; do
  NETKEIBA_THROTTLE_MIN="${NETKEIBA_THROTTLE_MIN:-1.8}" NETKEIBA_THROTTLE_MAX="${NETKEIBA_THROTTLE_MAX:-2.2}" KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scraper.backfill --phase past "$@" >>"$LOG" 2>&1
  RC=$?
  echo "===== $(date -Is) END rc=${RC} (試行 ${attempt}) =====" >>"$LOG"
  [ "$RC" -eq 75 ] || break
  sleep 60
done
exit "$RC"
