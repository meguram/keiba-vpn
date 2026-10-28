#!/usr/bin/env bash
# 気象庁の時別値（風向・風速など）を、直近 3 日の補い → 2016 年以降の取得（新しい日から）の順に進める。毎時起動・flock で重複なし。
# エラー（HTTP が 200 以外・表が読めない）は 1 件で中断し、data/local/meta/jma_weather_halt.json を書く。確認して削除するまで再開しない。
# 仕様・運用: docs/operations/jma_weather.md   Usage: bash scripts/cron/jma_weather_run.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs data/local/meta
LOG="logs/jma_weather.log"; HALT="data/local/meta/jma_weather_halt.json"; PY="${PYTHON:-/opt/conda/bin/python3}"
exec 9>"/tmp/keiba_jma_weather.lock"
flock -n 9 || exit 0
if [ -e "$HALT" ]; then echo "[$(date -Is)] 中断印があるため始めません（確認して ${HALT} を削除）" >>"$LOG"; exit 0; fi
echo "===== $(date -Is) START =====" >>"$LOG"
run() { KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scraper.jma_weather "$@" >>"$LOG" 2>&1; }
run daily; RC=$?
[ "$RC" -eq 0 ] && { run fetch --from 20160101; RC=$?; }
[ "$RC" -eq 3 ] && printf '{"at":"%s","rc":%s,"log":"%s"}\n' "$(date -Is)" "$RC" "$LOG" > "$HALT"
echo "===== $(date -Is) END rc=${RC} =====" >>"$LOG"
exit "$RC"
