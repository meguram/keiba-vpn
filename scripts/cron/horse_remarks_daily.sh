#!/usr/bin/env bash
# 馬の戦績ページの 備考・スタート指数等（horse_race_remarks）の日次の増分取得。
# 直近 14 日に走った馬のうち、直近の出走の行が保存済みの戦績に無い馬だけ取り直す（出走から 5 日たっても行が出ない馬は諦める）。
# 1 日に取得する頭数は走った馬の数程度（数百頭）。他のスクレイピングが動いていれば待つ。Usage: bash scripts/cron/horse_remarks_daily.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
PY="${PYTHON:-/opt/conda/bin/python3}"
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. "$PY" -m src.scripts.scraping.backfill_horse_race_remarks --ran-within-days 14
