#!/bin/bash
# auto_scrape.sh — JRA スクレイピング cron ラッパー
#
# crontab から positional arg で呼び出される。
# 各 mode を src.scraper.auto_scrape の --task に変換して実行する。
#
# 使い方:
#   auto_scrape.sh thu           → raceday-eve           (木 20:00: 翌日が開催日なら出馬表取得)
#   auto_scrape.sh fri_night     → raceday-eve           (金 22:00: 明日=土曜=開催日の出馬表・追い切り)
#   auto_scrape.sh runner        → raceday-runner        (土日 07:30: 開催日常駐 T-15 スクレイプ)
#   auto_scrape.sh result_runner → raceday-result-runner (土日 07:30: 各R発走T+15速報結果取得)
#   auto_scrape.sh raceday       → raceday-evening       (土日 18:00: 本日レース結果取得)
#   auto_scrape.sh catchup       → catchup-missing       (毎日 09:00: 欠損補完)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/hirokiakataoka/miniconda3/bin/python3"
LOG_TAG="auto_scrape.sh[$1]"

cd "$PROJECT_DIR"

case "${1:-}" in
  thu)
    TASK="raceday-eve"
    ;;
  fri_night)
    TASK="raceday-eve"
    ;;
  runner)
    TASK="raceday-runner"
    if pgrep -f "auto_scrape.*--task.*raceday-runner" > /dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG already running, skip"
      exit 0
    fi
    ;;
  result_runner)
    TASK="raceday-result-runner"
    if pgrep -f "auto_scrape.*--task.*raceday-result-runner" > /dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG already running, skip"
      exit 0
    fi
    ;;
  raceday)
    TASK="raceday-evening"
    ;;
  catchup)
    # NOTE: crontab に catchup-missing の直接 Python 呼び出しも 09:00 に登録されている。
    # 重複実行を避けるため、既に同タスクが実行中なら終了する。
    TASK="catchup-missing"
    if pgrep -f "auto_scrape.*--task.*catchup-missing" > /dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG already running, skip"
      exit 0
    fi
    ;;
  *)
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG unknown mode: '${1:-}'" >&2
    exit 1
    ;;
esac

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG start → task=$TASK"
"$PYTHON" -m src.scraper.auto_scrape --task "$TASK"
echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $LOG_TAG done → task=$TASK"
