#!/usr/bin/env bash
# スクレイピングの完了ステータスを Slack に通知する（alerts / summary / coverage）。有効化は KEIBA_SLACK_NOTIFY=1（.env）。
# Usage: bash scripts/cron/scrape_status_notify.sh <alerts|summary-eve|summary-morning|coverage> [PROJECT_DIR]
#   summary-eve     : 開催日の夜のサマリー（非開催日は何もしない）／ summary-morning : 非開催日の朝のサマリー（開催日は何もしない）
#   alerts   : 15 分おき（異常の即時通知。同じ異常は復旧まで 60 分おきに最大 3 回、復旧時に 1 回）
#   summary  : 開催日は 21:30 JST、非開催日は 07:30 JST に 1 回（cron の時刻は UTC: 12:30 / 22:30 のように）
#   coverage : 月曜 08:00 JST に 1 回
set -u
MODE="${1:?mode: alerts|summary-eve|summary-morning|coverage}"
PROJECT_DIR="${2:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
[ "${KEIBA_SLACK_NOTIFY:-0}" = "1" ] || { echo "KEIBA_SLACK_NOTIFY!=1 のため何もしません"; exit 0; }
export TZ="${TZ:-Asia/Tokyo}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
case "$MODE" in
  summary-eve)     ARGS="--mode summary --when raceday" ;;
  summary-morning) ARGS="--mode summary --when nonraceday" ;;
  alerts|coverage) ARGS="--mode $MODE" ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
exec nice -n 10 "$PY" -m src.scripts.data.scrape_status_notify $ARGS --send
