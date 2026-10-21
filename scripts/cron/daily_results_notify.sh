#!/usr/bin/env bash
# 開催日の結果通知（予測成績の集計・推奨馬券の回収率と的中率）を Slack に送る。cron から 20 分おきに呼ぶ想定。
# 結果が揃っていなければ何もせず終了（終了コード 10 は正常な待機）。1 日 1 回だけ送る（送信済みフラグ data/local/meta/slack_sent/）。
# 有効化: .env に KEIBA_SLACK_NOTIFY=1 と SLACK_WEBHOOK_URL（任意で SLACK_WEBHOOK_URL_RESULTS）。未設定なら何もしない。
# 20:30 JST を過ぎても確定しない日は、暫定のまま 1 回送る（--allow-provisional）。
# Usage: bash scripts/cron/daily_results_notify.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
[ "${KEIBA_SLACK_NOTIFY:-0}" = "1" ] || { echo "KEIBA_SLACK_NOTIFY!=1 のため何もしません"; exit 0; }
export TZ="${TZ:-Asia/Tokyo}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
EXTRA=""
if [ "$(date +%H%M)" -ge 2030 ]; then EXTRA="--allow-provisional"; fi
nice -n 10 "$PY" -m src.scripts.data.daily_results_notify --date "$(date +%Y%m%d)" --refresh --send $EXTRA
rc=$?
case "$rc" in
  0|10|11|12) exit 0 ;;   # 送信済み・待機・開催なしは正常
  *) echo "daily_results_notify rc=${rc}"; exit "$rc" ;;
esac
