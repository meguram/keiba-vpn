#!/usr/bin/env bash
# Slack 通知の cron を登録する（冪等。タグ KEIBA-SLACK-NOTIFY の行だけを入れ替える）。ホストで root として実行。登録はユーザーの承認後に行うこと。
# 時刻は cron のシステム時刻（UTC）。JST = UTC+9。
#   結果通知   : 土日の 15:00〜20:59 JST（06:00〜11:59 UTC）に 20 分おき（結果が揃うまで待機。KEIBA_SLACK_NOTIFY_ON_FINAL=1 なら確定時に即時にも送る）
#   異常の通知 : 15 分おき
#   日次サマリー: 開催日は 21:30 JST（12:30 UTC）、非開催日は 07:30 JST（22:30 UTC）。どちらも毎日 cron を回し、通知側が開催日かで片方だけ送る
#   週次カバレッジ: 月曜 08:00 JST（日曜 23:00 UTC）
# Usage: bash scripts/cron/install_slack_notify_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-SLACK-NOTIFY"
W="cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh"
L=">> ${PROJECT_DIR}/logs/slack_notify.log 2>&1 # ${TAG}"
LINES="*/20 6-11 * * 0,6 ${W} slack-results -- bash ${PROJECT_DIR}/scripts/cron/daily_results_notify.sh ${PROJECT_DIR} ${L}
*/15 * * * * ${W} slack-scrape-alerts -- bash ${PROJECT_DIR}/scripts/cron/scrape_status_notify.sh alerts ${PROJECT_DIR} ${L}
30 12 * * * ${W} slack-scrape-summary-eve -- bash ${PROJECT_DIR}/scripts/cron/scrape_status_notify.sh summary-eve ${PROJECT_DIR} ${L}
30 22 * * * ${W} slack-scrape-summary-morning -- bash ${PROJECT_DIR}/scripts/cron/scrape_status_notify.sh summary-morning ${PROJECT_DIR} ${L}
0 23 * * 0 ${W} slack-scrape-coverage -- bash ${PROJECT_DIR}/scripts/cron/scrape_status_notify.sh coverage ${PROJECT_DIR} ${L}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_slack_notify_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
printf '%s\n' "$LINES" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"
