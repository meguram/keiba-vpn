#!/usr/bin/env bash
# 凍結候補の前向き追跡を crontab に登録する（冪等。タグ KEIBA-FROZEN-TRACKING の行だけを入れ替える）。ホストで root として実行する。登録はユーザーの承認後に行うこと。
# 登録前に crontab -l を data/local/meta/cron_backups/ に退避する。
# Usage: bash scripts/cron/install_frozen_tracking_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-FROZEN-TRACKING"
# 月曜 10:30 JST = 月曜 01:30 UTC（cron はシステム時刻 = UTC）。drift_monitor_weekly（月曜 01:00 UTC）の後。
LINE="30 1 * * 1 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh frozen-tracking-weekly -- bash ${PROJECT_DIR}/scripts/cron/frozen_tracking_weekly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/frozen_tracking_weekly.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_frozen_tracking_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました: ${LINE}"
