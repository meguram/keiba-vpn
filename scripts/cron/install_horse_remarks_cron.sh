#!/usr/bin/env bash
# horse_remarks_daily を crontab に登録する（冪等。タグ KEIBA-HORSE-REMARKS の行だけを入れ替える）。ホストで root として実行する。
# Usage: bash scripts/cron/install_horse_remarks_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-HORSE-REMARKS"
# 毎日 03:00 JST = 18:00 UTC（cron はシステム時刻 = UTC）。夜間のエクスポート・recent-retry の後
LINE="0 18 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh horse-remarks-daily -- bash ${PROJECT_DIR}/scripts/cron/horse_remarks_daily.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/horse_remarks_daily.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました: ${LINE}"
