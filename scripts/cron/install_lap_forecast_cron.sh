#!/usr/bin/env bash
# lap_forecast_daily を crontab に登録する（冪等。タグ KEIBA-LAP-FORECAST の行だけを入れ替える）。ホストで root として実行する。
# 登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。
# Usage: bash scripts/cron/install_lap_forecast_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-LAP-FORECAST"
# cron はシステム時刻（UTC）。eve: 19:30 JST = 10:30 UTC（前日夕方の raceday-eve（09:00 UTC 開始、金曜は約 75 分）の後）。
# morning: 07:30 JST = 22:30 UTC（22:00 UTC の export_tables_daily の後。スクリプト内でも同じロックを待つ）
EVE="30 10 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh lap-forecast-eve -- bash ${PROJECT_DIR}/scripts/cron/lap_forecast_daily.sh eve ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/lap_forecast_daily.cron.log 2>&1 # ${TAG}"
MORNING="30 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh lap-forecast-morning -- bash ${PROJECT_DIR}/scripts/cron/lap_forecast_daily.sh morning ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/lap_forecast_daily.cron.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_lap_forecast_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
echo "$EVE" >> "$TMP"
echo "$MORNING" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました:"; echo "$EVE"; echo "$MORNING"
