#!/usr/bin/env bash
# odds_drift の週次レポートを crontab に登録する（冪等。タグ KEIBA-ODDS-DRIFT の行だけを入れ替える）。ホストで root として実行する。
# Usage: bash scripts/cron/install_odds_drift_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-ODDS-DRIFT"
# 日曜 22:00 JST = 13:00 UTC（cron はシステム時刻 = UTC）。夕方の確定オッズ（17:30 JST）の取得後
LINE="0 13 * * 0 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh odds-drift-weekly -- bash ${PROJECT_DIR}/scripts/cron/odds_drift_weekly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/odds_drift_weekly.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました: ${LINE}"
