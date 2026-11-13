#!/usr/bin/env bash
# ドリフト週次監視を crontab に登録する（冪等。タグ KEIBA-DRIFT-MONITOR の行だけを入れ替える）。ホストで root として実行する。登録はユーザーの承認後に行うこと。
# 事前に crontab -l を data/local/meta/cron_backups/ に退避する（このスクリプトも退避する）。
# Usage: bash scripts/cron/install_drift_monitor_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-DRIFT-MONITOR"
# 月曜 10:00 JST = 月曜 01:00 UTC（cron はシステム時刻 = UTC）。週末の結果・払戻・確定オッズが揃った後に週 1 回（追走難度 v2 の突き合わせは月曜 08:30 JST）
LINE="0 1 * * 1 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh drift-monitor-weekly -- bash ${PROJECT_DIR}/scripts/cron/drift_monitor_weekly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/drift_monitor_weekly.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
# 安全な入れ替え（root でなければ・crontab -l が失敗したら・少なすぎる crontab なら中止、退避、タグの行だけ、行数の確認）。2026-11-13
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINE"
echo "登録しました: ${LINE}"
