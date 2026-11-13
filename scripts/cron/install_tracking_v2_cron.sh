#!/usr/bin/env bash
# 追走難度 v2 の週次の突き合わせを crontab に登録する（冪等。タグ KEIBA-TRACKING-V2 の行だけを入れ替える）。ホストで root として実行する。
# 事前に crontab -l を data/local/meta/cron_backups/ に退避すること（このスクリプトも退避する）。
# Usage: bash scripts/cron/install_tracking_v2_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-TRACKING-V2"
# 月曜 08:30 JST = 日曜 23:30 UTC（cron はシステム時刻 = UTC）。土日の結果（確定は金曜の weekly-update まで暫定の日もある）を突き合わせる
LINE="30 23 * * 0 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh tracking-v2-weekly -- bash ${PROJECT_DIR}/scripts/cron/tracking_v2_weekly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/tracking_v2_weekly.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
# 安全な入れ替え（root でなければ・crontab -l が失敗したら・少なすぎる crontab なら中止、退避、タグの行だけ、行数の確認）。2026-11-13
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINE"
echo "登録しました: ${LINE}"
