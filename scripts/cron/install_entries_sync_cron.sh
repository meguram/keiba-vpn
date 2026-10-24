#!/usr/bin/env bash
# 出馬表の段階収集（entries-sync）を crontab に登録する（冪等。タグ KEIBA-ENTRIES-SYNC の行だけを入れ替える）。ホストで root として実行する。登録はユーザーの承認後に行うこと。
# 10 分おきに起動するが、config/entries_schedule.json の checkpoint（JST）の時刻から window 以内で未実施のときだけ取得する（それ以外は数秒で終了）。
# cron は UTC: JST 08:00–22:00 = UTC 23:00–13:00 の範囲だけ起動する。--remove で外す。
# Usage: bash scripts/cron/install_entries_sync_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
REMOVE=0
for a in "$@"; do [ "$a" = "--remove" ] && REMOVE=1; done
TAG="KEIBA-ENTRIES-SYNC"
LINE="*/10 23,0-13 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh entries-sync -- bash ${PROJECT_DIR}/scripts/cron/yield_to_rescrape.sh entries-sync -- bash ${PROJECT_DIR}/scripts/cron/run_auto_scrape_logged.sh ${PROJECT_DIR} entries-sync logs/entries_sync.log # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_entries_sync_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
[ "$REMOVE" = 1 ] || echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
if [ "$REMOVE" = 1 ]; then echo "外しました（${TAG}）"; else echo "登録しました: ${LINE}"; fi
