#!/usr/bin/env bash
# horse_training の穴埋め（夜間）を crontab に登録する（冪等。タグ KEIBA-TRAINING-BACKFILL の行だけを入れ替える）。ホストで root として実行する。
# 登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。--remove で外す。
# Usage: bash scripts/cron/install_horse_training_backfill_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-TRAINING-BACKFILL"
# cron は UTC。13:00 UTC = 22:00 JST の月〜木（開催日の前夜[金・土]と開催日[土・日]を避ける。21:00 JST の new-horse-pedigree の後）
LINE="0 13 * * 1-4 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh horse-training-backfill -- bash ${PROJECT_DIR}/scripts/cron/horse_training_backfill_night.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/horse_training_backfill.cron.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_training_backfill_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
if [ "${2:-}" != "--remove" ]; then echo "$LINE" >> "$TMP"; fi
crontab "$TMP"
rm -f "$TMP"
if [ "${2:-}" = "--remove" ]; then echo "外しました"; else echo "登録しました:"; echo "$LINE"; fi
