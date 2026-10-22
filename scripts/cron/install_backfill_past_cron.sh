#!/usr/bin/env bash
# 過去分（2016〜2019 年）の夜間補填の cron を登録する（冪等。タグ KEIBA-BACKFILL-PAST の行だけを入れ替える）。ホストで root として実行する。
# ユーザーの確認後にだけ実行する（既定では登録しない運用）。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。--remove で外す。
# Usage: bash scripts/cron/install_backfill_past_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-BACKFILL-PAST"
# cron は UTC。17:45 UTC（日〜木）= 02:45 JST（月〜金）。時間窓 02:30-05:45 JST の内側で、05:45 JST までに止まる。
# 土日（開催日）の朝は走らない。21:00 UTC（06:00 JST）の backfill-all-horse より前に終わる。他の取得が動いていれば最大 30 分待ち、待ちきれなければその夜は始めない。
LINE="45 17 * * 0-4 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh backfill-past -- bash ${PROJECT_DIR}/scripts/cron/yield_to_rescrape.sh backfill-past -- bash ${PROJECT_DIR}/scripts/cron/backfill_past_night.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/backfill_past.cron.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_backfill_past_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
if [ "${2:-}" != "--remove" ]; then echo "$LINE" >> "$TMP"; fi
crontab "$TMP"
rm -f "$TMP"
if [ "${2:-}" = "--remove" ]; then echo "外しました"; else echo "登録しました:"; echo "$LINE"; fi
