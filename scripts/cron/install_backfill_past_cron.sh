#!/usr/bin/env bash
# 過去分（2016〜2019 年）の夜間補填の cron を登録する（冪等。タグ KEIBA-BACKFILL-PAST の行だけを入れ替える）。ホストで root として実行する。
# ユーザーの確認後にだけ実行する（既定では登録しない運用）。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。--remove で外す。
# Usage: bash scripts/cron/install_backfill_past_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-BACKFILL-PAST"
# 毎時 05 分に起動する（実行中は flock で何もしない＝実質つねに 1 本が一定ペースで進み、終わったら 1 時間以内に再開）。時間帯・曜日では分けない。
# ペースは backfill_guard が一定間隔（既定 62.5 件/時）に保ち、他の取得が動いている間は待ち、日の総件数が上限に達したら止める。
LINE="5 * * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh backfill-past -- bash ${PROJECT_DIR}/scripts/cron/yield_to_rescrape.sh backfill-past -- bash ${PROJECT_DIR}/scripts/cron/backfill_past_night.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/backfill_past.cron.log 2>&1 # ${TAG}"
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
