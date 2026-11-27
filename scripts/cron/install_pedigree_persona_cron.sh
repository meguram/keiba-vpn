#!/usr/bin/env bash
# 血統予想家の 2 つの処理を crontab に登録する（冪等。タグ KEIBA-PEDIGREE-PERSONA の行だけを入れ替える）。ホストで root として実行する。
# 入れるのは 12/1 以降の平日（統括役の決定。開催日の 07:30〜17:00 に動く新しい処理のため）。12/6 の開催は影だけ（画面に出す目印
# data/local/meta/pedigree_persona_show.on を置かない）。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。
# Usage: bash scripts/cron/install_pedigree_persona_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-PEDIGREE-PERSONA"
# cron はシステム時刻（UTC）。eve: 20:30 JST = 11:30 UTC（raceday-eve 18:00・lap_forecast 19:30 の後）。
# t60: 09:00〜16:55 JST = 00:00〜07:55 UTC に 5 分おき（開催日でなければ前の夜の段の出力が無く、すぐ終わる）。
EVE="30 11 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh pedigree-persona-eve -- bash ${PROJECT_DIR}/scripts/cron/pedigree_persona.sh eve ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/pedigree_persona.cron.log 2>&1 # ${TAG}"
T60="*/5 0-7 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh pedigree-persona-t60 -- bash ${PROJECT_DIR}/scripts/cron/pedigree_persona.sh t60 ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/pedigree_persona.cron.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_pedigree_persona_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
echo "$EVE" >> "$TMP"
echo "$T60" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました:"; echo "$EVE"; echo "$T60"
