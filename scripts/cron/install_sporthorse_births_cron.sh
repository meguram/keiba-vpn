#!/usr/bin/env bash
# Sporthorse 夜間バッチを crontab に登録（冪等。タグ KEIBA-SPORTHORSE-BIRTHS の行だけ入れ替え）。--remove で外す。
# 2026-11-13 に外した（研究専用・不採用の結論済みで、毎日「SKIP: セッション未登録」だけだった。root と jovyan の両方の crontab に行があった）。
# 安全策: crontab -l が「crontab が無い」以外で失敗したら中止。入れ替え前に data/local/meta/cron_backups/ へ退避し、行数の増減を確かめる。
# Usage: bash scripts/cron/install_sporthorse_births_cron.sh [PROJECT_DIR] [--remove]（--remove はどの位置でもよい）
set -eu
MODE=install; PROJECT_DIR=""
for a in "$@"; do
  case "$a" in
    --remove) MODE=--remove ;;
    -*) echo "不明な引数: $a"; exit 2 ;;
    *) PROJECT_DIR="$a" ;;
  esac
done
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
[ -d "$PROJECT_DIR" ] || { echo "PROJECT_DIR がありません: $PROJECT_DIR"; exit 2; }
TAG="KEIBA-SPORTHORSE-BIRTHS"
# 毎日 22:00 JST = 13:00 UTC
LINE="0 13 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh sporthorse-births-nightly -- bash ${PROJECT_DIR}/scripts/cron/sporthorse_births_nightly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/sporthorse_births_nightly.log 2>&1 # ${TAG}"
BK_DIR="${KEIBA_CRON_BACKUP_DIR:-${PROJECT_DIR}/data/local/meta/cron_backups}"
mkdir -p "$BK_DIR"
NOW="$(mktemp)"; NEW="$(mktemp)"
if ! crontab -l > "$NOW" 2>"$NEW"; then
  grep -qi "no crontab" "$NEW" || { echo "crontab -l に失敗。中止: $(cat "$NEW")"; exit 1; }
  : > "$NOW"
fi
cp "$NOW" "${BK_DIR}/crontab_before_sporthorse_$(id -un)_$(date +%Y%m%d_%H%M%S).txt"
grep -v "# ${TAG}" "$NOW" > "$NEW" || true
if [ "$MODE" != "--remove" ]; then
  NIGHTLY="${PROJECT_DIR}/scripts/cron/sporthorse_births_nightly.sh"
  [ -x "$NIGHTLY" ] || chmod +x "$NIGHTLY" 2>/dev/null || true
  mkdir -p "${PROJECT_DIR}/logs"
  echo "$LINE" >> "$NEW"
fi
WANT=$(( $(grep -vcE '^\s*(#|$)' "$NOW" || true) - $(grep "# ${TAG}" "$NOW" | grep -vcE '^\s*#' || true) + $([ "$MODE" = "--remove" ] && echo 0 || echo 1) ))
[ "$(grep -vcE '^\s*(#|$)' "$NEW" || true)" -eq "$WANT" ] || { echo "行数が合いません。中止"; exit 1; }
crontab "$NEW"; rm -f "$NOW" "$NEW"
[ "$MODE" = "--remove" ] && echo "外しました（$(id -un) の crontab）" || { echo "登録しました:"; echo "  $LINE"; }
