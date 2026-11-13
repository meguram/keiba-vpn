#!/usr/bin/env bash
# 共通ログインの月次更新 cron を登録する（冪等。タグ KEIBA-SHARED-LOGIN の行だけを入れ替える）。ホストで root として実行する。--remove で外す。
# 安全策: 現在の crontab が空・極端に少ないときは何もしない。入れ替え前に data/local/meta/cron_backups/ へ退避し、行数の増減を確かめる。
# Usage: bash scripts/cron/install_shared_login_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-SHARED-LOGIN"
# 毎日 00:05 JST = 15:05 UTC（ホストの cron は UTC。行頭の TZ=Asia/Tokyo はコマンドの環境にしか効かず、起動時刻は変わらない）。
# 2026-10-23〜11-13 は `5 0`（UTC＝JST 09:05）と書いていて、毎月 1 日の更新が 09:05 になり、その時刻にそれまでのログインが切れていた
LINE="5 15 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh shared-login -- bash ${PROJECT_DIR}/scripts/cron/shared_login_daily.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/shared_login.cron.log 2>&1 # ${TAG}"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_shared_login_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$BK")"
NOW="$(mktemp)"; NEW="$(mktemp)"
crontab -l > "$NOW" || { echo "crontab -l に失敗。中止"; exit 1; }
[ "$(grep -vcE '^\s*(#|$)' "$NOW")" -ge 20 ] || { echo "有効な行が少なすぎる crontab（$(grep -vcE '^\s*(#|$)' "$NOW") 行）。中止"; exit 1; }
cp "$NOW" "$BK"; echo "crontab を退避しました: $BK"
grep -v "# ${TAG}" "$NOW" > "$NEW" || true
[ "${2:-}" = "--remove" ] || echo "$LINE" >> "$NEW"
WANT=$(( $(grep -vcE '^\s*(#|$)' "$NOW") - $(grep -c "# ${TAG}" "$NOW" || true) + $([ "${2:-}" = "--remove" ] && echo 0 || echo 1) ))
[ "$(grep -vcE '^\s*(#|$)' "$NEW")" -eq "$WANT" ] || { echo "行数が合いません。中止"; exit 1; }
crontab "$NEW"; rm -f "$NOW" "$NEW"
[ "${2:-}" = "--remove" ] && echo "外しました" || { echo "登録しました:"; echo "$LINE"; }
