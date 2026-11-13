#!/usr/bin/env bash
# 共通ログインの月次更新 cron を登録する（冪等。タグ KEIBA-SHARED-LOGIN の行だけを入れ替える）。ホストで root として実行する。--remove で外す。
# 安全策: 現在の crontab が空・極端に少ないときは何もしない。入れ替え前に data/local/meta/cron_backups/ へ退避し、行数の増減を確かめる。
# Usage: bash scripts/cron/install_shared_login_cron.sh [PROJECT_DIR] [--remove]（--remove はどの位置でもよい）
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
TAG="KEIBA-SHARED-LOGIN"
# 毎日 00:05 JST = 15:05 UTC（ホストの cron は UTC。行頭の TZ=Asia/Tokyo はコマンドの環境にしか効かず、起動時刻は変わらない）。
# 2026-10-23〜11-13 は `5 0`（UTC＝JST 09:05）と書いていて、毎月 1 日の更新が 09:05 になり、その時刻にそれまでのログインが切れていた
LINE="5 15 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh shared-login -- bash ${PROJECT_DIR}/scripts/cron/shared_login_daily.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/shared_login.cron.log 2>&1 # ${TAG}"
# 安全な入れ替え（root でなければ・crontab -l が失敗したら・少なすぎる crontab なら中止、退避、タグの行だけ、行数の確認）。2026-11-13
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINE" "$MODE"
[ "$MODE" = "--remove" ] && echo "外しました" || { echo "登録しました:"; echo "$LINE"; }
