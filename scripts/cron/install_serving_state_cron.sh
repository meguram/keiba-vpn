#!/usr/bin/env bash
# v4 サービングの日次状態の cron を登録する（冪等。タグ KEIBA-SERVING-STATE の行だけを入れ替える）。ホストで root として実行する。--remove で外す。
# - serving-state-precompute（毎日 07:35 JST = 22:35 UTC）: 開催日の朝に、その日の全レースの馬主・馬自身の群（v4_day.parquet）を作る。
#   ホストの馬体重ランナー（v6）がこれを使い回す（GCP の day-prep が止まった分の肩代わり。無いと最初のレースで作るので 70〜90 秒遅れる）。
# - serving-state-profile（毎日 08:10 JST = 23:10 UTC）: 履歴にいて馬主が未登録の馬を補う（profile_dyn.parquet）。要否を確認中（2026-11-13）。
# 2026-10-19 に手で登録した行（タグなし）を、2026-11-13 にこのスクリプトでタグ付きに入れ替えた。タグの無い旧行（同じジョブ名の cron_run.sh の行）も消す。
# 安全策: 現在の crontab が空・極端に少ないときは何もしない。入れ替え前に data/local/meta/cron_backups/ へ退避し、行数の増減を確かめる。
# Usage: bash scripts/cron/install_serving_state_cron.sh [PROJECT_DIR] [--remove]（--remove はどの位置でもよい）
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
TAG="KEIBA-SERVING-STATE"
L1="35 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash scripts/cron/cron_run.sh serving-state-precompute -- env PYTHONPATH=. KEIBA_ENV=stg /opt/conda/bin/python3 -m src.scripts.data.update_serving_state precompute --date \$(TZ=Asia/Tokyo date +\\%Y\\%m\\%d) # ${TAG}"
L2="10 23 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash scripts/cron/cron_run.sh serving-state-profile -- env PYTHONPATH=. KEIBA_ENV=stg /opt/conda/bin/python3 -m src.scripts.data.update_serving_state profile # ${TAG}"
BK="${KEIBA_CRON_BACKUP_DIR:-${PROJECT_DIR}/data/local/meta/cron_backups}/crontab_before_serving_state_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$BK")"
NOW="$(mktemp)"; NEW="$(mktemp)"
crontab -l > "$NOW" || { echo "crontab -l に失敗。中止"; exit 1; }
[ "$(grep -vcE '^\s*(#|$)' "$NOW")" -ge "${KEIBA_CRON_MIN_LINES:-20}" ] || { echo "有効な行が少なすぎる crontab（$(grep -vcE '^\s*(#|$)' "$NOW") 行）。中止"; exit 1; }
cp "$NOW" "$BK"; echo "crontab を退避しました: $BK"
OLD_RE="# ${TAG}|cron_run\.sh serving-state-(precompute|profile) |^# v4 サービングの日次状態更新|^# 履歴にいて馬主が未登録の馬を補う"   # 旧行とその見出しのコメント
grep -vE "$OLD_RE" "$NOW" > "$NEW" || true
[ "$MODE" = "--remove" ] || printf '%s\n%s\n' "$L1" "$L2" >> "$NEW"
N_OLD=$(grep -E "$OLD_RE" "$NOW" | grep -vcE '^\s*#' || true)
WANT=$(( $(grep -vcE '^\s*(#|$)' "$NOW") - N_OLD + $([ "$MODE" = "--remove" ] && echo 0 || echo 2) ))
[ "$(grep -vcE '^\s*(#|$)' "$NEW")" -eq "$WANT" ] || { echo "行数が合いません。中止"; exit 1; }
crontab "$NEW"; rm -f "$NOW" "$NEW"
[ "$MODE" = "--remove" ] && echo "外しました" || { echo "登録しました:"; echo "$L1"; echo "$L2"; }
