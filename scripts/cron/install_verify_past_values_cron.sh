#!/usr/bin/env bash
# 過去分（2016〜2019 年）の値の検査（src.scripts.data.verify_past_values）の cron を登録する（冪等。タグ KEIBA-PAST-VALUE-CHECK の行だけを入れ替える）。
# ホストで root として実行。2026-11-13: 2 時間おき・cron_run.sh なしの手入れの行を、1 日 1 回（JST 04:25 = UTC 19:25）・cron_run.sh で包む形に入れ替える。
# 費用: GCS の全件一覧は取らない（280a6ab0）。未検査が 0 件の日は読み出しも 0。最悪でも 1 回 500 件（月 約 1 円）。
# Usage: bash scripts/cron/install_verify_past_values_cron.sh [PROJECT_DIR] [--remove]
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
TAG="KEIBA-PAST-VALUE-CHECK"
PY="${PY:-/opt/conda/bin/python3}"
LINES="25 19 * * * cd \"${PROJECT_DIR}\" && TZ=Asia/Tokyo bash \"${PROJECT_DIR}/scripts/cron/cron_run.sh\" verify-past-values -- env KEIBA_ENV=stg PYTHONPATH=. nice -n 10 \"${PY}\" -m src.scripts.data.verify_past_values >> \"${PROJECT_DIR}/logs/verify_past_values.log\" 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINES" "$MODE"
[ "$MODE" = "--remove" ] && echo "外しました（${TAG}）" || { echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"; }
