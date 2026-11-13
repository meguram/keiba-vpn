#!/usr/bin/env bash
# gcs_mirror_push を crontab に登録する（冪等。タグ KEIBA-GCS-MIRROR の行だけを入れ替える）。ホストで root として実行する。
# Usage: bash scripts/cron/install_gcs_mirror_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-GCS-MIRROR"
# 5 分おき（直近 2 日に更新されたファイルだけを見る。差分が無ければ GCS の一覧 5 回だけ）
LINE="*/5 * * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh gcs-mirror-push -- bash ${PROJECT_DIR}/scripts/cron/gcs_mirror_push.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/gcs_mirror_push.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
# 安全な入れ替え（root でなければ・crontab -l が失敗したら・少なすぎる crontab なら中止、退避、タグの行だけ、行数の確認）。2026-11-13
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINE"
echo "登録しました: ${LINE}"
