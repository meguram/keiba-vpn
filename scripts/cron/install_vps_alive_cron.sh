#!/usr/bin/env bash
# VPS 全体の停止の検知（src.ops.vps_alive）の cron を登録する（冪等。タグ KEIBA-VPS-ALIVE の行だけを入れ替える）。ホストで root として実行。
# 15 分おきに公開の画面の /version.json（Cloudflare → VPS。Cloud Run を通らない）を確かめ、2 回続けて届かなければ Slack、戻ったら Slack。
# 本番 VPS の通知なので KEIBA_ENV=prod（Slack の宛先・見出しのラベル）。GCS は使わない。
# 費用: 0 円（GCP・GCS に触れない）。一覧の正: config/scheduled_jobs.json の host-vps-alive。
# Usage: bash scripts/cron/install_vps_alive_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-VPS-ALIVE"
PY="${PY:-/opt/conda/bin/python3}"
LINES="*/15 * * * * cd \"${PROJECT_DIR}\" && bash \"${PROJECT_DIR}/scripts/cron/cron_run.sh\" vps-alive -- env KEIBA_ENV=prod PYTHONPATH=. \"${PY}\" -m src.ops.vps_alive >> \"${PROJECT_DIR}/logs/vps_alive.log\" 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
# 安全な入れ替え（root でなければ・crontab -l が失敗したら中止、退避、タグの行だけ、行数の確認）。2026-11-13
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_crontab.sh"
KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}" crontab_replace_tag "$TAG" "$LINES"
echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"
