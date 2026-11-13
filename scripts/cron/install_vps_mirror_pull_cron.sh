#!/usr/bin/env bash
# VPS が取ったレース一覧・発走時刻表（ops/mirror_vps/）を、ホストに取り込む cron を登録する（冪等。タグ KEIBA-VPS-MIRROR-PULL の行だけ入れ替える）。
# ホストで root として実行。docs/operations/vps_only_fetch.md の段 1。
# 取り込むのは、その木を作る取得の担当が VPS に移った木だけ（config/fetch_owner.json）。移っていなければ GCS に触れずに終わる（費用 0）。
# 時刻は cron のシステム時刻（UTC）。5 分おき。JRA の馬場は受け渡さない（ホストが取る。docs/operations/vps_only_fetch.md）。
# 費用（最悪・月）: 目録 1 件（約 5KB）× 288 回/日 × 30 ≒ 43MB・約 1.5 円 ＋ 変わったファイル（1 件約 1KB）。全件の一覧はしない。
# Usage: bash scripts/cron/install_vps_mirror_pull_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-VPS-MIRROR-PULL"
PY="${PY:-/opt/conda/bin/python3}"
W="cd \"${PROJECT_DIR}\" && TZ=Asia/Tokyo bash \"${PROJECT_DIR}/scripts/cron/cron_run.sh\""
L=">> \"${PROJECT_DIR}/logs/vps_mirror_pull.log\" 2>&1 # ${TAG}"
LINES="*/5 * * * * ${W} vps-mirror-pull-lists -- env KEIBA_ENV=stg PYTHONPATH=. \"${PY}\" -m src.ops.gcs_mirror pull-vps --trees race_lists,race_day_schedule ${L}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_vps_mirror_pull_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
printf '%s\n' "$LINES" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"
