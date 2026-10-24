#!/usr/bin/env bash
# 並走期間だけ使う: ホストの成果物（job_map の parity パス）を毎朝 GCS の host_snapshot/<日付>/ に上げる cron を登録／削除する（冪等。タグ KEIBA-HOST-SNAPSHOT）。
# GCP 側の parity-check（08:00 JST）がこれとGCP の成果物を突き合わせる。切替が済んだら --remove で外す。ホストで root として実行。登録はユーザーの承認後に行うこと。
# Usage: bash scripts/cron/install_host_snapshot_cron.sh [PROJECT_DIR] | --remove [PROJECT_DIR]
set -eu
MODE=install
if [ "${1:-}" = "--remove" ]; then MODE=remove; shift; fi
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-HOST-SNAPSHOT"
# 07:50 JST = 前日 22:50 UTC（cron はシステム時刻 = UTC）。朝のジョブ（〜07:40）が終わった後、GCP の parity-check（08:00 JST）の前
LINE="50 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh host-snapshot -- env PYTHONPATH=. /opt/conda/bin/python3 -m src.ops.state_sync snapshot --root ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/host_snapshot.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_host_snapshot_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
if [ "$MODE" = install ]; then echo "$LINE" >> "$TMP"; fi
crontab "$TMP"
rm -f "$TMP"
if [ "$MODE" = install ]; then echo "登録しました: ${LINE}"; else echo "削除しました（タグ ${TAG}）"; fi
