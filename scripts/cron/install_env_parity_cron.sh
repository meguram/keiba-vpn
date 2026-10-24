#!/usr/bin/env bash
# stg と prod の照合（毎日）を crontab に登録する（冪等。タグ KEIBA-ENV-PARITY の行だけを入れ替える）。ホストで root として実行する。登録はユーザーの承認後に行うこと。
# 事前に crontab -l を data/local/meta/cron_backups/ に退避する（このスクリプトも退避する）。
# Usage: bash scripts/cron/install_env_parity_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-ENV-PARITY"
# 毎日 JST 04:10 = 19:10 UTC（メンテナンス窓の自動反映 03:05 の後）
LINE="10 19 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh env-parity-check -- bash ${PROJECT_DIR}/scripts/cron/env_parity_check.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/env_parity_check.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_env_parity_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました: ${LINE}"
