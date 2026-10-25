#!/usr/bin/env bash
# 出馬表の取りこぼし検査（shutuba-audit）を crontab に登録する（冪等。タグ KEIBA-SHUTUBA-AUDIT の行だけを入れ替える）。ホストで root として実行する。登録はユーザーの承認後に行うこと。
# 毎日 JST 06:40（= UTC 21:40）に 1 回。過去 7 日で馬番が確定しないままの出馬表を再取得キューに入れる（1 レース 3 回まで）。--remove で外す。
# Usage: bash scripts/cron/install_shutuba_audit_cron.sh [PROJECT_DIR] [--remove]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
REMOVE=0
for a in "$@"; do [ "$a" = "--remove" ] && REMOVE=1; done
TAG="KEIBA-SHUTUBA-AUDIT"
LINE="40 21 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh shutuba-audit -- bash ${PROJECT_DIR}/scripts/cron/yield_to_rescrape.sh shutuba-audit -- bash ${PROJECT_DIR}/scripts/cron/run_auto_scrape_logged.sh ${PROJECT_DIR} shutuba-audit logs/shutuba_audit.log # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_shutuba_audit_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
[ "$REMOVE" = 1 ] || echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
if [ "$REMOVE" = 1 ]; then echo "外しました（${TAG}）"; else echo "登録しました: ${LINE}"; fi
