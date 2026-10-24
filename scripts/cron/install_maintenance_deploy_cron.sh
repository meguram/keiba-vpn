#!/usr/bin/env bash
# メンテナンス時間帯の自動反映ランナー（と、任意で開始/終了の Slack 通知）を crontab に登録する。ホストで root として実行する。
# 冪等（タグ KEIBA-MAINTENANCE-DEPLOY / KEIBA-MAINTENANCE-NOTIFY の行だけを入れ替える）。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。--remove で外す。
# cron は UTC: 18:05 UTC = 03:05 JST（窓 03:00–04:00 JST の中）。反映が無い日は何もしない（pending が空なら即終了）。
# Usage: bash scripts/cron/install_maintenance_deploy_cron.sh [PROJECT_DIR] [--with-notify] [--remove]
set -eu
PROJECT_DIR="/home/jovyan/work/keiba-vpn"; WITH_NOTIFY=0; REMOVE=0
for a in "$@"; do
  case "$a" in
    --with-notify) WITH_NOTIFY=1 ;;
    --remove) REMOVE=1 ;;
    *) PROJECT_DIR="$a" ;;
  esac
done
TAG1="KEIBA-MAINTENANCE-DEPLOY"; TAG2="KEIBA-MAINTENANCE-NOTIFY"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_maintenance_deploy_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v -e "# ${TAG1}" -e "# ${TAG2}" > "$TMP" || true
if [ "$REMOVE" = 0 ]; then
  echo "5 18 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh maintenance-deploy -- bash ${PROJECT_DIR}/scripts/ops/maintenance_deploy.sh ${PROJECT_DIR} # ${TAG1}" >> "$TMP"
  if [ "$WITH_NOTIFY" = 1 ]; then
    echo "0 18 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh maintenance-notify-start -- bash ${PROJECT_DIR}/scripts/cron/maintenance_notify.sh start ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/maintenance_notify.log 2>&1 # ${TAG2}" >> "$TMP"
    echo "2 19 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh maintenance-notify-end -- bash ${PROJECT_DIR}/scripts/cron/maintenance_notify.sh end ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/maintenance_notify.log 2>&1 # ${TAG2}" >> "$TMP"
  fi
fi
crontab "$TMP"; rm -f "$TMP"
if [ "$REMOVE" = 1 ]; then echo "外しました（${TAG1} / ${TAG2}）"; else echo "登録しました:"; crontab -l | grep -e "# ${TAG1}" -e "# ${TAG2}"; fi
