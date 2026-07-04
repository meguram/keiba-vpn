#!/usr/bin/env bash
# =============================================================================
# git pull 1 時間ごと cron セットアップ
#
# Usage:
#   bash scripts/cron/setup_git_pull_cron.sh install
#   bash scripts/cron/setup_git_pull_cron.sh remove
#   bash scripts/cron/setup_git_pull_cron.sh show
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
GIT_PULL="${PROJECT_DIR}/scripts/cron/git_pull_hourly.sh"
LOG_DIR="${PROJECT_DIR}/logs"
CRON_TAG="# KEIBA_GIT_PULL"

chmod +x "$GIT_PULL" 2>/dev/null || true
mkdir -p "$LOG_DIR"

generate_entry() {
  cat <<EOF
# ── git pull（1 時間ごと） ─────────────────────────────────────── ${CRON_TAG}
0 * * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${GIT_PULL} ${CRON_TAG}
EOF
}

remove_existing() {
  crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab - 2>/dev/null || true
}

case "${1:-show}" in
  install)
    remove_existing
    {
      crontab -l 2>/dev/null || true
      echo ""
      generate_entry
    } | crontab -
    echo "✅ git pull 1 時間ごと cron を登録しました"
    crontab -l | grep "$CRON_TAG" || true
    ;;
  remove)
    remove_existing
    echo "✅ KEIBA_GIT_PULL cron を削除しました"
    ;;
  show)
    echo "=== git pull cron プレビュー ==="
    generate_entry
    echo ""
    echo "登録: bash $0 install"
    ;;
  *)
    echo "Usage: $0 {show|install|remove}"
    exit 1
    ;;
esac
