#!/usr/bin/env bash
# cron_prune_apply.sh の前の crontab に戻す。 sudo bash scripts/ops/cron_prune_restore.sh [バックアップのパス]（省略時は最新）
set -euo pipefail
BACKUP_DIR="${CRON_BACKUP_DIR:-/var/backups/keiba-cron}"
B="${1:-$(ls -t "$BACKUP_DIR"/crontab_*.txt 2>/dev/null | head -1)}"
[ -n "$B" ] && [ -f "$B" ] || { echo "バックアップが見つかりません" >&2; exit 1; }
crontab "$B"; echo "戻しました: $B"
