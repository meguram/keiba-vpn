#!/usr/bin/env bash
# 直前の版（または指定した版）に即座に戻す。sudo /opt/keiba-ui/bin/rollback.sh [<リリース名>]
set -euo pipefail
. /etc/keiba-ui/vars.env 2>/dev/null || true
APP_ROOT="${APP_ROOT:-/srv/keiba-ui}"; APP_PORT="${APP_PORT:-3000}"
LOG=/var/log/keiba-ui/deploy.log
if [ -n "${1:-}" ]; then
  case "$1" in *[!A-Za-z0-9._-]*) echo "不正な名前" >&2; exit 2;; esac
  TARGET="$APP_ROOT/releases/$1"
else
  TARGET="$(cat "$APP_ROOT/shared/previous" 2>/dev/null || true)"
fi
[ -n "$TARGET" ] && [ -d "$TARGET" ] || { echo "戻し先がありません。一覧: ls -1t $APP_ROOT/releases" >&2; exit 2; }
CUR="$(readlink -f "$APP_ROOT/current" || true)"
ln -sfn "$TARGET" "$APP_ROOT/current.tmp" && mv -T "$APP_ROOT/current.tmp" "$APP_ROOT/current"
[ -n "$CUR" ] && echo "$CUR" > "$APP_ROOT/shared/previous"
systemctl restart keiba-ui
for i in $(seq 1 30); do sleep 1; curl -s -m 3 "http://127.0.0.1:$APP_PORT/version.json" | grep -q '"release"' && { echo "[$(date -Is)] ROLLBACK 完了: $(basename "$TARGET")" | tee -a "$LOG"; exit 0; }; done
echo "[$(date -Is)] ROLLBACK 後の確認に失敗: $(basename "$TARGET")" | tee -a "$LOG"; exit 1
