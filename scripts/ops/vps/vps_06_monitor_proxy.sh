#!/usr/bin/env bash
# 監視ポータルの VPS 側の入口を作る/外す(冪等。root で実行)。
#   sudo bash vps_06_monitor_proxy.sh [--dry-run]        # 作る/更新する
#   sudo bash vps_06_monitor_proxy.sh --remove           # 外す(ロールバック)
# 前提: /etc/keiba-ui/vars.env に MONITOR_CLOUDRUN_HOST=<keiba-monitor-….a.run.app>(https:// なし)、
#       /etc/nginx/keiba-proxykey.conf(render_nginx_proxykey.sh。API と同じ鍵)、keiba-ui-zones.conf(limit_req ゾーン)。
# 作ったあと 127.0.0.1:MONITOR_PORT(既定 8082)の MONITOR_HEALTH_PATH(既定 /api/overview。nginx が共有鍵を付けるので 200 なら鍵も通っている)が
# 200 でなければ、自動で元に戻す。※ /healthz は Cloud Run の前段(Google Frontend)が横取りして 404 を返すため使えない。
# Cloudflare 側(ユーザー作業): prod のトンネルに monitor.megu-keiba.com → http://localhost:8082 を追加し、ホストのトンネルからは外す。
set -euo pipefail
DRY=0; REMOVE=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --remove) REMOVE=1;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done
[ "$(id -u)" = 0 ] || [ "$DRY" = 1 ] || { echo "root で実行してください" >&2; exit 2; }
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VARS="${VARS_FILE:-/etc/keiba-ui/vars.env}"
SITE=/etc/nginx/sites-available/keiba-monitor; LINK=/etc/nginx/sites-enabled/keiba-monitor
TPL="${TEMPLATE:-$HERE/templates/nginx-keiba-monitor.conf}"
run() { printf '+'; printf ' %q' "$@"; printf '\n'; [ "$DRY" = 1 ] || "$@"; }

if [ "$REMOVE" = 1 ]; then
  run rm -f "$LINK" "$SITE"
  [ "$DRY" = 1 ] || { nginx -t >/dev/null 2>&1 && systemctl reload nginx; }
  echo "外しました(監視ポータルの入口)"; exit 0
fi

# shellcheck disable=SC1090
[ -r "$VARS" ] && . "$VARS"
MONITOR_PORT="${MONITOR_PORT:-8082}"; MONITOR_HEALTH_PATH="${MONITOR_HEALTH_PATH:-/api/overview}"
[ -n "${MONITOR_CLOUDRUN_HOST:-}" ] || { echo "vars.env に MONITOR_CLOUDRUN_HOST がありません(例: keiba-monitor-xxxx-an.a.run.app。https:// は付けない)" >&2; exit 2; }
case "$MONITOR_CLOUDRUN_HOST" in *[!A-Za-z0-9.-]*) echo "MONITOR_CLOUDRUN_HOST に使えない文字があります" >&2; exit 2;; esac
[ "$DRY" = 1 ] || [ -s /etc/nginx/keiba-proxykey.conf ] || { echo "/etc/nginx/keiba-proxykey.conf がありません(先に render_nginx_proxykey.sh)" >&2; exit 2; }

RE="$(printf '%s' "$MONITOR_CLOUDRUN_HOST" | sed 's/\./\\./g')"; RE="${RE//\\/\\\\}"
NEW="$(mktemp)"; trap 'rm -f "$NEW"' EXIT
sed -e "s/__MONITOR_CLOUDRUN_HOST_RE__/${RE}/g" -e "s/__MONITOR_CLOUDRUN_HOST__/${MONITOR_CLOUDRUN_HOST}/g" -e "s/__MONITOR_PORT__/${MONITOR_PORT}/g" "$TPL" > "$NEW"
if grep -q '__[A-Z_]*__' "$NEW"; then echo "置換されていない項目が残っています" >&2; exit 1; fi
echo "監視ポータルの入口: 127.0.0.1:${MONITOR_PORT} → https://${MONITOR_CLOUDRUN_HOST}"
[ "$DRY" = 1 ] && { echo "(dry-run) ${SITE} に置いて有効化し、nginx を再読み込みします"; exit 0; }

BAK=""; [ -f "$SITE" ] && { BAK="/etc/nginx/keiba-monitor.bak.$(date +%Y%m%d%H%M%S)"; cp -a "$SITE" "$BAK"; }
install -m 644 "$NEW" "$SITE"; ln -sfn "$SITE" "$LINK"
revert() { if [ -n "$BAK" ]; then cp -a "$BAK" "$SITE"; else rm -f "$LINK" "$SITE"; fi; nginx -t >/dev/null 2>&1 && systemctl reload nginx || true; echo "元に戻しました: $1" >&2; exit 1; }
nginx -t >/dev/null 2>&1 || revert "nginx -t 失敗"
systemctl reload nginx; sleep 2
c="$(curl -s -o /dev/null -w '%{http_code}' -m 30 "http://127.0.0.1:${MONITOR_PORT}${MONITOR_HEALTH_PATH}" || true)"
[ "$c" = 200 ] || revert "${MONITOR_HEALTH_PATH} が ${c}"
echo "OK: 監視ポータルの入口を有効にしました(${MONITOR_HEALTH_PATH} 200)。次: Cloudflare のトンネルに monitor.megu-keiba.com → http://localhost:${MONITOR_PORT}"
