#!/usr/bin/env bash
# VPS の nginx が /api/・/data/image/ を取りにいく先を切り替える。root で実行（/opt/keiba-ui/bin/ に入る）。
#   switch_api_upstream.sh status             # 今どちらか
#   switch_api_upstream.sh cloudrun [--force] # Cloud Run へ（事前: vars.env の CLOUDRUN_HOST、proxykey.env、render_nginx_proxykey.sh）
#   switch_api_upstream.sh host               # ホスト機（api-internal）へ戻す（ロールバック）
# 切替後に http://127.0.0.1:PROXY_PORT/api/health が 200 でなければ、自動で直前の設定に戻して終了コード 1。nginx -t が通らなければ置かない。
set -euo pipefail
MODE="${1:-status}"; FORCE="${2:-}"
. /etc/keiba-ui/vars.env
PROXY_PORT="${PROXY_PORT:-8080}"; APP_PORT="${APP_PORT:-3000}"
TPL=/opt/keiba-ui/templates; SITE=/etc/nginx/sites-available/keiba-ui
STATE=/etc/keiba-ui/api_upstream; LOG=/var/log/keiba-ui-switch.log
cur() { cat "$STATE" 2>/dev/null || echo host; }
log() { echo "$(date -Is) $*" | tee -a "$LOG" >/dev/null; }

if [ "$MODE" = status ]; then echo "$(cur)"; exit 0; fi

render() {  # $1=host|cloudrun → 標準出力
  case "$1" in
    host)
      local re; re="$(printf '%s' "$UPSTREAM_HOST" | sed 's/\./\\./g')"; re="${re//\\/\\\\}"
      sed -e "s/__UPSTREAM_HOST_RE__/${re}/g" -e "s/__UPSTREAM_HOST__/${UPSTREAM_HOST}/g" -e "s/__PROXY_PORT__/${PROXY_PORT}/g" -e "s/__APP_PORT__/${APP_PORT}/g" "$TPL/nginx-keiba-ui.conf" ;;
    cloudrun)
      [ -n "${CLOUDRUN_HOST:-}" ] || { echo "vars.env に CLOUDRUN_HOST がありません（例: keiba-api-xxxx-an.a.run.app。https:// は付けない）" >&2; return 2; }
      local re; re="$(printf '%s' "$CLOUDRUN_HOST" | sed 's/\./\\./g')"; re="${re//\\/\\\\}"
      sed -e "s/__CLOUDRUN_HOST_RE__/${re}/g" -e "s/__CLOUDRUN_HOST__/${CLOUDRUN_HOST}/g" -e "s/__PROXY_PORT__/${PROXY_PORT}/g" -e "s/__APP_PORT__/${APP_PORT}/g" "$TPL/nginx-keiba-ui-cloudrun.conf" ;;
    *) echo "host | cloudrun | status" >&2; return 2 ;;
  esac
}

[ "$MODE" = host ] || [ "$MODE" = cloudrun ] || { echo "host | cloudrun | status" >&2; exit 2; }
if [ "$MODE" = cloudrun ]; then
  [ -s /etc/nginx/keiba-proxykey.conf ] && grep -q 'X-Keiba-Proxy-Key' /etc/nginx/keiba-proxykey.conf || { echo "/etc/nginx/keiba-proxykey.conf がありません（先に render_nginx_proxykey.sh）" >&2; exit 2; }
  # 切替前に Cloud Run が応答するか（鍵なしの /api/health は除外パス。200 でなければ切り替えない）
  if [ "$FORCE" != "--force" ]; then
    c="$(curl -s -o /dev/null -w '%{http_code}' -m 15 "https://${CLOUDRUN_HOST:-invalid}/api/health" || true)"
    [ "$c" = 200 ] || { echo "Cloud Run の /api/health が $c です。切り替えません（--force で強制）" >&2; exit 1; }
  fi
fi

NEW="$(mktemp)"; trap 'rm -f "$NEW"' EXIT
render "$MODE" > "$NEW"
BAK="/etc/nginx/keiba-ui.bak.$(date +%Y%m%d%H%M%S)"; cp -a "$SITE" "$BAK"; PREV="$(cur)"
install -m 644 "$NEW" "$SITE"
revert() { cp -a "$BAK" "$SITE"; nginx -t >/dev/null 2>&1 && systemctl reload nginx || true; log "revert to $PREV ($1)"; echo "直前の設定に戻しました: $1" >&2; exit 1; }
nginx -t >/dev/null 2>&1 || revert "nginx -t 失敗"
systemctl reload nginx
sleep 2
c="$(curl -s -o /dev/null -w '%{http_code}' -m 20 "http://127.0.0.1:${PROXY_PORT}/api/health" || true)"
[ "$c" = 200 ] || revert "切替後の /api/health が $c"
echo "$MODE" > "$STATE"; log "switch $PREV -> $MODE (backup $BAK)"
echo "OK: API の上流を $MODE にしました（/api/health 200）。戻すとき: switch_api_upstream.sh $PREV"
