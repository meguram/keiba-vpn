#!/usr/bin/env bash
# VPS の死活・資源の確認（systemd timer が 5 分おき）。サービスが落ちていれば 1 回だけ再起動し、Slack に知らせる（/etc/keiba-ui/slack.env があれば）。
# 異常の通知は同じ内容を 1 時間に 1 回まで。状態は /var/lib/keiba-ui/health.json。
set -u
. /etc/keiba-ui/vars.env 2>/dev/null || true
APP_PORT="${APP_PORT:-3000}"; PROXY_PORT="${PROXY_PORT:-8080}"
STATE=/var/lib/keiba-ui; mkdir -p "$STATE"
issues=()
for u in keiba-ui nginx cloudflared-prod; do
  if ! systemctl is-active --quiet "$u"; then issues+=("$u が停止"); systemctl restart "$u" 2>/dev/null; sleep 3; systemctl is-active --quiet "$u" && issues+=("$u を再起動して復旧"); fi
done
c="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$APP_PORT/version.json" || true)"; [ "$c" = 200 ] || issues+=("UI が応答しない($c)")
# nginx 経由でも /api/ は叩かない: /api/ は Cloud Run(keiba-api)に転送され、API は『CPU を常に割り当て』(インスタンス課金)なので、
# 15 分以内の定期リクエストがあると落ちずに月 約 7,800 円になる（docs/operations/cost_recalc_20261028.md）。API の死活は監視ポータル側で、人が開いたときに見る。
c="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PROXY_PORT/version.json" || true)"; [ "$c" = 200 ] || issues+=("nginx 経由の画面が応答しない($c)")
disk="$(df --output=pcent / | tail -1 | tr -dc 0-9)"; [ "${disk:-0}" -lt 85 ] || issues+=("ディスク ${disk}%")
mem="$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"; [ "${mem:-999}" -gt 150 ] || issues+=("空きメモリ ${mem}MB")
ok=true; [ ${#issues[@]} -eq 0 ] || ok=false
printf '{"at":"%s","ok":%s,"issues":"%s","disk_pct":%s,"mem_avail_mb":%s}\n' "$(date -Is)" "$ok" "$(IFS=';'; echo "${issues[*]:-}")" "${disk:-0}" "${mem:-0}" > "$STATE/health.json"
if [ "$ok" = false ] && [ -r /etc/keiba-ui/slack.env ]; then
  # shellcheck disable=SC1091
  . /etc/keiba-ui/slack.env
  key="$(printf '%s' "${issues[*]}" | md5sum | cut -c1-8)"; last="$(cat "$STATE/notified.$key" 2>/dev/null || echo 0)"; now="$(date +%s)"
  if [ -n "${SLACK_WEBHOOK_URL:-}" ] && [ $((now-last)) -gt 3600 ]; then
    msg="[prod UI] $(hostname): $(IFS=';'; echo "${issues[*]}")"
    curl -s -m 10 -X POST -H 'Content-type: application/json' --data "$(printf '{"text":"%s"}' "$msg")" "$SLACK_WEBHOOK_URL" >/dev/null || true
    echo "$now" > "$STATE/notified.$key"
  fi
fi
[ "$ok" = true ]
