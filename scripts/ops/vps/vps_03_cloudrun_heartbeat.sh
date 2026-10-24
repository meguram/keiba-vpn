#!/usr/bin/env bash
# Cloud Run 化に必要な VPS 側の部品を入れる（冪等。root で実行。vps_02 の後）:
#   - API の上流切替（switch_api_upstream.sh）・共有ヘッダ鍵の nginx include 生成（render_nginx_proxykey.sh）
#   - ハートビート送信（heartbeat_client.py + systemd タイマー 5 分おき）。設定が揃うまでタイマーは有効にしない
# 秘密は入れない。手順の続き: docs/operations/cloud_run_api.md §4
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "root で実行してください" >&2; exit 2; }
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
install -d -m 755 /opt/keiba-ui/bin /opt/keiba-ui/templates /etc/keiba-ui /var/lib/keiba-ui
for f in switch_api_upstream.sh render_nginx_proxykey.sh heartbeat_client.py; do install -m 755 -o root -g root "$HERE/$f" "/opt/keiba-ui/bin/$f"; done
install -m 644 "$HERE/templates/nginx-keiba-ui.conf" "$HERE/templates/nginx-keiba-ui-cloudrun.conf" /opt/keiba-ui/templates/
install -m 644 "$HERE/templates/keiba-heartbeat.service" "$HERE/templates/keiba-heartbeat.timer" /etc/systemd/system/
[ -e /etc/keiba-ui/proxykey.env ] || { umask 077; : > /etc/keiba-ui/proxykey.env; }
[ -e /etc/keiba-ui/heartbeat.env ] || { umask 077; : > /etc/keiba-ui/heartbeat.env; }
chmod 600 /etc/keiba-ui/proxykey.env /etc/keiba-ui/heartbeat.env
systemctl daemon-reload
if grep -q '^HEARTBEAT_URL=.' /etc/keiba-ui/heartbeat.env && grep -q '^KEIBA_HEARTBEAT_SECRET=.' /etc/keiba-ui/heartbeat.env; then
  systemctl enable --now keiba-heartbeat.timer
  echo "ハートビートのタイマーを有効にしました。確認: python3 /opt/keiba-ui/bin/heartbeat_client.py --print / journalctl -u keiba-heartbeat -n 5"
else
  echo "ハートビートの設定が未完了です（タイマーは未有効）。次を実行してください:"
  echo "  printf '%s' 'https://<Cloud Run のホスト>/ops/heartbeat' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/heartbeat.env HEARTBEAT_URL"
  echo "  printf '%s' '<署名鍵>' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/heartbeat.env KEIBA_HEARTBEAT_SECRET"
  echo "  sudo systemctl enable --now keiba-heartbeat.timer"
fi
grep -q '^CLOUDRUN_HOST=' /etc/keiba-ui/vars.env 2>/dev/null || echo "注意: /etc/keiba-ui/vars.env に CLOUDRUN_HOST=<keiba-api-xxxx-an.a.run.app> を追記してください（https:// なし）"
echo "完了。API の上流切替: sudo /opt/keiba-ui/bin/switch_api_upstream.sh status | cloudrun | host"
