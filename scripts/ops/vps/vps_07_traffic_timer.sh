#!/usr/bin/env bash
# アクセス集計の送信（traffic_client.py + systemd タイマー 15 分おき）を入れる/外す(冪等。root で実行。vps_03 の後)。
#   sudo bash vps_07_traffic_timer.sh [--dry-run]     # 入れる
#   sudo bash vps_07_traffic_timer.sh --remove        # 外す
# 前提: /etc/keiba-ui/heartbeat.env に HEARTBEAT_URL(=監視ポータル keiba-monitor の /ops/heartbeat)と KEIBA_HEARTBEAT_SECRET が入っていること。
#       TRAFFIC_URL は未設定なら HEARTBEAT_URL の /ops/heartbeat を /ops/traffic に読み替える。
# 送り先は監視ポータル(keiba-monitor)。keiba-api には送らない(インスタンス課金で、15 分以内の ping は月 約 7,800 円。docs/operations/cost_recalc_20261028.md)。
# 何を読むか: /var/log/nginx/access.log*（読み取りだけ。root のサービスが読む。ログの権限は変えない）。生の IP・UA は VPS の外に出ない。
set -euo pipefail
DRY=0; REMOVE=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --remove) REMOVE=1;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done
[ "$(id -u)" = 0 ] || [ "$DRY" = 1 ] || { echo "root で実行してください" >&2; exit 2; }
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
run() { printf '+'; printf ' %q' "$@"; printf '\n'; [ "$DRY" = 1 ] || "$@"; }
if [ "$REMOVE" = 1 ]; then
  run systemctl disable --now keiba-traffic.timer || true
  run rm -f /etc/systemd/system/keiba-traffic.service /etc/systemd/system/keiba-traffic.timer /opt/keiba-ui/bin/traffic_client.py /opt/keiba-ui/bin/vps_traffic_probe.py
  run systemctl daemon-reload
  echo "外しました(蓄積した /var/lib/keiba-ui/traffic は残しています)"; exit 0
fi
[ -f /opt/keiba-ui/bin/heartbeat_client.py ] || { echo "heartbeat_client.py がありません(先に vps_03_cloudrun_heartbeat.sh)" >&2; exit 1; }
run install -d -m 755 /opt/keiba-ui/bin
run install -d -m 700 /var/lib/keiba-ui/traffic
for f in traffic_client.py vps_traffic_probe.py; do run install -m 755 -o root -g root "$HERE/$f" "/opt/keiba-ui/bin/$f"; done
run install -m 644 "$HERE/templates/keiba-traffic.service" "$HERE/templates/keiba-traffic.timer" /etc/systemd/system/
run systemctl daemon-reload
if [ "$DRY" = 1 ]; then echo "(dry-run) 変更は何もしていません"; exit 0; fi
if grep -q '^KEIBA_HEARTBEAT_SECRET=.' /etc/keiba-ui/heartbeat.env 2>/dev/null && grep -Eq '^(TRAFFIC_URL|HEARTBEAT_URL)=.' /etc/keiba-ui/heartbeat.env 2>/dev/null; then
  python3 /opt/keiba-ui/bin/traffic_client.py --print
  systemctl enable --now keiba-traffic.timer
  echo "有効にしました。確認: journalctl -u keiba-traffic -n 5 / systemctl list-timers keiba-traffic.timer"
else
  echo "heartbeat.env の設定が未完了です(タイマーは未有効)。HEARTBEAT_URL と KEIBA_HEARTBEAT_SECRET を入れてから: sudo systemctl enable --now keiba-traffic.timer"
fi
