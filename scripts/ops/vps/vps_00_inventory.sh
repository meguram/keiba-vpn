#!/usr/bin/env bash
# 【読み取り専用】VPS で今何が動いているかの棚卸し。旧サイト（ML-AutoPilot Keiba）の残骸を把握する。root で実行。
# 出力は画面と /root/vps_inventory_<日時>.txt。秘密（.env の中身など）は出さない。
set -u
OUT="/root/vps_inventory_$(date +%Y%m%d_%H%M%S).txt"
{
echo "== 基本"; hostnamectl 2>/dev/null | head -8; uptime; free -m; df -h / | tail -1
echo; echo "== 待ち受けポート（外から見えるもの）"; ss -ltnp 2>/dev/null | awk 'NR==1 || /0\.0\.0\.0|\[::\]|\*:/' | cut -c1-160
echo; echo "== 動いているサービス"; systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | awk '{print $1}'
echo; echo "== Web サーバ・証明書・コンテナ・プロセス管理"
for u in nginx apache2 httpd caddy certbot.timer docker pm2-root supervisor gunicorn uvicorn; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${u}"; then echo "$u: $(systemctl is-active "$u" 2>/dev/null) / enabled=$(systemctl is-enabled "$u" 2>/dev/null)"; fi
done
command -v docker >/dev/null && { echo "-- docker ps"; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Ports}}' 2>/dev/null; }
command -v pm2 >/dev/null && { echo "-- pm2"; pm2 ls 2>/dev/null | head -20; }
echo; echo "== cron"; crontab -l 2>/dev/null | grep -v '^#' | head -20; ls /etc/cron.d 2>/dev/null
echo; echo "== 大きいディレクトリ（上位）"; du -xh --max-depth=2 /var/www /srv /opt /home /root 2>/dev/null | sort -rh | head -15
echo; echo "== Let's Encrypt 証明書の期限"; ls /etc/letsencrypt/live 2>/dev/null; for c in /etc/letsencrypt/live/*/cert.pem; do [ -f "$c" ] && echo "$c: $(openssl x509 -enddate -noout -in "$c" 2>/dev/null)"; done
echo; echo "== OS"; . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}"; echo "SSH: $(sshd -T 2>/dev/null | grep -E '^(passwordauthentication|permitrootlogin|port) ' | tr '\n' ' ')"
echo; echo "== ファイアウォール"; ufw status 2>/dev/null | head -10; iptables -S 2>/dev/null | head -10
} 2>&1 | tee "$OUT"
echo; echo "保存しました: $OUT（旧サイトのデータが要るなら ConoHa のコンソールで『イメージ保存』を先に）"
