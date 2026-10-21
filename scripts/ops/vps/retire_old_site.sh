#!/usr/bin/env bash
# 旧サイトの常駐を止める（削除はしない）。root で実行。先に vps_00_inventory.sh と ConoHa の『イメージ保存』を済ませること。
#   bash retire_old_site.sh             → 止める候補を表示するだけ
#   bash retire_old_site.sh --apply --i-have-a-backup   → 候補を stop + disable（データ・設定ファイルは残す）
set -u
APPLY=0; BACKUP=0
for a in "$@"; do case "$a" in --apply) APPLY=1;; --i-have-a-backup) BACKUP=1;; esac; done
# nginx は止めない: vps_02 のあとは nginx(127.0.0.1:8080) が新しい UI の入口になっている。止めると prod が落ちる
CANDS=(keiba-api uwsgi apache2 httpd caddy certbot.timer certbot docker pm2-root supervisor gunicorn uvicorn)
echo "止める候補（存在するもの）:"
FOUND=()
for u in "${CANDS[@]}"; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${u}[.]\|^${u}\$"; then FOUND+=("$u"); echo "  - $u ($(systemctl is-active "$u" 2>/dev/null))"; fi
done
[ ${#FOUND[@]} -gt 0 ] || { echo "  （なし）"; exit 0; }
echo "注意: nginx は新構成（vps_02）で使うので、旧設定（/etc/nginx/conf.d/myapp_nginx.conf など）は vps_02 が退避してから入れ直します。ここで止めても問題ありません。"
echo "旧サイトの実体（この VPS の実測）: keiba-api.service(uvicorn :8000) / uwsgi.service(megu-ai) / nginx(80,443) / certbot.timer。データは /var/www/megu-keiba と /home/<ユーザー> に残す（削除しない）"
if [ "$APPLY" != 1 ]; then echo "→ 実行するなら --apply --i-have-a-backup"; exit 0; fi
[ "$BACKUP" = 1 ] || { echo "バックアップ確認の --i-have-a-backup が必要です"; exit 3; }
for u in "${FOUND[@]}"; do systemctl stop "$u" 2>/dev/null; systemctl disable "$u" 2>/dev/null; echo "停止: $u"; done
mkdir -p /root/old_site_configs
[ -d /etc/nginx ] && cp -a /etc/nginx /root/old_site_configs/nginx_$(date +%Y%m%d) 2>/dev/null
[ -d /etc/apache2 ] && cp -a /etc/apache2 /root/old_site_configs/apache2_$(date +%Y%m%d) 2>/dev/null
echo "旧設定を /root/old_site_configs に退避しました。データ（/var/www など）は残してあります。"
