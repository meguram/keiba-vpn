#!/usr/bin/env bash
# VPS の nginx に、画像(/data/image/)のキャッシュを入れる。root で実行。冪等・失敗したら元に戻す。
#   sudo bash scripts/ops/vps/vps_04_image_cache.sh [--dry-run|--remove]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="${SITE:-/etc/nginx/sites-available/keiba-ui}"
ZONES=/etc/nginx/conf.d/keiba-ui-zones.conf
SNIP=/etc/nginx/snippets/keiba-img-cache.conf
CACHE_DIR=/var/cache/nginx/keiba_img
INC='    include /etc/nginx/snippets/keiba-img-cache.conf;'
MODE="${1:-apply}"
[ -f "$SITE" ] || { echo "サイト設定がありません: $SITE" >&2; exit 1; }
BK="/var/backups/keiba-nginx/$(date +%Y%m%d_%H%M%S)"
if [ "$MODE" = "--dry-run" ]; then
  echo "(dry-run) 追加: $CACHE_DIR / $SNIP / $ZONES の proxy_cache_path / $SITE の include"; exit 0
fi
mkdir -p "$BK" /etc/nginx/snippets; cp -p "$SITE" "$ZONES" "$BK/" 2>/dev/null || true
restore() { cp -p "$BK/$(basename "$SITE")" "$SITE"; cp -p "$BK/$(basename "$ZONES")" "$ZONES"; rm -f "$SNIP"; nginx -t >/dev/null 2>&1 && systemctl reload nginx || true; echo "元に戻しました: $BK" >&2; }
if [ "$MODE" = "--remove" ]; then
  sed -i '\#keiba-img-cache.conf#d' "$SITE"; sed -i '\#proxy_cache_path /var/cache/nginx/keiba_img#d' "$ZONES"; rm -f "$SNIP"
  nginx -t && systemctl reload nginx; echo "画像キャッシュを外しました"; exit 0
fi
mkdir -p "$CACHE_DIR"; chown -R www-data:www-data /var/cache/nginx 2>/dev/null || chown -R nginx:nginx /var/cache/nginx
install -m 644 "$HERE/templates/nginx-img-cache.conf" "$SNIP"
grep -q 'proxy_cache_path /var/cache/nginx/keiba_img' "$ZONES" || \
  echo 'proxy_cache_path /var/cache/nginx/keiba_img levels=1:2 keys_zone=keiba_img:10m max_size=200m inactive=7d use_temp_path=off;' >> "$ZONES"
if ! grep -q 'keiba-img-cache.conf' "$SITE"; then
  # API の location の直前に include を入れる（正規表現 location は先に書いたものが優先）
  awk -v inc="$INC" '/location ~ \^\/\(api\/\|data\/image\/\)/ && !done {print inc; done=1} {print}' "$SITE" > "$SITE.new" && mv "$SITE.new" "$SITE"
fi
if nginx -t; then systemctl reload nginx; echo "画像キャッシュを有効にしました（バックアップ: $BK）"; else restore; exit 1; fi
