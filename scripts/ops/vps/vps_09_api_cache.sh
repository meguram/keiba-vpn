#!/usr/bin/env bash
# VPS の nginx に、API 応答のキャッシュ（匿名・許可リストのみ）を入れる。root で実行。冪等・失敗したら元に戻す。
#   sudo bash scripts/ops/vps/vps_09_api_cache.sh [--dry-run|--remove]
# 対象: keiba-ui（本番）と keiba-ui-stg。サイト設定の /api/ の location に include を1行足すだけ（上流の認証ヘッダには触れない）。
# 上流の切替（switch_api_upstream.sh）はテンプレートから再生成するが、テンプレート側にも同じ include があるので外れない。
# 仕組みと安全性は templates/nginx-api-cache.conf の冒頭を参照。検証は python3 scripts/ops/vps/verify_api_cache.py。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NGX="${NGX_ETC:-/etc/nginx}"
ZONES="$NGX/conf.d/keiba-ui-zones.conf"
SNIP="$NGX/snippets/keiba-api-cache.conf"
CACHE_DIR="${API_CACHE_DIR:-/var/cache/nginx/keiba_api}"
SITES="${SITES:-$NGX/sites-available/keiba-ui $NGX/sites-available/keiba-ui-stg}"
INC='        include /etc/nginx/snippets/keiba-api-cache*.conf;'
MODE="${1:-apply}"
TARGETS=""; for s in $SITES; do [ -f "$s" ] && TARGETS="$TARGETS $s"; done
[ -n "$TARGETS" ] || { echo "サイト設定がありません: $SITES" >&2; exit 1; }
[ -f "$ZONES" ] || { echo "ゾーン設定がありません: $ZONES" >&2; exit 1; }
if [ "$MODE" = "--dry-run" ]; then
  echo "(dry-run) 追加: $CACHE_DIR / $SNIP / $ZONES の proxy_cache_path+map / include を次のサイトへ:$TARGETS"; exit 0
fi
BK="${BK_ROOT:-/var/backups/keiba-nginx}/api-cache-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK" "$NGX/snippets"
for f in $TARGETS "$ZONES"; do cp -p "$f" "$BK/$(basename "$f")"; done
HAD_SNIP=0; [ -f "$SNIP" ] && { cp -p "$SNIP" "$BK/snippet.bak"; HAD_SNIP=1; }
restore() {
  for f in $TARGETS "$ZONES"; do cp -p "$BK/$(basename "$f")" "$f"; done
  if [ "$HAD_SNIP" = 1 ]; then cp -p "$BK/snippet.bak" "$SNIP"; else rm -f "$SNIP"; fi
  nginx -t >/dev/null 2>&1 && systemctl reload nginx || true
  echo "元に戻しました: $BK" >&2
}
if [ "$MODE" = "--remove" ]; then
  for s in $TARGETS; do sed -i '\#keiba-api-cache#d' "$s"; done
  sed -i -e '/^# API 応答のキャッシュ置き場/d' -e '/proxy_cache_path .*keiba_apicache/d' -e '/^map "\$http_cookie\$http_authorization"/,/^}/d' "$ZONES"
  rm -f "$SNIP"
  if nginx -t; then systemctl reload nginx; echo "API キャッシュを外しました（バックアップ: $BK）"; else restore; exit 1; fi
  exit 0
fi
[ -f "$HERE/templates/nginx-api-cache.conf" ] || { echo "templates/nginx-api-cache.conf がありません" >&2; exit 1; }
mkdir -p "$CACHE_DIR"; chown -R www-data:www-data "$(dirname "$CACHE_DIR")" 2>/dev/null || chown -R nginx:nginx "$(dirname "$CACHE_DIR")" 2>/dev/null || true
install -m 644 "$HERE/templates/nginx-api-cache.conf" "$SNIP"
# 上流の切替（switch_api_upstream.sh）はここのテンプレートから再生成するので、最新の include 入りを置く
if [ -d /opt/keiba-ui/templates ]; then install -m 644 "$HERE/templates/nginx-keiba-ui.conf" "$HERE/templates/nginx-keiba-ui-cloudrun.conf" /opt/keiba-ui/templates/; fi
if ! grep -q 'keiba_apicache' "$ZONES"; then
  sed -n '/^# API 応答のキャッシュ置き場/,$p' "$HERE/templates/nginx-zones.conf" | sed "s#/var/cache/nginx/keiba_api#$CACHE_DIR#" >> "$ZONES"
fi
# 既存のゾーンの保持設定（確定済みを 1 週間置くため max_size/inactive）を最新にする
sed -i -E 's#(keys_zone=keiba_apicache:10m) max_size=[0-9]+[kmg] inactive=[0-9]+[smhd]#\1 max_size=200m inactive=8d#' "$ZONES"
for s in $TARGETS; do
  if ! grep -q 'keiba-api-cache' "$s"; then
    # /api/ の location の proxy_buffering の直後に入れる（上流の認証 include より後ろ・location の中）
    awk -v inc="$INC" '/location ~ \^\/\(api\// {inapi=1} {print} inapi && /proxy_buffering on;/ {print inc; inapi=0}' "$s" > "$s.new" && mv "$s.new" "$s"
    grep -q 'keiba-api-cache' "$s" || { echo "include を入れる位置が見つかりません: $s" >&2; restore; exit 1; }
  fi
done
if nginx -t; then systemctl reload nginx; echo "API キャッシュを有効にしました（バックアップ: $BK）"; else restore; exit 1; fi
