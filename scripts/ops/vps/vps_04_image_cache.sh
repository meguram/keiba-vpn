#!/usr/bin/env bash
# VPS の nginx に、画像(/data/image/)のキャッシュを入れる。root で実行。冪等・失敗したら元に戻す。
#   sudo bash scripts/ops/vps/vps_04_image_cache.sh [--dry-run|--remove]
# 仕組み: サイト設定のテンプレートが /data/image/ 専用の location を持ち、その中で keiba-img-cache*.conf を include する。
#   このスクリプトは (1) スニペットとキャッシュ置き場を置き、(2) テンプレートを /opt/keiba-ui/templates へ同期し、
#   (3) 本番(keiba-ui)を switch_api_upstream.sh で・stg(keiba-ui-stg)を同じ置換で再生成する。
#   上流の切替で再生成されても外れない（以前は再生成のたびに画像キャッシュが消えていた）。
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; T="$HERE/templates"
NGX="${NGX_ETC:-/etc/nginx}"
ZONES="$NGX/conf.d/keiba-ui-zones.conf"
SNIP="$NGX/snippets/keiba-img-cache.conf"
CACHE_DIR=/var/cache/nginx/keiba_img
OPT_TPL=/opt/keiba-ui/templates
MODE="${1:-apply}"
if [ "$MODE" = "--dry-run" ]; then
  echo "(dry-run) 追加: $CACHE_DIR / $SNIP / $ZONES の proxy_cache_path / $OPT_TPL へテンプレート同期 / 本番と stg のサイト再生成"; exit 0
fi
[ "$(id -u)" = 0 ] || { echo "root で実行してください" >&2; exit 2; }
[ -f "$ZONES" ] || { echo "ゾーン設定がありません: $ZONES" >&2; exit 1; }
BK="/var/backups/keiba-nginx/img-cache-$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BK" "$NGX/snippets"
for f in "$NGX/sites-available/keiba-ui" "$NGX/sites-available/keiba-ui-stg" "$ZONES"; do [ -f "$f" ] && cp -p "$f" "$BK/"; done
reload_or_fail() { if nginx -t; then systemctl reload nginx; else return 1; fi; }
restore() {
  for f in keiba-ui keiba-ui-stg; do [ -f "$BK/$f" ] && cp -p "$BK/$f" "$NGX/sites-available/$f"; done
  cp -p "$BK/$(basename "$ZONES")" "$ZONES"; rm -f "$SNIP"
  nginx -t >/dev/null 2>&1 && systemctl reload nginx || true; echo "元に戻しました: $BK" >&2
}
if [ "$MODE" = "--remove" ]; then
  rm -f "$SNIP"; reload_or_fail && echo "画像キャッシュを外しました（include は未配置なら何も読まない）"; exit 0
fi

mkdir -p "$CACHE_DIR"; chown -R www-data:www-data /var/cache/nginx 2>/dev/null || chown -R nginx:nginx /var/cache/nginx
install -m 644 "$T/nginx-img-cache.conf" "$SNIP"
grep -q 'proxy_cache_path /var/cache/nginx/keiba_img' "$ZONES" || \
  echo 'proxy_cache_path /var/cache/nginx/keiba_img levels=1:2 keys_zone=keiba_img:10m max_size=200m inactive=7d use_temp_path=off;' >> "$ZONES"
if [ -d "$OPT_TPL" ]; then install -m 644 "$T/nginx-keiba-ui.conf" "$T/nginx-keiba-ui-cloudrun.conf" "$OPT_TPL/"; fi

# 本番: 現在の上流の種類のまま、テンプレートから再生成（切替スクリプトが nginx -t・/api/health 確認・自動復旧をする）
if [ -x /opt/keiba-ui/bin/switch_api_upstream.sh ]; then
  CUR="$(cat /etc/keiba-ui/api_upstream 2>/dev/null || echo host)"
  /opt/keiba-ui/bin/switch_api_upstream.sh "$CUR" --force || { restore; exit 1; }
fi

# stg: vps_05 と同じ置換で再生成
STG="$NGX/sites-available/keiba-ui-stg"
if [ -f "$STG" ]; then
  CH="$(. /etc/keiba-ui/vars.env 2>/dev/null; echo "${CLOUDRUN_HOST:-}")"
  [ -n "$CH" ] || { echo "vars.env に CLOUDRUN_HOST がありません。stg は更新していません" >&2; restore; exit 1; }
  CR_HOST="rstg---${CH}"; CR_RE="$(printf '%s' "$CR_HOST" | sed 's/\./\\./g')"; CR_RE="${CR_RE//\\/\\\\}"
  NEW="$(mktemp)"; trap 'rm -f "$NEW"' EXIT
  sed -e "s/__CLOUDRUN_HOST_RE__/${CR_RE}/g" -e "s/__CLOUDRUN_HOST__/${CR_HOST}/g" -e "s/__PROXY_PORT__/8081/g" -e "s/__APP_PORT__/3002/g" "$T/nginx-keiba-ui-stg.conf" > "$NEW"
  install -m 644 "$NEW" "$STG"
fi
if reload_or_fail; then echo "画像キャッシュを有効にしました（バックアップ: $BK）"; else restore; exit 1; fi
