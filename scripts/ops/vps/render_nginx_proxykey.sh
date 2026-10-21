#!/usr/bin/env bash
# /etc/keiba-ui/proxykey.env（KEIBA_PROXY_KEY。set_secret.sh で書く）から nginx の include を作って reload する。root で実行。値は表示しない。
# 鍵のローテーション: 新しい鍵を Cloud Run 側の KEIBA_PROXY_KEY_NEXT に入れて動かす → 本スクリプトで VPS の鍵を新しい値に → Cloud Run の古い鍵を外す。
set -euo pipefail
ENVF=/etc/keiba-ui/proxykey.env; OUT=/etc/nginx/keiba-proxykey.conf
[ -r "$ENVF" ] || { echo "$ENVF がありません（先に: printf '%s' '<鍵>' | sudo bash set_secret.sh $ENVF KEIBA_PROXY_KEY）" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENVF"
[ -n "${KEIBA_PROXY_KEY:-}" ] || { echo "KEIBA_PROXY_KEY が空です" >&2; exit 2; }
case "$KEIBA_PROXY_KEY" in *\"*|*\;*|*\$*|*\\*|*" "*) echo "値に使えない文字が含まれています" >&2; exit 2;; esac
umask 037
{ echo "# 自動生成（render_nginx_proxykey.sh）。手で編集しない。権限 640 root:www-data"
  printf 'proxy_set_header X-Keiba-Proxy-Key "%s";\n' "$KEIBA_PROXY_KEY"; } > "$OUT"
chown root:www-data "$OUT"; chmod 640 "$OUT"
nginx -t >/dev/null 2>&1 && systemctl reload nginx && echo "更新しました: $OUT（長さ ${#KEIBA_PROXY_KEY}。nginx を再読み込み）" || { echo "nginx -t が失敗しました" >&2; exit 1; }
