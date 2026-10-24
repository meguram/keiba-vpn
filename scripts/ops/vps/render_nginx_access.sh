#!/usr/bin/env bash
# /etc/keiba-ui/access.env（CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET。set_secret.sh で書く）から nginx の include を作って reload する。root で実行。値は表示しない。
set -euo pipefail
ENVF=/etc/keiba-ui/access.env; OUT=/etc/nginx/keiba-access.conf
[ -r "$ENVF" ] || { echo "$ENVF がありません（先に set_secret.sh で CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET を書いてください）" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENVF"
[ -n "${CF_ACCESS_CLIENT_ID:-}" ] && [ -n "${CF_ACCESS_CLIENT_SECRET:-}" ] || { echo "CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET が空です" >&2; exit 2; }
case "$CF_ACCESS_CLIENT_ID$CF_ACCESS_CLIENT_SECRET" in *\"*|*\;*|*\$*|*\\*) echo "値に使えない文字が含まれています" >&2; exit 2;; esac
umask 037
{
  echo "# 自動生成（render_nginx_access.sh）。手で編集しない。権限 640 root:www-data"
  printf 'proxy_set_header CF-Access-Client-Id "%s";\n' "$CF_ACCESS_CLIENT_ID"
  printf 'proxy_set_header CF-Access-Client-Secret "%s";\n' "$CF_ACCESS_CLIENT_SECRET"
} > "$OUT"
chown root:www-data "$OUT"; chmod 640 "$OUT"
nginx -t >/dev/null 2>&1 && systemctl reload nginx && echo "更新しました: $OUT（nginx を再読み込み）" || { echo "nginx -t が失敗しました" >&2; exit 1; }
