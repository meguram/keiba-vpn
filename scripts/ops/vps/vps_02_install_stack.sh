#!/usr/bin/env bash
# Node.js・nginx・cloudflared・systemd・sudoers を入れる。root で実行。vps_01 の後。
#   bash vps_02_install_stack.sh [--vars /etc/keiba-ui/vars.env] [--node-major 22]
# 秘密（トンネルのトークン、Access のサービストークン）はここでは扱わない（set_secret.sh で別に渡す）。
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "root で実行してください" >&2; exit 1; }
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VARS=/etc/keiba-ui/vars.env; NODE_MAJOR=22
while [ $# -gt 0 ]; do case "$1" in --vars) VARS="$2"; shift 2;; --node-major) NODE_MAJOR="$2"; shift 2;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -r "$VARS" ] || { [ -r "$HERE/vars.env" ] && install -D -m 644 "$HERE/vars.env" /etc/keiba-ui/vars.env && VARS=/etc/keiba-ui/vars.env; }
[ -r "$VARS" ] || { echo "vars.env がありません。vars.env.example をコピーして編集し、/etc/keiba-ui/vars.env に置いてください" >&2; exit 2; }
# shellcheck disable=SC1090
. "$VARS"
: "${UPSTREAM_HOST:?}" "${APP_ROOT:=/srv/keiba-ui}" "${APP_PORT:=3000}" "${PROXY_PORT:=8080}"
export DEBIAN_FRONTEND=noninteractive
[ "$(uname -m)" = x86_64 ] || { echo "x86_64 以外は未対応（Node の取得先を変更してください）" >&2; exit 1; }

echo "== 1/7 Node.js ${NODE_MAJOR}.x（公式バイナリ + SHA256 検証）"
if ! command -v node >/dev/null || ! node -v | grep -q "^v${NODE_MAJOR}\."; then
  BASE="https://nodejs.org/dist/latest-v${NODE_MAJOR}.x"
  curl -fsSL "$BASE/SHASUMS256.txt" -o /tmp/node.sums
  FN="$(awk '/linux-x64\.tar\.xz$/ {print $2; exit}' /tmp/node.sums)"; [ -n "$FN" ] || { echo "Node の配布ファイル名を取得できません" >&2; exit 1; }
  curl -fsSL "$BASE/$FN" -o "/tmp/$FN"
  ( cd /tmp && grep " $FN\$" node.sums | sha256sum -c - )
  VER="${FN#node-}"; VER="${VER%-linux-x64.tar.xz}"
  install -d /opt/node; tar -xJf "/tmp/$FN" -C /opt/node
  ln -sfn "/opt/node/node-${VER}-linux-x64" /opt/node/current
  ln -sfn /opt/node/current/bin/node /usr/local/bin/node
  rm -f "/tmp/$FN" /tmp/node.sums
fi
node -v

echo "== 2/7 cloudflared（Cloudflare 公式 apt リポジトリ）"
if ! command -v cloudflared >/dev/null; then
  install -d -m 0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" > /etc/apt/sources.list.d/cloudflared.list
  apt-get update -y && apt-get install -y cloudflared
fi
cloudflared --version

echo "== 3/7 管理用スクリプト（root 所有。deploy は書き換えられない）"
install -d -m 755 /opt/keiba-ui/bin
for f in install_release.sh rollback.sh healthcheck.sh render_nginx_access.sh set_secret.sh; do install -m 755 -o root -g root "$HERE/$f" "/opt/keiba-ui/bin/$f"; done
install -m 644 "$VARS" /etc/keiba-ui/vars.env 2>/dev/null || true
install -d -m 755 /etc/cloudflared; touch /etc/cloudflared/prod.env; chmod 600 /etc/cloudflared/prod.env

echo "== 4/7 UI の環境ファイル（秘密なし）"
cat > /etc/keiba-ui/ui.env <<UIENV
NODE_ENV=production
PORT=${APP_PORT}
HOSTNAME=127.0.0.1
# ビルド時にも同じ値が埋め込まれる（next.config.js）。ブラウザの API 呼び出しは同一オリジンで nginx が上流へ中継する
KEIBA_API_URL=http://127.0.0.1:${PROXY_PORT}
KEIBA_FASTAPI_URL=http://127.0.0.1:${PROXY_PORT}
NEXT_TELEMETRY_DISABLED=1
NODE_OPTIONS=--max-old-space-size=384
UIENV
chmod 644 /etc/keiba-ui/ui.env

echo "== 5/7 nginx（127.0.0.1:${PROXY_PORT} のみ。旧設定は退避）"
[ -d /etc/nginx/sites-enabled ] && { install -d /root/old_site_configs; cp -a /etc/nginx/sites-enabled /root/old_site_configs/nginx_sites_enabled_$(date +%Y%m%d%H%M) 2>/dev/null || true; }
rm -f /etc/nginx/sites-enabled/default
# 旧サイトの設定（conf.d/myapp_nginx.conf など。80/443 を握り、期限切れ証明書を使っている）を退避。新しい keiba-ui-zones.conf 以外を動かさない
install -d /root/old_site_configs/conf.d
for f in /etc/nginx/conf.d/*.conf; do
  [ -e "$f" ] || continue
  [ "$(basename "$f")" = keiba-ui-zones.conf ] && continue
  mv -v "$f" "/root/old_site_configs/conf.d/$(basename "$f").$(date +%Y%m%d%H%M)"
done
UP_RE="$(printf '%s' "$UPSTREAM_HOST" | sed 's/\./\\./g')"
UP_RE_SED="${UP_RE//\\/\\\\}"      # sed の置換側で \ が消えないように二重にする
install -m 644 "$HERE/templates/nginx-zones.conf" /etc/nginx/conf.d/keiba-ui-zones.conf
sed -e "s/__UPSTREAM_HOST_RE__/${UP_RE_SED}/g" -e "s/__UPSTREAM_HOST__/${UPSTREAM_HOST}/g" -e "s/__PROXY_PORT__/${PROXY_PORT}/g" -e "s/__APP_PORT__/${APP_PORT}/g" "$HERE/templates/nginx-keiba-ui.conf" > /etc/nginx/sites-available/keiba-ui
ln -sfn /etc/nginx/sites-available/keiba-ui /etc/nginx/sites-enabled/keiba-ui
[ -f /etc/nginx/keiba-access.conf ] || { printf '# render_nginx_access.sh が作る（未設定）\n' > /etc/nginx/keiba-access.conf; chown root:www-data /etc/nginx/keiba-access.conf; chmod 640 /etc/nginx/keiba-access.conf; }
nginx -t && systemctl enable --now nginx && systemctl reload nginx

echo "== 6/7 systemd・logrotate・sudoers"
install -m 644 "$HERE/templates/keiba-ui.service" /etc/systemd/system/keiba-ui.service
install -m 644 "$HERE/templates/cloudflared-prod.service" /etc/systemd/system/cloudflared-prod.service
install -m 644 "$HERE/templates/keiba-ui-health.service" /etc/systemd/system/keiba-ui-health.service
install -m 644 "$HERE/templates/keiba-ui-health.timer" /etc/systemd/system/keiba-ui-health.timer
install -m 644 "$HERE/templates/logrotate-keiba-ui" /etc/logrotate.d/keiba-ui
install -m 440 "$HERE/templates/sudoers-keiba-deploy" /etc/sudoers.d/keiba-deploy
visudo -cf /etc/sudoers.d/keiba-deploy >/dev/null || { rm -f /etc/sudoers.d/keiba-deploy; echo "sudoers が不正のため削除しました" >&2; exit 1; }
systemctl daemon-reload
systemctl enable keiba-ui.service keiba-ui-health.timer cloudflared-prod.service >/dev/null 2>&1
systemctl start keiba-ui-health.timer

echo "== 7/7 次の手順"
cat <<NEXT
  1) トンネルのトークンを渡す（Cloudflare の『meguai-prod』トンネルの画面のコマンドを、チャットに貼らずに）:
       printf '%s' '<画面のコマンド全体 or eyJ…>' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/cloudflared/prod.env TUNNEL_TOKEN
       sudo systemctl restart cloudflared-prod && systemctl status cloudflared-prod --no-pager | head -5
  2) Access のサービストークン（上流 ${UPSTREAM_HOST} 用）を渡す:
       printf '%s' '<Client ID>'     | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/access.env CF_ACCESS_CLIENT_ID
       printf '%s' '<Client Secret>' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/access.env CF_ACCESS_CLIENT_SECRET
       sudo /opt/keiba-ui/bin/render_nginx_access.sh
  3) 最初のリリースを配布（ホスト側から promote.sh）。
NEXT
