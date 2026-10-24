#!/usr/bin/env bash
# stg 環境の画面を、VPS に 2 つ目の画面として入れる。root で実行。何度実行しても同じ結果（冪等）。
#   bash vps_05_stg_instance.sh [--dry-run] [--cloudrun-host <rstg---keiba-api-….run.app>] [--pull-now]
#
# できあがるもの（本番の画面 keiba-ui とは、プロセス・ポート・置き場・設定がすべて別）:
#   画面     keiba-ui-stg        Next.js 127.0.0.1:3002     /srv/keiba-ui-stg     メモリ上限 160M（本番の画面は 200M）
#   nginx    keiba-ui-stg        127.0.0.1:8081             ページ→3002、/api→Cloud Run の stg タグ付きリビジョン（rstg）
#   取り込み keiba-release-pull@ui-stg.timer  5 分おき。stg ブランチの push で GCS に公開された画面を、窓を待たずに入れる
# 公開側: stg ブランチへの push（.github/workflows/deploy-stg.yml）が ops/releases/ui-stg/ に画面を置く。
#
# 前提（無ければ止まる）:
#   - vps_01〜03 が済んでいる。vps_03_units.sh --with-pull-venv で pull 用の venv があり、/etc/keiba-pull/pull.env に GCS の読み取り資格情報が入っている
#   - /etc/nginx/keiba-proxykey.conf がある（render_nginx_proxykey.sh。X-Keiba-Proxy-Key。本番と同じ Cloud Run サービスなので同じ鍵）
#   - Cloud Run の stg タグ URL のホスト名: --cloudrun-host、または /etc/keiba-ui/vars.env の CLOUDRUN_HOST に rstg--- を付けたもの
# この後、Cloudflare で stg.megu-keiba.com の転送先を『この VPS のトンネル → http://localhost:8081』に付け替える（docs/operations/github_promote.md）。
# 本番の画面・nginx の本番の設定には触れない。--dry-run は何も変更しない。
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; T="$HERE/templates"
DRY=0; CR_HOST=""; PULL_NOW=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1; shift;; --cloudrun-host) CR_HOST="${2:?ホスト名}"; shift 2;; --pull-now) PULL_NOW=1; shift;;
  *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ "$DRY" = 1 ] || [ "$(id -u)" = 0 ] || { echo "root で実行してください（確認だけなら --dry-run）" >&2; exit 1; }

APP_ROOT=/srv/keiba-ui-stg; APP_PORT=3002; PROXY_PORT=8081
if [ -z "$CR_HOST" ] && [ -r /etc/keiba-ui/vars.env ]; then
  # shellcheck disable=SC1091
  CH="$(. /etc/keiba-ui/vars.env 2>/dev/null; echo "${CLOUDRUN_HOST:-}")"; [ -z "$CH" ] || CR_HOST="rstg---${CH}"
fi
if [ -z "$CR_HOST" ]; then
  [ "$DRY" = 1 ] && CR_HOST="rstg---keiba-api-XXXX.asia-northeast1.run.app" || { echo "--cloudrun-host が要ります（例: rstg---keiba-api-244891111662.asia-northeast1.run.app）" >&2; exit 2; }
fi
case "$CR_HOST" in *[!A-Za-z0-9.-]*|"") echo "ホスト名が不正です: $CR_HOST" >&2; exit 2;; esac
CR_RE="$(printf '%s' "$CR_HOST" | sed 's/\./\\./g')"; CR_RE="${CR_RE//\\/\\\\}"

CHANGED=0
say() { echo "$*"; }
run() { if [ "$DRY" = 1 ]; then echo "  (dry-run) $*"; else "$@"; fi; }
install_if_changed() {  # <mode> <src> <dst>: 内容が違うときだけ置く
  local mode="$1" src="$2" dst="$3"
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then return 0; fi
  say "  更新: $dst"; CHANGED=1; run install -D -m "$mode" -o root -g root "$src" "$dst"
}
write_if_missing() {  # <mode> <dst> <内容(stdin)>
  local mode="$1" dst="$2" body; body="$(cat)"
  [ -e "$dst" ] && return 0
  say "  新規: $dst"
  if [ "$DRY" != 1 ]; then install -d "$(dirname "$dst")"; printf '%s\n' "$body" > "$dst"; chmod "$mode" "$dst"; fi
}

say "== 0/6 前提の確認"
if [ "$DRY" != 1 ]; then
  [ -x /opt/keiba-pull/venv/bin/python ] || { echo "pull 用の venv がありません（先に vps_03_units.sh --with-pull-venv）" >&2; exit 3; }
  [ -s /etc/keiba-pull/pull.env ] || { echo "/etc/keiba-pull/pull.env がありません（GCS の読み取り資格情報。set_secret.sh で入れる）" >&2; exit 3; }
  [ -s /etc/nginx/keiba-proxykey.conf ] || { echo "/etc/nginx/keiba-proxykey.conf がありません（先に render_nginx_proxykey.sh）" >&2; exit 3; }
  id keibaui >/dev/null 2>&1 && id deploy >/dev/null 2>&1 || { echo "ユーザー keibaui / deploy がありません（vps_01）" >&2; exit 3; }
else say "  (dry-run) venv・pull.env・keiba-proxykey.conf・ユーザーの有無は実行時に確認します"; fi

say "== 1/6 管理用スクリプト（導入に INSTANCE=stg、取り込みに ui-stg を足した版。本番の既定は変わらない）"
install_if_changed 755 "$HERE/install_release.sh" /opt/keiba-ui/bin/install_release.sh
install_if_changed 755 "$HERE/pull_release.py" /opt/keiba-ui/bin/pull_release.py

say "== 2/6 ディレクトリ"
for d in "$APP_ROOT" "$APP_ROOT/releases" "$APP_ROOT/incoming" "$APP_ROOT/shared"; do run install -d -m 755 -o deploy -g keibaui "$d"; done

say "== 3/6 環境ファイル（秘密なし）"
write_if_missing 644 /etc/keiba-ui/vars-stg.env <<EOF
# stg 環境の画面（INSTANCE=stg）。install_release.sh が読む
APP_ROOT=${APP_ROOT}
APP_PORT=${APP_PORT}
PROXY_PORT=${PROXY_PORT}
EOF
write_if_missing 644 /etc/keiba-ui/ui-stg.env <<EOF
NODE_ENV=production
PORT=${APP_PORT}
HOSTNAME=127.0.0.1
KEIBA_API_URL=http://127.0.0.1:${PROXY_PORT}
KEIBA_FASTAPI_URL=http://127.0.0.1:${PROXY_PORT}
NEXT_TELEMETRY_DISABLED=1
NODE_OPTIONS=--max-old-space-size=192
EOF

say "== 4/6 systemd（画面・取り込みタイマー）"
install_if_changed 644 "$T/keiba-ui-stg.service" /etc/systemd/system/keiba-ui-stg.service
install_if_changed 644 "$T/keiba-release-pull-ui-stg.timer" /etc/systemd/system/keiba-release-pull-ui-stg.timer
install_if_changed 644 "$T/keiba-release-pull@.service" /etc/systemd/system/keiba-release-pull@.service

say "== 5/6 nginx（stg 用のサイトだけを足す。本番の keiba-ui には触れない）"
TMP_CONF="$(mktemp)"; trap 'rm -f "$TMP_CONF"' EXIT
sed -e "s/__CLOUDRUN_HOST_RE__/${CR_RE}/g" -e "s/__CLOUDRUN_HOST__/${CR_HOST}/g" -e "s/__PROXY_PORT__/${PROXY_PORT}/g" -e "s/__APP_PORT__/${APP_PORT}/g" "$T/nginx-keiba-ui-stg.conf" > "$TMP_CONF"
if [ ! -f /etc/nginx/sites-available/keiba-ui-stg ] || ! cmp -s "$TMP_CONF" /etc/nginx/sites-available/keiba-ui-stg; then
  say "  更新: /etc/nginx/sites-available/keiba-ui-stg（API の上流: ${CR_HOST}）"; CHANGED=1
  run install -m 644 "$TMP_CONF" /etc/nginx/sites-available/keiba-ui-stg
fi
run ln -sfn /etc/nginx/sites-available/keiba-ui-stg /etc/nginx/sites-enabled/keiba-ui-stg

say "== 6/6 反映"
if [ "$DRY" = 1 ]; then
  say "  (dry-run) nginx -t && systemctl reload nginx / daemon-reload / enable --now keiba-release-pull-ui-stg.timer / enable keiba-ui-stg"
else
  nginx -t && systemctl reload nginx
  systemctl daemon-reload
  systemctl enable keiba-ui-stg.service >/dev/null 2>&1 || true        # 画面の初回導入（install_release.sh）で起動する
  systemctl enable --now keiba-release-pull-ui-stg.timer
  [ "$PULL_NOW" != 1 ] || systemctl start keiba-release-pull@ui-stg.service || true
fi
say "完了。変更あり=${CHANGED}。確認:"
say "  systemctl list-timers | grep ui-stg ; journalctl -u keiba-release-pull@ui-stg -n 20 --no-pager"
say "  curl -s http://127.0.0.1:${APP_PORT}/version.json   # 初回の導入後（stg ブランチの push で公開された版）"
say "次: Cloudflare で stg.megu-keiba.com の転送先を『この VPS のトンネル → http://localhost:${PROXY_PORT}』に付け替える。"
