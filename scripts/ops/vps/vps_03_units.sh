#!/usr/bin/env bash
# VPS の常駐最小構成の仕上げ（architecture_target.md 3.2 / 5.2 / 5.6）。root で実行。何度実行しても同じ結果（冪等）。vps_01・vps_02 の後。
#   bash vps_03_units.sh [--dry-run] [--with-pull-venv] [--with-fetch-venv <requirements-fetch.txt>] [--enable-fetch-agent]
#
# やること:
#   1) ユーザー keibafetch・ディレクトリ・環境ファイルの雛形（秘密なし。値は set_secret.sh で後から入れる）
#   2) heartbeat_client.py と pull_release.py を /opt/keiba-ui/bin（root 所有）へ
#   3) systemd: keiba-fetch.slice・取得の実行役・nginx/cloudflared/Next.js のメモリ上限（MemoryHigh/Max）・ハートビート・pull 型デプロイ
#   4) 任意: --with-pull-venv（GCS 読み取り用の小さな venv）、--with-fetch-venv（取得の実行役の venv。requirements-fetch.txt を渡す）
#   5) タイマーの有効化（秘密の環境ファイルが埋まっているものだけ）。取得の実行役は --enable-fetch-agent と fetch.env の ENABLED=1 が揃ったときだけ
# --dry-run は何も変更せず、何をするかだけ表示する。
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; T="$HERE/templates"
DRY=0; PULL_VENV=0; FETCH_VENV=""; ENABLE_AGENT=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1; shift;; --with-pull-venv) PULL_VENV=1; shift;;
  --with-fetch-venv) FETCH_VENV="${2:?requirements-fetch.txt のパス}"; shift 2;;
  --enable-fetch-agent) ENABLE_AGENT=1; shift;;
  *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ "$DRY" = 1 ] || [ "$(id -u)" = 0 ] || { echo "root で実行してください（確認だけなら --dry-run）" >&2; exit 1; }
CHANGED=0
say() { echo "$*"; }
run() { if [ "$DRY" = 1 ]; then echo "  (dry-run) $*"; else "$@"; fi; }
# install_if_changed <mode> <src> <dst>: 内容が違うときだけ置く
install_if_changed() {
  local mode="$1" src="$2" dst="$3"
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then return 0; fi
  say "  更新: $dst"; CHANGED=1
  run install -D -m "$mode" -o root -g root "$src" "$dst"
}
# write_if_missing <mode> <dst> <内容(stdin)>
write_if_missing() {
  local mode="$1" dst="$2" body; body="$(cat)"
  [ -e "$dst" ] && return 0
  say "  新規(雛形): $dst"
  if [ "$DRY" != 1 ]; then install -d "$(dirname "$dst")"; umask 077; printf '%s\n' "$body" > "$dst"; chmod "$mode" "$dst"; fi
}

say "== 1/6 ユーザー・ディレクトリ"
id keibafetch >/dev/null 2>&1 || { say "  ユーザー keibafetch を作成（ログイン不可）"; run useradd --system --home-dir /opt/keiba-fetch --shell /usr/sbin/nologin keibafetch; }
for d in /opt/keiba-fetch /opt/keiba-fetch/releases /opt/keiba-fetch/incoming /opt/keiba-pull /var/lib/keiba-pull; do run install -d -m 755 "$d"; done
run install -d -m 750 -o root -g keibafetch /etc/keiba-fetch
run install -d -m 700 /etc/keiba-pull
run install -d -m 755 -o keibafetch -g keibafetch /var/lib/keiba-fetch /var/lib/keiba-fetch/data /var/lib/keiba-fetch/logs

say "== 2/6 スクリプト（root 所有。deploy は書き換えられない）"
install_if_changed 755 "$HERE/heartbeat_client.py" /opt/keiba-ui/bin/heartbeat_client.py
install_if_changed 755 "$HERE/pull_release.py" /opt/keiba-ui/bin/pull_release.py

say "== 3/6 systemd ユニット"
install_if_changed 644 "$T/keiba-fetch.slice" /etc/systemd/system/keiba-fetch.slice
install_if_changed 644 "$T/keiba-fetch-agent.service" /etc/systemd/system/keiba-fetch-agent.service
install_if_changed 644 "$T/keiba-heartbeat.service" /etc/systemd/system/keiba-heartbeat.service
install_if_changed 644 "$T/keiba-heartbeat.timer" /etc/systemd/system/keiba-heartbeat.timer
install_if_changed 644 "$T/keiba-release-pull@.service" /etc/systemd/system/keiba-release-pull@.service
install_if_changed 644 "$T/keiba-release-pull-ui.timer" /etc/systemd/system/keiba-release-pull-ui.timer
install_if_changed 644 "$T/keiba-release-pull-fetch.timer" /etc/systemd/system/keiba-release-pull-fetch.timer
# メモリ上限（MemoryHigh=ソフト、MemoryMax=ハード）。既存ユニットは書き換えず drop-in で上書きする
install_if_changed 644 "$T/dropin-nginx-memory.conf" /etc/systemd/system/nginx.service.d/50-keiba-memory.conf
install_if_changed 644 "$T/dropin-cloudflared-prod-memory.conf" /etc/systemd/system/cloudflared-prod.service.d/50-keiba-memory.conf
install_if_changed 644 "$T/dropin-keiba-ui-memory.conf" /etc/systemd/system/keiba-ui.service.d/50-keiba-memory.conf
[ "$CHANGED" = 0 ] || run systemctl daemon-reload

say "== 4/6 環境ファイルの雛形（秘密なし）"
write_if_missing 600 /etc/keiba-fetch/fetch.env <<'ENVF'
# 取得の実行役の環境。秘密の値は set_secret.sh で入れる（画面・ログに出さない）。有効化は KEIBA_FETCH_AGENT_ENABLED=1
KEIBA_ENV=prod
KEIBA_RUN_LOCATION=vps
KEIBA_FETCH_AGENT_ENABLED=0
# 影モード: ホストが取得を続けたまま、同じ取得を shadow/ 以下へ並走して本番と比べる（docs/operations/vps_agent.md の影モードの章）。既定オフ
KEIBA_FETCH_SHADOW_ENABLED=0
GCS_BUCKET=
GOOGLE_APPLICATION_CREDENTIALS=/etc/keiba-fetch/sa.json
# netkeiba のログイン等（.env.example の NETKEIBA_* と同じ名前）はここにだけ置く
ENVF
write_if_missing 600 /etc/keiba-pull/pull.env <<'ENVF'
# pull 型デプロイ。GCS_BUCKET と、読み取り中心の専用サービスアカウントの鍵ファイル
KEIBA_ENV=prod
KEIBA_RUN_LOCATION=vps
GCS_BUCKET=
GOOGLE_APPLICATION_CREDENTIALS=/etc/keiba-pull/sa.json
ENVF

say "== 5/6 venv（任意）"
if [ "$PULL_VENV" = 1 ]; then
  if [ ! -x /opt/keiba-pull/venv/bin/python ]; then
    run python3 -m venv /opt/keiba-pull/venv
    run /opt/keiba-pull/venv/bin/pip install -q --disable-pip-version-check google-cloud-storage
  else say "  pull 用 venv は作成済み"; fi
fi
if [ -n "$FETCH_VENV" ]; then
  [ -r "$FETCH_VENV" ] || { echo "requirements-fetch.txt を読めません: $FETCH_VENV" >&2; exit 1; }
  [ -x /opt/keiba-fetch/venv/bin/python ] || run python3 -m venv /opt/keiba-fetch/venv
  run /opt/keiba-fetch/venv/bin/pip install -q --disable-pip-version-check -r "$FETCH_VENV"
fi

say "== 6/6 タイマーの有効化"
filled() { [ -s "$1" ] && grep -q "^$2=.\+" "$1"; }          # 値が空でない行があるか（値は表示しない）
if filled /etc/keiba-ui/heartbeat.env HEARTBEAT_URL; then run systemctl enable --now keiba-heartbeat.timer; else say "  ハートビート: /etc/keiba-ui/heartbeat.env が未設定のため有効化しません（docs/operations/vps_agent.md 手順 3）"; fi
if filled /etc/keiba-pull/pull.env GCS_BUCKET && [ -s /etc/keiba-pull/sa.json ] && [ -x /opt/keiba-pull/venv/bin/python ]; then
  run systemctl enable --now keiba-release-pull-ui.timer keiba-release-pull-fetch.timer
else say "  pull 型デプロイ: pull.env / sa.json / venv が揃っていないため有効化しません"; fi
if [ "$ENABLE_AGENT" = 1 ]; then
  if grep -q '^KEIBA_FETCH_AGENT_ENABLED=1' /etc/keiba-fetch/fetch.env 2>/dev/null && [ -L /opt/keiba-fetch/current ] && [ -x /opt/keiba-fetch/venv/bin/python ]; then
    run systemctl enable --now keiba-fetch-agent.service
  else say "  取得の実行役: ENABLED=1・current(コードバンドル)・venv が揃っていないため有効化しません"; fi
else say "  取得の実行役は有効化しません（--enable-fetch-agent で。既定はホスト cron のまま）"; fi
[ "$DRY" = 1 ] || { systemctl list-timers --no-pager 2>/dev/null | grep -E "keiba|NEXT" || true; }
say "完了"
