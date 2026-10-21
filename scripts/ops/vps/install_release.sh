#!/usr/bin/env bash
# リリース（standalone の tar.gz）を VPS に導入する。root（sudo）で実行: sudo /opt/keiba-ui/bin/install_release.sh /srv/keiba-ui/incoming/<名前>.tar.gz
# 順序: sha256 検証 → 展開 → 別ポートで起動テスト → current を切替 → 再起動 → 確認 → 失敗なら直前の版へ自動で戻す。
set -euo pipefail
TARBALL="${1:?tar.gz のパス}"
. /etc/keiba-ui/vars.env 2>/dev/null || true
APP_ROOT="${APP_ROOT:-/srv/keiba-ui}"; APP_PORT="${APP_PORT:-3000}"; PROXY_PORT="${PROXY_PORT:-8080}"; KEEP="${KEEP_RELEASES:-5}"
LOG=/var/log/keiba-ui/deploy.log; mkdir -p /var/log/keiba-ui
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

case "$TARBALL" in "$APP_ROOT"/incoming/*.tar.gz) ;; *) echo "incoming 以外のファイルは導入しません: $TARBALL" >&2; exit 2;; esac
[ -f "$TARBALL" ] && [ -f "$TARBALL.sha256" ] || { echo "tar.gz と .sha256 が必要です" >&2; exit 2; }
ID="$(basename "$TARBALL" .tar.gz)"
case "$ID" in *[!A-Za-z0-9._-]*) echo "不正なリリース名: $ID" >&2; exit 2;; esac
REL="$APP_ROOT/releases/$ID"
( cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "$TARBALL").sha256" ) >/dev/null || { log "NG sha256 が一致しません: $ID"; exit 3; }
log "導入開始: $ID"

if [ -d "$REL" ]; then log "すでに展開済み: $ID（そのまま切替だけ行う）"; else
  mkdir -p "$REL.partial"
  tar -xzf "$TARBALL" -C "$REL.partial" --no-same-owner --no-same-permissions
  [ -f "$REL.partial/server.js" ] || { rm -rf "$REL.partial"; log "NG server.js がありません: $ID"; exit 3; }
  mkdir -p "$REL.partial"/.next-prod/cache
  chown -R deploy:keibaui "$REL.partial"; chmod -R u+rwX,g+rX,o-rwx "$REL.partial"
  chmod -R g+rwX "$REL.partial"/.next-prod/cache
  mv -T "$REL.partial" "$REL"
fi

# 別ポートで起動テスト（サービスは触らない）
TP=3101
( cd "$REL" && sudo -u keibaui env $(grep -v '^#' /etc/keiba-ui/ui.env | grep -v '^$' | xargs) PORT=$TP HOSTNAME=127.0.0.1 node server.js >/tmp/keiba-ui-smoke.log 2>&1 ) &
SP=$!
OK=0
for i in $(seq 1 30); do sleep 1; c="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$TP/version.json" || true)"; [ "$c" = 200 ] && { OK=1; break; }; done
pkill -P "$SP" 2>/dev/null || true; kill "$SP" 2>/dev/null || true; sleep 1
[ "$OK" = 1 ] || { log "NG 起動テストに失敗: $ID（切り替えません）"; tail -5 /tmp/keiba-ui-smoke.log | tee -a "$LOG"; exit 4; }

PREV="$(readlink -f "$APP_ROOT/current" 2>/dev/null || true)"
ln -sfn "$REL" "$APP_ROOT/current.tmp" && mv -T "$APP_ROOT/current.tmp" "$APP_ROOT/current"
[ -n "$PREV" ] && echo "$PREV" > "$APP_ROOT/shared/previous"
systemctl restart keiba-ui
OK=0
for i in $(seq 1 30); do sleep 1; v="$(curl -s -m 3 "http://127.0.0.1:$APP_PORT/version.json" || true)"; echo "$v" | grep -q "\"release\"" && { OK=1; break; }; done
if [ "$OK" = 1 ]; then
  c="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PROXY_PORT/version.json" || true)"
  [ "$c" = 200 ] || { log "警告: nginx 経由の確認が $c（cloudflared / nginx を確認）"; }
fi
if [ "$OK" != 1 ]; then
  log "NG 切替後の確認に失敗。直前の版へ戻します: ${PREV:-なし}"
  if [ -n "$PREV" ] && [ -d "$PREV" ]; then ln -sfn "$PREV" "$APP_ROOT/current.tmp" && mv -T "$APP_ROOT/current.tmp" "$APP_ROOT/current"; systemctl restart keiba-ui; fi
  exit 5
fi
log "OK 切替完了: $ID  $(curl -s -m 3 "http://127.0.0.1:$APP_PORT/version.json" | tr -d '\n' | cut -c1-200)"

# 古い版を整理（現在と直前は残す）
CUR="$(readlink -f "$APP_ROOT/current")"
ls -1dt "$APP_ROOT"/releases/*/ 2>/dev/null | tail -n +$((KEEP+1)) | while read -r d; do d="${d%/}"; [ "$d" = "$CUR" ] && continue; [ "$d" = "$PREV" ] && continue; rm -rf -- "$d"; log "古い版を削除: $(basename "$d")"; done
# 配布ファイルは直近 3 つだけ残す
ls -1t "$APP_ROOT"/incoming/*.tar.gz 2>/dev/null | tail -n +4 | while read -r f; do rm -f -- "$f" "$f.sha256" "${f%.tar.gz}.manifest.json"; done
