#!/usr/bin/env bash
# Cloudflare Tunnel（cloudflared）の起動・停止・状態。トークンは .env の CLOUDFLARE_TUNNEL_TOKEN から読む（画面・ログ・ps に出さない: 環境変数 TUNNEL_TOKEN で渡す）。
#
# Usage:
#   bash scripts/server/tunnel_cloudflared.sh start     # バックグラウンド + 自動再接続（すでに動いていれば何もしない）
#   bash scripts/server/tunnel_cloudflared.sh status    # 動作確認（トークンは出さない）
#   bash scripts/server/tunnel_cloudflared.sh stop
#
# 事前: cloudflared を入れ（dpkg -i cloudflared-linux-amd64.deb）、.env に CLOUDFLARE_TUNNEL_TOKEN=<Zero Trust のトンネル作成画面のトークン> を書く。
# 公開ホスト名 → http://localhost:23200 の対応は Cloudflare の画面（Public Hostname）で設定する。
set -u
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
LOG="logs/cloudflared.log"
PIDFILE="$ROOT/.cloudflared.pid"
ACTION="${1:-status}"

load_token() {
  if [ -f .env ]; then
    TUNNEL_TOKEN="$(grep -E '^CLOUDFLARE_TUNNEL_TOKEN=' .env | tail -n1 | cut -d= -f2-)"
  fi
  TUNNEL_TOKEN="${TUNNEL_TOKEN:-${CLOUDFLARE_TUNNEL_TOKEN:-}}"
  if [ -z "$TUNNEL_TOKEN" ]; then
    echo ".env に CLOUDFLARE_TUNNEL_TOKEN がありません（値は表示しません）" >&2
    return 1
  fi
  export TUNNEL_TOKEN
}

is_running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "$ACTION" in
  start)
    command -v cloudflared >/dev/null 2>&1 || { echo "cloudflared が見つかりません" >&2; exit 1; }
    if is_running; then echo "すでに動いています (pid $(cat "$PIDFILE"))"; exit 0; fi
    load_token || exit 1
    (
      # 切れたら 10 秒後に再接続する簡易ループ（トークンは環境変数のまま渡す）
      while true; do
        cloudflared tunnel --no-autoupdate --loglevel info run >>"$LOG" 2>&1
        echo "[$(date -Is)] cloudflared が終了しました。10 秒後に再接続します" >>"$LOG"
        sleep 10
      done
    ) &
    echo $! >"$PIDFILE"
    sleep 5
    echo "起動しました (pid $(cat "$PIDFILE"))。ログ: $LOG"
    grep -E "Registered tunnel connection|ERR|error" "$LOG" | tail -n 3 | sed -E 's/eyJ[A-Za-z0-9_.-]+/<masked>/g'
    ;;
  status)
    if is_running; then echo "動作中 (pid $(cat "$PIDFILE"))"; else echo "停止中"; fi
    [ -f "$LOG" ] && grep -E "Registered tunnel connection|ERR" "$LOG" | tail -n 4 | sed -E 's/eyJ[A-Za-z0-9_.-]+/<masked>/g'
    ;;
  stop)
    if is_running; then
      pid="$(cat "$PIDFILE")"
      pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
      rm -f "$PIDFILE"
      echo "停止しました"
    else
      rm -f "$PIDFILE"; echo "動いていません"
    fi
    ;;
  *)
    echo "Usage: $0 start|status|stop" >&2; exit 2 ;;
esac
