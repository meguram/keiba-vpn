#!/usr/bin/env bash
# ホスト PC(stg)の画面サーバを、この PC の localhost から見られるようにする SSH ポートフォワード。
#
#   bash scripts/server/tunnel_stg_ui.sh start    # バックグラウンドで張る（既にあれば何もしない）
#   bash scripts/server/tunnel_stg_ui.sh status
#   bash scripts/server/tunnel_stg_ui.sh stop
#
# 見られるもの（ポートは config/ports.env の stg。ローカルのポートはホストと同じ。監視ポータルだけ +1 する）:
#   MLflow          http://localhost:23230
#   フロント(stg)   http://localhost:23200
#   Flask API       http://localhost:23210/api/v1/health
#   FastAPI         http://localhost:23220/api/health
#   監視ポータル    http://localhost:23901   （ローカルの dev 用監視ポータルが 23900 を使うため）
#
# 接続先は ssh の別名(既定 keiba-remote。~/.ssh/config に設定済み)。セッションを閉じる(stop)と見られなくなる。
# 経路は ngrok 経由なので、画像などの大きなファイルを何度も開くと転送量を使う。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
# shellcheck source=ports.sh
source "$ROOT/scripts/server/ports.sh"
load_keiba_ports "$ROOT"
HOST="${KEIBA_REMOTE_HOST:-keiba-remote}"
SOCK="${TMPDIR:-/tmp}/keiba-stg-tunnel.sock"
MON_LOCAL="${KEIBA_STG_MONITOR_LOCAL_PORT:-$((KEIBA_PORT_MONITOR + 1))}"

forwards=(
  -L "${KEIBA_PORT_STG_MLFLOW}:127.0.0.1:${KEIBA_PORT_STG_MLFLOW}"
  -L "${KEIBA_PORT_STG_FRONTEND}:127.0.0.1:${KEIBA_PORT_STG_FRONTEND}"
  -L "${KEIBA_PORT_STG_FLASK}:127.0.0.1:${KEIBA_PORT_STG_FLASK}"
  -L "${KEIBA_PORT_STG_FASTAPI}:127.0.0.1:${KEIBA_PORT_STG_FASTAPI}"
  -L "${MON_LOCAL}:127.0.0.1:${KEIBA_PORT_MONITOR}"
)

alive() { ssh -S "$SOCK" -O check "$HOST" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    if alive; then echo "既に張られています"; exit 0; fi
    rm -f "$SOCK"
    ssh -M -S "$SOCK" -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o BatchMode=yes "${forwards[@]}" "$HOST" </dev/null >/dev/null 2>&1
    sleep 1; alive && echo "張りました" || { echo "失敗しました（ポートが使用中か、接続できません）" >&2; exit 1; }
    ;;
  stop)
    if alive; then ssh -S "$SOCK" -O exit "$HOST" >/dev/null 2>&1 && echo "閉じました"; else echo "張られていません"; fi
    ;;
  status)
    if alive; then echo "張られています"; else echo "張られていません"; exit 1; fi
    ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
if [[ "${1:-status}" != "stop" ]] && alive; then
  for pair in "MLflow:${KEIBA_PORT_STG_MLFLOW}:/" "フロント:${KEIBA_PORT_STG_FRONTEND}:/" "Flask:${KEIBA_PORT_STG_FLASK}:/api/v1/health" "FastAPI:${KEIBA_PORT_STG_FASTAPI}:/api/health" "監視:${MON_LOCAL}:/login"; do
    IFS=: read -r name port path <<<"$pair"
    code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}${path}" || true)"
    printf '  %-8s http://localhost:%s%s  → %s\n' "$name" "$port" "$path" "$code"
  done
fi
