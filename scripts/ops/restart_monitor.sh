#!/usr/bin/env bash
# 監視ポータル(src/monitor)を再起動する（テンプレート・app.py の変更を反映）。ホストで root として実行する。
#   sudo bash scripts/ops/restart_monitor.sh
# 落とすと cron の ensure_monitor.sh（5 分おき）でも戻るが、ここで即起動する。冪等。
set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT" || exit 1
source "$ROOT/scripts/server/ports.sh"; load_keiba_ports "$ROOT"
PORT="${KEIBA_PORT_MONITOR:?KEIBA_PORT_MONITOR が未設定}"
PIDS="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $NF}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"
# ss でポートの持ち主が見えない環境があるので、見つからなければコマンドラインで探す
[ -n "$PIDS" ] || PIDS="$(pgrep -f "src.monitor.app --host 127.0.0.1 --port $PORT" || true)"
[ -n "$PIDS" ] && { echo "停止: $PIDS"; kill $PIDS; sleep 2; }
bash "$ROOT/scripts/server/ensure_monitor.sh"
sleep 3
code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/login" || true)"
echo "監視ポータル: HTTP $code（200 なら起動済み）"
[ "$code" = "200" ]
