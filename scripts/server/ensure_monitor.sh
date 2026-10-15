#!/usr/bin/env bash
# 開発者監視ポータル（src/monitor、config/ports.env の KEIBA_PORT_MONITOR）が落ちていれば起動する。冪等。
#
# cron から呼ぶ想定（コンテナ起動時と数分おき）:
#   @reboot sleep 30 && /path/to/keiba-vpn/scripts/server/ensure_monitor.sh
#   */5 * * * * /path/to/keiba-vpn/scripts/server/ensure_monitor.sh
#
# 既定は 127.0.0.1 で待ち受ける（ssh -L / トンネル経由で見る。外へ直接は出さない）。変える場合は KEIBA_MONITOR_HOST。
# 前提: .env に MONITOR_PASSWORD / MONITOR_SECRET_KEY（start_monitor.sh と同じ）。
# ログ: logs/ensure_monitor.log

set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT" || exit 1
export PATH="/opt/conda/bin:${PATH:-}:/usr/local/bin:/usr/bin:/bin"
export TZ="${TZ:-Asia/Tokyo}"

# shellcheck source=ports.sh
source "$ROOT/scripts/server/ports.sh"
load_keiba_ports "$ROOT"

PORT="${KEIBA_PORT_MONITOR:?KEIBA_PORT_MONITOR が未設定}"
HOST="${KEIBA_MONITOR_HOST:-127.0.0.1}"
LOG="$ROOT/logs/ensure_monitor.log"
mkdir -p "$ROOT/logs"

up() {
  local code
  code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/login" 2>/dev/null || true)"
  [[ "$code" =~ ^(200|302|401|403)$ ]]
}

up && exit 0

exec 9>"$ROOT/.ensure_monitor.lock"
flock -n 9 || exit 0   # 同時実行は 1 本だけ
up && exit 0

{
  echo "=== $(date -Iseconds) 監視ポータルが応答しないため起動します (host=${HOST} port=${PORT}) ==="
  bash "$ROOT/scripts/server/start_monitor.sh" --bg --host "$HOST" 9>&-
  for _ in $(seq 1 20); do up && { echo "起動を確認しました"; exit 0; }; sleep 1; done
  echo "起動後 20 秒経っても応答しません。logs/monitor_*.log を確認してください"
} >>"$LOG" 2>&1
up
