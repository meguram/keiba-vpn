#!/usr/bin/env bash
# 会議の記録のビューア（src/scripts/docs/council_viewer.py、config/ports.env の KEIBA_PORT_COUNCIL）が落ちていれば起動する。冪等。
#
# cron から呼ぶ想定（ensure_monitor.sh と同じ。コンテナ起動時と 5 分おき）:
#   @reboot sleep 30 && /path/to/keiba-vpn/scripts/server/ensure_council_viewer.sh # KEIBA-VPN-COUNCIL
#   */5 * * * * /path/to/keiba-vpn/scripts/server/ensure_council_viewer.sh # KEIBA-VPN-COUNCIL
#
# 5 分おきの確認は 127.0.0.1 への curl 1 回だけ（GCS・外の網は使わない。費用 0 円）。
# 待ち受けは 127.0.0.1 に固定する（確かめも 127.0.0.1。認証が無いので外へは出さない）。
# ログ: logs/ensure_council_viewer.log

set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT" || exit 1
export PATH="/opt/conda/bin:${PATH:-}:/usr/local/bin:/usr/bin:/bin"
export TZ="${TZ:-Asia/Tokyo}"

# shellcheck source=ports.sh
source "$ROOT/scripts/server/ports.sh"
load_keiba_ports "$ROOT"

PORT="${KEIBA_PORT_COUNCIL:?KEIBA_PORT_COUNCIL が未設定}"
LOG="$ROOT/logs/ensure_council_viewer.log"
mkdir -p "$ROOT/logs"

up() {
  [[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/healthz" 2>/dev/null || true)" == "200" ]]
}

up && exit 0

exec 9>"$ROOT/.ensure_council_viewer.lock"
flock -n 9 || exit 0   # 同時実行は 1 本だけ
up && exit 0

{
  echo "=== $(date -Iseconds) 会議の記録のビューアが応答しないため起動します (port=${PORT}) ==="
  bash "$ROOT/scripts/server/start_council_viewer.sh" --bg --host 127.0.0.1 --port "$PORT" 9>&-
  for _ in $(seq 1 20); do up && { echo "起動を確認しました"; exit 0; }; sleep 1; done
  echo "起動後 20 秒経っても応答しません。logs/council_viewer.log を確認してください"
} >>"$LOG" 2>&1
up
