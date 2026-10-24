#!/usr/bin/env bash
# dev 用ローカルサーバ（モック UI 確認用）の自動起動。冪等。
#
# 起動対象: Next.js モック / Flask / FastAPI / 監視ポータル（ポートは config/ports.env）
# 呼び出し元: Claude Code SessionStart フック（.claude/settings.json）/ Cursor sessionStart フック
# 安全策: KEIBA_ENV が dev 以外（stg/prod/未設定）なら何もしない。起動は即座にバックグラウンド化する。
#
# Usage:
#   scripts/server/autostart_dev.sh            # 必要なら起動（ログ: logs/autostart_dev.log）
#   scripts/server/autostart_dev.sh --status   # 4 サービスの死活だけ表示
#   KEIBA_AUTOSTART_DEV=0 ...                  # 自動起動を無効化

set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"

# フックの stdin（JSON）を読み捨てる
if [[ ! -t 0 ]]; then cat >/dev/null || true; fi

# shellcheck source=ports.sh
source "$ROOT/scripts/server/ports.sh"
load_keiba_ports "$ROOT"

MODE=start
[[ "${1:-}" == "--status" ]] && MODE=status

dev_env() {
  local v="${KEIBA_ENV:-}"
  if [[ -z "$v" && -f "$ROOT/.env" ]]; then
    v="$(grep -E '^KEIBA_ENV=' "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
  fi
  [[ "$v" == "dev" ]]
}

code() { curl -s -m 3 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || true; }
up() { [[ "$(code "$1")" =~ ^(200|302|304|401|403)$ ]]; }

FRONT="http://127.0.0.1:${KEIBA_PORT_DEV_FRONTEND}/"
FLASK="http://127.0.0.1:${KEIBA_PORT_DEV_FLASK}/api/v1/health"
FAST="http://127.0.0.1:${KEIBA_PORT_DEV_FASTAPI}/api/health"
MON="http://127.0.0.1:${KEIBA_PORT_MONITOR}/login"

if [[ "$MODE" == "status" ]]; then
  for pair in "Next.js(mock):$FRONT" "Flask:$FLASK" "FastAPI:$FAST" "Monitor:$MON"; do
    if up "${pair#*:}"; then echo "OK   ${pair%%:*}  ${pair#*:}"; else echo "DOWN ${pair%%:*}  ${pair#*:}"; fi
  done
  exit 0
fi

[[ "${KEIBA_AUTOSTART_DEV:-1}" == "0" ]] && exit 0
dev_env || exit 0

LOG="$ROOT/logs/autostart_dev.log"
mkdir -p "$ROOT/logs"

run() {
  # 同時に複数セッションから呼ばれても 1 本だけ走らせる（子サーバに fd を継がせないよう 9>&- で閉じる）
  exec 9>"$ROOT/.autostart_dev.lock"
  flock -n 9 || exit 0
  echo "=== $(date -Iseconds) autostart_dev ==="
  # 別 PC でも同じ表示になるよう、dev ストア（モック + git 配布の実データ日）を揃える。冪等
  python3 -m src.scripts.data.dev_fixture ensure 9>&- || true
  if ! up "$FRONT" || ! up "$FLASK" || ! up "$FAST"; then
    bash "$ROOT/scripts/server/service_start.sh" --env dev --mock --with-fastapi 9>&- || true
  fi
  if ! up "$MON"; then
    bash "$ROOT/scripts/server/start_monitor.sh" --bg --host 127.0.0.1 9>&- || true
  fi
  echo "=== done ==="
}

if up "$FRONT" && up "$FLASK" && up "$FAST" && up "$MON"; then
  exit 0
fi

export -f run code up dev_env
export ROOT FRONT FLASK FAST MON KEIBA_PORT_DEV_FRONTEND KEIBA_PORT_DEV_FLASK KEIBA_PORT_DEV_FASTAPI KEIBA_PORT_MONITOR
nohup setsid bash -c run >>"$LOG" 2>&1 </dev/null &
disown || true
exit 0
