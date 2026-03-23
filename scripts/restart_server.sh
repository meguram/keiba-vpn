#!/usr/bin/env bash
# keiba-vpn API サーバーを停止してから起動し直す（手動・デプロイ後用）
#
# Usage:
#   ./scripts/restart_server.sh           # 本番相当: main.py --prod（複数ワーカー）
#   ./scripts/restart_server.sh dev       # 開発: ホットリロード（単一ワーカー）
#   PORT=9000 ./scripts/restart_server.sh
#
# ログ: logs/server_restart.log（追記）
# PID:  .server.pid（watchdog と同じファイル。競合する場合はどちらか一方だけ使う）

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
PYTHON="${PYTHON:-$(command -v python3)}"
LOG_DIR="$ROOT/logs"
PID_FILE="$ROOT/.server.pid"
LOG_FILE="$LOG_DIR/server_restart.log"
HEALTH_URL="http://127.0.0.1:${PORT}/api/health"

MODE="prod"
if [[ "${1:-}" == "dev" || "${1:-}" == "--dev" ]]; then
  MODE="dev"
fi

mkdir -p "$LOG_DIR"

http_code() {
  curl -sf -m 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000"
}

stop_server() {
  echo "[restart] 停止処理 (port=${PORT})..."
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      kill "$old_pid" 2>/dev/null || true
      sleep 2
      if kill -0 "$old_pid" 2>/dev/null; then
        kill -9 "$old_pid" 2>/dev/null || true
        sleep 1
      fi
    fi
    rm -f "$PID_FILE"
  fi

  # nohup 親が PID ファイルに無い・子だけ残っている場合
  local pids
  pids=$(pgrep -f "main\.py.*--port[[:space:]]*${PORT}" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "[restart] main.py を停止: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "main\.py.*--port[[:space:]]*${PORT}" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi

  pids=$(pgrep -f "uvicorn.*api\.app:app" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "[restart] uvicorn を停止: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "uvicorn.*api\.app:app" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi

  # まだポートが塞がっている場合（子プロセス残り）
  if command -v fuser >/dev/null 2>&1; then
    if fuser "${PORT}/tcp" >/dev/null 2>&1; then
      echo "[restart] 警告: ポート ${PORT} にまだプロセスあり — fuser -k を試行"
      fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
      sleep 2
    fi
  fi
}

start_server() {
  {
    echo "======== $(date -Iseconds) restart_server MODE=${MODE} PORT=${PORT} ========"
  } >>"$LOG_FILE"

  if [[ "$MODE" == "dev" ]]; then
    echo "[restart] 起動: ${PYTHON} main.py --port ${PORT}（reload）"
    nohup "$PYTHON" main.py --host 0.0.0.0 --port "$PORT" >>"$LOG_FILE" 2>&1 &
  else
    echo "[restart] 起動: ${PYTHON} main.py --port ${PORT} --prod"
    nohup "$PYTHON" main.py --host 0.0.0.0 --port "$PORT" --prod >>"$LOG_FILE" 2>&1 &
  fi
  echo $! >"$PID_FILE"
  echo "[restart] PID=$(cat "$PID_FILE") ログ=$LOG_FILE"
}

wait_health() {
  local max="${1:-20}"
  local i=0
  while [[ "$i" -lt "$max" ]]; do
    code="$(http_code)"
    if [[ "$code" == "200" ]]; then
      echo "[restart] OK /api/health → HTTP ${code}（約 $((i + 1)) 秒）"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "[restart] 警告: ${max} 秒以内に /api/health が 200 になりませんでした（HTTP ${code:-?}）。ログを確認してください。"
  return 1
}

case "${1:-}" in
  -h|--help)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

stop_server
start_server
wait_health 25 || true
