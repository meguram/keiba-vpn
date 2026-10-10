#!/usr/bin/env bash
# keiba-vpn 開発者監視ポータル 起動スクリプト
#
# Usage:
#   bash scripts/server/start_monitor.sh             # デフォルト :9090
#   bash scripts/server/start_monitor.sh --port 9191 # ポート指定
#   bash scripts/server/start_monitor.sh --bg        # バックグラウンド起動
#
# 前提: .env に MONITOR_PASSWORD / MONITOR_SECRET_KEY が設定済みであること

set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"

PORT=9090
BG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port|-p) PORT="$2"; shift 2 ;;
    --bg)      BG=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--port PORT] [--bg]"
      exit 0
      ;;
    *) shift ;;
  esac
done

# Python 解決
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "/opt/venv/bin/python3" ]]; then
  PYTHON="/opt/venv/bin/python3"
else
  PYTHON="$(command -v python3)"
fi

# .env 確認（MONITOR_PASSWORD チェック）
if [[ -f "$ROOT/.env" ]]; then
  source_val=$(grep -E '^MONITOR_PASSWORD=' "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)
  if [[ -z "$source_val" ]]; then
    echo "[monitor] 警告: .env に MONITOR_PASSWORD が未設定です"
  fi
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if $BG; then
  LOG_FILE="$LOG_DIR/monitor_$(date +%Y%m%d_%H%M%S).log"
  PID_FILE="$ROOT/.monitor.pid"

  # 既存プロセス停止
  if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[monitor] 既存プロセス (PID=$old_pid) を停止中..."
      kill "$old_pid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "[monitor] バックグラウンド起動 (port=${PORT}) ログ=${LOG_FILE}"
  nohup "$PYTHON" -m src.monitor.app --host 0.0.0.0 --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[monitor] PID=$(cat "$PID_FILE") — http://127.0.0.1:${PORT}/"
else
  echo "[monitor] 起動: http://127.0.0.1:${PORT}/"
  exec "$PYTHON" -m src.monitor.app --host 0.0.0.0 --port "$PORT"
fi
