#!/usr/bin/env bash
# 「絶対回収率プラスにするマン」の会議の記録のビューア（src/scripts/docs/council_viewer.py）の起動・停止。
#
# Usage:
#   bash scripts/server/start_council_viewer.sh                # 前面で起動（config/ports.env の KEIBA_PORT_COUNCIL）
#   bash scripts/server/start_council_viewer.sh --bg           # バックグラウンド起動（nice 19。PID: .council_viewer.pid）
#   bash scripts/server/start_council_viewer.sh --stop         # 停止
#   bash scripts/server/start_council_viewer.sh --bg --port 8090
#   KEIBA_PORT_COUNCIL=8090 bash scripts/server/start_council_viewer.sh --bg   # 環境変数でも替えられる
#
# 認証が無いので 127.0.0.1 だけで待ち受ける（Cursor の Remote-SSH の「ポート」・ssh -L で手元へ転送して見る）。
# 変える場合は --host か KEIBA_COUNCIL_HOST。読むのは docs/ の下だけ。GCS は読まない（費用 0 円）。

set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"
# markdown パッケージが入っているのは conda の python（cron の限られた PATH でも見えるよう先頭に足す）
export PATH="/opt/conda/bin:${PATH:-/usr/bin:/bin}"

# shellcheck source=ports.sh
source "$ROOT/scripts/server/ports.sh"
load_keiba_ports "$ROOT"

PORT="${KEIBA_PORT_COUNCIL:?KEIBA_PORT_COUNCIL が未設定}"
HOST="${KEIBA_COUNCIL_HOST:-127.0.0.1}"
BG=false
STOP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port|-p) PORT="${2:?--port に番号が要る}"; shift 2 ;;
    --host)    HOST="${2:?--host に番地が要る}"; shift 2 ;;
    --bg)      BG=true; shift ;;
    --stop)    STOP=true; shift ;;
    --help|-h) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "[council] 知らない引数: $1（--help を参照）" >&2; exit 2 ;;
  esac
done

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

PID_FILE="$ROOT/.council_viewer.pid"

# PID ファイルの番号がビューア自身のときだけ止める（再起動の後に同じ番号を引き継いだ別のプロセスを止めない）
is_viewer() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
    && tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q 'src\.scripts\.docs\.council_viewer'
}

stop_old() {
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_viewer "$old_pid"; then
      echo "[council] 既存プロセス (PID=$old_pid) を停止します"
      kill "$old_pid" 2>/dev/null || true
      for _ in $(seq 1 10); do is_viewer "$old_pid" || break; sleep 0.5; done
      is_viewer "$old_pid" && kill -9 "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}

if $STOP; then
  stop_old
  exit 0
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if $BG; then
  stop_old
  LOG_FILE="$LOG_DIR/council_viewer.log"   # 名前を固定して追記（起動に失敗し続けてもファイルが増えない）
  nohup nice -n 19 "$PYTHON" -m src.scripts.docs.council_viewer --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[council] PID=$(cat "$PID_FILE") ログ=${LOG_FILE}"
  echo "Listening on http://${HOST}:${PORT}/"
else
  exec nice -n 19 "$PYTHON" -m src.scripts.docs.council_viewer --host "$HOST" --port "$PORT"
fi
