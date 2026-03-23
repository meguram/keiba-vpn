#!/usr/bin/env bash
# ================================================================
# keiba-vpn 全サービス監視 & 自動再起動スクリプト
#
# 監視対象:
#   1) keiba-vpn FastAPI サーバー (port 8000)
#   2) MLflow Tracking サーバー   (port 5000)
#
# cron から定期実行される。
# 各サービスのヘルスチェックを行い、ダウン時は自動再起動する。
#
# Usage:
#   ./scripts/server_watchdog.sh              # 全サービス監視
#   ./scripts/server_watchdog.sh --api-only   # API のみ
#   ./scripts/server_watchdog.sh --status     # 状態表示のみ (再起動なし)
# ================================================================

set -euo pipefail

# ── 設定 ──
PROJECT_DIR="/home/hirokiakataoka/project/myproject/keiba-vpn"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
MAX_LOG_LINES=5000

# サービス定義
API_PORT=8000
API_PID_FILE="${PROJECT_DIR}/.server.pid"
API_HEALTH_URL="http://127.0.0.1:${API_PORT}/api/health"

MLFLOW_PORT=5000
MLFLOW_PID_FILE="${PROJECT_DIR}/.mlflow.pid"
MLFLOW_HEALTH_URL="http://127.0.0.1:${MLFLOW_PORT}/health"
MLFLOW_BACKEND_STORE="${PROJECT_DIR}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT="${PROJECT_DIR}/mlruns/artifacts"

PYTHON="/home/hirokiakataoka/miniconda3/bin/python3"
MLFLOW_CMD="/home/hirokiakataoka/.local/bin/mlflow"

# Python / MLflow がなければ fallback
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(which python3 2>/dev/null || echo /usr/bin/python3)"
fi
if [ ! -x "$MLFLOW_CMD" ]; then
    MLFLOW_CMD="$(which mlflow 2>/dev/null || echo "")"
fi

mkdir -p "$LOG_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*" >> "$LOG_FILE"
}

# ── ログローテーション ──
if [ -f "$LOG_FILE" ]; then
    line_count=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$line_count" -gt "$MAX_LOG_LINES" ]; then
        tail -n 2000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
        log "ログローテーション実行 (${line_count} → 2000行)"
    fi
fi

# ═══════════════════════════════════════════════════════
# 汎用ヘルパー
# ═══════════════════════════════════════════════════════

http_check() {
    local url="$1"
    local code
    code=$(curl -sf -m 10 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null) || code="000"
    echo "$code"
}

is_pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

kill_by_pid_file() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null || echo "")
        if is_pid_alive "$old_pid"; then
            log "  プロセス停止中 (PID=$old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            sleep 2
            if is_pid_alive "$old_pid"; then
                kill -9 "$old_pid" 2>/dev/null || true
                sleep 1
            fi
        fi
        rm -f "$pid_file"
    fi
}

kill_by_pattern() {
    local pattern="$1"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "  残存プロセスを停止 ($pattern): $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
}

wait_for_health() {
    local url="$1"
    local max_wait="${2:-15}"
    local waited=0
    while [ $waited -lt "$max_wait" ]; do
        sleep 1
        waited=$((waited + 1))
        local code
        code=$(http_check "$url")
        if [ "$code" = "200" ]; then
            echo "$waited"
            return 0
        fi
    done
    echo "0"
    return 1
}

increment_counter() {
    local file="$1"
    local count=0
    if [ -f "$file" ]; then
        count=$(cat "$file" 2>/dev/null || echo 0)
    fi
    count=$((count + 1))
    echo "$count" > "$file"
    echo "$count"
}

reset_counter() {
    local file="$1"
    echo "0" > "$file"
}

# ═══════════════════════════════════════════════════════
# keiba-vpn API サーバー
# ═══════════════════════════════════════════════════════

stop_api() {
    kill_by_pid_file "$API_PID_FILE"
    kill_by_pattern "uvicorn.*api.app:app"
}

start_api() {
    log "[API] サーバー起動中 (port=${API_PORT})..."
    cd "$PROJECT_DIR"

    nohup "$PYTHON" main.py --port "$API_PORT" --prod \
        >> "${LOG_DIR}/server.log" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$API_PID_FILE"

    local secs
    secs=$(wait_for_health "$API_HEALTH_URL" 15) || true
    if [ "$secs" != "0" ]; then
        log "[API] 起動完了 (PID=$new_pid, ${secs}秒)"
        return 0
    fi
    log "[API] WARNING: 起動したが応答確認できず (PID=$new_pid)"
    return 1
}

check_api() {
    local code
    code=$(http_check "$API_HEALTH_URL")

    if [ "$code" = "200" ]; then
        return 0
    fi

    log "[API] ALERT: 応答なし (HTTP=${code})"

    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.api_fail_count")

    if [ "$fail_count" -ge 5 ]; then
        log "[API] ERROR: ${fail_count}回連続失敗 — 60秒待機後に再起動"
        sleep 60
    fi

    stop_api
    if start_api; then
        reset_counter "${LOG_DIR}/.api_fail_count"
        log "[API] 再起動成功"
    else
        log "[API] ERROR: 再起動後もヘルスチェック失敗"
    fi
    return 1
}

# ═══════════════════════════════════════════════════════
# MLflow Tracking サーバー
# ═══════════════════════════════════════════════════════

stop_mlflow() {
    kill_by_pid_file "$MLFLOW_PID_FILE"
    kill_by_pattern "mlflow.*server.*${MLFLOW_PORT}"
}

start_mlflow() {
    if [ -z "$MLFLOW_CMD" ]; then
        log "[MLflow] WARNING: mlflow コマンドが見つかりません — スキップ"
        return 1
    fi

    log "[MLflow] サーバー起動中 (port=${MLFLOW_PORT})..."
    mkdir -p "$(dirname "$MLFLOW_BACKEND_STORE")" "$MLFLOW_ARTIFACT_ROOT"
    cd "$PROJECT_DIR"

    nohup "$MLFLOW_CMD" server \
        --host 127.0.0.1 \
        --port "$MLFLOW_PORT" \
        --backend-store-uri "sqlite:///${MLFLOW_BACKEND_STORE}" \
        --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
        --serve-artifacts \
        >> "${LOG_DIR}/mlflow.log" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$MLFLOW_PID_FILE"

    local secs
    secs=$(wait_for_health "$MLFLOW_HEALTH_URL" 20) || true
    if [ "$secs" != "0" ]; then
        log "[MLflow] 起動完了 (PID=$new_pid, ${secs}秒)"
        return 0
    fi

    # MLflow /health が無いバージョンもあるので、API で再チェック
    local alt_code
    alt_code=$(http_check "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1")
    if [ "$alt_code" = "200" ]; then
        log "[MLflow] 起動完了 — API応答OK (PID=$new_pid)"
        return 0
    fi

    log "[MLflow] WARNING: 起動したが応答確認できず (PID=$new_pid)"
    return 1
}

check_mlflow() {
    if [ -z "$MLFLOW_CMD" ]; then
        return 0
    fi

    # /health または experiments API で確認
    local code
    code=$(http_check "$MLFLOW_HEALTH_URL")
    if [ "$code" = "200" ]; then
        return 0
    fi
    code=$(http_check "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1")
    if [ "$code" = "200" ]; then
        return 0
    fi

    log "[MLflow] ALERT: 応答なし (port=${MLFLOW_PORT})"

    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.mlflow_fail_count")

    if [ "$fail_count" -ge 5 ]; then
        log "[MLflow] ERROR: ${fail_count}回連続失敗 — 60秒待機後に再起動"
        sleep 60
    fi

    stop_mlflow
    if start_mlflow; then
        reset_counter "${LOG_DIR}/.mlflow_fail_count"
        log "[MLflow] 再起動成功"
    else
        log "[MLflow] ERROR: 再起動後もヘルスチェック失敗"
    fi
    return 1
}

# ═══════════════════════════════════════════════════════
# ステータス表示
# ═══════════════════════════════════════════════════════

show_status() {
    echo "=== keiba-vpn サービス状態 ==="
    echo ""

    # API
    local api_code
    api_code=$(http_check "$API_HEALTH_URL")
    local api_pid=""
    if [ -f "$API_PID_FILE" ]; then api_pid=$(cat "$API_PID_FILE" 2>/dev/null || echo ""); fi
    if [ "$api_code" = "200" ]; then
        local body
        body=$(curl -sf -m 5 "$API_HEALTH_URL" 2>/dev/null || echo "{}")
        echo "  [API]    ✅ 稼働中  port=${API_PORT}  PID=${api_pid:-?}  ${body}"
    else
        echo "  [API]    ❌ 停止    port=${API_PORT}  HTTP=${api_code}"
    fi

    # MLflow
    if [ -n "$MLFLOW_CMD" ]; then
        local ml_code
        ml_code=$(http_check "$MLFLOW_HEALTH_URL")
        if [ "$ml_code" != "200" ]; then
            ml_code=$(http_check "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1")
        fi
        local ml_pid=""
        if [ -f "$MLFLOW_PID_FILE" ]; then ml_pid=$(cat "$MLFLOW_PID_FILE" 2>/dev/null || echo ""); fi
        if [ "$ml_code" = "200" ]; then
            echo "  [MLflow] ✅ 稼働中  port=${MLFLOW_PORT}  PID=${ml_pid:-?}"
        else
            echo "  [MLflow] ❌ 停止    port=${MLFLOW_PORT}  HTTP=${ml_code}"
        fi
    else
        echo "  [MLflow] ⚠️  mlflow コマンド未検出 — スキップ"
    fi

    echo ""
}

# ═══════════════════════════════════════════════════════
# メイン処理
# ═══════════════════════════════════════════════════════

MODE="${1:-all}"

case "$MODE" in
    --status|-s)
        show_status
        exit 0
        ;;
    --api-only)
        check_api
        exit 0
        ;;
    --mlflow-only)
        check_mlflow
        exit 0
        ;;
esac

# ── 全サービス監視 ──
api_ok=true
mlflow_ok=true

check_api  || api_ok=false
check_mlflow || mlflow_ok=false

if $api_ok && $mlflow_ok; then
    local_count=$(increment_counter "${LOG_DIR}/.check_count")
    if [ $((local_count % 5)) -eq 0 ]; then
        log "OK: 全サービス正常 (API=:${API_PORT}, MLflow=:${MLFLOW_PORT}, check#${local_count})"
    fi
fi
