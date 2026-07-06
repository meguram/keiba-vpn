#!/usr/bin/env bash
# ================================================================
# keiba-vpn 全サービス監視 & 自動再起動スクリプト
#
# 監視対象:
#   1) keiba-vpn FastAPI サーバー (port 8000)
#   2) MLflow Tracking サーバー   (port 5000)
#   3) Flask API サーバー         (port 5100 dev / 5000 stg/prod)
#   4) Next.js フロントエンド     (port 3000 dev / 3001 stg/prod)
#
# cron から定期実行される（*/3 * * * *）。
# 各サービスのヘルスチェックを行い、ダウン時は自動再起動する。
# サービスが起動済みか（PID ファイル存在）の場合のみ監視・復旧対象とする。
#
# Usage:
#   ./scripts/server/server_watchdog.sh              # 全サービス監視
#   ./scripts/server/server_watchdog.sh --api-only   # FastAPI のみ
#   ./scripts/server/server_watchdog.sh --status     # 状態表示のみ (再起動なし)
# ================================================================

set -euo pipefail

export TZ="${TZ:-Asia/Tokyo}"

# ── 設定 ──
# PROJECT_DIR: スクリプト自身の位置から動的に解決（ハードコード不要）
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
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
MLFLOW_BACKEND_STORE="${PROJECT_DIR}/mlflow/runs/mlflow.db"
MLFLOW_ARTIFACT_ROOT="${PROJECT_DIR}/mlflow/runs/artifacts"

# ── アクティブプロファイルの読み込み ──
# service_start.sh が書き込む .service_start.profile から「dev/stg/prod」を取得し
# Flask・Next.js の起動パラメータ（ポート・スクリプト・モック）を決定する。
PROFILE_FILE="${PROJECT_DIR}/.service_start.profile"
ACTIVE_PROFILE=""
if [ -f "$PROFILE_FILE" ]; then
    ACTIVE_PROFILE=$(awk '{print $1}' "$PROFILE_FILE" 2>/dev/null || echo "")
fi

# プロファイル別デフォルト（service_start.profiles.sh と対応）
case "$ACTIVE_PROFILE" in
    stg|prod)
        FLASK_PORT="${FLASK_PORT:-5000}"
        FLASK_DEBUG_MODE=0
        FRONTEND_PORT="${FRONTEND_PORT:-3001}"
        FRONTEND_NPM_SCRIPT="dev"
        FRONTEND_USE_MOCK="false"
        ;;
    *)  # dev またはプロファイル不明（dev 扱い）
        FLASK_PORT="${FLASK_PORT:-5100}"
        FLASK_DEBUG_MODE=1
        FRONTEND_PORT="${FRONTEND_PORT:-3000}"
        FRONTEND_NPM_SCRIPT="dev"
        FRONTEND_USE_MOCK="true"
        ;;
esac

FLASK_PID_FILE="${PROJECT_DIR}/.flask.pid"
FLASK_HEALTH_URL="http://127.0.0.1:${FLASK_PORT}/api/v1/health"

FRONTEND_PID_FILE="${PROJECT_DIR}/.frontend.pid"
FRONTEND_HEALTH_URL="http://127.0.0.1:${FRONTEND_PORT}/"
NODE_ENV="${NODE_ENV:-development}"

KEIBA_API_URL="${KEIBA_API_URL:-http://127.0.0.1:${FLASK_PORT}}"

# Python / MLflow: PATH から動的に解決
PYTHON="${KEIBA_PYTHON:-$(which python3 2>/dev/null || echo /usr/bin/python3)}"
MLFLOW_CMD="${KEIBA_MLFLOW:-$(which mlflow 2>/dev/null || echo "")}"

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
    kill_by_pattern "uvicorn.*src\.api\.app:app"
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
# Flask API サーバー
# ═══════════════════════════════════════════════════════

# PID ファイルが存在する場合のみ監視対象とする（service_start が起動した場合のみ）
flask_should_monitor() {
    [ -f "$FLASK_PID_FILE" ]
}

stop_flask() {
    kill_by_pid_file "$FLASK_PID_FILE"
    kill_by_pattern "main\.py.*--flask-api"
}

start_flask() {
    local log_file="${LOG_DIR}/flask_$(date +%Y%m%d_%H%M%S).log"
    log "[Flask] サーバー起動中 (port=${FLASK_PORT} debug=${FLASK_DEBUG_MODE}) ログ=${log_file}"
    cd "$PROJECT_DIR"

    FLASK_PORT="$FLASK_PORT" \
    FLASK_DEBUG="$FLASK_DEBUG_MODE" \
    KEIBA_API_URL="$KEIBA_API_URL" \
    nohup "$PYTHON" main.py --flask-api >> "$log_file" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$FLASK_PID_FILE"

    local secs
    secs=$(wait_for_health "$FLASK_HEALTH_URL" 25) || true
    if [ "$secs" != "0" ]; then
        log "[Flask] 起動完了 (PID=$new_pid, ${secs}秒)"
        return 0
    fi
    log "[Flask] WARNING: 起動したが応答確認できず (PID=$new_pid)"
    return 1
}

check_flask() {
    flask_should_monitor || return 0  # PID ファイルがなければスキップ

    local code
    code=$(http_check "$FLASK_HEALTH_URL")
    if [ "$code" = "200" ]; then
        reset_counter "${LOG_DIR}/.flask_fail_count"
        return 0
    fi

    # PID が生きているか確認
    local flask_pid=""
    if [ -f "$FLASK_PID_FILE" ]; then flask_pid=$(cat "$FLASK_PID_FILE" 2>/dev/null || echo ""); fi
    if is_pid_alive "$flask_pid"; then
        # プロセスは生きているが応答なし → まだ起動中かもしれない
        local fail_count
        fail_count=$(increment_counter "${LOG_DIR}/.flask_fail_count")
        if [ "$fail_count" -lt 3 ]; then
            log "[Flask] 応答なし (HTTP=${code}, 連続${fail_count}回) — プロセスは生存中、様子見"
            return 0
        fi
    fi

    log "[Flask] ALERT: 応答なし (HTTP=${code})"
    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.flask_fail_count")

    if [ "$fail_count" -ge 5 ]; then
        log "[Flask] ERROR: ${fail_count}回連続失敗 — 30秒待機後に再起動"
        sleep 30
    fi

    stop_flask
    if start_flask; then
        reset_counter "${LOG_DIR}/.flask_fail_count"
        log "[Flask] 再起動成功"
    else
        log "[Flask] ERROR: 再起動後もヘルスチェック失敗"
    fi
    return 1
}

# ═══════════════════════════════════════════════════════
# Next.js フロントエンド
# ═══════════════════════════════════════════════════════

frontend_should_monitor() {
    [ -f "$FRONTEND_PID_FILE" ]
}

stop_frontend() {
    kill_by_pid_file "$FRONTEND_PID_FILE"
    kill_by_pattern "next.*dev.*-p ${FRONTEND_PORT}"
    kill_by_pattern "next.*start.*-p ${FRONTEND_PORT}"
}

start_frontend() {
    local log_file="${LOG_DIR}/frontend_$(date +%Y%m%d_%H%M%S).log"
    log "[Next.js] フロントエンド起動中 (port=${FRONTEND_PORT} script=${FRONTEND_NPM_SCRIPT} mock=${FRONTEND_USE_MOCK}) ログ=${log_file}"

    # node_modules の確認
    if [ ! -f "${PROJECT_DIR}/frontend/node_modules/next/dist/compiled/commander/index.js" ]; then
        log "[Next.js] WARNING: node_modules が未インストール — npm install を実行"
        (cd "${PROJECT_DIR}/frontend" && npm install --ignore-scripts) || {
            log "[Next.js] ERROR: npm install 失敗"
            return 1
        }
    fi

    (
        cd "${PROJECT_DIR}/frontend"
        export KEIBA_API_URL="$KEIBA_API_URL"
        if [ "$FRONTEND_USE_MOCK" = "true" ]; then
            export NEXT_PUBLIC_MOCK=true
        else
            unset NEXT_PUBLIC_MOCK 2>/dev/null || true
        fi
        export NODE_ENV="${NODE_ENV:-development}"
        nohup npm run "$FRONTEND_NPM_SCRIPT" -- -p "$FRONTEND_PORT" >> "$log_file" 2>&1 &
        echo $! > "$FRONTEND_PID_FILE"
    )

    # Next.js の起動は遅いため待機時間を長めに設定
    local wait_sec=45
    if [ "$FRONTEND_NPM_SCRIPT" = "start" ]; then
        wait_sec=20
    fi
    local secs
    secs=$(wait_for_health "$FRONTEND_HEALTH_URL" "$wait_sec") || true
    local front_pid
    front_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo "?")
    if [ "$secs" != "0" ]; then
        log "[Next.js] 起動完了 (PID=${front_pid}, ${secs}秒)"
        return 0
    fi
    log "[Next.js] WARNING: 起動したが応答確認できず (PID=${front_pid})"
    return 1
}

check_frontend() {
    frontend_should_monitor || return 0  # PID ファイルがなければスキップ

    local code
    code=$(http_check "$FRONTEND_HEALTH_URL")
    if [[ "$code" =~ ^(200|304)$ ]]; then
        reset_counter "${LOG_DIR}/.frontend_fail_count"
        return 0
    fi

    # PID が生きているか確認
    local front_pid=""
    if [ -f "$FRONTEND_PID_FILE" ]; then front_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo ""); fi
    if is_pid_alive "$front_pid"; then
        local fail_count
        fail_count=$(increment_counter "${LOG_DIR}/.frontend_fail_count")
        if [ "$fail_count" -lt 4 ]; then
            log "[Next.js] 応答なし (HTTP=${code}, 連続${fail_count}回) — 起動中の可能性あり、様子見"
            return 0
        fi
    fi

    log "[Next.js] ALERT: 応答なし (HTTP=${code})"
    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.frontend_fail_count")

    if [ "$fail_count" -ge 5 ]; then
        log "[Next.js] ERROR: ${fail_count}回連続失敗 — 30秒待機後に再起動"
        sleep 30
    fi

    stop_frontend
    if start_frontend; then
        reset_counter "${LOG_DIR}/.frontend_fail_count"
        log "[Next.js] 再起動成功"
    else
        log "[Next.js] ERROR: 再起動後もヘルスチェック失敗"
    fi
    return 1
}

# ═══════════════════════════════════════════════════════
# ステータス表示
# ═══════════════════════════════════════════════════════

show_status() {
    echo "=== keiba-vpn サービス状態 ==="
    echo "  アクティブプロファイル: ${ACTIVE_PROFILE:-不明}"
    echo ""

    # API
    local api_code
    api_code=$(http_check "$API_HEALTH_URL")
    local api_pid=""
    if [ -f "$API_PID_FILE" ]; then api_pid=$(cat "$API_PID_FILE" 2>/dev/null || echo ""); fi
    if [ "$api_code" = "200" ]; then
        local body
        body=$(curl -sf -m 5 "$API_HEALTH_URL" 2>/dev/null || echo "{}")
        echo "  [FastAPI] ✅ 稼働中  port=${API_PORT}  PID=${api_pid:-?}  ${body}"
    else
        echo "  [FastAPI] ❌ 停止    port=${API_PORT}  HTTP=${api_code}"
    fi

    # Flask
    local flask_code
    flask_code=$(http_check "$FLASK_HEALTH_URL")
    local flask_pid=""
    if [ -f "$FLASK_PID_FILE" ]; then flask_pid=$(cat "$FLASK_PID_FILE" 2>/dev/null || echo ""); fi
    local flask_monitor_label=""
    if flask_should_monitor; then
        flask_monitor_label=" [監視中]"
    else
        flask_monitor_label=" [未起動/監視外]"
    fi
    if [ "$flask_code" = "200" ]; then
        echo "  [Flask]   ✅ 稼働中  port=${FLASK_PORT}  PID=${flask_pid:-?}${flask_monitor_label}"
    else
        echo "  [Flask]   ❌ 停止    port=${FLASK_PORT}  HTTP=${flask_code}${flask_monitor_label}"
    fi

    # Next.js
    local front_code
    front_code=$(http_check "$FRONTEND_HEALTH_URL")
    local front_pid=""
    if [ -f "$FRONTEND_PID_FILE" ]; then front_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo ""); fi
    local front_monitor_label=""
    if frontend_should_monitor; then
        front_monitor_label=" [監視中]"
    else
        front_monitor_label=" [未起動/監視外]"
    fi
    if [[ "$front_code" =~ ^(200|304)$ ]]; then
        echo "  [Next.js] ✅ 稼働中  port=${FRONTEND_PORT}  PID=${front_pid:-?}${front_monitor_label}"
    else
        echo "  [Next.js] ❌ 停止    port=${FRONTEND_PORT}  HTTP=${front_code}${front_monitor_label}"
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
            echo "  [MLflow]  ✅ 稼働中  port=${MLFLOW_PORT}  PID=${ml_pid:-?}"
        else
            echo "  [MLflow]  ❌ 停止    port=${MLFLOW_PORT}  HTTP=${ml_code}"
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
    --flask-only)
        check_flask
        exit 0
        ;;
    --frontend-only)
        check_frontend
        exit 0
        ;;
    --mlflow-only)
        check_mlflow
        exit 0
        ;;
esac

# ── 全サービス監視 ──
api_ok=true
flask_ok=true
frontend_ok=true
mlflow_ok=true

check_api      || api_ok=false
check_flask    || flask_ok=false
check_frontend || frontend_ok=false
check_mlflow   || mlflow_ok=false

if $api_ok && $flask_ok && $frontend_ok && $mlflow_ok; then
    local_count=$(increment_counter "${LOG_DIR}/.check_count")
    if [ $((local_count % 5)) -eq 0 ]; then
        log "OK: 全サービス正常 (FastAPI=:${API_PORT}, Flask=:${FLASK_PORT}, Next.js=:${FRONTEND_PORT}, profile=${ACTIVE_PROFILE:-?}, check#${local_count})"
    fi
fi
