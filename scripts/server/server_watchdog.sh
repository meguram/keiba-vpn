#!/usr/bin/env bash
# ================================================================
# keiba-vpn 全サービス監視 & 自動再起動スクリプト
#
# 監視対象:
#   1) keiba-vpn FastAPI サーバー
#   2) MLflow Tracking サーバー
#   3) Flask API サーバー
#   4) Next.js フロントエンド
#   ※ ポートは config/ports.env（プロファイルごと）
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

# cron（PATH が限定的）や sudo -n（secure_path）でも npm / node が見えるように補う（先頭には足さない）
for _d in /opt/conda/bin /usr/local/bin; do
    if [ -d "$_d" ]; then
        case ":${PATH}:" in *":${_d}:"*) ;; *) PATH="${PATH}:${_d}" ;; esac
    fi
done
export PATH

# ── CLI 引数パース（--profile / --mode 等）──
# --profile dev|stg|prod  → .service_start.profile を上書きして指定プロファイルを強制
PROFILE_OVERRIDE=""
_PASSTHROUGH_ARGS=()
_skip_next=false
for _arg in "$@"; do
    if $_skip_next; then
        PROFILE_OVERRIDE="$_arg"
        _skip_next=false
        continue
    fi
    case "$_arg" in
        --profile=*) PROFILE_OVERRIDE="${_arg#--profile=}" ;;
        --profile)   _skip_next=true ;;
        *)           _PASSTHROUGH_ARGS+=("$_arg") ;;
    esac
done
set -- "${_PASSTHROUGH_ARGS[@]+"${_PASSTHROUGH_ARGS[@]}"}"

# ── 設定 ──
# PROJECT_DIR: スクリプト自身の位置から動的に解決（ハードコード不要）
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
MAX_LOG_LINES=5000

# ポート定義の正は config/ports.env（環境変数 KEIBA_PORT_* で上書き可）
# shellcheck source=ports.sh
source "${PROJECT_DIR}/scripts/server/ports.sh"
load_keiba_ports "$PROJECT_DIR"

# ── アクティブプロファイルの読み込み ──
# --profile 引数があればそちらを優先、なければ .service_start.profile から読む。
PROFILE_FILE="${PROJECT_DIR}/.service_start.profile"
if [ -n "$PROFILE_OVERRIDE" ]; then
    ACTIVE_PROFILE="$PROFILE_OVERRIDE"
elif [ -f "$PROFILE_FILE" ]; then
    ACTIVE_PROFILE=$(awk '{print $1}' "$PROFILE_FILE" 2>/dev/null || echo "")
else
    ACTIVE_PROFILE=""
fi

# プロファイル別デフォルト（service_start.profiles.sh と対応）
case "$ACTIVE_PROFILE" in
    stg|prod)
        PROFILE_KEY="$ACTIVE_PROFILE"
        FLASK_PORT="${FLASK_PORT:-$(keiba_port "$PROFILE_KEY" flask)}"
        FLASK_DEBUG_MODE=0
        FRONTEND_PORT="${FRONTEND_PORT:-$(keiba_port "$PROFILE_KEY" frontend)}"
        FRONTEND_NPM_SCRIPT="dev"
        FRONTEND_USE_MOCK="false"
        # stg/prod 用環境変数（.env.${ACTIVE_PROFILE} が存在すれば読み込む）
        if [ -f "${PROJECT_DIR}/.env.${ACTIVE_PROFILE}" ]; then
            set -a; source "${PROJECT_DIR}/.env.${ACTIVE_PROFILE}"; set +a
        fi
        # stg は本番ビルド（next build + next start）が既定。従来の next dev に戻すときは KEIBA_STG_NEXT_MODE=dev
        # （環境変数、または .env.stg に書いて watchdog を再起動）。prod プロファイルは従来どおり。
        if [ "$ACTIVE_PROFILE" = "stg" ] && [ "${KEIBA_STG_NEXT_MODE:-prod}" != "dev" ]; then
            FRONTEND_NPM_SCRIPT="start"
        fi
        ;;
    *)  # dev またはプロファイル不明（dev 扱い）
        PROFILE_KEY=dev
        FLASK_PORT="${FLASK_PORT:-$KEIBA_PORT_DEV_FLASK}"
        FLASK_DEBUG_MODE=1
        FRONTEND_PORT="${FRONTEND_PORT:-$KEIBA_PORT_DEV_FRONTEND}"
        FRONTEND_NPM_SCRIPT="dev"
        FRONTEND_USE_MOCK="true"
        ;;
esac

# next build の要否判定（BUILD_ID・入力の更新時刻・ビルド時環境のシグネチャ）
# shellcheck source=frontend_build.sh
source "${PROJECT_DIR}/scripts/server/frontend_build.sh"

# サービス定義（プロファイル別ポート）
API_PORT="$(keiba_port "$PROFILE_KEY" fastapi)"
API_PID_FILE="${PROJECT_DIR}/.server-${API_PORT}.pid"
API_HEALTH_URL="http://127.0.0.1:${API_PORT}/api/health"

MLFLOW_PORT="$(keiba_port "$PROFILE_KEY" mlflow)"
MLFLOW_PID_FILE="${PROJECT_DIR}/.mlflow.pid"
MLFLOW_HEALTH_URL="http://127.0.0.1:${MLFLOW_PORT}/health"
MLFLOW_BACKEND_STORE="${PROJECT_DIR}/mlflow/runs/mlflow.db"
MLFLOW_ARTIFACT_ROOT="${PROJECT_DIR}/mlflow/runs/artifacts"

# --profile が指定された場合は PID/カウンタファイルをプロファイル別にする
# （複数プロファイルを同時に watchdog で監視しても競合しないように）
if [ -n "$PROFILE_OVERRIDE" ]; then
    PROFILE_SUFFIX="-${ACTIVE_PROFILE}"
else
    PROFILE_SUFFIX=""
fi

FLASK_PID_FILE="${PROJECT_DIR}/.flask${PROFILE_SUFFIX}.pid"
FLASK_HEALTH_URL="http://127.0.0.1:${FLASK_PORT}/api/v1/health"

FRONTEND_PID_FILE="${PROJECT_DIR}/.frontend${PROFILE_SUFFIX}.pid"
FRONTEND_HEALTH_URL="http://127.0.0.1:${FRONTEND_PORT}/"
NODE_ENV="${NODE_ENV:-development}"
if [ "$FRONTEND_NPM_SCRIPT" = "start" ]; then
    NODE_ENV=production
fi

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

    env ${KEIBA_ENV:+KEIBA_ENV="$KEIBA_ENV"} nohup "$PYTHON" main.py --port "$API_PORT" --prod \
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
        reset_counter "${LOG_DIR}/.api_fail_count"
        return 0
    fi

    # 5xx サーバーエラーは即時再起動
    if [[ "$code" =~ ^5 ]]; then
        log "[API] ALERT: サーバーエラー (HTTP=${code}) — 即時再起動"
        stop_api
        if start_api; then
            reset_counter "${LOG_DIR}/.api_fail_count"
            log "[API] 再起動成功"
        else
            log "[API] ERROR: 再起動後もヘルスチェック失敗"
        fi
        return 1
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
    # プロファイル指定がある場合はプロファイル別 PID ファイルのみを参照
    # （旧 .flask.pid への自動フォールバックは行わない: 別プロファイルの PID を誤殺するため）
    [ -f "$FLASK_PID_FILE" ]
}

stop_flask() {
    kill_by_pid_file "$FLASK_PID_FILE"
    # プロファイル指定がある場合は当該ポートを使っているプロセスのみ停止する
    # （kill_by_pattern は全 --flask-api プロセスを停止するため、別プロファイルを誤殺しない）
    if [ -n "$PROFILE_SUFFIX" ]; then
        local port_pid
        port_pid=$(fuser -n tcp "${FLASK_PORT}" 2>/dev/null | tr -d ' \n' || true)
        if [ -n "$port_pid" ]; then
            log "  Flask ポート ${FLASK_PORT} の残存プロセスを停止 (PID=$port_pid)"
            kill "$port_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$port_pid" 2>/dev/null || true
        fi
    else
        kill_by_pattern "main\.py.*--flask-api"
    fi
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
        reset_counter "${LOG_DIR}/.flask_fail_count${PROFILE_SUFFIX}"
        return 0
    fi

    # 5xx サーバーエラー → 即時再起動（起動中猶予なし）
    if [[ "$code" =~ ^5 ]]; then
        log "[Flask${PROFILE_SUFFIX}] ALERT: サーバーエラー (HTTP=${code}) — 即時再起動"
        stop_flask
        if start_flask; then
            reset_counter "${LOG_DIR}/.flask_fail_count${PROFILE_SUFFIX}"
            log "[Flask${PROFILE_SUFFIX}] 再起動成功"
        else
            log "[Flask${PROFILE_SUFFIX}] ERROR: 再起動後もヘルスチェック失敗"
        fi
        return 1
    fi

    # PID が生きているか確認
    local flask_pid=""
    if [ -f "$FLASK_PID_FILE" ]; then flask_pid=$(cat "$FLASK_PID_FILE" 2>/dev/null || echo ""); fi
    if is_pid_alive "$flask_pid"; then
        # プロセスは生きているが応答なし → まだ起動中かもしれない
        local fail_count
        fail_count=$(increment_counter "${LOG_DIR}/.flask_fail_count${PROFILE_SUFFIX}")
        if [ "$fail_count" -lt 3 ]; then
            log "[Flask${PROFILE_SUFFIX}] 応答なし (HTTP=${code}, 連続${fail_count}回) — プロセスは生存中、様子見"
            return 0
        fi
    fi

    log "[Flask${PROFILE_SUFFIX}] ALERT: 応答なし (HTTP=${code})"
    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.flask_fail_count${PROFILE_SUFFIX}")

    if [ "$fail_count" -ge 5 ]; then
        log "[Flask${PROFILE_SUFFIX}] ERROR: ${fail_count}回連続失敗 — 30秒待機後に再起動"
        sleep 30
    fi

    stop_flask
    if start_flask; then
        reset_counter "${LOG_DIR}/.flask_fail_count${PROFILE_SUFFIX}"
        log "[Flask${PROFILE_SUFFIX}] 再起動成功"
    else
        log "[Flask${PROFILE_SUFFIX}] ERROR: 再起動後もヘルスチェック失敗"
    fi
    return 1
}

# ═══════════════════════════════════════════════════════
# Next.js フロントエンド
# ═══════════════════════════════════════════════════════

frontend_should_monitor() {
    # プロファイル指定がある場合はプロファイル別 PID ファイルのみを参照
    # （旧 .frontend.pid への自動フォールバックは行わない: 別プロファイルの PID を誤殺するため）
    # PID ファイルが無くても、ビルド失敗/高負荷待ちで起動できていない目印があれば監視を続ける
    # （目印を PID ファイルに書くと、プロファイル指定の有無で別ファイルになり復旧判定が壊れるため、別ファイルにしている）
    [ -f "$FRONTEND_PID_FILE" ] || [ -f "$FRONTEND_WANT_FILE" ]
}

# フロントエンドを確実に止める（ss / netstat / lsof に依存しない。/proc で子孫とポートの持ち主を特定）。
# 親（npm / sh）だけを止めても、子の next-server が残ってポートを掴み続け、新ビルドに切り替わらないことがあった。
# ポートが空かなければ 1 を返す（呼び出し側は起動・入れ替えを中止すること）。
stop_frontend() {
    frontend_stop_all "$FRONTEND_PORT" "${PROJECT_DIR}/frontend" "$FRONTEND_PID_FILE"
}

# 本番ビルド運用（stg）。ビルドは別ディレクトリ（.next-stg-build）に作り、動いているサーバを止めない。
# 切り替えは『next を止める → .next-stg を退避 → ビルドを .next-stg に入れ替え → next start』の数秒。
# ロック・目印は、プロファイル指定の有無（PID ファイル名が変わる）に関係なく共有するため PROFILE_KEY で名前を付ける。
FRONTEND_LOCK_FILE="${LOG_DIR}/.frontend-build-${PROFILE_KEY}.lock"
FRONTEND_FAIL_STAMP="${LOG_DIR}/.frontend-build-failed-${PROFILE_KEY}"
FRONTEND_WANT_FILE="${LOG_DIR}/.frontend-want-${PROFILE_KEY}"
FRONTEND_LAST_BUILD_ID=""

frontend_lock_held_by_other() {
    # 他のプロセスがロック中なら 0（真）。自分は取得して即解放するだけ。
    [ "$FRONTEND_NPM_SCRIPT" = "start" ] || return 1
    (
        exec 8>"$FRONTEND_LOCK_FILE"
        flock -n 8
    ) && return 1
    return 0
}

frontend_dist_name() {
    case "${KEIBA_ENV:-dev}" in
        stg)  echo ".next-stg"  ;;
        prod) echo ".next-prod" ;;
        *)    echo ".next-dev"  ;;
    esac
}

# 自動ビルドを見送るか（高負荷）。見送るなら 0 を返して理由を stdout に出す。KEIBA_FORCE_BUILD=1 なら見送らない。
frontend_build_deferred_for_load() {
    [ "${KEIBA_FORCE_BUILD:-0}" = "1" ] && return 1
    local load cores max
    load=$(frontend_current_load1); cores=$(nproc 2>/dev/null || echo 4)
    if [ -n "${KEIBA_BUILD_MAX_LOAD:-}" ]; then
        max="$KEIBA_BUILD_MAX_LOAD"
        awk -v l="$load" -v m="$max" 'BEGIN{exit !(l>m)}' || return 1
    else
        frontend_load_too_high "$load" "$cores" 0.8 || return 1
        max=$(awk -v c="$cores" 'BEGIN{printf "%.1f", c*0.8}')
    fi
    echo "load1=${load} が閾値 ${max} を超えている（コア数 ${cores}）"
    return 0
}

# 別ディレクトリへビルドする（稼働中のサーバには触れない）。ロックは呼び出し側が持つ。
#   戻り値 0=ビルド成功 1=失敗 2=不要 3=高負荷で見送り 4=直近の失敗で見送り
build_frontend_aside() {
    local fdir="${PROJECT_DIR}/frontend" active build_name build_path reason="" sig
    active=$(frontend_dist_name); build_name="${active}-build"; build_path="${fdir}/${build_name}"
    sig=$(frontend_env_signature)
    if [ "${KEIBA_FORCE_BUILD:-0}" != "1" ]; then
        if ! reason=$(frontend_build_needed "$fdir" "$active" "$sig"); then
            log "[Next.js] ビルド済み（${active}）— 再ビルドは不要"
            return 2
        fi
        local why
        if why=$(frontend_build_deferred_for_load); then
            log "[Next.js] 高負荷のためビルドを見送り: ${why}（${reason}）"
            return 3
        fi
        if [ -f "$FRONTEND_FAIL_STAMP" ] && [ -n "$(find "$FRONTEND_FAIL_STAMP" -mmin -10 2>/dev/null)" ]; then
            log "[Next.js] 直近 10 分以内にビルドが失敗している — 今回は再試行しません（${reason}）"
            return 4
        fi
    fi
    reason="${reason:-強制ビルド}"
    log "[Next.js] 本番ビルドを実行（別ディレクトリ ${build_name}、nice 19・並列 ${NEXT_BUILD_CPUS:-2}）: ${reason}"
    frontend_remove_dist "$build_path"
    local build_log
    build_log="${LOG_DIR}/frontend_build_$(date +%Y%m%d_%H%M%S).log"
    local t0=$SECONDS
    # ビルドは /tmp の作業コピーで行う。ホストの repo は 9p（Windows ドライブ）上にあり、小ファイルの作成が /tmp の約 300 倍遅い。
    # 9p 上で直接ビルドすると「Collecting page data」から 7 分以上進まなかった（2026-10-19）。/tmp 上では 23 秒。
    # 成果物（キャッシュを除く約 6MB・240 ファイル）だけを repo 側の ${build_name} に置く。
    local work_dir="${FRONTEND_BUILD_WORK_DIR:-/tmp/keiba-vpn-fe-build${PROFILE_SUFFIX}}"
    local work_dist
    work_dist=$(frontend_dist_name)
    if (
        set -e
        mkdir -p "$work_dir"
        for d in app components lib styles public data; do rm -rf "${work_dir:?}/$d"; done
        tar -C "$fdir" --exclude=node_modules --exclude='.next*' --exclude='*.test.mjs' -cf - . | tar -C "$work_dir" -xf -
        ln -sfn "$(readlink -f "$fdir/node_modules")" "$work_dir/node_modules"
        cd "$work_dir"
        export KEIBA_API_URL="$KEIBA_API_URL" KEIBA_ENV="${KEIBA_ENV:-dev}" NODE_ENV=production
        export NEXT_BUILD_CPUS="${NEXT_BUILD_CPUS:-2}"
        unset NEXT_DIST_DIR
        export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}"
        if [ "$FRONTEND_USE_MOCK" = "true" ]; then export NEXT_PUBLIC_MOCK=true; else unset NEXT_PUBLIC_MOCK 2>/dev/null || true; fi
        if command -v ionice >/dev/null 2>&1; then
            nice -n 19 ionice -c3 npm run build
        else
            nice -n 19 npm run build
        fi
        test -f "${work_dir}/${work_dist}/BUILD_ID"
        rm -rf "${build_path}.tmp"
        mkdir -p "${build_path}.tmp"
        tar -C "${work_dir}/${work_dist}" --exclude=cache -cf - . | tar -C "${build_path}.tmp" -xf -
        rm -rf "$build_path"
        mv "${build_path}.tmp" "$build_path"
    ) 9>&- > "$build_log" 2>&1; then
        printf '%s' "$sig" > "${build_path}/.build-env"
        rm -f "$FRONTEND_FAIL_STAMP"
        log "[Next.js] ビルド成功 ($((SECONDS - t0))秒) 出力=${build_name} ログ=${build_log}"
        return 0
    fi
    log "[Next.js] ERROR: ビルド失敗 ($((SECONDS - t0))秒) ログ=${build_log} — 稼働中の ${active} は変更していません"
    : > "$FRONTEND_FAIL_STAMP"
    tail -n 15 "$build_log" 2>/dev/null | while IFS= read -r l; do log "  | $l"; done
    frontend_remove_dist "$build_path"
    return 1
}

# 起動本体（ロックは呼び出し側が持つ）。script=start で BUILD_ID が無い場合だけ、その場でビルドする（落ちている最中なので止まる影響は無い）
_start_frontend_inner() {
    local log_file
    log_file="${LOG_DIR}/frontend_$(date +%Y%m%d_%H%M%S).log"
    log "[Next.js] フロントエンド起動中 (port=${FRONTEND_PORT} script=${FRONTEND_NPM_SCRIPT} mock=${FRONTEND_USE_MOCK}) ログ=${log_file}"

    # node_modules の確認
    if [ ! -f "${PROJECT_DIR}/frontend/node_modules/next/dist/compiled/commander/index.js" ]; then
        log "[Next.js] WARNING: node_modules が未インストール — npm install を実行"
        (cd "${PROJECT_DIR}/frontend" && npm install --ignore-scripts) || {
            log "[Next.js] ERROR: npm install 失敗"
            return 1
        }
    fi

    local keiba_env="${KEIBA_ENV:-dev}"
    local dist_dir dist_path
    dist_dir=$(frontend_dist_name)
    dist_path="${PROJECT_DIR}/frontend/${dist_dir}"
    # production ビルド残骸（required-server-files.json）が dev 起動の妨げになる場合にクリア
    if [ "$FRONTEND_NPM_SCRIPT" = "dev" ] && [ -f "${dist_path}/required-server-files.json" ]; then
        log "[Next.js] 旧 production キャッシュを検出 — ${dist_dir} をクリアします"
        frontend_remove_dist "$dist_path"
    fi

    if [ "$FRONTEND_NPM_SCRIPT" = "start" ]; then
        if [ ! -f "${dist_path}/BUILD_ID" ]; then
            # 使える本番ビルドが無い: 別ディレクトリにビルドして入れ替える（高負荷なら待つ。旧ビルドが無いので起動できない）
            local rc=0
            build_frontend_aside || rc=$?
            if [ "$rc" = "0" ]; then
                frontend_remove_dist "$dist_path"
                mv "${dist_path}-build" "$dist_path"
            else
                case "$rc" in
                    3) log "[Next.js] 使える本番ビルドが無く、高負荷でビルドも見送り — 負荷が下がるまで待機（次の周期で再試行）" ;;
                    4) log "[Next.js] 使える本番ビルドが無く、直近のビルド失敗から 10 分以内 — 待機（次の周期で再試行）" ;;
                    *) log "[Next.js] ERROR: 使える本番ビルドが無く、ビルドもできない — 起動しません（next dev へは自動で戻しません。KEIBA_STG_NEXT_MODE=dev で戻せます）" ;;
                esac
                touch "$FRONTEND_WANT_FILE"
                return 1
            fi
        else
            local why sig
            sig=$(frontend_env_signature)
            if why=$(frontend_build_needed "${PROJECT_DIR}/frontend" "$dist_dir" "$sig"); then
                log "[Next.js] 注意: 本番ビルドが最新ではない（${why}）— 旧ビルドで起動します。更新は frontend_build.sh → --switch-frontend"
            fi
        fi
    fi

    (
        cd "${PROJECT_DIR}/frontend"
        export KEIBA_API_URL="$KEIBA_API_URL"
        export KEIBA_ENV="$keiba_env"
        if [ "$FRONTEND_USE_MOCK" = "true" ]; then
            export NEXT_PUBLIC_MOCK=true
        else
            unset NEXT_PUBLIC_MOCK 2>/dev/null || true
        fi
        export NODE_ENV="${NODE_ENV:-development}"
        # 子プロセスにビルドロックの fd(9) を引き継がせない（next-server が動いている間ロックが残らないように）
        nohup npm run "$FRONTEND_NPM_SCRIPT" -- -p "$FRONTEND_PORT" >> "$log_file" 2>&1 9>&- &
        echo $! > "$FRONTEND_PID_FILE"
    )

    # Next.js の起動は遅いため待機時間を長めに設定（start はビルド済みなので短め）
    local wait_sec=45
    if [ "$FRONTEND_NPM_SCRIPT" = "start" ]; then
        wait_sec=30
    fi
    local secs front_pid
    secs=$(wait_for_health "$FRONTEND_HEALTH_URL" "$wait_sec") || true
    front_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo "?")
    if [ "$secs" != "0" ]; then
        if [ "$FRONTEND_NPM_SCRIPT" = "start" ] && [ -f "${dist_path}/BUILD_ID" ]; then
            local bid listener
            bid=$(cat "${dist_path}/BUILD_ID")
            if ! frontend_verify_build "$FRONTEND_PORT" "$bid" "$front_pid"; then
                log "[Next.js] ERROR: 起動したサーバが期待したビルド(${bid})ではない — 失敗扱い"
                stop_frontend || true
                return 1
            fi
            listener=$(frontend_listener_pids "$FRONTEND_PORT")
            FRONTEND_LAST_BUILD_ID="$bid"
            rm -f "$FRONTEND_WANT_FILE"
            log "[Next.js] 起動完了 (PID=${front_pid}, リスナーPID=${listener:-?}, BUILD_ID=${bid}, ${secs}秒)"
            return 0
        fi
        rm -f "$FRONTEND_WANT_FILE"
        log "[Next.js] 起動完了 (PID=${front_pid}, ${secs}秒)"
        return 0
    fi
    log "[Next.js] WARNING: 起動したが応答確認できず (PID=${front_pid})"
    return 1
}

start_frontend() {
    if [ "$FRONTEND_NPM_SCRIPT" != "start" ]; then
        _start_frontend_inner
        return $?
    fi
    exec 9>"$FRONTEND_LOCK_FILE"
    if ! flock -n 9; then
        log "[Next.js] 別の watchdog がビルド/切替中 — 今回は何もしません"
        return 1
    fi
    local rc=0
    _start_frontend_inner || rc=$?
    flock -u 9 2>/dev/null || true
    return $rc
}

# 明示実行: 稼働中のサーバを止めずに別ディレクトリへビルド（frontend_build.sh から呼ぶ）
build_frontend_cli() {
    if [ "$FRONTEND_NPM_SCRIPT" != "start" ]; then
        echo "このプロファイルは next dev 運用です（KEIBA_STG_NEXT_MODE=dev または stg 以外）。本番ビルドは不要です"
        return 0
    fi
    exec 9>"$FRONTEND_LOCK_FILE"
    if ! flock -n 9; then echo "別のビルド/切替が実行中です"; return 1; fi
    local rc=0
    build_frontend_aside || rc=$?
    flock -u 9 2>/dev/null || true
    case "$rc" in
        0) echo "ビルド成功: 切り替えは  bash scripts/server/server_watchdog.sh --profile ${ACTIVE_PROFILE} --switch-frontend"; return 0 ;;
        2) echo "ビルド不要（最新）。強制するには KEIBA_FORCE_BUILD=1"; return 0 ;;
        3) echo "高負荷のため見送り（logs/watchdog.log 参照）。強制するには KEIBA_FORCE_BUILD=1"; return 3 ;;
        4) echo "直近のビルド失敗から 10 分以内のため見送り。強制するには KEIBA_FORCE_BUILD=1"; return 4 ;;
        *) echo "ビルド失敗（logs/watchdog.log と logs/frontend_build_*.log を参照）"; return 1 ;;
    esac
}

# 明示実行: 別ディレクトリのビルドに入れ替えて next start（ダウンタイムは数秒）。失敗したら元に戻す。
switch_frontend() {
    if [ "$FRONTEND_NPM_SCRIPT" != "start" ]; then echo "このプロファイルは next dev 運用です。切替は不要です"; return 0; fi
    local fdir="${PROJECT_DIR}/frontend" active build_name sig
    active=$(frontend_dist_name); build_name="${active}-build"; sig=$(frontend_env_signature)
    if [ ! -f "${fdir}/${build_name}/BUILD_ID" ]; then
        echo "ビルド済みの ${build_name} がありません。先に frontend_build.sh を実行してください"; return 1
    fi
    if [ "$(cat "${fdir}/${build_name}/.build-env" 2>/dev/null)" != "$sig" ] && [ "${KEIBA_FORCE_SWITCH:-0}" != "1" ]; then
        echo "ビルド時の環境が現在と違います（API の URL など）。作り直すか KEIBA_FORCE_SWITCH=1 で強制してください"; return 1
    fi
    exec 9>"$FRONTEND_LOCK_FILE"
    if ! flock -n 9; then echo "別のビルド/切替が実行中です"; return 1; fi
    log "[Next.js] 切替開始: ${build_name} → ${active}（next を止めて入れ替え）"
    if ! stop_frontend; then
        log "[Next.js] ERROR: 旧プロセスを止められない — 切替を中止（${active} は入れ替えていません。${build_name} は残しています）"
        flock -u 9 2>/dev/null || true
        echo "旧プロセスを止められないため切替を中止しました（旧ビルドのまま。ログ参照）"
        return 1
    fi
    frontend_remove_dist "${fdir}/${active}.old"
    if [ -e "${fdir}/${active}" ]; then mv "${fdir}/${active}" "${fdir}/${active}.old"; fi
    mv "${fdir}/${build_name}" "${fdir}/${active}"
    local rc=0
    _start_frontend_inner || rc=$?
    if [ "$rc" = "0" ]; then
        frontend_remove_dist "${fdir}/${active}.old"
        log "[Next.js] 切替完了: 新プロセス PID=$(cat "$FRONTEND_PID_FILE" 2>/dev/null) リスナーPID=$(frontend_listener_pids "$FRONTEND_PORT") BUILD_ID=${FRONTEND_LAST_BUILD_ID:-?}（旧ディレクトリを削除）"
        flock -u 9 2>/dev/null || true
        echo "切替完了 (BUILD_ID=${FRONTEND_LAST_BUILD_ID:-?})"
        return 0
    fi
    log "[Next.js] ERROR: 切替後の起動/ビルド確認に失敗 — 元のディレクトリに戻します"
    stop_frontend || log "[Next.js] WARNING: 失敗したサーバを止めきれていません"
    frontend_remove_dist "${fdir}/${active}.failed"
    mv "${fdir}/${active}" "${fdir}/${active}.failed"
    if [ -e "${fdir}/${active}.old" ]; then
        mv "${fdir}/${active}.old" "${fdir}/${active}"
        _start_frontend_inner || true
    fi
    flock -u 9 2>/dev/null || true
    echo "切替に失敗したため元に戻しました（失敗した出力: ${active}.failed）"
    return 1
}

check_frontend() {
    frontend_should_monitor || return 0  # PID ファイルがなければスキップ
    if frontend_lock_held_by_other; then
        log "[Next.js${PROFILE_SUFFIX}] 別の watchdog がビルド/起動中 — 今回は監視をスキップ"
        return 0
    fi

    local code
    code=$(http_check "$FRONTEND_HEALTH_URL")
    if [[ "$code" =~ ^(200|304)$ ]]; then
        reset_counter "${LOG_DIR}/.frontend_fail_count${PROFILE_SUFFIX}"
        return 0
    fi

    # 5xx サーバーエラー → 即時再起動
    if [[ "$code" =~ ^5 ]]; then
        log "[Next.js${PROFILE_SUFFIX}] ALERT: サーバーエラー (HTTP=${code}) — 即時再起動"
        if ! stop_frontend; then log "[Next.js${PROFILE_SUFFIX}] ERROR: 旧プロセスを止められない — 起動しません"; return 1; fi
        if start_frontend; then
            reset_counter "${LOG_DIR}/.frontend_fail_count${PROFILE_SUFFIX}"
            log "[Next.js${PROFILE_SUFFIX}] 再起動成功"
        else
            log "[Next.js${PROFILE_SUFFIX}] ERROR: 再起動後もヘルスチェック失敗"
        fi
        return 1
    fi

    # PID が生きているか確認
    local front_pid=""
    if [ -f "$FRONTEND_PID_FILE" ]; then front_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo ""); fi
    if is_pid_alive "$front_pid"; then
        local fail_count
        fail_count=$(increment_counter "${LOG_DIR}/.frontend_fail_count${PROFILE_SUFFIX}")
        if [ "$fail_count" -lt 4 ]; then
            log "[Next.js${PROFILE_SUFFIX}] 応答なし (HTTP=${code}, 連続${fail_count}回) — 起動中の可能性あり、様子見"
            return 0
        fi
    fi

    log "[Next.js${PROFILE_SUFFIX}] ALERT: 応答なし (HTTP=${code})"
    local fail_count
    fail_count=$(increment_counter "${LOG_DIR}/.frontend_fail_count${PROFILE_SUFFIX}")

    if [ "$fail_count" -ge 5 ]; then
        log "[Next.js${PROFILE_SUFFIX}] ERROR: ${fail_count}回連続失敗 — 30秒待機後に再起動"
        sleep 30
    fi

    if ! stop_frontend; then log "[Next.js${PROFILE_SUFFIX}] ERROR: 旧プロセスを止められない — 起動しません"; return 1; fi
    if start_frontend; then
        reset_counter "${LOG_DIR}/.frontend_fail_count${PROFILE_SUFFIX}"
        log "[Next.js${PROFILE_SUFFIX}] 再起動成功"
    else
        log "[Next.js${PROFILE_SUFFIX}] ERROR: 再起動後もヘルスチェック失敗"
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
    --build-frontend)
        build_frontend_cli
        exit $?
        ;;
    --switch-frontend)
        switch_frontend
        exit $?
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
    local_count=$(increment_counter "${LOG_DIR}/.check_count${PROFILE_SUFFIX}")
    if [ $((local_count % 5)) -eq 0 ]; then
        log "OK: 全サービス正常 (FastAPI=:${API_PORT}, Flask=:${FLASK_PORT}, Next.js=:${FRONTEND_PORT}, profile=${ACTIVE_PROFILE:-?}, check#${local_count})"
    fi
fi
