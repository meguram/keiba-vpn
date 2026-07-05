#!/usr/bin/env bash
# keiba-vpn ローカル開発 — HTTP サービス一括起動
#
# Usage:
#   ./service_start                      # 既定: --env dev（Next.js モック UI :3001）
#   ./service_start --env dev            # 上記と同じ（モックのみ）
#   ./service_start --env dev --full     # ホットリロード開発（FastAPI + Flask + 実 API）
#   ./service_start --env stg            # 本番相当 + KEIBA_ENV=stg + tcpexposer stg
#   ./service_start --env prod           # 本番相当（FastAPI --prod / next build+start）
#   ./service_start --minimal            # FastAPI + MLflow のみ（dev 時は reload）
#   ./service_start --mock               # Next.js モックのみ
#   ./service_start --with-model-serve   # Docker MLflow model serve（:5001 等）
#   ./service_start --status             # 状態表示
#   ./service_start --help
#
# 上書き: 環境変数 / scripts/server/service_start.local.env（任意）
# ポート: dev モック :3001 / stg・prod Next.js :3000 / stg・prod Flask :5000 / dev Flask :5100

set -euo pipefail

export TZ="${TZ:-Asia/Tokyo}"

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=service_start.profiles.sh
source "$ROOT/scripts/server/service_start.profiles.sh"

if [[ -f "$ROOT/scripts/server/service_start.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/server/service_start.local.env"
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

FLASK_PID_FILE="$ROOT/.flask.pid"
FRONTEND_PID_FILE="$ROOT/.frontend.pid"
MLFLOW_PID_FILE="$ROOT/.mlflow.pid"
PROFILE_FILE="$ROOT/.service_start.profile"

# .env の秘密鍵等は bash で source しない（Python / Node が各自読み込む）。

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif [[ -x "/opt/venv/bin/python3" ]]; then
    PYTHON="/opt/venv/bin/python3"
  else
    PYTHON="$(command -v python3)"
  fi
fi

http_code() {
  local url="$1"
  local code
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || true)
  echo "${code:-000}"
}

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_open() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln | grep -q ":${port} "
    return $?
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$port" 2>/dev/null
    return $?
  fi
  return 1
}

run_with_profile_env() {
  local -a env_args=()
  local item
  while IFS= read -r -d '' item; do
    env_args+=("$item")
  done < <(profile_exports_for_child)
  env "${env_args[@]}" "$@"
}

wait_health() {
  local url="$1"
  local label="$2"
  local max="${3:-25}"
  local i=0
  while [[ "$i" -lt "$max" ]]; do
    if [[ "$(http_code "$url")" == "200" ]]; then
      echo "[service_start] OK ${label} → ${url}"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "[service_start] 警告: ${label} が ${max} 秒以内に応答しません (${url})"
  return 1
}

frontend_node_modules_ok() {
  local marker="$ROOT/frontend/node_modules/next/dist/compiled/commander/index.js"
  [[ -f "$marker" ]] || return 1
  local size
  size=$(wc -c <"$marker" 2>/dev/null || echo 0)
  [[ "$size" -gt 20000 ]]
}

frontend_dev_cache_needs_reset() {
  local next_dir="$ROOT/frontend/.next"
  [[ -d "$next_dir" ]] || return 1
  # production ビルド残骸や不完全キャッシュで next dev が 500 になるのを防ぐ
  if [[ -f "$next_dir/required-server-files.json" ]]; then
    return 0
  fi
  if [[ -f "$next_dir/routes-manifest.json" ]] && [[ ! -d "$next_dir/cache" ]]; then
    return 0
  fi
  return 1
}

prepare_frontend_dev_cache() {
  if [[ "${FRONTEND_NPM_SCRIPT:-dev}" != "dev" ]]; then
    return 0
  fi
  if frontend_dev_cache_needs_reset; then
    echo "[service_start] frontend/.next をリセット（dev 起動のため production/不完全キャッシュを削除）..."
    rm -rf "$ROOT/frontend/.next"
  fi
}

ensure_frontend_node_modules() {
  local frontend="$ROOT/frontend"
  local nm="$frontend/node_modules"
  local staging="${KEIBA_NPM_STAGING:-/tmp/keiba-vpn-npm-staging}"

  if frontend_node_modules_ok; then
    return 0
  fi

  echo "[service_start] frontend node_modules を準備中（破損検出時は ${staging} 経由）..."

  if [[ -e "$nm" ]]; then
    rm -rf "$nm"
  fi

  mkdir -p "$staging"
  if ! frontend_node_modules_ok; then
    local staging_marker="$staging/node_modules/next/dist/compiled/commander/index.js"
    if [[ ! -f "$staging_marker" ]] || [[ $(wc -c <"$staging_marker" 2>/dev/null || echo 0) -le 20000 ]]; then
      rm -rf "$staging/node_modules"
      cp "$frontend/package.json" "$staging/"
      if [[ -f "$frontend/package-lock.json" ]]; then
        cp "$frontend/package-lock.json" "$staging/"
      fi
      (cd "$staging" && npm install --ignore-scripts)
    fi
    ln -s "$staging/node_modules" "$nm"
  fi

  if ! frontend_node_modules_ok; then
    echo "[service_start] エラー: frontend/node_modules の準備に失敗しました" >&2
    return 1
  fi
}

check_postgres() {
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h localhost -p 5432 -q 2>/dev/null
    return $?
  fi
  port_open 5432
}

check_redis() {
  if command -v redis-cli >/dev/null 2>&1; then
    [[ "$(redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null || true)" == "PONG" ]]
    return $?
  fi
  port_open 6379
}

warn_infra() {
  local ok=true
  if check_postgres; then
    echo "[service_start] PostgreSQL :5432 — 接続可能"
  else
    echo "[service_start] 警告: PostgreSQL :5432 に接続できません（別途起動してください）"
    ok=false
  fi
  if check_redis; then
    echo "[service_start] Redis :6379 — 接続可能"
  else
    echo "[service_start] 警告: Redis :6379 に接続できません（別途起動してください）"
    ok=false
  fi
  $ok
}

show_status() {
  echo "=== keiba-vpn サービス状態 ==="
  if [[ -f "$PROFILE_FILE" ]]; then
    echo "  最終起動プロファイル: $(cat "$PROFILE_FILE")"
  fi
  echo ""

  local code
  code="$(http_code "http://127.0.0.1:${PORT:-8000}/api/health")"
  if [[ "$code" == "200" ]]; then
    echo "  [FastAPI]  ✅  :${PORT:-8000}  GET /api/health → 200"
  else
    echo "  [FastAPI]  ❌  :${PORT:-8000}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${FLASK_PORT:-5100}/api/v1/health")"
  if [[ "$code" == "200" ]]; then
    echo "  [Flask]    ✅  :${FLASK_PORT:-5100}  GET /api/v1/health → 200"
  else
    echo "  [Flask]    ❌  :${FLASK_PORT:-5100}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${FRONTEND_PORT:-3000}/")"
  if [[ "$code" =~ ^(200|304)$ ]]; then
    echo "  [Next.js]  ✅  :${FRONTEND_PORT:-3000}"
  else
    echo "  [Next.js]  ❌  :${FRONTEND_PORT:-3000}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${MLFLOW_PORT:-5000}/health")"
  if [[ "$code" != "200" ]]; then
    code="$(http_code "http://127.0.0.1:${MLFLOW_PORT:-5000}/api/2.0/mlflow/experiments/search?max_results=1")"
  fi
  local flask_on_mlflow_port=false
  if [[ "${FLASK_PORT:-5000}" == "${MLFLOW_PORT:-5000}" ]] && \
     [[ "$(http_code "http://127.0.0.1:${FLASK_PORT}/api/v1/health")" == "200" ]]; then
    flask_on_mlflow_port=true
  fi
  if [[ "$code" == "200" ]]; then
    echo "  [MLflow]   ✅  :${MLFLOW_PORT:-5000}"
  elif $flask_on_mlflow_port; then
    echo "  [MLflow]   ⏭️  :${MLFLOW_PORT:-5000}  （Flask が使用中・MLflow ローカル未起動）"
  else
    echo "  [MLflow]   ❌  :${MLFLOW_PORT:-5000}  HTTP=${code}"
  fi

  if check_postgres; then
    echo "  [Postgres] ✅  :5432"
  else
    echo "  [Postgres] ❌  :5432"
  fi

  if check_redis; then
    echo "  [Redis]    ✅  :6379"
  else
    echo "  [Redis]    ❌  :6379"
  fi

  echo ""
  echo "  Model serve（任意）: :5001 :5002 :5003 :5004 :5005 :5006 :5010"
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -E ':(5001|5002|5003|5004|5005|5006|5010)\s' || true
  fi
  echo ""
}

start_mlflow() {
  if [[ "${START_MLFLOW_LOCAL}" != "true" ]]; then
    echo "[service_start] MLflow ローカル起動スキップ（profile=${SERVICE_PROFILE}）"
    return 0
  fi

  local health_url="http://127.0.0.1:${MLFLOW_PORT}/health"
  if [[ "$(http_code "$health_url")" == "200" ]]; then
    echo "[service_start] MLflow 既に稼働中 (:${MLFLOW_PORT})"
    return 0
  fi
  local alt_code
  alt_code="$(http_code "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1")"
  if [[ "$alt_code" == "200" ]]; then
    echo "[service_start] MLflow 既に稼働中 (:${MLFLOW_PORT})"
    return 0
  fi

  if [[ -f "$MLFLOW_PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$MLFLOW_PID_FILE" 2>/dev/null || true)"
    if is_pid_alive "$old_pid"; then
      kill "$old_pid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$MLFLOW_PID_FILE"
  fi

  mkdir -p "$ROOT/mlflow/runs/artifacts"
  local log_file="$LOG_DIR/mlflow_${TS}.log"
  echo "[service_start] MLflow 起動 (:${MLFLOW_PORT}) ログ=${log_file}"
  nohup "$PYTHON" -m mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "sqlite:///${ROOT}/mlflow/runs/mlflow.db" \
    --default-artifact-root "$ROOT/mlflow/runs/artifacts" \
    --serve-artifacts \
    >>"$log_file" 2>&1 &
  echo $! >"$MLFLOW_PID_FILE"
  wait_health "$health_url" "MLflow" 30 || \
    wait_health "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1" "MLflow" 10 || true
}

start_fastapi() {
  echo "[service_start] FastAPI (:${PORT}) mode=${FASTAPI_MODE}..."
  if [[ "$FASTAPI_MODE" == "dev" ]]; then
    run_with_profile_env env PORT="$PORT" PYTHON="$PYTHON" "$ROOT/scripts/server/restart_server.sh" dev
  else
    run_with_profile_env env PORT="$PORT" PYTHON="$PYTHON" "$ROOT/scripts/server/restart_server.sh"
  fi
}

start_flask() {
  local health_url="http://127.0.0.1:${FLASK_PORT}/api/v1/health"
  local need_restart=false

  if [[ "$(http_code "$health_url")" == "200" ]]; then
    if [[ "${FLASK_DEBUG:-0}" == "1" ]]; then
      need_restart=true
      echo "[service_start] Flask を dev 設定（FLASK_DEBUG=1）で再起動..."
    else
      echo "[service_start] Flask 既に稼働中 (:${FLASK_PORT})"
      return 0
    fi
  fi

  if [[ -f "$FLASK_PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$FLASK_PID_FILE" 2>/dev/null || true)"
    if is_pid_alive "$old_pid"; then
      kill "$old_pid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$FLASK_PID_FILE"
  fi

  if $need_restart; then
    local pids
    pids=$(pgrep -f "main\.py.*--flask-api" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      sleep 1
    fi
  fi

  local log_file="$LOG_DIR/flask_${TS}.log"
  echo "[service_start] Flask 起動 (:${FLASK_PORT} debug=${FLASK_DEBUG}) ログ=${log_file}"
  run_with_profile_env nohup env FLASK_PORT="$FLASK_PORT" "$PYTHON" main.py --flask-api >>"$log_file" 2>&1 &
  echo $! >"$FLASK_PID_FILE"
  wait_health "$health_url" "Flask" 25 || true
}

start_frontend() {
  local mock_override="${1:-}"
  local use_mock="${FRONTEND_USE_MOCK:-false}"
  if [[ "$mock_override" == "true" ]]; then
    use_mock=true
  elif [[ "$mock_override" == "false" ]]; then
    use_mock=false
  fi
  local url="http://127.0.0.1:${FRONTEND_PORT}/"

  if [[ "$(http_code "$url")" =~ ^(200|304)$ ]]; then
    echo "[service_start] Next.js 既に稼働中 (:${FRONTEND_PORT})"
    return 0
  fi

  ensure_frontend_node_modules || return 1
  prepare_frontend_dev_cache

  if [[ "$FRONTEND_NPM_SCRIPT" == "start" ]]; then
    echo "[service_start] Next.js 本番ビルド (npm run build)..."
    run_with_profile_env bash -c "cd '$ROOT/frontend' && npm run build"
  fi

  if [[ -f "$FRONTEND_PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$FRONTEND_PID_FILE" 2>/dev/null || true)"
    if is_pid_alive "$old_pid"; then
      kill "$old_pid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$FRONTEND_PID_FILE"
  fi

  local log_file="$LOG_DIR/frontend_${TS}.log"
  echo "[service_start] Next.js 起動 (npm run ${FRONTEND_NPM_SCRIPT} :${FRONTEND_PORT} mock=${use_mock}) KEIBA_API_URL=${KEIBA_API_URL} ログ=${log_file}"
  (
    cd "$ROOT/frontend"
    if [[ "$use_mock" == "true" ]]; then
      export NEXT_PUBLIC_MOCK=true
    else
      unset NEXT_PUBLIC_MOCK
    fi
    export KEIBA_API_URL="$KEIBA_API_URL"
    if [[ "$FRONTEND_NPM_SCRIPT" == "dev" ]]; then
      export NODE_ENV=development
    fi
    run_with_profile_env nohup npm run "$FRONTEND_NPM_SCRIPT" -- -p "$FRONTEND_PORT" >>"$log_file" 2>&1 &
    echo $! >"$FRONTEND_PID_FILE"
  )
  local wait_sec=40
  if [[ "$FRONTEND_NPM_SCRIPT" == "start" ]]; then
    wait_sec=20
  fi
  wait_health "$url" "Next.js" "$wait_sec" || true
}

start_model_serve_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[service_start] 警告: docker コマンドがありません — model serve をスキップ"
    return 1
  fi
  echo "[service_start] Docker MLflow model serve 起動..."
  (
    cd "$ROOT/mlflow/server"
    docker compose --profile all-models up -d
  )
}

record_profile() {
  echo "${SERVICE_PROFILE} $(date -Iseconds)" >"$PROFILE_FILE"
}

start_tcpexposer_tunnel() {
  local profile="${1:-}"
  if [[ -z "$profile" ]]; then
    return 0
  fi
  if [[ ! -x "$ROOT/scripts/server/tunnel_tcpexposer.sh" ]]; then
    return 0
  fi
  echo "[service_start] tcpexposer トンネル (${profile}) を起動..."
  bash "$ROOT/scripts/server/tunnel_tcpexposer.sh" "$profile" autostart || \
    echo "[service_start] 警告: tcpexposer ${profile} の起動に失敗（手動: ./scripts/server/tunnel_tcpexposer.sh ${profile} background）"
}

free_flask_port_if_needed() {
  if [[ "${FLASK_PORT}" != "5000" ]]; then
    return 0
  fi
  if [[ "$(http_code "http://127.0.0.1:5000/api/v1/health")" == "200" ]]; then
    return 0
  fi
  if [[ -f "$MLFLOW_PID_FILE" ]]; then
    local mpid
    mpid="$(cat "$MLFLOW_PID_FILE" 2>/dev/null || true)"
    if is_pid_alive "$mpid"; then
      echo "[service_start] Flask :5000 のため MLflow (PID=${mpid}) を停止..."
      kill "$mpid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$MLFLOW_PID_FILE"
  fi
  if [[ "$(http_code "http://127.0.0.1:5000/health")" == "200" ]] || \
     [[ "$(http_code "http://127.0.0.1:5000/api/v1/health")" != "200" ]]; then
    pkill -f "mlflow server.*--port 5000" 2>/dev/null || true
    pkill -f "uvicorn.*port 5000.*mlflow" 2>/dev/null || true
    sleep 1
  fi
}

mode_full() {
  echo "[service_start] パターン A — フルセット (${SERVICE_PROFILE})"
  describe_profile
  warn_infra || true
  start_mlflow
  free_flask_port_if_needed
  start_flask
  start_fastapi
  start_frontend false
  record_profile
  echo ""
  echo "[service_start] 起動完了。状態確認: ./service_start --status"
  echo "  FastAPI  http://127.0.0.1:${PORT}/"
  echo "  Flask    http://127.0.0.1:${FLASK_PORT}/api/v1/health"
  echo "  Next.js  http://127.0.0.1:${FRONTEND_PORT}/"
  if [[ "${START_MLFLOW_LOCAL}" == "true" ]]; then
    echo "  MLflow   http://127.0.0.1:${MLFLOW_PORT}/"
  fi
  if [[ -n "${TCPEXPOSER_PROFILE:-}" ]]; then
    start_tcpexposer_tunnel "$TCPEXPOSER_PROFILE"
    case "$TCPEXPOSER_PROFILE" in
      dev) echo "  tcpexposer  https://meguai-dev.tcpexposer.com/" ;;
      stg) echo "  tcpexposer  https://meguai-stg.tcpexposer.com/" ;;
    esac
  fi
}

mode_minimal() {
  echo "[service_start] パターン B — API + MLflow (${SERVICE_PROFILE})"
  describe_profile
  if [[ "$FASTAPI_MODE" == "dev" ]]; then
    start_mlflow
    start_fastapi
  else
    "$ROOT/scripts/server/server_watchdog.sh"
  fi
  record_profile
  echo ""
  echo "[service_start] 起動完了。状態確認: ./service_start --status"
}

mode_mock() {
  echo "[service_start] パターン C — Next.js モックのみ (${SERVICE_PROFILE})"
  describe_profile
  start_frontend true
  record_profile
  echo ""
  echo "[service_start] 起動完了 → http://127.0.0.1:${FRONTEND_PORT}/ （NEXT_PUBLIC_MOCK=true）"
  if [[ -n "${TCPEXPOSER_PROFILE:-}" ]]; then
    start_tcpexposer_tunnel "$TCPEXPOSER_PROFILE"
    case "$TCPEXPOSER_PROFILE" in
      dev) echo "  公開 URL: https://meguai-dev.tcpexposer.com/" ;;
      stg) echo "  公開 URL: https://meguai-stg.tcpexposer.com/" ;;
    esac
  fi
}

print_help() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "プロファイル:"
  echo "  dev   モック UI のみ (:3001) → meguai-dev.tcpexposer.com"
  echo "  dev --full  FastAPI reload / Flask debug / next dev（実 API 開発）"
  echo "  stg   本番相当 (:3000 Flask :5000) + KEIBA_ENV=stg → meguai-stg.tcpexposer.com"
  echo "  prod  stg と同構成（VPS デプロイ想定）"
}

MODE=""
SERVICE_PROFILE="${KEIBA_SERVICE_PROFILE:-dev}"
WITH_MODEL_SERVE=false
DEV_FULL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--profile)
      SERVICE_PROFILE="${2:?--env には dev|stg|prod を指定}"
      shift 2
      ;;
    --env=*|--profile=*)
      SERVICE_PROFILE="${1#*=}"
      shift
      ;;
    --dev)
      SERVICE_PROFILE=dev
      shift
      ;;
    --stg)
      SERVICE_PROFILE=stg
      shift
      ;;
    --prod)
      SERVICE_PROFILE=prod
      shift
      ;;
    --full)
      DEV_FULL=true
      MODE=full
      shift
      ;;
    --minimal|-m)
      MODE=minimal
      shift
      ;;
    --mock)
      MODE=mock
      shift
      ;;
    --with-model-serve)
      WITH_MODEL_SERVE=true
      shift
      ;;
    --status|-s)
      prof="${SERVICE_PROFILE:-dev}"
      if [[ -f "$PROFILE_FILE" ]]; then
        prof="$(awk '{print $1}' "$PROFILE_FILE")"
      fi
      apply_service_profile "$prof" || exit 1
      show_status
      exit 0
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "不明なオプション: $1（--help を参照）" >&2
      exit 1
      ;;
  esac
done

apply_service_profile "$SERVICE_PROFILE" || exit 1

if [[ -z "$MODE" ]]; then
  case "$SERVICE_PROFILE" in
    dev)
      if $DEV_FULL; then
        MODE=full
      else
        MODE=mock
      fi
      ;;
    stg|prod)
      MODE=full
      ;;
    *)
      MODE=full
      ;;
  esac
fi

case "$MODE" in
  full)
    mode_full
    ;;
  minimal)
    mode_minimal
    ;;
  mock)
    mode_mock
    ;;
esac

if $WITH_MODEL_SERVE; then
  start_model_serve_docker || true
fi
