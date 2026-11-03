#!/usr/bin/env bash
# keiba-vpn ローカル開発 — HTTP サービス一括起動
#
# Usage:
#   ./service_start                      # 既定: --env dev（Next.js モック UI。ポートは config/ports.env）
#   ./service_start --env dev            # 上記と同じ（モックのみ）
#   ./service_start --env dev --full     # ホットリロード開発（FastAPI + Flask + 実 API）
#   ./service_start --env prod           # 本番相当（FastAPI --prod / next build+start）
#                                        # ※ホストで動かすのは dev のみ。stg/prod は VPS・GCP の運用
#   ./service_start --minimal            # FastAPI + MLflow のみ（dev 時は reload）
#   ./service_start --mock               # Next.js モック + Flask（ログイン用）
#   ./service_start --mock --with-fastapi # 上記 + FastAPI（レガシー UI / モックデータ API）
#   ./service_start --with-model-serve   # Docker MLflow model serve（:5001 等）
#   ./service_start --monitor            # 開発者監視ポータル（KEIBA_PORT_MONITOR）も同時起動
#   ./service_start --status             # 状態表示
#   ./service_start --help
#
# 上書き: 環境変数 / scripts/server/service_start.local.env（任意）
# ポート: config/ports.env が唯一の正（ホストで動かすのは dev と監視ポータルのみ。cron 用の FastAPI / MLflow は別管理）

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

frontend_next_dist_dir() {
  case "${KEIBA_ENV:-dev}" in
    stg)  echo ".next-stg" ;;
    prod) echo ".next-prod" ;;
    *)    echo ".next-dev" ;;
  esac
}

frontend_dev_cache_needs_reset() {
  local dist_name
  dist_name="$(frontend_next_dist_dir)"
  local next_dir="$ROOT/frontend/$dist_name"
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
    local dist_name
    dist_name="$(frontend_next_dist_dir)"
    echo "[service_start] frontend/${dist_name} をリセット（dev 起動のため production/不完全キャッシュを削除）..."
    rm -rf "$ROOT/frontend/$dist_name"
  fi
  # 旧 .next（distDir 分離前の残骸）も削除
  if [[ -d "$ROOT/frontend/.next" ]]; then
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

pg_port() { echo 5432; }
redis_port() { echo 6379; }

check_postgres() {
  local port; port="$(pg_port)"
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h localhost -p "$port" -q 2>/dev/null
    return $?
  fi
  port_open "$port"
}

check_redis() {
  local port; port="$(redis_port)"
  if command -v redis-cli >/dev/null 2>&1; then
    [[ "$(redis-cli -h 127.0.0.1 -p "$port" ping 2>/dev/null || true)" == "PONG" ]]
    return $?
  fi
  port_open "$port"
}

# .env から非秘密キーだけを安全に読む（bash で source しない方針）。
# 優先順: 環境変数 > .env.<KEIBA_ENV> > .env（アプリの load_project_dotenv と同じ重ね順）
dotenv_get() {
  local key="$1" val="${!1:-}" env_name f line
  if [[ -n "$val" ]]; then
    printf '%s' "$val"
    return 0
  fi
  env_name="${KEIBA_ENV:-}"
  if [[ -z "$env_name" && -f "$ROOT/.env" ]]; then
    env_name="$(grep -E '^KEIBA_ENV=' "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
  fi
  for f in "$ROOT/.env.${env_name}" "$ROOT/.env"; do
    [[ "$f" == "$ROOT/.env." ]] && continue
    [[ -f "$f" ]] || continue
    line="$(grep -E "^${key}=" "$f" 2>/dev/null | tail -n1 || true)"
    if [[ -n "$line" ]]; then
      val="${line#*=}"
      val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
      printf '%s' "$val"
      return 0
    fi
  done
  return 0
}

# Cloud SQL 利用時（stg/prod の既定）はローカル PostgreSQL :5432 は不要。
uses_cloud_sql() {
  [[ "$(dotenv_get KEIBA_DB_BACKEND | tr '[:upper:]' '[:lower:]')" == "cloud_sql" ]]
}

# ローカルの Postgres/Redis が未起動なら docker-compose.dev.yml から自動起動する（無効化: KEIBA_AUTO_START_INFRA=0）
start_infra_docker() {
  local service="$1"
  [[ "${KEIBA_AUTO_START_INFRA:-1}" == "0" ]] && return 1
  command -v docker >/dev/null 2>&1 || return 1
  docker info >/dev/null 2>&1 || return 1
  [[ -f "$ROOT/docker-compose.dev.yml" ]] || return 1
  (cd "$ROOT" && docker compose -f docker-compose.dev.yml up -d "$service") >/dev/null 2>&1
}

wait_until() {
  local fn="$1" i
  for i in $(seq 1 30); do
    "$fn" && return 0
    sleep 1
  done
  return 1
}

warn_infra() {
  local ok=true
  if uses_cloud_sql; then
    echo "[service_start] PostgreSQL — Cloud SQL を使用（KEIBA_DB_BACKEND=cloud_sql）のためローカル :5432 は不要"
  elif check_postgres; then
    echo "[service_start] PostgreSQL :$(pg_port) — 接続可能"
  elif start_infra_docker postgres && wait_until check_postgres; then
    echo "[service_start] PostgreSQL :$(pg_port) — docker compose (docker-compose.dev.yml) で自動起動しました"
  else
    echo "[service_start] 警告: PostgreSQL :$(pg_port) に接続できません（docker compose -f docker-compose.dev.yml up -d postgres で起動してください）"
    ok=false
  fi
  if check_redis; then
    echo "[service_start] Redis :$(redis_port) — 接続可能"
  elif start_infra_docker redis && wait_until check_redis; then
    echo "[service_start] Redis :$(redis_port) — docker compose (docker-compose.dev.yml) で自動起動しました"
  else
    echo "[service_start] 警告: Redis :$(redis_port) に接続できません（docker compose -f docker-compose.dev.yml up -d redis で起動してください）"
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
  code="$(http_code "http://127.0.0.1:${PORT}/api/health")"
  if [[ "$code" == "200" ]]; then
    echo "  [FastAPI]  ✅  :${PORT}  GET /api/health → 200"
  else
    echo "  [FastAPI]  ❌  :${PORT}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${FLASK_PORT}/api/v1/health")"
  if [[ "$code" == "200" ]]; then
    echo "  [Flask]    ✅  :${FLASK_PORT}  GET /api/v1/health → 200"
  else
    echo "  [Flask]    ❌  :${FLASK_PORT}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${FRONTEND_PORT}/")"
  if [[ "$code" =~ ^(200|304)$ ]]; then
    echo "  [Next.js]  ✅  :${FRONTEND_PORT}"
  else
    echo "  [Next.js]  ❌  :${FRONTEND_PORT}  HTTP=${code}"
  fi

  code="$(http_code "http://127.0.0.1:${MLFLOW_PORT}/health")"
  if [[ "$code" != "200" ]]; then
    code="$(http_code "http://127.0.0.1:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1")"
  fi
  if [[ "$code" == "200" ]]; then
    echo "  [MLflow]   ✅  :${MLFLOW_PORT}"
  elif [[ "${START_MLFLOW_LOCAL}" != "true" ]]; then
    echo "  [MLflow]   ⏭️  :${MLFLOW_PORT}  （このプロファイルではローカル MLflow を起動しない）"
  else
    echo "  [MLflow]   ❌  :${MLFLOW_PORT}  HTTP=${code}"
  fi

  if uses_cloud_sql; then
    echo "  [Postgres] ☁️   Cloud SQL（ローカル :5432 は不要）"
  elif check_postgres; then
    echo "  [Postgres] ✅  :$(pg_port)"
  else
    echo "  [Postgres] ❌  :$(pg_port)"
  fi

  if check_redis; then
    echo "  [Redis]    ✅  :$(redis_port)"
  else
    echo "  [Redis]    ❌  :$(redis_port)"
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
    run_with_profile_env env PORT="$PORT" PYTHON="$PYTHON" bash "$ROOT/scripts/server/restart_server.sh" dev
  else
    run_with_profile_env env PORT="$PORT" PYTHON="$PYTHON" bash "$ROOT/scripts/server/restart_server.sh"
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
      # 作業ツリーが 9p（WSL の D:\ など）だとファイル変更の通知が届かず、保存しても画面に反映されないことがある。
      # 変更を定期的に見に行く方式にする（無効化: WATCHPACK_POLLING=false）
      export WATCHPACK_POLLING="${WATCHPACK_POLLING:-true}"
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

mode_full() {
  echo "[service_start] パターン A — フルセット (${SERVICE_PROFILE})"
  describe_profile
  warn_infra || true
  start_mlflow
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
      monitor) echo "  tcpexposer  https://meguai-monitor.tcpexposer.com/" ;;
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
  # ログイン (/api/v1/auth/*) は Flask 経由のため、モック UI でも dev Flask を起動する
  start_flask
  if $WITH_FASTAPI; then
    start_fastapi
  fi
  start_frontend "$FRONTEND_USE_MOCK"
  record_profile
  echo ""
  echo "[service_start] 起動完了 → http://127.0.0.1:${FRONTEND_PORT}/ （mock=${FRONTEND_USE_MOCK}。false は dev ストアのデータを表示）"
  if [[ -n "${TCPEXPOSER_PROFILE:-}" ]]; then
    start_tcpexposer_tunnel "$TCPEXPOSER_PROFILE"
    case "$TCPEXPOSER_PROFILE" in
      dev) echo "  公開 URL: https://meguai-dev.tcpexposer.com/" ;;
      monitor) echo "  公開 URL: https://meguai-monitor.tcpexposer.com/" ;;
    esac
  fi
}

print_help() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "プロファイル:"
  echo "  dev   モック UI (:${KEIBA_PORT_DEV_FRONTEND}) → meguai-dev.tcpexposer.com"
  echo "  dev --full  FastAPI reload / Flask debug / next dev（実 API 開発）"
  echo "  prod  本番ビルド構成（ホストでは通常使わない。stg/prod は VPS・GCP）"
}

MODE=""
SERVICE_PROFILE="${KEIBA_SERVICE_PROFILE:-dev}"
WITH_MODEL_SERVE=false
WITH_MONITOR=false
WITH_FASTAPI=false
DEV_FULL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--profile)
      SERVICE_PROFILE="${2:?--env には dev|prod を指定}"
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
    --with-fastapi)
      WITH_FASTAPI=true
      shift
      ;;
    --monitor)
      WITH_MONITOR=true
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
    prod)
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

if $WITH_MONITOR; then
  echo "[service_start] 監視ポータル (:${KEIBA_PORT_MONITOR}) をバックグラウンド起動..."
  bash "$ROOT/scripts/server/start_monitor.sh" --bg || \
    echo "[service_start] 警告: 監視ポータルの起動に失敗（手動: bash scripts/server/start_monitor.sh）"
  echo "[service_start] tcpexposer monitor トンネルを起動..."
  bash "$ROOT/scripts/server/tunnel_tcpexposer.sh" monitor autostart || \
    echo "[service_start] 警告: tcpexposer monitor の起動に失敗（手動: ./scripts/server/tunnel_tcpexposer.sh monitor background）"
  echo "  監視ポータル URL: https://meguai-monitor.tcpexposer.com/"
fi
