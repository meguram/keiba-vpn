# shellcheck shell=bash
# service_start.sh 用 — 実行プロファイル（dev / stg / prod）
#
# dev … モック UI のみ（:3000）。実 API は不要。
# stg … 本 PC 上の本番相当（Flask :5000 / FastAPI :8000 / Next :3001 + KEIBA_ENV=stg）
# prod … stg と同構成（KEIBA_ENV=prod 想定）。将来 VPS へ stg をクローン。
#
# 環境変数で上書き可能（例: FLASK_PORT=5200 ./service_start --env dev --full）
# ローカル上書き: scripts/server/service_start.local.env（git 管理外・任意）

apply_service_profile() {
  local profile="${1:-dev}"

  SERVICE_PROFILE="$profile"
  FASTAPI_MODE=dev
  FLASK_DEBUG=0
  FRONTEND_NPM_SCRIPT=dev
  FRONTEND_USE_MOCK=false
  MLFLOW_PORT=5000
  MLFLOW_HOST=127.0.0.1
  START_MLFLOW_LOCAL=false
  PROFILE_EXPORT_KEYS=()
  TCPEXPOSER_PROFILE=""

  case "$profile" in
    dev)
      # UI モック確認専用（tcpexposer: meguai-dev → :3000）
      FASTAPI_MODE=dev
      FLASK_DEBUG=1
      FRONTEND_NPM_SCRIPT=dev
      FRONTEND_USE_MOCK=true
      : "${PORT:=8000}"
      : "${FLASK_PORT:=5100}"
      FRONTEND_PORT="${FRONTEND_PORT:-3000}"
      : "${NODE_ENV:=development}"
      TCPEXPOSER_PROFILE=dev
      START_MLFLOW_LOCAL=false
      ;;
    stg)
      # 本番相当（本 PC）。prod 将来は VPS へ同構成をデプロイ。
      FASTAPI_MODE=prod
      FLASK_DEBUG=0
      # PC stg は next dev（実 API）。VPS prod は npm run start。
      FRONTEND_NPM_SCRIPT=dev
      FRONTEND_USE_MOCK=false
      : "${PORT:=8000}"
      : "${FLASK_PORT:=5000}"
      FRONTEND_PORT="${FRONTEND_PORT:-3001}"
      : "${NODE_ENV:=development}"
      START_MLFLOW_LOCAL="${START_MLFLOW_LOCAL:-false}"
      PROFILE_EXPORT_KEYS=(KEIBA_ENV KEIBA_DEPLOYMENT_LABEL)
      KEIBA_ENV=stg
      KEIBA_DEPLOYMENT_LABEL="${KEIBA_DEPLOYMENT_LABEL:-STG}"
      TCPEXPOSER_PROFILE=stg
      ;;
    prod)
      FASTAPI_MODE=prod
      FLASK_DEBUG=0
      FRONTEND_NPM_SCRIPT=start
      FRONTEND_USE_MOCK=false
      : "${PORT:=8000}"
      : "${FLASK_PORT:=5000}"
      FRONTEND_PORT="${FRONTEND_PORT:-3001}"
      : "${NODE_ENV:=production}"
      START_MLFLOW_LOCAL="${START_MLFLOW_LOCAL:-false}"
      TCPEXPOSER_PROFILE=prod
      ;;
    *)
      echo "[service_start] 不明なプロファイル: $profile（dev / stg / prod）" >&2
      return 1
      ;;
  esac

  KEIBA_API_URL="${KEIBA_API_URL:-http://127.0.0.1:${FLASK_PORT}}"
}

profile_exports_for_child() {
  local -a args=()
  local key
  for key in "${PROFILE_EXPORT_KEYS[@]}"; do
    args+=("$key=${!key}")
  done
  if [[ "${FLASK_DEBUG:-0}" == "1" ]]; then
    args+=(FLASK_DEBUG=1)
  fi
  if [[ -n "${NODE_ENV:-}" ]]; then
    args+=(NODE_ENV="$NODE_ENV")
  fi
  if [[ "${FRONTEND_USE_MOCK}" == "true" ]]; then
    args+=(NEXT_PUBLIC_MOCK=true)
  fi
  printf '%s\0' "${args[@]}"
}

describe_profile() {
  local mock_note=""
  if [[ "${FRONTEND_USE_MOCK}" == "true" ]]; then
    mock_note=" mock=1"
  fi
  echo "  profile=${SERVICE_PROFILE}  fastapi=${FASTAPI_MODE}  flask=:${FLASK_PORT}  frontend=:${FRONTEND_PORT} npm run ${FRONTEND_NPM_SCRIPT}${mock_note}  mlflow_local=${START_MLFLOW_LOCAL}"
}
