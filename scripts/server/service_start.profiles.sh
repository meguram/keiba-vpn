# shellcheck shell=bash
# service_start.sh 用 — 実行プロファイル（dev / stg / prod）
#
# dev … モック UI 確認用（Next.js モック + Flask + FastAPI）。
# stg … 本 PC 上の本番相当（KEIBA_ENV=stg）
# prod … stg と同構成（KEIBA_ENV=prod 想定）。将来 VPS へ stg をクローン。
#
# ポート番号は config/ports.env が唯一の正（scripts/server/ports.sh で読み込み）。
# 環境変数で上書き可能（例: FLASK_PORT=23999 ./service_start --env dev --full）
# ローカル上書き: scripts/server/service_start.local.env（git 管理外・任意）

# shellcheck source=ports.sh
source "$(dirname "${BASH_SOURCE[0]}")/ports.sh"
load_keiba_ports "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

apply_service_profile() {
  local profile="${1:-dev}"

  SERVICE_PROFILE="$profile"
  FASTAPI_MODE=dev
  FLASK_DEBUG=0
  FRONTEND_NPM_SCRIPT=dev
  FRONTEND_USE_MOCK=false
  MLFLOW_PORT="$(keiba_port "$profile" mlflow)"
  MLFLOW_HOST=127.0.0.1
  START_MLFLOW_LOCAL=false
  PROFILE_EXPORT_KEYS=()
  TCPEXPOSER_PROFILE=""

  case "$profile" in
    dev)
      # dev: dev ストアのデータを表示（tcpexposer: meguai-dev）
      FASTAPI_MODE=dev
      FLASK_DEBUG=1
      FRONTEND_NPM_SCRIPT=dev
      # 既定は dev ストア（data/dev_mock = 実データ取り込み + モック）をバックエンド経由で表示。
      # フロント内蔵のモックだけで見たいときは KEIBA_DEV_UI_MOCK=1
      if [[ "${KEIBA_DEV_UI_MOCK:-0}" == "1" ]]; then FRONTEND_USE_MOCK=true; else FRONTEND_USE_MOCK=false; fi
      : "${NEXT_PUBLIC_PICKER_PAST_DAYS:=150}"
      : "${NEXT_PUBLIC_DEFAULT_RACE_DATE:=20260906}"
      # dev はこの PC からの直アクセスを自動でログイン済みにする（公開トンネル経由は対象外）。止めるなら KEIBA_DEV_AUTO_LOGIN=0
      if [[ "${KEIBA_DEV_AUTO_LOGIN:-1}" == "0" ]]; then NEXT_PUBLIC_DEV_AUTO_LOGIN=0; else NEXT_PUBLIC_DEV_AUTO_LOGIN=1; fi
      # .env が KEIBA_ENV=stg の PC（運用 cron が動くホスト）でも、dev のサーバは必ず dev（GCP 遮断・dev ストア）で動かす
      KEIBA_ENV=dev
      PROFILE_EXPORT_KEYS=(KEIBA_ENV NEXT_PUBLIC_PICKER_PAST_DAYS NEXT_PUBLIC_DEFAULT_RACE_DATE NEXT_PUBLIC_DEV_AUTO_LOGIN)
      : "${PORT:=$KEIBA_PORT_DEV_FASTAPI}"
      : "${FLASK_PORT:=$KEIBA_PORT_DEV_FLASK}"
      FRONTEND_PORT="${FRONTEND_PORT:-$KEIBA_PORT_DEV_FRONTEND}"
      : "${NODE_ENV:=development}"
      TCPEXPOSER_PROFILE=dev
      START_MLFLOW_LOCAL=false
      ;;
    stg)
      # 本番相当（本 PC）。prod 将来は VPS へ同構成をデプロイ。
      FASTAPI_MODE=prod
      FLASK_DEBUG=0
      # PC stg は本番ビルド（next build + next start）が既定。従来の next dev に戻すときは KEIBA_STG_NEXT_MODE=dev。
      # （server_watchdog.sh --profile stg も同じ変数を見る。ビルドは起動時に必要なら自動実行）
      FRONTEND_USE_MOCK=false
      : "${PORT:=$KEIBA_PORT_STG_FASTAPI}"
      : "${FLASK_PORT:=$KEIBA_PORT_STG_FLASK}"
      FRONTEND_PORT="${FRONTEND_PORT:-$KEIBA_PORT_STG_FRONTEND}"
      if [[ "${KEIBA_STG_NEXT_MODE:-prod}" == "dev" ]]; then
        FRONTEND_NPM_SCRIPT=dev
        NODE_ENV=development
      else
        FRONTEND_NPM_SCRIPT=start
        NODE_ENV=production
      fi
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
      : "${PORT:=$KEIBA_PORT_PROD_FASTAPI}"
      : "${FLASK_PORT:=$KEIBA_PORT_PROD_FLASK}"
      FRONTEND_PORT="${FRONTEND_PORT:-$KEIBA_PORT_PROD_FRONTEND}"
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
  KEIBA_FASTAPI_URL="${KEIBA_FASTAPI_URL:-http://127.0.0.1:${PORT}}"
  NEXT_PUBLIC_MONITOR_URL="${NEXT_PUBLIC_MONITOR_URL:-http://localhost:${KEIBA_PORT_MONITOR}}"
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
  args+=(KEIBA_API_URL="$KEIBA_API_URL" KEIBA_FASTAPI_URL="$KEIBA_FASTAPI_URL" NEXT_PUBLIC_MONITOR_URL="$NEXT_PUBLIC_MONITOR_URL")
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
