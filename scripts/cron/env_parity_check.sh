#!/usr/bin/env bash
# stg と prod の実効リリース・マニフェスト（コード・モデル・config・フラグ・フロント）を照合し、意図しない差があれば Slack に通知する。
# 読み取り専用。URL は .env の KEIBA_PARITY_STG_API / KEIBA_PARITY_PROD_API（と _UI の version.json）。トークンは KEIBA_MANIFEST_TOKEN（値は表示しない）。
# 窓の自動反映（JST 03:05）の後、毎日 JST 04:10 に回す想定。終了コード 0=一致 1=差あり 2=取得不可。
# Usage: bash scripts/cron/env_parity_check.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
export TZ="${TZ:-Asia/Tokyo}" PYTHONPATH=.
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
: "${KEIBA_PARITY_STG_API:=https://stg.megu-keiba.com}"; : "${KEIBA_PARITY_PROD_API:=https://app.megu-keiba.com}"
ARGS=(compare --stg-api "$KEIBA_PARITY_STG_API" --prod-api "$KEIBA_PARITY_PROD_API" --slack)
[ -n "${KEIBA_PARITY_STG_UI:-}" ] && ARGS+=(--stg-ui "$KEIBA_PARITY_STG_UI")
[ -n "${KEIBA_PARITY_PROD_UI:-}" ] && ARGS+=(--prod-ui "$KEIBA_PARITY_PROD_UI")
"$PY" -m src.scripts.ops.env_parity_check "${ARGS[@]}"
