#!/usr/bin/env bash
# 予測ファイル・出馬表の状態を GCS（ops/mirror/）へ上げる（差分だけ。Cloud Run が読む）。数分おきに cron から実行する。
# Usage: bash scripts/cron/gcs_mirror_push.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR" KEIBA_ENV="${KEIBA_ENV:-stg}"
PY="${PYTHON:-/opt/conda/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec nice -n 19 "$PY" -m src.ops.gcs_mirror push --recent-days "${MIRROR_RECENT_DAYS:-2}"
