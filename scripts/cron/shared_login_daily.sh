#!/usr/bin/env bash
# 共通ログインの毎月 1 日の更新（毎日 00:05 JST に起動。月が替わっていなければ何もしない）。仕様: docs/operations/shared_login.md
# Usage: bash scripts/cron/shared_login_daily.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. "${PYTHON:-/opt/conda/bin/python3}" -m src.scripts.ops.shared_login ensure >>logs/shared_login.log 2>&1
