#!/usr/bin/env bash
# メンテナンスの開始/終了を Slack に 1 通ずつ通知する（config/maintenance.json の notify.slack_start_end が true、かつ KEIBA_SLACK_NOTIFY=1 のときだけ）。
# Usage: bash scripts/cron/maintenance_notify.sh start|end [PROJECT_DIR]
set -u
EVENT="${1:?start|end}"; PROJECT_DIR="${2:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
export TZ="${TZ:-Asia/Tokyo}" KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=.
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m src.scripts.ops.maintenance_notify "$EVENT" --send
