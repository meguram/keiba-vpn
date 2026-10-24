#!/usr/bin/env bash
# 会員名簿（data/local/auth/members.json。ハッシュだけ）を GCS（ops/mirror/）へ上げる。Cloud Run がログインの照合に使う。5 分おきに root の cron から。
# 名簿はホストでは root だけが読める場所にあるので、通常の同期(jovyan)とは別。変わっていなければ何も上げない（md5 比較）。
# Usage: sudo bash scripts/cron/gcs_mirror_push_auth.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR" KEIBA_ENV="${KEIBA_ENV:-stg}"
PY="${PYTHON:-/opt/conda/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec nice -n 19 "$PY" -m src.ops.gcs_mirror push --only-auth
