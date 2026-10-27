#!/usr/bin/env bash
# SporthorseData からドルメロ用の誕生月日を夜間バッチで取得する。
# 前提: data/meta/dormello/sporthorse_session.json（ブラウザで Cloudflare 通過後に登録）
#
# Usage: bash scripts/cron/sporthorse_births_nightly.sh [PROJECT_DIR]
set -euo pipefail
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"

if [ -x "/opt/conda/bin/python3" ]; then
  PYTHON="/opt/conda/bin/python3"
else
  PYTHON="$(command -v python3)"
fi

DORMELLO="${KEIBA_DORMELLO_DIR:-${PROJECT_DIR}/data/meta/dormello}"
ANCESTORS="${KEIBA_ANCESTOR_BIRTHDAYS:-/tmp/ancestor_birthdays.csv}"
NK="${KEIBA_NK_BIRTH_JSONL:-/tmp/nk_birth_results.jsonl}"
SIRES="${KEIBA_ANCESTOR_SIRES:-/tmp/ancestor_sires.json}"
DELAY="${KEIBA_SPORTHORSE_DELAY:-45}"
MAX_SEC="${KEIBA_SPORTHORSE_MAX_SECONDS:-25200}"
LIMIT="${KEIBA_SPORTHORSE_QUEUE_LIMIT:-3000}"

SIRES_ARG=()
[ -f "$SIRES" ] && SIRES_ARG=(--sires "$SIRES")

exec "$PYTHON" -m src.research.pedigree.sporthorse_births run-nightly \
  --ancestors "$ANCESTORS" \
  --zone "${DORMELLO}/zone_review.csv" \
  --nk "$NK" \
  --dir "$DORMELLO" \
  --delay "$DELAY" \
  --max-seconds "$MAX_SEC" \
  --limit "$LIMIT" \
  "${SIRES_ARG[@]}"
