#!/usr/bin/env bash
# 追走難度 v2: 前夜に保存した予測を、直近の開催日の実測と突き合わせて日次・週次の JSON を作る（読み取りと data/local への書き込みのみ。netkeiba へはアクセスしない）。
# 既定は直近 10 日（土日の結果が確定した月曜以降に実行）。結果: data/local/inference/tracking_v2_eval/{YYYYMMDD}.json、weekly/{週の月曜}.json、_trend.json
# 自動実行は KEIBA_TRACKING_V2_EVAL=1 のときだけ（金曜の weekly-update 後にも同じ処理が走る）。このスクリプトは手動・cron 用で、環境変数に依らず実行する。
# Usage: bash scripts/cron/tracking_v2_weekly.sh [PROJECT_DIR] [LOOKBACK_DAYS]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
LOOKBACK="${2:-10}"
cd "$PROJECT_DIR" || exit 1
export TZ="${TZ:-Asia/Tokyo}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
TO="$(date +%Y%m%d)"
FROM="$(date -d "${LOOKBACK} days ago" +%Y%m%d)"
nice -n 19 "$PY" -m src.scripts.data.tracking_v2_eval --from "$FROM" --to "$TO"
rc=$?
echo "tracking_v2 weekly: ${FROM}〜${TO} rc=${rc}"
exit "$rc"
