#!/usr/bin/env bash
# odds_drift の週次レポート: 直近 28 日の「T-15 → 最新（final など）」「T-60 → 最新」のオッズ変化率を、オッズ帯別に集計する。
# 読み取りのみ（GCS の race_odds を読む。netkeiba へはアクセスしない）。結果: data/local/meta/odds_drift/weekly/{YYYYMMDD}_{early}.json と latest_{early}.json
# Usage: bash scripts/cron/odds_drift_weekly.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
export TZ="${TZ:-Asia/Tokyo}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
TO="$(date +%Y%m%d)"
FROM="$(date -d '28 days ago' +%Y%m%d)"
OUT="data/local/meta/odds_drift/weekly"
mkdir -p "$OUT"
rc=0
for EARLY in t15 t60; do
  "$PY" -m src.pipeline.betting.odds_drift --from "$FROM" --to "$TO" --early "$EARLY" --out "${OUT}/${TO}_${EARLY}.json" || rc=$?
  [ -f "${OUT}/${TO}_${EARLY}.json" ] && cp "${OUT}/${TO}_${EARLY}.json" "data/local/meta/odds_drift/latest_${EARLY}.json"
done
echo "odds_drift weekly: ${FROM}〜${TO} rc=${rc}"
exit "$rc"
