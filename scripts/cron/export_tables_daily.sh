#!/usr/bin/env bash
# 推論モードの履歴（race_result_flat など）を最新に保つため、当年（1 月は前年も）のテーブルを増分エクスポートする。
# --refresh: 各カテゴリが自分の flat に無いレースを GCS から追記する（これが無いと race_result に新しいレースが入らない）。
# Usage: bash scripts/cron/export_tables_daily.sh [PROJECT_DIR] [categories]
set -u
export TZ="${TZ:-Asia/Tokyo}"
PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CATS="${2:-race_result,race_shutuba,race_result_lap,race_result_on_time}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
LOG="logs/export_tables_daily.log"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
YEAR="$(date +%Y)"
YEARS="$YEAR"
[ "$(date +%m)" = "01" ] && YEARS="$((YEAR - 1)),$YEAR"
exec 9>"/tmp/keiba_export_tables_daily.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] export_tables_daily already running, skip" >>"$LOG"
  exit 0
fi
echo "===== $(date -Is) START years=$YEARS categories=$CATS =====" >>"$LOG"
"$PYTHON" -m src.scraper.export_tables --years "$YEARS" --categories "$CATS" --refresh >>"$LOG" 2>&1
EC=$?
# パース失敗の検査（必須項目の欠損）。ページ表記の変更でパーサが黙って空を返す事故を、取得の翌日までに見つける
"$PYTHON" -m src.scraper.parse_audit --years "$YEARS" >>"$LOG" 2>&1
AUD=$?
[ "$AUD" -ne 0 ] && echo "PARSE_AUDIT_FAIL: 必須項目の欠損があります（上の行を確認。docs/operations/stg_cron_schedule_20261014.md 参照）" >>"$LOG"
# 全 flat テーブルの列ごとの空・文字化けの急増（基準 2020〜前年 と 当年の直近）と、GCS の保存データ（最近保存した文書）の同様の検査
"$PYTHON" -m src.scraper.parse_drift --baseline "2020-$((YEAR - 1))" --recent "$YEAR" --months 3 >>"$LOG" 2>&1
DRIFT=$?
"$PYTHON" -m src.scraper.parse_drift_docs --baseline-year "$((YEAR - 1))" --recent-year "$YEAR" --sample 40 >>"$LOG" 2>&1
DOCS=$?
[ "$DRIFT" -ne 0 ] && echo "PARSE_DRIFT_FAIL: flat テーブルの列で空・文字化けの急増があります（上の行を確認）" >>"$LOG"
[ "$DOCS" -ne 0 ] && echo "PARSE_DRIFT_DOCS_FAIL: 最近保存した文書で空・文字化けの急増があります（上の行を確認）" >>"$LOG"
echo "===== $(date -Is) END exit=$EC audit=$AUD drift=$DRIFT docs=$DOCS =====" >>"$LOG"
exit "$EC"
