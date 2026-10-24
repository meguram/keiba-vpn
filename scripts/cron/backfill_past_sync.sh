#!/usr/bin/env bash
# 過去分（2016〜2019 年）を、学習が読む形に揃える（GCS の派生カテゴリ → ローカルのフラット表 → 整合の検証）。取得とは別に、手動または週次で実行する。
# 取得先（netkeiba）へは一切アクセスしない。GCS は読み取りと、派生カテゴリの『新しいキーの保存』のみ（既存データの上書き・削除はしない）。
# 重い処理なので /tmp/heavy_run.sh 経由で直列に実行する。モデルの学習は行わない。
# Usage: bash scripts/cron/backfill_past_sync.sh [PROJECT_DIR] [YEARS(カンマ区切り。既定 2016,2017,2018,2019)]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
YEARS="${2:-2016,2017,2018,2019}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
LOG="logs/backfill_past_sync.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
HEAVY="${HEAVY_RUN:-/tmp/heavy_run.sh}"
run() { if [ -x "$HEAVY" ]; then "$HEAVY" "$@"; else "${@:2}"; fi; }   # run <label> <cmd...>
exec 9>"/tmp/keiba_backfill_past_sync.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>"$LOG"; exit 0; fi
echo "===== $(date -Is) START years=${YEARS} =====" >>"$LOG"
Y0="${YEARS%%,*}"; Y1="${YEARS##*,}"
export KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=.
HFLAG="--include-horses"; [ "${SYNC_HORSES:-1}" = "0" ] && HFLAG=""   # 馬の派生は全馬キーを走査するため重い。レースだけ確認するときは SYNC_HORSES=0
run backfill-past-derived "$PY" -m src.scripts.scraping.migrate_row_data_to_unique_paths --year-start "$Y0" --year-end "$Y1" $HFLAG >>"$LOG" 2>&1
run backfill-past-flat "$PY" -m src.scraper.export_tables --years "$YEARS" --categories race_result,race_shutuba,race_index,race_odds,race_oikiri,race_result_lap,race_shutuba_past,race_result_on_time,race_pair_odds --workers 2 --refresh >>"$LOG" 2>&1
"$PY" -m src.scripts.data.verify_past_backfill --years ${YEARS//,/ } --progress-ok --json-out data/local/meta/backfill_past_verify.json >>"$LOG" 2>&1
RC=$?
echo "===== $(date -Is) END verify_rc=${RC} =====" >>"$LOG"
exit "$RC"
