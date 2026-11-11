#!/usr/bin/env bash
# 2016〜2019 年だけに走った馬の調教ページ（horse_training）を全ページ取り直す（手動起動・再開できる）。
# Usage: sudo nohup bash scripts/cron/horse_training_past.sh [PROJECT_DIR] &
# 環境変数: TRAINING_PAST_HORSES（対象の馬 ID。既定 data/local/meta/horse_training_past_targets.parquet）、TRAINING_PAST_LIMIT（既定 0＝残り全部）
# 進捗の表示: bash scripts/dev/fetch_status.sh（「調教の過去分の取り直し」）。進捗: data/local/meta/horse_training_past_progress.json
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs
LOG="logs/horse_training_past.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
exec 9>"/tmp/keiba_horse_training_past.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>"$LOG"; exit 0; fi
# 他の取得が長く動いて待ちきれなかったとき（rc=4）だけ、少し置いて続きから再開する（アクセス制限・失敗の連続は rc=3 で止まったまま）
for try in 1 2 3 4; do
  echo "===== $(date -Is) START =====" >>"$LOG"
  # 同時に取得するのは自分を含めて 1 本（他のプロセスがスロットを持っている間は待つ）
  # クライアントの間隔（2.2〜4 秒）はそのまま。それに重ねている追加の待ち（ページ間・1 頭ごと）だけ短くする
  HORSE_TRAINING_INTER_PAGE_MIN_SEC="${HORSE_TRAINING_INTER_PAGE_MIN_SEC:-0.2}" HORSE_TRAINING_INTER_PAGE_MAX_SEC="${HORSE_TRAINING_INTER_PAGE_MAX_SEC:-0.6}" \
  HORSE_TRAINING_BACKFILL_MIN_SEC="${HORSE_TRAINING_BACKFILL_MIN_SEC:-0.2}" HORSE_TRAINING_BACKFILL_MAX_SEC="${HORSE_TRAINING_BACKFILL_MAX_SEC:-0.6}" \
  NETKEIBA_GLOBAL_MAX_SLOTS="${NETKEIBA_GLOBAL_MAX_SLOTS:-1}" KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scripts.scraping.backfill_horse_training_past \
    --horses-file "${TRAINING_PAST_HORSES:-data/local/meta/horse_training_past_targets.parquet}" --limit "${TRAINING_PAST_LIMIT:-0}" >>"$LOG" 2>&1
  rc=$?
  echo "===== $(date -Is) END rc=$rc =====" >>"$LOG"
  [ "$rc" = 4 ] || break
  sleep 300
done
