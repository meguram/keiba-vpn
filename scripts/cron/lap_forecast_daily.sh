#!/usr/bin/env bash
# ラップ予想(tp-55)の日次生成。出馬表（race_shutuba_flat）を書き出してから、対象日の全レースのハロン別予想を作って
# data/local/inference/lap_forecast/<日付>/<race_id>.json に保存する（予測 API が KEIBA_LAP_FORECAST=1 で読む）。
# Usage: bash scripts/cron/lap_forecast_daily.sh eve|morning [PROJECT_DIR]
#   eve     … 前日の夕方に、翌日（JST）のレース分を作る（出馬表は前日夕方の raceday-eve で保存される）
#   morning … 当日の朝に、当日（JST）のレース分を作り直す（07:00 JST のエクスポートで前日の結果が履歴に入った後）
# 開催の無い日は、出馬表が無いので 0 件で終わる（害は無い）。モデルは data/local/inference/lap_forecast_models/（tp-55 の A_full / A_noten）。
set -u
MODE="${1:?eve|morning}"
PROJECT_DIR="${2:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs
LOG="logs/lap_forecast_daily.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
MODEL_DIR="${KEIBA_LAP_MODEL_DIR:-$PROJECT_DIR/data/local/inference/lap_forecast_models}"
case "$MODE" in
  eve) DATE="$(TZ=Asia/Tokyo date -d tomorrow +%F)" ;;
  morning) DATE="$(TZ=Asia/Tokyo date +%F)" ;;
  *) echo "usage: lap_forecast_daily.sh eve|morning [PROJECT_DIR]" >&2; exit 2 ;;
esac
DATE="${LAP_FORECAST_DATE:-$DATE}"      # 手動で日付を指定して作り直すとき（例 LAP_FORECAST_DATE=2026-10-18）
YEAR="${DATE%%-*}"
echo "===== $(date -Is) START mode=$MODE date=$DATE =====" >>"$LOG"
# エクスポートと同じロックを使う（書き出し中の flat を読まない・書き出しを妨げない）。最大 1 時間待つ
exec 9>"/tmp/keiba_export_tables_daily.lock"
if ! flock -w 3600 9; then
  echo "[$(date -Is)] export のロックを取れず（1 時間待った）。スキップ" >>"$LOG"; exit 0
fi
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. "$PY" -m src.scraper.export_tables --years "$YEAR" --categories race_shutuba --refresh >>"$LOG" 2>&1 || echo "WARN: race_shutuba のエクスポートに失敗（既存の flat で続行）" >>"$LOG"
exec 9>&-
# 対象日の出馬表が flat に無ければ（開催の無い日）、何もしないで正常終了
N="$("$PY" -B -c "
import sys, pyarrow.parquet as pq, pandas as pd
d = sys.argv[1]
try:
    t = pq.read_table(f'data/page_reference/tables/{d[:4]}/race_shutuba_flat.parquet', columns=['date']).to_pandas()
    print(int((pd.to_datetime(t['date']).dt.strftime('%Y-%m-%d') == d).sum()))
except Exception:
    print(0)
" "$DATE" 2>>"$LOG")"
if [ "${N:-0}" -eq 0 ]; then
  echo "出馬表の行が無い（開催なし、または未保存）: $DATE。スキップ" >>"$LOG"
  echo "===== $(date -Is) END mode=$MODE date=$DATE exit=0 (skip) =====" >>"$LOG"; exit 0
fi
echo "出馬表の行数 $N ($DATE)" >>"$LOG"
KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scripts.data.tp55_backfill --dates "$DATE" --model-dir "$MODEL_DIR" \
  --model-full model_final2026_A_full.pkl --model-noten model_final2026_A_noten.pkl >>"$LOG" 2>&1
EC=$?
echo "===== $(date -Is) END mode=$MODE date=$DATE exit=$EC =====" >>"$LOG"
exit "$EC"
