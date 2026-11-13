#!/usr/bin/env bash
# 推論モードの履歴（race_result_flat など）を最新に保つため、当年（1 月は前年も）のテーブルを増分エクスポートする。
# --refresh --from-local-results: 各カテゴリが自分の flat に無いレースを追記する（これが無いと race_result に新しいレースが入らない）。
#   足すレースはローカルの記録（race_lists・出馬表／速報／確定結果の flat）から決め、名前を指定して読む。GCS の一覧は取らない
#   （2026-11-13 まで、毎朝 約 30 回の全件一覧で月 約 50 円だった。export_tables.export_year_local）。
# Usage: bash scripts/cron/export_tables_daily.sh [PROJECT_DIR] [categories]
#   categories を渡すときは export_tables.LOCAL_CATEGORIES の中から選ぶ（それ以外は --from-local-results が受け付けず、その年が止まる）。
set -u
export TZ="${TZ:-Asia/Tokyo}"
PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CATS="${2:-race_result,race_shutuba,race_result_lap,race_result_on_time,race_odds,race_pair_odds}"
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
# 付随カテゴリ（馬柱・指数・バロメーター・SmartRC・追い切り・パドック・厩舎コメント）も同じ呼び出しで足す（確定結果のあるレースだけ）
LOCAL_CATS="${LOCAL_CATS:-race_shutuba_past,race_index,race_barometer,smartrc_race,race_oikiri,race_paddock,race_trainer_comment}"
ALL_CATS="$CATS${LOCAL_CATS:+,$LOCAL_CATS}"
echo "===== $(date -Is) START years=$YEARS categories=$ALL_CATS =====" >>"$LOG"
"$PYTHON" -m src.scraper.export_tables --years "$YEARS" --categories "$ALL_CATS" --refresh --from-local-results >>"$LOG" 2>&1
EC=$?
[ "$EC" -ne 0 ] && echo "LOCAL_EXPORT_FAIL: 一覧なしの追記で失敗したカテゴリがあります（上の行を確認）" >>"$LOG"
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
# めぐ分析の走法の軸（run_axes）: GCS の horse_lap（バロメーターの取得が同じページから保存）を、足りないレースだけ名前で読んで追記し、
# 追記があれば作り直す。事前計算の前に行う（指紋に run_axes が入っているので、変わった日は窓の分析が作り直される）。失敗しても本体には影響させない
"$PYTHON" -m src.scripts.data.update_run_axes >>"$LOG" 2>&1 || echo "RUN_AXES_FAIL: 走法の軸の更新に失敗しました（上の行を確認）" >>"$LOG"
# めぐ分析の事前計算（結果が更新されたので、過去30日〜先14日のレースの分析を作り直す。父の対応表も先に差分更新する）。失敗しても本体の終了コードには影響させない
"$PYTHON" -m src.scripts.data.precompute_race_analysis >>"$LOG" 2>&1 || echo "RACE_ANALYSIS_FAIL: めぐ分析の事前計算に失敗しました（上の行を確認）" >>"$LOG"
echo "===== $(date -Is) END exit=$EC audit=$AUD drift=$DRIFT docs=$DOCS =====" >>"$LOG"
exit "$EC"
