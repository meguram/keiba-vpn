#!/usr/bin/env bash
# ドリフトの週次監視（特徴量の抜き取り込み）。金曜の weekly-update（18:00 JST）で先週末の確定結果が入った後に回す。
# 1) 先週末（直近 8 日の土日）のうち開催のあった日から 10 レースずつ、推論経路の特徴量を抜き取って保存（data/local/inference/drift_features/<年>/、重い。nice 19・load>14 で待つ）
# 2) drift_monitor を回して latest.json / latest.html / history.jsonl を作る（モデルと市場の差の管理図・CUSUM・較正・特徴量の PSI・v4/v5 shadow）
# 読み取りと data/local への書き込みのみ。GCS/DB/netkeiba へはアクセスしない。指標は報告・警報のみで、モデルや閾値の選択には使わない（config/drift_periods.json）。
# 自動実行（auto_scrape からの軽量版、特徴量なし）は KEIBA_DRIFT_MONITOR=1 のときだけ。このスクリプトは手動・cron 用で、環境変数に依らず実行する。
# KEIBA_DRIFT_DUMP=0 で抜き取りを省く。Usage: bash scripts/cron/drift_monitor_weekly.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
export TZ="${TZ:-Asia/Tokyo}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
DAYS=""
for back in 1 2 3 4 5 6 7 8; do
  if [ "$(date -d "${back} days ago" +%u)" -ge 6 ]; then DAYS="${DAYS:+$DAYS,}$(date -d "${back} days ago" +%Y%m%d)"; fi
done
if [ "${KEIBA_DRIFT_DUMP:-1}" != "0" ] && [ -n "$DAYS" ]; then
  KEIBA_INFERENCE_ENGINE=full nice -n 19 "$PY" -m src.scripts.data.drift_dump_features --days "$DAYS" --max-load 14
  echo "drift dump: days=${DAYS} rc=$?"
fi
nice -n 19 "$PY" -m src.scripts.data.drift_monitor --root "$PROJECT_DIR"
rc=$?
echo "drift monitor: rc=${rc}"
exit "$rc"
