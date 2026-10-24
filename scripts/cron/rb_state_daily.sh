#!/usr/bin/env bash
# v6 の RB（めぐ指数/netkeiba 指数の履歴 16 列）の推論用の履歴状態を、前日までに確定した走で作り直す（日次）。
# tp65_prep（走ごとの指数の表）→ build_rb_state build → 状態の置き場 data/local/inference/state/rb_v6/ をファイル単位で原子的に差し替える。
# 失敗しても古い版が残る（検査に通らなければ差し替えない）。GCS へは書かない。所要は十数秒・メモリ 1GB 未満（heavy_run は使わない）。
# Usage: bash scripts/cron/rb_state_daily.sh [PROJECT_DIR]
# 状態の消費側: KEIBA_V6_SHADOW=1 のときの V3Serving（KEIBA_V6_RB_STATE、既定は同じ場所）。v6 を配備していない間も作っておく（害は無い）。
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs
LOG="logs/rb_state_daily.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
STATE="${KEIBA_V6_RB_STATE:-${KEIBA_INFERENCE_DIR:-$PROJECT_DIR/data/local/inference}/state/rb_v6}"
echo "===== $(date -Is) START state=$STATE =====" >>"$LOG"
# エクスポートと同じロックを使う（書き出し中の flat を読まない）。最大 1 時間待つ
exec 9>"/tmp/keiba_export_tables_daily.lock"
if ! flock -w 3600 9; then
  echo "[$(date -Is)] export のロックを取れず（1 時間待った）。スキップ" >>"$LOG"; exit 0
fi
exec 9>&-
PARENT="$(dirname "$STATE")"
mkdir -p "$PARENT"
TMP="$(mktemp -d "$PARENT/.rb_v6_build.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
YEARS="$(seq -s ' ' 2020 "$(TZ=Asia/Tokyo date +%Y)")"
# shellcheck disable=SC2086
PYTHONPATH=. OMP_NUM_THREADS=2 nice -n 10 "$PY" -m src.scripts.data.tp65_prep --out "$TMP/prep" --years $YEARS >>"$LOG" 2>&1 || { echo "tp65_prep 失敗。古い状態を残す" >>"$LOG"; exit 1; }
PYTHONPATH=. OMP_NUM_THREADS=2 nice -n 10 "$PY" -m src.scripts.data.build_rb_state build --runs "$TMP/prep/runs.parquet" --out "$TMP/state" >>"$LOG" 2>&1 || { echo "build_rb_state 失敗。古い状態を残す" >>"$LOG"; exit 1; }
# 検査: 履歴の件数が前回より減っていない・最終日が戻っていない
OK="$("$PY" -B - "$STATE" "$TMP/state" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
old, new = Path(sys.argv[1]), Path(sys.argv[2])
n = pd.read_parquet(new / "hist.parquet", columns=["date"])
pn = json.loads((new / "params.json").read_text(encoding="utf-8"))
if len(n) < 100000:
    print("NG 件数が少なすぎる", len(n)); raise SystemExit
if (old / "hist.parquet").exists():
    o = pd.read_parquet(old / "hist.parquet", columns=["date"])
    po = json.loads((old / "params.json").read_text(encoding="utf-8"))
    if len(n) < len(o) * 0.99:
        print("NG 件数が減った", len(o), "->", len(n)); raise SystemExit
    if str(pn.get("cut")) < str(po.get("cut")):
        print("NG 最終日が戻った", po.get("cut"), "->", pn.get("cut")); raise SystemExit
    print(f"OK 件数 {len(o):,} -> {len(n):,}、最終日 {po.get('cut')} -> {pn.get('cut')}")
else:
    print(f"OK 新規 件数 {len(n):,}、最終日 {pn.get('cut')}")
PY
)"
echo "[$(date -Is)] 検査: $OK" >>"$LOG"
case "$OK" in OK*) ;; *) echo "検査に通らない。古い状態を残す" >>"$LOG"; exit 1 ;; esac
mkdir -p "$STATE"
# ファイル単位で原子的に差し替える（同じファイルシステム上の mv -f）。履歴 → 定数の順
cp "$TMP/state/hist.parquet" "$STATE/.hist.parquet.new" && mv -f "$STATE/.hist.parquet.new" "$STATE/hist.parquet" \
  && cp "$TMP/state/params.json" "$STATE/.params.json.new" && mv -f "$STATE/.params.json.new" "$STATE/params.json" \
  || { echo "差し替えに失敗" >>"$LOG"; exit 1; }
echo "===== $(date -Is) END exit=0 =====" >>"$LOG"
exit 0
