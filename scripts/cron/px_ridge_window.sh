#!/usr/bin/env bash
# v5/v6 の血統 ALL_BEST（PX）のリッジを、次の半年（1/1・7/1）に入る前に作っておく（無いと最初の予測で自動学習して遅い／失敗しうる）。
# 毎週日曜に動くが、新しい半年の初日の 30 日前以降で、まだ ridge_<T0>.npz が無いときだけ build_px_state --windows <T0> を実行する（それ以外は何もしない）。
# リッジは T0 より前の確定結果で学習する（as-of。特徴量パイプラインの一部で、モデルの学習ではない）。祖先表・SVD の成分は学習時のまま固定（更新しない）。
# 数分・メモリ数 GB のため /tmp/heavy_run.sh（共有スロット）経由。出力は state/px_v5/ridge_<T0>.npz（一時ファイルへ書いて差し替え）。GCS へは書かない。
# Usage: bash scripts/cron/px_ridge_window.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs
LOG="logs/px_ridge_window.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
STATE="${KEIBA_INFERENCE_DIR:-$PROJECT_DIR/data/local/inference}/state/px_v5"
TODAY="${PX_RIDGE_TODAY:-$(TZ=Asia/Tokyo date +%F)}"      # PX_RIDGE_TODAY は動作確認用（状態の置き場も KEIBA_INFERENCE_DIR で一時の場所へ向けること）
read -r T0 DAYS <<<"$("$PY" -B -c "
import sys, datetime as dt
d = dt.date.fromisoformat(sys.argv[1])
t0 = dt.date(d.year, 7, 1) if d.month <= 6 else dt.date(d.year + 1, 1, 1)
print(t0.isoformat(), (t0 - d).days)
" "$TODAY")"
echo "===== $(date -Is) START today=$TODAY 次の窓 T0=$T0（あと ${DAYS} 日） =====" >>"$LOG"
if [ "$DAYS" -gt 30 ]; then echo "まだ早い（30 日より先）。何もしない" >>"$LOG"; exit 0; fi
if [ -f "$STATE/ridge_${T0}.npz" ]; then echo "ridge_${T0}.npz は既にある。何もしない" >>"$LOG"; exit 0; fi
if [ ! -f "$STATE/svd.npz" ]; then echo "PX の状態（svd.npz）が無い。何もしない" >>"$LOG"; exit 1; fi
HR="${HEAVY_RUN:-/tmp/heavy_run.sh}"
if [ -x "$HR" ]; then WRAP=("$HR" px-ridge-window); else WRAP=(); fi
# build_px_state は STATE_DIR に直接書くので、一時の場所へ書かせてから差し替える
TMP="$(mktemp -d "$STATE/.ridge_build.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
for f in svd.npz anc.parquet ancestors.parquet horses.parquet state_meta.json; do ln -s "$STATE/$f" "$TMP/$f"; done
mkdir -p "$TMP/inf/state"; ln -s "$TMP" "$TMP/inf/state/px_v5"
KEIBA_ENV="${KEIBA_ENV:-stg}" KEIBA_INFERENCE_DIR="$TMP/inf" PYTHONPATH=. OMP_NUM_THREADS=4 "${WRAP[@]}" nice -n 10 "$PY" -m src.scripts.data.build_px_state --windows "$T0" >>"$LOG" 2>&1 || { echo "build_px_state 失敗。何も差し替えない" >>"$LOG"; exit 1; }
[ -f "$TMP/ridge_${T0}.npz" ] || { echo "ridge_${T0}.npz ができていない" >>"$LOG"; exit 1; }
cp "$TMP/ridge_${T0}.npz" "$STATE/.ridge_${T0}.npz.new" && mv -f "$STATE/.ridge_${T0}.npz.new" "$STATE/ridge_${T0}.npz"
echo "作成: $STATE/ridge_${T0}.npz ($(stat -c %s "$STATE/ridge_${T0}.npz") bytes)" >>"$LOG"
echo "===== $(date -Is) END exit=0 =====" >>"$LOG"
exit 0
