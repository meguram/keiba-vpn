#!/usr/bin/env bash
# 2026 年の v6e バックフィル（別の場所に算出済み）を、画面が見る場所へ差し替える。ホストで root（sudo -n）で実行。
#   sudo -n bash scripts/ops/swap_pred_v6e.sh [--min-ratio 0.98] [--yes]
# 手順: 件数を確認（出馬表のあるレースに対する割合）→ 既存の bundle_v6_2025 を退避 → staging を 1 回の mv で差し替え
#       → 予測成績の日次記録を退避して作り直し → キャッシュの世代を更新。途中で失敗したら退避から戻す。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"; cd "$ROOT"
INF="${KEIBA_INFERENCE_DIR:-$ROOT/data/local/inference}"
ST="$INF/predictions_backfill_v6e_staging/bundle_v6_2025"; DST="$INF/predictions_backfill/bundle_v6_2025"
MIN=0.98; YES=0
while [ $# -gt 0 ]; do case "$1" in --min-ratio) MIN="$2"; shift 2;; --yes) YES=1; shift;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
PY="${PYTHON:-/opt/conda/bin/python3}"; export PYTHONPATH="$ROOT" KEIBA_ENV="${KEIBA_ENV:-stg}"
[ -d "$ST" ] || { echo "staging がありません: $ST" >&2; exit 1; }
n_st=$(find "$ST" -name '*.parquet' | wc -l)
n_all=$(PYTHONPATH="$ROOT" "$PY" - <<'EOF'
import polars as pl
d = pl.read_parquet("data/page_reference/tables/2026/race_shutuba_flat.parquet", columns=["race_id"]).unique()
print(sum(1 for r in d["race_id"].cast(pl.Utf8).to_list() if len(r) == 12 and r[4:6] in {f"{i:02d}" for i in range(1, 11)}))
EOF
)
echo "staging: $n_st 件 / 2026 年の JRA レース: $n_all 件"
ok=$("$PY" -c "print(1 if $n_all and $n_st / $n_all >= $MIN else 0)")
[ "$ok" = 1 ] || { echo "中止: 件数が足りません（min-ratio=$MIN）。算出が終わっていない可能性があります" >&2; exit 1; }
if [ "$YES" != 1 ]; then read -r -p "差し替えますか？ [y/N] " a; [ "$a" = y ] || { echo "中止"; exit 1; }; fi
TS="$(date +%Y%m%d%H%M%S)"; OLD=""
if [ -d "$DST" ]; then OLD="$INF/predictions_backfill_old_bundle_v6_2025_$TS"; mv "$DST" "$OLD"; echo "退避: $OLD"; fi
mkdir -p "$(dirname "$DST")"; mv "$ST" "$DST"; echo "差し替え: $DST"
PRB="$ROOT/data/local/prediction_results"
if [ -d "$PRB" ]; then cp -a "$PRB" "${PRB}_before_v6e_$TS"; echo "予測成績の記録を退避: ${PRB}_before_v6e_$TS"; fi
last=$(ls "$PRB" 2>/dev/null | grep -E '^[0-9]{8}\.json$' | tail -1 | cut -c1-8 || true)
if [ -n "${last:-}" ]; then
  nice -n 15 "$PY" -m src.scripts.data.prediction_results --from 20260101 --to "$last" || { echo "予測成績の再集計に失敗（記録は退避済み: ${PRB}_before_v6e_$TS）" >&2; exit 1; }
fi
"$PY" -c "from src.utils import cache_epoch; cache_epoch.bump('v6e backfill swap')" && echo "キャッシュ世代を更新"
echo "完了。戻すには: mv $DST <捨てる場所> && mv ${OLD:-<退避したディレクトリ>} $DST（予測成績は ${PRB}_before_v6e_$TS から戻す）"
