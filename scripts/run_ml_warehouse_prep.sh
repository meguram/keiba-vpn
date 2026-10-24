#!/usr/bin/env bash
# ML 向けローカルデータ準備（バックグラウンド推奨）
#
# Usage:
#   nohup bash scripts/run_ml_warehouse_prep.sh > logs/ml_warehouse_prep.log 2>&1 &
#   bash scripts/watch_ml_warehouse.sh

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

YEARS="${ML_PREP_YEARS:-2020,2021,2022,2023,2024,2025}"
SYNC_INTERVAL="${ML_SYNC_INTERVAL:-0.05}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PROGRESS="$LOG_DIR/ml_warehouse_progress.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

write_progress() {
  python3 - "$PROGRESS" "$@" << 'PY'
import json, sys, time
from pathlib import Path
p, phase, detail = sys.argv[1], sys.argv[2], sys.argv[3]
doc = {}
if Path(p).exists():
    try:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        doc = {}
doc["updated_at"] = time.time()
doc["phase"] = phase
doc["detail"] = detail
Path(p).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

log "=== ML warehouse prep start (years=$YEARS) ==="
write_progress "start" "years=$YEARS"

# 1) Parquet（欠損分のみ GCS 取得。コアが揃っていればスキップ可）
log "--- Step 1: export_tables (Parquet) ---"
write_progress "export_parquet" "running"
SKIP_EXPORT=$(python3 - << PY
import sys
from pathlib import Path
years = [y.strip() for y in "$YEARS".split(",") if y.strip()]
core = ["race_result","race_shutuba","race_index","race_odds","race_oikiri","race_result_lap","race_pair_odds"]
ok = True
for y in years:
    d = Path("data/local/tables") / y
    if not d.is_dir():
        ok = False
        break
    for c in core:
        if not (d / f"{c}_flat.parquet").exists():
            ok = False
            break
print("1" if ok else "0")
PY
)
if [ "$SKIP_EXPORT" = "1" ] && [ "${ML_FORCE_EXPORT:-0}" != "1" ]; then
  log "Step 1 skipped (core Parquet already present; set ML_FORCE_EXPORT=1 to re-run)"
else
  python3 -m src.scripts.data.build_ml_warehouse \
    --years "$YEARS" \
    --export-parquet \
    --no-sqlite \
    --base-dir "$ROOT_DIR"
fi
write_progress "export_parquet" "done"

# 2) horse_result → ローカル cache（キャッシュが全頭揃っていれば GCS read ゼロでスキップ）
log "--- Step 2: sync horse_result (GCS→cache) ---"
write_progress "horse_result_sync" "running"
SKIP_SYNC=$(
  python3 -c "
from pathlib import Path
import sys

from src.scraper.storage import HybridStorage
from src.scripts.data.ml_warehouse.sqlite_builder import horse_ids_from_race_results
from src.scripts.data.sync_horse_result_cache import _disk_cache_usable

years = [y.strip() for y in sys.argv[1].split(',') if y.strip()]
base = Path(sys.argv[2]).resolve()
ids = horse_ids_from_race_results(base, years)
if not ids:
    print('1')
    raise SystemExit(0)
st = HybridStorage(str(base))
for hid in ids:
    if not _disk_cache_usable(st._local_cache_path('horse_result', str(hid))):
        print('0')
        raise SystemExit(0)
print('1')
" "$YEARS" "$ROOT_DIR"
)
if [ "$SKIP_SYNC" = "1" ] && [ "${ML_FORCE_SYNC:-0}" != "1" ]; then
  log "Step 2 skipped (all horse_result JSON already in data/cache; ML_FORCE_SYNC=1 to re-run)"
else
  python3 -m src.scripts.data.sync_horse_result_cache \
    --years "$YEARS" \
    --base-dir "$ROOT_DIR" \
    --interval "$SYNC_INTERVAL" \
    2>&1 | tee -a "$LOG_DIR/sync_horse_result.log"
fi
write_progress "horse_result_sync" "done"

# 3) 馬 SQLite シャード
log "--- Step 3: horse SQLite shards ---"
write_progress "sqlite" "running"
python3 -m src.scripts.data.build_ml_warehouse \
  --years "$YEARS" \
  --sqlite-only \
  --race-linked-horses \
  --base-dir "$ROOT_DIR"
write_progress "sqlite" "done"

# 4) 陳列 + manifest
log "--- Step 4: warehouse layout ---"
write_progress "layout" "running"
python3 -m src.scripts.data.build_ml_warehouse \
  --years "$YEARS" \
  --layout-only \
  --no-sqlite \
  --base-dir "$ROOT_DIR"
write_progress "layout" "done"

log "=== ML warehouse prep finished ==="
write_progress "done" "all steps complete"
