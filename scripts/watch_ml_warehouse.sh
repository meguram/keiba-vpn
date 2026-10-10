#!/bin/bash
# ML ローカルデータ準備の進捗モニター（プログレスバー付き）
#
# Usage:
#   bash scripts/watch_ml_warehouse.sh
#   bash scripts/watch_ml_warehouse.sh --interval 3
#   bash scripts/watch_ml_warehouse.sh --years 2024,2025

set -euo pipefail
cd "$(dirname "$0")/.."

INTERVAL=2
YEARS="2020,2021,2022,2023,2024,2025"

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="${2:-2}"; shift 2 ;;
    --years) YEARS="${2:-$YEARS}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/watch_ml_warehouse.sh [--interval SEC] [--years YYYY,...]"
      exit 0
      ;;
    *) shift ;;
  esac
done

export ML_WATCH_YEARS="$YEARS"

while true; do
  clear
  ML_WATCH_YEARS="$YEARS" python3 << 'PY'
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

years = [y.strip() for y in os.environ.get("ML_WATCH_YEARS", "2025").split(",") if y.strip()]
base = Path(".")
log_prep = base / "logs/ml_warehouse_prep.log"
log_sync = base / "logs/sync_horse_result.log"
prog_json = base / "logs/ml_warehouse_progress.json"
tables = base / "data/local/tables"
wh = base / "data/ml/warehouse"
cat_db = wh / "sqlite/catalog.sqlite3"

def bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "░" * width + "  N/A"
    p = max(0.0, min(done / total, 1.0))
    f = int(width * p)
    return "█" * f + "░" * (width - f) + f" {p*100:5.1f}%"

def pgrep_pat(pat: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False

def count_json(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.json"))

# --- ジョブ状態 ---
phase = "idle"
detail = ""
if prog_json.exists():
    try:
        j = json.loads(prog_json.read_text(encoding="utf-8"))
        phase = j.get("phase", phase)
        detail = j.get("detail", "")
    except Exception:
        pass

running_export = pgrep_pat("build_ml_warehouse.*export-parquet") or pgrep_pat("export_tables")
running_sync = pgrep_pat("sync_horse_result_cache")
running_sqlite = pgrep_pat("build_ml_warehouse.*sqlite-only")
running_prep = pgrep_pat("run_ml_warehouse_prep")

if running_prep or running_export or running_sync or running_sqlite:
    state = "実行中"
elif phase == "done":
    state = "完了"
else:
    state = "停止?"

print("=" * 62)
print(f" ML データ準備 監視   {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 62)
print(f"  状態: {state}   フェーズ: {phase}  ({detail})")
print(f"  プロセス: prep={running_prep} export={running_export} "
      f"horse_sync={running_sync} sqlite={running_sqlite}")
print()

# --- 1) Parquet export ---
print("[1] レース系 Parquet (data/local/tables)")
core_cats = [
    "race_result", "race_shutuba", "race_index", "race_odds",
    "race_oikiri", "race_result_lap", "race_pair_odds",
]
extra_cats = [
    "race_info", "race_lap", "race_return", "race_detail",
    "horse_lap", "race_predictions",
]
for y in years:
    rep = tables / y / "_export_report.json"
    n_core = 0
    if (tables / y).is_dir():
        for c in core_cats:
            if (tables / y / f"{c}_flat.parquet").exists():
                n_core += 1
    total_core = len(core_cats)
    line = f"  {y} コア {bar(n_core, total_core, 24)}  ({n_core}/{total_core})"
    if rep.exists():
        try:
            r = json.loads(rep.read_text(encoding="utf-8"))
            skipped = sum(1 for s in r.get("categories", []) if s.get("skipped"))
            print(line + f"  export済 skip={skipped}")
        except Exception:
            print(line)
    else:
        print(line + "  (report なし)")

print()

# --- 2) horse_result cache ---
print("[2] horse_result (GCS→cache)")
# 期待頭数: race_result parquet から
import pyarrow.parquet as pq
expected = set()
for y in years:
    p = tables / y / "race_result_flat.parquet"
    if p.exists():
        try:
            t = pq.read_table(p, columns=["horse_id"])
            for h in t.column(0).to_pylist():
                if h:
                    expected.add(str(h))
        except Exception:
            pass
cached = count_json(base / "data/cache/horse_result")
local_hr = count_json(base / "data/local/horse_result")
have = cached + local_hr  # 重複あり得るが概算
print(f"  期待(出走馬ユニーク): {len(expected):>6}")
print(f"  cache JSON:          {cached:>6}")
print(f"  local JSON:          {local_hr:>6}")
if expected:
    # 粗い進捗: cache のみでは不足 — sync ログから ok 数
    prog_sync = base / "logs/sync_horse_result_progress.json"
    done = ok = total = 0
    if prog_sync.exists():
        try:
            sj = json.loads(prog_sync.read_text(encoding="utf-8"))
            done = int(sj.get("done", 0))
            ok = int(sj.get("ok", 0))
            total = int(sj.get("total", len(expected)))
        except Exception:
            pass
    if total <= 0:
        total = len(expected)
    print(f"  同期: {bar(done, total, 24)}  ({done}/{total})  ok={ok}")
print()

# --- 3) SQLite shards ---
print("[3] 馬 SQLite シャード (data/ml/warehouse/sqlite/horses)")
if cat_db.exists():
    con = sqlite3.connect(str(cat_db))
    rows = con.execute(
        "SELECT SUM(profile_count), SUM(history_count), SUM(training_count), COUNT(*) "
        "FROM shard_registry"
    ).fetchone()
    n_lookup = con.execute("SELECT COUNT(*) FROM horse_lookup").fetchone()[0]
    con.close()
    prof, hist, train, n_sh = rows[0] or 0, rows[1] or 0, rows[2] or 0, rows[3] or 0
    print(f"  シャード数: {n_sh}")
    print(f"  profile: {prof:>7}  history: {hist:>7}  training: {train:>9}")
    print(f"  horse_lookup: {n_lookup}")
    if expected:
        print(f"  profile率(対出走馬): {bar(int(prof), len(expected), 24)}")
else:
    print("  catalog 未作成")
print()

# --- 4) 陳列 ---
print("[4] warehouse 陳列 (data/ml/warehouse)")
man = wh / "manifest.json"
n_links = 0
for y in years:
    d = wh / "by_year" / y / "scraper_flat"
    if d.is_dir():
        n_links += sum(1 for _ in d.iterdir() if _.is_symlink() or _.exists())
print(f"  manifest: {'あり' if man.exists() else 'なし'}")
print(f"  by_year symlink 数: {n_links}")
print()
print("  ログ:")
print(f"    tail -f logs/ml_warehouse_prep.log")
print(f"    tail -f logs/sync_horse_result.log")
print()
print("  Ctrl+C で終了（バックグラウンドジョブは止まりません）")
PY
  sleep "$INTERVAL"
done
