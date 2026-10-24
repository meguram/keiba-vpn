#!/usr/bin/env bash
# scrape_missing_5gen_for_10gen 完了後に、10gen 側の生成物を連鎖実行する。
#
# 前提:
#   - リポジトリルートで実行（cd はスクリプトが行う）
#   - data/research/bloodline_meta_cluster/race_records.parquet が存在すること
#     （build_pedigree_10gen_index / 3view が参照）
#
# Usage:
#   # 現在走っている scrape_missing_5gen_for_10gen が終わるまで待ってから実行
#   nohup bash scripts/run_after_scrape_missing_5gen_10gen_chain.sh \
#     >> logs/scraping/after_5gen_10gen_chain.log 2>&1 &
#
#   # 待たずに直ちに 10gen 連鎖だけ実行（5gen 取得が既に完了している場合）
#   bash scripts/run_after_scrape_missing_5gen_10gen_chain.sh --no-wait
#
# 実行内容（順）:
#   1. build_horse_pedigree_10gen … 5gen JOIN で data/local/horse_pedigree_10gen/*.json
#   2. build_pedigree_10gen_index … father/mother 用 parquet (data/research/pedigree_10gen/)
#   3. build_pedigree_10gen_3view_index … 3 視点 parquet (data/research/pedigree_10gen_3view/)
#   4. build_ancestor_name_map … 5gen+10gen 全 JSON から ancestor_id→name（上記と整合）

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export ROOT
# Python が glibc の per-thread arena で RSS を膨らませるのを抑える (長時間バッチ向け)
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
mkdir -p logs/scraping

WAIT=1
for a in "$@"; do
  case "$a" in
    --no-wait) WAIT=0 ;;
    -h|--help)
      grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

PATTERN='src.scripts.scraping.scrape_missing_5gen_for_10gen'
PROGRESS="data/research/pedigree_10gen_3view/scrape_missing_5gen_progress.json"

if [ "$WAIT" = "1" ]; then
  log "scrape_missing_5gen_for_10gen の終了を待機 ($PATTERN) …"
  while pgrep -f "$PATTERN" >/dev/null 2>&1; do
    sleep 45
  done
  log "対象プロセスは終了しました。進捗 JSON を確認します。"
  if [ -f "$PROGRESS" ]; then
    python3 - "$PROGRESS" << 'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception as e:
    print(f"[warn] progress 読込失敗: {e}")
    raise SystemExit(0)
fin = d.get("finished_at")
if fin:
    print(f"[info] finished_at={fin}")
else:
    print("[warn] finished_at 無し（異常終了の可能性）。続行します。")
PY
  else
    log "進捗 JSON なし（未実行または別パス）。続行します。"
  fi
else
  log "--no-wait: 待機スキップ"
fi

log "=== 10gen 連鎖開始 (ROOT=$ROOT) ==="

log "[1/4] build_horse_pedigree_10gen"
python3 -m src.research.pedigree.build_horse_pedigree_10gen

log "[2/4] build_pedigree_10gen_index"
python3 -m src.research.pedigree.build_pedigree_10gen_index

log "[3/4] build_pedigree_10gen_3view_index"
python3 -m src.research.pedigree.build_pedigree_10gen_3view_index

log "[4/4] build_ancestor_name_map"
python3 -m src.research.pedigree.build_ancestor_name_map

log "=== 10gen 連鎖 完了 ==="
