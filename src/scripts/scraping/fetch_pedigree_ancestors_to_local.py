"""ローカル 5gen 馬の祖先で 5gen データを持たない馬を取得しローカルへ保存。

GCS → ローカル の取得経路:
    1. 既存 ``data/local/horse_pedigree_5gen/`` をスキャンして、全祖先 ID 集合を作る
    2. ローカルに無い祖先 ID を抽出 (未取得 = 42,390 程度)
    3. ThreadPoolExecutor で GCS から並列ロード → ローカル強制保存
    4. GCS にも無い (None 戻り) 馬を todo として記録 → 次フェーズでスクレイピング

ローカル保存パス:
    data/local/horse_pedigree_5gen/{prefix}/{horse_id}.json
    prefix: horse_id の先頭 4 文字 (例: '2022', '000a')

Usage:
    python -m src.scripts.scraping.fetch_pedigree_ancestors_to_local \
        [--workers 16] [--limit N] [--dry-run]

"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scraper.storage import HybridStorage  # noqa: E402

logger = logging.getLogger("fetch_ancestors_to_local")

PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
TODO_LIST_PATH = ROOT / "data/research/pedigree_10gen_3view/scrape_todo.json"


def _prefix_of(horse_id: str) -> str:
    """horse_id の prefix 4 文字 (例: '2022', '000a')。"""
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def _local_path(horse_id: str) -> Path:
    return PED_DIR / _prefix_of(horse_id) / f"{horse_id}.json"


def _save_local(horse_id: str, data: dict) -> None:
    p = _local_path(horse_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def scan_local_state() -> tuple[set[str], set[str]]:
    """既存 5gen データの horse_id と祖先 ID を返す。

    Returns:
        (existing_ids, all_ancestor_ids)
    """
    existing_ids: set[str] = set()
    all_ancestors: set[str] = set()
    n_files = 0
    print(f"[scan] ローカル 5gen をスキャン: {PED_DIR}", flush=True)
    t0 = time.time()
    for prefix_dir in PED_DIR.iterdir():
        if not prefix_dir.is_dir():
            continue
        for f in prefix_dir.iterdir():
            if f.suffix != ".json":
                continue
            existing_ids.add(f.stem)
            try:
                data = json.loads(f.read_text())
                for a in data.get("ancestors", []):
                    hid = (a.get("horse_id") or "").strip()
                    if hid:
                        all_ancestors.add(hid)
            except Exception as e:
                logger.warning("読み込み失敗 %s: %s", f, e)
            n_files += 1
            if n_files % 10000 == 0:
                print(f"[scan]   {n_files:,} files / "
                      f"{len(all_ancestors):,} ancestors / "
                      f"{time.time()-t0:.1f}s", flush=True)
    print(f"[scan] 完了: existing={len(existing_ids):,} "
          f"ancestors={len(all_ancestors):,} ({time.time()-t0:.1f}s)", flush=True)
    return existing_ids, all_ancestors


def fetch_one(horse_id: str, storage: HybridStorage) -> tuple[str, str]:
    """1馬分を GCS から取得しローカル保存。

    Returns:
        (horse_id, status) where status in {'ok', 'not_in_gcs', 'error'}
    """
    try:
        data = storage.load("horse_pedigree_5gen", horse_id, bypass_cache=True)
    except Exception as e:
        logger.warning("load 例外 %s: %s", horse_id, e)
        return (horse_id, "error")
    if data is None:
        return (horse_id, "not_in_gcs")
    try:
        _save_local(horse_id, data)
    except Exception as e:
        logger.warning("save 例外 %s: %s", horse_id, e)
        return (horse_id, "error")
    return (horse_id, "ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16,
                        help="GCS 並列取得スレッド数 (default: 16)")
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N 馬のみ試行 (動作確認用)")
    parser.add_argument("--dry-run", action="store_true",
                        help="todo リストのみ作成 (取得しない)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    existing, ancestors = scan_local_state()
    missing = sorted(ancestors - existing)
    print(f"[main] 取得対象 (祖先で 5gen なし): {len(missing):,}", flush=True)
    if args.limit:
        missing = missing[:args.limit]
        print(f"[main] --limit により {len(missing):,} に絞る", flush=True)

    if args.dry_run:
        TODO_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        TODO_LIST_PATH.write_text(json.dumps(missing, ensure_ascii=False, indent=1))
        print(f"[main] dry-run: todo を保存 → {TODO_LIST_PATH}", flush=True)
        return 0

    storage = HybridStorage()
    counts = {"ok": 0, "not_in_gcs": 0, "error": 0}
    not_in_gcs: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exec:
        futures = {exec.submit(fetch_one, hid, storage): hid for hid in missing}
        for i, fut in enumerate(as_completed(futures)):
            hid, status = fut.result()
            counts[status] += 1
            if status == "not_in_gcs":
                not_in_gcs.append(hid)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(missing) - i - 1) / rate if rate > 0 else 0
                print(f"[main] {i+1:,}/{len(missing):,} "
                      f"ok={counts['ok']} ng={counts['not_in_gcs']} err={counts['error']} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"[main] 完了: {counts} ({(time.time()-t0)/60:.1f}min)", flush=True)

    # GCS にも無い馬 → スクレイピング todo として保存
    TODO_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TODO_LIST_PATH.write_text(json.dumps(sorted(not_in_gcs), ensure_ascii=False, indent=1))
    print(f"[main] GCS にも存在しない {len(not_in_gcs):,} 馬を todo に保存 → {TODO_LIST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
