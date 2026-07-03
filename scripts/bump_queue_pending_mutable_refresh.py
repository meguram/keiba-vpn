#!/usr/bin/env python3
"""
保留中・precheck のキュー行から ``smart_skip`` を外し、タスク単位ポリシー（可変=上書き既定）を適用させる。

旧ジョブは ``smart_skip: true`` が JSON に残っており、race_result 等がスキップされ続けることがある。
本スクリプトは pending / precheck のみを改変する（running / completed は触らない）。

Usage:
  cd keiba-vpn && python3 scripts/bump_queue_pending_mutable_refresh.py
  python3 scripts/bump_queue_pending_mutable_refresh.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.scraper.job_queue import (
        QUEUE_FILE,
        ScrapeJobQueue,
        _exclusive_queue_json_lock,
    )
    from src.scraper.queue_tasks import normalize_tasks
    from src.scraper.scrape_policy import (
        QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP,
        _catalog_task_ids,
    )

    if not QUEUE_FILE.exists():
        print("no queue file:", QUEUE_FILE, file=sys.stderr)
        return 0

    mut_atomic = _catalog_task_ids() - QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP - frozenset({"race_all"})

    changed = 0
    scanned = 0
    with _exclusive_queue_json_lock():
        q = ScrapeJobQueue()
        jobs = q._load_queue_nolock()
        for j in jobs:
            st = str(j.get("status") or "")
            if st not in ("pending", "precheck"):
                continue
            if j.get("overwrite") is True:
                continue
            tasks = normalize_tasks(j.get("tasks"))
            if not tasks or "race_all" in tasks:
                continue
            if not all(t in mut_atomic for t in tasks):
                continue
            if "smart_skip" not in j:
                continue
            scanned += 1
            if args.dry_run:
                print("would strip smart_skip:", j.get("job_id"), tasks, "was", j.get("smart_skip"))
                changed += 1
                continue
            j.pop("smart_skip", None)
            changed += 1
        if not args.dry_run and changed:
            q._save_queue_nolock(jobs)

    print(f"scanned={scanned} updated={changed} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
