#!/usr/bin/env python3
"""待機中の horse_profile ジョブに skip_pedigree=True を付与する。"""
from __future__ import annotations

import sys

from src.scraper.job_queue import ScrapeJobQueue, _exclusive_queue_json_lock


def main() -> int:
    q = ScrapeJobQueue()
    with _exclusive_queue_json_lock():
        jobs = q._load_queue_nolock()
        n = 0
        for job in jobs:
            if job.get("status") != "pending":
                continue
            tasks = job.get("tasks") or job.get("types") or []
            if "horse_pedigree_5gen" in tasks:
                job["status"] = "completed"
                job["error"] = "skipped: horse_pedigree_5gen (user request)"
                job["completed_at"] = job.get("completed_at")
                n += 1
                continue
            if "horse_profile" in tasks:
                job["skip_pedigree"] = True
                n += 1
        q._save_queue_nolock(jobs)
    print(f"updated_or_skipped={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
