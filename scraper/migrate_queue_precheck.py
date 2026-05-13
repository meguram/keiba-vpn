"""
既存の ``scrape_queue.json`` 内で、まだ ``pending`` かつ上書きなし・スマートスキップの行を
``precheck`` に揃える（全タスク種。新投入と同じ二段階キュー）。

  python3 -m scraper.migrate_queue_precheck --dry-run
  python3 -m scraper.migrate_queue_precheck
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.keiba_logging import script_basic_config  # noqa: E402

script_basic_config()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pending → precheck（上書きなし・スキップありの既存行）"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="scrape_queue.json を書き換えず件数のみ",
    )
    args = ap.parse_args()
    from scraper.job_queue import ScrapeJobQueue

    q = ScrapeJobQueue()
    out = q.migrate_pending_to_storage_precheck(dry_run=bool(args.dry_run))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
