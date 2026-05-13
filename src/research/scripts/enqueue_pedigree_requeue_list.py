"""
horse_id リストを ``horse_pedigree_5gen`` タスクでスクレイピングキューに投入する。

既定: data/meta/pedigree_requeue_59_horse_ids.json
smart_skip=False で netkeiba から再取得（既存 GCS でも上書き取得）。

例::

    python3 -m src.research.scripts.enqueue_pedigree_requeue_list
    python3 -m src.research.scripts.enqueue_pedigree_requeue_list --ids-json path/to.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scraper.job_queue import PRIORITY_URGENT_PEDIGREE_5GEN, ScrapeJobQueue  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402

DEFAULT_IDS = Path("data/meta/pedigree_requeue_59_horse_ids.json")


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="horse_pedigree_5gen をキュー投入")
    ap.add_argument(
        "--ids-json",
        type=Path,
        default=DEFAULT_IDS,
        help="horse_id の JSON 配列",
    )
    args = ap.parse_args()

    raw = json.loads(args.ids_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("ids-json は配列である必要があります")
    horse_ids = [str(x).strip() for x in raw if str(x).strip()]
    if not horse_ids:
        raise SystemExit("horse_id が空です")

    q = ScrapeJobQueue()
    stats = q.add_horse_jobs_bulk(
        horse_ids,
        ["horse_pedigree_5gen"],
        priority=PRIORITY_URGENT_PEDIGREE_5GEN,
        smart_skip=False,
        skip_pedigree_5gen_if_complete=False,
    )
    logger = logging.getLogger(__name__)
    logger.info("投入 %d 頭 → %s", len(horse_ids), stats)


if __name__ == "__main__":
    main()
