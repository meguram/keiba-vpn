#!/usr/bin/env python3
"""追走難度を過去全レース分まとめて事前計算し、calculated_data に保存する。

例:
  python3 -m src.scripts.maintenance.precompute_tracking_difficulty_all --dry-run
  python3 -m src.scripts.maintenance.precompute_tracking_difficulty_all --skip-existing
  python3 -m src.scripts.maintenance.precompute_tracking_difficulty_all --limit 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.config.data_paths import TRACKING_DIFFICULTY_DIR  # noqa: E402
from src.pipeline.inference.tracking_difficulty_service import (  # noqa: E402
    build_tracking_difficulty_response,
    save_cached_response,
)
from src.pipeline.inference.tracking_difficulty_store import (  # noqa: E402
    count_local,
    exists_local,
    update_index_meta,
)
from src.scraper.storage import HybridStorage  # noqa: E402
from src.scripts.maintenance.batch_inference_all_races import (  # noqa: E402
    collect_race_ids,
    _has_shutuba,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("PrecomputeTrackingDifficulty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-race-lists", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--race-id", action="append", default=[])
    args = parser.parse_args(argv)

    storage = HybridStorage()
    if args.race_id:
        race_ids = args.race_id
    else:
        race_ids = collect_race_ids(storage, from_lists=args.from_race_lists)

    if args.limit > 0:
        race_ids = race_ids[: args.limit]

    existing = count_local()
    logger.info(
        "追走難度バッチ: races=%d skip_existing=%s store=%s (既存 %d 件)",
        len(race_ids),
        args.skip_existing,
        TRACKING_DIFFICULTY_DIR,
        existing,
    )

    if args.dry_run:
        return 0

    TRACKING_DIFFICULTY_DIR.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    t0 = time.perf_counter()

    for i, rid in enumerate(race_ids, 1):
        if not _has_shutuba(storage, rid):
            fail += 1
            continue
        if args.skip_existing and exists_local(rid):
            skip += 1
            continue
        try:
            payload = build_tracking_difficulty_response(
                rid,
                storage,
                allow_scrape=False,
                pre_race_only=True,
            )
            if payload.get("entries"):
                save_cached_response(storage, rid, payload, source="precompute_all")
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            logger.warning("FAIL %s: %s", rid, exc)
            fail += 1

        if i % 50 == 0 or i == len(race_ids):
            logger.info(
                "[%d/%d] ok=%d skip=%d fail=%d",
                i,
                len(race_ids),
                ok,
                skip,
                fail,
            )

    update_index_meta(batch_source="precompute_tracking_difficulty_all")
    logger.info(
        "完了 %.1fs — ok=%d skip=%d fail=%d total_local=%d",
        time.perf_counter() - t0,
        ok,
        skip,
        fail,
        count_local(),
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
