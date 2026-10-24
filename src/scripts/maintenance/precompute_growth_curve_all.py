#!/usr/bin/env python3
"""成長曲線を馬名インデックス対象分、calculated_data に事前計算する。

例:
  python3 -m src.scripts.maintenance.precompute_growth_curve_all --dry-run
  python3 -m src.scripts.maintenance.precompute_growth_curve_all --limit 100
  python3 -m src.scripts.maintenance.precompute_growth_curve_all --skip-existing
  python3 -m src.scripts.maintenance.precompute_growth_curve_all --weekly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.pipeline.inference.growth_curve_service import (  # noqa: E402
    filter_horse_ids_with_horse_result,
    iter_index_horse_ids,
    precompute_growth_curves,
    run_weekly_growth_curve_update,
)
from src.pipeline.inference.growth_curve_store import count_local  # noqa: E402
from src.scraper.storage import HybridStorage  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("PrecomputeGrowthCurve")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weekly",
        action="store_true",
        help="金曜週次更新（horse_result あり・7日以内キャッシュはスキップ）",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.set_defaults(skip_existing=True)
    p.add_argument("--force-refresh", action="store_true", help="ローカルキャッシュを無視して再計算")
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--horse-id", action="append", default=[])
    p.add_argument(
        "--no-filter-horse-result",
        action="store_true",
        help="horse_result 存在チェックを省略（非推奨）",
    )
    args = p.parse_args(argv)

    if args.weekly:
        if args.dry_run:
            storage = HybridStorage(str(_ROOT))
            ids = iter_index_horse_ids(_ROOT)
            filtered = filter_horse_ids_with_horse_result(storage, ids)
            logger.info(
                "週次更新 dry-run: インデックス %d → horse_result %d 頭",
                len(ids),
                len(filtered),
            )
            return 0
        result = run_weekly_growth_curve_update(_ROOT)
        logger.info("完了: %s", result)
        return 0 if result.get("fail", 0) == 0 else 1

    if args.horse_id:
        horse_ids = args.horse_id
    else:
        horse_ids = iter_index_horse_ids(_ROOT)

    if not args.no_filter_horse_result:
        storage = HybridStorage(str(_ROOT))
        before = len(horse_ids)
        # --limit 時は全インデックス走査を避け、候補を絞ってから存在チェック
        candidates = horse_ids
        if args.limit > 0 and not args.weekly:
            candidates = horse_ids[: max(args.limit * 20, args.limit)]
        horse_ids = filter_horse_ids_with_horse_result(storage, candidates)
        logger.info("horse_result フィルタ: %d → %d 頭", before, len(horse_ids))

    if args.limit > 0:
        horse_ids = horse_ids[: args.limit]

    logger.info(
        "成長曲線バッチ: targets=%d skip_existing=%s force_refresh=%s 既存=%d",
        len(horse_ids),
        args.skip_existing,
        args.force_refresh,
        count_local(),
    )

    if args.dry_run:
        return 0

    storage = HybridStorage(str(_ROOT))
    result = precompute_growth_curves(
        storage,
        horse_ids,
        skip_existing=args.skip_existing,
        force_refresh=args.force_refresh,
        max_age_days=args.max_age_days,
        workers=args.workers,
    )
    logger.info("完了: %s (local=%d)", result, count_local())
    return 0 if result.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
