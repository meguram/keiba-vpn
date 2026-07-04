"""レース振り分け CLI (2026年以降など任意期間)"""

from __future__ import annotations

import argparse
import logging

from src.research.race.track_speed_engine import TrackSpeedEngine
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="馬場速度レース振り分け")
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    years = args.years
    if years is None:
        y0 = int(args.date_from[:4])
        y1 = int((args.date_to or "2026-12-31")[:4])
        years = [str(y) for y in range(y0, y1 + 1)]

    eng = TrackSpeedEngine(args.base_dir)
    if not eng.load_baselines():
        raise SystemExit("先に build_track_speed_baselines を実行してください")

    def log(msg: str) -> None:
        logger.info(msg)
        print(msg)

    paths = eng.assign_races(
        years=years,
        date_min=args.date_from.replace("-", ""),
        date_max=(args.date_to or "").replace("-", "") or None,
        progress_cb=log,
    )
    for y, p in paths.items():
        log(f"  {y}: {p}")


if __name__ == "__main__":
    main()
