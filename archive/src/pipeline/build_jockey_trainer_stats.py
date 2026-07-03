"""
騎手・調教師の成績統計 Parquet を生成する。

  python -m src.pipeline.build_jockey_trainer_stats
  python -m src.pipeline.build_jockey_trainer_stats --years 2022,2023,2024

既定は JRA（venue_code 01–10）のみ。地方を含める場合は ``--nar``。

定期実行はリポジトリの ``scripts/cron/update_jockey_trainer_stats.sh`` と
``scripts/cron/setup_jockey_trainer_stats_cron.sh``（``install``）を利用。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.features.jockey_trainer_stats import write_jockey_trainer_stats
from src.utils.keiba_logging import script_basic_config


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(
        description="騎手・調教師統計を data/features/race_horse_tbl/ 等へ出力（マニフェストは jockey_trainer_stats/）"
    )
    ap.add_argument(
        "--years",
        type=str,
        default="",
        help="カンマ区切り年（省略時は tables にある全年）",
    )
    ap.add_argument(
        "--nar",
        action="store_true",
        help="地方（JRA 以外）も含める（既定は JRA のみ）",
    )
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力ディレクトリ（既定: <base-dir>/data/features/jockey_trainer_stats）",
    )
    args = ap.parse_args()
    years: list[str] | None = None
    if args.years.strip():
        years = [y.strip() for y in args.years.split(",") if y.strip()]

    try:
        manifest = write_jockey_trainer_stats(
            years=years,
            base_dir=args.base_dir,
            jra_only=not args.nar,
            out_dir=args.out_dir,
        )
    except Exception as e:
        logging.error("%s", e)
        return 1
    logging.info("manifest: %s", manifest.get("paths", {}).get("manifest"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
