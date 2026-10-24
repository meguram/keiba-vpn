"""
追走難度レスポンスを storage に事前計算して保存する CLI。

例:
  python -m src.scripts.maintenance.precompute_tracking_difficulty 202603010301
  python -m src.scripts.maintenance.precompute_tracking_difficulty --from-file races.txt
"""

from __future__ import annotations

import argparse
import sys
import time

from src.pipeline.inference.tracking_difficulty_service import (
    build_tracking_difficulty_response,
    save_cached_response,
)
from src.scraper.storage import HybridStorage
from src.utils.logger import get_logger

logger = get_logger("PrecomputeTracking")


def _get_storage() -> HybridStorage:
    return HybridStorage()


def race_ids_for_date(storage, date_yyyymmdd: str) -> list[str]:
    """開催日 prefix (YYYYMMDD) に一致する race_shutuba キー一覧。"""
    prefix = date_yyyymmdd.strip().replace("-", "")[:8]
    if len(prefix) != 8 or not prefix.isdigit():
        return []
    return sorted(
        k.replace(".json", "")
        for k in storage.list_keys("race_shutuba")
        if k.replace(".json", "").startswith(prefix)
    )


def precompute_one(
    storage,
    race_id: str,
    *,
    allow_scrape: bool,
    pre_race_only: bool = True,
) -> dict:
    payload = build_tracking_difficulty_response(
        race_id,
        storage,
        allow_scrape=allow_scrape,
        pre_race_only=pre_race_only,
    )
    if payload.get("entries"):
        save_cached_response(storage, race_id, payload, source="precompute_cli")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="追走難度キャッシュの事前計算")
    parser.add_argument(
        "race_ids",
        nargs="*",
        help="レース ID（複数可）",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="1行1レース ID のファイル",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="ローカルに無い場合スクレイプする",
    )
    parser.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help="開催日で race_shutuba がある全レースを対象にする",
    )
    parser.add_argument(
        "--include-result-fallback",
        action="store_true",
        help="当該レースの race_result も参照する（事後データ混入の恐れあり）",
    )
    args = parser.parse_args(argv)

    storage = _get_storage()
    pre_race_only = not args.include_result_fallback

    ids = list(args.race_ids or [])
    if args.date:
        ids.extend(race_ids_for_date(storage, args.date))
        ids = sorted(set(ids))
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            for line in f:
                rid = line.strip()
                if rid and not rid.startswith("#"):
                    ids.append(rid)

    if not ids:
        parser.error("race_id、または --date を指定してください")

    ok = 0
    fail = 0
    for race_id in ids:
        t0 = time.perf_counter()
        try:
            payload = precompute_one(
                storage,
                race_id,
                allow_scrape=args.scrape,
                pre_race_only=pre_race_only,
            )
            meta = payload.get("_compute_meta") or {}
            ms = meta.get("elapsed_ms") or round((time.perf_counter() - t0) * 1000, 1)
            n = len(payload.get("entries") or [])
            if n:
                logger.info(
                    "OK %s horses=%d backend=%s %.0fms",
                    race_id,
                    n,
                    meta.get("lgbm_backend", "?"),
                    ms,
                )
                ok += 1
            else:
                logger.warning("SKIP %s: %s", race_id, payload.get("error", "no entries"))
                fail += 1
        except Exception as e:
            logger.exception("FAIL %s: %s", race_id, e)
            fail += 1

    logger.info("完了: ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
