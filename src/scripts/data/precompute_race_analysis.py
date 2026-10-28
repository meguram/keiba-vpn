"""めぐ分析の事前計算（バッチ）。範囲内のレースの分析結果を計算して data/local/race_analysis/ に保存する。画面は保存済みの結果を読むだけ。

python -m src.scripts.data.precompute_race_analysis                 # 過去 past 日〜先 lookahead 日。指紋が変わったレースだけ作り直す
python -m src.scripts.data.precompute_race_analysis --force         # 全部作り直す
python -m src.scripts.data.precompute_race_analysis --races 202608041011,202608041010

呼ばれる場所: 毎朝の export_tables_daily.sh（結果が更新された後）、entries_sync（出馬表を取り直した直後。そのレースだけ）。
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger("precompute_race_analysis")

KEEP_DAYS = 90


def race_ids_in_range(storage: Any, today: date, past_days: int, lookahead_days: int) -> list[str]:
    from src.scraper.missing_races import is_jra_race_id

    ids: list[str] = []
    for i in range(-past_days, lookahead_days + 1):
        ymd = (today + timedelta(days=i)).strftime("%Y%m%d")
        try:
            races = (storage.load("race_lists", ymd) or {}).get("races") or []
        except Exception:
            logger.exception("race_lists の読み込みに失敗: %s", ymd)
            continue
        ids += [str(r["race_id"]) for r in races if r.get("race_id") and is_jra_race_id(str(r["race_id"]))]
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--past-days", type=int, default=30)
    ap.add_argument("--lookahead-days", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--races", help="race_id のカンマ区切り（指定すると日付の範囲は使わない）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.research.race_analysis import store
    from src.research.race_analysis.precompute import refresh_for_races
    from src.scraper.storage import HybridStorage
    from src.scripts.data.build_horse_sire_map import result_horse_ids

    storage = HybridStorage()
    ids = [r for r in (args.races or "").split(",") if r] or race_ids_in_range(storage, date.today(), args.past_days, args.lookahead_days)
    res = refresh_for_races(storage, ids, force=args.force, extra_horse_ids=result_horse_ids())
    pruned = store.prune(KEEP_DAYS)
    logger.info("race_analysis: races=%d %s pruned=%d", len(ids), res, pruned)
    return 1 if res.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
