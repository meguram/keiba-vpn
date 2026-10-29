"""馬場速度指数の日次計算 CLI（確定結果の flat から races_{year}.parquet へアップサート）。

  python -m src.scripts.data.track_speed_daily --from 20261020 --to 20261029

flat の更新（export_tables）は含まない。GCP Job では export-tables-daily の直後の手順として動かす。
確定結果が 0 件の年・期間は「まだ無い」として飛ばす（異常ではない）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("scripts.track_speed_daily")


def _years_in_range(d_from: str, d_to: str) -> list[str]:
    return [str(y) for y in range(int(d_from[:4]), int(d_to[:4]) + 1)]


def run(d_from: str, d_to: str, base_dir: str | Path | None = None) -> dict[str, str]:
    """年ごとに assign_races を呼ぶ。戻り値は {年: ok|no_results}。ベースライン欠落は例外のまま上げる。"""
    from src.research.race.track_speed_engine import TrackSpeedEngine

    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[3]
    eng = TrackSpeedEngine(str(base))
    out: dict[str, str] = {}
    for year in _years_in_range(d_from, d_to):
        lo = d_from if d_from[:4] == year else f"{year}0101"
        hi = d_to if d_to[:4] == year else f"{year}1231"
        try:
            eng.assign_races(years=[year], date_min=lo, date_max=hi, progress_cb=lambda m: logger.info("[馬場速度] %s", m))
            out[year] = "ok"
        except ValueError as exc:
            logger.info("[馬場速度] %s %s-%s は確定結果が未取得のため見送り: %s", year, lo, hi, exc)
            out[year] = "no_results"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to", dest="d_to", required=True)
    a = ap.parse_args(argv)
    for v in (a.d_from, a.d_to):
        datetime.strptime(v, "%Y%m%d")
    if a.d_from > a.d_to:
        ap.error("--from は --to 以前にしてください")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run(a.d_from, a.d_to)
    logger.info("[馬場速度] 完了: %s", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
