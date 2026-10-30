"""馬ごとの 200m 区間ラップ（db.netkeiba.com の ajax_race_result_horse_laptime）が取れるかのパイロット（ホストで実行）。

python -m src.scripts.data.acq_horse_lap_pilot --per-year 2 --out /home/jovyan/acq/horse_lap_pilot.jsonl

1 レース 1 リクエスト。パース結果（区間列と各馬の区間ラップ・位置）だけを保存し、生 HTML は保存しない。既存の保存先には書かない。
他のスクレイピング・高負荷の間は待つ。4.5〜7 秒間隔、連続 8 件失敗で中断（終了コード 3）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])


def pick_races(per_year: int, years: list[int], seed: int) -> list[str]:
    import polars as pl

    rnd = random.Random(seed)
    out: list[str] = []
    for y in years:
        f = ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet"
        if f.exists():
            ids = pl.read_parquet(f, columns=["race_id"])["race_id"].unique().sort().to_list()
            out += rnd.sample(ids, min(per_year, len(ids)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year", type=int, default=2)
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2022, 2024, 2025])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="/home/jovyan/acq/horse_lap_pilot.jsonl")
    ap.add_argument("--race-ids", nargs="+", default=[], help="指定すると無作為抽出の代わりにこのレースだけ取得")
    ap.add_argument("--no-wait", action="store_true", help="他の取得や高負荷の待機をしない")
    a = ap.parse_args()

    from src.scripts.data.backfill_trifecta_odds import wait_ok
    from src.scraper.parsers import DbHorseLaptimeTableParser
    from src.scraper.run import ScraperRunner

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runner = ScraperRunner()
    parser = DbHorseLaptimeTableParser()
    streak = 0
    for n, rid in enumerate(a.race_ids or pick_races(a.per_year, a.years, a.seed), 1):
        if not a.no_wait:
            wait_ok()
        try:
            html = runner.client.fetch(parser.AJAX_URL.format(race_id=rid), use_cache=False, encoding="euc-jp")
            d = parser.parse(html, race_id=rid)
            ok = bool(d.get("entries"))
        except Exception as e:  # noqa: BLE001
            d, ok = {"race_id": rid, "error": str(e)[:200], "entries": []}, False
        streak = 0 if ok else streak + 1
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(rid, "ok" if ok else "NG", len(d.get("entries", [])), flush=True)
        if streak >= 8:
            raise SystemExit(3)
        time.sleep(random.uniform(4.5, 7.0))


if __name__ == "__main__":
    main()
