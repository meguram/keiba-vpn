"""馬プロフィール表（db.netkeiba.com/horse/<id>/ の db_prof_table）の全行を抜き出すパイロット（ホストで実行）。

python -m src.scripts.data.acq_profile_pilot --per-year 6 --out data/local/research/acquisition/profile_pilot.jsonl

目的: セリ取引価格・馬主 ID 等が表に載るか、載る馬の割合（世代別）、書式を確かめる。保存は表の「見出し・本文・リンク先」だけ
（生 HTML は保存しない）。既存データは書き換えない。1 馬 1 リクエスト、4.5〜7 秒間隔、100 件ごとに 30〜60 秒休む、
連続 8 件失敗で中断（終了コード 3）。他のスクレイピング・高負荷の間は待つ（backfill_trifecta_odds.wait_ok と同じ）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
SHUTUBA_HORSE = ROOT / "data/local/features/horse_tbl/shutuba_horse.parquet"


def extract_rows(html: str) -> dict:
    """プロフィール表を {見出し: {text, hrefs}} にする。表が無ければ空。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.db_prof_table") or soup.select_one("div.db_prof_area table")
    rows = []
    if table:
        for tr in table.select("tr"):
            th, td = tr.select_one("th"), tr.select_one("td")
            if th is None or td is None:
                continue
            rows.append({
                "th": th.get_text(" ", strip=True),
                "td": td.get_text(" ", strip=True),
                "hrefs": [a.get("href", "") for a in td.select("a")],
            })
    name = soup.select_one("div.horse_title h1")
    return {"name": name.get_text(strip=True) if name else "", "rows": rows}


def sample_horses(per_year: int, seed: int, years: list[int]) -> list[str]:
    import polars as pl

    ids = pl.read_parquet(SHUTUBA_HORSE, columns=["horse_id"])["horse_id"].to_list()
    by_year: dict[int, list[str]] = {}
    for h in ids:
        if h and len(h) == 10 and h[:4].isdigit():
            by_year.setdefault(int(h[:4]), []).append(h)
    rnd = random.Random(seed)
    out: list[str] = []
    for y in years:
        pool = by_year.get(y, [])
        out += rnd.sample(pool, min(per_year, len(pool)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year", type=int, default=6)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2012, 2025)))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horse-ids", nargs="*", default=None, help="指定した馬だけ（サンプリングしない）")
    ap.add_argument("--ids-file", default=None, help="1 行 1 馬 ID の一覧（本取得用。先頭から順に取り、取得済みは飛ばす）")
    ap.add_argument("--out", default="data/local/research/acquisition/profile_pilot.jsonl")
    a = ap.parse_args()

    from src.scripts.data.backfill_trifecta_odds import wait_ok
    from src.scraper.run import ScraperRunner

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["horse_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    if a.ids_file:
        horses = [l.strip() for l in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        horses = a.horse_ids or sample_horses(a.per_year, a.seed, a.years)
    todo = [h for h in horses if h not in done]
    print(f"対象 {len(horses)} 頭、未取得 {len(todo)} 頭", flush=True)

    runner = ScraperRunner()
    streak = 0
    for n, h in enumerate(todo, 1):
        wait_ok()
        try:
            html = runner.client.fetch(runner.HORSE_URL.format(horse_id=h), use_cache=False)  # fetch 自身がスロットを取る
            rec = {"horse_id": h, **extract_rows(html)}
            ok = bool(rec["rows"])
        except Exception as e:  # noqa: BLE001
            rec, ok = {"horse_id": h, "error": str(e)[:200], "rows": []}, False
        streak = 0 if ok else streak + 1
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if streak >= 8:
            print("連続 8 件失敗（アクセス制限の疑い）で中断", flush=True)
            raise SystemExit(3)
        if n % 20 == 0:
            print(f"{n}/{len(todo)}", flush=True)
        time.sleep(random.uniform(30, 60) if n % 100 == 0 else random.uniform(4.5, 7.0))
    print("完了", flush=True)


if __name__ == "__main__":
    main()
