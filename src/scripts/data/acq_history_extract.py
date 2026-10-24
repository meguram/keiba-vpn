"""非 JRA（地方・海外）を含む全走の履歴表と、馬→母・父の対応表を作る（ホストで実行・読み取りのみ）。

python -m src.scripts.data.acq_history_extract --out /home/jovyan/work/acq

出力:
  runs_all.parquet  horse_id, race_id, date, is_jra, venue, surface, distance, field_size, finish_position, time_sec, last_3f, weight
  ped_dam.parquet   horse_id, sire_id, dam_id, dam_sire_id（horse_pedigree_5gen の 1 世代目・2 世代目）
入力: horse_result（race_history に地方・海外の走も入っている）、data/local/horse_pedigree_5gen。
市場の値（オッズ・人気）は出力しない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
JRA_CODES = {f"{i:02d}" for i in range(1, 11)}
RUN_FIELDS = ("venue", "surface", "distance", "field_size", "finish_position", "time_sec", "last_3f", "weight")


def is_jra_race_id(race_id: str) -> bool:
    r = str(race_id or "")
    return len(r) == 12 and r[4:6] in JRA_CODES


def read_ped(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sire = dam = dsire = ""
    for a in d.get("ancestors") or []:
        g, p = a.get("generation"), a.get("position")
        if g == 1 and p == 0:
            sire = str(a.get("horse_id") or "")
        elif g == 1 and p == 1:
            dam = str(a.get("horse_id") or "")
        elif g == 2 and p == 2:
            dsire = str(a.get("horse_id") or "")
    return {"horse_id": str(d.get("horse_id") or path.stem), "sire_id": sire, "dam_id": dam, "dam_sire_id": dsire}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/acq")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ped_dir = ROOT / "data/local/horse_pedigree_5gen"
    files = sorted(ped_dir.rglob("*.json"))
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        peds = [p for p in ex.map(read_ped, files) if p]
    ped = pd.DataFrame(peds).drop_duplicates("horse_id")
    ped.to_parquet(out / "ped_dam.parquet", index=False)
    print(f"血統 {len(ped)} 頭 ({time.time() - t0:.0f}s)", flush=True)

    from src.scraper.storage import HybridStorage

    st = HybridStorage(base_dir=str(ROOT))
    ids = [h for h in ped["horse_id"] if len(h) == 10 and h.isdigit() and "2009" <= h[:4] <= "2026"]
    if a.limit:
        ids = ids[: a.limit]

    def load(h: str) -> list[dict]:
        try:
            d = st.load("horse_result", h)
        except Exception:
            return []
        rows = []
        for r in (d or {}).get("race_history") or []:
            rid = str(r.get("race_id") or "")
            if not rid:
                continue
            rows.append({"horse_id": h, "race_id": rid, "date": str(r.get("date") or "").replace("/", "-"), "is_jra": int(is_jra_race_id(rid)),
                         **{k: r.get(k) for k in RUN_FIELDS}})
        return rows

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, rs in enumerate(ex.map(load, ids), 1):
            rows += rs
            if i % 2000 == 0:
                print(f"{i}/{len(ids)} 走 {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
                while os.getloadavg()[0] >= 16:
                    time.sleep(20)
    df = pd.DataFrame(rows)
    for c in ("distance", "field_size", "finish_position", "weight"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("time_sec", "last_3f"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out / "runs_all.parquet", index=False)
    print(f"完了 馬 {len(ids)} 走 {len(df)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
