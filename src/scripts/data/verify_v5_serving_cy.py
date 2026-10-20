"""推論側の CY（serving_v5）が、学習側の CY（tp40/groups/CY.parquet）と一致するかを過去の日で確かめる。

KEIBA_ENV=stg python -m src.scripts.data.verify_v5_serving_cy --days 20250614 20250215 --max-races 8 --out /tmp/verify_v5_cy.json
馬別調教ページの記録は storage.load('horse_training')（保存物）から読む。基準表は tp35/tl_reference_v2.parquet（再取得版の追い切りから作った参照表）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_v5 as sv5

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
TABLES = ROOT / "data/page_reference/tables"
KEY = ["race_id", "horse_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--max-races", type=int, default=8)
    ap.add_argument("--ref", default=str(V4 / "tp35/tl_reference_v2.parquet"), help="基準表。v5 は再取得版の追い切りから作った tl_reference_v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    ref = pd.read_parquet(a.ref)
    train = pd.read_parquet(V4 / "tp40/groups/CY.parquet")
    train["race_id"], train["horse_id"] = train["race_id"].astype(str), train["horse_id"].astype(str)
    train = train.drop_duplicates(KEY)
    rep: dict = {}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        flat = pl.read_parquet(TABLES / day[:4] / "race_shutuba_flat.parquet", columns=["race_id", "horse_id", "date"]).filter(pl.col("date").cast(pl.Utf8) == d).to_pandas()
        ids = sorted(flat["race_id"].unique())[: a.max_races]
        races = flat[flat["race_id"].isin(ids)].drop_duplicates(KEY)
        records = {h: storage.load("horse_training", str(h)) for h in races["horse_id"].astype(str).unique()}
        got = sv5.cy_columns_for_races(races[KEY + ["date"]], records, ref)
        m = got.merge(train, on=KEY, how="inner", suffixes=("", "__r"))
        cols = [c for c in got.columns if c not in KEY and c + "__r" in m.columns]
        bad = {}
        for c in cols:
            x, y = m[c].to_numpy(dtype=float), m[c + "__r"].to_numpy(dtype=float)
            ok = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-3, atol=1e-4)
            if not ok.all():
                bad[c] = float(1 - ok.mean())
        rep[day] = {"rows": int(len(m)), "cols": len(cols), "bad": len(bad), "worst": dict(sorted(bad.items(), key=lambda kv: -kv[1])[:5]),
                    "missing_in_serving": sorted(c for c in train.columns if c not in KEY and c not in got.columns)[:10]}
        print(day, json.dumps(rep[day], ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
