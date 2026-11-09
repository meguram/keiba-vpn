"""2016〜2019 年だけに出走した馬の静的プロファイル（馬主・生産者・産地・毛色・性・誕生日）を 1 回だけ集め、既存の表に足す（再学習 2026-11）。

    python -m src.scripts.data.fetch_horse_profile_2016 [--out /home/jovyan/work/retrain_out/base_2016/profile]

- 既存の factory/conn/horse_profile.parquet・factory/self/horse_attrs.parquet にある馬は読まない（差分だけ）。
- 読み取り元は HybridStorage（horse_profile → 無ければ horse_result）。ホストから GCS を読むので、繰り返し実行しない
  （見積もり: 約 1.85 万頭 × 約 35KB ≈ 0.65 GiB ≈ 12 円・Class B 約 3.7 万回 ≈ 2 円、1 回限り）。
出力: <out>/horse_profile.parquet（conn 用）、<out>/horse_attrs.parquet（self 用）。どちらも既存＋追加。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
CONN = ROOT / "data/local/modeling/training_pipeline/factory/conn/horse_profile.parquet"
SELF = ROOT / "data/local/modeling/training_pipeline/factory/self/horse_attrs.parquet"
FIELDS = ["owner", "breeder", "birthplace", "color", "sex", "birthday"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/profile")
    ap.add_argument("--years", default="2016-2025")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(x) for x in a.years.split("-"))
    ids: set[str] = set()
    for y in range(y0, y1 + 1):
        ids |= set(pl.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet", columns=["horse_id"])["horse_id"].drop_nulls().cast(pl.Utf8).to_list())
    prof = pd.read_parquet(CONN)
    prof["horse_id"] = prof["horse_id"].astype(str)
    attrs = pd.read_parquet(SELF)
    attrs["horse_id"] = attrs["horse_id"].astype(str)
    todo = sorted(ids - (set(prof["horse_id"]) & set(attrs["horse_id"])))
    print(f"対象 {len(ids)} 頭、未取得 {len(todo)} 頭", flush=True)

    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    t0 = time.time()

    def one(h: str) -> dict:
        try:
            d = st.load("horse_profile", h) or st.load("horse_result", h) or {}
        except Exception:
            d = {}
        return {"horse_id": h, **{k: d.get(k) for k in FIELDS}}

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"[{time.time() - t0:.0f}s] {i}/{len(todo)}", flush=True)
    new = pd.DataFrame(rows, columns=["horse_id", *FIELDS])
    print(f"取得 {len(new)} 頭 owner有 {new['owner'].notna().mean():.3f} birthday有 {new['birthday'].notna().mean():.3f}", flush=True)

    p2 = pd.concat([prof, new[~new["horse_id"].isin(prof["horse_id"])][prof.columns.intersection(new.columns)]], ignore_index=True)
    p2.to_parquet(out / "horse_profile.parquet", index=False)
    na = new[~new["horse_id"].isin(attrs["horse_id"])][["horse_id", "birthday", "birthplace", "color", "owner", "breeder"]].copy()
    ymd = na["birthday"].astype(str).str.extract(r"(\d{4})年(\d{1,2})月(\d{1,2})日").astype(float)
    ymd.columns = ["year", "month", "day"]
    na["birthday"] = pd.to_datetime(ymd, errors="coerce")
    a2 = pd.concat([attrs, na[attrs.columns.intersection(na.columns)]], ignore_index=True)
    a2.to_parquet(out / "horse_attrs.parquet", index=False)
    print(f"保存 profile {len(p2)} 頭 / attrs {len(a2)} 頭", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
