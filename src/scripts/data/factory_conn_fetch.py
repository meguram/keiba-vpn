"""ファクトリー G2: 馬の静的プロファイル（馬主・生産者・産地・毛色・誕生日）を horse_profile（GCS/ローカルキャッシュ）から集めて 1 枚の表にする。

python -m src.scripts.data.factory_conn_fetch   # ホスト（KEIBA_ENV=stg）で実行。読み取りのみ
出力: data/local/modeling/training_pipeline/factory/conn/horse_profile.parquet（horse_id, owner, breeder, birthplace, color, sex, birthday）
注意: 馬主・生産者は ID が無く名前だけ（正規化して使う）。馬主は取得時点の値（途中の名義変更は反映されない）。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline/factory/conn"
FIELDS = ["owner", "breeder", "birthplace", "color", "sex", "birthday"]


def main(workers: int = 6) -> None:
    from src.scraper.storage import HybridStorage

    OUT.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    for y in range(2020, 2027):
        f = ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet"
        if f.exists():
            ids |= set(pl.read_parquet(f, columns=["horse_id"])["horse_id"].drop_nulls().to_list())
    ids = sorted(ids)
    st = HybridStorage()
    t0 = time.time()

    def one(h: str) -> dict:
        try:
            d = st.load("horse_profile", h) or st.load("horse_result", h) or {}
        except Exception:
            d = {}
        return {"horse_id": h, **{k: d.get(k) for k in FIELDS}}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, ids), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"[{time.time() - t0:.0f}s] {i}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "horse_profile.parquet", index=False)
    print(f"保存 {len(df)} 頭 owner有 {df['owner'].notna().mean():.3f} breeder有 {df['breeder'].notna().mean():.3f} birthplace有 {df['birthplace'].notna().mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
