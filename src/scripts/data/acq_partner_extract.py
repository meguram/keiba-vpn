"""追い切りの併せ馬メモ（GCS の horse_training、読み取りのみ）を (horse_id, train_date, partner_notes) の表にする（ホストで実行・読み取りのみ）。

python -m src.scripts.data.acq_partner_extract --out /home/jovyan/work/acq/workouts_partner.parquet
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/acq/workouts_partner.parquet")
    ap.add_argument("--ped", default="/home/jovyan/work/acq/ped_dam.parquet", help="対象の馬の一覧（acq_history_extract の出力）")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    st = HybridStorage(base_dir=str(ROOT))
    ids = [h for h in pd.read_parquet(a.ped)["horse_id"] if len(h) == 10 and h.isdigit() and h[:4] >= "2012"]
    if a.limit:
        ids = ids[: a.limit]

    def read(h: str) -> list[dict]:
        try:
            d = st.load("horse_training", h) or {}
        except Exception:
            return []
        return [{"horse_id": h, "train_date": e.get("date"), "partner_notes": e.get("partner_notes") or ""} for e in d.get("entries") or [] if e.get("date")]

    with ThreadPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(read, ids) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    print(f"馬 {len(ids)} 追い切り {len(df)} 併せ馬メモあり {(df.partner_notes != '').sum()} 馬数(追い切りあり) {df.horse_id.nunique()}")


if __name__ == "__main__":
    main()
