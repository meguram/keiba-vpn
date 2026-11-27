"""前の夜の段の 4 列と、前向きの記録の CSV（forward/<日付>.csv）の照合（学習と推論のずれの確かめ）。

    nice -n 19 /opt/conda/bin/python -m src.research.pedigree_persona.verify 2026-11-21 2026-11-22 2026-11-23

- 線（数字の前に固定）: dirt_sire・dirt_bms・dirt_mat5・dirt_lv5 の |差| ≤ 1e-6（CSV は小数 6 桁に丸めてあるので、丸めの差 5e-7 を含む）、
  馬の組（race_id × horse_id）が同じ、no_pedigree が同じ。合わなければ終了コード 1。
- 読むのは記録の生の値だけ（着順・払戻は読まない。記録の CSV に着順は無い）。記録の CSV は書かない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 1e-6
COLS = ["dirt_sire", "dirt_bms", "dirt_mat5", "dirt_lv5"]
FORWARD = Path(__file__).resolve().parents[3] / "docs/operations/pedigree_lab/forward"


def compare(day: str) -> dict:
    import numpy as np
    import pandas as pd

    from src.research.pedigree_persona import columns as pc

    out, _ = pc.build_day(pd.Timestamp(day))
    rec = pd.read_csv(FORWARD / f"{day}.csv", dtype={"race_id": str, "horse_id": str})
    out["race_id"], out["horse_id"] = out["race_id"].astype(str), out["horse_id"].astype(str)
    m = rec.merge(out, on=["race_id", "horse_id"], how="outer", suffixes=("_rec", "_new"), indicator=True)
    res = {"date": day, "rows_rec": int(len(rec)), "rows_new": int(len(out)), "unmatched": int((m["_merge"] != "both").sum())}
    b = m[m["_merge"] == "both"]
    for c in COLS:
        res[f"max_abs_{c}"] = float(np.max(np.abs(b[f"{c}_rec"].to_numpy(float) - b[f"{c}_new"].round(6).to_numpy(float)))) if len(b) else None
    res["no_pedigree_diff"] = int((b["no_pedigree_rec"].astype(bool) != b["no_pedigree_new"].astype(bool)).sum())
    res["ok"] = res["unmatched"] == 0 and res["no_pedigree_diff"] == 0 and all((res[f"max_abs_{c}"] or 0) <= TOL for c in COLS)
    return res


def main() -> None:
    days = sys.argv[1:] or ["2026-11-21", "2026-11-22", "2026-11-23"]
    rs = [compare(d) for d in days]
    print(json.dumps(rs, ensure_ascii=False, indent=1), flush=True)
    if not all(r["ok"] for r in rs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
