"""tp-64: 展開利の特徴量（pb_*）が V5 の既存の列とどれだけ重なるか（2023 年の行で、各 pb_* と最も相関の高い既存列）。

python -m src.scripts.data.tp64_overlap   # 出力: $TP64_OUT/overlap.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp27_eval as t
from src.pipeline.features import tp64_pace_benefit as pb
from src.scripts.data import tp64_eval as ev

OUT = Path(os.environ.get("TP64_OUT") or "/home/jovyan/work/tp64_out")


def main() -> int:
    df, base, meta, folds, lp = t.setup()
    G = pd.read_parquet(ev.FEATS)
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    m = df[["race_id", "horse_id", "year"]].merge(G, on=["race_id", "horse_id"], how="left")
    sel_rows = (df["year"] == 2023).to_numpy()
    num = df.loc[sel_rows].select_dtypes("number").drop(columns=["year"], errors="ignore")
    num = num.loc[:, num.std() > 0]
    P = m.loc[sel_rows, ev.HIST + ev.GAP].reset_index(drop=True)
    num = num.reset_index(drop=True)
    sample = np.random.default_rng(0).choice(len(num), size=min(40000, len(num)), replace=False)
    nr, pr = num.iloc[sample].rank(), P.iloc[sample].rank()
    res = {}
    for c in P.columns:
        x = pr[c]
        ok = x.notna()
        if ok.sum() < 1000:
            continue
        cors = nr[ok].corrwith(x[ok]).dropna()
        top = cors.abs().sort_values(ascending=False).head(3)
        res[c] = [[k, round(float(cors[k]), 3)] for k in top.index]
    (OUT / "overlap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
