"""データ版の国別の型の 2 本目の線: 父の群を父の父から受け継いだ行の版 − (A)（log.md「調べ F」）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.sire_type_inh
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import data


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "sire_type_inh", "受け継いだ行の版 − (A)（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "split"])
    T = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]].merge(
        pd.read_parquet(data.OUT / "sire_type_wf.parquet"), on=["race_id", "horse_id"]).merge(X, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")]
    w = T["win"].to_numpy(float)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(w * np.log(p) + (1 - w) * np.log(1 - p))
    g = ll(T["p_A"].to_numpy()) - ll(T["p_TB"].to_numpy())
    mk = T["_inh"].to_numpy(bool)
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, di = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    d = np.bincount(di[mk], weights=g[mk], minlength=len(du))
    n = np.bincount(di[mk], minlength=len(du)).astype(float)
    bs = (Wb @ d) / (Wb @ n)
    pt, se, zb = float(d.sum() / n.sum()), float(bs.std()), float(norm.ppf(1 - 0.025 / 30))
    res = {"行": int(mk.sum()), "版 − (A)": round(pt, 5), "95%": np.percentile(bs, [2.5, 97.5]).round(5).tolist(),
           "割り引き（30 本）": [round(pt - zb * se, 5), round(pt + zb * se, 5)]}
    print(res, flush=True)
    path = data.OUT / "sire_type_inh.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
