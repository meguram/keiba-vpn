"""牝系の家族の版の、母の側に情報が無い行の記述（log.md「調べ D」で先に決めた記述の群。dam_all_n = 0 かつ dd_all_n = 0）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.family_group

- 比べ先は同じ窓の (A)（base2018_wf）。行は compare_null と同じ（開発の行・父が空でない行・2020〜24 年）。区間は開催日のブートストラップ。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "family_group", "家族の版の母の側に情報が無い行の記述（尤度だけ）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "split", "dam_all_n", "dd_all_n"])
    T = pd.read_parquet(data.OUT / "base2018_wf.parquet").rename(columns={"p_TB": "pa"}).merge(
        pd.read_parquet(data.OUT / "family_wf.parquet"), on=["race_id", "horse_id"]).merge(X, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")]
    w = T["win"].to_numpy(float)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(w * np.log(p) + (1 - w) * np.log(1 - p))
    g = ll(T["pa"].to_numpy()) - ll(T["p_TB"].to_numpy())
    mk = ((T["dam_all_n"] == 0) & (T["dd_all_n"] == 0)).to_numpy()
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, di = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    d = np.bincount(di[mk], weights=g[mk], minlength=len(du))
    n = np.bincount(di[mk], minlength=len(du)).astype(float)
    bs = (Wb @ d) / (Wb @ n)
    res = {"行": int(mk.sum()), "版 − (A)": round(float(g[mk].mean()), 5), "95%": np.percentile(bs, [2.5, 97.5]).round(5).tolist()}
    print(res, flush=True)
    path = data.OUT / "family_group.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
