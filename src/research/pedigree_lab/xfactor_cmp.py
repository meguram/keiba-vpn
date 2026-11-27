"""X 因子の版の比べ（log.md「公開の手法を規則で作った列」）。父の母の父の列を、牝の馬に入れた版 X と牡・騸に入れた版 XP で比べる。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.xfactor_cmp

- 牝の行での X の改善（X − (A)）と、牡・騸の行での XP の改善（XP − (A)）の差も出す。ただし (A) に性の列が無く、列の欠けが性の印になるので、
  この差は性の印の効きの違いを含み、X 因子の分としては読めない（critic）。読むのは牝の行の X − 帰無。
- 牝の行の X − 帰無（借り手の血統・1 種）も出す。区間は開催日のブートストラップ（1,000 回・種 3）。行は compare_null と同じ。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "xfactor_cmp", "X 因子の版の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "split", "sex"])
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)]
    T = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    for nm, f in (("X", "public_X_wf.parquet"), ("XP", "public_XP_wf.parquet"), ("XN", "public_X_wf_null1.parquet")):
        T = T.merge(pd.read_parquet(data.OUT / f)[["race_id", "horse_id", "p_TB"]].rename(columns={"p_TB": nm}), on=["race_id", "horse_id"])
    T = T.merge(X, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    w = T["win"].to_numpy(float)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(w * np.log(p) + (1 - w) * np.log(1 - p))
    fem = T["sex"].astype(str).eq("牝").to_numpy()
    lA = ll(T["p_A"].to_numpy())
    gX, gXP, gXN = lA - ll(T["X"].to_numpy()), lA - ll(T["XP"].to_numpy()), lA - ll(T["XN"].to_numpy())
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)

    def mean_bs(v, mk):
        d = np.bincount(dinv[mk], weights=v[mk], minlength=len(du))
        n = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        return d.sum() / n.sum(), (Wb @ d) / (Wb @ n)
    pf, bf = mean_bs(gX, fem)
    pm, bm = mean_bs(gXP, ~fem)
    pn, bn = mean_bs(gX - gXN, fem)
    res = {"rows_f": int(fem.sum()), "rows_m": int((~fem).sum())}
    for k, (pt, bs) in {"牝の行の X − (A)": (pf, bf), "牡・騸の行の XP − (A)": (pm, bm),
                        "差（性の印の分を含む）": (pf - pm, bf - bm), "牝の行の X − 帰無": (pn, bn)}.items():
        res[k] = [round(float(pt), 6), np.percentile(bs, [2.5, 97.5]).round(6).tolist()]
        print(k, res[k], flush=True)
    path = data.OUT / "xfactor_cmp.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
