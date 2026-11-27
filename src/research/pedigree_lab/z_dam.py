"""z の効きと母の側の情報の確かさ・z の切り詰めの確かめ（log.md「調べ S」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_dam

- 行と式は z_bands.py と同じ（帯ごとの市場の曲線を入れる）。
- (a) 母の子の as-of の走りの数 dam_all_n が 0（兄姉が走っていない）− 1 以上 の z の傾きの差。
- (b) z を全体の 1%・99% 点で切り詰めた版の、全体の z の傾き（切り詰めない版と並べる）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.z_bands import fit, load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_dam", "z の効きと母の側の情報・切り詰め（記述）"))
    X = load()
    D = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "dam_all_n"])
    n0 = len(X)
    X = X.merge(D, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    assert X["dam_all_n"].notna().all()
    rng = np.random.default_rng(11)
    res = {"行": int(len(X))}
    res["(a) 母の子の走り 0"] = fit(X, X["dam_all_n"].eq(0).to_numpy().astype(int), ["1 以上", "0"], rng)
    print(json.dumps(res["(a) 母の子の走り 0"], ensure_ascii=False), flush=True)
    lo, hi = np.percentile(X["z"], [1, 99])
    out = {}
    for nm, zz in (("切り詰めない", X["z"]), ("1%・99% で切り詰め", X["z"].clip(lo, hi))):
        out[nm] = fit_one(X.assign(z=zz), rng)
    res["(b) z の切り詰め"] = out | {"切り詰めの点": [round(float(lo), 3), round(float(hi), 3)]}
    print(json.dumps(res["(b) z の切り詰め"], ensure_ascii=False), flush=True)
    path = data.OUT / "z_dam.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


def fit_one(D: pd.DataFrame, rng, nb: int = 300) -> list:
    from src.research.pedigree_lab.sire_market import irls
    lq, z = D["lq"].to_numpy(), D["z"].to_numpy()
    M = np.column_stack([np.ones(len(D)), lq, lq ** 2, z]); y = D["win"].to_numpy(float)
    b0 = irls(M, y)[3]
    du, di = np.unique(D["date"].dt.strftime("%Y%m%d"), return_inverse=True)
    idx = [np.flatnonzero(di == j) for j in range(len(du))]
    bs = [irls(M[ii], y[ii])[3] for ii in (np.concatenate([idx[j] for j in rng.integers(0, len(du), len(du))]) for _ in range(nb))]
    return [round(float(b0), 4), np.percentile(bs, [2.5, 97.5]).round(4).tolist()]


if __name__ == "__main__":
    main()
