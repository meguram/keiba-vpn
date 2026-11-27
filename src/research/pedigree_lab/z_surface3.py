"""調べ U3: 芝ダの z の傾きの差を、3 次の市場の曲線 × クラスの群で切り分ける（記述・2 号の確かめの 2 本目）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_surface3

- 帯 = 芝ダ × 群（新馬・未勝利・1 勝以上）の 6 つ。帯ごとに 1・z・logit(q)・logit(q)²・logit(q)³ を入れる（基準の帯も同じ）。
- 群ごとの差（ダート − 芝）と、群の件数で重みを付けた平均の差。区間は開催日の単位（300 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls
from src.research.pedigree_lab.z_bands import load

GROUPS = ["新馬", "未勝利", "1 勝以上"]


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_surface3", "芝ダ × クラスの群・3 次の曲線（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface", "cls"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    assert X["surface"].notna().all() and X["cls"].notna().all()
    X = X[X["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    assert len(X) == n0                                                      # 今の行は芝ダだけ（障害は行に無い）
    g = np.where(X["cls"].eq("NEW"), 0, np.where(X["cls"].eq("MDN"), 1, 2))
    band = g * 2 + X["surface"].eq("ダート").to_numpy().astype(int)   # 0 新馬芝・1 新馬ダ・2 未勝利芝・…
    lq, z = X["lq"].to_numpy(), X["z"].to_numpy()
    cols, zpos = [], []
    for b in range(6):
        ind = (band == b).astype(float)
        zpos.append(len(cols) + 1)
        cols += [ind, z * ind, lq * ind, lq ** 2 * ind, lq ** 3 * ind]
    M = np.column_stack(cols); y = X["win"].to_numpy(float)
    n = np.bincount(band, minlength=6)
    w = np.array([n[2 * k] + n[2 * k + 1] for k in range(3)], float); w /= w.sum()

    def summ(c):
        sl = np.array([c[p] for p in zpos])
        d = np.array([sl[2 * k + 1] - sl[2 * k] for k in range(3)])
        return np.r_[sl, d, (d * w).sum()]
    s0 = summ(irls(M, y))
    du, di = np.unique(X["date"].dt.strftime("%Y%m%d"), return_inverse=True)
    idx = [np.flatnonzero(di == j) for j in range(len(du))]
    rng = np.random.default_rng(29)
    bs = np.array([summ(irls(M[ii], y[ii])) for ii in (np.concatenate([idx[j] for j in rng.integers(0, len(du), len(du))]) for _ in range(300))])
    ci = lambda k: [round(float(s0[k]), 4), np.percentile(bs[:, k], [2.5, 97.5]).round(4).tolist()]
    res = {"走り": {f"{GROUPS[k // 2]}・{'ダート' if k % 2 else '芝'}": int(n[k]) for k in range(6)},
           "z の傾き": {f"{GROUPS[k // 2]}・{'ダート' if k % 2 else '芝'}": ci(k) for k in range(6)},
           "差（ダート − 芝）": {GROUPS[k]: ci(6 + k) for k in range(3)},
           "群の件数で重みを付けた差": ci(9)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_surface3.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
