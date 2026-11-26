"""z の傾きを、レースの中の条件付きロジットで見直す（log.md「調べ T」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_clogit

- 行は z_bands.py の load と同じ（2019〜24 年の開発の行・235,714 走）。
- 版 1（Benter の 2 段の形）: 勝ち ~ β log q + α log p_A（レースの中）。
- 版 3（2 次も入れる）: 勝ち ~ log q + (log q)² + z（z = log p_A − log q）。二値のロジット（調べ Q の +0.08）と比べる。
- 区間は開催日の単位のブートストラップ（200 回）。同じ日が 2 回引かれたら、そのレースは別の群として数える。
"""
from __future__ import annotations

import json

import numpy as np

from src.research.pedigree_lab import data
from src.research.pedigree_lab.us_dirt_wet import clogit
from src.research.pedigree_lab.z_bands import load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_clogit", "z の傾きの条件付きロジット（記述）"))
    X = load()
    rid = X["race_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
    y = X["win"].to_numpy(float)
    lq = np.log(X["q"].clip(1e-4, 1).to_numpy())
    lp = np.log(X["pA"].clip(1e-6, 1).to_numpy())
    z = X["z"].to_numpy()
    lqc = lq + 2.6
    designs = {"版 1（log q・log p_A）": np.column_stack([lq, lp]),
               "版 3（log q・(log q)²・z）": np.column_stack([lqc, lqc ** 2, z])}
    du, di = np.unique(X["date"].dt.strftime("%Y%m%d"), return_inverse=True)
    idx = [np.flatnonzero(di == j) for j in range(len(du))]
    rng = np.random.default_rng(13)
    draws = [rng.integers(0, len(du), len(du)) for _ in range(200)]
    res = {"走り": int(len(X)), "レース": int(rid.max() + 1)}
    for nm, Z in designs.items():
        b0 = clogit(Z, y, rid)
        bs = []
        for dr in draws:
            ii = np.concatenate([idx[j] for j in dr])
            t = np.concatenate([np.full(len(idx[j]), k, dtype=np.int64) for k, j in enumerate(dr)])
            g = np.unique(rid[ii] * 100000 + t, return_inverse=True)[1]
            bs.append(clogit(Z[ii], y[ii], g))
        bs = np.array(bs)
        res[nm] = {f"係数{c}": [round(float(b0[c]), 4), np.percentile(bs[:, c], [2.5, 97.5]).round(4).tolist()] for c in range(Z.shape[1])}
        print(nm, json.dumps(res[nm], ensure_ascii=False), flush=True)
    path = data.OUT / "z_clogit.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
