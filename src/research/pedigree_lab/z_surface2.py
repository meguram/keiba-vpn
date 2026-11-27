"""調べ U2: 芝ダの違いを、レースの中の条件付きロジットの α と nats の改善で見直す・初ダートの切り分け（記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_surface2

- (1) 芝・ダートそれぞれで、勝ち ~ β log q（市場だけ）と 勝ち ~ β log q + α log p_A を当て、
      α と 1 レースあたりの対数尤度の改善（nats）を出す。差（ダート − 芝）を開催日の単位で 200 回引き直す（2 号）。
- (2) ダートの行を初ダート（first_surf == "Y"）とダート経験ありに分け、z の傾きの差（z_bands.fit・500 回）を出す（1 号）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.us_dirt_wet import clogit
from src.research.pedigree_lab.z_bands import fit, load


def cll(Z: np.ndarray, y: np.ndarray, rid: np.ndarray, b: np.ndarray) -> float:
    eta = Z @ b
    nr = rid.max() + 1
    m = np.full(nr, -np.inf); np.maximum.at(m, rid, eta)
    s = np.bincount(rid, weights=np.exp(eta - m[rid]), minlength=nr)
    wr = np.bincount(rid, weights=y, minlength=nr)                       # 同着の 2 頭は log s を 2 回引く（clogit の目的関数と同じ・critic）
    return float((y * (eta - m[rid])).sum() - (wr * np.log(s)).sum())


def one(D: pd.DataFrame, rid: np.ndarray) -> tuple[float, float]:
    y = D["win"].to_numpy(float)
    lq = np.log(D["q"].clip(1e-4, 1).to_numpy())[:, None]
    Z = np.column_stack([lq[:, 0], np.log(D["pA"].clip(1e-6, 1).to_numpy())])
    b1 = clogit(lq, y, rid); b2 = clogit(Z, y, rid)
    nr = rid.max() + 1
    return float(b2[1]), (cll(Z, y, rid, b2) - cll(lq, y, rid, b1)) / nr


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_surface2", "芝ダの α と nats・初ダート（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface", "first_surf"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    res = {}
    parts = {}
    for sf in ("芝", "ダート"):
        D = X[X["surface"] == sf].reset_index(drop=True)
        rid = D["race_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
        du, di = np.unique(D["date"].dt.strftime("%Y%m%d"), return_inverse=True)
        parts[sf] = (D, rid, du, [np.flatnonzero(di == j) for j in range(len(du))])
    alldays = sorted(set(parts["芝"][2]) | set(parts["ダート"][2]))
    rng = np.random.default_rng(19)
    draws = [rng.integers(0, len(alldays), len(alldays)) for _ in range(200)]
    pt = {sf: one(parts[sf][0], parts[sf][1]) for sf in parts}
    bs = {sf: [] for sf in parts}
    for dr in draws:
        days = [alldays[k] for k in dr]
        for sf, (D, rid, du, idx) in parts.items():
            pos = {d: j for j, d in enumerate(du)}
            sel = [(t, pos[d]) for t, d in enumerate(days) if d in pos]
            ii = np.concatenate([idx[j] for _, j in sel])
            tt = np.concatenate([np.full(len(idx[j]), t, dtype=np.int64) for t, j in sel])
            g = np.unique(rid[ii] * 100000 + tt, return_inverse=True)[1]
            bs[sf].append(one(D.iloc[ii].reset_index(drop=True), g))
    A = {sf: np.array(v) for sf, v in bs.items()}
    res["(1) 条件付きロジット"] = {sf: {"α": [round(pt[sf][0], 4), np.percentile(A[sf][:, 0], [2.5, 97.5]).round(4).tolist()],
                                     "1 レースあたりの nats の改善": [round(pt[sf][1], 5), np.percentile(A[sf][:, 1], [2.5, 97.5]).round(5).tolist()],
                                     "レース": int(parts[sf][1].max() + 1)} for sf in parts}
    res["(1) 差（ダート − 芝）"] = {"α": [round(pt["ダート"][0] - pt["芝"][0], 4), np.percentile(A["ダート"][:, 0] - A["芝"][:, 0], [2.5, 97.5]).round(4).tolist()],
                                 "nats": [round(pt["ダート"][1] - pt["芝"][1], 5), np.percentile(A["ダート"][:, 1] - A["芝"][:, 1], [2.5, 97.5]).round(5).tolist()]}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    D = X[X["surface"] == "ダート"].reset_index(drop=True)
    res["(2) ダートの中の初ダート"] = fit(D, D["first_surf"].astype(str).eq("Y").to_numpy().astype(int), ["ダート経験あり", "初ダート"], np.random.default_rng(23))
    print(json.dumps(res["(2) ダートの中の初ダート"], ensure_ascii=False), flush=True)
    path = data.OUT / "z_surface2.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
