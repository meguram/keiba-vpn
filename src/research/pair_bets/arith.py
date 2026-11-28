"""段 1 (a): 単勝の上乗せを、馬連・ワイドの組に伸ばす上限の算数（着順・払戻は読まない）。

    KEIBA_SESSION=pair-bets nice -n 19 python -m src.research.pair_bets.arith

- 元: 会議の土台の表（course_out/rules_council/full.parquet）の race_id・year・q（確定の単勝の和 1）・pop_final だけ。
- 植え方: 軸の馬 a の勝率を、和 1 に直した後で q_a × r になるように置き、ほかの馬は同じ割合で縮める（lessons「植える効きは和 1 の後」）。
  「両方」は、軸 a と相手 b の両方に r を植える。
- 組の確率は、同じ型（Harville・LBS）で p と q から作り、その比を「組の上乗せ」とする。型の偏りは比で消える（市場が単勝の q と同じ型で組を値付けしたときの上限）。
- 要る倍率: 馬連・ワイドの控除 22.5% から 1/0.775 ≈ 1.29。判定のオッズの楽観 s* 0.86（単勝の T-60）を掛けると 1.50。
- 重みは q からの組の確率（組の市場に比例して買う形）。帯は軸と相手の確定の人気。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council.place_w import LBS, ordered_probs

FULL = Path(os.environ.get("RC_FULL") or "/home/jovyan/work/course_out/rules_council/full.parquet")
OUT = Path(__file__).resolve().parents[3] / "docs/operations/pair_bets/arith.json"
R_LIST = (1.07, 1.15, 1.28)
MODELS = {"harville": (1.0, 1.0, 1.0), "lbs": LBS}
POP_BANDS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 99)]
N_RACES = 1500
TAKE = 0.775
S_STAR = 0.86


def band(pop: float) -> int:
    for i, (lo, hi) in enumerate(POP_BANDS):
        if lo <= pop <= hi:
            return i
    return len(POP_BANDS) - 1


def pair_tables(p: np.ndarray, lam) -> tuple[np.ndarray, np.ndarray]:
    """馬連（1・2 着の順なし）とワイド（3 着内の 2 頭）の組の確率（n×n、対称）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        P3 = np.nan_to_num(ordered_probs(p, 3, lam))
    P2 = P3.sum(axis=2)
    umaren = P2 + P2.T
    w = P3.sum(axis=2) + P3.sum(axis=1) + P3.sum(axis=0)         # i,j が (1,2)・(1,3)・(2,3) の位置
    wide = w + w.T
    return umaren, wide


def plant(q: np.ndarray, idx: list[int], r: float) -> np.ndarray:
    """idx の馬が和 1 の後で q × r になるように置く。"""
    p = q.copy()
    tgt = q[idx] * r
    rest = np.setdiff1d(np.arange(len(q)), idx)
    p[rest] = q[rest] * (1 - tgt.sum()) / q[rest].sum()
    p[idx] = tgt
    return p


def main() -> None:
    d = pd.read_parquet(FULL, columns=["race_id", "year", "q", "pop_final"], filters=[("year", ">=", 2022), ("year", "<=", 2024)])
    assert d["year"].max() <= 2024
    rng = np.random.default_rng(20261127)
    rids = d["race_id"].unique()
    pick = set(rng.choice(rids, size=min(N_RACES, len(rids)), replace=False))
    d = d[d["race_id"].isin(pick)]
    nb = len(POP_BANDS)
    # acc[kind][model][scenario][r] = (num, den) の [軸の帯, 相手の帯]
    acc = {k: {m: {s: {r: np.zeros((2, nb, nb)) for r in R_LIST} for s in ("axis", "both")} for m in MODELS} for k in ("umaren", "wide")}
    for _, g in d.groupby("race_id"):
        q = g["q"].to_numpy(float)
        if len(q) < 5 or not np.isfinite(q).all():
            continue
        q = q / q.sum()
        bands = np.array([band(x) for x in np.nan_to_num(g["pop_final"].to_numpy(float), nan=99)])
        n = len(q)
        for m, lam in MODELS.items():
            base_u, base_w = pair_tables(q, lam)
            for r in R_LIST:
                for a in range(n):
                    if q[a] * r >= 0.95:
                        continue
                    pu, pw = pair_tables(plant(q, [a], r), lam)
                    for kind, base, pp in (("umaren", base_u, pu), ("wide", base_w, pw)):
                        A = acc[kind][m]["axis"][r]
                        for b in range(n):
                            if b == a:
                                continue
                            A[0, bands[a], bands[b]] += pp[a, b]
                            A[1, bands[a], bands[b]] += base[a, b]
                    for b in range(a + 1, n):
                        if (q[a] + q[b]) * r >= 0.95:
                            continue
                        pu2, pw2 = pair_tables(plant(q, [a, b], r), lam)
                        ba, bb = sorted((bands[a], bands[b]))
                        for kind, base, pp in (("umaren", base_u, pu2), ("wide", base_w, pw2)):
                            A = acc[kind][m]["both"][r]
                            A[0, ba, bb] += pp[a, b]
                            A[1, ba, bb] += base[a, b]
    names = [f"{lo}" if lo == hi else (f"{lo}〜{hi}" if hi < 99 else f"{lo}〜") for lo, hi in POP_BANDS]
    res = {"要る倍率": {"控除だけ": round(1 / TAKE, 3), "s* 0.86 の後": round(1 / TAKE / S_STAR, 3)}, "races": len(pick), "bands": names, "table": {}}
    for kind in acc:
        for m in acc[kind]:
            for s in acc[kind][m]:
                for r in R_LIST:
                    A = acc[kind][m][s][r]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = A[0] / A[1]
                    tot = float(A[0].sum() / A[1].sum())
                    res["table"][f"{kind}|{m}|{s}|{r}"] = {"全体": round(tot, 4), "帯": np.round(np.nan_to_num(ratio), 4).tolist()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res["table"].items():
        print(k, v["全体"], v["帯"][0][:3] if "axis" in k else "")


if __name__ == "__main__":
    main()
