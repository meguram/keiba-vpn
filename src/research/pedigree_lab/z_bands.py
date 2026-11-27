"""血統と市場の差 z の効きが、情報の確かさ・2 歳の時期・産地で変わるか（log.md「調べ Q」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_bands

- 行: 開発の行（split 0・1）の 2019〜24 年の全出走（p_A のある行）。q は確定オッズの逆数をレースの中で和 1。
- z = log p_A − log q（p_A はレースの中で和 1）。
- 回帰: 勝ち ~ logit(q) + logit(q)² + z + Σ_帯 (1[帯] + z × 1[帯] + logit(q) × 1[帯] + logit(q)² × 1[帯])（帯ごとの市場の曲線を入れる）。帯ごとの z の傾きと、差を出す。
- 帯（4 本の差・数を見る前に固定）:
  (1) 父の産駒の as-of の走りの数 sire_all_n の年ごとの 3 分位: 多い − 少ない
  (2) 2 歳の行だけで、6〜9 月 − 10〜12 月
  (3a) 外国産（horse_id の 5・6 桁目が 11）− 内国産
  (3b) 地方からの転入は外した（JRA の初出走が 1 勝クラス以上では、3 歳秋のデビュー・地方の遠征馬と分けられない・表の前の初出走が漏れる。2 号・1 号）
- 区間は開催日の単位のブートストラップ（500 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls


def load() -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "odds", "sire_all_n", "age"])
    X = X[X["date"] <= data.DEV_END]
    X["foreign"] = X["horse_id"].astype(str).str[4:6].eq("11")
    P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    X = X.merge(P, on=["race_id", "horse_id"])
    X["odds"] = pd.to_numeric(X["odds"], errors="coerce")
    X = X[X["odds"] > 0].copy()
    X["q"] = (1 / X["odds"]) / (1 / X["odds"]).groupby(X["race_id"]).transform("sum")
    X["pA"] = X["p_A"] / X.groupby("race_id")["p_A"].transform("sum")
    X = X[X["split"].isin([0, 1])].copy()
    qq = X["q"].clip(1e-4, 1 - 1e-4)
    X["lq"] = np.log(qq / (1 - qq))
    X["z"] = np.log(X["pA"].clip(1e-6)) - np.log(X["q"])
    X["y"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    return X.reset_index(drop=True)


def fit(D: pd.DataFrame, band: np.ndarray, names: list[str], rng, nb: int = 500) -> dict:
    """band は 0..k-1。基準の帯 0 に z、他の帯に z × 帯と帯の主効果。"""
    lq, z = D["lq"].to_numpy(), D["z"].to_numpy()
    cols = [np.ones(len(D)), lq, lq ** 2, z]
    k = int(band.max()) + 1
    for b in range(1, k):
        ind = (band == b).astype(float)
        cols += [ind, z * ind, lq * ind, lq ** 2 * ind]   # 帯ごとの市場の曲線も入れる（2 号）
    M = np.column_stack(cols); y = D["win"].to_numpy(float)

    def slopes(c):
        s = [c[3]] + [c[3] + c[5 + 4 * (b - 1)] for b in range(1, k)]
        return s
    s0 = slopes(irls(M, y))
    du, di = np.unique(D["date"].dt.strftime("%Y%m%d"), return_inverse=True)
    idx = [np.flatnonzero(di == j) for j in range(len(du))]
    bs = []
    for _ in range(nb):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(du), len(du))])
        bs.append(slopes(irls(M[ii], y[ii])))
    bs = np.array(bs)
    out = {"走り": {names[b]: int((band == b).sum()) for b in range(k)},
           "勝ち": {names[b]: int(y[band == b].sum()) for b in range(k)},
           "z の sd・lq との相関": {names[b]: [round(float(z[band == b].std()), 3), round(float(np.corrcoef(z[band == b], lq[band == b])[0, 1]), 3)] for b in range(k)},
           "z の傾き": {names[b]: [round(float(s0[b]), 4), np.percentile(bs[:, b], [2.5, 97.5]).round(4).tolist()] for b in range(k)}}
    d = bs[:, k - 1] - bs[:, 0]
    out[f"差（{names[k - 1]} − {names[0]}）"] = [round(float(s0[k - 1] - s0[0]), 4), np.percentile(d, [2.5, 97.5]).round(4).tolist()]
    return out


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_bands", "z の効きの帯ごとの違い（記述）"))
    X = load()
    rng = np.random.default_rng(5)
    res = {"行": int(len(X))}
    t = X.groupby("y")["sire_all_n"].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False)).to_numpy().astype(int)
    res["(1) 父の産駒の件数（3 分位）"] = fit(X, t, ["少ない", "中", "多い"], rng)
    print(json.dumps(res["(1) 父の産駒の件数（3 分位）"], ensure_ascii=False), flush=True)
    D = X[X["age"].eq(2) & X["month"].ge(6)].reset_index(drop=True)
    res["(2) 2 歳の時期"] = fit(D, (D["month"] <= 9).to_numpy().astype(int), ["10〜12 月", "6〜9 月"], rng)
    print(json.dumps(res["(2) 2 歳の時期"], ensure_ascii=False), flush=True)
    res["(3a) 外国産"] = fit(X, X["foreign"].to_numpy().astype(int), ["内国産", "外国産"], rng)
    print(json.dumps(res["(3a) 外国産"], ensure_ascii=False), flush=True)
    path = data.OUT / "z_bands.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
