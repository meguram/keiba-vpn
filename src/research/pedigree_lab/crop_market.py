"""2〜3 年目の父の産駒の印を、市場の見込みに対して別の期間で確かめる（log.md「調べ E」。記述・買い方の設計ではない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.crop_market

- 行: 開発の行（split 0・1）の 2017〜19 年（確かめ）と 2020〜24 年（見つけた期間）。世代 0（その父の一番早い世代）の行は除く。
- 印: 2・3 番目の世代の産駒（生年 − その父の一番早い世代の生年が 1 か 2。一番早い世代が 2013 年生まれ以降の父だけ）。初年度の世代の行は除く。q は確定の単勝オッズの逆数をレースの中で和 1 に直した値。
- 主: 全帯で 1 つの回帰（勝ち ~ 帯ごとの 1・logit(q)・logit(q)² + 印）の印の係数。帯ごとの係数も出す。区間は種牡馬の単位のブートストラップ（200 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls

BANDS = ["1〜3", "4〜6", "7〜9", "10〜"]


def design(D: pd.DataFrame, pooled: bool) -> np.ndarray:
    lq = D["lq"].to_numpy()
    cols = []
    if pooled:
        for b in BANDS:
            m = (D["band"] == b).to_numpy(float)
            cols += [m, m * lq, m * lq ** 2]
    else:
        cols += [np.ones(len(D)), lq, lq ** 2]
    cols.append(D["new"].to_numpy(float))
    return np.column_stack(cols)


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "crop_market", "2〜3 年目の父の産駒の印と市場の見込み（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "win", "odds", "popularity"])
    X = X[X["date"] <= data.DEV_END].copy()
    X["y"] = X["date"].dt.year
    od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
    inv = 1 / od
    X["q"] = inv / inv.groupby(X["race_id"]).transform("sum")
    X = X[X["sire"].fillna("").ne("") & X["q"].notna()].copy()
    q = X["q"].clip(1e-4, 1 - 1e-4)
    X["lq"] = np.log(q / (1 - q))
    # 何番目の世代か = 生年 − その父の一番早い世代の生年（手元の行は 2016 年からなので、一番早い世代が 2013 年生まれ以降の父だけ数える）。
    # 暦の上の初出で決めると、2017・18 年はすべての父が「新しい」になる（手元の行が 2016 年から）ので、世代で決める
    X["by"] = pd.to_numeric(X["horse_id"].str[:4], errors="coerce")
    fc = X.groupby("sire")["by"].min()
    X["crop"] = X["by"] - X["sire"].map(fc)
    X["known"] = X["sire"].map(fc) >= 2013
    X = X[X["split"].isin([0, 1]) & (X["crop"] >= 1)].reset_index(drop=True)   # 世代 0 の行は除く（調べ B の除き方〔暦の初出の年の行〕とは違う）
    X["new"] = (X["crop"].between(1, 2) & X["known"]).astype(int)
    X["band"] = pd.cut(pd.to_numeric(X["popularity"], errors="coerce"), [0, 3, 6, 9, 99], labels=BANDS).astype(str)
    X = X[X["band"].isin(BANDS)].reset_index(drop=True)
    rng = np.random.default_rng(3)
    res = {}
    for per, (a, b) in (("2017〜19 年（確かめ）", (2017, 2019)), ("2020〜24 年（見つけた期間）", (2020, 2024))):
        D0 = X[X["y"].between(a, b)].reset_index(drop=True)
        out = {"走り": int(len(D0)), "印の割合": round(float(D0["new"].mean()), 3)}
        for nm, D, pooled in [("全帯（主）", D0, True)] + [(bd, D0[D0["band"] == bd].reset_index(drop=True), False) for bd in BANDS]:
            y = D["win"].to_numpy(float)
            coef = irls(design(D, pooled), y)[-1]
            sires = D["sire"].to_numpy()
            su = np.unique(sires)
            idx = {s: np.flatnonzero(sires == s) for s in su}
            bs = []
            for _ in range(200):
                ii = np.concatenate([idx[s] for s in rng.choice(su, len(su))])
                bs.append(irls(design(D.iloc[ii], pooled), y[ii])[-1])
            out[nm] = [round(float(coef), 3), np.percentile(bs, [2.5, 97.5]).round(3).tolist(), int(y.sum())]
        res[per] = out
        print(per, out, flush=True)
    path = data.OUT / "crop_market.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
