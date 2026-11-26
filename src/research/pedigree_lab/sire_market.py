"""種牡馬の価値（賞金で見た as-of の av_sire）が市場で消えるか・人気の帯ごと（log.md「延長の続き」の調べ B。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.sire_market

- 行: 2020〜24 年の開発の行（split 0・1）。q = 確定の単勝オッズの逆数をレースの中で和 1 に直した値。
- 3 分位: av_sire（aei_cols.parquet）の、前の年の行の 1/3・2/3 点で切る（as-of）。学習の期間に父の産駒が無い行は除く。
- 帯ごと（1〜3・4〜6・7〜9・10 番人気以下）× 層（全レース・新馬戦・それ以外）に、勝ち ~ logit(q) + logit(q)² + 中の分位 + 上の分位
  のロジスティック回帰を当て、上の分位の係数（下の分位に対する対数オッズの差）を読む（本命と大穴の偏りを logit(q) の 2 次で吸う。2 号）。
  区間は種牡馬の単位のブートストラップ（200 回）。
- 2〜3 年目の産駒（父の初出が前の年か前々の年）の印を、同じ回帰に別に入れた版も出す（1 号の「初年度・2 年目」は、初年度を除く決まりと両立しないので 1 年ずれる）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data


def irls(Xm: np.ndarray, y: np.ndarray, it: int = 25) -> np.ndarray:
    b = np.zeros(Xm.shape[1])
    for _ in range(it):
        eta = np.clip(Xm @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        H = Xm.T @ (Xm * W[:, None]) + 1e-6 * np.eye(Xm.shape[1])
        step = np.linalg.solve(H, Xm.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def design(D: pd.DataFrame, with_new: bool) -> np.ndarray:
    lq = D["lq"].to_numpy()
    cols = [np.ones(len(D)), lq, lq ** 2, (D["t"] == 1).to_numpy(float), (D["t"] == 2).to_numpy(float)]
    if with_new:
        cols.append(D["new"].to_numpy(float))
    return np.column_stack(cols)


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "sire_market", "種牡馬の価値と市場の勝ち確率の比べ（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "win", "odds", "popularity", "race_class"])
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "av_sire"])
    X = X.merge(C, on=["race_id", "horse_id"])
    X = X[X["date"] <= data.DEV_END].copy()
    X["y"] = X["date"].dt.year
    # q は父の空の行を落とす前のレースの全頭で和 1 に直す。オッズが 0 以下の行は欠け（critic）
    od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
    inv = 1 / od
    X["q"] = inv / inv.groupby(X["race_id"]).transform("sum")
    X = X[X["sire"].fillna("").ne("") & X["av_sire"].notna() & X["q"].notna()].copy()
    fr = X.groupby("sire")["date"].min()
    X["first"] = X["sire"].map(fr)
    q = X["q"].clip(1e-4, 1 - 1e-4)
    X["lq"] = np.log(q / (1 - q))
    cut = X.groupby("y")["av_sire"].quantile([1 / 3, 2 / 3]).unstack()
    lo = X["y"].map(lambda y: cut.loc[y - 1, 1 / 3] if y - 1 in cut.index else np.nan)
    hi = X["y"].map(lambda y: cut.loc[y - 1, 2 / 3] if y - 1 in cut.index else np.nan)
    X["t"] = np.where(X["av_sire"] >= hi, 2, np.where(X["av_sire"] >= lo, 1, 0))
    yr0 = pd.to_datetime(X["y"].astype(str) + "-01-01")
    X = X[X["split"].isin([0, 1]) & X["y"].between(2020, 2024) & (X["first"] < yr0) & X["lq"].notna()].reset_index(drop=True)
    # 初年度の産駒は「学習の期間に産駒が無い父」として除くので、印は 2〜3 年目の産駒（初出が前の年か前々の年）を指す（critic）
    yr0 = pd.to_datetime(X["y"].astype(str) + "-01-01")              # 絞った後の行で作り直す（添字のずれ）
    X["new"] = (X["first"] >= yr0 - pd.DateOffset(years=2)).astype(int)
    X["debut"] = X["race_class"].astype(str).str.contains("新馬")
    pop = pd.to_numeric(X["popularity"], errors="coerce")
    X["band"] = pd.cut(pop, [0, 3, 6, 9, 99], labels=["1〜3", "4〜6", "7〜9", "10〜"])
    rng = np.random.default_rng(3)
    res = {"rows": int(len(X)), "新しい父の行の割合": round(float(X["new"].mean()), 3), "結果": {}}
    for layer, mk in (("全レース", np.ones(len(X), bool)), ("新馬戦", X["debut"].to_numpy()), ("それ以外", ~X["debut"].to_numpy())):
        for band in ["1〜3", "4〜6", "7〜9", "10〜"]:
            D = X[mk & (X["band"] == band).to_numpy()].reset_index(drop=True)
            if D["win"].sum() < 50:
                res["結果"][f"{layer}・{band}"] = {"走り": int(len(D)), "勝ち": int(D["win"].sum()), "飛ばした": "勝ちが 50 未満"}
                continue
            y = D["win"].to_numpy(float)
            b = irls(design(D, False), y)
            bn = irls(design(D, True), y)
            sires = D["sire"].to_numpy()
            su = np.unique(sires)
            idx = {s: np.flatnonzero(sires == s) for s in su}
            bs, bsn = [], []
            for _ in range(200):
                pick = rng.choice(su, len(su))
                ii = np.concatenate([idx[s] for s in pick])
                Db = D.iloc[ii]
                yb = y[ii]
                bs.append(irls(design(Db, False), yb)[4])
                bsn.append(irls(design(Db, True), yb)[5])
            r = {"走り": int(len(D)), "勝ち": int(y.sum()),
                 "上の分位 − 下の分位（対数オッズ）": [round(float(b[4]), 3), np.percentile(bs, [2.5, 97.5]).round(3).tolist()],
                 "中の分位 − 下の分位": round(float(b[3]), 3),
                 "2〜3 年目の産駒の印": [round(float(bn[5]), 3), np.percentile(bsn, [2.5, 97.5]).round(3).tolist()]}
            res["結果"][f"{layer}・{band}"] = r
            print(layer, band, r, flush=True)
    path = data.OUT / "sire_market.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
