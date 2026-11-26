"""券種の間の歪みと血統（log.md「調べ P」と「流す前の直し」。記述・買い方の設計ではない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.pair_market

- 元: data/page_reference/tables/{2020..2024}/race_pair_odds_flat.parquet の馬連・馬単（確定オッズ。2025・26 年は開けない）。
  ワイドは、3 着までの組で 1 レースに 3 組が当たり、払い戻しが幅なので外した（critic・2 号・市場の 2 号）。
- 組: 開発の行のレースで、1・2 着に同着の無いレース。組の 2 頭のうち、ちょうど 1 頭が単勝 6 番人気以下（人気薄）で、相手が 1〜5 番人気。
- H: 単勝の q からの Harville。m: 組のオッズの逆数を、馬を突き合わせた後に券種 × レースの中で和 1 に直した値。
- 血統の帯: 人気薄の馬の z = log p_A − log q が、同じ年 × 同じ升（人気の順 6〜16・17 以上 × その中の log q の十分位）の人気薄の中で上位 2 割（良）・下位 2 割（悪）。
- 升: 人気薄の升 × 相手の人気の順（1〜5）。(1) log(m/H) の差（良 − 悪）と (2) A/E の差（良 − 悪）を、
  升の中で出して、升の件数で重み付けて平均する。帰無は、z を同じ年 × 同じ人気薄の升の中で入れ替える（200 回）。q は固定なので、升の中に残る z と q の結び付きも壊す（守りにならない）。
- (3) 回収率（当たり × オッズの平均）と、上位 1 本の払い戻しを除いた値。控除の線は A/E で馬連 1.29・馬単 1.33（市場の 2 号）。
- 区間は開催日の単位のブートストラップ（1,000 回・日ごとに集めた和を引き直す）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

ROOT = Path("/home/jovyan/work/keiba-vpn/data/page_reference/tables")
TYPES = ("umaren", "umatan")


def horses() -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "odds", "popularity", "horse_number", "finish_position"])
    X = X[(X["date"] >= "2020-01-01") & (X["date"] <= data.DEV_END)].copy()
    P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    X = X.merge(P, on=["race_id", "horse_id"])
    X["odds"] = pd.to_numeric(X["odds"], errors="coerce")
    X = X[X["odds"] > 0].copy()
    X["q"] = (1 / X["odds"]) / (1 / X["odds"]).groupby(X["race_id"]).transform("sum")
    X["pA"] = X["p_A"] / X.groupby("race_id")["p_A"].transform("sum")
    X["z"] = np.log(X["pA"].clip(1e-6)) - np.log(X["q"])
    X["pop"] = pd.to_numeric(X["popularity"], errors="coerce")
    X["fp"] = pd.to_numeric(X["finish_position"], errors="coerce")
    X["hn"] = pd.to_numeric(X["horse_number"], errors="coerce")
    X["y"] = X["date"].dt.year
    X = X[X["split"].isin([0, 1])]
    tie = X[X["fp"] <= 2].groupby("race_id")["fp"].agg(lambda s: s.duplicated().any())
    X = X[~X["race_id"].map(tie).fillna(False)].reset_index(drop=True)       # 1・2 着に同着のあるレースを除く（critic）
    # 人気薄の升: 人気の順（6〜16・17 以上）× 人気の順の中の log q の十分位。
    # 人気の順の中でも z と log q の相関が −0.8 前後あり、升を粗くすると「血統の良い」が「q のより小さい馬」になる（流した後の直し）
    X["pb_"] = np.where(X["pop"] >= 17, 17, X["pop"])
    ls = X["pop"] >= 6
    X["qd"] = -1
    X.loc[ls, "qd"] = X[ls].groupby("pb_")["q"].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False)).astype(int)
    X["lsb"] = X["pb_"] * 100 + X["qd"]
    return X


def bands(X: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    ls = (X["pop"] >= 6).to_numpy()
    key = X["y"].astype(str) + "|" + X["lsb"].astype(str)
    s = pd.Series(z).where(ls)
    hi = s.groupby(key).transform(lambda v: v.quantile(0.8)).to_numpy()
    lo = s.groupby(key).transform(lambda v: v.quantile(0.2)).to_numpy()
    return np.where(ls & (z >= hi), 2, np.where(ls & (z <= lo), 1, 0))


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "pair_market", "券種の間の歪みと血統（記述）"))
    X = horses()
    X["band"] = bands(X, X["z"].to_numpy())
    X["hrow"] = np.arange(len(X))
    dev = set(X["race_id"])
    rng = np.random.default_rng(3)
    res = {"控除の線（A/E）": {"umaren": round(1 / 0.775, 3), "umatan": round(1 / 0.75, 3)}}
    for ty in TYPES:
        F = pd.concat([pd.read_parquet(ROOT / str(y) / "race_pair_odds_flat.parquet", columns=["race_id", "date", "odds_type", "pair", "odds"],
                                       filters=[("odds_type", "==", ty)]) for y in range(2020, 2025)], ignore_index=True)
        F["race_id"] = F["race_id"].astype(str)
        F = F[F["race_id"].isin(dev)]
        pr = F["pair"].astype(str).str.strip("[]").str.split(",", expand=True)
        F = F.assign(a=pd.to_numeric(pr[0], errors="coerce"), b=pd.to_numeric(pr[1], errors="coerce"), o=pd.to_numeric(F["odds"], errors="coerce"))
        F = F[(F["o"] > 0)][["race_id", "date", "a", "b", "o"]]
        H = X[["race_id", "hn", "q", "pop", "fp", "hrow"]]
        F = F.merge(H.rename(columns={"hn": "a", "q": "qa", "pop": "pa", "fp": "fa", "hrow": "ra"}), on=["race_id", "a"])
        F = F.merge(H.rename(columns={"hn": "b", "q": "qb", "pop": "pb", "fp": "fb", "hrow": "rb"}), on=["race_id", "b"])
        F["m"] = (1 / F["o"]) / (1 / F["o"]).groupby(F["race_id"]).transform("sum")   # 突き合わせた後に和 1（critic）
        hab = F["qa"] * F["qb"] / (1 - F["qa"]); hba = F["qb"] * F["qa"] / (1 - F["qb"])
        F["H"] = hab if ty == "umatan" else hab + hba
        F["hit"] = ((F["fa"].eq(1) & F["fb"].eq(2)) if ty == "umatan" else (F["fa"].le(2) & F["fb"].le(2))).astype(float)
        one = (F["pa"] >= 6) ^ (F["pb"] >= 6)
        F = F[one & (np.minimum(F["pa"], F["pb"]) <= 5)].reset_index(drop=True)   # ちょうど 1 頭が人気薄・相手は 1〜5 番人気
        F["lsrow"] = np.where(F["pa"] >= 6, F["ra"], F["rb"])
        F["pt"] = np.minimum(F["pa"], F["pb"]).astype(int)
        F["lsb"] = X["lsb"].to_numpy()[F["lsrow"]]
        F["st"] = pd.factorize(F["lsb"].astype(str) + "|" + F["pt"].astype(str))[0]
        F["lr"] = np.log(F["m"] / F["H"].clip(1e-9))
        day = pd.to_datetime(F["date"]).dt.strftime("%Y%m%d").to_numpy()
        du, di = np.unique(day, return_inverse=True)
        ns, nd = F["st"].max() + 1, len(du)
        lr, hit, m, ho = F["lr"].to_numpy(), F["hit"].to_numpy(), F["m"].to_numpy(), (F["hit"] * F["o"]).to_numpy()
        st, lsrow = F["st"].to_numpy(), F["lsrow"].to_numpy()

        def cube(bd):
            # [日, 升, 帯] の和（件数・lr・当たり・m・当たり×オッズ）
            k = (di * ns + st) * 3 + bd
            out = [np.bincount(k, weights=w, minlength=nd * ns * 3).reshape(nd, ns, 3) for w in (np.ones(len(F)), lr, hit, m, ho)]
            return out

        def summ(C, wday=None):
            if wday is not None:
                C = [np.tensordot(wday, c, axes=(0, 0)) for c in C]
            else:
                C = [c.sum(axis=0) for c in C]
            n, s_lr, s_hit, s_m, s_ho = C
            ok = (n[:, 1] > 0) & (n[:, 2] > 0)
            w = (n[:, 1] + n[:, 2])[ok]
            d_lr = ((s_lr[ok, 2] / n[ok, 2]) - (s_lr[ok, 1] / n[ok, 1]))
            d_ae = ((s_hit[ok, 2] / np.maximum(s_m[ok, 2], 1e-12)) - (s_hit[ok, 1] / np.maximum(s_m[ok, 1], 1e-12)))
            tot = [c.sum(axis=0) for c in (n, s_lr, s_hit, s_m, s_ho)]
            return {"lr": float((d_lr * w).sum() / w.sum()), "ae": float((d_ae * w).sum() / w.sum()),
                    "AE": [float(tot[2][b_] / tot[3][b_]) for b_ in range(3)], "ROI": [float(tot[4][b_] / tot[0][b_]) for b_ in range(3)]}
        bd0 = X["band"].to_numpy()[lsrow]
        C0 = cube(bd0)
        s0 = summ(C0)
        boots = []
        for _ in range(1000):
            wd = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
            boots.append(summ(C0, wd))
        nul_lr, nul_ae = [], []
        zz = X["z"].to_numpy().copy()
        key = (X["y"].astype(str) + "|" + X["lsb"].astype(str)).to_numpy()
        ls_idx = np.flatnonzero((X["pop"] >= 6).to_numpy())
        groups = pd.Series(ls_idx).groupby(key[ls_idx]).apply(lambda s: s.to_numpy()).to_dict()
        for _ in range(200):
            zp = zz.copy()
            for g in groups.values():
                zp[g] = zz[rng.permutation(g)]
            sN = summ(cube(bands(X, zp)[lsrow]))
            nul_lr.append(sN["lr"]); nul_ae.append(sN["ae"])
        nm = ["それ以外", "血統の悪い人気薄を含む", "血統の良い人気薄を含む"]
        bd_ = np.array([[b["lr"], b["ae"]] + b["AE"] + b["ROI"] for b in boots])
        # 上位 1 本の払い戻しを除いた回収率
        roi_x1 = []
        for b_ in range(3):
            v = ho[bd0 == b_]
            roi_x1.append(float((v.sum() - v.max()) / max(len(v) - 1, 1)) if len(v) else np.nan)
        yrs = pd.to_datetime(F["date"]).dt.year.to_numpy()
        by_year = {}
        for yy in range(2020, 2025):
            mk = yrs == yy
            k = (st[mk]) * 3 + bd0[mk]
            n = np.bincount(k, minlength=ns * 3).reshape(ns, 3); sl = np.bincount(k, weights=lr[mk], minlength=ns * 3).reshape(ns, 3)
            ok = (n[:, 1] > 0) & (n[:, 2] > 0); w = (n[:, 1] + n[:, 2])[ok]
            by_year[yy] = round(float(((sl[ok, 2] / n[ok, 2] - sl[ok, 1] / n[ok, 1]) * w).sum() / w.sum()), 4)
        res[ty] = {"組": int(len(F)), "組の数（帯ごと）": {nm[k_]: int((bd0 == k_).sum()) for k_ in range(3)},
                   "当たり（帯ごと）": {nm[k_]: int(hit[bd0 == k_].sum()) for k_ in range(3)},
                   "(1) log(m/H) の差（良 − 悪・升の中）": [round(s0["lr"], 4), np.percentile(bd_[:, 0], [2.5, 97.5]).round(4).tolist(),
                                                     "帰無の 95% 範囲", np.percentile(nul_lr, [2.5, 97.5]).round(4).tolist(),
                                                     "帰無が絶対値で上回る割合", round(float(np.mean(np.abs(np.array(nul_lr) - np.mean(nul_lr)) >= abs(s0["lr"] - np.mean(nul_lr)))), 3)],
                   "(1) 年ごと": by_year,
                   "(2) A/E の差（良 − 悪・升の中）": [round(s0["ae"], 4), np.percentile(bd_[:, 1], [2.5, 97.5]).round(4).tolist(),
                                                "帰無の 95% 範囲", np.percentile(nul_ae, [2.5, 97.5]).round(4).tolist()],
                   "(2) A/E（帯ごと）": {nm[k_]: [round(s0["AE"][k_], 3), np.percentile(bd_[:, 2 + k_], [2.5, 97.5]).round(3).tolist()] for k_ in range(3)},
                   "(3) 回収率（帯ごと）": {nm[k_]: [round(s0["ROI"][k_], 3), np.percentile(bd_[:, 5 + k_], [2.5, 97.5]).round(3).tolist(), "上位 1 本を除く", round(roi_x1[k_], 3)] for k_ in range(3)}}
        print(ty, json.dumps(res[ty], ensure_ascii=False), flush=True)
    path = data.OUT / "pair_market_v2.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
