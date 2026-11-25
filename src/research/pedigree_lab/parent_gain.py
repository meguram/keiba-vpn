"""親モデル（めぐ AI の v5bag・年ごとの前向きの予測）に、血統のモデルの点がまだ情報を足すか（尤度だけ・回収率は見ない）。

開発の行（split 0・1）のうち、v5bag の予測がある行で、条件付きロジット（勝ち）を当てる。
- 基準 M0: [log p_v5bag]、M1: [log p_v5bag, log q_確定]
- 足す: log p_血統（第 4 版 pure・2 代母の単位の OOF）、log p_血統（第 5 版 E）
レースの単位の 5 分割の交差で、基準からの対数損失の改善を出す（区間は開催の単位のブートストラップ）。場面（初ダート）の上乗せも記述で出す。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.parent_gain
出力: OUT/parent_gain.json。費用 0 円。2025・26 年と保留は読まない。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates
from src.research.pedigree_lab.hyp_tests import clogit

V5BAG = Path("/home/jovyan/work/megu_cycles_out/rich_pred_v5bag.parquet")


def softmax(eta: np.ndarray, rid: np.ndarray) -> np.ndarray:
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def cv_gain(X: pd.DataFrame, base: list[str], extra: list[str]) -> dict:
    y = X["win"].to_numpy(float)
    rid = X["race_id"].to_numpy()
    rc = pd.factorize(rid)[0]
    fo = np.random.default_rng(7).permutation(rc.max() + 1)[rc] % 5
    pb, pe = np.zeros(len(X)), np.zeros(len(X))
    coefs = []
    for k in range(5):
        A = fo != k
        ra = pd.factorize(rid[A])[0]
        b0 = clogit(X.loc[A, base].to_numpy(), y[A], ra)
        b1 = clogit(X.loc[A, base + extra].to_numpy(), y[A], ra)
        coefs.append(b1.round(4).tolist())
        pb[~A] = softmax(X.loc[~A, base].to_numpy() @ b0, rid[~A])
        pe[~A] = softmax(X.loc[~A, base + extra].to_numpy() @ b1, rid[~A])
    w = y == 1
    d = np.log(pe[w]) - np.log(pb[w])
    mt = pd.factorize(X.loc[w, "meeting"])[0]
    nu = mt.max() + 1
    s, c = np.bincount(mt, d, minlength=nu), np.bincount(mt, minlength=nu)
    rng = np.random.default_rng(3)
    bs = [((wb := np.bincount(rng.integers(0, nu, nu), minlength=nu)) * s).sum() / (wb * c).sum() for _ in range(1000)]
    return {"gain": round(float(d.mean()), 6), "ci": np.percentile(bs, [2.5, 97.5]).round(6).tolist(), "coefs": coefs}


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "parent_gain",
                                   "親モデル（v5bag）への血統の上乗せ（尤度だけ・回収率は見ない）"))
    V = pd.read_parquet(V5BAG)[["race_id", "horse_id", "pr_win"]].dropna()
    V["race_id"], V["horse_id"] = V["race_id"].astype(str), V["horse_id"].astype(str)
    K4 = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    K4 = K4[K4["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    KE = pd.read_parquet(data.OUT / "kf_v5E_dd.parquet").rename(columns={"p_win": "p_e"})
    F = gates.scenes(pd.read_parquet(data.OUT / "feat_v4.parquet", columns=gates.FCOLS))
    X = K4.merge(KE, on=["race_id", "horse_id"]).merge(V, on=["race_id", "horse_id"]).merge(
        F[["race_id", "horse_id", "meeting", "win", "odds", "horse_number", "FD", "date"]], on=["race_id", "horse_id"])
    X = X[(X["odds"] > 0) & X["p_e"].notna()].copy()
    # v5bag の予測があるレースだけ（レースの全頭が揃うものに限る）
    n_all = F.groupby("race_id").size()
    full = X.groupby("race_id")["horse_id"].transform("size").to_numpy() == n_all.reindex(X["race_id"]).to_numpy()
    X = X[full].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    for c, nm in (("pr_win", "lv"), ("p_win", "lp"), ("p_e", "le")):
        v = X[c].clip(1e-5, 1).to_numpy()
        X[nm] = np.log(v / pd.Series(v).groupby(rid).transform("sum").to_numpy())
    inv = 1 / X["odds"].to_numpy()
    X["lq"] = np.log(inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy())
    res = {"rows": len(X), "races": int(X["race_id"].nunique()), "years": sorted(X["date"].dt.year.unique().tolist())}
    res["v5bag+ped_v4"] = cv_gain(X, ["lv"], ["lp"])
    res["v5bag+ped_E"] = cv_gain(X, ["lv"], ["le"])
    res["v5bag+mkt+ped_v4"] = cv_gain(X, ["lv", "lq"], ["lp"])
    res["v5bag+mkt+ped_E"] = cv_gain(X, ["lv", "lq"], ["le"])
    fd = X["FD"].to_numpy().astype(float)
    lpc = X["lp"].to_numpy() - pd.Series(X["lp"].to_numpy()).groupby(rid).transform("mean").to_numpy()
    X["lpc_fd"] = lpc * fd
    X["fd"] = fd
    res["v5bag+mkt+ped_v4+FD"] = cv_gain(X, ["lv", "lq", "lp"], ["lpc_fd", "fd"])
    path = data.OUT / "parent_gain.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "coefs"} | {"coef_last": v["coefs"][-1]})
                      for k, v in res.items()}, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
