"""1 号の R2 の確かめ（log.md「確かめ 5 本」。未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.explore6 <pop|yoshiba>

- pop: 人気の種牡馬の係数の別の説明の区別（B 曲がり・前年末までの人気度・C 産駒の数の帯・D 2 歳戦）。
- yoshiba: 北海道の芝の 3 歳以上・4 歳以上戦だけを群 A にした洋芝の γ0（市場なし）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import attach, diff_col, frame, groups, sim_ci, zcol
from src.research.pedigree_lab.hyp_tests import clogit


def ci(b, bs, j):
    return {"coef": round(float(b[j]), 4), "ci": np.percentile(bs[:, j], [2.5, 97.5]).round(4).tolist()}


def run_pop(n_boot: int) -> dict:
    X = attach(frame())
    X["gap"] = X["lq"] - X["lp"]
    s, c = asof(X, ["sire"], ["gap"])
    X["pop"] = s[:, 0] / (c + 30)
    # 前年の年末までの走りだけの人気度（年 t の行には t−1 年末までの父の平均を当てる）
    X["year"] = X["date"].dt.year
    yr = X.groupby(["sire", "year"])["gap"].agg(["sum", "size"]).reset_index().sort_values(["sire", "year"])
    yr["cs"] = yr.groupby("sire")["sum"].cumsum() - yr["sum"]
    yr["cn"] = yr.groupby("sire")["size"].cumsum() - yr["size"]
    X = X.merge(yr[["sire", "year", "cs", "cn"]], on=["sire", "year"], how="left")
    # 自分の前年までの走りを引く（critic 4 回目の 3）
    hy = X.groupby(["horse_id", "year"])["gap"].agg(["sum", "size"]).reset_index().sort_values(["horse_id", "year"])
    hy["hs"] = hy.groupby("horse_id")["sum"].cumsum() - hy["sum"]
    hy["hn"] = hy.groupby("horse_id")["size"].cumsum() - hy["size"]
    X = X.merge(hy[["horse_id", "year", "hs", "hn"]], on=["horse_id", "year"], how="left")
    X = X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X["pop_prev"] = ((X["cs"] - X["hs"]) / (X["cn"] - X["hn"] + 30)).fillna(0.0)
    z = zcol(X, X["pop"].to_numpy())
    res = {}
    # B: 曲がりを足す
    Z = np.c_[X["lp"], X["lq"], X["lp"] ** 2, X["lq"] ** 2, z]
    b, bs = fit_boot(X, Z, n_boot)
    res["B_curvature"] = ci(b, bs, 4)
    # 漏れ: 前年末までの人気度
    zp = zcol(X, X["pop_prev"].to_numpy())
    b, bs = fit_boot(X, np.c_[X["lp"], X["lq"], zp], n_boot)
    res["prev_year_pop"] = ci(b, bs, 2)
    # C: 父の産駒の数の帯
    few = (np.expm1(X["sire_all_n"].to_numpy()) < 1000).astype(float)
    b, bs = fit_boot(X, np.c_[X["lp"], X["lq"], z * few, z * (1 - few)], n_boot)
    res["C_sire_runs_lt1000"], res["C_sire_runs_ge1000"] = ci(b, bs, 2), ci(b, bs, 3)
    # D: 2 歳戦とそれ以外
    two = (X["rage"] == "A2").to_numpy().astype(float)
    b, bs = fit_boot(X, np.c_[X["lp"], X["lq"], z * two, z * (1 - two)], n_boot)
    res["D_2yo"], res["D_other"] = ci(b, bs, 2), ci(b, bs, 3)
    return res


def run_yoshiba(n_boot: int) -> dict:
    F = frame()
    G = groups(F)
    turf = (F["surf"] == "T").to_numpy()
    hok = turf & F["venue"].isin(["札幌", "函館"]).to_numpy()
    old = ~F["rage"].isin(["A2", "A3"]).to_numpy()
    A, B = hok & old, turf & ~F["venue"].isin(["札幌", "函館"]).to_numpy() & old
    d = diff_col(F, A, B)
    F["z_y"] = np.where(A, d, np.where(B, -d, 0.0))
    X = attach(F)
    b, bs = fit_boot(X, np.c_[X["lp"], zcol(X, X["z_y"].to_numpy())], n_boot)
    return {"rows_A_B": [int(A.sum()), int(B.sum())], "gamma0_old_only": ci(b, bs, 1)}


def run_yoshiba2(n_boot: int) -> dict:
    """2 号の確かめ 1・2: 洋芝の z を北海道の部分とそれ以外の部分に分け、父の芝の質を制御に足す。"""
    F = frame()
    G = groups(F)
    A, B = G["H17 洋芝"]
    d = diff_col(F, A, B)
    F["zA"], F["zB"] = np.where(A, d, 0.0), np.where(B, -d, 0.0)
    F2 = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "sire_surf_fpct"])
    X = attach(F).merge(F2, on=["race_id", "horse_id"], how="left").sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    qual = zcol(X, -X["sire_surf_fpct"].fillna(X["sire_surf_fpct"].mean()).to_numpy())
    zA, zB = zcol(X, X["zA"].to_numpy()), zcol(X, X["zB"].to_numpy())
    res = {}
    b, bs = fit_boot(X, np.c_[X["lp"], zA, zB], n_boot)
    res["split_no_market"] = {"A_hokkaido": ci(b, bs, 1), "B_other_turf": ci(b, bs, 2)}
    zfull = zcol(X, (X["zA"] + X["zB"]).to_numpy())
    b, bs = fit_boot(X, np.c_[X["lp"], zfull, qual], n_boot)
    res["with_turf_quality_control"] = {"yoshiba": ci(b, bs, 1), "quality": ci(b, bs, 2)}
    return res


def run_popq(n_boot: int) -> dict:
    """2 号の確かめ 4 と任意の帰無: 人気度に父の質の列を足す・人気度の計算でだけ父の札を入れ替える。"""
    X = attach(frame())
    F2 = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "sire_all_fpct", "sire_v_win", "by", "sexb"])
    X = X.merge(F2, on=["race_id", "horse_id"], how="left").sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X["gap"] = X["lq"] - X["lp"]
    s, c = asof(X, ["sire"], ["gap"])
    z = zcol(X, s[:, 0] / (c + 30))
    q1 = zcol(X, -X["sire_all_fpct"].fillna(X["sire_all_fpct"].mean()).to_numpy())
    q2 = zcol(X, X["sire_v_win"].fillna(X["sire_v_win"].mean()).to_numpy())
    res = {}
    b, bs = fit_boot(X, np.c_[X["lp"], X["lq"], z, q1, q2], n_boot)
    res["with_quality"] = {"pop": ci(b, bs, 2), "sire_all_fpct": ci(b, bs, 3), "sire_v_win": ci(b, bs, 4)}
    # 帰無: 父の札を、生年 × 性の中で馬の単位に入れ替えて人気度を作る
    # 帰無（父の札を生年 × 性の中で馬の単位に入れ替えて人気度を作る）を 5 回（critic 4 回目の 4: 1 回では入れ替えの散らばりが入らない）
    H = X.drop_duplicates("horse_id")[["horse_id", "sire", "by", "sexb"]].reset_index(drop=True)
    grp = pd.Series(np.arange(len(H))).groupby(H["by"].astype(str) + "_" + H["sexb"].astype(str)).indices
    y = X["win"].to_numpy(float)
    rc = pd.factorize(X["race_id"])[0]
    nulls = []
    for k in range(5):
        rng = np.random.default_rng(99 + k)
        perm = np.arange(len(H))
        for ii in grp.values():
            perm[ii] = rng.permutation(ii)
        fake = dict(zip(H["horse_id"], H["sire"].to_numpy()[perm]))
        s, c = asof(X.assign(sire=X["horse_id"].map(fake)), ["sire"], ["gap"])
        zn = zcol(X, s[:, 0] / (c + 30))
        nulls.append(round(float(clogit(np.c_[X["lp"], X["lq"], zn], y, rc)[2]), 4))
    res["null_pop_5"] = {"coefs": nulls, "mean": round(float(np.mean(nulls)), 4), "sd": round(float(np.std(nulls, ddof=1)), 4)}
    return res


def run_popfwd(n_boot: int) -> dict:
    """critic 4 回目の 2: 人気度の元の lp を、時間の順の walk-forward の予測（年 t を t より前の年で学んだ模型。2019〜24）に替える。"""
    W = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_wf"})
    X = attach(frame()).merge(W, on=["race_id", "horse_id"], how="inner").sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    v = X["p_wf"].clip(1e-5, 1).to_numpy()
    X["lpw"] = np.log(v / pd.Series(v).groupby(rid).transform("sum").to_numpy())
    X["gap"] = X["lq"] - X["lpw"]
    s, c = asof(X, ["sire"], ["gap"])
    z = zcol(X, s[:, 0] / (c + 30))
    b, bs = fit_boot(X, np.c_[X["lpw"], X["lq"], z], n_boot)
    return {"rows": len(X), "pop_wf": ci(b, bs, 2), "lp_wf": ci(b, bs, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pop", "yoshiba", "yoshiba2", "popq", "popfwd"])
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"explore6_{a.what}", f"1 号の R2 の確かめ: {a.what}（尤度だけ）"))
    res = {"pop": run_pop, "yoshiba": run_yoshiba, "yoshiba2": run_yoshiba2, "popq": run_popq, "popfwd": run_popfwd}[a.what](a.boot)
    path = data.OUT / f"explore6_{a.what}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
