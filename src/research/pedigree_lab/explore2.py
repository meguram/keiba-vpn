"""探す側の記述の 2 巡目（2 体の R1 を受けた 5 本。log.md「次の 5 本」。未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.explore2 <joint|bands|career|layoff|nulldrop>

- joint: 12 場面を 1 本の条件付きロジットに同時に入れる。開催の束ね 1,000 回の最大 |t| の同時の区間と、経験ベイズで縮めた効果。
- bands: 父の産駒の as-of の走数・母の産駒の頭数の帯ごとの、血統の点の傾き（市場なし）と α の上乗せ（市場あり）。結果で条件付けない。
- career: 本人の中央の出走の数の帯ごとの α の上乗せ（2014 年以降の生まれ）。1 自由度の判定「1〜2 > 0 かつ 1〜2 > 6〜」。
- layoff: 休み明けの行を、レースの年齢条件 × 父の早熟度（3 分位）の 6 つに分けた α の上乗せ。
- nulldrop: 帰無（回 0）で、母父・母・きょうだいの束を落としたときの改善の減り。
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates, scene_dist
from src.research.pedigree_lab.hyp_tests import clogit
from src.research.pedigree_lab.scene_map import RACE_LEVEL, masks


def frame() -> pd.DataFrame:
    cols = list(dict.fromkeys(gates.FCOLS + ["dband", "going", "distance", "dchg", "rage", "sexb", "sire_all_n", "dam_v_nrun",
                                             "sire_rasurf_top3", "sire_surf_top3", "layoff"]))
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    F = scene_dist.marks(gates.scenes(F))
    for k, v in masks(F).items():
        F[f"m_{k}"] = v
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    X["lp"], X["lq"] = np.log(np.clip(X["p_win"], 1e-5, 1)), np.log(np.clip(X["q"], 1e-5, 1))
    X["lpc"] = X["lp"] - pd.Series(X["lp"].to_numpy()).groupby(rid).transform("mean").to_numpy()
    X["lqc"] = X["lq"] - pd.Series(X["lq"].to_numpy()).groupby(rid).transform("mean").to_numpy()
    return X


def fit_boot(X: pd.DataFrame, Z: np.ndarray, n_boot: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    y = X["win"].to_numpy(float)
    rc = pd.factorize(X["race_id"])[0]
    b = clogit(Z, y, rc)
    u = pd.factorize(X["meeting"])[0]
    nu = u.max() + 1
    rng = np.random.default_rng(seed)
    bs = np.array([clogit(Z, y, rc, n_iter=4, b0=b, w=np.bincount(rng.integers(0, nu, nu), minlength=nu)[u].astype(float))
                   for _ in range(n_boot)])
    return b, bs


def interaction_cols(X: pd.DataFrame, mk: dict[str, np.ndarray], level: bool = True, market: bool = True):
    cols, idx = [X["lp"].to_numpy(), X["lq"].to_numpy()] if market else [X["lp"].to_numpy()], {}
    for nm, m in mk.items():
        m = m.astype(float)
        idx[nm] = len(cols)
        cols.append(X["lpc"].to_numpy() * m)
        if market:
            cols.append(X["lqc"].to_numpy() * m)
        if level and nm not in RACE_LEVEL:
            cols.append(m)
    return np.c_[tuple(cols)], idx


def summarize(b, bs, idx) -> dict:
    out = {}
    for nm, j in idx.items():
        out[nm] = {"alpha_extra": round(float(b[j]), 4), "ci": np.percentile(bs[:, j], [2.5, 97.5]).round(4).tolist(),
                   "se": round(float(bs[:, j].std()), 4)}
    return out


def run_joint(X, n_boot):
    mk = {k[2:]: X[k].to_numpy() for k in X.columns if k.startswith("m_")}
    Z, idx = interaction_cols(X, mk)
    b, bs = fit_boot(X, Z, n_boot)
    res = summarize(b, bs, idx)
    # 最大 |t| の同時の区間（初ダートを除く 11 本と、全 12 本）
    names = list(idx)
    se = np.array([bs[:, idx[n]].std() for n in names])
    t = (bs[:, [idx[n] for n in names]] - b[[idx[n] for n in names]]) / se
    cmax = float(np.percentile(np.abs(t).max(1), 95))
    for i, n in enumerate(names):
        res[n]["sim_ci"] = [round(float(b[idx[n]] - cmax * se[i]), 4), round(float(b[idx[n]] + cmax * se[i]), 4)]
    # 経験ベイズの縮め（初ダートを除く 11 本の効果の間の τ²）
    est = np.array([b[idx[n]] for n in names if n != "初ダート"])
    ses = np.array([se[i] for i, n in enumerate(names) if n != "初ダート"])
    tau2 = max(float(np.var(est) - np.mean(ses ** 2)), 0.0)
    mu = float(np.average(est, weights=1 / (ses ** 2 + tau2 + 1e-12)))
    for i, n in enumerate(names):
        w = tau2 / (tau2 + se[i] ** 2) if tau2 > 0 else 0.0
        res[n]["eb"] = round(mu + w * (b[idx[n]] - mu), 4)
    res["_meta"] = {"c_maxT": round(cmax, 3), "tau2": round(tau2, 5), "mu": round(mu, 4), "n_boot": n_boot}
    return res


def band(v, edges, labels):
    return pd.cut(v, edges, labels=labels).astype(str).to_numpy()


def run_bands(X, n_boot):
    sire_n = np.expm1(X["sire_all_n"].to_numpy())
    sb = band(sire_n, [-1, 300, 1000, 3000, 1e12], ["父〜300", "父300〜1000", "父1000〜3000", "父3000〜"])
    dam_n = np.expm1(X["dam_v_nrun"].to_numpy())
    db = band(dam_n, [-1, 0.5, 1.5, 2.5, 1e9], ["母の産駒0", "母の産駒1", "母の産駒2", "母の産駒3〜"])
    res = {}
    for nm, bb, base in (("sire", sb, "父〜300"), ("dam", db, "母の産駒0")):
        mk = {lv: bb == lv for lv in sorted(set(bb)) if lv != base and lv != "nan"}
        for market in (False, True):
            Z, idx = interaction_cols(X, mk, level=True, market=market)
            b, bs = fit_boot(X, Z, n_boot)
            r = summarize(b, bs, idx)
            r["_base"] = {"band": base, "slope_lp": round(float(b[0]), 4)}
            res[f"{nm}_{'with_market' if market else 'no_market'}"] = r
    return res


def run_career(X, n_boot):
    F = X.sort_values(["horse_id", "date"], kind="stable")
    ok = (pd.to_numeric(F["by"], errors="coerce").fillna(0) >= 2014) & (F.groupby("horse_id")["cls"].transform("first") == "NEW")
    c = F["career"].to_numpy()
    lab = np.where(c == 0, "0", np.where(c <= 2, "1〜2", np.where(c <= 5, "3〜5", "6〜")))
    lab = pd.Series(np.where(ok, lab, "対象外"), index=F.index).sort_index().to_numpy()
    mk = {lv: lab == lv for lv in ("0", "1〜2", "3〜5", "6〜")}
    Z, idx = interaction_cols(X, mk)
    b, bs = fit_boot(X, Z, n_boot)
    res = summarize(b, bs, idx)
    d1 = bs[:, idx["1〜2"]] - bs[:, idx["0"]]
    d2 = bs[:, idx["1〜2"]] - bs[:, idx["6〜"]]
    res["_test"] = {"1〜2 − 0": [round(float(b[idx["1〜2"]] - b[idx["0"]]), 4), np.percentile(d1, [2.5, 97.5]).round(4).tolist()],
                    "1〜2 − 6〜": [round(float(b[idx["1〜2"]] - b[idx["6〜"]]), 4), np.percentile(d2, [2.5, 97.5]).round(4).tolist()],
                    "pass": bool(np.percentile(d1, 2.5) > 0 and np.percentile(d2, 2.5) > 0)}
    return res


def run_layoff(X, n_boot):
    lay = X["m_休み明け（90 日以上）"].to_numpy()
    early = (X["sire_rasurf_top3"] - X["sire_surf_top3"]).to_numpy()      # 父 × レースの年齢条件の上乗せ（H6'）
    young = X["rage"].isin(["A2", "A3"]).to_numpy() | (X["rage"].eq("A3U").to_numpy() & (X["date"].dt.month.to_numpy() <= 8))
    q = pd.qcut(pd.Series(early[lay]).rank(method="first"), 3, labels=["早熟", "中", "晩成"]).astype(str).to_numpy()
    tert = np.full(len(X), "", dtype=object)
    tert[lay] = q
    mk = {}
    for a in ("2〜3 歳", "4 歳以上"):
        for t in ("早熟", "中", "晩成"):
            m = lay & (tert == t) & (young if a == "2〜3 歳" else ~young)
            mk[f"{a}×{t}"] = m
    Z, idx = interaction_cols(X, mk)
    b, bs = fit_boot(X, Z, n_boot)
    res = summarize(b, bs, idx)
    for k in mk:
        res[k]["rows"] = int(mk[k].sum())
    return res


def run_nulldrop():
    from src.research.pedigree_lab import ablate, features4, v5
    from src.research.pedigree_lab.report_stats import GROUPS
    r = data.load_dev(reason="帰無（回 0）で母父・母の束を落とす", version="nulldrop_0")
    ped = data.load_ped()
    sx = r.drop_duplicates("horse_id").set_index("horse_id")["sex"].reindex(ped["horse_id"]).fillna("?").to_numpy()
    grp = ped["horse_id"].str[:4].to_numpy() + "_" + sx
    rng = np.random.default_rng(1000)
    perm = np.arange(len(ped))
    for ii in pd.Series(np.arange(len(ped))).groupby(grp).indices.values():
        perm[ii] = rng.permutation(ii)
    P = ped.iloc[perm].reset_index(drop=True)
    P["horse_id"] = ped["horse_id"].to_numpy()
    Xn, _ = features4.build(r, P, sub=False)
    Xn["split"] = data.split_meetings(Xn["meeting"])
    Xn = Xn[Xn["date"].dt.year >= 2017].reset_index(drop=True)
    dev = Xn["split"].isin([0, 1]).to_numpy()
    full = ablate.col_sets(Xn)["pure"]
    g0 = v5.gain(Xn, v5.oof_for(Xn, full, "dd")["win"], dev)
    pats = dict(GROUPS)
    res = {"null_full": g0}
    for nm in ("母父", "母・きょうだい"):
        cols = [c for c in full if not re.search(pats[nm], c)]
        g = v5.gain(Xn, v5.oof_for(Xn, cols, "dd")["win"], dev)
        res[nm] = {"null_drop_loss": round(g0 - g, 5)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["joint", "bands", "career", "layoff", "nulldrop"])
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"explore2_{a.what}", f"探す側の記述の 2 巡目: {a.what}（尤度だけ）"))
    if a.what == "nulldrop":
        res = run_nulldrop()
    else:
        X = frame()
        res = {"joint": run_joint, "bands": run_bands, "career": run_career, "layoff": run_layoff}[a.what](X, a.boot)
    path = data.OUT / f"explore2_{a.what}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
