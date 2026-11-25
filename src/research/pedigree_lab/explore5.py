"""市場の側の切り口と、公開の血統理論からの仮説（log.md「市場の側の切り口」「(5) 公開の血統理論からの仮説」。未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.explore5 <course|sides|newsire|sirepop>

- course: H15〜H20 の 6 本の父の場の差の列 z を 1 本の条件付きロジットに同時に入れ、γ0（市場なし）と γ（市場あり）を同時の区間で読む。
- sides: 父系側・母系側に分けた OOF（kf_v4_pat_side_dd・kf_v4_mat_side_dd）を Benter に両方入れる。
- newsire: そのレースの前に父の産駒が中央で 1 走もしていない 2 歳戦の馬の α の上乗せ。
- sirepop: 父の「市場の人気度」（その父の産駒のそれまでの走りの log q − log p_血統 の平均。as-of・自分を引く）を Benter の第 3 の項に。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.hyp_tests import clogit

K = 200.0
STEEP = {"中山", "阪神", "中京"}
FLAT = {"京都", "新潟", "福島", "小倉", "札幌", "函館"}
LEFT = {"東京", "新潟", "中京"}
TURF_START = {("東京", 1600), ("中山", 1200), ("阪神", 1400), ("阪神", 2000), ("京都", 1400), ("中京", 1400), ("新潟", 1200), ("福島", 1150)}


def long_straight(v, d):
    if v in ("東京", "新潟"):
        return True
    if v == "阪神":
        return d in (1600, 1800, 2400)
    if v == "京都":
        return d in (1800, 2200, 2400)
    return False


def groups(F: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    v, d, s = F["venue"].to_numpy(), F["distance"].to_numpy(), F["surf"].to_numpy()
    m = F["date"].dt.month.to_numpy()
    good = F["going"].to_numpy() == "G"
    turf = s == "T"
    ls = np.array([long_straight(a, b) for a, b in zip(v, d)])
    short = turf & np.isin(v, ["中山", "福島", "小倉", "札幌", "函館", "阪神", "京都"]) & ~ls
    return {
        "H15 坂": (np.isin(v, list(STEEP)), np.isin(v, list(FLAT))),
        "H16 直線": (turf & ls, short),
        "H17 洋芝": (turf & np.isin(v, ["札幌", "函館"]), turf & ~np.isin(v, ["札幌", "函館"])),
        "H18 冬のダート": ((s == "D") & good & np.isin(m, [12, 1, 2]), (s == "D") & good & np.isin(m, [6, 7, 8])),
        "H19 左右": (np.isin(v, list(LEFT)), ~np.isin(v, list(LEFT))),
    }


def diff_col(F: pd.DataFrame, A: np.ndarray, B: np.ndarray, keys=("sire", "surf")) -> np.ndarray:
    """父の as-of の「A の平均 − B の平均」（良さ = −分位）を全体の差へ縮め、全体の差を引いて 0 中心にする。"""
    g = -F["fpct"].to_numpy()
    D = F.assign(_a=g * A, _na=A.astype(float), _b=g * B, _nb=B.astype(float))
    s, _ = asof(D, list(keys), ["_a", "_na", "_b", "_nb"])
    # 基準の群の平均は、鍵に芝ダがあるときだけ芝ダで分ける（H20 は芝 − ダートの差なので芝ダで分けない。critic 4 回目の 1）
    base, _ = asof(D.assign(_k="1"), ["_k", "surf"] if "surf" in keys else ["_k"], ["_a", "_na", "_b", "_nb"])
    mA0 = base[:, 0] / np.maximum(base[:, 1], 1)
    mB0 = base[:, 2] / np.maximum(base[:, 3], 1)
    mA = (s[:, 0] + K * mA0) / (s[:, 1] + K)
    mB = (s[:, 2] + K * mB0) / (s[:, 3] + K)
    return (mA - mB) - (mA0 - mB0)


def frame() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "meeting", "date", "win", "fpct", "odds", "horse_number", "sire", "surf", "venue", "distance",
            "going", "rage", "career", "sire_all_n", "split"]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    return F


def attach(F: pd.DataFrame, kf: str = "kf_v4_pure_dd") -> pd.DataFrame:
    O = pd.read_parquet(data.OUT / f"{kf}.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    X["lp"], X["lq"] = np.log(np.clip(X["p_win"], 1e-5, 1)), np.log(np.clip(X["q"], 1e-5, 1))
    for c in ("lp", "lq"):
        X[f"{c}c"] = X[c] - pd.Series(X[c].to_numpy()).groupby(rid).transform("mean").to_numpy()
    return X


def zcol(X: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    rid = X["race_id"].to_numpy()
    v = raw - pd.Series(raw).groupby(rid).transform("mean").to_numpy()
    sd = v.std()
    return v / (sd if sd > 0 else 1.0)


def sim_ci(b, bs, idx):
    names = list(idx)
    se = np.array([bs[:, idx[n]].std() for n in names])
    t = (bs[:, [idx[n] for n in names]] - b[[idx[n] for n in names]]) / se
    c = float(np.percentile(np.abs(t).max(1), 95))
    return c, {n: {"gamma": round(float(b[idx[n]]), 4), "se": round(float(se[i]), 4),
                   "ci": np.percentile(bs[:, idx[n]], [2.5, 97.5]).round(4).tolist(),
                   "sim_ci": [round(float(b[idx[n]] - c * se[i]), 4), round(float(b[idx[n]] + c * se[i]), 4)]} for i, n in enumerate(names)}


def run_course(n_boot: int) -> dict:
    F = frame()
    G = groups(F)
    raw = {}
    for nm, (A, B) in G.items():
        d = diff_col(F, A, B)
        raw[nm] = np.where(A, d, np.where(B, -d, 0.0))
    # H20: 父の芝 − ダートの差（全体の芝ダの群で作る）を、一覧のダートで +・ほかのダートで −
    dTD = diff_col(F, (F["surf"] == "T").to_numpy(), (F["surf"] == "D").to_numpy(), keys=("sire",))
    ts = np.array([(a, b) in TURF_START for a, b in zip(F["venue"], F["distance"])]) & (F["surf"] == "D").to_numpy()
    od = (F["surf"] == "D").to_numpy() & ~ts
    raw["H20 芝スタートのダート"] = np.where(ts, dTD, np.where(od, -dTD, 0.0))
    for nm, v in raw.items():
        F[f"z_{nm}"] = v
    X = attach(F)
    names = list(raw)
    Z = np.c_[tuple(zcol(X, X[f"z_{n}"].to_numpy()) for n in names)]
    res = {}
    for market in (False, True):
        base = [X["lp"].to_numpy(), X["lq"].to_numpy()] if market else [X["lp"].to_numpy()]
        M = np.c_[tuple(base) + (Z,)]
        idx = {n: len(base) + i for i, n in enumerate(names)}
        b, bs = fit_boot(X, M, n_boot)
        c, r = sim_ci(b, bs, idx)
        res["with_market" if market else "no_market"] = {"c_maxT": round(c, 3), **r}
    res["rows_A_B"] = {n: [int(G[n][0].sum()), int(G[n][1].sum())] for n in G}
    return res


def run_sides(n_boot: int) -> dict:
    P = pd.read_parquet(data.OUT / "kf_v4_pat_side_dd.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_pat"})
    Mt = pd.read_parquet(data.OUT / "kf_v4_mat_side_dd.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_mat"})
    X = attach(frame()).merge(P, on=["race_id", "horse_id"]).merge(Mt, on=["race_id", "horse_id"])
    X = X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X["lpat"], X["lmat"] = np.log(np.clip(X["p_pat"], 1e-5, 1)), np.log(np.clip(X["p_mat"], 1e-5, 1))
    res = {}
    for market in (False, True):
        cols = ["lpat", "lmat"] + (["lq"] if market else [])
        b, bs = fit_boot(X, X[cols].to_numpy(), n_boot)
        res["with_market" if market else "no_market"] = {c: {"coef": round(float(b[i]), 4), "ci": np.percentile(bs[:, i], [2.5, 97.5]).round(4).tolist()}
                                                        for i, c in enumerate(cols)}
        d = bs[:, 1] - bs[:, 0]
        res["with_market" if market else "no_market"]["mat_minus_pat"] = [round(float(b[1] - b[0]), 4), np.percentile(d, [2.5, 97.5]).round(4).tolist()]
    return res


def run_newsire(n_boot: int) -> dict:
    X = attach(frame())
    m = (X["rage"] == "A2").to_numpy() & (np.expm1(X["sire_all_n"].to_numpy()) < 0.5)
    Z = np.c_[X["lp"], X["lq"], X["lpc"] * m, X["lqc"] * m, m.astype(float)]
    b, bs = fit_boot(X, Z, n_boot)
    return {"rows": int(m.sum()), "alpha_extra": round(float(b[2]), 4), "ci": np.percentile(bs[:, 2], [2.5, 97.5]).round(4).tolist(),
            "alpha_all": round(float(b[0]), 4)}


def run_sirepop(n_boot: int) -> dict:
    X = attach(frame())
    X["gap"] = X["lq"] - X["lp"]
    s, c = asof(X, ["sire"], ["gap"])
    pop = (s[:, 0] + 30 * 0.0) / (c + 30)                 # 0 へ縮める（擬似件数 30）
    X["pop"] = pop
    z = zcol(X, pop)
    Z = np.c_[X["lp"], X["lq"], z]
    b, bs = fit_boot(X, Z, n_boot)
    return {"coef_pop": round(float(b[2]), 4), "ci": np.percentile(bs[:, 2], [2.5, 97.5]).round(4).tolist(),
            "coef_lp": round(float(b[0]), 4), "coef_lq": round(float(b[1]), 4),
            "pop_sd": round(float(np.std(pop)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["course", "sides", "newsire", "sirepop"])
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"explore5_{a.what}", f"市場の側の切り口・公開の理論: {a.what}（尤度だけ）"))
    res = {"course": run_course, "sides": run_sides, "newsire": run_newsire, "sirepop": run_sirepop}[a.what](a.boot)
    path = data.OUT / f"explore5_{a.what}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
