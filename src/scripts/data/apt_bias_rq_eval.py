"""適性マップの点数化(APT)・馬場の当日予測(BIAS)・レース質が、親モデルの予測の残差に残っているか。再学習 #3（2026-11）。

    python apt_bias_rq_eval.py build            # 候補列（as-of）を作る → <OUT>/aprq_features.parquet
    python apt_bias_rq_eval.py score            # 第 1 段: 残差のスコア検定 + 残差の木
    python apt_bias_rq_eval.py tier2 BIAS,APT   # 第 2 段: test 2025 に 1 回だけ適用（第 1 段を通った供給元だけ）

合格線は docs/operations/retrain_trials/README.md「適性マップの点数化 APT・馬場の当日予測 BIAS・レース質」に測る前に固定した。
2026 年は使わない。市場オッズは使わない。統計量はすべて「そのレース日より前」だけ（同日を含まない累積）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

R = Path("/home/jovyan/work/retrain_out")
BASE = R / "base_2016"
OUT = R / "aprq16"
RQ = Path("/home/jovyan/work/keiba-vpn/data/local/modeling/shared_cache/race_quality/race_quality.parquet")
OOF = BASE / "tp27_v6e/ens/v6.parquet"
T25 = BASE / "final_eval_v6e16/test2025_predictions.parquet"
M = 20.0
NB = 8
KERNEL = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])[None, :, :, None]
BOOT_YEARS = (2022, 2023, 2024)


def load_table() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "grade", "field_size", "horse_number", "finish_position",
            "a_style", "a_l3fz_ewm2"]
    t = pd.read_parquet(BASE / "features/train_table.parquet", columns=cols)
    t["horse_id"] = t["horse_id"].astype(str)
    t["race_id"] = t["race_id"].astype(str)
    t["date"] = pd.to_datetime(t["date"].astype(str).str[:10])
    t["year"] = t["date"].dt.year
    t["venue_code"] = t["venue_code"].astype(str)
    t["surface"] = t["surface"].astype(str)
    return t


# ------------------------------------------------------------------ APT
def _edges(x: pd.Series, mask: pd.Series, n: int) -> np.ndarray:
    return np.nanquantile(x[mask].to_numpy(), np.linspace(0, 1, n + 1)[1:-1])


def _smooth(A: np.ndarray) -> np.ndarray:
    """A: (D, NB*NB, C) の累積 → セル平面で 3×3 平滑化。"""
    D, _, C = A.shape
    return convolve(A.reshape(D, NB, NB, C), KERNEL, mode="constant").reshape(D, NB * NB, C)


def _cum_excl(rows_d: np.ndarray, rows_c: np.ndarray, vals: np.ndarray, nd: int) -> np.ndarray:
    T = np.zeros((nd, NB * NB, vals.shape[1]))
    np.add.at(T, (rows_d, rows_c), vals)
    return np.cumsum(T, axis=0) - T


def apt_features(t: pd.DataFrame) -> pd.DataFrame:
    """座標 (a_style, a_l3fz_ewm2) の 8×8 セルごとの勝ち・3 着内の倍率を、コース別・馬場別に as-of で作る。"""
    flat = t["surface"].isin(["芝", "ダート"]) & (t["field_size"] >= 5) & (t["finish_position"] >= 1)
    fit = t["year"] <= 2019
    es = _edges(t["a_style"], fit & flat, NB)
    ek = _edges(t["a_l3fz_ewm2"], fit & flat, NB)
    bs = np.searchsorted(es, t["a_style"].to_numpy())
    bk = np.searchsorted(ek, t["a_l3fz_ewm2"].to_numpy())
    cell = np.where(t[["a_style", "a_l3fz_ewm2"]].notna().all(axis=1) & flat, bs * NB + bk, -1)
    n = t["field_size"].to_numpy().astype(float)
    fin = t["finish_position"].to_numpy()
    vals = np.column_stack([(fin == 1), 1.0 / n, (fin <= 3), np.minimum(3, n) / n]).astype(float)   # O_win, E_win, O_t3, E_t3
    dates = np.sort(t["date"].unique())
    didx = np.searchsorted(dates, t["date"].to_numpy())
    ok = cell >= 0
    good = (t["track_condition"].astype(str) == "良").to_numpy()
    ckey = (t["venue_code"] + "|" + t["surface"] + "|" + t["distance"].astype(str)).to_numpy()
    gkey = (t["surface"] + "|" + np.where(good, "良", "道悪")).to_numpy()

    G = _smooth(_cum_excl(didx[ok], cell[ok], vals[ok], len(dates)))                      # 全コース共通
    res = {k: np.full(len(t), np.nan) for k in ("course_win", "course_t3", "going_win", "going_t3")}

    def lifts(Cm: np.ndarray, gl: np.ndarray, o: int, e: int) -> np.ndarray:
        g = gl[:, o] / np.maximum(gl[:, e], 1e-9)
        lc = (Cm[:, o] + M * g) / (Cm[:, e] + M)
        return np.where(g > 0, np.log(np.maximum(lc, 1e-12) / np.maximum(g, 1e-12)), 0.0)

    for name, key in (("course", ckey), ("going", gkey)):
        codes = pd.factorize(pd.Series(key).where(ok))[0]
        order = np.argsort(codes, kind="stable")
        starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
        ends = np.append(starts[1:], len(order))
        for a, b in zip(starts, ends):
            idx = order[a:b]
            ds, inv = np.unique(didx[idx], return_inverse=True)
            C = _smooth(_cum_excl(inv, cell[idx], vals[idx], len(ds)))
            Cm = C[inv, cell[idx]]
            gl = G[didx[idx], cell[idx]]
            res[f"{name}_win"][idx] = lifts(Cm, gl, 0, 1)
            res[f"{name}_t3"][idx] = lifts(Cm, gl, 2, 3)
    o = pd.DataFrame({"race_id": t["race_id"].to_numpy(), "horse_id": t["horse_id"].to_numpy()})
    for m in ("win", "t3"):
        o[f"apt_course_{m}"] = res[f"course_{m}"]
        o[f"apt_going_{m}"] = res[f"going_{m}"]
        o[f"apt_sum_{m}"] = o[f"apt_course_{m}"] + 0.5 * o[f"apt_going_{m}"]
    return o


# ------------------------------------------------------------------ BIAS
def bias_features(t: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_parquet(R / "bias/races.parquet", columns=["race_id"])
    p = pd.read_parquet(R / "bias/pred_try3.parquet")
    s = pd.read_parquet(R / "bias/static_pred.parquet").add_prefix("st_")
    b = pd.concat([r.reset_index(drop=True), p.reset_index(drop=True), s.reset_index(drop=True)], axis=1)
    b["race_id"] = b["race_id"].astype(str)
    b = b[["race_id", "pred_b_F", "pred_b_D", "st_b_F", "st_b_D", "sw_b_F", "sw_b_D"]]
    b["d_b_F"] = b["pred_b_F"] - b["st_b_F"]
    b["d_b_D"] = b["pred_b_D"] - b["st_b_D"]
    return b


# ------------------------------------------------------------------ レース質
JOINT = [f"{p}|{s}" for p in ("slow", "mid", "high") for s in ("compact", "normal", "stretched")]


def rq_features(t: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_parquet(RQ)
    q["race_id"] = q["race_id"].astype(str)
    q["date"] = pd.to_datetime(q["date"].astype(str).str[:10])
    pj = q[[f"p_joint_{j}" for j in JOINT]].to_numpy()
    out = pd.DataFrame({"race_id": q["race_id"]})
    out["rq_p_high"] = pj[:, 6:9].sum(1)
    out["rq_p_slow"] = pj[:, 0:3].sum(1)
    out["rq_p_l3f_fast"] = q["p_l3f_fast"].to_numpy()
    out["rq_p_l3f_slow"] = q["p_l3f_slow"].to_numpy()
    out["rq_p_slow_kick"] = (q["p_lap_slow_kick"] + q["p_lap_slow_spurt"]).to_numpy()
    out["rq_p_high_collapse"] = q["p_lap_high_collapse"].to_numpy()

    # r7・r8: 馬のグループ g × 実測の展開タイプ q の倍率を as-of で作り、事前確率で重み付き平均
    flat = t["surface"].isin(["芝", "ダート"]) & (t["field_size"] >= 5) & (t["finish_position"] >= 1)
    fit = t["year"] <= 2019
    es = np.nanquantile(t.loc[fit & flat, "a_style"], [1 / 3, 2 / 3])
    ek = np.nanquantile(t.loc[fit & flat, "a_l3fz_ewm2"], [1 / 3, 2 / 3])
    g = np.where(t[["a_style", "a_l3fz_ewm2"]].notna().all(axis=1), np.searchsorted(es, t["a_style"].to_numpy()) * 3 + np.searchsorted(ek, t["a_l3fz_ewm2"].to_numpy()), -1)
    qm = t[["race_id"]].merge(q[["race_id", "q_joint"]], on="race_id", how="left")["q_joint"].map({j: i for i, j in enumerate(JOINT)}).to_numpy()
    n = t["field_size"].to_numpy().astype(float)
    fin = t["finish_position"].to_numpy()
    vals = np.column_stack([(fin == 1), 1.0 / n, (fin <= 3), np.minimum(3, n) / n]).astype(float)
    dates = np.sort(t["date"].unique())
    didx = np.searchsorted(dates, t["date"].to_numpy())
    use = (g >= 0) & flat.to_numpy() & ~np.isnan(qm)
    Tq = np.zeros((len(dates), 9, 9, 4))
    np.add.at(Tq, (didx[use], g[use], qm[use].astype(int)), vals[use])
    Cq = np.cumsum(Tq, axis=0) - Tq                      # 同日を含まない
    Cg = Cq.sum(2)                                        # g 全体（q をまとめる）
    P = t[["race_id"]].merge(q[["race_id"] + [f"p_joint_{j}" for j in JOINT]], on="race_id", how="left").iloc[:, 1:].to_numpy()
    res = {}
    for m, (o, e) in (("win", (0, 1)), ("t3", (2, 3))):
        r = np.full(len(t), np.nan)
        ii = np.flatnonzero(g >= 0)
        og, eg = Cg[didx[ii], g[ii], o], Cg[didx[ii], g[ii], e]
        lg = og / np.maximum(eg, 1e-9)
        oq, eq = Cq[didx[ii], g[ii], :, o], Cq[didx[ii], g[ii], :, e]
        lq = (oq + M * lg[:, None]) / (eq + M)
        lr = np.where(lg[:, None] > 0, np.log(np.maximum(lq, 1e-12) / np.maximum(lg[:, None], 1e-12)), 0.0)
        r[ii] = (P[ii] * lr).sum(1)
        r[np.isnan(P).any(axis=1)] = np.nan
        res[m] = r
    o = pd.DataFrame({"race_id": t["race_id"].to_numpy(), "horse_id": t["horse_id"].to_numpy(), "rq_te_win": res["win"], "rq_te_t3": res["t3"]})
    return out, o


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_table()
    print("table", t.shape, flush=True)
    apt = apt_features(t)
    print("apt done", apt.drop(columns=["race_id", "horse_id"]).describe().round(3).T, flush=True)
    bias = bias_features(t)
    rq_race, rq_run = rq_features(t)
    d = t[["race_id", "horse_id", "date", "year", "venue_code", "surface", "distance", "track_condition", "grade", "field_size", "horse_number", "finish_position", "a_style", "a_l3fz_ewm2"]]
    d = d.merge(apt, on=["race_id", "horse_id"], how="left").merge(bias, on="race_id", how="left").merge(rq_race, on="race_id", how="left").merge(rq_run, on=["race_id", "horse_id"], how="left")
    d["gate"] = (d["horse_number"] - 1) / (d["field_size"] - 1).clip(lower=1)
    d.to_parquet(OUT / "aprq_features.parquet")
    print("saved", d.shape, flush=True)


# ------------------------------------------------------------------ 検定
CAND = {
    "BIAS": [("b1_pred_F_x_s", "pred_b_F", "s"), ("b2_pred_D_x_g", "pred_b_D", "g"), ("b3_delta_F_x_s", "d_b_F", "s"), ("b4_delta_D_x_g", "d_b_D", "g"),
             ("b5_static_F_x_s", "st_b_F", "s"), ("b6_static_D_x_g", "st_b_D", "g")],
    "APT": [(f"a{i + 1}_{c}", c, None) for i, c in enumerate(["apt_course_win", "apt_going_win", "apt_sum_win", "apt_course_t3", "apt_going_t3", "apt_sum_t3"])],
    "RQ": [("r1_high_x_s", "rq_p_high", "s"), ("r2_slow_x_s", "rq_p_slow", "s"), ("r3_l3ffast_x_k", "rq_p_l3f_fast", "k"), ("r4_l3fslow_x_k", "rq_p_l3f_slow", "k"),
           ("r5_slowkick_x_k", "rq_p_slow_kick", "k"), ("r6_highcollapse_x_s", "rq_p_high_collapse", "s"), ("r7_te_win", "rq_te_win", None), ("r8_te_t3", "rq_te_t3", None)],
}
SRC_COLS = {
    "BIAS": ["pred_b_F", "pred_b_D", "st_b_F", "st_b_D", "d_b_F", "d_b_D"],
    "APT": ["apt_course_win", "apt_going_win", "apt_sum_win", "apt_course_t3", "apt_going_t3", "apt_sum_t3"],
    "RQ": ["rq_p_high", "rq_p_slow", "rq_p_l3f_fast", "rq_p_l3f_slow", "rq_p_slow_kick", "rq_p_high_collapse", "rq_te_win", "rq_te_t3"],
}
CTX = ["venue_code", "surface", "distance", "grade", "track_condition", "field_size", "a_style", "a_l3fz_ewm2", "gate"]


def _prep(oof_path: Path, years=None) -> pd.DataFrame:
    f = pd.read_parquet(OUT / "aprq_features.parquet")
    o = pd.read_parquet(oof_path)[["race_id", "horse_id", "year", "p_win"]].copy()
    o["race_id"] = o["race_id"].astype(str)
    o["horse_id"] = o["horse_id"].astype(str)
    d = o.merge(f.drop(columns=["year"]), on=["race_id", "horse_id"], how="left")
    d["y"] = (d["finish_position"] == 1).astype(float)
    d["res"] = d["y"] - d["p_win"]
    d["s"] = d["a_style"] - d["a_style"].mean()
    d["k"] = d["a_l3fz_ewm2"] - d["a_l3fz_ewm2"].mean()
    d["g"] = d["gate"] - d["gate"].mean()
    return d


def score_test() -> list[dict]:
    d = _prep(OOF)
    rows = []
    for src, cands in CAND.items():
        for name, col, hz in cands:
            x = d[col] * (d[hz].fillna(0) if hz else 1.0)
            ok = d[col].notna() & (d[hz].notna() if hz else True)
            s = d[ok].assign(x=x[ok])
            s["xc"] = s["x"] - s["x"].mean()
            pr = (s["res"] * s["xc"]).groupby(s["race_id"]).sum()
            z = float(pr.sum() / np.sqrt((pr ** 2).sum()))
            beta = float((s["res"] * s["xc"]).sum() / (s["p_win"] * (1 - s["p_win"]) * s["xc"] ** 2).sum())
            yz = {}
            for y, sy in s.groupby("year"):
                p2 = (sy["res"] * sy["xc"]).groupby(sy["race_id"]).sum()
                yz[int(y)] = round(float(p2.sum() / np.sqrt((p2 ** 2).sum())), 2)
            rows.append({"src": src, "name": name, "n": int(len(s)), "sd_x": round(float(s["x"].std()), 4), "z": round(z, 2), "beta": round(beta, 4), "z_by_year": yz,
                         "pass": bool(abs(z) >= 3.48 and len({np.sign(v) for v in yz.values()}) == 1)})
            print(rows[-1], flush=True)
    return rows


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


PRM = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=50.0, verbose=-1, num_threads=4, seed=7)


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    import lightgbm as lgb

    cat_cols = [c for c in ("venue_code", "surface", "grade", "track_condition") if c in cols]
    a, b = tr[cols].copy(), te[cols].copy()
    for c in cat_cols:
        cat = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        a[c] = pd.Categorical(tr[c].astype(str), categories=cat)
        b[c] = pd.Categorical(te[c].astype(str), categories=cat)
    init = lambda p: np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
    m = lgb.train(PRM, lgb.Dataset(a, tr["y"], init_score=init(tr["p_win"]).to_numpy()), num_boost_round=300)
    return 1 / (1 + np.exp(-(init(te["p_win"]).to_numpy() + m.predict(b, raw_score=True))))


def boost_test() -> dict:
    d = _prep(OOF)
    d = d[d["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    out = {}
    sets = {"control": CTX, **{k: CTX + v for k, v in SRC_COLS.items()}, "ALL": CTX + sum(SRC_COLS.values(), [])}
    gains = {}
    for name, cols in sets.items():
        g = []
        for hold in BOOT_YEARS:
            tr, te = d[d["year"] != hold], d[d["year"] == hold]
            pr = _fit_predict(tr, te, cols)
            g.append(_ll(te["y"].to_numpy(), pr) - _ll(te["y"].to_numpy(), te["p_win"].to_numpy()))
        gains[name] = g
    for name in sets:
        if name == "control":
            continue
        diff = [a - b for a, b in zip(gains[name], gains["control"])]
        out[name] = {"gain_vs_base": [round(x, 6) for x in gains[name]], "diff_vs_control": [round(x, 6) for x in diff], "mean_diff": round(float(np.mean(diff)), 6),
                     "pass": bool(np.mean(diff) <= -0.0002 and all(x < 0 for x in diff))}
        print(name, out[name], flush=True)
    out["control"] = [round(x, 6) for x in gains["control"]]
    print("control", out["control"], flush=True)
    return out


def score() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = score_test()
    (OUT / "score_test.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    b = boost_test()
    (OUT / "boost.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")


def tier2(srcs: list[str]) -> None:
    """全年の OOF の残差に木を当て（対照／対照＋供給元）、test 2025 に 1 回だけ適用。"""
    tr = _prep(OOF)
    tr = tr[tr["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    f = pd.read_parquet(OUT / "aprq_features.parquet")
    te = pd.read_parquet(T25)[["race_id", "horse_id", "p_win"]].copy()
    te["race_id"] = te["race_id"].astype(str)
    te["horse_id"] = te["horse_id"].astype(str)
    te = te.merge(f.drop(columns=["year"]), on=["race_id", "horse_id"], how="left")
    te["y"] = (te["finish_position"] == 1).astype(float)
    te = te[te["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    y = te["y"].to_numpy()
    base = te["p_win"].to_numpy()
    ctrl = _fit_predict(tr, te, CTX)
    res = {"rows": int(len(te)), "ll_base": _ll(y, base), "ll_control": _ll(y, ctrl)}
    rng = np.random.default_rng(0)
    races = te["race_id"].to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    for s in srcs:
        cols = CTX + SRC_COLS[s]
        pr = _fit_predict(tr, te, cols)
        ll_i = lambda p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        dif = ll_i(pr) - ll_i(ctrl)
        per = np.bincount(inv, weights=dif) / np.bincount(inv, weights=np.ones(len(dif)))
        w = np.bincount(inv)
        boots = []
        for _ in range(2000):
            ix = rng.integers(0, len(uniq), len(uniq))
            boots.append(np.bincount(inv, weights=dif)[ix].sum() / w[ix].sum())
        res[s] = {"ll": _ll(y, pr), "diff_vs_control": float(dif.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                  "adopt": bool(dif.mean() <= -0.0002)}
        print(s, res[s], flush=True)
    (OUT / "tier2_TEST2025_EVALUATED.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": build, "score": score}.get(cmd, lambda: tier2(sys.argv[2].split(",")))()
