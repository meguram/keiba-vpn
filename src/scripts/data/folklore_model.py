"""通説の当てはまりを、市場の見込みを土台にしたモデルで束ねて、回収率で評価する（通説の検証の章の第 10 節）。

    python -m src.scripts.data.folklore_model flags   # 全出走 × 全通説の当てはまり → <FOLKLORE_OUT>/flags.parquet
    python -m src.scripts.data.folklore_model fit     # 学習・評価 → <FOLKLORE_OUT>/model.json（章は folklore_verify report で作る）

第 9 節（格の重みの得点）の課題と、ここでの対応:
1. 回収率を目標にしていない（得点の 1 位 = 人気馬）
   → 市場の確率を土台（オフセット）にして通説で直したモデルの確率 × オッズ = 期待値で買う。1 レースで 0 頭も複数頭もある。
2. 格は確かさ（q 値）で、効き目の大きさではない
   → モデルの係数（効き目）で重みを付ける。モデルを使わない版として、人気をそろえた差 Δ を経験ベイズで縮めた重みの得点も作る。
3. 通説どうしの重なりを二重に数える
   → 全部を 1 つの回帰に入れて同時に推定する（L2 の縮小。強さは時系列の内側の検証で選ぶ）。
4. 目的変数（3 着内）と賭け方（単勝）がずれる
   → 単勝は 1 着の条件付きロジット（レース内の確率の合計 = 1）、複勝は 3 着内のロジスティック回帰で別々に作る。
5. 確定オッズの情報を使っている
   → 人気・オッズを使う通説を外した版と、確定オッズの代わりに『レース前の情報だけの想定オッズ』（odds_pred_eval の share_raw、
     年次 walk-forward）を土台にした版を作る。払戻は実際の確定オッズで計算する。
6. 今のモデルに足して上積みがあるか
   → 本番系モデル（v6 の OOF、2022〜2024）の p_win・p_top3 を土台にして、通説の得点を足したときの上積みを見る。

学習は 2016〜2022 年（想定オッズ・本番系モデルの OOF は 2022 年からしか無いので、その版は 2022 年）、評価は 2023〜2025 年（本番系は 2023〜2024）。
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data.folklore_verify import MIN_N0, MIN_N1, MIN_R1, OUT, _payouts, hyps  # noqa: E402

WP = ROOT / "data/local/modeling/odds_pred/wp_oof_pred.parquet"
WP_RAW = ROOT / "data/local/modeling/odds_pred/wp_raw_oof.parquet"
# 当日の人気・オッズから作る列（これを使う通説は確定オッズが無いと当てはめられない）
ODDS_TOKENS = ('"pop"', '"odds"', '"q"', "jday_minpop", "tr_pop_rank", "own_pop_rank", "top2q", "fav_odds", "has_pacemaker", '"p_mkt"')
TRAIN, TEST = (2016, 2022), (2023, 2025)


# ---------------------------------------------------------------- 当てはまりの表
def odds_dependent(h) -> bool:
    src = ""
    for fn in (h.elig, h.flag):
        try:
            src += inspect.getsource(fn)
        except (OSError, TypeError):
            pass
    return any(t in src for t in ODDS_TOKENS)


def flags() -> None:
    d = pd.read_pickle(OUT / "_prep.pkl")
    cols, meta = {}, []
    for h in hyps():
        if h.elig is None or h.hid.startswith("Z"):
            continue
        try:
            e = np.asarray(pd.Series(h.elig(d), index=d.index).fillna(False).astype(bool))
            f = np.asarray(pd.Series(h.flag(d), index=d.index).fillna(False).astype(bool)) & e
        except Exception:  # noqa: BLE001
            continue
        if f.sum() < 300:
            continue
        cols[h.hid] = f.astype(np.int8)
        meta.append({"hid": h.hid, "dir": h.direction, "odds": odds_dependent(h), "n": int(f.sum()), "how": h.how})
    base = d[["race_id", "horse_id", "horse_number", "year", "fin", "odds", "pop", "q", "p_mkt", "field"]].copy()
    base["race_id"] = base["race_id"].astype(str)
    base["horse_id"] = base["horse_id"].astype(str)
    base["horse_number"] = pd.to_numeric(base["horse_number"], errors="coerce")
    F = pd.DataFrame(cols, index=d.index)
    out = pd.concat([base, F], axis=1)
    out = out.merge(_payouts(sorted(out["year"].unique().tolist())), on=["race_id", "horse_number"], how="left")
    out["win"] = out["win"].fillna(0.0)
    out["plc"] = out["plc"].fillna(0.0)
    # 複勝オッズ（確定、下限）
    po = []
    for y in sorted(out["year"].unique()):
        o = pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_odds_flat.parquet", columns=["race_id", "horse_number", "place_odds_min", "place_odds_max"])
        po.append(o)
    po = pd.concat(po)
    po["race_id"] = po["race_id"].astype(str)
    po["horse_number"] = pd.to_numeric(po["horse_number"], errors="coerce")
    for c in ("place_odds_min", "place_odds_max"):
        po[c] = pd.to_numeric(po[c], errors="coerce")
    out = out.merge(po.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    out = out.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    out.to_parquet(OUT / "flags.parquet.tmp", index=False)
    os.replace(OUT / "flags.parquet.tmp", OUT / "flags.parquet")  # 読む側が書き込み途中のファイルを見ないように入れ替える
    json.dump(meta, open(OUT / "flags_meta.json", "w"), ensure_ascii=False, indent=1)
    print("flags", out.shape, "odds-dependent", sum(m["odds"] for m in meta), "of", len(meta))


# ---------------------------------------------------------------- 推定（L2 つき）
def _starts(race: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, race[1:] != race[:-1]])


def fit_clogit(X, off, y, race, lam, pen) -> np.ndarray:
    """条件付きロジット: P(i が 1 着) = softmax_race(off_i + X_i β)。y はレース内で合計 1（同着は等分）。pen: 罰則を掛ける列。"""
    st = _starts(race)
    n, k = X.shape

    def f(b):
        eta = off + X @ b
        mx = np.maximum.reduceat(eta, st)
        rep = np.repeat(mx, np.diff(np.r_[st, n]))
        ex = np.exp(eta - rep)
        s = np.add.reduceat(ex, st)
        lse = np.log(s) + mx
        p = ex / np.repeat(s, np.diff(np.r_[st, n]))
        loss = lse.sum() - (y * eta).sum() + 0.5 * lam * (pen * b * b).sum()
        g = X.T @ (p - y) + lam * pen * b
        return loss, np.asarray(g).ravel()

    r = minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x


def pred_clogit(X, off, race, b) -> np.ndarray:
    st = _starts(race)
    n = len(off)
    eta = off + X @ b
    mx = np.maximum.reduceat(eta, st)
    ex = np.exp(eta - np.repeat(mx, np.diff(np.r_[st, n])))
    return ex / np.repeat(np.add.reduceat(ex, st), np.diff(np.r_[st, n]))


def fit_logit(X, off, y, lam, pen) -> np.ndarray:
    k = X.shape[1]

    def f(b):
        eta = off + X @ b
        loss = (np.logaddexp(0, eta) - y * eta).sum() + 0.5 * lam * (pen * b * b).sum()
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (p - y) + lam * pen * b
        return loss, np.asarray(g).ravel()

    return minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def pred_logit(X, off, b) -> np.ndarray:
    return 1 / (1 + np.exp(-(off + X @ b)))


def _design(d: pd.DataFrame, fl: list[str], base_cols: list[str]):
    """列 = [土台の補正（罰則なし）…, 通説の当てはまり（罰則あり）…]。"""
    B = d[base_cols].to_numpy(float) if base_cols else np.zeros((len(d), 0))
    F = sparse.csr_matrix(d[fl].to_numpy(np.float32)) if fl else sparse.csr_matrix((len(d), 0))
    X = sparse.hstack([sparse.csr_matrix(B), F]).tocsr()
    pen = np.r_[np.zeros(B.shape[1]), np.ones(F.shape[1])]
    return X, pen


def _ll_win(p, y, race):
    """レースごとの 1 着の対数損失（勝ち馬の -log p）。"""
    return -(y * np.log(np.clip(p, 1e-9, 1))).sum() / len(np.unique(race))


def _ll_bin(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _race_se_diff(a, b, race):
    """レース単位で束ねた損失の差の平均と標準誤差。"""
    s = pd.Series(a - b).groupby(race).sum()
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))


def model_win(tr, te, fl, base_cols, off_col, lams=(1, 10, 100, 300, 1000, 3000, 10000), lam_fixed=None):
    """単勝: 条件付きロジット。λ は学習期間の最後の 2 年を検証にして選ぶ。"""
    def Xy(d):
        X, pen = _design(d, fl, base_cols)
        y = (d["fin"] == 1).astype(float).to_numpy()
        y = y / pd.Series(y).groupby(d["race_id"].to_numpy()).transform("sum").replace(0, 1).to_numpy()
        return X, pen, d[off_col].to_numpy(float), y, d["race_id"].to_numpy()

    best = (np.inf, lam_fixed if lam_fixed is not None else lams[0])
    if fl and lam_fixed is None:
        yrs = sorted(tr["year"].unique())
        cut = yrs[-3] if len(yrs) >= 5 else yrs[-2]  # 内側の検証 = 学習期間の最後の 2 年（短い期間は最後の 1 年）
        inner_tr, inner_va = tr[tr["year"] <= cut], tr[tr["year"] > cut]
        X1, pen, o1, y1, r1 = Xy(inner_tr)
        X2, _, o2, y2, r2 = Xy(inner_va)
        for lam in lams:
            b = fit_clogit(X1, o1, y1, r1, lam, pen)
            ll = _ll_win(pred_clogit(X2, o2, r2, b), y2, r2)
            if ll < best[0]:
                best = (ll, lam)
    X, pen, o, y, r = Xy(tr)
    b = fit_clogit(X, o, y, r, best[1], pen)
    Xt, _, ot, _, rt = Xy(te)
    return pred_clogit(Xt, ot, rt, b), b, best[1]


def model_top3(tr, te, fl, base_cols, off_col, lams=(1, 10, 100, 300, 1000, 3000, 10000), lam_fixed=None):
    def Xy(d):
        X, pen = _design(d, fl, base_cols + ["_one"])
        return X, pen, d[off_col].to_numpy(float), (d["fin"] <= 3).astype(float).to_numpy()

    tr, te = tr.assign(_one=1.0), te.assign(_one=1.0)
    best = (np.inf, lam_fixed if lam_fixed is not None else lams[0])
    if fl and lam_fixed is None:
        yrs = sorted(tr["year"].unique())
        cut = yrs[-3] if len(yrs) >= 5 else yrs[-2]  # 内側の検証 = 学習期間の最後の 2 年（短い期間は最後の 1 年）
        inner_tr, inner_va = tr[tr["year"] <= cut], tr[tr["year"] > cut]
        X1, pen, o1, y1 = Xy(inner_tr)
        X2, _, o2, y2 = Xy(inner_va)
        for lam in lams:
            b = fit_logit(X1, o1, y1, lam, pen)
            ll = _ll_bin(pred_logit(X2, o2, b), y2)
            if ll < best[0]:
                best = (ll, lam)
    X, pen, o, y = Xy(tr)
    b = fit_logit(X, o, y, best[1], pen)
    Xt, _, ot, _ = Xy(te)
    return pred_logit(Xt, ot, b), b, best[1]



def _cmp(pa, pb, y, race, kind):
    """2 つの予測の対数損失の差（a − b、負なら a が良い）と、レースで束ねた標準誤差。win はレースあたり、top3 は 1 頭あたり。"""
    if kind == "win":
        la, lb = -(y * np.log(np.clip(pa, 1e-9, 1))), -(y * np.log(np.clip(pb, 1e-9, 1)))
        m, se = _race_se_diff(la, lb, race)
        return {"d": m, "se": se}
    la = -(y * np.log(np.clip(pa, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pa, 1e-6, 1)))
    lb = -(y * np.log(np.clip(pb, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pb, 1e-6, 1)))
    m, se = _race_se_diff(la, lb, race)
    k = len(y) / len(np.unique(race))
    return {"d": m / k, "se": se / k}


# ---------------------------------------------------------------- 期待値で買ったときの成績
EV_GRID = [0.0, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]


def ev_curve(ev: np.ndarray, ret: np.ndarray, hit: np.ndarray, race: np.ndarray, pop: np.ndarray) -> list[dict]:
    rows = []
    nr = len(np.unique(race))
    for th in EV_GRID:
        m = ev >= th
        n = int(m.sum())
        if n < 30:
            continue
        r = ret[m] / 100
        # レースで束ねた標準誤差（同じレースで複数頭買うと払戻が連動する）
        s = pd.DataFrame({"r": r, "k": race[m]}).groupby("k")["r"].agg(["sum", "count"])
        roi = s["sum"].sum() / s["count"].sum()
        z = s["sum"] - roi * s["count"]
        se = float(np.sqrt((z ** 2).sum()) / s["count"].sum())
        rows.append({"th": th, "n": n, "races": int(len(s)), "race_share": len(s) / nr, "hit": float(hit[m].mean()), "roi": float(roi), "se": se,
                     "pop": float(np.nanmean(pop[m]))})
    return rows


def _evaluate(name, te, p_win, p3, win_odds, plc_odds, out):
    race = te["race_id"].to_numpy()
    pop = te["pop"].to_numpy(float)
    res = {}
    if p_win is not None:
        res["win"] = ev_curve(p_win * win_odds, te["win"].to_numpy(), (te["fin"] == 1).to_numpy(), race, pop)
    if p3 is not None:
        res["plc"] = ev_curve(p3 * plc_odds, te["plc"].to_numpy(), (te["fin"] <= 3).to_numpy(), race, pop)
    out["ev"][name] = res


def fit() -> None:
    d = pd.read_parquet(OUT / "flags.parquet")
    meta = json.load(open(OUT / "flags_meta.json"))
    allf = [m["hid"] for m in meta]
    nodds = [m["hid"] for m in meta if not m["odds"]]
    d = d[d["odds"] > 0].copy()
    d["lq"] = np.log(d["q"].clip(1e-4))
    d["lgt_mkt"] = np.log(d["p_mkt"].clip(1e-4, 1 - 1e-4) / (1 - d["p_mkt"].clip(1e-4, 1 - 1e-4)))
    d["zero"] = 0.0
    # 複勝の市場（確定の複勝オッズの下限・上限）。単勝の市場だけを土台にすると、複勝の売り場が嫌った馬を『期待値が高い』と拾ってしまう
    d["lpmin"] = np.log(d["place_odds_min"].where(d["place_odds_min"] > 0).fillna(d["odds"] / 3).clip(1.0))
    d["lpmax"] = np.log(d["place_odds_max"].where(d["place_odds_max"] > 0).fillna(d["odds"] / 3).clip(1.0))
    # 土台の補正を曲線にする（直線だと複勝の売り場の情報を使い切れず、期待値の高い馬ほど見込み違いになった）
    d["lq2"] = d["lq"] ** 2
    d["lq3"] = d["lq"] ** 3
    for c in ("lgt_mkt", "lpmin", "lpmax"):
        d[c + "2"] = d[c] ** 2
    d["lgt_x_pmin"] = d["lgt_mkt"] * d["lpmin"]
    d["pgap"] = d["lpmin"] - d["lpmax"]
    WB = ["lq", "lq2", "lq3"]
    PB = ["lgt_mkt", "lpmin", "lpmax", "lgt_mkt2", "lpmin2", "lpmax2", "lgt_x_pmin", "pgap"]
    tr = d[d["year"].between(*TRAIN)]
    te = d[d["year"].between(*TEST)].copy()
    out: dict = {"n_flags": len(allf), "n_flags_nodds": len(nodds), "odds_hids": sorted(set(allf) - set(nodds)), "ev": {}, "ll": {}, "coef": {}, "lam": {}}
    yw = (te["fin"] == 1).astype(float).to_numpy()
    y3 = (te["fin"] <= 3).astype(float).to_numpy()
    race = te["race_id"].to_numpy()
    plc_odds = te["place_odds_min"].fillna(0).to_numpy(float)
    win_odds = te["odds"].to_numpy(float)

    # 1) 確定オッズの市場を土台にする（補正の係数 α は罰則なしで推定 = 本命・大穴の偏りを取り込む）
    preds = {}
    for name, fl in (("M0", []), ("M1", allf), ("M2", nodds)):
        pw, bw, lw = model_win(tr, te, fl, WB, "zero")
        p3, b3, l3 = model_top3(tr, te, fl, PB, "zero")
        preds[name] = (pw, p3)
        out["lam"][name] = [lw, l3]
        if fl:
            out["coef"][name] = {"win": dict(zip(fl, map(float, bw[len(WB):]))), "top3": dict(zip(fl, map(float, b3[len(PB):-1])))}
        out["coef"].setdefault("base", {})[name] = {"win_base": list(map(float, bw[:len(WB)])), "top3_base": list(map(float, b3[:len(PB)])), "icpt_top3": float(b3[-1])}
        _evaluate(name, te, pw, p3, win_odds, plc_odds, out)
        print(name, "lam", lw, l3, flush=True)
    q = te["q"].to_numpy(float)
    pm = te["p_mkt"].to_numpy(float)
    llw = {k: -(yw * np.log(np.clip(v[0], 1e-9, 1))) for k, v in preds.items()}
    llw["MKT"] = -(yw * np.log(np.clip(q, 1e-9, 1)))
    ll3 = {k: -(y3 * np.log(np.clip(v[1], 1e-6, 1)) + (1 - y3) * np.log(np.clip(1 - v[1], 1e-6, 1))) for k, v in preds.items()}
    ll3["MKT"] = -(y3 * np.log(np.clip(pm, 1e-6, 1)) + (1 - y3) * np.log(np.clip(1 - pm, 1e-6, 1)))
    for k in llw:
        mw, sw = _race_se_diff(llw[k], llw["MKT"], race)
        m3, s3 = _race_se_diff(ll3[k], ll3["MKT"], race)
        out["ll"][k] = {"win": float(llw[k].sum() / len(np.unique(race))), "win_d": mw, "win_se": sw, "top3": float(ll3[k].mean()), "top3_d": m3 / te.groupby("race_id").size().mean(), "top3_se": s3 / te.groupby("race_id").size().mean()}
    _evaluate("MKT", te, q, pm, win_odds, plc_odds, out)  # 市場そのまま（期待値はほぼ 0.8 で一定）
    out["cmp"] = {}
    for a_, b_ in (("M1", "M0"), ("M2", "M0"), ("M0", "MKT")):
        pa = preds[a_]
        pb = preds.get(b_, (q, pm))
        out["cmp"][f"{a_}-{b_}"] = {"win": _cmp(pa[0], pb[0], yw, race, "win"), "top3": _cmp(pa[1], pb[1], y3, race, "top3")}

    # 2) 効き目の重み（モデルを使わない版）: 発見期間の人気をそろえた差 Δ を経験ベイズで縮める
    tr_raw = pd.read_parquet(OUT / "tests_raw.parquet")
    g = tr_raw[(tr_raw["period"] == "disc") & (tr_raw["dim"] == "全体") & (tr_raw["n1"] >= MIN_N1) & (tr_raw["r1"] >= MIN_R1) & (tr_raw["n0"] >= MIN_N0)
               & tr_raw["hid"].isin(allf)].copy()
    tau2 = max(float(g["adj"].var() - (g["adj_se"] ** 2).mean()), 1e-8)
    g["w"] = g["adj"] * tau2 / (tau2 + g["adj_se"] ** 2)
    out["eb"] = {"tau": float(np.sqrt(tau2)), "w": dict(zip(g["hid"], map(float, g["w"])))}
    ww = g.set_index("hid")["w"]
    for name, fl in (("EB1", [h for h in allf if h in ww.index]), ("EB2", [h for h in nodds if h in ww.index])):
        add = te[fl].to_numpy(float) @ ww[fl].to_numpy()
        p3 = np.clip(preds["M0"][1] + add, 0.005, 0.995)  # 土台は M0（単複の市場）の 3 着内確率
        _evaluate(name, te, None, p3, win_odds, plc_odds, out)
        out["ll"][name] = {"top3": float(_ll_bin(p3, y3))}

    # 3) レース前の情報だけの想定オッズ（share_raw）を土台にする版。人気・オッズの通説は使えないので外す
    raw = pd.read_parquet(WP_RAW)
    raw["race_id"] = raw["race_id"].astype(str)
    raw["horse_id"] = raw["horse_id"].astype(str)
    e = d.merge(raw[["race_id", "horse_id", "share_raw", "place_raw"]], on=["race_id", "horse_id"], how="inner").sort_values(["race_id", "horse_number"])
    e = e[e["share_raw"].notna() & e["place_raw"].notna()]
    ok = e.groupby("race_id")["share_raw"].transform("size") == e.groupby("race_id")["field"].transform("first")
    e = e[ok]
    etr, ete = e[e["year"] == 2022], e[e["year"].between(*TEST)].copy()  # 想定オッズの OOF は 2022 年から
    out["pre_n"] = {"train": int(etr["race_id"].nunique()), "test": int(ete["race_id"].nunique())}
    pp = {}
    yw2 = (ete["fin"] == 1).astype(float).to_numpy()
    y32 = (ete["fin"] <= 3).astype(float).to_numpy()
    r2 = ete["race_id"].to_numpy()
    for name, fl in (("P0", []), ("P2", nodds)):
        # 学習が 1 年だけなので、λ は確定オッズの版（M2）で選んだ値を使う
        pw, bw, lw = model_win(etr, ete, fl, ["share_raw"], "zero", lam_fixed=out["lam"]["M2"][0])
        p3, b3, l3 = model_top3(etr, ete, fl, ["place_raw"], "zero", lam_fixed=out["lam"]["M2"][1])
        pp[name] = (pw, p3)
        out["lam"][name] = [lw, l3]
        # (a) 確定オッズで期待値を測る（締切直前のオッズで買える場合の上限）
        _evaluate(name, ete, pw, p3, ete["odds"].to_numpy(float), ete["place_odds_min"].fillna(0).to_numpy(float), out)
        out["ll"][name] = {"win": float(-(yw2 * np.log(np.clip(pw, 1e-9, 1))).sum() / ete["race_id"].nunique()), "top3": float(_ll_bin(p3, y32))}
        print(name, "lam", lw, l3, flush=True)
    # (b) レース前の想定オッズだけで判断し、払戻は確定オッズ（前日に買う場合）。
    #     想定単勝オッズ = 0.8 / 想定の勝つ確率（P0）、想定複勝オッズ = exp(a + b·place_raw)（odds_pred_config の係数）
    cfg = json.load(open(ROOT / "data/local/modeling/odds_pred/odds_pred_config.json"))
    win_hat = 0.8 / np.clip(pp["P0"][0], 1e-4, 1)
    plc_hat = np.exp(cfg["place_mid"]["a"] + cfg["place_mid"]["b"] * ete["place_raw"].to_numpy(float))
    for name in ("P0", "P2"):
        out["ev"][name + "_pre"] = {"win": ev_curve(pp[name][0] * win_hat, ete["win"].to_numpy(), yw2 == 1, r2, ete["pop"].to_numpy(float)),
                                    "plc": ev_curve(pp[name][1] * plc_hat, ete["plc"].to_numpy(), y32 == 1, r2, ete["pop"].to_numpy(float))}
    out["cmp"]["P2-P0"] = {"win": _cmp(pp["P2"][0], pp["P0"][0], yw2, r2, "win"), "top3": _cmp(pp["P2"][1], pp["P0"][1], y32, r2, "top3")}

    # 4) 本番系モデル（v6 の OOF の p_win・p_top3、2022〜2024）に通説を足す
    wp = pd.read_parquet(WP, columns=["race_id", "horse_id", "p_win", "p_top3"])
    wp["race_id"] = wp["race_id"].astype(str)
    wp["horse_id"] = wp["horse_id"].astype(str)
    v = d.merge(wp, on=["race_id", "horse_id"], how="inner").sort_values(["race_id", "horse_number"])
    v = v[v["p_win"].notna() & v["p_top3"].notna()]
    ok = v.groupby("race_id")["p_win"].transform("size") == v.groupby("race_id")["field"].transform("first")
    v = v[ok].copy()
    v["lpw"] = np.log(v["p_win"].clip(1e-4))
    p3c = v["p_top3"].clip(1e-4, 1 - 1e-4)
    v["lp3"] = np.log(p3c / (1 - p3c))
    vtr, vte = v[v["year"] == 2022], v[v["year"].between(2023, 2024)].copy()
    out["own_n"] = {"train": int(vtr["race_id"].nunique()), "test": int(vte["race_id"].nunique())}
    # 通説の得点は 2016〜2021 の M2（オッズを使わない通説）の係数で作り、2022 で本番系モデルとの重みだけを当てはめる
    tr21 = d[d["year"].between(2016, 2021)]
    _, bw21, _ = model_win(tr21, tr21.head(1000), nodds, WB, "zero")
    _, b321, _ = model_top3(tr21, tr21.head(1000), nodds, PB, "zero")
    for z in (vtr, vte):
        z["fs_w"] = z[nodds].to_numpy(float) @ bw21[len(WB):]
        z["fs_3"] = z[nodds].to_numpy(float) @ b321[len(PB):-1]
    ep = {}
    for name, bw_cols, b3_cols in (("E0", ["lpw"], ["lp3"]), ("E1", ["lpw", "fs_w"], ["lp3", "fs_3"]),
                                   ("E0m", ["lpw"] + WB, ["lp3"] + PB), ("E1m", ["lpw"] + WB + ["fs_w"], ["lp3"] + PB + ["fs_3"])):
        pw, bw, _ = model_win(vtr, vte, [], bw_cols, "zero")
        p3, b3, _ = model_top3(vtr, vte, [], b3_cols, "zero")
        _evaluate(name, vte, pw, p3, vte["odds"].to_numpy(float), vte["place_odds_min"].fillna(0).to_numpy(float), out)
        yw2 = (vte["fin"] == 1).astype(float).to_numpy()
        y32 = (vte["fin"] <= 3).astype(float).to_numpy()
        out["ll"][name] = {"win": float(-(yw2 * np.log(np.clip(pw, 1e-9, 1))).sum() / vte["race_id"].nunique()), "top3": float(_ll_bin(p3, y32)),
                           "coef_win": list(map(float, bw)), "coef_top3": list(map(float, b3))}
        ep[name] = (pw, p3)
        print(name, out["ll"][name], flush=True)
    rv = vte["race_id"].to_numpy()
    ywv, y3v = (vte["fin"] == 1).astype(float).to_numpy(), (vte["fin"] <= 3).astype(float).to_numpy()
    for a_, b_ in (("E1", "E0"), ("E1m", "E0m")):
        out["cmp"][f"{a_}-{b_}"] = {"win": _cmp(ep[a_][0], ep[b_][0], ywv, rv, "win"), "top3": _cmp(ep[a_][1], ep[b_][1], y3v, rv, "top3")}
    json.dump(out, open(OUT / "model.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out["cmp"], ensure_ascii=False))
    for k, v_ in out["ev"].items():
        for kind, rows in v_.items():
            print(k, kind, [(r["th"], r["n"], round(r["roi"], 3), round(r["se"], 3)) for r in rows])
    print(json.dumps(out["ll"], ensure_ascii=False, indent=0)[:3000])


# ---------------------------------------------------------------- 第 12 節: ベース（実力）+ エッジ（人気より走る条件）+ 期待値
def _edges(meta_odds: set) -> dict:
    """エッジ = 発見期間（2016〜2022）だけで『単独で人気より走る（走らない）』と確かめた条件。重みは人気をそろえた差を経験ベイズで縮めた値（pt）。
    通説の型（オッズを使わないもの、全体の判定）と、血統 × 条件替わりの (a)（条件の中でその血統）。BH の q < 0.10。"""
    from scipy.stats import norm as _n

    from src.scripts.data.folklore_verify import bh as _bh

    tr = pd.read_parquet(OUT / "tests_raw.parquet")
    g = tr[(tr["period"] == "disc") & (tr["dim"] == "全体") & (tr["n1"] >= MIN_N1) & (tr["r1"] >= MIN_R1) & (tr["n0"] >= MIN_N0)
           & ~tr["hid"].str.startswith("Z") & ~tr["hid"].isin(meta_odds)].copy()
    g["q"] = _bh(np.asarray(2 * _n.sf(np.abs(g["adj"] / g["adj_se"])), float))
    tau2 = max(float(g["adj"].var() - (g["adj_se"] ** 2).mean()), 1e-8)
    g["w"] = g["adj"] * tau2 / (tau2 + g["adj_se"] ** 2)
    meta = {m["hid"]: m for m in json.load(open(OUT / "meta.json"))}
    g["agree"] = [np.sign(a) == meta[h]["dir"] for h, a in zip(g["hid"], g["adj"])]
    hyp = g[g["q"] < 0.10][["hid", "w", "agree"]]
    st = pd.read_parquet(OUT / "sire_tests_raw.parquet")
    b = st[(st["period"] == "disc") & (st["level"] != "placebo") & (st["n1"] >= 200) & (st["r1"] >= 150) & (st["a_se"] > 0)].copy()
    b["p"] = 2 * _n.sf(np.abs(b["a"] / b["a_se"]))
    b["q"] = b.groupby("level")["p"].transform(lambda x: pd.Series(_bh(x.to_numpy()), index=x.index))
    tau2b = max(float((b["a"] ** 2).mean() - (b["a_se"] ** 2).mean()), 1e-8)
    b["w"] = b["a"] * tau2b / (tau2b + b["a_se"] ** 2)
    blood = b[b["q"] < 0.10][["level", "cond", "key", "w"]]
    return {"hyp": hyp, "blood": blood}


def laps() -> None:
    from src.scripts.data.folklore_sire import add_blood, conditions

    d = pd.read_parquet(OUT / "flags.parquet")
    meta = json.load(open(OUT / "flags_meta.json"))
    odds_h = {m["hid"] for m in meta if m["odds"]}
    ed = _edges(odds_h)
    # 血統のエッジの当てはまり
    prep = pd.read_pickle(OUT / "_prep.pkl")
    prep = prep[prep["year"] >= 2022].copy()
    pb = add_blood(prep)
    cm = conditions(pb)
    eb = np.zeros(len(pb))
    for r in ed["blood"].itertuples():
        e, f = cm[r.cond]
        hit = np.asarray((f & e).fillna(False), bool) & (pb[r.level].astype(object) == r.key).to_numpy()
        eb += r.w * hit
    be = pd.DataFrame({"race_id": pb["race_id"].astype(str).to_numpy(), "horse_id": pb["horse_id"].astype(str).to_numpy(), "e_blood": eb})
    wp = pd.read_parquet(WP, columns=["race_id", "horse_id", "p_win", "p_top3", "share_raw", "place_raw"])
    wp["race_id"] = wp["race_id"].astype(str)
    wp["horse_id"] = wp["horse_id"].astype(str)
    v = d[d["year"].between(2022, 2024) & (d["odds"] > 0)].merge(wp, on=["race_id", "horse_id"], how="inner").merge(be, on=["race_id", "horse_id"], how="left")
    v = v[v["p_win"].notna() & v["p_top3"].notna()].sort_values(["race_id", "horse_number"])
    v = v[v.groupby("race_id")["p_win"].transform("size") == v.groupby("race_id")["field"].transform("first")].copy()
    v["e_blood"] = v["e_blood"].fillna(0.0)
    hw = ed["hyp"].set_index("hid")
    hh = [h for h in hw.index if h in v.columns]
    hh_ok = [h for h in hh if hw.loc[h, "agree"]]
    v["e_hyp"] = v[hh].to_numpy(float) @ hw.loc[hh, "w"].to_numpy() if hh else 0.0
    v["e_hyp_ok"] = v[hh_ok].to_numpy(float) @ hw.loc[hh_ok, "w"].to_numpy() if hh_ok else 0.0
    v["E_all"] = v["e_hyp"] + v["e_blood"]
    v["E_ok"] = v["e_hyp_ok"]
    v["lpw"] = np.log(v["p_win"].clip(1e-4))
    p3c = v["p_top3"].clip(1e-4, 1 - 1e-4)
    v["lp3"] = np.log(p3c / (1 - p3c))
    v["zero"] = 0.0
    vtr, vte = v[v["year"] == 2022], v[v["year"].between(2023, 2024)].copy()
    out = {"n_edge_hyp": len(hh), "n_edge_hyp_ok": len(hh_ok), "n_edge_blood": len(ed["blood"]), "edge_hyp": {h: [float(hw.loc[h, "w"]), bool(hw.loc[h, "agree"])] for h in hh},
           "edge_blood": [[r.level, r.cond, r.key, float(r.w)] for r in ed["blood"].itertuples()], "races": int(vte["race_id"].nunique()), "strat": {}, "ev": {}}
    race = vte["race_id"].to_numpy()
    odds = vte["odds"].to_numpy(float)
    po = vte["place_odds_min"].fillna(0).to_numpy(float)
    rw, rp = vte["win"].to_numpy(), vte["plc"].to_numpy()
    yw, y3 = (vte["fin"] == 1).to_numpy(), (vte["fin"] <= 3).to_numpy()
    pop = vte["pop"].to_numpy(float)
    # 前日判断用の想定オッズ（odds_pred の share_raw・place_raw）
    cfg = json.load(open(ROOT / "data/local/modeling/odds_pred/odds_pred_config.json"))
    sh = np.exp(vte["share_raw"].to_numpy(float))
    sh = sh / pd.Series(sh).groupby(race).transform("sum").to_numpy()
    win_hat = 0.8 / np.clip(sh, 1e-4, 1)
    plc_hat = np.exp(cfg["place_mid"]["a"] + cfg["place_mid"]["b"] * vte["place_raw"].to_numpy(float))

    def stat(m, ret, hit):
        n = int(m.sum())
        if n < 30:
            return None
        s_ = pd.DataFrame({"r": ret[m] / 100, "k": race[m]}).groupby("k")["r"].agg(["sum", "count"])
        roi = s_["sum"].sum() / s_["count"].sum()
        se = float(np.sqrt(((s_["sum"] - roi * s_["count"]) ** 2).sum()) / s_["count"].sum())
        return {"n": n, "hit": float(hit[m].mean()), "roi": float(roi), "se": se, "pop": float(np.nanmean(pop[m]))}

    for name, ecol in (("B", None), ("BE", "E_all"), ("BEok", "E_ok")):
        pw, bw, _ = model_win(vtr, vte, [], ["lpw"] + ([ecol] if ecol else []), "zero")
        p3, b3, _ = model_top3(vtr, vte, [], ["lp3"] + ([ecol] if ecol else []), "zero")
        out[f"coef_{name}"] = {"win": list(map(float, bw)), "top3": list(map(float, b3))}
        top = pd.Series(pw).groupby(race).transform("max").to_numpy() == pw
        top3_ = pd.Series(p3).groupby(race).transform("max").to_numpy() == p3
        ev_w, ev_p = pw * odds, p3 * po
        st_ = {
            "指数 1 位・全レース（単勝）": stat(top, rw, yw), "指数 1 位・全レース（複勝）": stat(top3_, rp, y3),
            "期待値 1.2 以上（単勝・確定オッズ）": stat(ev_w >= 1.2, rw, yw), "期待値 1.2 以上（複勝・確定オッズ）": stat(ev_p >= 1.2, rp, y3),
            "期待値 1.2 以上・1 倍台を除く（単勝）": stat((ev_w >= 1.2) & (odds >= 2.0), rw, yw),
            "期待値 1.2 以上（単勝・前日の想定オッズで判断）": stat(pw * win_hat >= 1.2, rw, yw),
            "期待値 1.2 以上（複勝・前日の想定オッズで判断）": stat(p3 * plc_hat >= 1.2, rp, y3),
        }
        out["strat"][name] = st_
        out["ev"][name] = {"win": ev_curve(ev_w, rw, yw, race, pop), "plc": ev_curve(ev_p, rp, y3, race, pop)}
        out[f"ll_{name}"] = {"win": float(-(yw * np.log(np.clip(pw, 1e-9, 1))).sum() / len(np.unique(race))), "top3": float(_ll_bin(p3, y3.astype(float)))}
        if name == "B":
            pB = (pw, p3)
        else:
            out[f"cmp_{name}"] = {"win": _cmp(pw, pB[0], yw.astype(float), race, "win"), "top3": _cmp(p3, pB[1], y3.astype(float), race, "top3")}
        print(name, json.dumps(st_, ensure_ascii=False), flush=True)
    m = json.load(open(OUT / "model.json"))
    m["laps"] = out
    json.dump(m, open(OUT / "model.json", "w"), ensure_ascii=False, indent=1)
    print(out["n_edge_hyp"], out["n_edge_hyp_ok"], out["n_edge_blood"], out.get("cmp_BE"), out.get("cmp_BEok"))


# ---------------------------------------------------------------- 章（第 10 節）
MODEL_NAMES = {
    "MKT": "市場そのまま", "M0": "市場の補正だけ（通説なし）", "M1": "市場 + 全部の通説", "M2": "市場 + オッズを使わない通説",
    "EB1": "効き目の重み（全部）", "EB2": "効き目の重み（オッズを使わない）",
    "P0": "想定オッズだけ", "P2": "想定オッズ + 通説", "P0_pre": "想定オッズだけ（前日判断）", "P2_pre": "想定オッズ + 通説（前日判断）",
    "E0": "本番系モデル", "E1": "本番系 + 通説", "E0m": "本番系 + 市場", "E1m": "本番系 + 市場 + 通説",
}
MODEL_COL = {"M0": "#3987e5", "M2": "#d95926", "M1": "#199e70", "EB2": "#c98500", "P0": "#3987e5", "P2": "#d95926", "P0_pre": "#3987e5", "P2_pre": "#d95926",
             "E0": "#3987e5", "E1": "#d95926", "E0m": "#199e70", "E1m": "#c98500"}


def model_section(w) -> None:
    f = OUT / "model.json"
    if not f.exists():
        return
    from src.scripts.data import folklore_verify as V

    o = json.load(open(f))
    for k, c in MODEL_COL.items():
        V.SER_COL[k] = c
        V.SER_NAME[k] = MODEL_NAMES[k]
    pc_ = lambda v: f"{v * 100:.0f}%"  # noqa: E731

    def best(name, kind, min_n=300):
        rows = [r for r in o["ev"].get(name, {}).get(kind, []) if r["n"] >= min_n and r["th"] >= 1.0]
        return max(rows, key=lambda r: r["roi"]) if rows else None

    def at(name, kind, th):
        for r in o["ev"].get(name, {}).get(kind, []):
            if abs(r["th"] - th) < 1e-9:
                return r
        return None

    def fmt(r):
        return "—" if r is None else f"{r['roi'] * 100:.0f}%±{r['se'] * 100:.0f}（{r['n']:,} 点）"

    def cm(key, kind):
        c = o["cmp"].get(key, {}).get(kind)
        if not c:
            return "—"
        z = c["d"] / c["se"] if c["se"] > 0 else 0
        return f"{c['d']:+.4f}（z={z:+.1f}）"

    w("<h2 id='model'>10. 課題への対応: 市場の見込みを土台にしたモデルで通説を束ね、期待値で買う</h2>")
    w("<p>第 9 節の得点の課題を 1 つずつ直した。表の各行が課題と対応で、結果は 2023〜2025 年（学習に使っていない期間）。"
      "期待値 = モデルの確率 × オッズ。期待値が閾値以上の馬だけを買う（1 レースで 0 頭も複数頭もある）。回収率の ± はレースで束ねた標準誤差。</p>")
    w("<div class='wrap'><table><thead><tr><th>#</th><th>課題</th><th>対応</th><th>結果</th></tr></thead><tbody>")
    rows = [
        ("1", "得点の 1 位を買うと人気馬になり、回収率を目標にしていない",
         "通説で直した確率 × オッズ = 期待値で買う（単勝・複勝それぞれ）",
         f"単勝 期待値 1.0 以上: 通説なし {fmt(at('M0', 'win', 1.0))} → 通説あり {fmt(at('M2', 'win', 1.0))}。複勝 1.0 以上: {fmt(at('M0', 'plc', 1.0))} → {fmt(at('M2', 'plc', 1.0))}"),
        ("2", "格は確かさで、効き目の大きさではない",
         "回帰の係数（効き目）で重み付け。モデルなしの版として、人気をそろえた差を経験ベイズで縮めた重みも作った",
         f"効き目の重み（オッズを使わない）の複勝 期待値 1.0 以上: {fmt(at('EB2', 'plc', 1.0))}"),
        ("3", "重なる通説を二重に数える",
         f"全通説を 1 つの回帰に入れて同時推定（L2 の縮小、強さ λ は学習期間の最後の 2 年で選んだ: 単勝 {o['lam']['M2'][0]}・3 着内 {o['lam']['M2'][1]}）",
         f"対数損失（小さいほど良い）の改善 市場の補正だけ → + 通説: 1 着 {cm('M2-M0', 'win')}／レース、3 着内 {cm('M2-M0', 'top3')}／頭"),
        ("4", "目的変数（3 着内）と単勝がずれる",
         "単勝は 1 着の条件付きロジット（レース内で確率の合計 1）、複勝は 3 着内のロジスティック回帰。複勝の土台には確定の複勝オッズも入れた",
         "単勝・複勝で別々の確率と期待値（下の図・表）"),
        ("5", "確定オッズ（発走直前の情報）を使っている",
         f"人気・オッズを使う通説 {len(o['odds_hids'])} 型を外した版（M2）と、レース前の情報だけの想定オッズを土台にした版（P、2022 年で学習）を作った",
         f"想定オッズ + 通説の対数損失の改善: 1 着 {cm('P2-P0', 'win')}、3 着内 {cm('P2-P0', 'top3')}。前日判断の単勝 期待値 1.0 以上 {fmt(at('P2_pre', 'win', 1.0))}・複勝 {fmt(at('P2_pre', 'plc', 1.0))}"),
        ("6", "今のモデルに足して上積みがあるか",
         "本番系モデル（v6 の OOF の p_win・p_top3、2022 年で重み、2023〜2024 年で評価）に通説の得点（2016〜2021 年の係数）を足した",
         f"対数損失の改善 本番系 → + 通説: 1 着 {cm('E1-E0', 'win')}、3 着内 {cm('E1-E0', 'top3')}。本番系 + 市場 → + 通説: 1 着 {cm('E1m-E0m', 'win')}、3 着内 {cm('E1m-E0m', 'top3')}"),
        ("7", "データの穴（追い切りの詳細は 2016〜2019 年、併せ・1 週前は無い）",
         "この PC には 2020 年以降の追い切りの詳細が無い（horse_training は 2026 年の 2 歳馬だけ）。前日オッズの記録も無い（出馬表のオッズの列は 0）ので、想定オッズで代えた",
         "未対応（データの取得が必要。下の『残り』）"),
        ("—", "（作り直しで見つかった不具合）取消馬の 0 倍のオッズでレース全体の市場確率が 0 になっていた（1,873 レース、5.5%）",
         "オッズが正の馬だけで正規化し直し、第 1〜9 節の検定・集計をすべて作り直した", "この章の数字はすべて直した後のもの"),
    ]
    for r in rows:
        w("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    w("</tbody></table></div>")

    def z(key, kind):
        c = o["cmp"][key][kind]
        return c["d"] / c["se"]

    m2w, m2p = at("M2", "win", 1.0), at("M2", "plc", 1.0)
    e0w, e0p = at("E0", "win", 1.5), at("E0", "plc", 1.5)
    e1w, e1p = at("E1", "win", 1.5), at("E1", "plc", 1.5)
    w("<div class='note'><b>結論.</b> "
      f"(1) 通説は市場の確率を<b>統計的には確かに直す</b>（オッズを使わない通説を足すと、1 着の対数損失が z={z('M2-M0', 'win'):.1f}、3 着内が z={z('M2-M0', 'top3'):.1f}）。"
      "ただし改善はレースあたり 0.005 nat 程度で小さく、"
      f"期待値で買っても回収率は 100% を超えない（単勝 期待値 1.0 以上 {fmt(m2w)}、複勝 {fmt(m2p)}）。"
      "(2) 複勝は期待値の閾値を上げるほど回収率が下がる。単勝・複勝のオッズを曲線（2〜3 次と単複の食い違い）で補正しても複勝の売り場の情報を使い切れず、期待値が高く見える馬ほど見込み違い（勝者の呪い）になる。"
      f"(3) 前日の想定オッズで判断すると（P、払戻は確定オッズ）、通説を足しても回収率は約 80% のまま（単勝 {fmt(at('P2_pre', 'win', 0.9))}・複勝 {fmt(at('P2_pre', 'plc', 1.0))}）。"
      f"(4) 本番系モデル（v6 の OOF）は、確定オッズで期待値 1.5 以上を買うと単勝 {fmt(e0w)}・複勝 {fmt(e0p)} だが、v6 の OOF は入力に軽い先読み（EASE2・ラップ予想）と選択バイアスを含みうるので、この数字は上限で、人気以上の情報を持つ証拠とは言えない（第 13 節で自作ベースを年ごとに抜く形で作ると、市場より弱く 100% を下回った。差の原因は切り分けていない）。"
      f"通説を足すと 1 着の対数損失はわずかに良くなる（z={z('E1-E0', 'win'):.1f}）が、回収率は単勝 {fmt(e1w)}・複勝 {fmt(e1p)} でほぼ変わらない。"
      "通説だけで控除率を越えるのは難しく、使い道は本番系モデルの特徴量・解釈の補助と、第 11 節のような『市場が織り込み切れていない所』の候補探し。</div>")
    # 図
    def series(names, kind):
        out = []
        for nm in names:
            rr = [r for r in o["ev"].get(nm, {}).get(kind, []) if r["th"] >= 0.8 and r["n"] >= 100]
            if rr:
                out.append((nm, [(r["th"], r["roi"], f"期待値 {r['th']:.2f} 以上: {r['n']:,} 点・的中 {r['hit'] * 100:.1f}% → 回収率 {r['roi'] * 100:.0f}%±{r['se'] * 100:.0f}") for r in rr]))
        return out

    w("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:18px'>")
    for title, names, kind in (("単勝（確定オッズで期待値）", ["M0", "M2", "M1"], "win"), ("複勝（確定オッズで期待値）", ["M0", "M2", "M1", "EB2"], "plc"),
                               ("単勝（前日判断: 想定オッズ）", ["P0_pre", "P2_pre"], "win"), ("複勝（前日判断: 想定オッズ）", ["P0_pre", "P2_pre"], "plc"),
                               ("単勝（本番系モデル + 通説、2023〜2024）", ["E0", "E1", "E0m", "E1m"], "win"), ("複勝（本番系モデル + 通説、2023〜2024）", ["E0", "E1", "E0m", "E1m"], "plc")):
        ss = series(names, kind)
        if not ss:
            continue
        w(f"<figure style='margin:0'><figcaption><b>{title}</b>: 期待値の閾値と回収率</figcaption>")
        w(V._svg_lines(ss, (0.8, 2.0), (0.3, 1.4), "期待値の閾値（以上を買う）", "回収率", lambda v: f"{v:.1f}", pc_, refs=[(1.0, "100%")]) + "</figure>")
    w("</div><p class='muted'>点にカーソルを置くと点数・的中率・回収率が出る。閾値が高いほど買う点数が減り、誤差が大きくなる。</p>")
    # 表
    for name in ("M0", "M2", "M1", "EB2", "P0_pre", "P2_pre", "P0", "P2", "E0", "E1", "E0m", "E1m"):
        if name not in o["ev"]:
            continue
        w(f"<details><summary>{MODEL_NAMES[name]}（{name}）: 期待値の閾値ごとの成績</summary><div class='wrap'><table><thead><tr><th>券種</th><th>期待値 ≥</th><th>点数</th><th>レース</th><th>的中率</th><th>回収率 ±SE</th><th>平均人気</th></tr></thead><tbody>")
        for kind, lab in (("win", "単勝"), ("plc", "複勝")):
            for r in o["ev"][name].get(kind, []):
                w(f"<tr><td>{lab}</td><td class='num'>{r['th']:.2f}</td><td class='num'>{r['n']:,}</td><td class='num'>{r['races']:,}</td><td class='num'>{r['hit'] * 100:.1f}%</td>"
                  f"<td class='num'>{r['roi'] * 100:.1f}% ±{r['se'] * 100:.1f}</td><td class='num'>{r['pop']:.1f}</td></tr>")
        w("</tbody></table></div></details>")
    # 係数（効き目）の上位
    co = o["coef"].get("M2", {})
    if co:
        for kind, lab in (("win", "1 着"), ("top3", "3 着内")):
            items = sorted(co[kind].items(), key=lambda kv: -abs(kv[1]))[:20]
            w(f"<details><summary>効き目の大きい通説（市場 + オッズを使わない通説、{lab}の係数の絶対値の上位 20）</summary><div class='wrap'><table><thead><tr><th>ID</th><th>係数（対数オッズ）</th><th>倍率</th></tr></thead><tbody>")
            for h, b in items:
                w(f"<tr><td class='id'>{h}</td><td class='num'>{b:+.3f}</td><td class='num'>×{np.exp(b):.2f}</td></tr>")
            w("</tbody></table></div></details>")
    w(f"<p class='muted'>点数の目安: 2023〜2025 年は約 {o.get('pre_n', {}).get('test', 0):,} レース（想定オッズの版）。本番系モデルの版は 2023〜2024 年の {o.get('own_n', {}).get('test', 0):,} レース。"
      "コード: <code>python -m src.scripts.data.folklore_model flags</code> → <code>fit</code>（→ <code>folklore_verify report</code>）。</p>")


LAPS_NAMES = {"B": "ベースだけ（本番系 v6）", "BE": "ベース + エッジ（全部）", "BEok": "ベース + エッジ（通説の向きと合うものだけ）"}
LAPS_COL = {"B": "#3987e5", "BE": "#d95926", "BEok": "#199e70"}


def laps_section(w) -> None:
    f = OUT / "model.json"
    if not f.exists():
        return
    o = json.load(open(f)).get("laps")
    if not o:
        return
    from src.scripts.data import folklore_verify as V

    for k in LAPS_NAMES:
        V.SER_COL[k] = LAPS_COL[k]
        V.SER_NAME[k] = LAPS_NAMES[k]
    pc_ = lambda v: f"{v * 100:.0f}%"  # noqa: E731

    def cell(r):
        if not r:
            return "<td class='num' colspan='4'>—</td>"
        return (f"<td class='num'>{r['n']:,}</td><td class='num'>{r['hit'] * 100:.1f}%</td><td class='num'>{r['roi'] * 100:.1f}% ±{r['se'] * 100:.1f}</td>"
                f"<td class='num'>{r['pop']:.1f}</td>")

    def zc(key, kind):
        c = o.get(key, {}).get(kind)
        return "—" if not c else f"{c['d']:+.4f}（z={c['d'] / c['se']:+.1f}）"

    w("<h2 id='laps'>12. 記事の構成（ベース + エッジ + 期待値 1.2）を手元のデータで再現</h2>")
    w("<p>参考: ユリス【UMASTカルテ】『新生【LAPS】始動』（note、2026-11-06、<a href='https://note.com/mgtkeiba/n/n1512b01f6755'>リンク</a>）。"
      "記事の構成は (1) <b>ベース</b> = オッズ・人気を使わず 3 着内で学習した実力、(2) <b>エッジ</b> = 『人気より走るか』を物差しに、条件ごとのノウハウを<b>1 つずつ単独で確かめてから</b>組み合わせる、"
      "(3) <b>確率 × オッズ = 期待値が 1.2 以上</b>に絞って買う。記事では検証期間で単勝 100% 超・複勝 90% 超とあるが、期間・件数・エッジの中身は公開されていない。</p>")
    w("<p>手元での対応:</p><ul>"
      "<li><b>ベース</b>: 本番系モデル v6 の p_win・p_top3（土台の特徴量にオッズ・人気の列は無い）。OOF は 2022〜2024 年。<b>検証年より前だけで学習した expanding だが、入力の EASE2・ラップ予想は leave-one-year-out で『軽い先読みがある』（v6 の章）、構成の選択も 2022〜2024 年の OOF 上。3 着内の AUC が確定オッズの市場に近く、この節の v6 の回収率は上限として読む（自作ベースで作り直した結果は第 13 節）。</b></li>"
      f"<li><b>エッジ</b>: 発見期間（2016〜2022 年）だけで、人気をそろえた差が単独で有意（BH の q &lt; 0.10）になった条件。通説の型（オッズを使わないもの）{o['n_edge_hyp']} 個と、血統 × 条件替わりの (a)（条件の中でその血統）{o['n_edge_blood']} 個。"
      "重みは差を経験ベイズで縮めた値。記事の『腹に落ちるものだけ』の代わりに、通説の向きと合うものだけの版（"
      f"{o['n_edge_hyp_ok']} 個）も作った。</li>"
      "<li><b>組み合わせ</b>: 2022 年でベースとエッジの重みだけを当てはめ（単勝は 1 着の条件付きロジット、複勝は 3 着内のロジスティック）、2023〜2024 年で評価"
      f"（{o['races']:,} レース）。</li>"
      "<li><b>期待値 1.2</b>: 記事の閾値をそのまま事前に決めた買い方として使う（評価期間を見て選んでいない）。単勝 = 1 着の確率 × 確定単勝オッズ、複勝 = 3 着内の確率 × 確定複勝オッズの下限。"
      "前日に買う場合として、想定オッズ（odds_pred）で期待値を判断し、払戻は確定オッズで計算した版も出した。</li></ul>")
    w("<div class='wrap'><table><thead><tr><th rowspan='2'>買い方（2023〜2024 年）</th>" + "".join(f"<th colspan='4'>{LAPS_NAMES[k]}</th>" for k in LAPS_NAMES) + "</tr><tr>"
      + "<th>点数</th><th>的中率</th><th>回収率 ±SE</th><th>平均人気</th>" * 3 + "</tr></thead><tbody>")
    for sname in o["strat"]["B"]:
        w(f"<tr><td>{sname}</td>" + "".join(cell(o["strat"][k].get(sname)) for k in LAPS_NAMES) + "</tr>")
    w("</tbody></table></div>")
    w(f"<p>対数損失の改善（ベースだけ → + エッジ、負なら改善）: 全部 1 着 {zc('cmp_BE', 'win')}／レース・3 着内 {zc('cmp_BE', 'top3')}／頭。"
      f"通説の向きと合うものだけ 1 着 {zc('cmp_BEok', 'win')}・3 着内 {zc('cmp_BEok', 'top3')}。</p>")
    def f_(r):
        return "—" if not r else f"{r['roi'] * 100:.0f}%±{r['se'] * 100:.0f}（{r['n']:,} 点）"

    zs = [((r["roi"] - 1) / r["se"], k, sn) for k in LAPS_NAMES for sn, r in o["strat"][k].items() if r and r["se"] > 0]
    zmax = max(zs)
    S1, S2, S3 = "期待値 1.2 以上（単勝・確定オッズ）", "期待値 1.2 以上（単勝・前日の想定オッズで判断）", "期待値 1.2 以上（複勝・確定オッズ）"
    w("<div class='note'><b>結論.</b> "
      f"(1) 指数 1 位を全レース買うと、平均 {o['strat']['B']['指数 1 位・全レース（単勝）']['pop']:.1f} 番人気で単勝 {f_(o['strat']['B']['指数 1 位・全レース（単勝）'])}（記事のベースだけの 82.6% と近い水準）。"
      f"(2) 期待値 1.2 以上: 確定オッズの単勝は ベース {f_(o['strat']['B'][S1])}・+エッジ {f_(o['strat']['BE'][S1])}、複勝は ベース {f_(o['strat']['B'][S3])}・+エッジ {f_(o['strat']['BE'][S3])}。"
      f"前日の想定オッズで判断した単勝は ベース {f_(o['strat']['B'][S2])}・+エッジ {f_(o['strat']['BE'][S2])}。"
      f"(3) 100% との差が最も大きいのは『{LAPS_NAMES[zmax[1]]}・{zmax[2]}』で標準誤差の {zmax[0]:.1f} 倍。どの買い方も 2 倍に届かず、100% を超えるとは言えない（評価は 2 年・約 6,100 レース）。"
      "(4) エッジは予測を統計的に良くする（上の対数損失）が、回収率の差は誤差の範囲。記事のエッジの中身（非公開）が手元の通説・血統・ラップより強い可能性はある。"
      "(5) <b>ただしこの節のベース（v6 の OOF）は入力に軽い先読みと選択バイアスを含みうるので、前日の想定オッズ判断の成績は上限</b>。自作ベース（年ごとに抜く形）で作り直すと 100% を下回った（第 13 節）。</div>")
    w("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:18px'>")
    for kind, title in (("win", "単勝"), ("plc", "複勝")):
        ss = []
        for k in LAPS_NAMES:
            rr = [r for r in o["ev"][k][kind] if r["th"] >= 0.8 and r["n"] >= 100]
            ss.append((k, [(r["th"], r["roi"], f"期待値 {r['th']:.2f} 以上: {r['n']:,} 点・的中 {r['hit'] * 100:.1f}% → 回収率 {r['roi'] * 100:.0f}%±{r['se'] * 100:.0f}") for r in rr]))
        w(f"<figure style='margin:0'><figcaption><b>{title}（確定オッズで期待値、2023〜2024）</b>: 期待値の閾値と回収率</figcaption>")
        w(V._svg_lines(ss, (0.8, 2.0), (0.3, 1.4), "期待値の閾値（以上を買う）", "回収率", lambda v: f"{v:.1f}", pc_, refs=[(1.0, "100%")]) + "</figure>")
    w("</div>")
    w("<details><summary>使ったエッジ（通説の型）</summary><div class='wrap'><table><thead><tr><th>ID</th><th>重み（pt）</th><th>通説の向きと</th></tr></thead><tbody>")
    for h, (wt, ag) in sorted(o["edge_hyp"].items(), key=lambda kv: -abs(kv[1][0])):
        w(f"<tr><td class='id'>{h}</td><td class='num'>{wt * 100:+.2f}</td><td>{'合う' if ag else '逆'}</td></tr>")
    w("</tbody></table></div></details>")
    w("<details><summary>使ったエッジ（血統 × 条件替わり）</summary><div class='wrap'><table><thead><tr><th>粒度</th><th>条件</th><th>血統</th><th>重み（pt）</th></tr></thead><tbody>")
    from src.scripts.data.folklore_sire import LEVELS as _LV
    for lv, cond, key, wt in sorted(o["edge_blood"], key=lambda r: -abs(r[3])):
        w(f"<tr><td>{_LV.get(lv, lv)}</td><td>{cond}</td><td>{key}</td><td class='num'>{wt * 100:+.2f}</td></tr>")
    w("</tbody></table></div></details>")
    w("<p class='muted'>コード: <code>python -m src.scripts.data.folklore_model laps</code>（folklore_sire run の後）。</p>")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    {"flags": flags, "fit": fit, "laps": laps}[cmd]()
