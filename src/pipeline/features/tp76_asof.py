"""tp-76: 学習前スクリーニング用の候補特徴量（A〜G）を、すべて as-of（その走より前の結果だけ）で作る純粋関数。

入力は 1 出走 1 行の表（tp72 の mining.parquet と同じ列名）。市場オッズ・人気は使わない。
- A 同日バイアス: 同じ日・同じ場・同じ面の、自分より前のレースの結果から作る
- B 潜在能力: 年齢×性別の曲線と種牡馬の事前分布を持つ局所水準モデル（カルマン）
- C 対戦関係: 着順から更新する Elo 型の強さ
- D 条件別適性: B の予測誤差を、条件ごとに馬自身で集計して種牡馬の事前分布へ縮約（血統は不完全な 10 世代表に基づく）
- E 上がりの形 / F 負荷 / G メンバー構成
母集団の定数（年齢曲線・ノイズ）は FIT_END までの走だけから作る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

FIT_END = pd.Timestamp("2021-12-31")
KEY = ["race_id", "horse_id"]

PARAMS = {
    "front_k": 20.0,          # A: 日内バイアスの縮約（レース重みの和に足す）
    "habit_hl": 3.0, "habit_k": 1.0,
    "kal_q_ratio": 0.08, "kal_sire_k": 30.0,
    "elo_k0": 0.15, "elo_k1": 0.85, "elo_c": 3.0,
    "apt_kh": 3.0, "apt_ks": 30.0,
    "ag_hl": 3.0, "ag_k": 1.0,
    "front_thr": 0.35,
}


# ---------------------------------------------------------------- 共通部品
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """型の整理・欠損の正規化・並び順（日付, ラウンド, 場, race_id, 馬番）。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "round", "venue_code", "race_id", "horse_number"], kind="mergesort").reset_index(drop=True)
    d["fin_ok"] = (d["finish_position"] > 0) & d["finish_position"].notna()
    d["rank_"] = d["finish_position"].where(d["fin_ok"])
    d["pos1_frac"] = d["pos1_frac"].where((d["pos1_frac"] >= 0) & (d["pos1_frac"] <= 1.0))
    d["posL_frac"] = d["posL_frac"].where((d["posL_frac"] >= 0) & (d["posL_frac"] <= 1.0))
    d["last_3f"] = d["last_3f"].where((d["last_3f"] > 30) & (d["last_3f"] < 50))
    d["time_sec"] = d["time_sec"].where(d["time_sec"] > 30)
    d["age_c"] = d["age"].clip(2, 7)
    d["sex_g"] = d["sex"].where(d["sex"].isin(["牡", "牝", "セ"]), "牡").replace({"セ": "牡"})
    d["n_fin"] = d.groupby("race_id")["fin_ok"].transform("sum")
    d["z_fin"] = np.where(d["fin_ok"] & (d["n_fin"] >= 5), ndtri((d["n_fin"] - d["rank_"] + 0.5) / d["n_fin"]), np.nan)
    d["agari_pct"] = np.where(d["agari_rank"] > 0, (d["agari_rank"] - 1) / (d["field_size"] - 1).clip(lower=1), np.nan)
    d["bracket_frac"] = (d["bracket_number"] - 0.5) / 8.0
    d["hn_frac"] = (d["horse_number"].astype("float") - 0.5) / d["field_size"].clip(lower=1)
    return d


def asof_group_sum(df: pd.DataFrame, keys: list[str], vals: list[str], date: str = "date") -> pd.DataFrame:
    """keys が同じ行のうち、date が自分より厳密に前の行の vals の合計と件数（列 `__n`）。NaN は 0。"""
    base = df[keys + [date]].copy()
    for v in vals:
        base[v] = df[v].fillna(0.0).to_numpy()
        base[v + "__m"] = df[v].notna().astype(float).to_numpy()
    g = base.groupby(keys + [date], sort=True).sum()
    lvl = list(range(len(keys)))
    cs = g.groupby(level=lvl).cumsum() - g
    cs = cs.reset_index()
    out = df[keys + [date]].merge(cs, on=keys + [date], how="left")
    out.index = df.index
    return out.drop(columns=keys + [date])


def ewm_asof(horse: np.ndarray, x: np.ndarray, hl: float, prior: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """馬ごと（行は馬・日付順）に、直前までの指数加重平均を prior へ縮約した値と、有効件数。"""
    lam = 0.5 ** (1.0 / hl)
    n = len(x)
    out = np.empty(n)
    eff = np.empty(n)
    num = den = 0.0
    prev = None
    for i in range(n):
        if horse[i] != prev:
            num = den = 0.0
            prev = horse[i]
        out[i] = (num + k * prior) / (den + k)
        eff[i] = den
        xi = x[i]
        if xi == xi:
            num = num * lam + xi
            den = den * lam + 1.0
    return out, eff


def _sorted_by_horse(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values(["horse_id", "date", "round", "race_id"], kind="mergesort")


# ---------------------------------------------------------------- A 同日バイアス
def _race_corr(d: pd.DataFrame, col: str) -> pd.Series:
    """レース内の Spearman 相関（col と着順）。8 頭未満は NaN。"""
    ok = d["fin_ok"] & d[col].notna()
    x = d[col].where(ok)
    r1 = x.groupby(d["race_id"]).rank(pct=True)
    r2 = d["rank_"].where(ok).groupby(d["race_id"]).rank(pct=True)
    t = pd.DataFrame({"race_id": d["race_id"], "a": r1, "b": r2, "ab": r1 * r2, "aa": r1 * r1, "bb": r2 * r2})
    g = t.groupby("race_id")
    n = g["a"].count()
    ma, mb = g["a"].mean(), g["b"].mean()
    cov = g["ab"].mean() - ma * mb
    va = g["aa"].mean() - ma * ma
    vb = g["bb"].mean() - mb * mb
    c = cov / np.sqrt(va * vb)
    return c.where(n >= 8)


def day_bias(d: pd.DataFrame, p: dict) -> pd.DataFrame:
    """レースごとの『前残り』『内有利』の相関を、同日・同場・同面の自分より前のレース（ラウンドが小さい）で集計する。"""
    rc = d.groupby("race_id").agg(date=("date", "first"), round=("round", "first"), venue_code=("venue_code", "first"), surface=("surface", "first"), n=("fin_ok", "sum")).reset_index()
    rc["c_front"] = rc["race_id"].map(_race_corr(d, "pos1_frac"))
    rc["c_inner"] = rc["race_id"].map(_race_corr(d, "bracket_frac"))
    rc["w"] = (rc["n"] - 1).clip(lower=0).where(rc["c_front"].notna(), 0.0)
    rc["wcf"] = (rc["w"] * rc["c_front"]).fillna(0.0)
    rc["wci"] = (rc["w"] * rc["c_inner"]).fillna(0.0)
    rc["one"] = rc["c_front"].notna().astype(float)
    rc = rc.sort_values(["date", "venue_code", "surface", "round", "race_id"], kind="mergesort").reset_index(drop=True)
    grp = rc.groupby(["date", "venue_code", "surface"], sort=False)
    for c in ("w", "wcf", "wci", "one"):
        rc["cs_" + c] = grp[c].cumsum() - rc[c]
    k = p["front_k"]
    rc["day_front"] = rc["cs_wcf"] / (rc["cs_w"] + k)
    rc["day_inner"] = rc["cs_wci"] / (rc["cs_w"] + k)
    rc["day_n"] = rc["cs_one"]
    return rc[["race_id", "day_front", "day_inner", "day_n"]]


def habit_position(d: pd.DataFrame, p: dict) -> pd.Series:
    """馬の普段の初角位置（直前までの指数平滑を 0.5 へ縮約）。"""
    s = _sorted_by_horse(d)
    out, _ = ewm_asof(s["horse_id"].to_numpy(), s["pos1_frac"].to_numpy(float), p["habit_hl"], 0.5, p["habit_k"])
    return pd.Series(out, index=s.index).reindex(d.index)


# ---------------------------------------------------------------- B カルマン
def fit_constants(d: pd.DataFrame, ycol: str = "z_fin") -> dict:
    """FIT_END までの走だけから、年齢×性別の平均・ノイズ・事前分散を求める。"""
    f = d[d["date"] <= FIT_END]
    mu = f.groupby(["sex_g", "age_c"])[ycol].mean().to_dict()
    var = float(f[ycol].var())
    s = _sorted_by_horse(f)
    prev = s.groupby("horse_id")[ycol].shift(1)
    m = prev.notna() & s[ycol].notna()
    rho = float(np.corrcoef(prev[m], s[ycol][m])[0, 1]) if m.sum() > 100 else 0.3
    rho = min(max(rho, 0.05), 0.8)
    return {"mu": mu, "var": var, "rho1": rho, "sigma2": var * (1 - rho), "tau2": var * rho}


def kalman_ability(d: pd.DataFrame, ycol: str, consts: dict, p: dict) -> pd.DataFrame:
    """局所水準モデル。y_t = mu(年齢,性別) + a_t + e。a は走ごとに分散 q だけ動き、休養が長いほど大きく動く。
    初出走の事前平均は、同じ種牡馬の自分より前の産駒の『初出走時の平均 − mu』（厳密に前の日付、縮約つき）。"""
    s = _sorted_by_horse(d).copy()
    mu = np.array([consts["mu"].get((a, b), 0.0) for a, b in zip(s["sex_g"], s["age_c"])])
    s["_mu"] = mu
    first = s.groupby("horse_id").cumcount() == 0
    s["_first_res"] = np.where(first & s[ycol].notna(), s[ycol] - mu, np.nan)
    sp = asof_group_sum(s, ["sire"], ["_first_res"])
    sire_prior = (sp["_first_res"] / (sp["_first_res__m"] + p["kal_sire_k"])).to_numpy()
    tau2, sig2 = consts["tau2"], consts["sigma2"]
    q = tau2 * p["kal_q_ratio"]
    horse = s["horse_id"].to_numpy()
    y = s[ycol].to_numpy(float)
    lay = s["lay_days"].fillna(30.0).clip(0, 270).to_numpy(float)
    n = len(s)
    m_pred = np.empty(n)
    P_pred = np.empty(n)
    innov = np.full(n, np.nan)
    a = Pc = 0.0
    prev = None
    for i in range(n):
        if horse[i] != prev:
            prev = horse[i]
            a = sire_prior[i] if sire_prior[i] == sire_prior[i] else 0.0
            Pc = tau2
        else:
            Pc = Pc + q * (1.0 + lay[i] / 90.0)
        m_pred[i] = mu[i] + a
        P_pred[i] = Pc
        if y[i] == y[i]:
            e = y[i] - mu[i] - a
            innov[i] = e
            K = Pc / (Pc + sig2)
            a = a + K * e
            Pc = (1 - K) * Pc
    out = pd.DataFrame({"m": m_pred, "sd": np.sqrt(P_pred), "innov": innov}, index=s.index)
    return out.reindex(d.index)


# ---------------------------------------------------------------- C Elo
def elo_ratings(d: pd.DataFrame, p: dict) -> pd.DataFrame:
    """レースを時系列に処理し、着順（負かした割合）から馬の強さを更新する。特徴量は更新前の値。"""
    rating: dict[str, float] = {}
    cnt: dict[str, int] = {}
    r_pre = np.zeros(len(d))
    n_pre = np.zeros(len(d))
    rel = np.zeros(len(d))
    order = d.groupby("race_id", sort=False).indices
    horses = d["horse_id"].to_numpy()
    rank = d["rank_"].to_numpy(float)
    for _, idx in order.items():
        h = horses[idx]
        r = np.array([rating.get(x, 0.0) for x in h])
        c = np.array([cnt.get(x, 0) for x in h], float)
        r_pre[idx] = r
        n_pre[idx] = c
        rel[idx] = r - r.mean()
        rk = rank[idx]
        ok = ~np.isnan(rk)
        m = int(ok.sum())
        if m < 5:
            continue
        rr, kk, ci = r[ok], rk[ok], c[ok]
        diff = rr[:, None] - rr[None, :]
        expct = (1 / (1 + np.exp(-diff)))
        np.fill_diagonal(expct, 0)
        e = expct.sum(1) / (m - 1)
        act = (m - kk) / (m - 1)
        K = p["elo_k0"] + p["elo_k1"] / (1 + ci / p["elo_c"])
        newr = rr + K * (act - e)
        for hh, v in zip(h[ok], newr):
            rating[hh] = float(v)
            cnt[hh] = cnt.get(hh, 0) + 1
    return pd.DataFrame({"elo": r_pre, "elo_rel": rel, "elo_unc": 1.0 / np.sqrt(1.0 + n_pre)}, index=d.index)


# ---------------------------------------------------------------- D 条件別適性
def conditional_aptitude(d: pd.DataFrame, innov: pd.Series, tag: str, p: dict) -> pd.DataFrame:
    """馬自身の『予測誤差』を条件ごとに集計し、種牡馬の同条件の産駒の平均（事前分布）へ縮約する。"""
    t = d[["horse_id", "sire", "date", "dist_band", "venue_code", "surface", "track_condition", "cush_bin"]].copy()
    t["_e"] = innov.to_numpy()
    t["vs"] = t["venue_code"].astype(str) + t["surface"].astype(str)
    axes = {"dist": "dist_band", "venue": "vs", "going": "track_condition", "cush": "cush_bin"}
    out = {}
    kh, ks = p["apt_kh"], p["apt_ks"]
    for nm, col in axes.items():
        t["_ax"] = t[col].astype(str).where(t[col].notna(), "NA")
        h = asof_group_sum(t, ["horse_id", "_ax"], ["_e"])
        s = asof_group_sum(t, ["sire", "_ax"], ["_e"])
        prior = (s["_e"] / (s["_e__m"] + ks)).fillna(0.0)
        post = (h["_e"] + kh * prior) / (h["_e__m"] + kh)
        raw = h["_e"] / (h["_e__m"] + kh)
        na = (t["_ax"] == "NA").to_numpy()
        out[f"apt_{tag}_{nm}"] = post.where(~na, 0.0)
        out[f"aptraw_{tag}_{nm}"] = raw.where(~na, 0.0)
    o = pd.DataFrame(out, index=d.index)
    o[f"apt_{tag}_all"] = o[[c for c in o.columns if c.startswith(f"apt_{tag}_") and not c.endswith("_all")]].sum(axis=1)
    o[f"aptraw_{tag}_all"] = o[[c for c in o.columns if c.startswith(f"aptraw_{tag}_") and not c.endswith("_all")]].sum(axis=1)
    return o


# ---------------------------------------------------------------- E 上がり
def agari_features(d: pd.DataFrame, p: dict) -> pd.DataFrame:
    rm = d.groupby("race_id").agg(date=("date", "first"), venue_code=("venue_code", "first"), surface=("surface", "first"), distance=("distance", "first"),
                                 med=("last_3f", "median")).reset_index()
    rm = rm.sort_values(["date", "race_id"], kind="mergesort").reset_index(drop=True)
    rm["_m"] = rm["med"]
    base = asof_group_sum(rm, ["venue_code", "surface", "distance"], ["_m"])
    rm["base"] = np.where(base["_m__m"] >= 5, base["_m"] / base["_m__m"].replace(0, np.nan), np.nan)
    rm["dev"] = rm["med"] - rm["base"]
    day = rm.groupby(["date", "venue_code", "surface"])["dev"].transform("mean")
    rm["day_dev"] = day
    d = d.copy()
    d["_base"] = d["race_id"].map(rm.set_index("race_id")["base"])
    d["_day_dev"] = d["race_id"].map(rm.set_index("race_id")["day_dev"])
    d["_med"] = d["race_id"].map(rm.set_index("race_id")["med"])
    d["ag_abs"] = d["last_3f"] - d["_base"]
    d["ag_day"] = d["ag_abs"] - d["_day_dev"]
    d["ag_rel"] = -(d["last_3f"] - d["_med"])
    d["unlucky"] = (d["fin_pct"].where(d["fin_ok"]) - d["agari_pct"]) * d["posL_frac"]
    s = _sorted_by_horse(d)
    h = s["horse_id"].to_numpy()
    out = pd.DataFrame(index=s.index)
    out["E_abs"], _ = ewm_asof(h, s["ag_abs"].to_numpy(float), p["ag_hl"], 0.0, p["ag_k"])
    out["E_day"], _ = ewm_asof(h, s["ag_day"].to_numpy(float), p["ag_hl"], 0.0, p["ag_k"])
    s["_sus"] = s["ag_rel"].where(s["posL_frac"] <= 0.4)
    s["_acc"] = s["ag_rel"].where(s["posL_frac"] >= 0.6)
    for nm, col in (("E_sustain", "_sus"), ("E_accel", "_acc")):
        cs = s[col].fillna(0.0).groupby(s["horse_id"]).cumsum() - s[col].fillna(0.0)
        cn = s[col].notna().astype(float).groupby(s["horse_id"]).cumsum() - s[col].notna().astype(float)
        out[nm] = cs / (cn + 2.0)
    g = s.groupby("horse_id")
    out["E_unlucky_prev"] = g["unlucky"].shift(1)
    out["E_unlucky_m3"] = g["unlucky"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    out["E_std5"] = g["agari_pct"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
    return out.reindex(d.index)


# ---------------------------------------------------------------- F 負荷・疲労
def load_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    win_t = d.groupby("race_id")["time_sec"].transform("min")
    loss = (d["time_sec"] - win_t).clip(0, 6).fillna(1.0)
    d["_burden"] = (d["distance"] / 1600.0) * (1.0 + loss / 3.0)
    s = _sorted_by_horse(d)
    day = s["date"].to_numpy("datetime64[D]").astype("int64")
    dist = s["distance"].fillna(0).to_numpy(float)
    bur = s["_burden"].fillna(0).to_numpy(float)
    horse = s["horse_id"].to_numpy()
    n = len(s)
    n30 = np.zeros(n); n90 = np.zeros(n); d90 = np.zeros(n); b90 = np.zeros(n); since = np.full(n, np.nan); tataki = np.zeros(n)
    bounds = np.flatnonzero(np.r_[True, horse[1:] != horse[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        dd = day[a:b]
        cd = np.r_[0.0, np.cumsum(dist[a:b])]
        cb = np.r_[0.0, np.cumsum(bur[a:b])]
        hi = np.searchsorted(dd, dd, side="left")        # 厳密に前
        lo30 = np.searchsorted(dd, dd - 30, side="left")
        lo90 = np.searchsorted(dd, dd - 90, side="left")
        n30[a:b] = hi - lo30
        n90[a:b] = hi - lo90
        d90[a:b] = cd[hi] - cd[lo90]
        b90[a:b] = cb[hi] - cb[lo90]
        gap = np.r_[np.nan, np.diff(dd)]
        since[a:b] = gap
        t = 0
        for j in range(b - a):
            if j == 0 or gap[j] >= 60:
                t = 0
            else:
                t += 1
            tataki[a + j] = t
    out = pd.DataFrame({"F_n30": n30, "F_n90": n90, "F_since": np.log1p(since), "F_dist90": d90, "F_burden90": b90, "F_tataki": tataki}, index=s.index)
    return out.reindex(d.index)


# ---------------------------------------------------------------- G メンバー構成
def field_features(d: pd.DataFrame, habit: pd.Series, p: dict) -> pd.DataFrame:
    t = pd.DataFrame({"race_id": d["race_id"], "hn": d["horse_number"].astype(float), "habit": habit, "field": d["field_size"]})
    t["front"] = (t["habit"] < p["front_thr"]).astype(float)
    t["n_front"] = t.groupby("race_id")["front"].transform("sum")
    t["front_rank"] = t.groupby("race_id")["habit"].rank(pct=True)
    s = t.sort_values(["race_id", "hn"], kind="mergesort")
    inner = s.groupby("race_id")["front"].cumsum() - s["front"]
    t["inner_front"] = inner.reindex(t.index)
    pressure = t["n_front"] / t["field"].clip(lower=1)
    out = pd.DataFrame({"G_n_front": t["n_front"], "G_front_rank": t["front_rank"], "G_inner_front": t["inner_front"],
                        "G_fit": pressure * (0.5 - t["habit"])}, index=d.index)
    return out


# ---------------------------------------------------------------- まとめ
def build_features(df: pd.DataFrame, consts: dict | None = None, params: dict | None = None, megu: pd.DataFrame | None = None) -> pd.DataFrame:
    p = {**PARAMS, **(params or {})}
    d = prepare(df)
    consts = consts or fit_constants(d)
    f = pd.DataFrame({"race_id": d["race_id"], "horse_id": d["horse_id"], "date": d["date"]}, index=d.index)
    # A
    db = day_bias(d, p).set_index("race_id")
    habit = habit_position(d, p)
    day_front = d["race_id"].map(db["day_front"])
    day_inner = d["race_id"].map(db["day_inner"])
    f["A_day_n"] = d["race_id"].map(db["day_n"])
    f["A_fit_front"] = day_front * (0.5 - habit)
    f["A_fit_inner"] = day_inner * (0.5 - d["bracket_frac"])
    # B
    kal = kalman_ability(d, "z_fin", consts, p)
    f["B_m"] = kal["m"]
    f["B_sd"] = kal["sd"]
    if megu is not None and "megu_z" in megu.columns:
        dm = d.merge(megu[["race_id", "horse_id", "megu_z"]], on=KEY, how="left")
        dm.index = d.index
        c2 = fit_constants(dm.assign(z_fin=dm["megu_z"]), "z_fin")
        k2 = kalman_ability(dm.assign(z_fin=dm["megu_z"]), "z_fin", c2, p)
        f["B_m_megu"] = k2["m"]
    # C
    f = f.join(elo_ratings(d, p).rename(columns={"elo": "C_elo", "elo_rel": "C_rel", "elo_unc": "C_unc"}))
    # D
    ap = conditional_aptitude(d, kal["innov"], "z", p)
    s = _sorted_by_horse(d)
    mu_ag, _ = ewm_asof(s["horse_id"].to_numpy(), s["agari_pct"].to_numpy(float), 5.0, 0.5, 1.0)
    ag_e = (s["agari_pct"] - pd.Series(mu_ag, index=s.index)).reindex(d.index)
    ap_ag = conditional_aptitude(d, ag_e, "ag", p)
    f["D_dist"] = ap["apt_z_dist"]
    f["D_venue"] = ap["apt_z_venue"]
    f["D_going"] = ap["apt_z_going"]
    f["D_cush"] = ap["apt_z_cush"]
    f["D_all"] = ap["apt_z_all"]
    f["D_raw_all"] = ap["aptraw_z_all"]
    f["D_ag_all"] = ap_ag["apt_ag_all"]
    # E, F, G
    f = f.join(agari_features(d, p))
    f = f.join(load_features(d))
    f = f.join(field_features(d, habit, p))
    return f.reset_index(drop=True)
