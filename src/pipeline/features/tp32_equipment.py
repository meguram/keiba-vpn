"""tp-32: 馬具（ブリンカー）と『初めて・替わり』系の状況特徴（as-of）。

どの特徴量も「当該レースの日より前の走だけ」から作る。入力 ``runs`` は factory_self.load_runs の形（JRA の完走のみ、1 走 = 1 行）。
各群は『その状況かどうかのフラグ』と『その状況での過去の成績』（馬自身の as-of 親和度・件数・初経験フラグ、全馬の as-of 成績率）を持つ。
市場オッズ・人気は使わない。馬具は SmartRC の fullresults（各戦績の当時の blinker。確定した過去の事実）から作る。

群: EQ（馬具）, DIST（距離の延長・短縮の段階）, SURF（芝↔ダートの替わり×距離）, FIRST（初芝・初ダート・初距離帯・初コース・初回り・初重馬場 ほか）,
    REST（休み明け×叩き何戦目）, CLASS（昇級・降級・格上挑戦）, TRAN（転厩）, JOCKEY（乗り替わりと騎手の格の差）, GELD（去勢明け）,
    WEIGHT（馬体重・斤量の変化）, TRANSPORT（東西の遠征）, AGE（年齢の節目）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features.factory_self import KEY, _f, _g, _prior_cumsum, lag, prep, seg_affinity

EAST = {"03", "04", "05", "06"}
WEST = {"07", "08", "09", "10"}
GROUPS = ["EQ", "DIST", "SURF", "FIRST", "REST", "CLASS", "TRAN", "JOCKEY", "GELD", "WEIGHT", "TRANSPORT", "AGE", "COMBO"]


# ───────────────────────── 部品
def pop_asof(d: pd.DataFrame, key: pd.Series, val: str, k: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """同じ key の行の val の『その日より前まで』の平均（全体の as-of 平均へ縮約）と件数。key が欠損の行は NaN。"""
    kk = key.astype("string").to_numpy()
    t = pd.DataFrame({"k": kk, "dn": d["dnum"].to_numpy(), "v": d[val].to_numpy(float)})
    ok = pd.notna(kk) & np.isfinite(t["v"].to_numpy())
    tv = t[ok]
    g = tv.groupby(["k", "dn"]).agg(s=("v", "sum"), n=("v", "size")).reset_index().sort_values(["k", "dn"])
    g["S"] = g.groupby("k")["s"].cumsum()
    g["N"] = g.groupby("k")["n"].cumsum()
    gg = tv.groupby("dn").agg(s=("v", "sum"), n=("v", "size")).reset_index().sort_values("dn")
    gg["GS"], gg["GN"] = gg["s"].cumsum(), gg["n"].cumsum()
    q = t[["k", "dn"]].reset_index().rename(columns={"index": "ix"})
    q = q[pd.notna(q["k"])].sort_values("dn")
    m = pd.merge_asof(q, g[["k", "dn", "S", "N"]].sort_values("dn"), on="dn", by="k", allow_exact_matches=False, direction="backward")
    m = pd.merge_asof(m.sort_values("dn"), gg[["dn", "GS", "GN"]], on="dn", allow_exact_matches=False, direction="backward")
    gm = (m["GS"] / m["GN"]).to_numpy(float)
    n = m["N"].fillna(0).to_numpy(float)
    est = (m["S"].fillna(0).to_numpy(float) + k * gm) / (n + k)
    out = np.full(len(d), np.nan)
    cnt = np.full(len(d), np.nan)
    pos = d.index.get_indexer(m["ix"])
    out[pos], cnt[pos] = est, n
    return out, cnt


def situ(d: pd.DataFrame, key: pd.Series, name: str, k: float = 3.0, pop: bool = True) -> pd.DataFrame:
    """ある『状況』（key。該当しない行は NaN）について、馬自身の過去の親和度・件数・初経験フラグ、全馬の as-of 成績率を作る。"""
    ks = key.astype("string")
    on = ks.notna().to_numpy()
    aff = seg_affinity(d, ks.fillna("__none__"), name, vals=("fp", "top3"), k=k)
    out = pd.DataFrame(index=d.index)
    for c in aff.columns:
        out[c] = np.where(on, aff[c].to_numpy(float), np.nan)
    out[f"{name}_first"] = np.where(on, (aff[f"{name}_n"].to_numpy(float) == 0).astype(float), np.nan)
    if pop:
        r, n = pop_asof(d, ks, "top3")
        out[f"{name}_pop_top3"] = r
        out[f"{name}_pop_n"] = n
    return out


def _tier(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(x, [-np.inf] + edges + [np.inf], labels=labels, right=False).astype("string")


def _dist_band(dist: pd.Series) -> pd.Series:
    return _tier(dist, [1400, 1700, 2000, 2500], ["S", "M1", "M2", "L1", "L2"])


# ───────────────────────── 群
def g_eq(d: pd.DataFrame, E: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """馬具（ブリンカー）。E は SmartRC の fullresults 由来の馬×戦績（horse_id, race_id, rdate, blinker, rank, entry）。
    当日の着用・初着用・外した・着用の継続・着用時と非着用時の成績の差。E の履歴は 2020 年より前の戦績も含む。"""
    if E is None or E.empty:
        return pd.DataFrame(index=d.index)
    e = E.copy()
    e["blk"] = pd.to_numeric(e["blinker"], errors="coerce").clip(0, 1)
    e["fpe"] = (pd.to_numeric(e["rank"], errors="coerce") - 1.0) / (pd.to_numeric(e["entry"], errors="coerce").clip(lower=2) - 1.0)
    e["fpe"] = e["fpe"].where(pd.to_numeric(e["rank"], errors="coerce") > 0)
    e["rd"] = pd.to_datetime(e["rdate"].astype(str), format="%Y%m%d", errors="coerce")
    e = e.dropna(subset=["rd"]).sort_values(["horse_id", "rd", "race_id"], kind="stable").reset_index(drop=True)
    gh = e.groupby("horse_id", sort=False)
    prev_blk = gh["blk"].shift(1)
    on_c = e["blk"].fillna(0.0)
    off_c = (1.0 - e["blk"]).fillna(0.0)
    n_on = _prior_cumsum(on_c, [e["horse_id"]])
    n_off = _prior_cumsum(off_c, [e["horse_id"]])
    s_on = _prior_cumsum((e["fpe"] * e["blk"]).fillna(0), [e["horse_id"]])
    s_off = _prior_cumsum((e["fpe"] * (1 - e["blk"])).fillna(0), [e["horse_id"]])
    k = 3.0
    allm = 0.5
    m_on = (s_on + k * allm) / (n_on + k)
    m_off = (s_off + k * allm) / (n_off + k)
    h = e["horse_id"].to_numpy()
    b = e["blk"].to_numpy(float)
    switch = (e["blk"] != prev_blk) | prev_blk.isna()
    run_id = switch.astype(int).groupby(e["horse_id"], sort=False).cumsum()
    run_len = e.assign(_r=run_id).groupby(["horse_id", "_r"], sort=False).cumcount() + 1                      # その走までの同じ状態の連続数
    run = run_len.groupby(e["horse_id"], sort=False).shift(1).to_numpy(float)                                # 1 走前の時点の連続数
    out = pd.DataFrame({
        "eq_blk": b, "eq_first": ((b == 1) & (n_on.to_numpy() == 0) & prev_blk.notna().to_numpy()).astype(float),
        "eq_off": ((b == 0) & (prev_blk.to_numpy() == 1)).astype(float), "eq_on_cont": ((b == 1) & (prev_blk.to_numpy() == 1)).astype(float),
        "eq_n_on": n_on.to_numpy(), "eq_n_off": n_off.to_numpy(), "eq_prev_blk": prev_blk.to_numpy(),
        "eq_fp_on": np.where(n_on > 0, m_on, np.nan), "eq_fp_off": np.where(n_off > 0, m_off, np.nan),
        "eq_fp_diff": np.where((n_on > 0) & (n_off > 0), m_on - m_off, np.nan),
        "eq_streak": run, "eq_first_after_n": np.where((b == 1) & (n_on.to_numpy() == 0), n_off.to_numpy(), np.nan),
    })
    out.loc[~np.isfinite(b), [c for c in out.columns if c != "eq_blk"]] = np.nan
    out["race_id"], out["horse_id"] = e["race_id"].astype(str).to_numpy(), e["horse_id"].astype(str).to_numpy()
    m = d[KEY].merge(out.drop_duplicates(KEY), on=KEY, how="left")
    m.index = d.index
    return m.drop(columns=KEY)


def g_dist(d: pd.DataFrame, **_) -> pd.DataFrame:
    dd = d["dist"] - lag(d, "dist")
    tier = np.sign(dd) * np.minimum(3, np.ceil(dd.abs() / 200.0))                     # ±1: 〜200m, ±2: 〜400m, ±3: それ以上
    tier = tier.where(dd.notna())
    sf = d["surface"].astype(str)
    key = pd.Series(np.where(tier.notna(), sf + "|" + tier.astype("Int64").astype(str), None), index=d.index, dtype="string")
    out = _f(d, dc_tier=tier, dc_ext=(dd > 0).astype(float), dc_short=(dd < 0).astype(float), dc_ext400=(dd >= 400).astype(float),
             dc_short400=(dd <= -400).astype(float), dc_ext600=(dd >= 600).astype(float), dc_short600=(dd <= -600).astype(float))
    out.loc[dd.isna(), out.columns] = np.nan
    ext = pd.Series(np.where(dd > 0, "ext", np.where(dd < 0, "short", np.where(dd == 0, "same", None))), index=d.index, dtype="string")
    return pd.concat([out, situ(d, key, "dcs"), situ(d, ext, "dce")], axis=1)


def g_surf(d: pd.DataFrame, **_) -> pd.DataFrame:
    ps, sc = lag(d, "surface").astype(object).to_numpy(), d["surface"].astype(object).to_numpy()
    has = pd.notna(ps)
    dd = (d["dist"] - lag(d, "dist")).to_numpy(float)
    dsign = np.where(dd > 0, "延長", np.where(dd < 0, "短縮", np.where(dd == 0, "同", None)))
    sw = np.where(has, (ps != sc).astype(float), np.nan)
    t2d, d2t = (ps == "芝") & (sc == "ダート"), (ps == "ダート") & (sc == "芝")
    out = _f(d, sf_switch=sw, sf_t2d=np.where(has, t2d.astype(float), np.nan), sf_d2t=np.where(has, d2t.astype(float), np.nan))
    ok = has & pd.notna(dsign)
    key = pd.Series(np.where(ok, pd.Series(ps).astype(str) + ">" + pd.Series(sc).astype(str) + "|" + pd.Series(dsign).astype(str), None), index=d.index, dtype="string")
    swk = pd.Series(np.where(has, pd.Series(ps).astype(str) + ">" + pd.Series(sc).astype(str), None), index=d.index, dtype="string")
    return pd.concat([out, situ(d, key, "sfk"), situ(d, swk, "sfw")], axis=1)


def g_first(d: pd.DataFrame, **_) -> pd.DataFrame:
    sf = d["surface"].astype("string")
    band = _dist_band(d["dist"])
    course = (d["venue_code"].astype(str) + "|" + sf + "|" + d["dist"].round().astype("Int64").astype(str)).where(d["dist"].notna()).astype("string")
    direction = d["direction"].astype("string").replace("", pd.NA)
    heavy = pd.Series(np.where(d["going"] >= 2, "heavy", None), index=d.index, dtype="string")
    longd = pd.Series(np.where(d["dist"] >= 2400, "long", None), index=d.index, dtype="string")
    shortd = pd.Series(np.where(d["dist"] <= 1300, "sprint", None), index=d.index, dtype="string")
    big = pd.Series(np.where(d["nfield"] >= 16, "big", None), index=d.index, dtype="string")
    venue = d["venue_code"].astype("string")
    parts = [situ(d, sf, "fsf"), situ(d, (sf + "|" + band).where(band.notna()), "fbd"), situ(d, course, "fcs", pop=False), situ(d, direction, "fdr"),
             situ(d, heavy, "fhv"), situ(d, longd, "fld"), situ(d, shortd, "fsp"), situ(d, big, "fbg"), situ(d, venue, "fvn", pop=False)]
    return pd.concat(parts, axis=1)


def g_rest(d: pd.DataFrame, **_) -> pd.DataFrame:
    days = d["dnum"] - lag(d, "dnum")
    band = _tier(days, [15, 36, 64, 91, 181], ["r0", "r1", "r2", "r3", "r4", "r5"])
    lay = (days >= 91) | days.isna()
    cyc = lay.astype(int).groupby(d["horse_id"], sort=False).cumsum()
    idx = d.assign(_c=cyc).groupby(["horse_id", "_c"], sort=False).cumcount().to_numpy(float)          # 0 = 休み明け初戦, 1 = 叩き 2 戦目 …
    out = _f(d, rs_days=days, rs_layoff=(days >= 91).astype(float), rs_tataki=np.where(days.notna(), idx, np.nan), rs_tataki_cap=np.where(days.notna(), np.minimum(idx, 5), np.nan))
    out.loc[days.isna(), ["rs_layoff"]] = np.nan
    key = band.where(days.notna())
    key2 = (band + "|t" + pd.Series(np.minimum(idx, 3), index=d.index).astype(int).astype(str)).where(days.notna())
    return pd.concat([out, situ(d, key, "rsb"), situ(d, key2, "rst")], axis=1)


def g_class(d: pd.DataFrame, **_) -> pd.DataFrame:
    pc = lag(d, "cls")
    up, dn = (d["cls"] > pc), (d["cls"] < pc)
    mx = _g(d.assign(_c=d["cls"]))["_c"].transform(lambda s: s.shift(1).cummax())
    win_cls = d.assign(_w=np.where(d["win"] == 1, d["cls"], np.nan))
    maxwin = _g(win_cls)["_w"].transform(lambda s: s.shift(1).cummax())
    challenge = (d["cls"] > maxwin).where(maxwin.notna() | (_g(d)["win"].transform(lambda s: s.shift(1).cummax()) == 1))
    out = _f(d, cl_up=up.astype(float), cl_down=dn.astype(float), cl_same=(d["cls"] == pc).astype(float), cl_vs_max=d["cls"] - mx,
             cl_challenge=challenge.astype(float), cl_vs_maxwin=d["cls"] - maxwin)
    out.loc[pc.isna(), ["cl_up", "cl_down", "cl_same"]] = np.nan
    key = pd.Series(np.where(pc.isna(), None, np.where(up, "up", np.where(dn, "down", "same"))), index=d.index, dtype="string")
    ch = pd.Series(np.where(challenge == True, "challenge", None), index=d.index, dtype="string")   # noqa: E712
    return pd.concat([out, situ(d, key, "clk"), situ(d, ch, "clc")], axis=1)


def g_tran(d: pd.DataFrame, **_) -> pd.DataFrame:
    pt, t = lag(d, "trainer_id").astype(object).to_numpy(), d["trainer_id"].astype(object).to_numpy()
    has = pd.notna(pt)
    chg_b = has & (pt != t)
    cyc = pd.Series(np.where(has, chg_b, True).astype(int), index=d.index).groupby(d["horse_id"], sort=False).cumsum()
    idx = d.assign(_c=cyc).groupby(["horse_id", "_c"], sort=False).cumcount().to_numpy(float)
    n_tr = pd.Series(chg_b.astype(float), index=d.index).groupby(d["horse_id"], sort=False).cumsum()
    out = _f(d, tr_chg=np.where(has, chg_b.astype(float), np.nan), tr_run_idx=np.where(has, idx, np.nan), tr_n_changes=n_tr)
    key = pd.Series(np.where(chg_b, "chg1", np.where(has & (idx <= 2), "chg_early", None)), index=d.index, dtype="string")
    return pd.concat([out, situ(d, key, "trk")], axis=1)


def g_jockey(d: pd.DataFrame, **_) -> pd.DataFrame:
    pj, jc = lag(d, "jockey_id").astype(object), d["jockey_id"].astype(object)
    pjn, jn_ = pj.to_numpy(), jc.to_numpy()
    has = pd.notna(pjn)
    chg_b = has & (pjn != jn_)
    jr, _n = pop_asof(d, jc.astype("string"), "top3", k=50.0)
    pr = _asof_with_other_key(d, jc.astype("string"), pj.astype("string"), "top3", 50.0)          # 前走の騎手の『今日時点の』成績率
    cont = d.assign(_j=jc.astype(str)).groupby(["horse_id", "_j"], sort=False).cumcount().to_numpy(float)
    diff = jr - pr
    out = _f(d, jk_chg=np.where(has, chg_b.astype(float), np.nan), jk_rate=jr, jk_rate_prev=np.where(has, pr, np.nan), jk_rate_diff=np.where(has, diff, np.nan),
             jk_ride_n=cont, jk_new_pair=(cont == 0).astype(float), jk_up=np.where(has, (diff > 0.03).astype(float), np.nan),
             jk_down=np.where(has, (diff < -0.03).astype(float), np.nan))
    key = pd.Series(np.where(chg_b, np.where(diff > 0.03, "up", np.where(diff < -0.03, "down", "flat")), None), index=d.index, dtype="string")
    return pd.concat([out, situ(d, key, "jkc")], axis=1)


def _asof_with_other_key(d: pd.DataFrame, own_key: pd.Series, query_key: pd.Series, val: str, k: float) -> np.ndarray:
    """own_key で集計した as-of 表を、query_key（同じ行の別のキー、例: 前走の騎手）で引く。"""
    kk = own_key.astype("string").to_numpy()
    t = pd.DataFrame({"k": kk, "dn": d["dnum"].to_numpy(), "v": d[val].to_numpy(float)})
    ok = pd.notna(kk) & np.isfinite(t["v"].to_numpy())
    tv = t[ok]
    g = tv.groupby(["k", "dn"]).agg(s=("v", "sum"), n=("v", "size")).reset_index().sort_values(["k", "dn"])
    g["S"] = g.groupby("k")["s"].cumsum()
    g["N"] = g.groupby("k")["n"].cumsum()
    gg = tv.groupby("dn").agg(s=("v", "sum"), n=("v", "size")).reset_index().sort_values("dn")
    gg["GS"], gg["GN"] = gg["s"].cumsum(), gg["n"].cumsum()
    q = pd.DataFrame({"ix": np.arange(len(d)), "k": query_key.astype("string").to_numpy(), "dn": d["dnum"].to_numpy()})
    q = q[pd.notna(q["k"])].sort_values("dn")
    m = pd.merge_asof(q, g[["k", "dn", "S", "N"]].sort_values("dn"), on="dn", by="k", allow_exact_matches=False, direction="backward")
    m = pd.merge_asof(m.sort_values("dn"), gg[["dn", "GS", "GN"]], on="dn", allow_exact_matches=False, direction="backward")
    est = (m["S"].fillna(0).to_numpy(float) + k * (m["GS"] / m["GN"]).to_numpy(float)) / (m["N"].fillna(0).to_numpy(float) + k)
    out = np.full(len(d), np.nan)
    out[m["ix"].to_numpy()] = est
    return out


def g_geld(d: pd.DataFrame, **_) -> pd.DataFrame:
    s, ps = d["sex"].astype(object).to_numpy(), lag(d, "sex").astype(object).to_numpy()
    first = (s == "セ") & pd.notna(ps) & (ps != "セ")
    n_since = pd.Series((s == "セ").astype(float), index=d.index).groupby(d["horse_id"], sort=False).cumsum().to_numpy()
    out = _f(d, gd_is_geld=(s == "セ").astype(float), gd_first=first.astype(float), gd_n_since=np.where(s == "セ", n_since - 1, np.nan))
    key = pd.Series(np.where(first, "geld_first", None), index=d.index, dtype="string")
    return pd.concat([out, situ(d, key, "gdk")], axis=1)


def g_weight(d: pd.DataFrame, **_) -> pd.DataFrame:
    wc = d["wt_chg"]
    tier = _tier(wc, [-10, -4, 4, 10, 20], ["w_m10", "w_m4", "w_0", "w_p4", "w_p10", "w_p20"])
    jwd = d["jw"] - lag(d, "jw")
    jtier = _tier(jwd, [-2, -0.5, 0.5, 2], ["j_dn2", "j_dn", "j_0", "j_up", "j_up2"])
    wmean = _g(d.assign(_w=d["wt"]))["_w"].transform(lambda s: s.shift(1).expanding().mean())
    out = _f(d, wt_big_up=(wc >= 10).astype(float), wt_big_dn=(wc <= -10).astype(float), wt_huge=(wc.abs() >= 20).astype(float),
             wt_vs_mean=d["wt"] - wmean, jw_chg=jwd, jw_up=(jwd > 0.5).astype(float), jw_dn=(jwd < -0.5).astype(float))
    out.loc[wc.isna(), ["wt_big_up", "wt_big_dn", "wt_huge"]] = np.nan
    out.loc[jwd.isna(), ["jw_up", "jw_dn"]] = np.nan
    return pd.concat([out, situ(d, tier.where(wc.notna()), "wtk"), situ(d, jtier.where(jwd.notna()), "jwk")], axis=1)


def g_transport(d: pd.DataFrame, home: pd.Series | None = None, **_) -> pd.DataFrame:
    """home: horse_id → 'E'/'W'（美浦/栗東）。無ければ馬ごとの出走場の多数派（2020 年以降の見える範囲）から推定する。"""
    v = d["venue_code"].astype(str).str.zfill(2)
    region = pd.Series(np.where(v.isin(EAST), "E", np.where(v.isin(WEST), "W", "H")), index=d.index)
    if home is not None:
        hm = d["horse_id"].astype(str).map(home)
    else:
        cnt = pd.crosstab(d["horse_id"], region)
        hm = d["horse_id"].map((cnt.get("W", 0) > cnt.get("E", 0)).map({True: "W", False: "E"}))
    away = ((region != "H") & (hm.notna()) & (hm != region)).astype(float).where(hm.notna())
    key = pd.Series(np.where(region == "H", "hokkaido", np.where(hm.notna(), hm.astype(str) + ">" + region, None)), index=d.index, dtype="string")
    out = _f(d, tp_away=away, tp_hokkaido=(region == "H").astype(float))
    prev_away = _g(d.assign(_a=away))["_a"].shift(1)
    out["tp_consec_away"] = ((away == 1) & (prev_away == 1)).astype(float).where(away.notna())
    return pd.concat([out, situ(d, key, "tpk")], axis=1)


def g_age(d: pd.DataFrame, **_) -> pd.DataFrame:
    age, m = d["age"], d["month"]
    season = pd.Series(np.where(m <= 5, "spring", np.where(m <= 8, "summer", "autumn")), index=d.index)
    key = pd.Series(np.where(age.notna(), age.fillna(0).astype(int).clip(upper=6).astype(str) + "|" + season, None), index=d.index, dtype="string")
    pa = lag(d, "age")
    first_old = ((age >= 4) & (pa <= 3)).astype(float).where(pa.notna())
    new_age = (age != pa).astype(float).where(pa.notna())
    out = _f(d, ag_age=age, ag_first_old=first_old, ag_new_age=new_age, ag_3_autumn=((age == 3) & (m >= 9)).astype(float),
             ag_3_spring=((age == 3) & (m <= 5)).astype(float), ag_month=m.astype(float))
    return pd.concat([out, situ(d, key, "agk", pop=True)], axis=1)


def g_combo(d: pd.DataFrame, E: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """状況の組み合わせ（休み明け・初経験・昇級・乗り替わり・馬具・面×距離の替わり など）。間隔そのものの特徴量は tp-34 の担当なので作らない。
    組み合わせごとに『馬自身のその組み合わせでの過去の成績』と『全馬のその組み合わせの as-of 成績率』を持つ。"""
    days = d["dnum"] - lag(d, "dnum")
    lay = (days >= 91).to_numpy()
    has_prev = days.notna().to_numpy()
    dd = (d["dist"] - lag(d, "dist")).to_numpy(float)
    dist_t = np.where(dd >= 200, "ext", np.where(dd <= -200, "short", "same"))
    ps, sc = lag(d, "surface").astype(object).to_numpy(), d["surface"].astype(object).to_numpy()
    sw = has_prev & pd.notna(ps) & (ps != sc)
    pc = lag(d, "cls").to_numpy(float)
    cl_up = has_prev & (d["cls"].to_numpy(float) > pc)
    cl_dn = has_prev & (d["cls"].to_numpy(float) < pc)
    pj = lag(d, "jockey_id").astype(object).to_numpy()
    jchg = has_prev & pd.notna(pj) & (pj != d["jockey_id"].astype(object).to_numpy())
    wc = d["wt_chg"].to_numpy(float)
    wbig = np.abs(wc) >= 10
    s_, ps_s = d["sex"].astype(object).to_numpy(), lag(d, "sex").astype(object).to_numpy()
    geld1 = (s_ == "セ") & pd.notna(ps_s) & (ps_s != "セ")
    age, mo = d["age"].to_numpy(float), d["month"].to_numpy(float)
    first_course = (_same_count(d, d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + d["dist"].round().astype("Int64").astype(str)) == 0)
    blk_first = np.zeros(len(d), dtype=bool)
    if E is not None and len(E):
        eq = g_eq(d, E=E)
        blk_first = (eq["eq_first"].to_numpy(float) == 1)

    def key(mask, label):
        return pd.Series(np.where(mask & has_prev, label, None), index=d.index, dtype="string")

    keys = {
        "cb_lay_blk": key(lay & blk_first, "lay_blk"), "cb_lay_short": key(lay & (dist_t == "short"), "lay_short"), "cb_lay_ext": key(lay & (dist_t == "ext"), "lay_ext"),
        "cb_short_cls": key((dist_t == "short") & cl_up, "short_up"), "cb_lay_sw": key(lay & sw, "lay_sw"), "cb_sw_dist": key(sw & (dist_t != "same"), "sw_distchg"),
        "cb_up_jchg": key(cl_up & jchg, "up_jchg"), "cb_dn_jchg": key(cl_dn & jchg, "dn_jchg"), "cb_lay_firstcourse": key(lay & first_course, "lay_fc"),
        "cb_lay_wbig": key(lay & wbig, "lay_wbig"), "cb_lay_geld": key(lay & geld1, "lay_geld"), "cb_blk_distchg": key(blk_first & (dist_t != "same"), "blk_distchg"),
        "cb_3aut_up": key((age == 3) & (mo >= 9) & cl_up, "3aut_up"),
    }
    parts = [situ(d, k, n, k=3.0) for n, k in keys.items()]
    return pd.concat(parts, axis=1)


def _same_count(d: pd.DataFrame, key: pd.Series) -> np.ndarray:
    return d.assign(_k=key.astype(str)).groupby(["horse_id", "_k"], sort=False).cumcount().to_numpy(float)


FUNCS = {"EQ": g_eq, "DIST": g_dist, "SURF": g_surf, "FIRST": g_first, "REST": g_rest, "CLASS": g_class, "TRAN": g_tran, "JOCKEY": g_jockey, "GELD": g_geld,
         "WEIGHT": g_weight, "TRANSPORT": g_transport, "AGE": g_age, "COMBO": g_combo}


def build_groups(runs: pd.DataFrame, E: pd.DataFrame | None = None, home: pd.Series | None = None, only: list[str] | None = None) -> dict[str, pd.DataFrame]:
    d = prep(runs)
    out = {}
    for name, fn in FUNCS.items():
        if only and name not in only:
            continue
        f = fn(d, E=E, home=home)
        if f.shape[1]:
            out[name] = f.astype("float32").assign(race_id=d["race_id"].to_numpy(), horse_id=d["horse_id"].to_numpy())
    return out


def sit_features(hist_runs: pd.DataFrame, target_rows: pd.DataFrame, groups: list[str], E: pd.DataFrame | None = None, home: pd.Series | None = None) -> pd.DataFrame:
    """推論用: 履歴（その日より前の確定結果）と当日の出走馬の行（結果は未確定）をつないで群を作り、当日の行だけ返す。
    学習時と同じ関数を使う（二重実装にしない）。target_rows の finish_position 等は無くてよい（欠損のまま）。"""
    cols = list(hist_runs.columns)
    T = target_rows.reindex(columns=cols).copy()
    allr = pd.concat([hist_runs, T], ignore_index=True).sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    out = build_groups(allr, E=E, home=home, only=groups)
    tk = pd.MultiIndex.from_frame(T[KEY].astype(str))
    parts = []
    for g in groups:
        if g not in out:
            continue
        f = out[g].set_index(KEY)
        parts.append(f.loc[f.index.isin(tk)])
    return pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(columns=KEY)
