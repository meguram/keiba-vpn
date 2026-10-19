"""tp-39: 成長曲線（馬体重の変化とパフォーマンスの関係を馬ごとに推定した『補正ポイント』）の特徴量。すべて as-of（そのレース日より前の走だけ）。

考え方
  パフォーマンス P（race 内で相対化した指標）から、年齢・性別・クラス・面・距離・休養間隔・前走の P で説明できる分を引いた残差 R（『期待以上度』）を作る。
  馬ごとに『R ~ 馬体重の変化 x_chg』『R ~ その馬自身の平均体重との差 x_lev』の傾き β を、その馬の過去の走だけから逐次に推定する
  （馬ごとの切片は過去平均で吸収。累積和を 1 行ずらして作るので、自分や未来の走は入らない）。傾きは、全馬共通の傾きへ経験ベイズで縮約する。
  『補正ポイント』cp = 縮約済みの傾き × 今回の体重変化（馬体重は T-60 で発表されるので推論で使える）。
  母集団の定数（年齢・性別・クラスなどの係数、縮約の強さ、母集団の応答表）は FIT_YEARS（2020–2021）の走だけから作り、全期間に同じ定数を当てる
  （valid の 2022 年以降には一切使われない）。

群
  POP_free / POP_megu  母集団の応答（性別×年齢×季節×馬格×体重変化の帯の平均応答、馬格の効果。free = 着順・タイム由来の指標 + 体重だけの列、megu = megu 系の指標）
  SL_m  馬ごとの傾き（chg / lev）、標準誤差、z、縮約後の傾き、補正ポイント、母集団の補正との差、R と体重の相関
  KR_m  馬ごとのカーネル回帰（今回と近い体重・近い体重変化のときの過去の R の平均 − その馬の平均）、最良時の体重との差
  GR_m  R の直近の傾き・指数平滑・最大/最小（成長の勢い）
  m = meg（自前の指数 megu v3a の相対 = 同世代・同クラスの 2 着馬が 50、レース内の中央値との差）/ mga（同じ指数の絶対。成長の軌跡用）/ pro（着順の正規スコア）/ pt（着順の正規スコアとタイム偏差の平均。megu 非依存）/ comp（meg・pro・タイム偏差の合成）
  megu の版は load_megu が attrs["megu_version"] に入れ、評価の結果に必ず記録する（megu は改善中で、版が変わると水準と順位が動く）。
市場オッズ・人気は使わない。キーは race_id / horse_id。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from src.pipeline.features import factory_self as fs
from src.pipeline.features.tp34_rest import band_id

KEY = ["race_id", "horse_id"]
FIT_YEARS = (2020, 2021)
METRICS = ("meg", "mga", "pro", "tim", "pt", "comp")
FEATURE_METRICS = ("meg", "mga", "pro", "pt", "comp")
MEGU_FREE_METRICS = ("pro", "pt")                      # megu_index に依存しない指標（着順の正規スコア / タイム偏差 / その平均）
MEGU_ROOT = "data/local/modeling/megu_recompute_20261019e"         # めぐ指数の最終候補 v3a（page_reference への反映前）。読み取り専用
MEGU_SPEC = {"col": "megu_index_v3a_feat", "abs_col": "megu_index_v3a_abs_feat", "status_col": "megu_v3a_status", "bad_status": (),
             "low_col": "megu_v3a_low_confidence", "ver_col": "megu_v3a_version"}
AGE_CATS = [2, 3, 4, 5, 6, 7]
SEXES = ["牡", "牝", "セ"]
CLS_CATS = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
SURFS = ["芝", "ダ"]
DIST_BINS = [0, 1300, 1700, 2100, 99999]
CHG_EDGES = [-99, -9.5, -4.5, -1.5, 1.5, 4.5, 9.5, 99]
LEV_EDGES = [-99, -9.5, -4.5, -1.5, 1.5, 4.5, 9.5, 99]
SZ_EDGES = [-99, -1.0, -0.33, 0.33, 1.0, 99]
GROUPS = ["POP_free", "POP_megu"] + [f"{p}_{m}" for p in ("SL", "KR", "GR") for m in FEATURE_METRICS] + ["SL0_pt", "SL0_comp"]
MEGU_FREE_GROUPS = ["POP_free"] + [f"{p}_{m}" for p in ("SL", "KR", "GR") for m in MEGU_FREE_METRICS] + ["SL0_pt"]
MEGU_GROUPS = [g for g in GROUPS if g not in MEGU_FREE_GROUPS]
MIN_RACE_N = 6
KINDS = ("chg", "lev", "abs", "labs")


# ───────────────────────── 指標
def load_megu(root, years: list[int], spec: dict | None = None) -> pd.DataFrame:
    """めぐ指数の parquet（<root>/<年>/megu_index_flat.parquet）から、相対の列 spec["col"] を megu_index、絶対の列 spec["abs_col"] を megu_abs として読む。
    status が bad_status の行・low_confidence の行・値が欠けた行は除く。読んだ版は DataFrame の attrs["megu"] に入れる（結果に必ず記録する）。
    spec の列が無い古い版は megu_index の列で代用する。"""
    import hashlib
    import json
    from pathlib import Path

    spec = {**MEGU_SPEC, **(spec or {})}
    parts, versions, nfiles = [], set(), 0
    for y in years:
        f = Path(root) / str(y) / "megu_index_flat.parquet"
        if not f.exists():
            continue
        nfiles += 1
        cols = list(pd.read_parquet(f).columns)
        col = spec["col"] if spec["col"] in cols else "megu_index"
        acol = spec["abs_col"] if spec["abs_col"] in cols else col
        want = [c for c in {"race_id", "horse_id", col, acol, spec["status_col"], spec["low_col"], spec["ver_col"]} if c in cols]
        m = pd.read_parquet(f, columns=want)
        if spec["status_col"] in m:
            m = m[m[spec["status_col"]].notna()]                                   # 状態の無い行 = 計算の対象外（障害戦など）
            if spec["bad_status"]:
                m = m[~m[spec["status_col"]].isin(list(spec["bad_status"]))]
        if spec["low_col"] in m:
            m = m[~m[spec["low_col"]].astype("boolean").fillna(False).astype(bool)]
        if spec["ver_col"] in m:
            versions |= set(m[spec["ver_col"]].dropna().astype(str).unique())
        m = m.assign(megu_index=m[col], megu_abs=m[acol])
        parts.append(m[["race_id", "horse_id", "megu_index", "megu_abs"]])
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["race_id", "horse_id", "megu_index", "megu_abs"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    m = m.dropna(subset=["megu_index"]).drop_duplicates(KEY)
    meta = {}
    for name in ("meta.json", "spec.json"):
        pth = Path("config/megu_v3a") / name
        if pth.exists():
            meta[name] = {"sha1": hashlib.sha1(pth.read_bytes()).hexdigest()[:12], **({"content": json.loads(pth.read_text(encoding="utf-8"))} if name == "meta.json" else {})}
    m.attrs["megu"] = {"root": str(root), "dir": Path(root).name, "files": nfiles, "version": sorted(versions) or ["(version 列なし)"], "col": spec["col"], "abs_col": spec["abs_col"], "rows": int(len(m)), "config": meta}
    return m


def prepare(runs: pd.DataFrame, megu: pd.DataFrame | None = None) -> pd.DataFrame:
    """runs（fs.load_runs の形、馬・日付順）に、race 内で相対化した生のパフォーマンス指標と、体重・休養の派生列を足す。各指標は同じ race の情報だけから決まる。"""
    d = fs.prep(runs)
    d["year"] = d["date"].dt.year
    d["gap"] = (d["dnum"] - fs.lag(d, "dnum")).astype(float)
    d["gapb"] = np.where(d["gap"].notna(), band_id(d["gap"].to_numpy()), -1.0)
    n = d["field_size"].astype(float)
    p = np.clip((d["fin"].to_numpy(float) - 0.5) / np.where(n > 0, n, np.nan), 0.01, 0.99)
    d["P_pro"] = np.where(n >= 5, ndtri(1.0 - p), np.nan)
    if megu is not None and len(megu):
        d = d.merge(megu[KEY + ["megu_index", "megu_abs"]], on=KEY, how="left")
    else:
        d["megu_index"], d["megu_abs"] = np.nan, np.nan
    cnt = d.groupby("race_id")["megu_index"].transform("count")
    med = d.groupby("race_id")["megu_index"].transform("median")
    d["raw_meg"] = np.where(cnt >= MIN_RACE_N, d["megu_index"] - med, np.nan)          # 相対（レース内の中央値との差）
    d["raw_mga"] = np.where(cnt >= MIN_RACE_N, d["megu_abs"], np.nan)                  # 絶対（成長の軌跡用。レースでそろえない）
    t = pd.to_numeric(d["time_sec"], errors="coerce").where(lambda s: s > 30)
    tc = t.groupby(d["race_id"]).transform("count")
    tm = t.groupby(d["race_id"]).transform("median")
    d["raw_tim"] = np.where(tc >= MIN_RACE_N, -(t - tm) / (d["dist"] / 200.0), np.nan)
    # 体重の変化（10kg 単位）と、その馬自身の過去平均との差
    d["x_chg"] = d["wt_chg"].clip(-30, 30) / 10.0
    cw = fs.excl_cum(d, ["horse_id"], "wt", True)
    sw = fs.excl_cum(d, ["horse_id"], "wt")
    prior_mean = np.where(cw >= 1, sw / np.maximum(cw, 1), np.nan)
    d["wt_prior_mean"] = prior_mean
    d["x_lev"] = (d["wt"] - prior_mean).clip(-30, 30) / 10.0
    d["x_abs"] = d["x_chg"].abs()                    # 変化の大きさ（増減どちらでも）
    d["x_labs"] = d["x_lev"].abs()
    d["chg_bin"] = pd.cut(d["wt_chg"], CHG_EDGES, labels=False).astype(float)
    d["lev_bin"] = pd.cut(d["wt"] - d["wt_prior_mean"], LEV_EDGES, labels=False).astype(float)
    return d


def apply_metrics(d: pd.DataFrame, consts: dict) -> pd.DataFrame:
    """生の指標を定数でそろえて P_meg / P_pro / P_tim / P_comp を作る（P は大きいほど良い）。"""
    d = d.copy()
    d["P_meg"] = (d["raw_meg"] / consts["scale"]["meg"]).clip(-4, 4)
    mu, sd = consts["scale"]["mga"]
    d["P_mga"] = ((d["raw_mga"] - mu) / sd).clip(-4, 4)
    d["P_tim"] = (d["raw_tim"] / consts["scale"]["tim"]).clip(-4, 4)
    M = np.column_stack([d["P_meg"].to_numpy(float), d["P_pro"].to_numpy(float), d["P_tim"].to_numpy(float)])
    k = np.isfinite(M).sum(1)
    d["P_comp"] = np.where(k >= 2, np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=1), np.nan)
    Mp = M[:, 1:3]                                                       # megu を含まない合成（着順の正規スコアとタイム偏差の平均）
    kp = np.isfinite(Mp).sum(1)
    d["P_pt"] = np.where(kp >= 1, np.nanmean(np.where(np.isfinite(Mp), Mp, np.nan), axis=1), np.nan)
    return d


def fit_scales(d: pd.DataFrame, years=FIT_YEARS) -> dict:
    s = d[d["year"].isin(years)]
    out: dict = {}
    for m, c in (("meg", "raw_meg"), ("tim", "raw_tim")):
        v = s[c].dropna().to_numpy(float)
        lo, hi = np.percentile(v, [1, 99])
        out[m] = float(np.std(np.clip(v, lo, hi)))
    v = s["raw_mga"].dropna().to_numpy(float)
    lo, hi = np.percentile(v, [1, 99])
    vc = np.clip(v, lo, hi)
    out["mga"] = [float(vc.mean()), float(vc.std())]
    return out


# ───────────────────────── 期待値（母集団）を引いた残差 R
def _design(d: pd.DataFrame, pcol: str) -> np.ndarray:
    age = d["age"].clip(2, 7).fillna(4).astype(int).to_numpy()
    cols = [np.ones(len(d))]
    cols += [(age == a).astype(float) for a in AGE_CATS]
    cols += [(d["sex"].to_numpy() == s).astype(float) for s in SEXES]
    cols += [(d["cls"].fillna(-1).astype(int).to_numpy() == c).astype(float) for c in CLS_CATS]
    cols += [(d["surface"].astype(str).to_numpy() == s).astype(float) for s in SURFS]
    db = np.digitize(d["dist"].to_numpy(float), DIST_BINS[1:-1])
    cols += [(db == i).astype(float) for i in range(len(DIST_BINS) - 1)]
    gb = d["gapb"].to_numpy(float)
    cols += [(gb == i).astype(float) for i in [-1, 0, 1, 2, 3, 4, 5, 6]]
    prev = fs.lag(d, pcol).to_numpy(float)
    cols += [np.where(np.isfinite(prev), prev, 0.0), (~np.isfinite(prev)).astype(float)]
    return np.column_stack(cols)


def fit_detrend(d: pd.DataFrame, years=FIT_YEARS) -> dict:
    """指標ごとに P ~ 年齢・性別・クラス・面・距離帯・休養間隔・前走の P を FIT_YEARS で最小二乗に当てる（リッジ微小）。"""
    out = {}
    fy = d["year"].isin(years).to_numpy()
    for m in METRICS:
        pc = f"P_{m}"
        y = d[pc].to_numpy(float)
        X = _design(d, pc)
        ok = fy & np.isfinite(y)
        A = X[ok]
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[ok])
        out[m] = coef.tolist()
    return out


def add_residuals(d: pd.DataFrame, consts: dict) -> pd.DataFrame:
    d = d.copy()
    for m in METRICS:
        pc = f"P_{m}"
        coef = np.asarray(consts["detrend"][m], dtype=float)
        d[f"R_{m}"] = d[pc].to_numpy(float) - _design(d, pc) @ coef
    return d


# ───────────────────────── 母集団の定数（縮約の強さ・応答表・標準体重）
def _horse_sums(d: pd.DataFrame, rcol: str, xcol: str, rows: np.ndarray) -> pd.DataFrame:
    v = rows & d[rcol].notna().to_numpy() & d[xcol].notna().to_numpy()
    s = d.loc[v, ["horse_id"]].copy()
    x, y = d.loc[v, xcol].to_numpy(float), d.loc[v, rcol].to_numpy(float)
    s["n"], s["x"], s["y"], s["xx"], s["xy"], s["yy"] = 1.0, x, y, x * x, x * y, y * y
    g = s.groupby("horse_id").sum()
    g["sxx"] = g["xx"] - g["x"] ** 2 / g["n"]
    g["sxy"] = g["xy"] - g["x"] * g["y"] / g["n"]
    g["syy"] = g["yy"] - g["y"] ** 2 / g["n"]
    return g


def fit_shrinkage(d: pd.DataFrame, years=FIT_YEARS) -> dict:
    """傾きの全体平均 β0、残差分散 σ²、馬ごとの真の傾きの分散 τ²（モーメント法。0 以下なら『馬ごとの違いは見えない』= 全馬共通へ強く縮約）。"""
    fy = d["year"].isin(years).to_numpy()
    out: dict = {}
    for m in FEATURE_METRICS + ("tim",):
        for kind in KINDS:
            g = _horse_sums(d, f"R_{m}", f"x_{kind}", fy)
            g = g[(g["n"] >= 6) & (g["sxx"] > 1e-3)]
            if len(g) < 100:
                out[f"{m}_{kind}"] = {"beta0": 0.0, "sigma2": 1.0, "tau2_raw": 0.0, "tau2": 1e-4, "horses": int(len(g))}
                continue
            beta0 = float(g["sxy"].sum() / g["sxx"].sum())
            bh = g["sxy"] / g["sxx"]
            s2i = (g["syy"] - bh * g["sxy"]).clip(lower=1e-6) / (g["n"] - 2).clip(lower=1)
            sigma2 = float(((g["n"] - 2).clip(lower=1) * s2i).sum() / (g["n"] - 2).clip(lower=1).sum())
            tau2 = float(np.mean((bh - beta0) ** 2 - sigma2 / g["sxx"]))
            out[f"{m}_{kind}"] = {"beta0": beta0, "sigma2": sigma2, "tau2_raw": tau2, "tau2": max(tau2, 1e-4), "horses": int(len(g)),
                                  "var_obs": float(np.var(bh)), "mean_noise": float(np.mean(sigma2 / g["sxx"]))}
    return out


def _season(month: pd.Series) -> pd.Series:
    """12〜2 月=0、3〜5 月=1、6〜8 月=2、9〜11 月=3。"""
    return ((month.astype(int) % 12) // 3).astype(int)


def _size_cols(d: pd.DataFrame, wt_std: dict) -> pd.DataFrame:
    """性別×年齢×季節ごとの標準体重（FIT_YEARS で作った表）との差の z と、その帯（馬格: 小〜大の 5 段階）。"""
    age_c = d["age"].clip(2, 7).fillna(4).astype(int).astype(str)
    season = _season(d["month"]).astype(str)
    k3 = d["sex"].astype(str) + "|" + age_c + "|" + season
    k2 = d["sex"].astype(str) + "|" + age_c
    mu = k3.map(lambda k: wt_std.get(k, [np.nan, np.nan])[0]).fillna(k2.map(lambda k: wt_std.get(k, [np.nan, np.nan])[0]))
    sd = k3.map(lambda k: wt_std.get(k, [np.nan, np.nan])[1]).fillna(k2.map(lambda k: wt_std.get(k, [np.nan, np.nan])[1]))
    z = ((d["wt"] - mu) / sd).clip(-4, 4)
    sz = pd.cut(z, SZ_EDGES, labels=False).astype(float)
    return pd.DataFrame({"age_c": age_c, "season": season, "wt_z": z.to_numpy(float), "sz_bin": sz.to_numpy(float), "wt_std_mu": mu.to_numpy(float)}, index=d.index)


def _tab(t: pd.DataFrame, keys: list[str], val: str, min_n: int) -> dict:
    g = t.groupby(keys)[val].agg(["mean", "size"])
    g = g[g["size"] >= min_n]
    return {"|".join(str(int(x)) if isinstance(x, (float, np.floating)) and np.isfinite(x) else str(x) for x in (idx if isinstance(idx, tuple) else (idx,))): float(r["mean"]) for idx, r in g.iterrows()}


def fit_population(d: pd.DataFrame, years=FIT_YEARS) -> dict:
    """母集団の成長曲線（FIT_YEARS の走だけから）。履歴の無い馬・少ない馬にも使える。
    chg  : (性別, 年齢, 季節, 馬格, 体重変化の帯) → 『その馬自身の平均より R がどれだけ高いか』。階層（細→粗）で件数が足りなければ粗い表へ。
    size : (性別, 年齢, 季節, 馬格) → R の平均（馬の履歴なしで使える。デビュー前後の馬格の効果）
    beta : (性別, 年齢, 馬格) ごとの、体重変化 / 体重差に対する R の傾き（馬ごとの傾きの縮約先）"""
    s = d[d["year"].isin(years)].copy()
    s["age_c"] = s["age"].clip(2, 7).fillna(4).astype(int).astype(str)
    s["season"] = _season(s["month"]).astype(str)
    w3 = s[s["wt"].notna()].groupby(["sex", "age_c", "season"])["wt"].agg(["mean", "std", "size"])
    w2 = s[s["wt"].notna()].groupby(["sex", "age_c"])["wt"].agg(["mean", "std", "size"])
    wt_std = {f"{a}|{b}": [float(r["mean"]), float(r["std"])] for (a, b), r in w2.iterrows()}
    wt_std.update({f"{a}|{b}|{c}": [float(r["mean"]), float(r["std"])] for (a, b, c), r in w3.iterrows() if r["size"] >= 200})
    sz = _size_cols(s, wt_std)
    s["sz_bin"], s["wt_z"] = sz["sz_bin"], sz["wt_z"]
    out: dict = {"wt_std": wt_std, "chg": {}, "lev": {}, "size": {}, "beta": {}}
    h = s["horse_id"]
    for m in FEATURE_METRICS:
        r = s[f"R_{m}"]
        cnt = r.groupby(h).transform("count")
        dev = (r - r.groupby(h).transform("mean")).where(cnt >= 3)
        t = s.assign(dev=dev, R=r)
        tc = t[t["dev"].notna() & t["chg_bin"].notna()]
        out["chg"][m] = {"l1": _tab(tc[tc["sz_bin"].notna()], ["sex", "age_c", "season", "sz_bin", "chg_bin"], "dev", 150),
                         "l2": _tab(tc, ["sex", "age_c", "chg_bin"], "dev", 100), "l3": _tab(tc, ["chg_bin"], "dev", 1)}
        tl = t[t["dev"].notna() & t["lev_bin"].notna()]
        out["lev"][m] = {"l2": _tab(tl, ["age_c", "lev_bin"], "dev", 100), "l3": _tab(tl, ["lev_bin"], "dev", 1)}
        tz = t[t["R"].notna() & t["sz_bin"].notna()]
        out["size"][m] = {"l1": _tab(tz, ["sex", "age_c", "season", "sz_bin"], "R", 150), "l2": _tab(tz, ["sex", "age_c", "sz_bin"], "R", 100), "l3": _tab(tz, ["sz_bin"], "R", 1)}
        # 傾きの母集団曲線（x と dev の馬内中心化済み回帰）
        for kind in KINDS:
            x = s[f"x_{kind}"]
            ok = dev.notna() & x.notna()
            xv = x.where(ok)
            xm = xv.groupby(h).transform("mean")
            dx, dy = (xv - xm), dev.where(ok)
            q = pd.DataFrame({"sex": s["sex"], "age_c": s["age_c"], "sz": s["sz_bin"], "xy": dx * dy, "xx": dx * dx})[ok.to_numpy()]
            tot = float(q["xy"].sum() / max(q["xx"].sum(), 1e-9))
            g = q.groupby(["sex", "age_c", "sz"])[["xy", "xx"]].sum()
            cell = {"|".join(str(int(v)) if isinstance(v, (float, np.floating)) else str(v) for v in idx): float(r["xy"] / r["xx"]) for idx, r in g.iterrows() if r["xx"] >= 200}
            out["beta"][f"{m}_{kind}"] = {"cells": cell, "global": tot}
    return out


def _lookup(keys: list[pd.Series], tabs: list[dict]) -> np.ndarray:
    """細い表から順に引き、無ければ粗い表へ。"""
    val = pd.Series(np.nan, index=keys[0].index)
    for k, t in zip(keys, tabs):
        val = val.fillna(k.map(t))
    return val.to_numpy(float)


def fit_constants(d: pd.DataFrame, years=FIT_YEARS) -> dict:
    """FIT_YEARS の走だけから、全期間に当てる定数をまとめて作る（prepare 済みの d を渡す）。"""
    consts: dict = {"fit_years": list(years), "scale": fit_scales(d, years)}
    d1 = apply_metrics(d, consts)
    consts["detrend"] = fit_detrend(d1, years)
    d2 = add_residuals(d1, consts)
    consts["shrink"] = fit_shrinkage(d2, years)
    consts["pop"] = fit_population(d2, years)
    return consts


# ───────────────────────── 群
def add_size(d: pd.DataFrame, consts: dict) -> pd.DataFrame:
    d = d.copy()
    z = _size_cols(d, consts["pop"]["wt_std"])
    for c in ("age_c", "season", "wt_z", "sz_bin", "wt_std_mu"):
        d[c] = z[c].to_numpy()
    return d


def _prior_beta(d: pd.DataFrame, m: str, kind: str, consts: dict, cell: bool) -> np.ndarray:
    """傾きの縮約先。cell=True なら (性別, 年齢, 馬格) ごとの母集団の傾き、False なら全体の傾き β0。"""
    sh = consts["shrink"][f"{m}_{kind}"]
    if not cell:
        return np.full(len(d), sh["beta0"])
    bt = consts["pop"]["beta"][f"{m}_{kind}"]
    k = d["sex"].astype(str) + "|" + d["age_c"].astype(str) + "|" + d["sz_bin"].map(lambda x: str(int(x)) if np.isfinite(x) else "nan")
    v = k.map(bt["cells"])
    return v.fillna(sh["beta0"]).to_numpy(float)


def _slope_block(d: pd.DataFrame, m: str, kind: str, consts: dict, cell_prior: bool = True, tag_prefix: str = "gc") -> pd.DataFrame:
    rc, xc = f"R_{m}", f"x_{kind}"
    sh = consts["shrink"][f"{m}_{kind}"]
    v = (d[rc].notna() & d[xc].notna())
    vf = v.astype(float)
    x, y = d[xc].where(v, 0.0), d[rc].where(v, 0.0)
    g = d["horse_id"]
    n = fs._prior_cumsum(vf, g).to_numpy()
    Sx, Sy = fs._prior_cumsum(x, g).to_numpy(), fs._prior_cumsum(y, g).to_numpy()
    Sxx, Sxy, Syy = (fs._prior_cumsum(x * x, g).to_numpy(), fs._prior_cumsum(x * y, g).to_numpy(), fs._prior_cumsum(y * y, g).to_numpy())
    b0 = _prior_beta(d, m, kind, consts, cell_prior)
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx = Sxx - Sx ** 2 / n
        sxy = Sxy - Sx * Sy / n
        syy = Syy - Sy ** 2 / n
        ok = (n >= 3) & (sxx > 1e-3)
        bh = np.where(ok, sxy / sxx, np.nan)
        s2 = sh["sigma2"]
        se = np.where(ok, np.sqrt(s2 / sxx), np.nan)
        k = s2 / sh["tau2"]
        bp = np.where(ok, (sxx * bh + k * b0) / (sxx + k), b0)
        corr = np.where(ok & (syy > 1e-6), sxy / np.sqrt(sxx * syy), np.nan)
        xbar = np.where(n >= 1, Sx / n, np.nan)
        ybar = np.where(n >= 1, Sy / n, np.nan)
    xnow = d[xc].to_numpy(float)
    t = f"{tag_prefix}_{kind}_{m}"
    return pd.DataFrame({
        f"{t}_n": n, f"{t}_beta": np.clip(bh, -4, 4), f"{t}_se": se, f"{t}_z": np.clip(bh / se, -8, 8), f"{t}_post": bp,
        f"{t}_cp": bp * xnow, f"{t}_cp_pop": b0 * xnow, f"{t}_cp_ex": (bp - b0) * xnow, f"{t}_cp_raw": np.clip(bh, -4, 4) * xnow,
        f"{t}_corr": corr, f"{t}_xbar": xbar, f"{t}_ybar": ybar, f"{t}_cpc": bp * (xnow - xbar), f"{t}_cpc_pop": b0 * (xnow - xbar),
    }, index=d.index)


def _kernel_block(d: pd.DataFrame, m: str, bw_kg: float = 5.0, bw_chg: float = 4.0, k0: float = 1.0, tau: float = 0.5) -> pd.DataFrame:
    """馬ごとのカーネル回帰。i 走目の値は 0..i-1 走目の配列だけから、長さ i の配列に対する厳密な和（math.fsum）で計算する。
    行列のまま和を取ると、後ろにある（未来の）走の数で丸めの端数が変わるため、行ごとに切り出して和の順序に依存しない形にしている。"""
    from math import fsum

    R = d[f"R_{m}"].to_numpy(float)
    wt = d["wt"].to_numpy(float)
    xc = d["wt_chg"].clip(-30, 30).to_numpy(float)
    h = d["horse_id"].to_numpy()
    N = len(d)
    starts = np.r_[0, np.flatnonzero(h[1:] != h[:-1]) + 1, N]
    names = ["adv_lev", "ev_lev", "adv_chg", "ev_chg", "wbest_gap", "wbest_gap_abs"]
    out = {nm: np.full(N, np.nan) for nm in names}
    for a, b in zip(starts[:-1], starts[1:]):
        n = b - a
        if n < 2:
            continue
        r, w, c = R[a:b], wt[a:b], xc[a:b]
        vr = np.isfinite(r)
        r0 = np.where(vr, r, 0.0)
        for t in range(1, n):
            vp = vr[:t]
            cnt = int(vp.sum())
            if cnt == 0:
                continue
            rp = r0[:t]
            rbar = fsum(rp.tolist()) / cnt
            for tag, z, bw in (("lev", w, bw_kg), ("chg", c, bw_chg)):
                dz = z[t] - z[:t]
                ok = vp & np.isfinite(dz)
                K = np.where(ok, np.exp(-0.5 * (np.where(ok, dz, 0.0) / bw) ** 2), 0.0)
                ks = fsum(K.tolist())
                out[f"adv_{tag}"][a + t] = (fsum((K * rp).tolist()) - rbar * ks) / (ks + k0)
                out[f"ev_{tag}"][a + t] = ks
            okw = vp & np.isfinite(w[:t])
            if okw.any() and np.isfinite(w[t]):
                E = np.where(okw, np.exp(np.where(okw, rp, 0.0) / tau), 0.0)
                es = fsum(E.tolist())
                wbest = fsum((E * np.where(okw, w[:t], 0.0)).tolist()) / es
                out["wbest_gap"][a + t] = w[t] - wbest
                out["wbest_gap_abs"][a + t] = abs(w[t] - wbest)
    return pd.DataFrame({f"gc_kr_{m}_{nm}": v for nm, v in out.items()}, index=d.index)


def _growth_block(d: pd.DataFrame, m: str) -> pd.DataFrame:
    rc = f"R_{m}"
    L5 = fs.lags(d, rc, 5)
    L8 = fs.lags(d, rc, 8)
    prior_n = fs.excl_cum(d, ["horse_id"], rc, True)
    prior_mean = np.where(prior_n > 0, fs.excl_cum(d, ["horse_id"], rc) / np.maximum(prior_n, 1), np.nan)
    ewm = fs.ewm_prior(d, rc, 3.0).to_numpy(float)
    with np.errstate(all="ignore"):
        mx = np.where(np.isfinite(L5).any(1), np.nanmax(np.where(np.isfinite(L5), L5, -np.inf), 1), np.nan)
        mn = np.where(np.isfinite(L5).any(1), np.nanmin(np.where(np.isfinite(L5), L5, np.inf), 1), np.nan)
    t = f"gc_gr_{m}"
    return pd.DataFrame({f"{t}_slope5": fs.slope_last(L5), f"{t}_slope8": fs.slope_last(L8), f"{t}_ewm3": ewm, f"{t}_mean": prior_mean,
                         f"{t}_ewm_minus_mean": ewm - prior_mean, f"{t}_max5": mx, f"{t}_min5": mn, f"{t}_lag1": L5[:, 0]}, index=d.index)


def _wt_slope(d: pd.DataFrame, k: int = 5) -> np.ndarray:
    """直近 k 走（今回を含まない）の体重の傾き（100 日あたり kg）。"""
    W = fs.lags(d, "wt", k)
    T = fs.lags(d, "dnum", k)
    m = np.isfinite(W) & np.isfinite(T)
    cnt = m.sum(1)
    tt = np.where(m, T, 0.0)
    tb = tt.sum(1) / np.maximum(cnt, 1)
    ww = np.where(m, W, 0.0)
    wb = ww.sum(1) / np.maximum(cnt, 1)
    dt = np.where(m, T - tb[:, None], 0.0)
    num = (dt * np.where(m, W - wb[:, None], 0.0)).sum(1)
    den = (dt ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den * 100.0
    return np.where((cnt >= 3) & (den > 0), s, np.nan)


def _pop_block(d: pd.DataFrame, consts: dict, metrics: tuple, with_weight: bool) -> pd.DataFrame:
    """母集団の成長曲線の参照値（馬の履歴を使わない。デビュー戦や履歴の浅い馬にも出る）。with_weight は体重だけの列（指標に依存しない）を含めるか。"""
    pop = consts["pop"]
    cols: dict[str, np.ndarray] = {}
    sex, age = d["sex"].astype(str), d["age_c"].astype(str)
    cb = d["chg_bin"].map(lambda x: str(int(x)) if np.isfinite(x) else "nan")
    lb = d["lev_bin"].map(lambda x: str(int(x)) if np.isfinite(x) else "nan")
    zb = d["sz_bin"].map(lambda x: str(int(x)) if np.isfinite(x) else "nan")
    for m in metrics:
        c = pop["chg"][m]
        cols[f"gc_pop_chg_{m}"] = _lookup([sex + "|" + age + "|" + d["season"] + "|" + zb + "|" + cb, sex + "|" + age + "|" + cb, cb], [c["l1"], c["l2"], c["l3"]])
        c = pop["lev"][m]
        cols[f"gc_pop_lev_{m}"] = _lookup([age + "|" + lb, lb], [c["l2"], c["l3"]])
        c = pop["size"][m]
        cols[f"gc_pop_size_{m}"] = _lookup([sex + "|" + age + "|" + d["season"] + "|" + zb, sex + "|" + age + "|" + zb, zb], [c["l1"], c["l2"], c["l3"]])
    if not with_weight:
        return pd.DataFrame(cols, index=d.index)
    wt = d["wt"].to_numpy(float)
    first = d.groupby("horse_id")["wt"].transform("first")
    ds = (d["dnum"] - d.groupby("horse_id")["dnum"].transform("first")).to_numpy(float)
    has_prior = fs.lag(d, "dnum").notna().to_numpy()
    cols["gc_wt_vs_std"] = wt - d["wt_std_mu"].to_numpy(float)
    cols["gc_wt_z"] = d["wt_z"].to_numpy(float)
    cols["gc_wt_slope100_5"] = _wt_slope(d, 5)
    cols["gc_wt_growth_total"] = np.where(has_prior, wt - first.to_numpy(float), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["gc_wt_growth_per100d"] = np.where(has_prior & (ds > 30), (wt - first.to_numpy(float)) / ds * 100.0, np.nan)
    return pd.DataFrame(cols, index=d.index)


def build_groups(d: pd.DataFrame, consts: dict) -> dict[str, pd.DataFrame]:
    """prepare 済みの d から、群ごとの特徴量表（d と同じ index）を作る。d は fs.load_runs と同じ並び（馬・日付順）であること。"""
    d = add_size(add_residuals(apply_metrics(d, consts), consts), consts)
    parts: dict[str, pd.DataFrame] = {"POP_free": _pop_block(d, consts, MEGU_FREE_METRICS, True), "POP_megu": _pop_block(d, consts, ("meg", "comp"), False)}
    for m in FEATURE_METRICS:
        parts[f"SL_{m}"] = pd.concat([_slope_block(d, m, k, consts) for k in KINDS], axis=1)
        parts[f"KR_{m}"] = _kernel_block(d, m)
        parts[f"GR_{m}"] = _growth_block(d, m)
    for m in ("pt", "comp"):
        parts[f"SL0_{m}"] = pd.concat([_slope_block(d, m, k, consts, cell_prior=False, tag_prefix="gc0") for k in KINDS], axis=1)
    out = {}
    for g, T in parts.items():
        T = T.replace([np.inf, -np.inf], np.nan)
        out[g] = pd.concat([d[KEY].reset_index(drop=True), T.reset_index(drop=True)], axis=1)
    return out


def build_all(runs: pd.DataFrame, megu: pd.DataFrame | None, consts: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    d = prepare(runs, megu)
    if consts is None:
        consts = fit_constants(d)
    return build_groups(d, consts), consts
