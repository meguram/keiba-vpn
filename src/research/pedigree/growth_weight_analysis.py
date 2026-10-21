"""馬体重の増減とめぐ指数の変化の結びつき（A 母集団 / B 馬ごと / C 血統）。ホストで実行する。

  python -m src.research.pedigree.growth_weight_analysis --dir <growth_weight_dataset の出力ディレクトリ>

入力: runs.parquet。出力（同じディレクトリ）: summary.json、horse_link.csv、sire_profile.csv、dam_sire_profile.csv ほか。
2026 年は使わない（データ未完）。指数は megu_index_v3a_abs_robust（斤量 55kg 換算済み、障害・低信頼は除く）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree import growth_weight_stats as S

log = logging.getLogger("growth_weight_analysis")
RNG = np.random.default_rng(20261021)

DW_EDGES = [-10, -4, -1, 2, 5, 10]
DW_ORDER = ["≤-11", "-10〜-5", "-4〜-2", "-1〜1", "2〜4", "5〜9", "≥10"]
DW_REF = "-1〜1"
LV_EDGES = [-12, -6, -2, 3, 7, 13]
LV_ORDER = ["≤-13", "-12〜-7", "-6〜-3", "-2〜2", "3〜6", "7〜12", "≥13"]
LV_REF = "-2〜2"
TRAIN_BORN, TEST_BORN = range(2017, 2020), range(2020, 2022)
TRAIN_END = pd.Timestamp("2023-12-31")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# ── A: 母集団 ─────────────────────────────────────────────────────

def pair_frame(df: pd.DataFrame, rest_lo=7, rest_hi=180, drop_low=True) -> pd.DataFrame:
    d = df[df["dweight"].notna() & df["didx"].notna() & df["stage"].notna() & df["rest_days"].between(rest_lo, rest_hi)
           & df["surface"].isin(["芝", "ダ"]) & df["sex"].isin(["牡", "牝", "セ"])].copy()
    if drop_low:
        d = d[~d["low_conf"].astype(bool) & ~d["prev_low_conf"].eq(True)]
    return d.reset_index(drop=True)


def controls_frame(d: pd.DataFrame, with_prev_idx=True):
    cols = {"log_rest": np.log(d["rest_days"].to_numpy(float)), "dist_km": d["distance"].to_numpy(float) / 1000.0,
            "prev_wt10": d["prev_weight"].to_numpy(float) / 10.0}
    if with_prev_idx:
        pi = d["prev_idx"].to_numpy(float)
        cols["prev_idx"] = pi - np.nanmean(pi)
        cols["prev_idx2"] = (pi - np.nanmean(pi)) ** 2 / 100.0
    cats = {"stage": d["stage"].to_numpy(object), "surface": d["surface"].to_numpy(object), "sex": d["sex"].to_numpy(object)}
    return cols, cats, {"stage": "3歳前半", "surface": "芝", "sex": "牡"}


def fit_bins(d: pd.DataFrame, y: np.ndarray, x: np.ndarray, edges, order, ref, cols, cats, cat_ref) -> dict:
    lab = S.bin_labels(x, edges)
    cats2 = dict(cats)
    cats2["bin"] = lab
    X, names = S.design_with_dummies(cols, cats2, {**cat_ref, "bin": ref})
    r = S.ols_cluster(X, y, d["race_id"].to_numpy())
    out = []
    for b in order:
        key = f"bin={b}"
        n_b = int(np.sum(lab == b))
        if b == ref:
            out.append({"bin": b, "n": n_b, "coef": 0.0, "lo": 0.0, "hi": 0.0, "ref": True})
        elif key in names:
            i = names.index(key)
            out.append({"bin": b, "n": n_b, "coef": r["beta"][i], "lo": r["lo"][i], "hi": r["hi"][i]})
    return {"bins": out, "n": r["n"], "races": r["clusters"]}


def fit_linear(d: pd.DataFrame, y: np.ndarray, x: np.ndarray, cols, cats, cat_ref, quad=False) -> dict:
    c = dict(cols)
    c["x10"] = x / 10.0
    if quad:
        c["x10_sq"] = (x / 10.0) ** 2
    X, names = S.design_with_dummies(c, cats, cat_ref)
    r = S.ols_cluster(X, y, d["race_id"].to_numpy())
    i = names.index("x10")
    out = {"coef": r["beta"][i], "lo": r["lo"][i], "hi": r["hi"][i], "n": r["n"], "races": r["clusters"]}
    if quad:
        j = names.index("x10_sq")
        out.update({"coef_sq": r["beta"][j], "lo_sq": r["lo"][j], "hi_sq": r["hi"][j]})
    return out


def analysis_a(df: pd.DataFrame) -> dict:
    out: dict = {}
    d = pair_frame(df)
    y, x = d["didx"].to_numpy(float), d["dweight"].to_numpy(float)
    cols, cats, cref = controls_frame(d)
    out["n_pairs"] = len(d)
    out["n_horses"] = int(d["horse_id"].nunique())
    out["n_races"] = int(d["race_id"].nunique())
    out["raw_corr"] = float(np.corrcoef(x, y)[0, 1])
    lab_all = S.bin_labels(x, DW_EDGES)
    out["raw_by_bin"] = [{"bin": b, "n": int((lab_all == b).sum()), "mean_didx": float(y[lab_all == b].mean())} for b in DW_ORDER if (lab_all == b).any()]
    out["bins"] = fit_bins(d, y, x, DW_EDGES, DW_ORDER, DW_REF, cols, cats, cref)
    out["linear"] = fit_linear(d, y, x, cols, cats, cref)
    out["quad"] = fit_linear(d, y, x, cols, cats, cref, quad=True)
    # 部分集団（10kg あたりの傾き）
    subs = {}
    for name, m in {
        **{f"段階={s}": (d["stage"] == s).to_numpy() for s in S.STAGES},
        **{f"性={s}": (d["sex"] == s).to_numpy() for s in ["牡", "牝", "セ"]},
        **{f"面={s}": (d["surface"] == s).to_numpy() for s in ["芝", "ダ"]},
        "休養≤35日": (d["rest_days"] <= 35).to_numpy(), "休養36〜180日": (d["rest_days"] > 35).to_numpy(),
        "前走の指数が下位1/3": (d["prev_idx"] <= d["prev_idx"].quantile(1 / 3)).to_numpy(),
        "前走の指数が上位1/3": (d["prev_idx"] >= d["prev_idx"].quantile(2 / 3)).to_numpy(),
        "前走体重 ≤460kg": (d["prev_weight"] <= 460).to_numpy(), "前走体重 ≥500kg": (d["prev_weight"] >= 500).to_numpy(),
    }.items():
        if m.sum() < 2000:
            continue
        dd = d[m].reset_index(drop=True)
        c2, k2, r2 = controls_frame(dd)
        subs[name] = fit_linear(dd, dd["didx"].to_numpy(float), dd["dweight"].to_numpy(float), c2, k2, r2)
    out["subgroups"] = subs
    # 感度: 休養条件・低信頼を含める
    for name, kw in {"休養7〜60日": dict(rest_hi=60), "休養7〜365日": dict(rest_hi=365), "低信頼を含む": dict(drop_low=False)}.items():
        dd = pair_frame(df, **kw)
        c2, k2, r2 = controls_frame(dd)
        out.setdefault("sensitivity", {})[name] = fit_linear(dd, dd["didx"].to_numpy(float), dd["dweight"].to_numpy(float), c2, k2, r2)
    # 2: 自分の過去平均からのずれ（適正体重付近が良いか）
    e = df[df["idx"].notna() & df["weight"].notna() & (df["prior_n_weight"] >= 3) & (df["prior_n_idx"] >= 3) & df["stage"].notna()
           & df["rest_days"].between(7, 180) & df["surface"].isin(["芝", "ダ"]) & df["sex"].isin(["牡", "牝", "セ"]) & ~df["low_conf"].astype(bool)].reset_index(drop=True)
    ye = (e["idx"] - e["prior_mean_idx"]).to_numpy(float)
    xe = (e["weight"] - e["prior_mean_weight"]).to_numpy(float)
    cols_e = {"log_rest": np.log(e["rest_days"].to_numpy(float)), "dist_km": e["distance"].to_numpy(float) / 1000.0,
              "prior_idx": e["prior_mean_idx"].to_numpy(float) - e["prior_mean_idx"].mean(), "prior_wt10": e["prior_mean_weight"].to_numpy(float) / 10.0}
    cats_e = {"stage": e["stage"].to_numpy(object), "surface": e["surface"].to_numpy(object), "sex": e["sex"].to_numpy(object)}
    out["own_mean"] = {"n": len(e), "bins": fit_bins(e, ye, xe, LV_EDGES, LV_ORDER, LV_REF, cols_e, cats_e, cref),
                       "linear": fit_linear(e, ye, xe, cols_e, cats_e, cref), "quad": fit_linear(e, ye, xe, cols_e, cats_e, cref, quad=True)}
    return out


# ── B: 馬ごと ─────────────────────────────────────────────────────

def _spearman_halves(horse_idx, x, y, mode: str, min_n=8) -> tuple[float, int, np.ndarray, np.ndarray]:
    """馬ごとに時系列順の走を 2 分割して傾きを出し、馬のあいだの順位相関を返す。horse_idx は馬ごとに連続・時系列順。"""
    order = np.argsort(horse_idx, kind="mergesort")
    h, x, y = horse_idx[order], x[order], y[order]
    # 馬内の通し番号
    first = np.r_[True, h[1:] != h[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(h)), 0))
    pos = np.arange(len(h)) - start
    cnt = np.bincount(h)[h]
    keep = cnt >= min_n
    h, x, y, pos, cnt = h[keep], x[keep], y[keep], pos[keep], cnt[keep]
    half = (pos % 2) if mode == "oddeven" else (pos >= cnt / 2).astype(int)
    _, hid = np.unique(h, return_inverse=True)
    gid = hid * 2 + half
    r = S.group_slopes(gid, x, y, min_n=3)
    a, b = r["slope"][0::2], r["slope"][1::2]
    return S.spearman(a, b), int(len(a)), a, b


def analysis_b(df: pd.DataFrame, out_dir: Path) -> dict:
    out: dict = {}
    d = pair_frame(df).sort_values(["horse_id", "date"]).reset_index(drop=True)
    cols, cats, cref = controls_frame(d)
    # 体重・指数それぞれから、段階・休養・距離・前走水準などの母集団の傾向を引いた残差
    X, _ = S.design_with_dummies(cols, cats, cref)
    ry = S.ols_cluster(X, d["didx"].to_numpy(float), d["race_id"].to_numpy())["resid"]
    c2, k2, r2 = controls_frame(d, with_prev_idx=False)
    X2, _ = S.design_with_dummies({**c2, "prev_idx": cols["prev_idx"]}, k2, r2)
    rx = S.ols_cluster(X2, d["dweight"].to_numpy(float), d["race_id"].to_numpy())["resid"]
    _, hidx = np.unique(d["horse_id"].to_numpy(), return_inverse=True)
    cnt = np.bincount(hidx)
    out["pairs_per_horse"] = {f">={k}": int((cnt >= k).sum()) for k in (4, 6, 8, 10, 12)}
    # (1) 変化×変化: 馬ごとの傾き（指数の変化 ÷ 体重の変化）。10kg あたりの指数ポイントで出す
    min_pairs = 6
    r = S.group_slopes(hidx, rx / 10.0, ry, min_n=min_pairs)
    ids = d["horse_id"].to_numpy()[np.r_[0, np.flatnonzero(np.diff(hidx)) + 1]]
    sl, se = r["slope"], r["se"]
    eb = S.eb_shrink(sl, se)
    ok = ~np.isnan(sl)
    out["delta"] = {"min_pairs": min_pairs, "n_horses": int(ok.sum()), "mu": eb["mu"], "tau2": eb["tau2"], "tau": float(np.sqrt(eb["tau2"])) if eb["tau2"] == eb["tau2"] else None,
                    "mean_se": float(np.nanmean(se)), "mean_weight": float(np.nanmean(eb["weight"])), "q": eb.get("q"), "k": eb["k"],
                    "slope_sd_raw": float(np.nanstd(sl))}
    pd.DataFrame({"horse_id": ids[ok], "n_pairs": r["n"][ok], "slope_per10kg": sl[ok], "se": se[ok], "shrunk": eb["shrunk"][ok], "reliability": eb["weight"][ok]}).to_csv(out_dir / "horse_link.csv", index=False)
    # 再現性: 奇数番目の走と偶数番目の走、前半と後半
    rel = {}
    for mode in ("oddeven", "firstsecond"):
        obs, n_h, a, b = _spearman_halves(hidx, rx / 10.0, ry, mode)
        # 帰無: 馬の中で体重の残差の順序をランダムに入れ替える
        null = []
        for _ in range(100):
            key = RNG.random(len(rx))
            perm = np.lexsort((key, hidx))                    # 馬ごとにランダム順
            srt = np.argsort(hidx, kind="mergesort")
            xp = np.empty_like(rx)
            xp[srt] = rx[perm]
            null.append(_spearman_halves(hidx, xp / 10.0, ry, mode)[0])
        null = np.array(null)
        rel[mode] = {"spearman": obs, "n_horses": n_h, "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
                     "null_p95": float(np.nanpercentile(null, 95)), "z": float((obs - np.nanmean(null)) / (np.nanstd(null) or np.nan))}
    out["delta"]["reliability"] = rel
    # (2) 水準×水準: 馬ごとに、母集団の（段階×性）の平均を引いた体重と指数の傾き
    e = df[df["idx"].notna() & df["weight"].notna() & df["stage"].notna() & df["sex"].isin(["牡", "牝", "セ"]) & ~df["low_conf"].astype(bool)].sort_values(["horse_id", "date"]).reset_index(drop=True)
    e["w_res"] = e["weight"] - e.groupby(["stage", "sex"])["weight"].transform("mean")
    e["i_res"] = e["idx"] - e.groupby(["stage", "sex", "surface"])["idx"].transform("mean")
    _, h2 = np.unique(e["horse_id"].to_numpy(), return_inverse=True)
    c2 = np.bincount(h2)
    rr = S.group_slopes(h2, e["w_res"].to_numpy() / 10.0, e["i_res"].to_numpy(), min_n=8)
    eb2 = S.eb_shrink(rr["slope"], rr["se"])
    rel2 = {}
    for mode in ("oddeven", "firstsecond"):
        obs, n_h, _, _ = _spearman_halves(h2, e["w_res"].to_numpy() / 10.0, e["i_res"].to_numpy(), mode)
        null = []
        for _ in range(100):
            key = RNG.random(len(e))
            perm = np.lexsort((key, h2))
            srt = np.argsort(h2, kind="mergesort")
            xp = np.empty(len(e))
            xp[srt] = e["w_res"].to_numpy()[perm]
            null.append(_spearman_halves(h2, xp / 10.0, e["i_res"].to_numpy(), mode)[0])
        null = np.array(null)
        rel2[mode] = {"spearman": obs, "n_horses": n_h, "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)), "z": float((obs - np.nanmean(null)) / (np.nanstd(null) or np.nan))}
    out["level"] = {"min_runs": 8, "n_horses": int((~np.isnan(rr["slope"])).sum()), "mu": eb2["mu"], "tau2": eb2["tau2"], "mean_se": float(np.nanmean(rr["se"])),
                    "mean_weight": float(np.nanmean(eb2["weight"])), "slope_sd_raw": float(np.nanstd(rr["slope"])), "reliability": rel2}
    return out


# ── C: 血統 ───────────────────────────────────────────────────────

def horse_metrics(df: pd.DataFrame, born, max_date) -> pd.DataFrame:
    """馬ごとの成長指標。2〜3 歳の平均 → 4 歳以上の平均（指数・体重）と、段階ごとの平均。"""
    d = df[df["age"].notna() & (df["date"] <= max_date) & df["sex"].isin(["牡", "牝", "セ"])].copy()
    d["born"] = d["horse_id"].str[:4].astype(int)
    d = d[d["born"].isin(list(born))]
    d.loc[d["low_conf"].astype(bool), "idx"] = np.nan
    d["early"] = d["age"].isin([2, 3])
    d["late"] = d["age"] >= 4
    rows = []
    g = d.groupby("horse_id", sort=False)
    stage_idx = d.groupby(["horse_id", "stage"])["idx"].mean().unstack().reindex(columns=S.STAGES)
    stage_wt = d.groupby(["horse_id", "stage"])["weight"].mean().unstack().reindex(columns=S.STAGES)
    stage_n = d.groupby(["horse_id", "stage"])["weight"].count().unstack().reindex(columns=S.STAGES).fillna(0)
    base = g.agg(sex=("sex", "first"), sire_id=("sire_id", "first"), dam_sire_id=("dam_sire_id", "first"), sire_name=("sire_name", "first"), dam_sire_name=("dam_sire_name", "first"), born=("born", "first"))
    for c in ["idx", "weight"]:
        for part in ("early", "late"):
            m = d[d[part] & d[c].notna()].groupby("horse_id")[c].agg(["mean", "count"])
            base[f"{c}_{part}"] = m["mean"]
            base[f"n_{c}_{part}"] = m["count"]
    base["late_minus_early_idx"] = (base["idx_late"] - base["idx_early"]).where((base["n_idx_early"] >= 2) & (base["n_idx_late"] >= 2))
    base["late_minus_early_wt"] = (base["weight_late"] - base["weight_early"]).where((base["n_weight_early"] >= 2) & (base["n_weight_late"] >= 2))
    prev = None
    for s in S.STAGES:
        if prev is not None:
            ok = (stage_n[s] >= 1) & (stage_n[prev] >= 1)
            base[f"dwt_{prev}→{s}"] = (stage_wt[s] - stage_wt[prev]).where(ok)
            base[f"didx_{prev}→{s}"] = (stage_idx[s] - stage_idx[prev])
        prev = s
    base = base.reset_index()
    # 性別の平均を引く（去勢馬・牝馬は伸び方が違うため）
    metric_cols = [c for c in base.columns if c.startswith(("late_minus", "dwt_", "didx_"))]
    for c in metric_cols:
        base[c + "_dev"] = base[c] - base.groupby("sex")[c].transform("mean")
    return base


def sire_table(h: pd.DataFrame, key: str, name_col: str, metric: str, min_n=3) -> pd.DataFrame:
    x = h[h[key].astype(str).ne("") & h[key].notna() & h[metric + "_dev"].notna()]
    sd_pop = float(x[metric + "_dev"].std())
    g = x.groupby(key)[metric + "_dev"].agg(["mean", "count"])
    g = g[g["count"] >= min_n].copy()
    g["se"] = sd_pop / np.sqrt(g["count"])
    eb = S.eb_shrink(g["mean"].to_numpy(), g["se"].to_numpy())
    g["shrunk"] = eb["shrunk"]
    g["reliability"] = eb["weight"]
    g.attrs.update({"mu": eb["mu"], "tau2": eb["tau2"], "sd_pop": sd_pop})
    nm = x.groupby(key)[name_col].agg(lambda s: next((v for v in s if v), ""))
    g["name"] = nm.reindex(g.index)
    return g.rename(columns={"mean": "raw", "count": "n"})


def evaluate_sire(train: pd.DataFrame, test: pd.DataFrame, key: str, name_col: str, metric_train: str, metric_test: str, min_train=10, min_test=5) -> dict:
    tr = sire_table(train, key, name_col, metric_train, min_n=3)
    trf = tr[tr["n"] >= min_train]
    te = test[test[key].astype(str).ne("") & test[key].notna() & test[metric_test + "_dev"].notna()]
    te = te[te[key].isin(trf.index)].copy()
    te["score"] = te[key].map(trf["shrunk"])
    te["score_raw"] = te[key].map(trf["raw"])
    per_sire = te.groupby(key).agg(test_mean=(metric_test + "_dev", "mean"), n=(metric_test + "_dev", "size"), score=("score", "first"), score_raw=("score_raw", "first"))
    per_sire = per_sire[per_sire["n"] >= min_test]
    out = {"metric_train": metric_train, "metric_test": metric_test, "n_sires_train": int(len(trf)), "n_sires_eval": int(len(per_sire)), "n_test_horses": int(len(te)),
           "tau2_train": tr.attrs.get("tau2"), "sd_pop_train": tr.attrs.get("sd_pop")}
    if len(per_sire) < 8:
        return out
    out["spearman_shrunk"] = S.spearman(per_sire["score"], per_sire["test_mean"])
    out["spearman_raw"] = S.spearman(per_sire["score_raw"], per_sire["test_mean"])
    # 馬単位の回帰（切片あり、種牡馬クラスタの頑健 SE）
    X = np.column_stack([np.ones(len(te)), te["score"].to_numpy(float)])
    r = S.ols_cluster(X, te[metric_test + "_dev"].to_numpy(float), te[key].to_numpy())
    out["horse_regression"] = {"slope": r["beta"][1], "lo": r["lo"][1], "hi": r["hi"][1], "n": r["n"], "sires": r["clusters"]}
    # 上位/下位 20% の差（種牡馬を単位にしたブートストラップ）
    q_lo, q_hi = te["score"].quantile(0.2), te["score"].quantile(0.8)
    hi_m, lo_m = te[te["score"] >= q_hi][metric_test + "_dev"], te[te["score"] <= q_lo][metric_test + "_dev"]
    sires = te[key].unique()
    boots = []
    grp = {s: g for s, g in te.groupby(key)}
    for _ in range(300):
        pick = RNG.choice(sires, len(sires))
        b = pd.concat([grp[s] for s in pick])
        a1, a2 = b[b["score"] >= q_hi][metric_test + "_dev"], b[b["score"] <= q_lo][metric_test + "_dev"]
        if len(a1) and len(a2):
            boots.append(a1.mean() - a2.mean())
    out["top_vs_bottom20"] = {"diff": float(hi_m.mean() - lo_m.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                              "n_top": int(len(hi_m)), "n_bottom": int(len(lo_m)), "score_top_mean": float(te[te["score"] >= q_hi]["score"].mean()), "score_bottom_mean": float(te[te["score"] <= q_lo]["score"].mean())}
    return out


def analysis_c(df: pd.DataFrame, out_dir: Path) -> dict:
    out: dict = {}
    train = horse_metrics(df, TRAIN_BORN, TRAIN_END)
    test = horse_metrics(df, TEST_BORN, pd.Timestamp("2025-12-31"))
    out["n_train_horses"] = {"all": int(len(train)), "idx_metric": int(train["late_minus_early_idx"].notna().sum()), "wt_metric": int(train["late_minus_early_wt"].notna().sum())}
    out["n_test_horses"] = {"all": int(len(test)), "idx_metric": int(test["late_minus_early_idx"].notna().sum()), "wt_metric": int(test["late_minus_early_wt"].notna().sum())}
    out["pop_means"] = {"train_late_minus_early_idx": float(train["late_minus_early_idx"].mean()), "train_late_minus_early_wt": float(train["late_minus_early_wt"].mean()),
                        "test_late_minus_early_idx": float(test["late_minus_early_idx"].mean()), "test_late_minus_early_wt": float(test["late_minus_early_wt"].mean())}
    # 馬の中で『体重の伸び』と『指数の伸び』は結びつくか
    both = train.dropna(subset=["late_minus_early_idx", "late_minus_early_wt"])
    out["horse_level_corr_train"] = {"n": int(len(both)), "pearson": float(both[["late_minus_early_idx", "late_minus_early_wt"]].corr().iloc[0, 1])}
    ev = {}
    for key, nc, label in (("sire_id", "sire_name", "sire"), ("dam_sire_id", "dam_sire_name", "dam_sire")):
        for mt, ms in (("late_minus_early_idx", "late_minus_early_idx"), ("late_minus_early_wt", "late_minus_early_wt"),
                       ("late_minus_early_idx", "late_minus_early_wt"), ("late_minus_early_wt", "late_minus_early_idx")):
            for mn in (10, 20):
                ev[f"{label}:{mt}->{ms}:min{mn}"] = evaluate_sire(train, test, key, nc, mt, ms, min_train=mn)
    out["eval"] = ev
    # 公開用: 全データ（生まれ 2017〜2021・〜2025 年）の種牡馬プロファイル
    allh = horse_metrics(df, range(2017, 2022), pd.Timestamp("2025-12-31"))
    for key, nc, fname in (("sire_id", "sire_name", "sire_profile.csv"), ("dam_sire_id", "dam_sire_name", "dam_sire_profile.csv")):
        tabs = {}
        for m in ["late_minus_early_idx", "late_minus_early_wt"] + [c for c in allh.columns if c.startswith(("dwt_", "didx_")) and not c.endswith("_dev")]:
            allh[m + "_dev"] = allh[m + "_dev"] if m + "_dev" in allh.columns else allh[m] - allh.groupby("sex")[m].transform("mean")
            t = sire_table(allh, key, nc, m, min_n=3)
            tabs[m] = t
        base = tabs["late_minus_early_idx"][["name", "n", "raw", "se", "shrunk", "reliability"]].rename(columns={"n": "n_idx", "raw": "banse_raw", "se": "banse_se", "shrunk": "banse_shrunk", "reliability": "banse_rel"})
        wt = tabs["late_minus_early_wt"][["n", "raw", "shrunk"]].rename(columns={"n": "n_wt", "raw": "wt_gain_raw", "shrunk": "wt_gain_shrunk"})
        prof = base.join(wt, how="outer")
        for m, t in tabs.items():
            if m.startswith(("dwt_", "didx_")):
                prof[m + "_n"] = t["n"]
                prof[m + "_shrunk"] = t["shrunk"]
        prof.index.name = key
        prof = prof[(prof["n_idx"].fillna(0) >= 8) | (prof["n_wt"].fillna(0) >= 8)]
        prof.sort_values("banse_shrunk", ascending=False).to_csv(out_dir / fname)
        out[f"profile_{key}"] = {"n_rows": int(len(prof)), "tau2_banse": tabs["late_minus_early_idx"].attrs.get("tau2"), "tau2_wt": tabs["late_minus_early_wt"].attrs.get("tau2"),
                                 "sd_pop_banse": tabs["late_minus_early_idx"].attrs.get("sd_pop"), "sd_pop_wt": tabs["late_minus_early_wt"].attrs.get("sd_pop")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--parts", default="ABC")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = Path(a.dir)
    df = pd.read_parquet(out_dir / "runs.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["year"] <= 2025].copy()
    df["surface"] = df["surface"].map(lambda s: "ダ" if isinstance(s, str) and s.startswith("ダ") else s)
    res: dict = {"rows": int(len(df)), "horses": int(df["horse_id"].nunique())}
    for part, fn in (("A", lambda: analysis_a(df)), ("B", lambda: analysis_b(df, out_dir)), ("C", lambda: analysis_c(df, out_dir))):
        if part in a.parts:
            log.info("部 %s 開始", part)
            res[part] = fn()
            (out_dir / "summary.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("部 %s 完了", part)


if __name__ == "__main__":
    main()
