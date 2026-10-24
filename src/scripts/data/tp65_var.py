"""tp-65: めぐ指数（v3a 相対）と netkeiba タイム指数の分散・外れ値の扱い・予測力の比較。ホストで実行。

python -m src.scripts.data.tp65_var dist   # 分布・条件別の分散・test-retest・レース水準・大差負けの圧縮・改訂・年月の水準 → dist.json
python -m src.scripts.data.tp65_var pred   # 次走予測力（条件付きロジット）と、めぐ指数のロバスト化の効果 → pred.json
入力: <TP65_OUT>/runs.parquet（tp65_prep が作る。1 行 = 1 走）。2026 年は観察のみ（選択・当てはめに使わない）。
指数 0 は欠損。障害は除く。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("TP65_OUT", "/home/jovyan/work/tp65_out"))
MEGU = "megu_index_v3a"
NK = "nk"
IDX = {"megu": MEGU, "megu_clip": "megu_index_v3a_robust", "megu_abs": "megu_index_v3a_abs_robust", "nk": NK}
QS = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def band_distance(d):
    d = pd.to_numeric(d, errors="coerce")
    return pd.cut(d, [0, 1300, 1700, 2100, 2500, 10**6], labels=["~1200", "1400-1600", "1800-2000", "2200-2400", "2500~"])


def class_bucket(race_class, grade) -> pd.Series:
    rc = race_class.fillna("").astype(str)
    gr = grade.fillna("").astype(str)
    out = pd.Series("他", index=rc.index)
    out[rc.str.contains("新馬")] = "新馬"
    out[rc.str.contains("未勝利")] = "未勝利"
    out[rc.str.contains("1勝|500万")] = "1勝"
    out[rc.str.contains("2勝|1000万")] = "2勝"
    out[rc.str.contains("3勝|1600万")] = "3勝"
    out[rc.str.contains("オープン|OP|リステッド|L")] = "OP"
    out[gr.str.contains("G1|G2|G3|GI|重賞", regex=True)] = "重賞"
    return out


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "runs.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["dband"] = band_distance(d["distance"])
    d["cls"] = class_bucket(d["race_class"], d["grade"])
    d["fp"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    d.loc[d["time_sec"] <= 0, "time_sec"] = np.nan
    d["loss_sec"] = d["time_sec"] - d.groupby("race_id")["time_sec"].transform("min")
    d["tc"] = d["track_condition"].map({"良": "良", "稍重": "稍重", "重": "重", "不良": "不良"}).fillna("他")
    steeple = d["megu_v3a_status"].astype(str).str.contains("steeple", na=False)
    d = d[~steeple].copy()
    d = d[d["surface"].isin(["芝", "ダート"])].copy()
    return d.reset_index(drop=True)


def mad_sd(x: pd.Series) -> float:
    x = x.dropna()
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    q = x.quantile(QS)
    vals, cnt = np.unique(x.round(6), return_counts=True)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()), "mad_sd": mad_sd(x), "sd_over_mad": float(x.std() / max(mad_sd(x), 1e-9)),
            "skew": float(x.skew()), "ex_kurt": float(x.kurt()), "min": float(x.min()), "max": float(x.max()),
            "q": {str(k): float(v) for k, v in q.items()},
            "share_min": float((x == x.min()).mean()), "share_max": float((x == x.max()).mean()),
            "top_value_share": float(cnt.max() / len(x)), "top_value": float(vals[cnt.argmax()]),
            "tail_ratio_up": float((q[0.999] - q[0.5]) / max(q[0.5] - q[0.001], 1e-9)),
            "gt_4mad": float((np.abs(x - x.median()) > 4 * mad_sd(x)).mean())}


def icc_oneway(x: pd.Series, g: pd.Series) -> dict:
    gr = x.groupby(g)
    n_i, m_i = gr.size(), gr.mean()
    N, K = n_i.sum(), len(n_i)
    ssb = (n_i * (m_i - x.mean()) ** 2).sum()
    ssw = ((x - gr.transform("mean")) ** 2).sum()
    msb, msw = ssb / (K - 1), ssw / (N - K)
    n0 = (N - (n_i ** 2).sum() / N) / (K - 1)
    var_b = max((msb - msw) / n0, 0.0)
    return {"horses": int(K), "within_sd": float(np.sqrt(msw)), "between_sd": float(np.sqrt(var_b)), "icc1": float(var_b / (var_b + msw))}


def cmd_dist(a) -> None:
    d = load()
    res: dict = {}
    both = d[d[MEGU].notna() & d[NK].notna() & d["year"].between(2020, 2025)].copy()
    res["n_runs_all"] = int(len(d))
    res["n_both_2020_25"] = int(len(both))
    res["cover"] = {k: float(d.loc[d["year"].between(2020, 2025), c].notna().mean()) for k, c in IDX.items()}
    # A. 全体の分布（両方ある走）
    res["A_dist"] = {k: summarize(both[c]) for k, c in IDX.items()}
    med = {k: float(both[c].median()) for k, c in IDX.items()}
    msd = {k: mad_sd(both[c]) for k, c in IDX.items()}
    for k, c in IDX.items():
        both["z_" + k] = (both[c] - med[k]) / msd[k]
    res["A_z_quantiles"] = {k: {str(q): float(both["z_" + k].quantile(q)) for q in QS} for k in IDX}
    ab = both["megu_index_v3a_abs_robust"]
    res["A_corr"] = {"pearson": float(both[MEGU].corr(both[NK])), "spearman": float(both[MEGU].corr(both[NK], method="spearman")),
                     "slope_nk_on_megu": float(np.polyfit(both[MEGU], both[NK], 1)[0]),
                     "abs_pearson": float(ab.corr(both[NK])), "abs_spearman": float(ab.corr(both[NK], method="spearman")),
                     "abs_slope_nk_on_megu": float(np.polyfit(ab.fillna(ab.median()), both[NK], 1)[0])}
    # B. 条件別
    sub = {}
    for gcol in ("surface", "dband", "cls", "tc"):
        g = {}
        for lv, x in both.groupby(gcol, observed=True):
            if len(x) < 500:
                continue
            g[str(lv)] = {"n": int(len(x)), "sd_megu": float(x[MEGU].std()), "sd_nk": float(x[NK].std()), "mad_megu": mad_sd(x[MEGU]), "mad_nk": mad_sd(x[NK]),
                          "corr": float(x[MEGU].corr(x[NK])), "med_megu": float(x[MEGU].median()), "med_nk": float(x[NK].median())}
        sub[gcol] = g
    res["B_by_group"] = sub
    # C. test-retest: 同一馬の連続する走（両方ある走だけ、間隔 ≤ 100 日）
    cellmed = both.groupby(["surface", "dband", "cls", "year"], observed=True)[NK].transform("median")
    both["nk_rel"] = both[NK] - cellmed
    both["megu_abs"] = both["megu_index_v3a_abs_robust"]
    b = both.sort_values(["horse_id", "date"]).copy()
    for c in (MEGU, NK, "nk_rel", "megu_abs", "surface", "distance", "tc", "date"):
        b["p_" + c] = b.groupby("horse_id")[c].shift(1)
    b["gap"] = (b["date"] - b["p_date"]).dt.days
    pair = b[b["gap"].between(1, 100)].copy()
    cond = (pair["surface"] == pair["p_surface"]) & ((pair["distance"] - pair["p_distance"]).abs() <= 200) & (pair["tc"].isin(["良", "稍重"]) == pair["p_tc"].isin(["良", "稍重"]))
    rt = {}
    for nm, p in (("全連続走", pair), ("同条件", pair[cond])):
        dm, dn = p[MEGU] - p["p_" + MEGU], p[NK] - p["p_" + NK]
        rt[nm] = {"n": int(len(p)), "diff_sd_megu": float(dm.std()), "diff_sd_nk": float(dn.std()), "diff_mad_megu": mad_sd(dm), "diff_mad_nk": mad_sd(dn),
                  "r_megu": float(p[MEGU].corr(p["p_" + MEGU])), "r_nk": float(p[NK].corr(p["p_" + NK])),
                  "rs_megu": float(p[MEGU].corr(p["p_" + MEGU], method="spearman")), "rs_nk": float(p[NK].corr(p["p_" + NK], method="spearman")),
                  "diff_gt20_megu": float((dm.abs() > 20).mean()), "diff_gt20_nk": float((dn.abs() > 20).mean())}
    res["C_test_retest"] = rt
    # 信頼性（ICC(1): 馬ごとの一元配置分散分析。4 走以上の馬）。めぐ指数（相対）はクラス基準で 50 に揃うので、公平に比べるために
    # めぐ指数の絶対版（megu_abs）と、netkeiba をクラスのセル（面×距離帯×クラス×年）の中央値で引いた相対版（nk_rel）も並べる
    cnt = both.groupby("horse_id")[MEGU].transform("size")
    h4 = both[cnt >= 4]
    icc = {}
    for k, col in (("megu(相対)", MEGU), ("megu(相対,0.5-99.5%クリップ)", None), ("megu_abs(絶対)", "megu_abs"), ("nk(絶対)", NK), ("nk_rel(クラス内相対)", "nk_rel")):
        x = h4[MEGU].clip(h4[MEGU].quantile(0.005), h4[MEGU].quantile(0.995)) if col is None else h4[col]
        icc[k] = icc_oneway(x, h4["horse_id"])
    res["C_icc"] = icc
    fair = {}
    for k, col in (("megu(相対)", MEGU), ("megu_abs(絶対)", "megu_abs"), ("nk(絶対)", NK), ("nk_rel(クラス内相対)", "nk_rel")):
        cur, prev = pair[col].to_numpy(float), pair["p_" + col].to_numpy(float)
        ok = ~(np.isnan(cur) | np.isnan(prev))
        dlt = cur[ok] - prev[ok]
        fair[k] = {"n": int(ok.sum()), "pearson": float(np.corrcoef(cur[ok], prev[ok])[0, 1]), "diff_sd": float(np.std(dlt)),
                   "diff_mad": float(1.4826 * np.median(np.abs(dlt - np.median(dlt)))), "total_sd": float(np.std(cur[ok]))}
    res["C_retest_fair"] = fair
    # D. レース内・勝ち馬の水準の揺れ（同面・距離帯・クラス・年のセル内の、レースごとの勝ち馬指数の sd）
    w = both[both["fp"] == 1].copy()
    cell = w.groupby(["surface", "dband", "cls", "year"], observed=True)
    rows = []
    for key, x in cell:
        if len(x) >= 25:
            rows.append({"n": len(x), "sd_megu": x[MEGU].std(), "sd_nk": x[NK].std(), "mad_megu": mad_sd(x[MEGU]), "mad_nk": mad_sd(x[NK])})
    cw = pd.DataFrame(rows)
    res["D_winner_cell"] = {"cells": int(len(cw)), "mean_sd_megu": float(cw["sd_megu"].mean()), "mean_sd_nk": float(cw["sd_nk"].mean()),
                            "mean_mad_megu": float(cw["mad_megu"].mean()), "mean_mad_nk": float(cw["mad_nk"].mean()),
                            "ratio_sd_to_global_megu": float(cw["sd_megu"].mean() / both[MEGU].std()), "ratio_sd_to_global_nk": float(cw["sd_nk"].mean() / both[NK].std())}
    rmed = both.groupby("race_id").agg(n=(MEGU, "size"), mm=(MEGU, "median"), mn=(NK, "median"))
    rmed = rmed[rmed["n"] >= 8]
    res["D_race_median_sd"] = {"races": int(len(rmed)), "sd_megu": float(rmed["mm"].std()), "sd_nk": float(rmed["mn"].std()), "corr": float(rmed["mm"].corr(rmed["mn"]))}
    # E. 年・月の水準（勝ち馬の中央値）
    lv = w.groupby("year").agg(megu=(MEGU, "median"), nk=(NK, "median"), n=(MEGU, "size")).reset_index()
    res["E_year_level"] = lv.to_dict("records")
    ml = w.groupby("month").agg(megu=(MEGU, "median"), nk=(NK, "median")).reset_index()
    res["E_month_range"] = {"megu": float(ml["megu"].max() - ml["megu"].min()), "nk": float(ml["nk"].max() - ml["nk"].min()),
                            "megu_sd": float(ml["megu"].std()), "nk_sd": float(ml["nk"].std())}
    # F. 大差負けでの圧縮: 同レースの勝ち馬との指数差 / 着差（秒）
    win_idx = w.groupby("race_id")[[MEGU, NK]].max().rename(columns={MEGU: "w_megu", NK: "w_nk"})
    x = both.join(win_idx, on="race_id")
    x["dm"] = x[MEGU] - x["w_megu"]
    x["dn"] = x[NK] - x["w_nk"]
    bins = [-0.001, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 1000]
    x["lb"] = pd.cut(x["loss_sec"], bins)
    comp = []
    for lb, g in x.groupby("lb", observed=True):
        if len(g) < 300:
            continue
        mid = float(g["loss_sec"].median())
        comp.append({"loss_bin": str(lb), "n": int(len(g)), "loss_sec_med": mid, "d_megu_med": float(g["dm"].median()), "d_nk_med": float(g["dn"].median()),
                     "pts_per_sec_megu": float(-g["dm"].median() / mid) if mid > 0.05 else None, "pts_per_sec_nk": float(-g["dn"].median() / mid) if mid > 0.05 else None,
                     "d_megu_p10": float(g["dm"].quantile(0.1)), "d_nk_p10": float(g["dn"].quantile(0.1)), "sd_megu": float(g["dm"].std()), "sd_nk": float(g["dn"].std())})
    res["F_loss_compression"] = comp
    # 外れ値の端: 値の集中（上限・下限の飽和）
    for k, c in (("nk", NK), ("megu", MEGU)):
        v = d.loc[d["year"].between(2020, 2025), c].dropna()
        vc = v.round(0).value_counts().head(8)
        res.setdefault("F_top_values", {})[k] = {str(int(i)): float(n / len(v)) for i, n in vc.items()}
        res.setdefault("F_edges", {})[k] = {"min": float(v.min()), "max": float(v.max()), "share_le0": float((v <= 0).mean()), "share_ge100": float((v >= 100).mean()),
                                              "share_ge120": float((v >= 120).mean()), "share_le_m50": float((v <= -50).mean())}
    # 差が大きい走（z 差）の内訳
    both["zdiff"] = both["z_nk"] - both["z_megu"]
    res["F_zdiff"] = {"sd": float(both["zdiff"].std()), "mad": mad_sd(both["zdiff"]), "q": {str(q): float(both["zdiff"].quantile(q)) for q in QS},
                      "share_abs_gt2": float((both["zdiff"].abs() > 2).mean())}
    dg = {}
    both["fpb"] = pd.cut(both["fp"], [0, 1, 3, 5, 9, 18], labels=["1", "2-3", "4-5", "6-9", "10+"])
    for gcol in ("fpb", "cls", "tc"):
        dg[gcol] = {str(k): {"n": int(len(g)), "zdiff_mean": float(g["zdiff"].mean()), "zdiff_sd": float(g["zdiff"].std()), "share_gt2": float((g["zdiff"].abs() > 2).mean())}
                    for k, g in both.groupby(gcol, observed=True) if len(g) >= 300}
    both["lowc"] = both["megu_v3a_low_confidence"].astype(str)
    dg["low_conf"] = {str(k): {"n": int(len(g)), "zdiff_sd": float(g["zdiff"].std()), "share_gt2": float((g["zdiff"].abs() > 2).mean())} for k, g in both.groupby("lowc")}
    res["F_zdiff_groups"] = dg
    # G. netkeiba の値の改訂（同じ走を次走・次々走・3走先の出走前ページで見たときの差）
    rev = {}
    for a_, b_ in (("nk_lag0", "nk_lag1"), ("nk_lag1", "nk_lag2")):
        x2 = d[d[a_].notna() & d[b_].notna() & d["year"].between(2020, 2025)]
        diff = x2[b_] - x2[a_]
        rev[f"{a_}->{b_}"] = {"n": int(len(x2)), "share_equal": float((diff == 0).mean()), "mean_diff": float(diff.mean()), "sd_diff": float(diff.std()),
                              "share_abs_ge3": float((diff.abs() >= 3).mean()), "share_abs_ge10": float((diff.abs() >= 10).mean()),
                              "by_year_mean": {str(y): float(g.mean()) for y, g in diff.groupby(x2["year"])}}
    res["G_revision"] = rev
    (OUT / "dist.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("dist.json saved")


# ── 次走予測 ───────────────────────────────────────────────────────────────
def huber_loc(M: np.ndarray, scale: float, c: float = 1.345, iters: int = 6) -> np.ndarray:
    mu = np.nanmedian(M, axis=1)
    for _ in range(iters):
        r = (M - mu[:, None]) / scale
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        w = np.where(np.isnan(M), 0.0, w)
        den = w.sum(axis=1)
        mu = np.where(den > 0, np.nansum(w * np.nan_to_num(M), axis=1) / np.maximum(den, 1e-9), np.nan)
    return mu


def past_matrix(v: pd.Series, groups: pd.Series, k: int = 5) -> np.ndarray:
    cols = [v.groupby(groups).shift(i).to_numpy() for i in range(1, k + 1)]
    return np.column_stack(cols)


def feature_variants(M: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = 0.65 ** np.arange(M.shape[1])
        ok = ~np.isnan(M)
        ewm = np.where(ok.any(axis=1), (np.nan_to_num(M) * w * ok).sum(axis=1) / np.maximum((w * ok).sum(axis=1), 1e-9), np.nan)
        s = np.sort(np.where(ok, M, np.nan), axis=1)
        n = ok.sum(axis=1)
        trimmed = np.where(n >= 4, (np.nansum(s, axis=1) - np.nanmin(s, axis=1) - np.nanmax(s, axis=1)) / np.maximum(n - 2, 1), np.nanmean(M, axis=1))
        return {"last1": M[:, 0], "mean3": np.nanmean(M[:, :3], axis=1), "mean5": np.nanmean(M, axis=1), "ewm5": ewm, "max5": np.nanmax(M, axis=1),
                "median5": np.nanmedian(M, axis=1), "trimmed5": trimmed, "huber5": huber_loc(M, scale)}


def race_softmax_ll(score: np.ndarray, ridx: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """レースごとの勝ち馬の対数尤度（softmax(beta*score)）を返す（勝ち馬のいるレースだけ。範囲外は NaN）。"""
    df = pd.DataFrame({"r": ridx, "s": beta * score, "y": y})
    mx = df.groupby("r")["s"].transform("max")
    e = np.exp(df["s"] - mx)
    lse = np.log(e.groupby(df["r"]).transform("sum")) + mx
    ll = (df["s"] - lse)[df["y"] == 1]
    return pd.Series(ll.to_numpy(), index=df.loc[df["y"] == 1, "r"].to_numpy())


def fit_beta(score, ridx, y, lo=0.0, hi=1.0) -> float:
    from scipy.optimize import minimize_scalar

    def nll(b):
        return -race_softmax_ll(score, ridx, y, b).mean()

    return float(minimize_scalar(nll, bounds=(lo, hi), method="bounded").x)


def race_spearman(b: pd.DataFrame) -> pd.Series:
    """レースごとの Spearman（スコア大 ↔ 着順小）をベクトル化して返す。b は ridx, s, fp を持つ。"""
    r1 = b.groupby("ridx")["s"].rank()
    r2 = (-b["fp"]).groupby(b["ridx"]).rank()
    a1 = r1 - r1.groupby(b["ridx"]).transform("mean")
    a2 = r2 - r2.groupby(b["ridx"]).transform("mean")
    num = (a1 * a2).groupby(b["ridx"]).sum()
    den = np.sqrt((a1 ** 2).groupby(b["ridx"]).sum() * (a2 ** 2).groupby(b["ridx"]).sum())
    return num / den.replace(0, np.nan)


def build_panel(d: pd.DataFrame) -> pd.DataFrame:
    """両方の指数がある走だけの履歴で、両指数に同じ走を使った as-of 特徴量を作る。"""
    p = d[d[MEGU].notna() & d[NK].notna()].sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    return p


def eval_feature(P: pd.DataFrame, score_col: str, years=(2022, 2023, 2024, 2025)) -> dict:
    """特徴量 1 列の次走予測力: 年ごとに、他の年（2025 以外）で当てはめた β での、勝ち馬の対数尤度の改善（一様に対する）と、レース内の着順との順位相関。"""
    out = {}
    base = P[P[score_col].notna() & P["year"].between(2021, 2025)].copy()
    cnt = base.groupby("race_id")[score_col].transform("size")
    base = base[(cnt >= 6)]
    base = base[base.groupby("race_id")["fp"].transform(lambda s: (s == 1).sum()) == 1].copy()
    base["ridx"] = pd.factorize(base["race_id"])[0]
    base["y"] = (base["fp"] == 1).astype(int)
    base["n_r"] = base.groupby("ridx")["y"].transform("size")
    sc = base[score_col].to_numpy()
    sc = (sc - base.groupby("ridx")[score_col].transform("mean").to_numpy()) / 10.0
    train_pool = [2022, 2023, 2024]
    for y in years:
        tr_years = [t for t in train_pool if t != y]
        tr = (base["year"].isin(tr_years)).to_numpy()
        te = (base["year"] == y).to_numpy()
        beta = fit_beta(sc[tr], base["ridx"].to_numpy()[tr], base["y"].to_numpy()[tr])
        ll = race_softmax_ll(sc[te], base["ridx"].to_numpy()[te], base["y"].to_numpy()[te], beta)
        nr = base.loc[te].groupby("ridx")["n_r"].first().reindex(ll.index)
        gain = ll.to_numpy() + np.log(nr.to_numpy())
        rng = np.random.default_rng(0)
        bs = [rng.choice(gain, len(gain)).mean() for _ in range(300)]
        sp = race_spearman(base.loc[te].assign(s=sc[te]))
        out[str(y)] = {"races": int(len(gain)), "beta": beta, "ll_gain": float(gain.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                       "rank_corr": float(sp.mean())}
    return out


def cmd_pred(a) -> None:
    d = load()
    P = build_panel(d)
    g = P["horse_id"]
    scales = {"megu": mad_sd(P[MEGU]), "nk": mad_sd(P[NK])}
    res: dict = {"panel_rows": int(len(P)), "scales": scales, "features": {}}
    # 変種: ロバスト化（めぐ指数に後付け）
    q_lo, q_hi = P.loc[P["year"].between(2020, 2023), MEGU].quantile([0.005, 0.995])
    cellq = P[P["year"].between(2020, 2023)].groupby(["surface", "dband", "cls"], observed=True)[MEGU].quantile([0.01, 0.995]).unstack()
    cellq.columns = ["lo", "hi"]
    P = P.join(cellq, on=["surface", "dband", "cls"])
    P["megu_win"] = P[MEGU].clip(q_lo, q_hi)
    P["megu_cell"] = P[MEGU].clip(P["lo"].fillna(q_lo), P["hi"].fillna(q_hi))
    win_idx = P[P["fp"] == 1].groupby("race_id")[MEGU].max().rename("w_megu")
    P = P.join(win_idx, on="race_id")
    P["megu_floor"] = np.where(P["w_megu"].notna(), np.maximum(P[MEGU], P["w_megu"] - 40.0), P[MEGU])
    # nk 側にも同じ処理をして比べる
    nq_lo, nq_hi = P.loc[P["year"].between(2020, 2023), NK].quantile([0.005, 0.995])
    P["nk_win"] = P[NK].clip(nq_lo, nq_hi)
    g = P["horse_id"]
    series = {"megu": MEGU, "megu_win": "megu_win", "megu_cell": "megu_cell", "megu_floor": "megu_floor", "megu_clip": "megu_index_v3a_robust", "nk": NK, "nk_win": "nk_win"}
    for nm, col in series.items():
        sc = scales["nk"] if nm.startswith("nk") else scales["megu"]
        M = past_matrix(P[col], g)
        for fv, arr in feature_variants(M, sc).items():
            P[f"f_{nm}__{fv}"] = arr
    cols = [c for c in P.columns if c.startswith("f_")]
    for c in cols:
        res["features"][c[2:]] = eval_feature(P, c)
        _log(f"{c}: {json.dumps({y: round(v['ll_gain'], 4) for y, v in res['features'][c[2:]].items()})}")
    # 2 指標を合わせる（ewm5 同士の 2 特徴ロジット）
    res["combo"] = combo_eval(P, "f_megu__ewm5", "f_nk__ewm5")
    (OUT / "pred.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    keep = ["race_id", "horse_id", "date"] + cols
    P[keep].to_parquet(OUT / "panel_features.parquet", index=False)
    _log("pred.json saved")


def weighted_huber(M: np.ndarray, decay: float, scale: float, c: float = 1.345, iters: int = 6) -> np.ndarray:
    """新しい走ほど重い（decay^k）加重つきの Huber 位置推定。外れ値の影響を有界にしつつ、直近を重視する。"""
    w0 = np.where(np.isnan(M), 0.0, decay ** np.arange(M.shape[1])[None, :])
    X = np.nan_to_num(M)
    den0 = w0.sum(axis=1)
    mu = np.where(den0 > 0, (w0 * X).sum(axis=1) / np.maximum(den0, 1e-9), np.nan)
    for _ in range(iters):
        r = (M - mu[:, None]) / scale
        hw = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        w = np.where(np.isnan(M), 0.0, w0 * hw)
        den = w.sum(axis=1)
        mu = np.where(den > 0, (w * X).sum(axis=1) / np.maximum(den, 1e-9), np.nan)
    return mu


def ewm_decay(M: np.ndarray, decay: float) -> np.ndarray:
    w = np.where(np.isnan(M), 0.0, decay ** np.arange(M.shape[1])[None, :])
    den = w.sum(axis=1)
    return np.where(den > 0, (w * np.nan_to_num(M)).sum(axis=1) / np.maximum(den, 1e-9), np.nan)


def cmd_pred2(a) -> None:
    """ロバスト化の細かい比較: 大差負けの下限（勝ち馬の指数 − m 点）の m、減衰率、加重 Huber。めぐ指数・netkeiba の両方。"""
    d = load()
    P = build_panel(d)
    scales = {"megu": mad_sd(P[MEGU]), "nk": mad_sd(P[NK])}
    g = P["horse_id"]
    res: dict = {"features": {}}
    for nm, col in (("megu", MEGU), ("nk", NK)):
        win = P[P["fp"] == 1].groupby("race_id")[col].max().rename("w")
        Pj = P.join(win, on="race_id")
        for m in (20.0, 30.0, 40.0, 60.0, None):
            x = Pj[col] if m is None else np.where(Pj["w"].notna(), np.maximum(Pj[col], Pj["w"] - m), Pj[col])
            x = pd.Series(np.asarray(x, dtype=float), index=Pj.index)
            M = past_matrix(x, g)
            tag = "raw" if m is None else f"floor{int(m)}"
            for dec in (0.5, 0.65, 0.8):
                P[f"f_{nm}_{tag}__ewm{dec}"] = ewm_decay(M, dec)
            P[f"f_{nm}_{tag}__whuber0.65"] = weighted_huber(M, 0.65, scales[nm])
    cols = [c for c in P.columns if c.startswith("f_")]
    for c in cols:
        res["features"][c[2:]] = eval_feature(P, c)
        _log(f"{c}: {json.dumps({y: round(v['ll_gain'], 4) for y, v in res['features'][c[2:]].items()})}")
    (OUT / "pred2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("pred2.json saved")


def combo_eval(P: pd.DataFrame, c1: str, c2: str) -> dict:
    from scipy.optimize import minimize

    base = P[P[c1].notna() & P[c2].notna() & P["year"].between(2021, 2025)].copy()
    cnt = base.groupby("race_id")["fp"].transform("size")
    base = base[cnt >= 6]
    base = base[base.groupby("race_id")["fp"].transform(lambda s: (s == 1).sum()) == 1].copy()
    base["ridx"] = pd.factorize(base["race_id"])[0]
    base["y"] = (base["fp"] == 1).astype(int)
    X = np.column_stack([(base[c] - base.groupby("ridx")[c].transform("mean")).to_numpy() / 10.0 for c in (c1, c2)])
    r, yv = base["ridx"].to_numpy(), base["y"].to_numpy()
    n_r = base.groupby("ridx")["y"].transform("size").to_numpy()
    out = {}
    for y in (2022, 2023, 2024, 2025):
        tr = base["year"].isin([t for t in (2022, 2023, 2024) if t != y]).to_numpy()
        te = (base["year"] == y).to_numpy()

        def nll(b):
            return -race_softmax_ll(X[tr] @ b, r[tr], yv[tr], 1.0).mean()

        b = minimize(nll, x0=np.array([0.1, 0.1]), method="Nelder-Mead").x
        ll = race_softmax_ll(X[te] @ b, r[te], yv[te], 1.0)
        nr = pd.Series(n_r[te], index=r[te]).groupby(level=0).first().reindex(ll.index)
        gain = ll.to_numpy() + np.log(nr.to_numpy())
        singles, paired = {}, {}
        rng = np.random.default_rng(0)
        for j, c in enumerate((c1, c2)):
            bj = fit_beta(X[tr][:, j], r[tr], yv[tr])
            llj = race_softmax_ll(X[te][:, j], r[te], yv[te], bj)
            gj = llj.reindex(ll.index).to_numpy() + np.log(nr.to_numpy())
            singles[c] = float(gj.mean())
            dif = gain - gj
            bs = [rng.choice(dif, len(dif)).mean() for _ in range(300)]
            paired[c] = {"diff": float(dif.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        out[str(y)] = {"races": int(len(gain)), "coef": [float(v) for v in b], "gain_both": float(gain.mean()), "gain_single": singles, "both_minus_single": paired}
    return out


def cmd_combo(a) -> None:
    d = load()
    P = build_panel(d)
    g = P["horse_id"]
    for nm, col in (("megu", MEGU), ("nk", NK)):
        M = past_matrix(P[col], g)
        P[f"f_{nm}__ewm5"] = feature_variants(M, 10.0)["ewm5"]
    res = combo_eval(P, "f_megu__ewm5", "f_nk__ewm5")
    (OUT / "combo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("combo.json saved")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dist")
    sub.add_parser("pred")
    sub.add_parser("pred2")
    sub.add_parser("combo")
    a = ap.parse_args()
    {"dist": cmd_dist, "pred": cmd_pred, "pred2": cmd_pred2, "combo": cmd_combo}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
