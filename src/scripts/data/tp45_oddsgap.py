"""tp-45: 予測オッズと確定オッズの乖離 g が回収率の差を生むか（軽い集計）。

g = log(p_pred / q)  （p_pred = 予測オッズから作ったレース内で合計 1 の確率、q = 確定オッズから作った同様の確率）。
g が負 = 市場が予測以上に買った（確定オッズが予測より低い）。

  python -m src.scripts.data.tp45_oddsgap fit      # 代理の最終オッズモデルを walk-forward で学習（2022–2025）
  python -m src.scripts.data.tp45_oddsgap analyze  # 回収率・回帰・EV との重なり・reality check・2026（bundle のみ・1 回）

最終オッズモデル（models/final_odds_bundle.json）は 2020–2024 で学習済みなので、確定の予測を持つのは 2025 年と 2026 年だけ。
2022–2025 の walk-forward 用には、同じ考え方（市場オッズ・人気を使わない公開情報から log 確定単勝オッズを予測）の LightGBM を
学習年より後の年にだけ当てはめる代理モデルを作る（V4 の train_table の特徴量）。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "data/local/modeling"
OUT_DIR = MOD / "betting/tp45"
TRAIN_TABLE = MOD / "training_pipeline_v4/features/train_table.parquet"
ODDS = MOD / "betting/odds_win_place.parquet"
FRAME = MOD / "betting/tp36/frame_all.parquet"
EVAL_CACHE = MOD / "final_odds_eval_cache/eval_overlay_2025_2026_oddsfix.parquet"
BUNDLE = ROOT / "models/final_odds_bundle.json"
YEARS = (2022, 2023, 2024, 2025)
DENY = {"finish_position", "race_id", "date", "horse_number", "has_speed_index", "start_time", "round"}
REL_COLS = ["speed_max", "speed_avg", "speed_recent_1", "horse_prior_win_rate", "horse_prior_top3_rate", "jockey_prior_win_rate",
            "trainer_prior_win_rate", "sr_perf_ewm2", "sr_perf_ewm6", "c_res_ewm2", "c_res_ewm6", "d_prev_fin", "b_dist_pred"]
BANDS = [(1.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 30.0), (30.0, 1e9)]


# ------------------------------------------------------------------ 純粋関数（テスト対象）
def norm_prob(race: np.ndarray, w: np.ndarray) -> np.ndarray:
    """レース内で合計 1 に正規化する。"""
    s = pd.Series(w).groupby(race).transform("sum").to_numpy()
    return w / s


def gap_g(race: np.ndarray, odds_final: np.ndarray, odds_pred: np.ndarray) -> np.ndarray:
    """g = log(p_pred / q)。p_pred, q はいずれもレース内で合計 1。負 = 市場が予測以上に買った。"""
    q = norm_prob(race, 1.0 / odds_final)
    p = norm_prob(race, 1.0 / odds_pred)
    return np.log(p / q)


def boot_ratio(stake: np.ndarray, ret: np.ndarray, race: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """レース単位ブートストラップの回収率（ret の和 / stake の和）と 95% 区間。"""
    d = pd.DataFrame({"r": race, "s": stake, "t": ret}).groupby("r")[["s", "t"]].sum()
    s, t = d["s"].to_numpy(), d["t"].to_numpy()
    if s.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    ss, tt = s[idx].sum(1), t[idx].sum(1)
    ok = ss > 0
    r = tt[ok] / ss[ok]
    return float(t.sum() / s.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """切片つき OLS とクラスタ頑健 SE（CR1）。X は切片なしの説明変数行列。"""
    Xc = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    b = XtX_inv @ Xc.T @ y
    e = y - Xc @ b
    codes, uniq = pd.factorize(groups)
    k = Xc.shape[1]
    meat = np.zeros((k, k))
    sc = np.zeros((len(uniq), k))
    np.add.at(sc, codes, Xc * e[:, None])
    meat = sc.T @ sc
    g, n = len(uniq), len(y)
    adj = (g / (g - 1)) * ((n - 1) / (n - k))
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"coef": b.tolist(), "se": se.tolist(), "t": (b / se).tolist(), "n": int(n), "clusters": int(g)}


def reality_check(profit: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """White の reality check。profit は レース × 戦略 の 1 レースあたり損益（賭けなしは 0）。
    統計量は戦略ごとの平均損益の t 値。中心化したレース単位ブートストラップの最大 t と比べる。"""
    R, S = profit.shape
    mu = profit.mean(0)
    sd = profit.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    t = mu / (sd / np.sqrt(R))
    rng = np.random.default_rng(seed)
    cen = profit - mu
    tmax = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, R, size=R)
        m = cen[idx].mean(0)
        s = cen[idx].std(0, ddof=1)
        s = np.where(s > 0, s, np.inf)
        tmax[b] = (m / (s / np.sqrt(R))).max()
    return {"max_t": float(t.max()), "p": float((tmax >= t.max()).mean()), "t": t.tolist()}


def top_overlap(race: np.ndarray, a: np.ndarray, b: np.ndarray, k: int = 3) -> dict:
    """レースごとに a の小さい方から k 頭と b の大きい方から k 頭が重なる割合（平均 Jaccard と 1 位一致率）。"""
    d = pd.DataFrame({"r": race, "a": a, "b": b})
    j, same = [], []
    for _, g in d.groupby("r"):
        if len(g) < k + 1:
            continue
        sa = set(g["a"].nsmallest(k).index)
        sb = set(g["b"].nlargest(k).index)
        j.append(len(sa & sb) / len(sa | sb))
        same.append(g["a"].idxmin() == g["b"].idxmax())
    return {"jaccard_top3": float(np.mean(j)), "top1_same": float(np.mean(same)), "races": len(j)}


# ------------------------------------------------------------------ データ
def load_labels() -> pd.DataFrame:
    w = pd.read_parquet(ODDS, columns=["race_id", "horse_number", "win_odds"])
    w = w[w["win_odds"] >= 1.0]
    inv = (1 / w["win_odds"]).groupby(w["race_id"]).sum()
    ok = set(inv.index[(inv >= 1.0) & (inv <= 1.6)])
    return w[w["race_id"].isin(ok)].copy()


def build_features() -> tuple[pd.DataFrame, list[str]]:
    t = pd.read_parquet(TRAIN_TABLE)
    t["year"] = t["race_id"].astype(str).str[:4].astype(int)
    cols = []
    for c in t.columns:
        if c in DENY or c == "year":
            continue
        dt = t[c].dtype
        if str(dt) == "category":
            if t[c].cat.categories.size <= 60:
                t[c] = t[c].cat.codes.astype("int16")
                cols.append(c)
        elif dt.kind in "fiub":
            cols.append(c)
    race = t["race_id"].astype(str)
    for c in REL_COLS:
        if c in t.columns and t[c].dtype.kind in "fiu":
            g = t.groupby(race)[c]
            t[c + "_rrank"] = g.rank(pct=True)
            t[c + "_rgap"] = t[c] - g.transform("mean")
            cols += [c + "_rrank", c + "_rgap"]
    return t, cols


def fit(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t, cols = build_features()
    lab = load_labels()
    t["race_id"] = t["race_id"].astype(str)
    lab["race_id"] = lab["race_id"].astype(str)
    d = t.merge(lab, on=["race_id", "horse_number"], how="inner")
    d["y"] = np.log(d["win_odds"])
    preds = []
    for y in YEARS:
        tr, te = d[d["year"] < y], d[d["year"] == y]
        m = lgb.LGBMRegressor(n_estimators=args.trees, learning_rate=args.lr, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=0)
        m.fit(tr[cols], tr["y"])
        p = np.maximum(np.exp(m.predict(te[cols])), 1.0)
        preds.append(pd.DataFrame({"race_id": te["race_id"].to_numpy(), "horse_number": te["horse_number"].to_numpy(), "year": y, "pred_odds": p}))
        print(y, "train", len(tr), "test", len(te), "MAE(log)", float(np.abs(np.log(p) - te["y"].to_numpy()).mean()), flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(OUT_DIR / "surrogate_pred.parquet", index=False)
    (OUT_DIR / "surrogate_cols.json").write_text(json.dumps(cols, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ 分析
def pct(x: float) -> float:
    return float(np.round(100 * x, 2))


def roi_table(d: pd.DataFrame, label: np.ndarray, kind: str) -> list[dict]:
    rows = []
    for lv in sorted(set(label)):
        s = d[label == lv]
        ret = s["ret_win"] if kind == "win" else s["ret_place"]
        ok = ret.notna()
        s, ret = s[ok], ret[ok]
        r, lo, hi = boot_ratio(np.full(len(s), 100.0), ret.to_numpy(), s["race_id"].to_numpy())
        rows.append({"bin": str(lv), "n": int(len(s)), "roi": pct(r), "lo": pct(lo), "hi": pct(hi), "win_rate": pct(float(s["win"].mean()))})
    return rows


def strategies() -> list[tuple]:
    out = []
    for side, q in (("low", 0.10), ("low", 0.20), ("low", 0.30), ("high", 0.70), ("high", 0.80), ("high", 0.90)):
        for kind in ("win", "place"):
            for band in ("all", "lt10", "ge10"):
                out.append((side, q, kind, band))
    return out


def select(d: pd.DataFrame, thr: dict, st: tuple) -> np.ndarray:
    side, q, kind, band = st
    cut = thr[(side, q)]
    m = (d["g"] <= cut) if side == "low" else (d["g"] >= cut)
    if band == "lt10":
        m &= d["win_odds"] < 10
    elif band == "ge10":
        m &= d["win_odds"] >= 10
    if kind == "place":
        m &= d["ret_place"].notna()
    return m.to_numpy()


def thresholds(g: np.ndarray) -> dict:
    return {(side, q): float(np.quantile(g, q)) for side, q in (("low", 0.10), ("low", 0.20), ("low", 0.30), ("high", 0.70), ("high", 0.80), ("high", 0.90))}


def strategy_profit(d: pd.DataFrame, thr: dict) -> tuple[np.ndarray, list[dict], list[str]]:
    races = pd.Index(d["race_id"].unique())
    pos = races.get_indexer(d["race_id"])
    mat = np.zeros((len(races), len(strategies())))
    rows = []
    for j, st in enumerate(strategies()):
        m = select(d, thr, st)
        ret = d["ret_win"] if st[2] == "win" else d["ret_place"]
        prof = np.where(m, (ret.fillna(0).to_numpy() - 100.0) / 100.0, 0.0)
        np.add.at(mat[:, j], pos, prof)
        s = d[m]
        r = ret[m]
        rr, lo, hi = boot_ratio(np.full(len(s), 100.0), r.to_numpy(), s["race_id"].to_numpy()) if len(s) else (float("nan"),) * 3
        rows.append({"side": st[0], "q": st[1], "kind": st[2], "band": st[3], "n": int(m.sum()), "roi": pct(rr), "lo": pct(lo), "hi": pct(hi)})
    return mat, rows, [str(st) for st in strategies()]


def load_frame() -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    f["race_id"] = f["race_id"].astype(str)
    return f


def band_of(odds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(odds), dtype=int)
    for k, (lo, hi) in enumerate(BANDS):
        out[(odds >= lo) & (odds < hi)] = k
    return out


def band_means(d: pd.DataFrame, gcol: str = "g_raw") -> dict:
    b = band_of(d["win_odds"].to_numpy(float))
    return {int(k): float(d[gcol].to_numpy()[b == k].mean()) for k in np.unique(b)}


def run_core(d: pd.DataFrame, tag: str, bm: dict | None = None) -> dict:
    """d: race_id, horse_number, year, win_odds, ret_win, ret_place, win, p_win, q_win, g_raw を持つ馬単位の表。"""
    res: dict = {}
    d = d.copy()
    d["q"] = norm_prob(d["race_id"].to_numpy(), 1.0 / d["win_odds"].to_numpy(float))
    d["logo"] = np.log(d["win_odds"])
    d["d_model"] = np.log(d["p_win"] / d["q_win"])
    d["ev"] = d["p_win"] * d["win_odds"]
    bm = bm or band_means(d)
    d["g"] = d["g_raw"] - np.array([bm[int(k)] for k in band_of(d["win_odds"].to_numpy(float))])   # オッズ帯の平均を引いた g（水準の影響を除く）
    res["g_summary"] = {"mean": float(d["g_raw"].mean()), "sd": float(d["g_raw"].std()), "q05": float(d["g_raw"].quantile(.05)), "q95": float(d["g_raw"].quantile(.95)),
                        "band_means": {str(k): round(v, 4) for k, v in bm.items()}}
    labs = ["Q1(市場が予測以上に買った)", "Q2", "Q3", "Q4", "Q5(予測より買われていない)"]
    qb = pd.qcut(d["g_raw"], 5, labels=labs)
    res["quintile_win"] = roi_table(d, qb.to_numpy(), "win")
    res["quintile_place"] = roi_table(d, qb.to_numpy(), "place")
    qb2 = pd.qcut(d["g"], 5, labels=labs)
    res["quintile_band_adjusted_win"] = roi_table(d, qb2.to_numpy(), "win")
    res["quintile_band_adjusted_place"] = roi_table(d, qb2.to_numpy(), "place")
    if d["year"].nunique() > 1:
        res["quintile_by_year_win"] = {str(y): roi_table(s_, qb2[s_.index].to_numpy(), "win") for y, s_ in d.groupby("year")}
    rows = []
    for lo_, hi_ in BANDS:
        s_ = d[(d["win_odds"] >= lo_) & (d["win_odds"] < hi_)].copy()
        s_["tb"] = pd.qcut(s_["g_raw"], 3, labels=["g低(買われた)", "g中", "g高(買われていない)"])
        for kind in ("win", "place"):
            for r in roi_table(s_, s_["tb"].to_numpy(), kind):
                rows.append({"band": f"{lo_:g}–{'' if hi_ > 1e6 else format(hi_, 'g')}", "kind": kind, **r})
    res["band_terciles"] = rows

    def reg(y, cols, name):
        X = d[cols].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        r = cluster_ols(y[ok], X[ok], d["race_id"].to_numpy()[ok])
        res.setdefault("regressions", {})[name] = {"vars": ["const"] + cols, **{k: (np.round(v, 5).tolist() if isinstance(v, list) else v) for k, v in r.items()}}

    ret = (d["ret_win"].to_numpy() - 100) / 100
    exc = d["win"].to_numpy(float) - d["q"].to_numpy()
    reg(ret, ["logo", "g_raw"], "単勝の収益率 ~ log確定オッズ + g")
    reg(exc, ["logo", "g_raw"], "的中−市場確率 ~ log確定オッズ + g")
    reg(exc, ["logo", "g_raw", "d_model"], "的中−市場確率 ~ log確定オッズ + g + log(モデル/市場)")
    reg(ret, ["logo", "g_raw", "d_model"], "単勝の収益率 ~ log確定オッズ + g + log(モデル/市場)")
    reg(exc, ["logo", "d_model"], "的中−市場確率 ~ log確定オッズ + log(モデル/市場)")
    ra, gv, dv = d["race_id"].to_numpy(), d["g_raw"].to_numpy(), d["d_model"].to_numpy()
    res["overlap_with_model_gap"] = {"spearman_g_vs_log_model_over_market": float(d["g_raw"].corr(d["d_model"], method="spearman")),
                                     "top3_g_high_vs_model_high": top_overlap(ra, -gv, dv),     # 予測より買われていない馬 と モデルが市場より高く見る馬
                                     "top3_g_low_vs_model_low": top_overlap(ra, gv, -dv)}       # 予測以上に買われた馬 と モデルが市場より低く見る馬
    d["evb"] = pd.cut(d["ev"], [0, 0.8, 1.0, 1.2, 99], labels=["EV<0.8", "0.8–1.0", "1.0–1.2", "EV≥1.2"])
    ev_rows = []
    for lv, s_ in d.groupby("evb", observed=True):
        s_ = s_.copy()
        s_["tb"] = pd.qcut(s_["g_raw"], 3, labels=["g低", "g中", "g高"])
        for r in roi_table(s_, s_["tb"].to_numpy(), "win"):
            ev_rows.append({"ev_band": str(lv), **r})
    res["ev_band_g_terciles_win"] = ev_rows
    thr = thresholds(d["g"].to_numpy())
    mat, srows, _ = strategy_profit(d, thr)
    ok = [i for i, r in enumerate(srows) if r["n"] >= 150]
    rc = reality_check(mat[:, ok])
    res["strategies"] = srows
    res["reality_check"] = {"tested": len(ok), "of": len(srows), "max_t": rc["max_t"], "p": rc["p"]}
    res["_thr"] = {f"{a}{q}": v for (a, q), v in thr.items()}
    res["_bm"] = bm
    res["_d"] = d
    return res


def analyze(args: argparse.Namespace) -> None:
    from src.scripts.data.eval_final_odds_bundles import predict_bundle, within_race_spearman

    f = load_frame()
    out: dict = {"n_strategies_total": 0}
    d = None
    spf = OUT_DIR / "surrogate_pred.parquet"
    if spf.exists():
        sp = pd.read_parquet(spf)
        sp["race_id"] = sp["race_id"].astype(str)
        d = f.merge(sp, on=["race_id", "horse_number", "year"], how="inner")
        d["g_raw"] = gap_g(d["race_id"].to_numpy(), d["win_odds"].to_numpy(float), d["pred_odds"].to_numpy(float))
        out["surrogate_quality"] = {
            str(y): {"races": int(s_["race_id"].nunique()), "mae_log": float(np.abs(np.log(s_["pred_odds"]) - np.log(s_["win_odds"])).mean()),
                     "spearman_within_race": within_race_spearman(s_["race_id"].to_numpy(), s_["pred_odds"].to_numpy(), s_["win_odds"].to_numpy())}
            for y, s_ in d.groupby("year")}
        core = run_core(d, "surrogate")
        out["surrogate"] = {k: v for k, v in core.items() if not k.startswith("_")}
        out["n_strategies_total"] += core["reality_check"]["tested"]
    else:
        out["surrogate_note"] = "代理モデルの予測が未作成（fit 未実行）。bundle のみ"

    ev = pd.read_parquet(EVAL_CACHE)
    ev["race_id"] = ev["race_id"].astype(str)
    ev["horse_number"] = ev["horse_number"].astype(int)
    pb = predict_bundle(BUNDLE, ev)
    out["bundle_feature_check"] = {"odds_or_popularity_columns": [c for c in json.loads(BUNDLE.read_text(encoding="utf-8"))["feature_cols"] if "odds" in c or "pop" in c],
                                   "note": "prev1_odds_vs_current_ratio は評価行で全て欠損（現在のオッズは入っていない）。過去走の人気差（prev*_pop_rank_diff）は過去の市場情報"}
    ev["pred_odds_bundle"] = pb["win"]
    bd = f.merge(ev[["race_id", "horse_number", "pred_odds_bundle"]], on=["race_id", "horse_number"], how="inner")
    bd["g_raw"] = gap_g(bd["race_id"].to_numpy(), bd["win_odds"].to_numpy(float), bd["pred_odds_bundle"].to_numpy(float))
    b25 = bd[bd["year"] == 2025].copy()
    out["bundle_quality_2025"] = {"races": int(b25["race_id"].nunique()),
                                  "spearman_within_race": within_race_spearman(b25["race_id"].to_numpy(), b25["pred_odds_bundle"].to_numpy(), b25["win_odds"].to_numpy())}
    if d is not None:
        s25 = d[d["year"] == 2025][["race_id", "horse_number", "g_raw"]].rename(columns={"g_raw": "g_surr"})
        cmp_ = b25.merge(s25, on=["race_id", "horse_number"])
        out["corr_bundle_vs_surrogate_g_2025"] = float(cmp_["g_raw"].corr(cmp_["g_surr"], method="spearman"))
    core_b = run_core(b25, "bundle2025")
    out["bundle_2025"] = {k: v for k, v in core_b.items() if not k.startswith("_")}
    out["n_strategies_total"] += core_b["reality_check"]["tested"]
    # 2026 は最後に 1 回だけ。しきい値とオッズ帯の平均は bundle の 2025 から固定する。
    b26 = bd[bd["year"] == 2026].copy()
    if len(b26):
        b26["g"] = b26["g_raw"] - np.array([core_b["_bm"][int(k)] for k in band_of(b26["win_odds"].to_numpy(float))])
        thr_b = {("low" if k.startswith("low") else "high", float(k[3:] if k.startswith("low") else k[4:])): v for k, v in core_b["_thr"].items()}
        _, rows26, _ = strategy_profit(b26, thr_b)
        out["bundle_2026"] = {"races": int(b26["race_id"].nunique()), "strategies": rows26,
                              "quintile_win": roi_table(b26, pd.qcut(b26["g"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).to_numpy(), "win")}
    (OUT_DIR / "gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    brief = {"surrogate_quality": out.get("surrogate_quality"), "bundle_quality_2025": out["bundle_quality_2025"],
             "bundle_2025_reality_check": out["bundle_2025"]["reality_check"], "bundle_2025_overlap": out["bundle_2025"]["overlap_with_model_gap"],
             "n_strategies_total": out["n_strategies_total"]}
    print(json.dumps(brief, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ 章への追記（proven_methods_roi.html の 5 と 6 の間）
CHAPTER = ROOT / "docs/html/modeling/chapters/proven_methods_roi.html"
START, END = "<!-- tp45:start -->", "<!-- tp45:end -->"


def _f(x) -> str:
    return "—" if x is None or x != x else f"{x:.1f}%"


def _roi_cells(rows: list[dict]) -> str:
    return "".join(f"<td class='num'>{_f(r['roi'])}<br><span class='muted'>[{_f(r['lo'])}–{_f(r['hi'])}] {r['n']:,}</span></td>" for r in rows)


def section_html(gap: dict) -> str:
    sg, bd = gap["surrogate"], gap["bundle_2025"]
    sq, bq = gap["surrogate_quality"], gap["bundle_quality_2025"]
    labs = ["Q1 買われた", "Q2", "Q3", "Q4", "Q5 買われていない"]

    def quint(c: dict, key: str) -> str:
        return "<tr><th>g の五分位</th>" + "".join(f"<th>{n}</th>" for n in labs) + "</tr>" + "<tr><td>単勝</td>" + _roi_cells(c[key.replace("X", "win")]) + "</tr>" + \
               "<tr><td>複勝</td>" + _roi_cells(c[key.replace("X", "place")]) + "</tr>"

    def band_tbl(c: dict) -> str:
        out = ["<tr><th>確定単勝オッズ帯</th><th>g 低（買われた）</th><th>g 中</th><th>g 高（買われていない）</th></tr>"]
        for band in dict.fromkeys(x["band"] for x in c["band_terciles"]):
            cells = []
            for lab in ("g低(買われた)", "g中", "g高(買われていない)"):
                r = next(x for x in c["band_terciles"] if x["band"] == band and x["kind"] == "win" and x["bin"] == lab)
                cells.append(f"<td class='num'>{_f(r['roi'])}<br><span class='muted'>[{_f(r['lo'])}–{_f(r['hi'])}] {r['n']:,}</span></td>")
            out.append(f"<tr><td>{band}</td>{''.join(cells)}</tr>")
        return "".join(out)

    def ev_tbl(c: dict) -> str:
        out = ["<tr><th>モデル単体の EV 帯（V4 の勝率×確定オッズ）</th><th>g 低</th><th>g 中</th><th>g 高</th></tr>"]
        for band in dict.fromkeys(x["ev_band"] for x in c["ev_band_g_terciles_win"]):
            cells = []
            for lab in ("g低", "g中", "g高"):
                r = next(x for x in c["ev_band_g_terciles_win"] if x["ev_band"] == band and x["bin"] == lab)
                cells.append(f"<td class='num'>{_f(r['roi'])}<br><span class='muted'>[{_f(r['lo'])}–{_f(r['hi'])}] {r['n']:,}</span></td>")
            out.append(f"<tr><td>{html.escape(band)}</td>{''.join(cells)}</tr>")
        return "".join(out)

    def reg_tbl(c: dict) -> str:
        out = ["<tr><th>回帰（クラスタ=レース）</th><th>説明変数</th><th>係数</th><th>t 値</th></tr>"]
        for name, r in c["regressions"].items():
            for v, b, t in zip(r["vars"][1:], r["coef"][1:], r["t"][1:]):
                out.append(f"<tr><td>{name}</td><td>{v}</td><td class='num'>{b:+.4f}</td><td class='num'>{t:+.1f}</td></tr>")
        return "".join(out)

    ov_s, ov_b = sg["overlap_with_model_gap"], bd["overlap_with_model_gap"]
    s26 = gap["bundle_2026"]
    best26 = max((x for x in s26["strategies"] if x["kind"] == "win" and x["n"] >= 100 and x["roi"] == x["roi"]), key=lambda x: x["roi"])
    rcs, rcb = sg["reality_check"], bd["reality_check"]
    qs = ", ".join(f"{y} 年 {v['spearman_within_race']:.3f}" for y, v in sq.items())

    def evc(c: dict, ev_band: str, g: str) -> dict:
        return next(x for x in c["ev_band_g_terciles_win"] if x["ev_band"] == ev_band and x["bin"] == g)

    s_lo, s_hi, b_lo, b_hi = evc(sg, "EV≥1.2", "g低"), evc(sg, "EV≥1.2", "g高"), evc(bd, "EV≥1.2", "g低"), evc(bd, "EV≥1.2", "g高")
    best_s = max((x for x in sg["strategies"] if x["n"] >= 150 and x["roi"] == x["roi"]), key=lambda x: x["roi"])
    gcoef = {k: dict(zip(v["vars"], zip(v["coef"], v["t"]))) for k, v in sg["regressions"].items()}
    gc1, gc2 = gcoef["的中−市場確率 ~ log確定オッズ + g"]["g_raw"], gcoef["的中−市場確率 ~ log確定オッズ + g + log(モデル/市場)"]["g_raw"]
    return f"""{START}
<h2>5b. 検証 3: 予測オッズと確定オッズの乖離 g（tp-45）</h2>
<p class="muted">問い: 『予測オッズ（公開情報から作る価格）と実際のオッズの乖離がある馬』は、直前のオッズ低下と実質的に同じシグナルか。ここでは過去の 2020〜2025 年で、乖離そのものが回収率の差を生むかを調べる（T-60 のスナップショットは使わない）。</p>
<ul>
 <li><b>g の定義</b>: g = log(p_pred / q)。p_pred は予測オッズ（の逆数）をレース内で合計 1 にした確率、q は確定オッズ（の逆数）を同様にした確率。<b>g が負なら『市場が予測以上に買った』</b>（確定オッズが予測より低い）。</li>
 <li><b>予測オッズ（a）現行の最終オッズ bundle（models/final_odds_bundle.json）</b>: 2020〜2024 年で学習済みなので、前向きの予測を持つのは 2025 年（{bq['races']:,} レース）と 2026 年（{s26['races']:,} レース、最後に 1 回だけ）。特徴量に『現在のオッズ』は入っていない（prev1_odds_vs_current_ratio は評価行で全て欠損）。過去走の人気差（prev*_pop_rank_diff 等、重要度の合計は約 4.5%）は過去の市場情報で、これは入っている。レース内の順位相関 {bq['spearman_within_race']:.3f}。</li>
 <li><b>予測オッズ（b）代理モデル</b>: bundle の学習年が 2024 年までなので 2022〜2024 年に前向きの予測が無い。そこで V4 の train_table（市場オッズ・人気を使わない）の特徴量とレース内の相対化列で log 確定単勝オッズを LightGBM で回帰し、学習年より後の年にだけ当てはめた（walk-forward、2022〜2025 年、{sg['quintile_win'][0]['n'] * 5 // 1:,} 頭、レース内の順位相関: {qs}）。bundle より高いのは V4 の特徴量が強いため。g の相関（2025 年）bundle×代理 = {gap['corr_bundle_vs_surrogate_g_2025']:.2f}。</li>
</ul>
<h3>g の五分位ごとの回収率（代理モデル、2022〜25 年）</h3>
<table>{quint(sg, 'quintile_X')}</table>
<p class="muted">g は確定オッズの水準と強く結びつく（人気薄ほど予測と確定の差が大きい）ため、単調な並びの多くは favorite–longshot bias の水準効果。オッズ帯の平均を引いた g の五分位では、単勝は {', '.join(_f(r['roi']) for r in sg['quintile_band_adjusted_win'])} でほぼ平らになる。</p>
<h3>確定オッズ帯を揃えた g 三分位の単勝回収率</h3>
<table><tr><th colspan="4">代理モデル（2022〜25 年）</th></tr>{band_tbl(sg)}</table>
<table><tr><th colspan="4">bundle（2025 年）</th></tr>{band_tbl(bd)}</table>
<h3>回帰: g は最終オッズ・モデルの見方を揃えても効くか</h3>
<table>{reg_tbl(sg)}</table>
<p class="muted">d_model = log(V4 の勝率 / 市場の勝率)。g と d_model は同じ向き（市場が過小に見ている馬ほど高い）に相関する（順位相関 代理 {ov_s['spearman_g_vs_log_model_over_market']:.2f}、bundle {ov_b['spearman_g_vs_log_model_over_market']:.2f}）が、上位 3 頭の重なりは代理で {ov_s['top3_g_high_vs_model_high']['jaccard_top3']:.2f}（1 位一致 {100 * ov_s['top3_g_high_vs_model_high']['top1_same']:.0f}%）、bundle で {ov_b['top3_g_high_vs_model_high']['jaccard_top3']:.2f}（{100 * ov_b['top3_g_high_vs_model_high']['top1_same']:.0f}%）。</p>
<h3>モデル単体の EV 帯を揃えた g 三分位の単勝回収率（代理モデル）</h3>
<table>{ev_tbl(sg)}</table>
<table><tr><th colspan="4">bundle（2025 年）</th></tr>{ev_tbl(bd)}</table>
<h3>g だけの買い方（オッズ帯の平均を引いた g の分位、36 戦略 × 2 つの予測オッズ）</h3>
<ul>
 <li>代理モデル（2022〜25 年）: 検定 {rcs['tested']} 本、reality check の最大 t = {rcs['max_t']:.2f}、p = {rcs['p']:.2f}。bundle（2025 年）: {rcb['tested']} 本、最大 t = {rcb['max_t']:.2f}、p = {rcb['p']:.2f}。<b>100% 超の区間下限は無し</b>。</li>
 <li>2026 年（bundle、2025 年で決めたしきい値のまま 1 回だけ）: 単勝の最良は {_f(best26['roi'])}［{_f(best26['lo'])}–{_f(best26['hi'])}］（{best26['n']:,} 点）。区間下限が 100% を超えたものは無し。</li>
</ul>
<div class="note"><b>結論（今回分かったこと）.</b>
 <ul>
  <li><b>g 単独は買いのシグナルにならない.</b>オッズ帯を揃えると五分位の回収率はほぼ平ら（代理モデル）で、g だけで戦略を作ると最良でも {_f(best_s['roi'])}［{_f(best_s['lo'])}–{_f(best_s['hi'])}］（{best_s['n']:,} 点）。</li>
  <li><b>ただし g は『当たりやすさ』に情報を持つ.</b>市場確率 q を超えた的中率は、g が低い（市場が予測以上に買った）馬ほど高い（的中−市場確率を g に回帰した係数 {gc1[0]:+.4f}、t = {gc1[1]:+.1f}）。モデルの見方（d_model）を揃えても残る（g と d_model は相関するので係数の大きさは参考、係数 {gc2[0]:+.4f}、t = {gc2[1]:+.1f}）。同じ EV 帯の中では g 低が g 高より高い（EV≥1.2 で 代理モデル {_f(s_lo['roi'])}［{_f(s_lo['lo'])}–{_f(s_lo['hi'])}］ 対 {_f(s_hi['roi'])}［{_f(s_hi['lo'])}–{_f(s_hi['hi'])}］、bundle でも {_f(b_lo['roi'])} 対 {_f(b_hi['roi'])}）。<b>これは 12 セルを見て後から気づいた探索的な結果で、区間下限は 100% に届いていない</b>。2026 年の最終評価には使っていない（新しいデータで確かめる対象）。</li>
  <li><b>直前のオッズ低下との関係.</b>g は『公開情報で説明できる価格』との差で、早い時間の資金も直前の資金も区別できない。論文の主張（最後の 5 分の低下が、最終オッズを揃えても回収率を高める）は、時刻のあるデータ（T-10・T-5・T-2）が貯まるまで検証できない。g が情報を持つこと自体は、『市場には公開情報（V4 を含む）にない情報が入っている』という読みと整合する。<b>分かったのは『乖離は実質的に同じではないが、同じ向きの情報を含む可能性がある』ことまで</b>。</li>
 </ul></div>
<p class="muted">試した戦略: g だけの買い方 36 本 × 2 つの予測オッズ = 72（点数 150 未満を除く）。EV 帯と回帰は記述（検定の対象外）。再現: <code>python -m src.scripts.data.tp45_oddsgap fit</code> / <code>analyze</code> / <code>report</code>（結果は <code>data/local/modeling/betting/tp45/gap.json</code>）。</p>
{END}
"""


def insert_section(text: str, block: str) -> str:
    if START in text:
        a, b = text.index(START), text.index(END) + len(END)
        return text[:a] + block.strip("\n") + text[b:]
    anchor = "<h2>6. 取り入れる価値のある手法と実装案</h2>"
    return text.replace(anchor, block.strip("\n") + "\n" + anchor, 1)


def report(args: argparse.Namespace) -> None:
    gap = json.loads(Path(args.gap).read_text(encoding="utf-8"))
    t = CHAPTER.read_text(encoding="utf-8")
    CHAPTER.write_text(insert_section(t, section_html(gap)), encoding="utf-8")
    print("updated", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "analyze", "report"])
    ap.add_argument("--gap", default=str(OUT_DIR / "gap.json"))
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.06)
    a = ap.parse_args()
    {"fit": fit, "analyze": analyze, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
