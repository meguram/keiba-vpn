"""期待値（的中確率×オッズ）の水準の監査。控除率を加味した水準（単勝・複勝 0.80、馬連・ワイド 0.775、馬単・3連複 0.75、3連単 0.725）に
整合しているか、系統的に高く出ていないかを、過去の確定オッズ・確定払戻・V4 の予測確率で調べる。

python -m src.scripts.data.audit_ev_calibration identities   # オッズ側・確率側の恒等式、全買いの回収率（払戻率の確認）
python -m src.scripts.data.audit_ev_calibration horse        # 単勝・連対（×1.65 の換算）・複勝の EV 水準と較正
python -m src.scripts.data.audit_ev_calibration exotic       # 馬連・ワイド・馬単: 本番の確率 × 確定オッズの EV 水準と較正、市場確率での検算
python -m src.scripts.data.audit_ev_calibration calib       # 補正パラメータ（複勝の実効オッズの重み w、連対の市場確率の指数 λ）の当てはめ
python -m src.scripts.data.audit_ev_calibration suggest      # 画面の買い目提案（suggest_race_bets）の EV と実現回収率
python -m src.scripts.data.audit_ev_calibration portfolio    # 推奨ポートフォリオ（BettingOptimizer）の期待回収率・色分けと実現回収率、縮約 k
結果は data/local/modeling/betting/ev_audit/*.json（ホストの出力先が書けないときは EV_AUDIT_OUT で指定）。

注意: オッズは締切後の確定オッズで、購入時点の値ではない（tp-36 と同じ限界）。予測は V4 の leave-one-year-out（2022〜24）と bundle_v4_2024（2025）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
OUT = Path(os.environ.get("EV_AUDIT_OUT", BET / "ev_audit"))

RET = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}
EV_BINS = (0.0, 0.8, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 1e9)
YEARS = (2022, 2023, 2024, 2025)
N_BOOT = 300


# ─────────────────────────────────────────── 小さな部品（テスト対象）
def ev_bin_labels() -> list[str]:
    return [f"[{lo:g},{'∞' if hi >= 1e8 else format(hi, 'g')})" for lo, hi in zip(EV_BINS[:-1], EV_BINS[1:])]


def bin_table(ev: np.ndarray, ret: np.ndarray, race: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """EV の帯ごとに、賭け数・平均 EV・実現回収率（1 点 1 単位）・レース単位ブートストラップの 95% 区間を返す。"""
    ev, ret = np.asarray(ev, float), np.asarray(ret, float)
    codes = pd.factorize(np.asarray(race))[0]
    rng = np.random.default_rng(seed)
    out = []
    for k, (lo, hi) in enumerate(zip(EV_BINS[:-1], EV_BINS[1:])):
        m = (ev >= lo) & (ev < hi)
        n = int(m.sum())
        row = {"bin": ev_bin_labels()[k], "n": n}
        if n:
            row["mean_ev"] = round(float(ev[m].mean()), 4)
            row["roi"] = round(float(ret[m].mean()), 4)
            if n >= 30:
                g = pd.DataFrame({"c": codes[m], "r": ret[m]}).groupby("c")["r"].agg(["sum", "count"])
                s, c = g["sum"].to_numpy(), g["count"].to_numpy()
                idx = rng.integers(0, len(g), size=(n_boot, len(g)))
                boot = s[idx].sum(axis=1) / np.maximum(c[idx].sum(axis=1), 1)
                row["roi_lo"], row["roi_hi"] = round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)
        out.append(row)
    return out


def fit_shrink_k(ev: np.ndarray, ret: np.ndarray, base: float) -> float | None:
    """realized − base = k × (EV − base) の原点を通る最小二乗の k（0〜1 に丸める）。EV が base からずれた賭けだけを使う。"""
    x, y = np.asarray(ev, float) - base, np.asarray(ret, float) - base
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = float((x * x).sum())
    if len(x) < 100 or d <= 0:
        return None
    return float(np.clip((x * y).sum() / d, 0.0, 1.0))


def identity_summary(vals: np.ndarray, expected: float, tol: float = 0.05) -> dict:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "p5": round(float(np.percentile(v, 5)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "expected": round(expected, 4), "ratio_to_expected": round(float(v.mean() / expected), 4),
            "within_tol": round(float((np.abs(v / expected - 1) <= tol).mean()), 4)}


# ─────────────────────────────────────────── データ
def _payoff_dicts(pay: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """bet_type → race_id → {combo: payout(円)}。"""
    out: dict[str, dict[str, dict]] = {}
    for bt, g in pay.groupby("bet_type"):
        d: dict[str, dict] = {}
        for r, c, p in zip(g["race_id"].to_numpy(), g["combo"].to_numpy(), g["payout"].to_numpy()):
            d.setdefault(str(r), {})[str(c)] = float(p)
        out[bt] = d
    return out


def _key2(a: int, b: int, ordered: bool = False) -> str:
    return f"{a}>{b}" if ordered else f"{min(a, b)}-{max(a, b)}"


def _key3(a: int, b: int, c: int) -> str:
    x = sorted((a, b, c))
    return f"{x[0]}-{x[1]}-{x[2]}"


def load_all():
    """tp36_edge の予測つき単勝・複勝の表と、払戻・組合せオッズ。"""
    from src.scripts.data.tp36_edge import load_frame

    d = load_frame()
    pay = pd.read_parquet(BET / "payoff.parquet")
    pay = pay[pay["race_id"].isin(set(d["race_id"]))]
    pairs = pd.read_parquet(BET / "odds_pair.parquet")
    pairs = pairs[pairs["race_id"].isin(set(d["race_id"]))]
    return d, pay, pairs


# ─────────────────────────────────────────── identities
def cmd_identities(a) -> None:
    d, pay, pairs = load_all()
    w = pd.read_parquet(BET / "odds_win_place.parquet")
    w = w[w["race_id"].isin(set(d["race_id"]))].copy()
    w["year"] = w["race_id"].str[:4].astype(int)
    res: dict = {"n_races": int(d["race_id"].nunique())}

    # 1. オッズ側 Σ(1/odds)
    g = w.groupby("race_id")
    n = g["horse_number"].count()
    year = g["year"].first()
    s_win = g["win_odds"].apply(lambda s: float((1 / s).sum()))
    pm = w[w["place_odds_min"] > 0].groupby("race_id")["place_odds_min"].apply(lambda s: float((1 / s).sum()))
    px = w[w["place_odds_max"] > 0].groupby("race_id")["place_odds_max"].apply(lambda s: float((1 / s).sum()))
    pmid = w.assign(m=(w["place_odds_min"] + w["place_odds_max"]) / 2).groupby("race_id")["m"].apply(lambda s: float((1 / s[s > 0]).sum()))
    res["odds_side"] = {
        "win": {"expected_sum_inv": 1 / RET["tansho"], **identity_summary(s_win.to_numpy(), 1 / RET["tansho"])},
        "place_min": {"expected_sum_inv": 3 / RET["fukusho"], **identity_summary(pm.to_numpy(), 3 / RET["fukusho"])},
        "place_max": identity_summary(px.to_numpy(), 3 / RET["fukusho"]),
        "place_mid": identity_summary(pmid.to_numpy(), 3 / RET["fukusho"]),
        "win_by_year": {str(y): identity_summary(s_win[year == y].to_numpy(), 1 / RET["tansho"]) for y in sorted(year.unique())},
        "win_by_field_size": {k: identity_summary(s_win[(n >= lo) & (n <= hi)].to_numpy(), 1 / RET["tansho"])
                              for k, (lo, hi) in {"<=8": (0, 8), "9-12": (9, 12), "13-15": (13, 15), "16+": (16, 99)}.items()},
    }
    pr = pairs.copy()
    pr["inv"] = 1 / pr["odds"]
    um = pr[pr["bet_type"] == "umaren"].groupby("race_id")["inv"].sum()
    ut = pr[pr["bet_type"] == "umatan"].groupby("race_id")["inv"].sum()
    for col in ("odds_min", "odds_max"):
        pr.loc[pr["bet_type"] == "wide", "inv"] = 1 / pr.loc[pr["bet_type"] == "wide", col]
        res["odds_side"][f"wide_{col}"] = identity_summary(pr[pr["bet_type"] == "wide"].groupby("race_id")["inv"].sum().to_numpy(), 3 / RET["wide"])
    res["odds_side"]["umaren"] = {"expected_sum_inv": 1 / RET["umaren"], **identity_summary(um.to_numpy(), 1 / RET["umaren"])}
    res["odds_side"]["umatan"] = {"expected_sum_inv": 1 / RET["umatan"], **identity_summary(ut.to_numpy(), 1 / RET["umatan"])}

    # 2. 全買い（その券種の全組合せを 100 円ずつ）の実現回収率 = 払戻率の実測
    ncomb = pairs.groupby(["race_id", "bet_type"]).size().unstack(fill_value=0)
    nn = n.reindex(ncomb.index)
    pay = pay.copy()
    pay["year"] = pay["race_id"].str[:4].astype(int)
    stake_n = {"tansho": nn, "fukusho": nn, "umaren": ncomb.get("umaren"), "wide": ncomb.get("wide"), "umatan": ncomb.get("umatan"),
               "sanrenpuku": nn.apply(lambda x: math.comb(int(x), 3)), "sanrentan": nn.apply(lambda x: int(x) * (int(x) - 1) * (int(x) - 2))}
    blanket: dict = {}
    for bt, ncol in stake_n.items():
        pp = pay[pay["bet_type"] == bt].groupby("race_id")["payout"].sum()
        df = pd.DataFrame({"ret": pp, "n": ncol}).dropna()
        df = df[df["n"] > 0]
        df["year"] = df.index.str[:4].astype(int)
        # 払戻が無い（取消・特払など）レースは 0 として扱わず除外する
        blanket[bt] = {"expected": RET[bt], "all": round(float(df["ret"].sum() / (100 * df["n"].sum())), 4), "n_races": int(len(df)),
                       "by_year": {str(y): round(float(g_["ret"].sum() / (100 * g_["n"].sum())), 4) for y, g_ in df.groupby("year")}}
    res["blanket_roi"] = blanket

    # 3. 確率側の恒等式（本番の割引 Harville と、V4 の出力）
    from src.pipeline.inference.betting import _RankContext

    rng = np.random.default_rng(0)
    rid = d["race_id"].unique()
    sample = rng.choice(rid, size=min(a.n_prob_races, len(rid)), replace=False)
    ds = d[d["race_id"].isin(set(sample))]
    dev: dict[str, list[float]] = {k: [] for k in ("model_win", "model_top2", "model_top3", "dist_top2", "dist_top3", "umaren", "wide", "sanrenpuku", "umatan")}
    for r, g_ in ds.groupby("race_id"):
        preds = [{"horse_number": int(h), "win_prob": float(p), "top2_prob": float(p2), "top3_prob": float(p3)}
                 for h, p, p2, p3 in zip(g_["horse_number"], g_["p_win"], g_["p_top2"], g_["p_top3"])]
        dev["model_win"].append(g_["p_win"].sum() - 1)
        dev["model_top2"].append(g_["p_top2"].sum() - 2)
        dev["model_top3"].append(g_["p_top3"].sum() - 3)
        rc = _RankContext(preds)
        dist = rc.dist
        dev["dist_top2"].append(float(dist.top2().sum()) - 2)
        dev["dist_top3"].append(float(dist.top3().sum()) - 3)
        dev["umaren"].append(sum(dist.all_umaren().values()) - 1)
        dev["wide"].append(sum(dist.all_wide().values()) - 3)
        dev["sanrenpuku"].append(sum(dist.all_sanrenpuku().values()) - 1)
        dev["umatan"].append(float(dist._p2.sum()) - 1)
    res["prob_identities"] = {k: {"mean_dev": round(float(np.mean(v)), 6), "max_abs_dev": round(float(np.max(np.abs(v))), 6)} for k, v in dev.items()}

    # 4. 旧経路（app.py の _compute_composite_scores）が pred_score=logit(p_win) を softmax し直すと確率がどうずれるか
    from src.utils.race_probabilities import derive_race_probabilities

    diffs, fav_ratio = [], []
    for r, g_ in ds.groupby("race_id"):
        p = g_["p_win"].to_numpy()
        sc = np.log(np.clip(p, 1e-9, 1) / np.clip(1 - p, 1e-9, 1))
        dp = np.array([x["win_prob"] for x in derive_race_probabilities(list(sc))])
        diffs.append(float(np.abs(dp - p).mean()))
        fav_ratio.append(float(dp[np.argmax(p)] / p.max()))
    res["legacy_softmax_of_logit"] = {"mean_abs_diff_win_prob": round(float(np.mean(diffs)), 5), "favorite_prob_ratio_mean": round(float(np.mean(fav_ratio)), 4)}
    _save("identities", res)


# ─────────────────────────────────────────── horse
def cmd_horse(a) -> None:
    from src.utils.race_probabilities import market_top2_fair_odds, place_effective_odds

    d, pay, pairs = load_all()
    d = d[d["year"].isin(YEARS)].copy()
    d["ev_win"] = d["p_win"] * d["win_odds"]
    d["ev_place_mid"] = d["p_top3"] * d["place_mid"]
    d["ev_place_min"] = d["p_top3"] * d["place_odds_min"]
    d["ev_top2_heur"] = d["p_top2"] * d["win_odds"] * 1.65
    d["place_eff"] = [place_effective_odds(a_, b_) or 0.0 for a_, b_ in zip(d["place_odds_min"], d["place_odds_max"])]
    d["ev_place_eff"] = d["p_top3"] * d["place_eff"]
    fair = {}
    for r, g in d.groupby("race_id"):
        f = market_top2_fair_odds({int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"])})
        for h, v in f.items():
            fair[(r, h)] = v
    d["top2_fair"] = [fair.get((r, int(h))) for r, h in zip(d["race_id"], d["horse_number"])]
    d["ev_top2_new"] = d["p_top2"] * d["top2_fair"]
    d["ret_win_u"] = d["ret_win"] / 100
    d["ret_place_u"] = d["ret_place"] / 100
    res: dict = {"levels": {}}
    for col in ("ev_win", "ev_place_mid", "ev_place_min", "ev_place_eff", "ev_top2_heur", "ev_top2_new"):
        res["levels"][col] = {"mean": round(float(d[col].mean()), 4), "median": round(float(d[col].median()), 4),
                              **{f"share_ge_{t}": round(float((d[col] >= t).mean()), 4) for t in (1.0, 1.05, 1.2, 1.5)},
                              "by_year": {str(y): round(float(g[col].mean()), 4) for y, g in d.groupby("year")}}
    res["calibration"] = {
        "ev_win": bin_table(d["ev_win"].to_numpy(), d["ret_win_u"].to_numpy(), d["race_id"].to_numpy()),
        "ev_place_mid": bin_table(d["ev_place_mid"].to_numpy(), d["ret_place_u"].to_numpy(), d["race_id"].to_numpy()),
        "ev_place_min": bin_table(d["ev_place_min"].to_numpy(), d["ret_place_u"].to_numpy(), d["race_id"].to_numpy()),
        "ev_place_eff": bin_table(d["ev_place_eff"].to_numpy(), d["ret_place_u"].to_numpy(), d["race_id"].to_numpy()),
    }
    res["shrink_k_all_years"] = {"win": fit_shrink_k(d["ev_win"].to_numpy(), d["ret_win_u"].to_numpy(), RET["tansho"]),
                                 "place": fit_shrink_k(d["ev_place_eff"].to_numpy(), d["ret_place_u"].to_numpy(), RET["fukusho"])}
    res["realized_blanket"] = {"win": round(float(d["ret_win_u"].mean()), 4), "place": round(float(d["ret_place_u"].mean()), 4)}
    # 複勝: 実払戻は min と max のどこに入るか（的中時の払戻 ÷ 中央値）
    hit = d[(d["ret_place"] > 0) & (d["place_mid"] > 0) & (d["place_odds_min"] > 0)]
    res["place_payout_vs_range"] = {"n_hits": int(len(hit)), "payout_over_mid": round(float((hit["ret_place_u"] / hit["place_mid"]).mean()), 4),
                                    "payout_over_min": round(float((hit["ret_place_u"] / hit["place_odds_min"]).mean()), 4)}
    # 連対の換算 ×1.65: 馬連の確定オッズから作る「市場の連対の公正オッズ」と比べる
    um = pairs[pairs["bet_type"] == "umaren"].copy()
    um[["h1", "h2"]] = um["combo"].str.split("-", expand=True).astype(int)
    inv = pd.concat([um[["race_id", "h1", "odds"]].rename(columns={"h1": "hn"}), um[["race_id", "h2", "odds"]].rename(columns={"h2": "hn"})])
    inv["inv"] = 1 / inv["odds"]
    s = inv.groupby(["race_id", "hn"])["inv"].sum().rename("sum_inv").reset_index().rename(columns={"hn": "horse_number"})
    d2 = d.merge(s, on=["race_id", "horse_number"], how="inner")
    d2["o2_fair"] = 1 / d2["sum_inv"]                      # 市場の連対確率 = 0.775×Σ(1/odds)。公正（控除後）の連対オッズ = 1/Σ(1/odds)
    d2["heur"] = d2["win_odds"] * 1.65
    d2["ratio"] = d2["heur"] / d2["o2_fair"]
    bands = [(1, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 1000)]
    res["top2_heuristic_vs_market"] = {
        "overall_ratio_mean": round(float(d2["ratio"].mean()), 4), "overall_ratio_median": round(float(d2["ratio"].median()), 4),
        "by_win_odds_band": {f"{lo}-{hi}": {"n": int(m.sum()), "ratio_mean": round(float(d2.loc[m, "ratio"].mean()), 4)}
                             for lo, hi in bands for m in [(d2["win_odds"] >= lo) & (d2["win_odds"] < hi)] if m.sum() > 0},
        "market_top2_prob_mean": round(float((0.775 * d2["sum_inv"]).mean()), 4)}
    d2["ev_top2_market"] = d2["p_top2"] * d2["o2_fair"]
    res["levels"]["ev_top2_market_equivalent"] = {"mean": round(float(d2["ev_top2_market"].mean()), 4), **{f"share_ge_{t}": round(float((d2["ev_top2_market"] >= t).mean()), 4) for t in (1.0, 1.2, 1.5)}}
    _save("horse", res)


# ─────────────────────────────────────────── calib
def market_top2_probs(q: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """市場の勝率 q（合計 1）から、q^λ で作る Harville の連対確率。"""
    from src.utils.race_probabilities import harville_top2_prob

    w = q ** lam
    return harville_top2_prob(w / w.sum())


def cmd_calib(a) -> None:
    d, pay, pairs = load_all()
    d = d[d["year"].isin(YEARS)].copy()
    res: dict = {}
    # (a) 複勝: 的中時の払戻が min〜max のどこにあるか、Σ(1/オッズ)=3/0.8 になる重み w
    hit = d[(d["ret_place"] > 0) & (d["place_odds_max"] > d["place_odds_min"]) & (d["place_odds_min"] > 0)]
    wh = ((hit["ret_place"] / 100 - hit["place_odds_min"]) / (hit["place_odds_max"] - hit["place_odds_min"])).clip(0, 1)
    res["place_payout_position"] = {"n_hits": int(len(hit)), "mean_w": round(float(wh.mean()), 4), "median_w": round(float(wh.median()), 4),
                                    "payout_over_mid": round(float((hit["ret_place"] / 100 / hit["place_mid"]).mean()), 4),
                                    "payout_over_min": round(float((hit["ret_place"] / 100 / hit["place_odds_min"]).mean()), 4)}
    dd = d[(d["place_odds_min"] > 0) & (d["place_odds_max"] >= d["place_odds_min"])]
    best, best_gap = None, 1e9
    for w in np.linspace(0, 1, 21):
        eff = dd["place_odds_min"] + w * (dd["place_odds_max"] - dd["place_odds_min"])
        sums = (1 / eff).groupby(dd["race_id"]).sum()
        gap = abs(float(sums.mean()) - 3 / RET["fukusho"])
        if gap < best_gap:
            best, best_gap = float(w), gap
    res["place_w_sum_identity"] = {"w": round(best, 3), "gap": round(best_gap, 4)}
    # (b) 連対: 馬連の確定オッズから作る市場の連対確率 vs 単勝の市場確率からの Harville
    um = pairs[pairs["bet_type"] == "umaren"].copy()
    hh = um["combo"].str.split("-", expand=True).astype(int)
    um["h1"], um["h2"] = hh[0].to_numpy(), hh[1].to_numpy()
    um["inv"] = 1 / um["odds"]
    inv = pd.concat([um[["race_id", "h1", "inv"]].rename(columns={"h1": "hn"}), um[["race_id", "h2", "inv"]].rename(columns={"h2": "hn"})])
    p2pool = (RET["umaren"] * inv.groupby(["race_id", "hn"])["inv"].sum()).rename("p2_pool").reset_index().rename(columns={"hn": "horse_number"})
    x = d.merge(p2pool, on=["race_id", "horse_number"], how="inner")
    rows = {}
    for lam in (0.7, 0.8, 0.9, 1.0, 1.1):
        err, ratios, cnt = 0.0, [], 0
        for r, g in x.groupby("race_id"):
            q = (1 / g["win_odds"].to_numpy())
            q = q / q.sum()
            ph = market_top2_probs(q, lam)
            lr = np.log(np.clip(g["p2_pool"].to_numpy(), 1e-6, 1)) - np.log(np.clip(ph, 1e-6, 1))
            err += float((lr ** 2).sum())
            cnt += len(lr)
            ratios.append(g["p2_pool"].to_numpy() / np.clip(ph, 1e-6, 1))
        rr = np.concatenate(ratios)
        rows[str(lam)] = {"rmse_log": round(math.sqrt(err / cnt), 4), "mean_ratio_pool_over_harville": round(float(rr.mean()), 4)}
    res["top2_market_harville_fit"] = rows
    # λ ごとに、勝率オッズ帯別の比（pool / Harville(λ)）
    lam_best = min(rows, key=lambda k: rows[k]["rmse_log"])
    res["top2_best_lambda"] = float(lam_best)
    bands = [(1, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 1000)]
    rb: dict = {}
    for lam in (1.0, float(lam_best)):
        arr = []
        for r, g in x.groupby("race_id"):
            q = (1 / g["win_odds"].to_numpy())
            q = q / q.sum()
            ph = market_top2_probs(q, lam)
            arr.append(pd.DataFrame({"wo": g["win_odds"].to_numpy(), "ratio": g["p2_pool"].to_numpy() / np.clip(ph, 1e-6, 1)}))
        z = pd.concat(arr)
        rb[str(lam)] = {f"{lo}-{hi}": round(float(z[(z["wo"] >= lo) & (z["wo"] < hi)]["ratio"].mean()), 4) for lo, hi in bands}
    res["top2_ratio_by_win_odds_band"] = rb
    _save("calib", res)


# ─────────────────────────────────────────── exotic
def _race_combo_rows(g: pd.DataFrame, um: dict, wd: dict, ut: dict, hit: dict, rc_cls, alpha_mode: str) -> list[tuple]:
    """1 レース分の (bet_type, ev, ret単位, odds, prob) を返す。alpha_mode: model（本番）/ harville_market（市場の単勝確率を α=1 で）。"""
    hn = g["horse_number"].astype(int).to_numpy()
    if alpha_mode == "model":
        win, t2, t3 = g["p_win"].to_numpy(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy()
    else:
        q = (1 / g["win_odds"]).to_numpy()
        win, t2, t3 = q / q.sum(), None, None
    preds = [{"horse_number": int(h), "win_prob": float(w_), **({"top2_prob": float(x), "top3_prob": float(y)} if t2 is not None else {})}
             for h, w_, x, y in zip(hn, win, t2 if t2 is not None else win, t3 if t3 is not None else win)]
    rc = rc_cls(preds)
    dist, ix = rc.dist, rc.idx
    p2 = dist._p2
    wide = dist.all_wide()
    out = []
    for (a_, b_), o in um.items():
        if a_ not in ix or b_ not in ix:
            continue
        i, j = ix[a_], ix[b_]
        pr = float(p2[i, j] + p2[j, i])
        out.append(("umaren", pr * o, hit["umaren"].get(_key2(a_, b_), 0.0) / 100, o, pr))
    for (a_, b_), (omin, omax) in wd.items():
        if a_ not in ix or b_ not in ix:
            continue
        i, j = sorted((ix[a_], ix[b_]))
        pr = float(wide[(i, j)])
        out.append(("wide", pr * omin, hit["wide"].get(_key2(a_, b_), 0.0) / 100, omin, pr))
    for (a_, b_), o in ut.items():
        if a_ not in ix or b_ not in ix:
            continue
        pr = float(p2[ix[a_], ix[b_]])
        out.append(("umatan", pr * o, hit["umatan"].get(_key2(a_, b_, True), 0.0) / 100, o, pr))
    return out


def cmd_exotic(a) -> None:
    from src.pipeline.inference.betting import _RankContext

    d, pay, pairs = load_all()
    d = d[d["year"].isin(YEARS)]
    pd_ = _payoff_dicts(pay)
    hit_all = {bt: pd_.get(bt, {}) for bt in ("umaren", "wide", "umatan")}
    pr = pairs[pairs["race_id"].isin(set(d["race_id"]))]
    um_g = {r: g for r, g in pr[pr["bet_type"] == "umaren"].groupby("race_id")}
    wd_g = {r: g for r, g in pr[pr["bet_type"] == "wide"].groupby("race_id")}
    ut_g = {r: g for r, g in pr[pr["bet_type"] == "umatan"].groupby("race_id")}
    races = sorted(d["race_id"].unique())
    if a.limit:
        races = races[: a.limit]
    res: dict = {"modes": {}}
    for mode in ("model", "harville_market"):
        rows: list[tuple] = []
        rid: list[str] = []
        for r in races:
            if r not in um_g or r not in wd_g or r not in ut_g:
                continue
            g = d[d["race_id"] == r]
            if len(g) < 4:
                continue
            um = {tuple(map(int, c.split("-"))): float(o) for c, o in zip(um_g[r]["combo"], um_g[r]["odds"])}
            wd = {tuple(map(int, c.split("-"))): (float(x), float(y)) for c, x, y in zip(wd_g[r]["combo"], wd_g[r]["odds_min"], wd_g[r]["odds_max"])}
            ut = {tuple(map(int, c.split(">"))): float(o) for c, o in zip(ut_g[r]["combo"], ut_g[r]["odds"])}
            hit = {bt: {k: v for k, v in hit_all[bt].get(r, {}).items()} for bt in hit_all}
            part = _race_combo_rows(g, um, wd, ut, hit, _RankContext, mode)
            rows.extend(part)
            rid.extend([r] * len(part))
        df = pd.DataFrame(rows, columns=["bt", "ev", "ret", "odds", "prob"])
        df["race_id"] = rid
        m: dict = {}
        for bt, gg in df.groupby("bt"):
            m[bt] = {"n_bets": int(len(gg)), "expected_return_rate": RET[bt],
                     "mean_ev_equal_weight": round(float(gg["ev"].mean()), 4), "median_ev": round(float(gg["ev"].median()), 4),
                     "prob_weighted_mean_ev": round(float((gg["ev"] * gg["prob"]).sum() / gg["prob"].sum()), 4),
                     "realized_blanket_roi": round(float(gg["ret"].mean()), 4),
                     **{f"share_ev_ge_{t}": round(float((gg["ev"] >= t).mean()), 4) for t in (1.05, 1.2, 1.5)},
                     "calibration": bin_table(gg["ev"].to_numpy(), gg["ret"].to_numpy(), gg["race_id"].to_numpy()),
                     "shrink_k_all_years": fit_shrink_k(gg["ev"].to_numpy(), gg["ret"].to_numpy(), RET[bt])}
        res["modes"][mode] = m
    _save("exotic", res)


# ─────────────────────────────────────────── suggest / portfolio
def _entries_for(g: pd.DataFrame) -> tuple[list[dict], dict[int, dict]]:
    ent = [{"horse_number": int(h), "horse_name": f"#{int(h)}", "win_prob": float(p), "top2_prob": float(p2), "top3_prob": float(p3)}
           for h, p, p2, p3 in zip(g["horse_number"], g["p_win"], g["p_top2"], g["p_top3"])]
    odds = {int(h): {"win_odds": float(w_), "place_odds_min": float(a_), "place_odds_max": float(b_), "popularity": int(po) if po == po else None}
            for h, w_, a_, b_, po in zip(g["horse_number"], g["win_odds"], g["place_odds_min"], g["place_odds_max"], g["popularity"])}
    return ent, odds


def _pick_return(p: dict, hit: dict, race: str) -> float:
    """買い目 1 点（100 円）の払戻（100 円単位）。"""
    h = p["horses"]
    bt = p["bet_type"]
    if bt == "tansho":
        return hit["tansho"].get(race, {}).get(str(h[0]), 0.0) / 100
    if bt == "fukusho":
        return hit["fukusho"].get(race, {}).get(str(h[0]), 0.0) / 100
    if bt == "umaren":
        return hit["umaren"].get(race, {}).get(_key2(h[0], h[1]), 0.0) / 100
    if bt == "wide":
        return hit["wide"].get(race, {}).get(_key2(h[0], h[1]), 0.0) / 100
    if bt == "umatan":
        return hit["umatan"].get(race, {}).get(_key2(h[0], h[1], True), 0.0) / 100
    if bt == "trio":
        return hit["sanrenpuku"].get(race, {}).get(_key3(*h), 0.0) / 100
    if bt == "tierce":
        return hit["sanrentan"].get(race, {}).get(f"{h[0]}>{h[1]}>{h[2]}", 0.0) / 100
    return 0.0


def cmd_suggest(a) -> None:
    from src.api.local_predictions import score_entries
    from src.utils.race_bet_suggestion import suggest_race_bets

    d, pay, pairs = load_all()
    d = d[d["year"].isin(YEARS)]
    hit = _payoff_dicts(pay)
    hit = {k: v for k, v in hit.items()}
    rows = []
    for r, g in d.groupby("race_id"):
        ent, odds = _entries_for(g)
        scored = score_entries(ent, odds)
        sug = suggest_race_bets(scored)
        for p in sug.get("picks", []):
            rows.append({"race_id": r, "year": int(r[:4]), "bet_type": p["bet_type"], "ev": p["ev"] or 0.0, "confidence": p["confidence"],
                         "ret": _pick_return(p, hit, r)})
    df = pd.DataFrame(rows)
    res: dict = {"n_picks": int(len(df)), "n_races_with_picks": int(df["race_id"].nunique()), "n_races": int(d["race_id"].nunique())}
    res["by_bet_type"] = {bt: {"n": int(len(g)), "mean_claimed_ev": round(float(g["ev"].mean()), 4), "realized_roi": round(float(g["ret"].mean()), 4),
                               "expected_return_rate": RET.get({"trio": "sanrenpuku", "tierce": "sanrentan"}.get(bt, bt), 0.8)} for bt, g in df.groupby("bet_type")}
    res["by_confidence"] = {c: {"n": int(len(g)), "mean_claimed_ev": round(float(g["ev"].mean()), 4), "realized_roi": round(float(g["ret"].mean()), 4)} for c, g in df.groupby("confidence")}
    res["by_year"] = {str(y): {"n": int(len(g)), "mean_claimed_ev": round(float(g["ev"].mean()), 4), "realized_roi": round(float(g["ret"].mean()), 4)} for y, g in df.groupby("year")}
    res["calibration_all"] = bin_table(df["ev"].to_numpy(), df["ret"].to_numpy(), df["race_id"].to_numpy())
    _save("suggest", res)


def _pair_odds_docs(pairs_r: pd.DataFrame) -> dict:
    out: dict = {"umaren": [], "wide": [], "umatan": []}
    for bt, c, o, omin, omax in zip(pairs_r["bet_type"], pairs_r["combo"], pairs_r["odds"], pairs_r["odds_min"], pairs_r["odds_max"]):
        if bt == "wide":
            out["wide"].append({"pair": [int(x) for x in c.split("-")], "odds_min": float(omin), "odds_max": float(omax)})
        elif bt == "umaren":
            out["umaren"].append({"pair": [int(x) for x in c.split("-")], "odds": float(o)})
        elif bt == "umatan":
            out["umatan"].append({"pair": [int(x) for x in c.split(">")], "odds": float(o)})
    return out


def cmd_portfolio(a) -> None:
    from src.api.v1.delegates import VALUE_LEVELS, value_level
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer

    d, pay, pairs = load_all()
    d = d[d["year"].isin(YEARS)]
    hit = _payoff_dicts(pay)
    pair_by_race = {r: g for r, g in pairs.groupby("race_id")}
    cfg = BettingConfig(max_total_bet=2000, bet_types=["tansho", "fukusho", "umaren", "wide"])
    opt = BettingOptimizer(cfg)
    races, cand_rows = [], []
    for r, g in d.groupby("race_id"):
        if r not in pair_by_race:
            continue
        ent, odds = _entries_for(g)
        preds = [{**e, "win_odds": odds[e["horse_number"]]["win_odds"], "place_odds_min": odds[e["horse_number"]]["place_odds_min"],
                  "place_odds_max": odds[e["horse_number"]]["place_odds_max"], "pred_score": math.log(max(e["win_prob"], 1e-9))} for e in ent]
        single = {"entries": [{"horse_number": hn, **o} for hn, o in odds.items()]}
        port = opt.optimize(preds, _pair_odds_docs(pair_by_race[r]), 100000, single_odds=single)
        cs = port["candidates"]
        if not cs:
            races.append({"race_id": r, "year": int(r[:4]), "n_bets": 0, "claimed": None, "realized": None, "stake": 0})
            continue
        stake = sum(c.bet_amount for c in cs)
        ret = 0.0
        for c in cs:
            k = {"tansho": "tansho", "fukusho": "fukusho", "umaren": "umaren", "wide": "wide", "umatan": "umatan"}[c.bet_type]
            key = str(c.pair[0]) if k in ("tansho", "fukusho") else (_key2(*c.pair, ordered=True) if k == "umatan" else _key2(*c.pair))
            u = hit[k].get(r, {}).get(key, 0.0) / 100
            ret += u * c.bet_amount
            cand_rows.append({"race_id": r, "year": int(r[:4]), "bet_type": c.bet_type, "ev": c.ev, "ret": u, "amount": c.bet_amount})
        races.append({"race_id": r, "year": int(r[:4]), "n_bets": len(cs), "claimed": float(port["expected_roi"]), "realized": float(ret / stake) if stake else None, "stake": stake,
                      "return": float(ret)})
    rdf, cdf = pd.DataFrame(races), pd.DataFrame(cand_rows)
    rdf["level"] = [value_level(c, n) for c, n in zip(rdf["claimed"], rdf["n_bets"])]
    res: dict = {"n_races": int(len(rdf)), "levels": {f"{n}>={f}": f for n, f in VALUE_LEVELS}}
    res["by_level"] = {lv: {"n_races": int(len(g)), "mean_claimed_roi": None if g["claimed"].isna().all() else round(float(g["claimed"].mean()), 4),
                            "realized_roi": None if g["stake"].sum() == 0 else round(float(g["return"].sum() / g["stake"].sum()), 4)}
                       for lv, g in rdf.groupby("level")}
    res["by_level_year"] = {f"{y}|{lv}": {"n_races": int(len(g)), "mean_claimed_roi": None if g["claimed"].isna().all() else round(float(g["claimed"].mean()), 4),
                                         "realized_roi": None if g["stake"].sum() == 0 else round(float(g["return"].sum() / g["stake"].sum()), 4)}
                            for (y, lv), g in rdf.groupby(["year", "level"])}
    res["candidates_by_bet_type"] = {bt: {"n": int(len(g)), "mean_ev": round(float(g["ev"].mean()), 4), "realized_roi_unweighted": round(float(g["ret"].mean()), 4),
                                          "realized_roi_stake_weighted": round(float((g["ret"] * g["amount"]).sum() / g["amount"].sum()), 4)} for bt, g in cdf.groupby("bet_type")}
    res["candidates_calibration"] = bin_table(cdf["ev"].to_numpy(), cdf["ret"].to_numpy(), cdf["race_id"].to_numpy())
    # 縮約 k: 評価年より前の年で当てはめ、評価年に適用（walk-forward）。EV_adj = base + k (EV − base)
    shr = {}
    for bt, g in cdf.groupby("bet_type"):
        base = RET[bt]
        for y in YEARS[1:]:
            tr = g[g["year"] < y]
            k = fit_shrink_k(tr["ev"].to_numpy(), tr["ret"].to_numpy(), base)
            shr[f"{bt}|{y}"] = None if k is None else round(k, 4)
    res["shrink_k_walk_forward"] = shr
    rr = rdf[rdf["n_bets"] > 0]
    res["portfolio_race_level"] = {"shrink_k_all_years": fit_shrink_k(rr["claimed"].to_numpy(), rr["realized"].to_numpy(), 0.8),
                                   "by_year": {str(y): fit_shrink_k(g["claimed"].to_numpy(), g["realized"].to_numpy(), 0.8) for y, g in rr.groupby("year")}}
    res["shrink_k_all_years"] = {bt: fit_shrink_k(g["ev"].to_numpy(), g["ret"].to_numpy(), RET[bt]) for bt, g in cdf.groupby("bet_type")}
    _save("portfolio", res)


# ─────────────────────────────────────────── 共通
def _save(name: str, res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {OUT / (name + '.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["identities", "horse", "calib", "exotic", "suggest", "portfolio"])
    ap.add_argument("--n-prob-races", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="exotic の対象レース数の上限（試走用）")
    a = ap.parse_args()
    {"identities": cmd_identities, "horse": cmd_horse, "calib": cmd_calib, "exotic": cmd_exotic, "suggest": cmd_suggest, "portfolio": cmd_portfolio}[a.cmd](a)


if __name__ == "__main__":
    main()
