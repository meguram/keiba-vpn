"""tp-36 (2) モデルと市場のずれ: v4 の確率と市場の確率（確定オッズ）の差を、単勝・複勝の回収率で測る。Benter 型の混合も評価。

python -m src.scripts.data.tp36_edge --out data/local/modeling/betting/tp36/edge.json
予測: valid 2022〜2024 は v4_e2 の leave-one-year-out、2025 は bundle_v4_2024（2020〜24 学習）の test 予測。
混合の重み (a, b) は『評価年より前の年』の勝者の対数尤度で当てはめる（walk-forward）。オッズは締切後の確定オッズ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP = ROOT / "data/local/modeling/training_pipeline"


def load_frame() -> pd.DataFrame:
    w = pd.read_parquet(BET / "odds_win_place.parquet")
    pay = pd.read_parquet(BET / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    bad_odds = set(w.loc[~(w["win_odds"] >= 1.0), "race_id"])     # 取消（-3.0）や欠損の馬がいるレースは除く
    w = w[~w["race_id"].isin(bad | bad_odds)].copy()
    e2 = pd.read_parquet(TP / "tp27/ens/v4_e2.parquet")[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]
    t25 = pd.read_parquet(TP / "final_eval_v4/test2025_predictions.parquet")[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]
    pred = pd.concat([e2, t25], ignore_index=True)
    pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
    w["horse_id"] = w["horse_id"].astype(str)
    d = w.merge(pred, on=["race_id", "horse_id"], how="inner")
    for bt, col in (("tansho", "ret_win"), ("fukusho", "ret_place")):
        p = pay[pay["bet_type"] == bt][["race_id", "combo", "payout"]].copy()
        p["horse_number"] = p["combo"].astype(int)
        d = d.merge(p[["race_id", "horse_number", "payout"]].rename(columns={"payout": col}), on=["race_id", "horse_number"], how="left")
        d[col] = d[col].fillna(0.0)
    d["year"] = d["race_id"].str[:4].astype(int)
    d["place_mid"] = (d["place_odds_min"] + d["place_odds_max"]) / 2
    # 全馬がそろったレースのみ（確率の合計が 1 から離れたレースを除く）
    s = d.groupby("race_id")["p_win"].transform("sum")
    d = d[(s > 0.97) & (s < 1.03)].copy()
    d["q_win"] = (1 / d["win_odds"]) / (1 / d["win_odds"]).groupby(d["race_id"]).transform("sum")
    d["win"] = (d["ret_win"] > 0).astype(float)
    return d


def fit_benter(d: pd.DataFrame) -> tuple[float, float]:
    """勝者の条件付き対数尤度で a (モデル) と b (市場) を当てはめる。softmax(a·log p + b·log q)。"""
    groups = [(np.log(np.clip(g["p_win"].to_numpy(), 1e-6, 1)), np.log(np.clip(g["q_win"].to_numpy(), 1e-6, 1)), int(np.argmax(g["win"].to_numpy()))) for _, g in d.groupby("race_id") if g["win"].sum() == 1]

    def nll(x):
        a, b = x
        t = 0.0
        for lp, lq, k in groups:
            z = a * lp + b * lq
            z = z - z.max()
            t -= z[k] - np.log(np.exp(z).sum())
        return t / len(groups)

    r = minimize(nll, x0=[0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 200})
    return float(r.x[0]), float(r.x[1])


def blend(d: pd.DataFrame, a: float, b: float) -> pd.Series:
    z = a * np.log(np.clip(d["p_win"], 1e-6, 1)) + b * np.log(np.clip(d["q_win"], 1e-6, 1))
    z = z - z.groupby(d["race_id"]).transform("max")
    e = np.exp(z)
    return e / e.groupby(d["race_id"]).transform("sum")


def logloss_win(d: pd.DataFrame, col: str) -> float:
    g = d[d["win"] == 1]
    return float(-np.log(np.clip(g[col], 1e-9, 1)).mean())


def boot(g: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float, int, float]:
    s = g["stake"].to_numpy(float)
    r = g["ret"].to_numpy(float)
    if s.sum() <= 0:
        return float("nan"), float("nan"), float("nan"), 0, float("nan")
    rng = np.random.default_rng(seed)
    n = len(s)
    idx = rng.integers(0, n, size=(n_boot, n))
    ss, rr = s[idx].sum(1), r[idx].sum(1)
    ok = ss > 0
    b = rr[ok] / ss[ok]
    hit = float((g["ret"] > 0).sum() / max(1, (g["stake"] > 0).sum()))
    return float(r.sum() / s.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), int(round(s.sum() / 100)), hit


def ev_bins(d: pd.DataFrame, ev: pd.Series, ret_col: str, edges: list[float], years: list[int]) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ev >= lo) & (ev < hi) & d["year"].isin(years)
        sub = d[m]
        if len(sub) == 0:
            continue
        agg = pd.DataFrame({"race_id": sub["race_id"], "stake": 100.0, "ret": sub[ret_col]}).groupby("race_id").sum().reset_index()
        roi, lo_, hi_, n, hit = boot(agg)
        row = {"ev": f"{lo:.2f}-{hi:.2f}", "roi": roi, "lo": lo_, "hi": hi_, "n_bets": n, "hit": hit}
        for y in years:
            a = sub[sub["year"] == y]
            row[str(y)] = float(a[ret_col].sum() / (100.0 * len(a))) if len(a) else None
        rows.append(row)
    return rows


def threshold_scan(d: pd.DataFrame, ev: pd.Series, ret_col: str, taus: list[float], years: list[int]) -> list[dict]:
    rows = []
    for t in taus:
        m = (ev >= t) & d["year"].isin(years)
        sub = d[m]
        if len(sub) == 0:
            continue
        agg = pd.DataFrame({"race_id": sub["race_id"], "stake": 100.0, "ret": sub[ret_col]}).groupby("race_id").sum().reset_index()
        roi, lo_, hi_, n, hit = boot(agg)
        rows.append({"tau": t, "roi": roi, "lo": lo_, "hi": hi_, "n_bets": n, "hit": hit, "races_bet": int(sub["race_id"].nunique())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BET / "tp36/edge.json"))
    a = ap.parse_args()
    d = load_frame()
    out: dict = {"rows": int(len(d)), "races": int(d["race_id"].nunique()), "years": sorted(d["year"].unique().tolist())}
    # 情報量: 勝者の対数確率
    out["logloss_win"] = {"model": logloss_win(d.assign(c=d["p_win"]), "c"), "market": logloss_win(d.assign(c=d["q_win"]), "c")}
    # Benter の当てはめ（walk-forward）
    d["p_blend"] = np.nan
    fits = {}
    for ey in (2023, 2024, 2025):
        tr = d[d["year"] < ey]
        a_, b_ = fit_benter(tr)
        fits[ey] = (a_, b_)
        m = d["year"] == ey
        d.loc[m, "p_blend"] = blend(d[m], a_, b_)
    out["benter_fit"] = {str(k): {"a_model": v[0], "b_market": v[1]} for k, v in fits.items()}
    dd = d[d["year"] >= 2023].copy()
    out["logloss_win_2023_2025"] = {"model": logloss_win(dd.assign(c=dd["p_win"]), "c"), "market": logloss_win(dd.assign(c=dd["q_win"]), "c"), "blend": logloss_win(dd.assign(c=dd["p_blend"]), "c")}
    years = [2023, 2024, 2025]
    edges = [0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 100.0]
    res = {}
    for name, pcol in (("model", "p_win"), ("blend", "p_blend")):
        ev_t = dd[pcol] * dd["win_odds"]
        res[f"tansho_{name}_bins"] = ev_bins(dd, ev_t, "ret_win", edges, years)
        res[f"tansho_{name}_scan"] = threshold_scan(dd, ev_t, "ret_win", [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5], years)
    # 複勝: 確率は p_top3（モデル）。ブレンドは単勝のブレンド比で p_top3 を補正する簡易版（勝率比で拡縮）。
    ratio = (dd["p_blend"] / dd["p_win"].clip(lower=1e-6)).clip(0.2, 5.0)
    p3_blend = (dd["p_top3"] * ratio).clip(upper=0.98)
    for name, p3 in (("model", dd["p_top3"]), ("blend", p3_blend)):
        ev_p = p3 * dd["place_mid"]
        res[f"fukusho_{name}_bins"] = ev_bins(dd, ev_p, "ret_place", edges, years)
        res[f"fukusho_{name}_scan"] = threshold_scan(dd, ev_p, "ret_place", [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3], years)
    out["results"] = res
    # モデルが市場より高く見る（比）別
    dd["ratio_w"] = dd["p_win"] / dd["q_win"]
    rb = [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 100]
    out["tansho_by_ratio"] = ev_bins(dd, dd["ratio_w"], "ret_win", rb, years)
    out["fukusho_by_ratio"] = ev_bins(dd, dd["ratio_w"], "ret_place", rb, years)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    d.to_parquet(Path(a.out).with_name("edge_frame.parquet"), index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
