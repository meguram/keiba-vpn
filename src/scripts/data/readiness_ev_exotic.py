"""導入可否の確認用: 期待値の較正・印の仕様・組合せ券の確率を、2026 年の v4 予測（バックフィル）と確定結果で判定する（readiness-audit）。

  python -m src.scripts.data.readiness_ev_exotic --out /home/jovyan/work/readiness_out/ev_exotic.json

基準は readiness_criteria.json の ev_calibration_marks / exotic_probabilities。オッズは確定オッズ（予測時点のオッズではない）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARK_KEYS = ("honmei", "pair", "show_val", "star", "anchor")
SINGLE_MARKS = ("honmei", "pair", "show_val", "star")


def ece_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float((np.abs(g["p"] - g["y"]) * g["n"]).sum() / len(p))


def binary_ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def mark_stats(counts: pd.DataFrame, limits: pd.Series) -> dict:
    """counts: 行=レース、列=MARK_KEYS の個数。limits: レースごとの △ の上限。"""
    total = counts.sum(axis=1)
    viol_single = int((counts[list(SINGLE_MARKS)] > 1).any(axis=1).sum())
    viol_anchor = int((counts["anchor"] > limits).sum())
    return {"races": int(len(counts)), "marks_per_race_mean": float(total.mean()), "marks_per_race_min": int(total.min()), "marks_per_race_max": int(total.max()),
            "mean_by_mark": {k: float(counts[k].mean()) for k in MARK_KEYS}, "races_single_mark_over1": viol_single, "races_anchor_over_limit": viol_anchor,
            "races_with_zero_honmei": int((counts["honmei"] == 0).sum())}


def decile_gap(ev: np.ndarray, ret: np.ndarray, race: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """EV の十分位ごとの平均 EV と実現回収（購入 1 あたりの払戻倍率）。最上位十分位のずれはレース単位ブートストラップ付き。"""
    df = pd.DataFrame({"ev": ev, "ret": ret, "race": race})
    df["dec"] = pd.qcut(df["ev"].rank(method="first"), 10, labels=False)
    tab = df.groupby("dec").agg(ev=("ev", "mean"), realized=("ret", "mean"), n=("ev", "size")).reset_index()
    top = df[df["dec"] == 9]
    gap = float(top["ev"].mean() - top["ret"].mean())
    g = top.groupby("race").agg(e=("ev", "sum"), r=("ret", "sum"), n=("ev", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    n = g["n"].to_numpy()[idx].sum(1)
    est = (g["e"].to_numpy()[idx].sum(1) - g["r"].to_numpy()[idx].sum(1)) / n
    return {"overall_ev": float(df["ev"].mean()), "overall_realized": float(df["ret"].mean()), "top_decile_gap_ev_minus_realized": gap,
            "top_decile_gap_ci95": [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))], "deciles": tab.round(4).to_dict("records"),
            "ev_ge_1": {"n": int((df["ev"] >= 1.0).sum()), "mean_ev": float(df.loc[df["ev"] >= 1.0, "ev"].mean()) if (df["ev"] >= 1.0).any() else None,
                        "realized": float(df.loc[df["ev"] >= 1.0, "ret"].mean()) if (df["ev"] >= 1.0).any() else None}}


def run_ev_marks(pred: pd.DataFrame, res: pd.DataFrame, odds: pd.DataFrame) -> dict:
    from src.pipeline.inference.prediction_results import parse_payoff
    from src.utils.race_probabilities import (
        assign_mece_marks, attach_calibrated_ev, market_top2_fair_odds, maybe_limit, place_effective_odds,
    )

    od = odds.set_index(["race_id", "horse_number"])
    rows, mark_rows, limits = [], [], {}
    r1 = res.drop_duplicates("race_id").set_index("race_id")["payoff"]
    pay = {rid: parse_payoff(json.loads(p) if isinstance(p, str) else p) for rid, p in r1.items() if isinstance(p, (str, dict))}
    odds_races = set(odds["race_id"])
    for rid, g in pred.merge(res[["race_id", "horse_id", "horse_number", "finish_position"]], on=["race_id", "horse_id"]).groupby("race_id"):
        g = g[g["finish_position"] >= 1]
        if len(g) < 5 or rid not in odds_races:
            continue
        o = {int(h): od.loc[(rid, h)] for h in g["horse_number"].astype(int) if (rid, h) in od.index}
        if len(o) != len(g):
            continue
        t2 = market_top2_fair_odds({h: float(v["win_odds"]) for h, v in o.items()})
        entries = []
        for _, r in g.iterrows():
            h = int(r["horse_number"])
            wo = float(o[h]["win_odds"])
            pe = place_effective_odds(float(o[h]["place_odds_min"] or 0), float(o[h]["place_odds_max"] or 0))
            wp, p2, p3 = float(r["p_win"]), float(r["p_top2"]), float(r["p_top3"])
            entries.append({"horse_number": h, "win_prob": wp, "top2_prob": p2, "top3_prob": p3, "win_odds": wo,
                            "ev_win": round(wp * wo, 3) if wo > 0 else None, "ev_top2": round(p2 * t2[h], 3) if t2.get(h) else None,
                            "ev_place": round(p3 * pe, 3) if pe else None, "finish": int(r["finish_position"])})
        for e in entries:
            attach_calibrated_ev(e)
        assign_mece_marks(entries)
        pm = pay.get(rid, {})
        for e in entries:
            h = e["horse_number"]
            rw = pm.get("tansho", {}).get((h,), 0) / 100.0
            rp = pm.get("fukusho", {}).get((h,), 0) / 100.0
            rows.append({"race_id": rid, "ev_win_cal": e["ev_win_cal"], "ev_place_cal": e["ev_place_cal"], "ev_top2_cal": e["ev_top2_cal"],
                         "ev_win_raw": e["ev_win"], "ev_place_raw": e["ev_place"], "ret_win": rw, "ret_place": rp,
                         "has_fukusho": len(pm.get("fukusho", {})) >= (3 if len(entries) >= 8 else 2), "has_tansho": "tansho" in pm})
        cnt = {k: sum(1 for e in entries if e["mark_type"] == k) for k in MARK_KEYS}
        mark_rows.append({"race_id": rid, **cnt})
        limits[rid] = maybe_limit(len(entries))
    ev = pd.DataFrame(rows)
    mk = pd.DataFrame(mark_rows).set_index("race_id")
    out = {"races": int(len(mk)), "marks": mark_stats(mk[list(MARK_KEYS)], pd.Series(limits).reindex(mk.index))}
    w = ev.dropna(subset=["ev_win_cal"])
    w = w[w["has_tansho"]]
    out["ev_win"] = decile_gap(w["ev_win_cal"].to_numpy(float), w["ret_win"].to_numpy(float), w["race_id"].to_numpy())
    out["ev_win_raw_overall"] = {"ev": float(w["ev_win_raw"].mean()), "realized": float(w["ret_win"].mean())}
    p = ev.dropna(subset=["ev_place_cal"])
    p = p[p["has_fukusho"]]
    out["ev_place"] = decile_gap(p["ev_place_cal"].to_numpy(float), p["ret_place"].to_numpy(float), p["race_id"].to_numpy())
    out["ev_place_raw_overall"] = {"ev": float(p["ev_place_raw"].mean()), "realized": float(p["ret_place"].mean())}
    return out


def run_exotic(pred: pd.DataFrame, res: pd.DataFrame, max_races: int, seed: int = 0) -> dict:
    from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution

    m = pred.merge(res[["race_id", "horse_id", "finish_position"]], on=["race_id", "horse_id"])
    m = m[m["finish_position"] >= 1]
    ids = [r for r, g in m.groupby("race_id") if 5 <= len(g) <= 18]
    rng = np.random.default_rng(seed)
    if len(ids) > max_races:
        ids = list(rng.choice(ids, max_races, replace=False))
    acc = {(k, mdl): ([], []) for k in ("umaren", "wide", "sanrenpuku") for mdl in ("fit", "harville")}
    alphas = []
    for rid in ids:
        g = m[m["race_id"] == rid].sort_values("p_win", ascending=False).reset_index(drop=True)
        fp = g["finish_position"].to_numpy()
        win, t2, t3 = g["p_win"].to_numpy(float), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float)
        dist, rep = fit_rank_distribution(win, t2, t3)
        alphas.append((rep.alpha2, rep.alpha3))
        hv = RankDistribution(win, 1.0, 1.0)
        n = len(g)
        first2 = {i for i in range(n) if fp[i] <= 2}
        first3 = {i for i in range(n) if fp[i] <= 3}
        if len(first2) != 2 or len(first3) != 3:
            continue
        for mdl, d in (("fit", dist), ("harville", hv)):
            um, wd, sp = d.all_umaren(), d.all_wide(), d.all_sanrenpuku()
            acc[("umaren", mdl)][0].extend(um.values())
            acc[("umaren", mdl)][1].extend(1.0 if set(k) == first2 else 0.0 for k in um)
            acc[("wide", mdl)][0].extend(wd.values())
            acc[("wide", mdl)][1].extend(1.0 if set(k) <= first3 else 0.0 for k in wd)
            acc[("sanrenpuku", mdl)][0].extend(sp.values())
            acc[("sanrenpuku", mdl)][1].extend(1.0 if set(k) == first3 else 0.0 for k in sp)
    out = {"races": len(ids), "alpha2_mean": float(np.mean([a[0] for a in alphas])), "alpha3_mean": float(np.mean([a[1] for a in alphas]))}
    for (k, mdl), (ps, ys) in acc.items():
        p, y = np.asarray(ps, float), np.asarray(ys, float)
        out.setdefault(k, {})[mdl] = {"combos": int(len(p)), "logloss": binary_ll(y, p), "ece": ece_bins(p, y), "mean_p": float(p.mean()), "mean_y": float(y.mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/page_reference/tables")
    ap.add_argument("--pred-dir", default="data/local/inference/predictions_backfill/bundle_v4_2025")
    ap.add_argument("--max-exotic-races", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tables = Path(a.tables)
    from src.scripts.data.readiness_eval import load_preds, load_results

    res = load_results(tables)
    full = pd.read_parquet(tables / "2026" / "race_result_flat.parquet", columns=["race_id", "horse_id", "horse_number", "payoff"])
    full["race_id"] = full["race_id"].astype(str)
    full["horse_id"] = full["horse_id"].astype(str)
    res = res.merge(full[["race_id", "horse_id", "payoff"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    pred = load_preds([Path(a.pred_dir)])
    o = pd.read_parquet(tables / "2026" / "race_odds_flat.parquet", columns=["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "snapshot_at"])
    o["race_id"] = o["race_id"].astype(str)
    o = o.sort_values("snapshot_at").drop_duplicates(["race_id", "horse_number"], keep="last")
    for c in ("win_odds", "place_odds_min", "place_odds_max"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = o[o["win_odds"] >= 1.0]
    out = {"ev_marks": run_ev_marks(pred, res, o), "exotic": run_exotic(pred, res, a.max_exotic_races)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"marks": out["ev_marks"]["marks"], "exotic": out["exotic"]}, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
