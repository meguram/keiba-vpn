"""想定単勝の帯別の精度と合格ライン。

2026-11-11 の測定で、厳しい絶対値の線（20 倍以内で差 1.0 以内 80% など）は、T-60 の実オッズを予測に使っても届かないと分かった
（蓄積 209 レース: T-60 実オッズ 41%、T-15 実オッズ 52%）。そこで合格の基準を次の 3 つに置き直した（ユーザー承認 2026-11-11）。
旧線は『旧線（参考）』として出力に残す。

    python -m src.scripts.data.odds_band_eval <OOF ディレクトリ> [<OOF ディレクトリ> ...] [--year 2025]

OOF ディレクトリには wp_raw_oof.parquet（share_raw=log 占有率の予測）がある。較正の直線(log 占有率 ← share_raw の 1 次式)は
2022〜2024 年の OOF で当てはめ、評価年(既定 2025)には当てはめない。帯は『実際の単勝』で切る（ユーザーの見る基準）。

新しい基準（A: 80% 区間、B: 人気順位、C: T-60 水準）:
- A: 想定オッズの帯（想定値で切る）ごとに、2022〜2024 年の OOF の log 誤差の 10%/90% 点で 80% 区間を作り、2025 年で区間に実際の単勝が入る割合が 77〜83%。
- B: 想定 1 位が実際の 1 番人気 75% 以上、実際の 1 番人気が想定上位 3 頭に入る 95% 以上、想定と実際の人気順位の差が ±1 以内 70% 以上。
- C: 差 1.0 以内の割合が T-60 実オッズの水準以上（〜5 倍 64%・5〜10 倍 43%・10〜20 倍 21%・20 倍以内全体 41%・20.1〜50 倍の相対 20% 以内 29%）。

旧線（参考。T-60 実オッズでも届かない）:
- 20 倍以内: 差 1.0 以内の馬が 80% 以上（全体）。さらに 〜5 倍 90%・5〜10 倍 80%・10〜20 倍 70% 以上。
- 20.1〜50 倍: 相対差（|想定−実際|/実際）が 20% 以内の馬が 70% 以上。
- 50 倍超: 5 倍刻みに丸めた想定の log 誤差の中央値が 0.25 以下（丸めてよいので、桁が合っていれば良い）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import odds_pred_eval as oe

PAYBACK = 0.797
T60_LEVEL = {"le5": 0.64, "5_10": 0.43, "10_20": 0.21, "le20_all": 0.41, "20_50_rel20": 0.29}
PRED_EDGES = [0, 2, 3, 5, 7, 10, 15, 20, 30, 50, 1e9]
PASS = {"le20_all": 0.80, "le5": 0.90, "5_10": 0.80, "10_20": 0.70, "20_50_rel20": 0.70, "gt50_logmed": 0.25}


def _round_odds(x: pd.Series) -> pd.Series:
    return (x / 5).round().clip(lower=1) * 5


def interval_and_rank(d: pd.DataFrame, year: int, fit_years: tuple[int, int]) -> dict:
    """d: pred・win_odds・race_id・year を持つ。80% 区間の被覆率（想定帯別）と人気順位の一致。"""
    d = d.copy()
    d["le"] = np.log(d["win_odds"] / d["pred"])
    d["pb"] = pd.cut(d["pred"], PRED_EDGES)
    tr, t = d[d["year"].between(*fit_years)], d[d["year"] == year].copy()
    q = tr.groupby("pb", observed=True)["le"].quantile([0.1, 0.9]).unstack()
    t = t.join(q, on="pb").dropna(subset=[0.1, 0.9])
    t["hit"] = (t["le"] >= t[0.1]) & (t["le"] <= t[0.9])
    cov = t.groupby("pb", observed=True).agg(n=("hit", "size"), coverage=("hit", "mean"), lo_mult=(0.1, lambda x: float(np.exp(x.iloc[0]))), hi_mult=(0.9, lambda x: float(np.exp(x.iloc[0]))))
    t["rk_p"] = t.groupby("race_id")["pred"].rank(method="first")
    t["rk_a"] = t.groupby("race_id")["win_odds"].rank(method="min")
    fav = t["rk_a"] == 1
    out = {
        "coverage_overall": float(t["hit"].mean()),
        "coverage_by_band": {str(k): {"n": int(r["n"]), "coverage": float(r["coverage"]), "lo_mult": r["lo_mult"], "hi_mult": r["hi_mult"]} for k, r in cov.iterrows()},
        "fav_is_pred_top1": float(((t["rk_p"] == 1) & fav).sum() / fav.sum()),
        "fav_in_pred_top3": float(((t["rk_p"] <= 3) & fav).sum() / fav.sum()),
        "rank_within1": float(((t["rk_p"] - t["rk_a"]).abs() <= 1).mean()),
    }
    out["pass_A"] = bool(0.77 <= out["coverage_overall"] <= 0.83 and all(0.74 <= v["coverage"] <= 0.86 for v in out["coverage_by_band"].values() if v["n"] >= 300))
    out["pass_B"] = bool(out["fav_is_pred_top1"] >= 0.75 and out["fav_in_pred_top3"] >= 0.95 and out["rank_within1"] >= 0.70)
    return out


def mode_shift(d: pd.DataFrame, year: int, fit_years: tuple[int, int]) -> dict:
    """差 1.0 以内の割合を上げるための『最頻値への補正』: pred × exp(−k·σ²)。σ は想定帯ごとの log 誤差の標準偏差（fit_years）、k は 1 個の係数。
    log 誤差が右に裾を引くので、中央値より少し小さい値の方が『差 1.0 以内』に入りやすい。k は fit_years の想定 20 倍以下で最大化し、評価年には当てはめない。"""
    mid = np.array([1.5, 2.5, 4, 6, 8.5, 12.5, 17.5, 25, 40, 100])
    f = d[d["year"].between(*fit_years)]
    sd = (np.log(f["win_odds"] / f["pred"])).groupby(pd.cut(f["pred"], PRED_EDGES), observed=False).std(ddof=1)
    sd = sd.ffill().bfill().to_numpy(float)
    sig2 = np.interp(np.log(d["pred"]), np.log(mid), sd) ** 2
    d = d.assign(sig2=sig2)
    trn = d[d["year"].between(*fit_years) & (d["win_odds"] <= 20)]
    ks = np.arange(0.0, 2.01, 0.1)
    hits = [(np.abs(trn["pred"] * np.exp(-k * trn["sig2"]) - trn["win_odds"]) <= 1.0).mean() for k in ks]
    k = float(ks[int(np.argmax(hits))])
    t = d[d["year"] == year]
    adj = t["pred"] * np.exp(-k * t["sig2"])
    err, o = (adj - t["win_odds"]).abs(), t["win_odds"]
    out: dict = {"k": k, "sigma_by_band": [float(x) for x in sd]}
    for name, lo, hi in (("le5", 0, 5), ("5_10", 5, 10), ("10_20", 10, 20), ("le20_all", 0, 20)):
        m = (o > lo) & (o <= hi)
        out[name] = float((err[m] <= 1.0).mean())
    m = (o > 20) & (o <= 50)
    out["20_50_rel20"] = float(((adj[m] - o[m]).abs() / o[m] <= 0.2).mean())
    return out


def evaluate(oof_dir: Path, year: int = 2025, fit_years: tuple[int, int] = (2022, 2024)) -> dict:
    if year >= fit_years[0] and year <= fit_years[1]:
        raise ValueError(f"評価年 {year} が当てはめ期間 {fit_years} に入っている")
    raw = pd.read_parquet(oof_dir / "wp_raw_oof.parquet")
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    wp = oe.load_wp()
    d = raw.merge(wp[oe.KEY + ["win_odds", "p_mkt"]], on=oe.KEY, how="inner").dropna(subset=["share_raw"])
    g = d["share_raw"] - np.log(np.exp(d["share_raw"]).groupby(d["race_id"]).transform("sum"))
    d["p_mkt"] = d["p_mkt"] / d.groupby("race_id")["p_mkt"].transform("sum")
    tr = d["year"].between(*fit_years)
    b1, a1 = np.polyfit(g[tr], np.log(d.loc[tr, "p_mkt"]), 1)
    s = np.exp(a1 + b1 * g)
    s = s / s.groupby(d["race_id"]).transform("sum")
    d["pred"] = np.clip(PAYBACK / s, 1.0, None)
    t = d[d["year"] == year]
    err = (t["pred"] - t["win_odds"]).abs()
    rel = err / t["win_odds"]
    o = t["win_odds"]
    res: dict = {"year": year, "n": int(len(t)), "line": [float(a1), float(b1)]}
    for name, lo, hi in (("le5", 0, 5), ("5_10", 5, 10), ("10_20", 10, 20), ("le20_all", 0, 20)):
        m = (o > lo) & (o <= hi)
        res[name] = {"n": int(m.sum()), "within1": float((err[m] <= 1.0).mean()), "median_abs": float(err[m].median()), "median_rel": float(rel[m].median())}
    m = (o > 20) & (o <= 50)
    res["20_50"] = {"n": int(m.sum()), "within_rel20": float((rel[m] <= 0.2).mean()), "median_rel": float(rel[m].median())}
    m = o > 50
    lr = np.abs(np.log(_round_odds(t.loc[m, "pred"])) - np.log(o[m]))
    res["gt50"] = {"n": int(m.sum()), "logerr_median_rounded": float(lr.median())}
    res["interval_rank"] = interval_and_rank(d, year, fit_years)
    res["mode_shift"] = mode_shift(d, year, fit_years)
    res["pass_C_t60_level"] = {"le5": res["le5"]["within1"] >= T60_LEVEL["le5"], "5_10": res["5_10"]["within1"] >= T60_LEVEL["5_10"], "10_20": res["10_20"]["within1"] >= T60_LEVEL["10_20"],
                               "le20_all": res["le20_all"]["within1"] >= T60_LEVEL["le20_all"], "20_50": res["20_50"]["within_rel20"] >= T60_LEVEL["20_50_rel20"]}
    res["pass_old_reference"] = {
        "le20_all": res["le20_all"]["within1"] >= PASS["le20_all"], "le5": res["le5"]["within1"] >= PASS["le5"],
        "5_10": res["5_10"]["within1"] >= PASS["5_10"], "10_20": res["10_20"]["within1"] >= PASS["10_20"],
        "20_50": res["20_50"]["within_rel20"] >= PASS["20_50_rel20"], "gt50": res["gt50"]["logerr_median_rounded"] <= PASS["gt50_logmed"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    import os

    if os.environ.get("REBUILD_2016") == "1":
        from src.scripts.data import rebuild_groups_2016 as rg

        rg.redirect()
        oe.BET = rg.BASE / "betting"
    for d in a.dirs:
        r = evaluate(Path(d), a.year)
        print(f"== {d}\n" + json.dumps(r, ensure_ascii=False, indent=1))
        ir = r["interval_rank"]
        print(f"   [要約] A 区間被覆 {ir['coverage_overall']:.3f} {'合格' if ir['pass_A'] else '不合格'} / B 1位一致 {ir['fav_is_pred_top1']:.3f}・上位3 {ir['fav_in_pred_top3']:.3f}・±1 {ir['rank_within1']:.3f} {'合格' if ir['pass_B'] else '不合格'} / C {r['pass_C_t60_level']}")


if __name__ == "__main__":
    main()
