"""『全券種・縛りなし・M=5』パターン探索（portfolio_pattern）の検証。現行の貪欲法（3連複のみ型・全券種型）と同じ候補表で比べる。

python -m src.scripts.data.portfolio_pattern_eval run      # 年ごとに並列。結果 pattern_outcomes.parquet・pattern_report.json
python -m src.scripts.data.portfolio_pattern_eval exact    # ビームサーチと厳密な全列挙（上位25件・K≤3）の差を数十レースで確認
"""

from __future__ import annotations

import json
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.pipeline.inference import portfolio_pattern as PP
from src.scripts.data import odds_pred_eval as OP_E
from src.scripts.data import portfolio_target_eval as E
from src.scripts.data import portfolio_tri as T

CONFIGS = [(k, tau) for k in (1.5, 2.0, 2.5) for tau in (0.9, 1.0, 1.1)]


def _parse(s: str, pos: dict[int, int]) -> tuple[int, ...]:
    return tuple(pos[int(x)] for x in re.split(r"[->]", str(s)))


def _race_inputs(R: dict, fr: pd.DataFrame):
    g = fr[fr["race_id"] == R["race_id"]].sort_values("horse_number")
    hn = g["horse_number"].to_numpy().astype(int)
    w = g["p_win"].to_numpy(float)
    Do = OP_E._dist(w / w.sum(), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float))
    perms = PP.Perms(len(hn))
    pp = Do._p3[perms.a, perms.b, perms.c]
    pos = {int(h): i for i, h in enumerate(hn)}
    combo = [_parse(c, pos) for c in R["combo"]]
    return perms, pp / pp.sum(), combo


def _year(args):
    year, configs, limit = args
    races = T.load_races([year])[: limit or None]
    fr = pd.read_parquet(E.FRAME, columns=["race_id", "horse_number", "p_win", "p_top2", "p_top3", "year"])
    fr = fr[fr["year"] == year].copy()
    fr["race_id"] = fr["race_id"].astype(str)
    fr_by = {rid: g for rid, g in fr.groupby("race_id")}
    rows, times = [], []
    for R in races:
        g = fr_by.get(R["race_id"])
        if g is None:
            continue
        perms, pp, combo = _race_inputs(R, g)
        for (k, tau) in configs:
            prm = PP.PatternParams(mult=5.0, k=k, tau=tau, budget=E.BUDGET)
            t0 = time.time()
            r = PP.select_pattern(R["bt"], combo, R["p"], R["o"], pp, perms, prm)
            times.append(time.time() - t0)
            idx, st = r["idx"], r["stakes"]
            if len(idx) == 0:
                rows.append((year, R["race_id"], R["date"], k, tau, 0.0, 0.0, 0, 0.0, np.nan)); continue
            oc = PT_outcome(st, R["pay"][idx])
            rows.append((year, R["race_id"], R["date"], k, tau, oc[0], oc[1], len(idx), r["hit_prob"], r["ev"]))
    df = pd.DataFrame(rows, columns=["year", "race_id", "date", "k", "tau", "spent", "pay", "n_bets", "hit_prob", "ev"])
    return df, float(np.mean(times)) if times else 0.0


def PT_outcome(stakes, pay100):
    return float(np.sum(stakes)), float(np.sum(np.asarray(stakes) / 100.0 * np.asarray(pay100)))


FRACS = (0.05, 0.10, 0.20, 0.30, 0.50)


def selection(df: pd.DataFrame) -> dict:
    """レース選択: パターンの期待回収率（想定オッズ・k で保守化）の高いレースだけ買う。しきい値は 2022〜24 年の分布で決め、2025 年にもそのまま当てる。"""
    out = {}
    for (k, tau), d in df.groupby(["k", "tau"]):
        sel = d[d["year"].isin(E.SEL_YEARS)]
        allr = len(sel)
        evs = np.sort(sel["ev"].dropna().to_numpy())[::-1]
        res = {"deciles_2022_2024": [], "frac": {}}
        qs = np.quantile(evs, np.linspace(0, 1, 6)) if len(evs) else []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = sel[(sel["ev"] >= lo) & (sel["ev"] <= hi)]
            res["deciles_2022_2024"].append({"ev_lo": round(float(lo), 3), "ev_hi": round(float(hi), 3), "races": len(m), **{x: E.summarize(m, 5.0)[x] for x in ("roi", "roi_lo", "roi_hi", "hit_rate")}})
        for f in FRACS:
            thr = float(evs[min(int(allr * f), len(evs) - 1)]) if len(evs) else np.inf
            row = {"ev_threshold": round(thr, 3)}
            for name, part in (("pooled_2022_2024", sel), ("2025", d[d["year"] == 2025])):
                keep = part.assign(spent=np.where(part["ev"] >= thr, part["spent"], 0.0), pay=np.where(part["ev"] >= thr, part["pay"], 0.0))
                s2 = E.summarize(keep, 5.0)
                row[name] = {**{x: s2[x] for x in ("bet_races", "hit_rate", "torigami_rate_of_hits", "reach_rate_of_hits", "roi", "roi_lo", "roi_hi", "max_drawdown_yen")},
                             "bet_races_per_year": round(s2["bet_races"] / (3 if name != "2025" else 1))}
            res["frac"][f"{int(f * 100)}%"] = row
        out[f"k{k}_tau{tau}"] = res
    return out


def run(limit: int = 0) -> None:
    with Pool(4) as pool:
        res = pool.map(_year, [(y, CONFIGS, limit) for y in E.YEARS])
    df = pd.concat([r[0] for r in res], ignore_index=True)
    df.to_parquet(E.OUT / "pattern_outcomes.parquet", index=False)
    rep = {"sec_per_race_config": float(np.mean([r[1] for r in res])), "designs": {}}
    for (k, tau) in CONFIGS:
        d = df[(df["k"] == k) & (df["tau"] == tau)]
        pooled = d[d["year"].isin(E.SEL_YEARS)]
        rep["designs"][f"k{k}_tau{tau}"] = {"pooled_2022_2024": E.summarize(pooled.assign(), 5.0), "2025": E.summarize(d[d["year"] == 2025], 5.0)}
        s, v = rep["designs"][f"k{k}_tau{tau}"]["pooled_2022_2024"], rep["designs"][f"k{k}_tau{tau}"]["2025"]
        print(f"k={k} tau={tau}: 22-24 roi={s['roi']} [{s['roi_lo']},{s['roi_hi']}] hit={s['hit_rate']} skip={s['skip_rate']} | 2025 roi={v['roi']} [{v['roi_lo']},{v['roi_hi']}] hit={v['hit_rate']} skip={v['skip_rate']}", flush=True)
    rep["selection"] = selection(df)
    (E.OUT / "pattern_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def exact(n: int = 40) -> None:
    races = T.load_races([2024])[:n]
    fr = pd.read_parquet(E.FRAME, columns=["race_id", "horse_number", "p_win", "p_top2", "p_top3", "year"])
    fr = fr[fr["year"] == 2024].copy(); fr["race_id"] = fr["race_id"].astype(str)
    diffs, same, t_ex = [], 0, []
    prm = PP.PatternParams(mult=5.0, k=2.0, tau=1.0, top_n=25, max_k=3)
    for R in races:
        perms, pp, combo = _race_inputs(R, fr[fr["race_id"] == R["race_id"]])
        a = PP.select_pattern(R["bt"], combo, R["p"], R["o"], pp, perms, prm)
        t0 = time.time(); b = PP.select_pattern(R["bt"], combo, R["p"], R["o"], pp, perms, prm, exhaustive=True); t_ex.append(time.time() - t0)
        diffs.append(b["hit_prob"] - a["hit_prob"]); same += int(abs(b["hit_prob"] - a["hit_prob"]) < 1e-9)
    print(f"ビーム vs 厳密: 一致 {same}/{len(diffs)}、厳密の方が高い平均差 {np.mean(diffs):.5f}、最大差 {np.max(diffs):.4f}、厳密の平均 {np.mean(t_ex):.2f}秒/レース")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        exact()
