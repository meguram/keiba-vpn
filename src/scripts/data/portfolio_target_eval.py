"""目標払戻つきの買い目配分（500% 設計・トリガミ回避）の構築と検証。

    python -m src.scripts.data.portfolio_target_eval build      # 年ごとの候補表（的中確率 MB60 × 想定オッズ × 実払戻）
    python -m src.scripts.data.portfolio_target_eval grid       # 2022〜2024 でパラメータ格子を評価し、設定を選ぶ
    python -m src.scripts.data.portfolio_target_eval final      # 選んだ設定とベースラインを 2022〜2025 で評価（区間つき）
    python -m src.scripts.data.portfolio_target_eval all

方針:
- 計画に使うのはレース前に分かる情報だけ: 自前の勝率/連対率/複勝率（tp36/frame_all の p_win/p_top2/p_top3）→ MB60 で各券種の的中確率、想定オッズ（odds_pred_eval の win_B/place_B と odds_H2）。
  T-X 時点オッズ・確定オッズは計画に使わない。回収率は実払戻（betting/payoff.parquet）で計算。
- パラメータの選択は 2022〜2024 のみ。2025 年は選択に使わず検証。2026 年は想定オッズが未作成のため対象外。
- データ監査の除外レース（odds_audit/races_*.txt の 4 本）と、複勝・ワイドの曖昧な払戻のレースは外す。
出力: data/local/modeling/portfolio_target/（PORTFOLIO_OUT で変更）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.pipeline.inference import portfolio_target as PT  # noqa: E402

try:
    from src.pipeline.inference.exotic_probabilities import fit_blend_distribution  # noqa: E402
except ImportError:  # ホストには MB60 版が別名で置かれている
    from src.pipeline.inference.exotic_probabilities_mb60 import fit_blend_distribution  # type: ignore  # noqa: E402

BET = ROOT / "data/local/modeling/betting"
OP = ROOT / "data/local/modeling/odds_pred"
FRAME = BET / "tp36/frame_all.parquet"
AUD = ROOT / "data/local/meta/odds_audit"
OUT = Path(os.environ.get("PORTFOLIO_OUT", ROOT / "data/local/modeling/portfolio_target"))
YEARS = [2022, 2023, 2024, 2025]
SEL_YEARS = [2022, 2023, 2024]
BT = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"]
BT_JA = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複"}
EXCLUDE_FILES = ["races_null_payoff_races.txt", "races_finish_fixed_wide.txt",
                 "races_finish_position_le0_with_finish_time.txt", "races_payoff_fix_differs.txt"]
TYPESETS = {"ALL": {0, 1, 2, 3, 4, 5}, "EXOTIC": {2, 3, 4, 5}, "TANSHO": {0}, "FUKUSHO": {1}, "UMAREN": {2},
            "WIDE": {3}, "UMATAN": {4}, "SANRENPUKU": {5}}
GRID = {"typeset": list(TYPESETS), "max_bets": [1, 3, None], "mult": [1.0, 2.0, 3.0, 5.0], "k": [1.0, 1.25, 1.5, 2.0], "tau": [0.0, 1.0]}
BUDGET = PT.DEFAULT_BUDGET
N_KEEP_EV, N_KEEP_P = 40, 25


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def excluded_races() -> set[str]:
    s: set[str] = set()
    for f in EXCLUDE_FILES:
        p = AUD / f
        if p.exists():
            s |= {x.strip() for x in p.read_text().split() if x.strip()}
    return s


def load_payoff() -> tuple[pd.DataFrame, set[str]]:
    pay = pd.read_parquet(BET / "payoff.parquet")
    pay["race_id"] = pay["race_id"].astype(str)
    amb = set(pay.loc[pay["ambiguous"].astype(bool) & pay["bet_type"].isin(["fukusho", "wide"]), "race_id"])
    pay = pay[pay["bet_type"].isin(BT)]
    pay = pay.drop_duplicates(["race_id", "bet_type", "combo"], keep="last")
    return pay[["race_id", "bet_type", "combo", "payout"]].rename(columns={"payout": "pay100"}), amb


# ───────────────────────── build ─────────────────────────

def _race_rows(rid: str, hn: np.ndarray, pw, p2, p3, win_B, place_lo, pop):
    """1 レースの候補（全券種）。hn は馬番の昇順。"""
    n = len(hn)
    dist, _ = fit_blend_distribution(pw, p2, p3)
    w = dist.win
    t3 = dist.top3() if n >= 8 else dist.top2()
    P2, P3 = dist._p2, dist._p3
    rows = {"bt": [], "combo": [], "p": [], "hn": [], "pop": [], "p_rank": [], "wB_rank": []}
    p_rank = (-w).argsort().argsort() + 1
    wb_rank = win_B.argsort().argsort() + 1
    for i in range(n):
        rows["bt"].append(0); rows["combo"].append(str(hn[i])); rows["p"].append(w[i]); rows["hn"].append(hn[i])
        rows["pop"].append(pop[i]); rows["p_rank"].append(p_rank[i]); rows["wB_rank"].append(wb_rank[i])
    for i in range(n):
        rows["bt"].append(1); rows["combo"].append(str(hn[i])); rows["p"].append(t3[i]); rows["hn"].append(hn[i])
        rows["pop"].append(pop[i]); rows["p_rank"].append(p_rank[i]); rows["wB_rank"].append(wb_rank[i])
    ii, jj = np.triu_indices(n, 1)
    pu = P2[ii, jj] + P2[jj, ii]
    for a, b, v in zip(ii, jj, pu):
        rows["bt"].append(2); rows["combo"].append(f"{hn[a]}-{hn[b]}"); rows["p"].append(v)
    tri = np.array(list(itertools.combinations(range(n), 3)))
    ti, tj, tk = tri[:, 0], tri[:, 1], tri[:, 2]
    pt = P3[ti, tj, tk] + P3[ti, tk, tj] + P3[tj, ti, tk] + P3[tj, tk, ti] + P3[tk, ti, tj] + P3[tk, tj, ti]
    W = np.zeros((n, n))
    np.add.at(W, (ti, tj), pt); np.add.at(W, (ti, tk), pt); np.add.at(W, (tj, tk), pt)
    for a, b in zip(ii, jj):
        rows["bt"].append(3); rows["combo"].append(f"{hn[a]}-{hn[b]}"); rows["p"].append(W[a, b])
    for a in range(n):
        for b in range(n):
            if a != b:
                rows["bt"].append(4); rows["combo"].append(f"{hn[a]}>{hn[b]}"); rows["p"].append(P2[a, b])
    for a, b, c, v in zip(ti, tj, tk, pt):
        rows["bt"].append(5); rows["combo"].append(f"{hn[a]}-{hn[b]}-{hn[c]}"); rows["p"].append(v)
    m = len(rows["bt"])
    for key in ("hn", "pop", "p_rank", "wB_rank"):
        rows[key] += [np.nan] * (m - len(rows[key]))
    # 単勝・複勝の想定オッズはここで付ける
    odds = np.full(m, np.nan)
    odds[:n] = win_B
    odds[n:2 * n] = place_lo
    rows["o"] = odds
    out = pd.DataFrame(rows)
    out.insert(0, "race_id", rid)
    out["n"] = n
    return out


def build_year(year: int, r_lo: float, excl: set[str], pay: pd.DataFrame, amb: set[str]) -> dict:
    t0 = time.time()
    fr = pd.read_parquet(FRAME, columns=["race_id", "date", "horse_number", "horse_id", "popularity", "p_win", "p_top2", "p_top3", "year"])
    fr = fr[(fr["year"] == year) & fr["p_win"].notna() & (fr["horse_number"] > 0)].copy()
    fr["race_id"], fr["horse_id"] = fr["race_id"].astype(str), fr["horse_id"].astype(str)
    wp = pd.read_parquet(OP / "wp_oof_pred.parquet", columns=["race_id", "horse_id", "win_B", "place_B"])
    wp["race_id"], wp["horse_id"] = wp["race_id"].astype(str), wp["horse_id"].astype(str)
    d = fr.merge(wp, on=["race_id", "horse_id"], how="inner")
    tan_ok = set(pay.loc[pay["bet_type"] == "tansho", "race_id"])
    drop = {"excluded": 0, "ambiguous": 0, "no_tansho": 0, "small_field": 0}
    parts, dates = [], {}
    for rid, g in d.groupby("race_id", sort=True):
        if rid in excl:
            drop["excluded"] += 1; continue
        if rid in amb:
            drop["ambiguous"] += 1; continue
        if rid not in tan_ok:
            drop["no_tansho"] += 1; continue
        if len(g) < 5:
            drop["small_field"] += 1; continue
        g = g.sort_values("horse_number")
        hn = g["horse_number"].to_numpy().astype(int)
        try:
            part = _race_rows(rid, hn, g["p_win"].to_numpy(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy(),
                              g["win_B"].to_numpy(), g["place_B"].to_numpy() * r_lo, g["popularity"].to_numpy().astype(float))
        except Exception as exc:  # noqa: BLE001
            drop["small_field"] += 1; _log(f"{rid}: {exc}"); continue
        parts.append(part)
        dates[rid] = str(g["date"].iloc[0])
    cand = pd.concat(parts, ignore_index=True)
    cand["bt_name"] = cand["bt"].map(dict(enumerate(BT)))
    # 複数頭券種の想定オッズ（odds_H2）
    for bt in BT[2:]:
        col = os.environ.get("PORTFOLIO_ODDS_COL", "odds_H2")  # H0=係数なし（テイクバック÷確率）、H2=2 次多項式（既定）
        od = pd.read_parquet(OP / f"odds_pred_{bt}.parquet", columns=["race_id", "combo", "year", col]).rename(columns={col: "odds_H2"})
        od = od[od["year"] == year][["race_id", "combo", "odds_H2"]]
        od["race_id"] = od["race_id"].astype(str)
        m = cand["bt_name"] == bt
        mm = cand.loc[m, ["race_id", "combo"]].merge(od, on=["race_id", "combo"], how="left")
        cand.loc[m, "o"] = mm["odds_H2"].to_numpy()
    cand = cand[cand["o"].notna()].copy()
    # 刈り込み（複数頭券種のみ）: EV 上位 + 的中確率上位
    cand["ev"] = cand["p"] * cand["o"]
    big = cand["bt"] >= 2
    sub = cand[big].copy()
    sub["r_ev"] = sub.groupby(["race_id", "bt"])["ev"].rank(ascending=False, method="first")
    sub["r_p"] = sub.groupby(["race_id", "bt"])["p"].rank(ascending=False, method="first")
    sub = sub[(sub["r_ev"] <= N_KEEP_EV) | (sub["r_p"] <= N_KEEP_P)]
    cand = pd.concat([cand[~big], sub.drop(columns=["r_ev", "r_p"])], ignore_index=True)
    cand = cand.merge(pay.rename(columns={"bet_type": "bt_name"}), on=["race_id", "bt_name", "combo"], how="left")
    cand["pay100"] = cand["pay100"].fillna(0.0)
    cand["year"] = year
    cand["date"] = cand["race_id"].map(dates)
    cand = cand.sort_values(["race_id", "bt", "combo"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cand.drop(columns=["bt_name"]).to_parquet(OUT / f"cands_{year}.parquet", index=False)
    info = {"year": year, "races": int(cand["race_id"].nunique()), "rows": int(len(cand)), "drop": drop, "sec": round(time.time() - t0)}
    _log(f"{year}: {info}")
    return info


def cmd_build(a) -> None:
    excl = excluded_races()
    pay, amb = load_payoff()
    wp = pd.read_parquet(OP / "wp_oof_pred.parquet", columns=["year", "place_odds_min", "place_mid"])
    w = wp[wp["year"].isin(SEL_YEARS)]
    r_lo = float((w["place_odds_min"] / w["place_mid"]).median())
    _log(f"複勝の下限/中央の比 r_lo={r_lo:.3f}, 除外レース {len(excl)}, 曖昧 {len(amb)}")
    years = [int(y) for y in a.years.split(",")] if a.years else YEARS
    with mp.get_context("fork").Pool(min(4, len(years))) as pool:
        res = pool.starmap(build_year, [(y, r_lo, excl, pay, amb) for y in years])
    (OUT / "build_info.json").write_text(json.dumps({"r_lo": r_lo, "years": res}, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── 評価 ─────────────────────────

def load_races(years: list[int]) -> list[dict]:
    d = pd.concat([pd.read_parquet(OUT / f"cands_{y}.parquet") for y in years], ignore_index=True)
    races = []
    for rid, g in d.groupby("race_id", sort=True):
        races.append({"race_id": rid, "year": int(g["year"].iloc[0]), "date": g["date"].iloc[0], "bt": g["bt"].to_numpy().astype(int),
                      "combo": g["combo"].to_numpy(), "p": g["p"].to_numpy(), "o": g["o"].to_numpy(), "pay": g["pay100"].to_numpy(),
                      "hn": g["hn"].to_numpy(), "pop": g["pop"].to_numpy(), "p_rank": g["p_rank"].to_numpy(), "wB_rank": g["wB_rank"].to_numpy()})
    return races


def _eval_chunk(args):
    races, configs = args
    acc = {c: np.zeros((len(YEARS), 6)) for c in configs}  # 年ごと: 買うR, 的中R, トリガミR, 到達R, 購入額, 払戻
    for R in races:
        yi = YEARS.index(R["year"])
        for c in configs:
            ts, mb, M, k, tau = c
            params = PT.TargetParams(mult=M, k=k, tau=tau, max_bets=mb)
            allowed = np.isin(R["bt"], list(TYPESETS[ts]))
            idx, st = PT.select_portfolio(R["p"], R["o"], params, allowed)
            if idx.size == 0:
                continue
            oc = PT.race_outcome(st, R["pay"][idx], M, BUDGET)
            a = acc[c][yi]
            a[0] += 1; a[1] += oc["hit"]; a[2] += oc["torigami"]; a[3] += oc["reach"]; a[4] += oc["spent"]; a[5] += oc["pay"]
    return acc


def cmd_grid(a) -> None:
    races = load_races(SEL_YEARS)
    configs = [(ts, mb, M, k, tau) for ts in GRID["typeset"] for mb in GRID["max_bets"] for M in GRID["mult"]
               for k in GRID["k"] for tau in GRID["tau"]]
    _log(f"格子 {len(configs)} 設定 × {len(races)} レース（2022〜2024）")
    nproc = int(os.environ.get("PORTFOLIO_PROCS", "4"))
    chunks = [races[i::nproc] for i in range(nproc)]
    with mp.get_context("fork").Pool(nproc) as pool:
        parts = pool.map(_eval_chunk, [(ch, configs) for ch in chunks])
    rows = []
    for c in configs:
        tot = sum(p[c] for p in parts)
        pooled = tot.sum(axis=0)
        row = {"typeset": c[0], "max_bets": c[1], "mult": c[2], "k": c[3], "tau": c[4], "bet_races": int(pooled[0]),
               "hit_races": int(pooled[1]), "torigami": int(pooled[2]), "reach": int(pooled[3]),
               "spent": float(pooled[4]), "pay": float(pooled[5]), "roi": float(pooled[5] / pooled[4]) if pooled[4] > 0 else np.nan}
        for yi, y in enumerate(YEARS[:3]):
            row[f"roi_{y}"] = float(tot[yi][5] / tot[yi][4]) if tot[yi][4] > 0 else np.nan
            row[f"bet_{y}"] = int(tot[yi][0])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "grid_results.parquet", index=False)
    _log(f"格子の評価完了 → grid_results.parquet（{len(df)} 行）")


def _boot_roi(pay: np.ndarray, spent: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(pay)
    if spent.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    pt = float(pay.sum() / spent.sum())
    ix = rng.integers(0, n, size=(n_boot, n))
    s, p = spent[ix].sum(axis=1), pay[ix].sum(axis=1)
    r = np.where(s > 0, p / np.where(s > 0, s, 1), np.nan)
    return pt, float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def summarize(outcomes: pd.DataFrame, mult: float) -> dict:
    """outcomes: 全レース 1 行（見送りは spent=0）。"""
    n_all = len(outcomes)
    bet = outcomes[outcomes["spent"] > 0]
    nb = len(bet)
    hit = bet[bet["pay"] > 0]
    pt, lo, hi = _boot_roi(outcomes["pay"].to_numpy(), outcomes["spent"].to_numpy())
    pnl = (bet["pay"] - bet["spent"]).to_numpy()
    srt = outcomes.sort_values(["date", "race_id"])
    cum = np.cumsum((srt["pay"] - srt["spent"]).to_numpy())
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
    return {"races": n_all, "bet_races": nb, "skip_rate": round(1 - nb / n_all, 3) if n_all else None,
            "hit_rate": round(len(hit) / nb, 4) if nb else None,
            "torigami_rate_of_hits": round(float((hit["pay"] < hit["spent"]).mean()), 3) if len(hit) else None,
            "reach_rate_of_hits": round(float((hit["pay"] >= mult * BUDGET - 1e-9).mean()), 3) if len(hit) else None,
            "roi": round(pt, 4) if pt == pt else None, "roi_lo": round(lo, 4) if lo == lo else None, "roi_hi": round(hi, 4) if hi == hi else None,
            "mean_pnl": round(float(pnl.mean()), 1) if nb else None,
            "pnl_p5_p50_p95": [round(float(np.percentile(pnl, q)), 0) for q in (5, 50, 95)] if nb else None,
            "mean_spent": round(float(bet["spent"].mean()), 0) if nb else None, "max_drawdown_yen": round(dd, 0)}


def run_design(races: list[dict], c: tuple) -> pd.DataFrame:
    ts, mb, M, k, tau = c
    params = PT.TargetParams(mult=M, k=k, tau=tau, max_bets=mb)
    rows = []
    for R in races:
        allowed = np.isin(R["bt"], list(TYPESETS[ts]))
        idx, st = PT.select_portfolio(R["p"], R["o"], params, allowed)
        if idx.size == 0:
            rows.append((R["race_id"], R["year"], R["date"], 0.0, 0.0, 0)); continue
        oc = PT.race_outcome(st, R["pay"][idx], M, BUDGET)
        rows.append((R["race_id"], R["year"], R["date"], oc["spent"], oc["pay"], len(idx)))
    return pd.DataFrame(rows, columns=["race_id", "year", "date", "spent", "pay", "n_bets"])


def baseline(races: list[dict], name: str) -> pd.DataFrame:
    rows = []
    for R in races:
        idx, st = np.empty(0, dtype=int), np.empty(0)
        bt = R["bt"]
        if name in ("fav_actual", "fav_pred", "top_pwin"):
            key = {"fav_actual": ("pop", 1), "fav_pred": ("wB_rank", 1), "top_pwin": ("p_rank", 1)}[name]
            j = np.flatnonzero((bt == 0) & (R[key[0]] == key[1]))
            if j.size:
                idx, st = j[:1], np.array([float(BUDGET)])
        elif name == "umaren_top2":
            top = R["hn"][(bt == 0) & (R["p_rank"] <= 2)]
            if len(top) == 2:
                combo = f"{int(min(top))}-{int(max(top))}"
                j = np.flatnonzero((bt == 2) & (R["combo"] == combo))
                if j.size:
                    idx, st = j[:1], np.array([float(BUDGET)])
        elif name == "ev_uniform":
            idx, st = PT.equal_split(R["p"] * R["o"], 5, BUDGET, 1.1)
        elif name == "kelly_like":
            idx, st = PT.kelly_stakes(R["p"], R["o"], BUDGET)
        if idx.size == 0:
            rows.append((R["race_id"], R["year"], R["date"], 0.0, 0.0, 0)); continue
        pay = PT.settle(st, R["pay"][idx])
        rows.append((R["race_id"], R["year"], R["date"], float(st.sum()), pay, len(idx)))
    return pd.DataFrame(rows, columns=["race_id", "year", "date", "spent", "pay", "n_bets"])


BASELINES = {"fav_actual": "1番人気の単勝（確定人気・参考）", "fav_pred": "想定オッズ最小馬の単勝", "top_pwin": "予測勝率1位の単勝",
             "umaren_top2": "予測勝率上位2頭の馬連1点", "ev_uniform": "想定EV≥1.1の上位5点を均等買い", "kelly_like": "EV≥1.05の分数ケリー（従来型の近似）"}


def pick_configs(df: pd.DataFrame, min_bets: int = 300, min_per_year: int = 60) -> dict[str, tuple]:
    ok = df[(df["bet_races"] >= min_bets) & (df[[f"bet_{y}" for y in SEL_YEARS]].min(axis=1) >= min_per_year)]
    picks = {}
    for M in GRID["mult"]:
        s = ok[ok["mult"] == M].sort_values("roi", ascending=False)
        if len(s):
            r = s.iloc[0]
            picks[f"best_M{M:g}"] = (r["typeset"], None if pd.isna(r["max_bets"]) else int(r["max_bets"]), float(r["mult"]), float(r["k"]), float(r["tau"]))
    return picks


def cmd_final(a) -> None:
    df = pd.read_parquet(OUT / "grid_results.parquet")
    picks = pick_configs(df)
    races = load_races(YEARS)
    by_year = {y: [r for r in races if r["year"] == y] for y in YEARS}
    report = {"picks": {k: list(v) for k, v in picks.items()}, "designs": {}, "baselines": {}}
    for name, c in picks.items():
        report["designs"][name] = {"config": list(c), "pooled_2022_2024": summarize(run_design([r for y in SEL_YEARS for r in by_year[y]], c), c[2]),
                                   **{str(y): summarize(run_design(by_year[y], c), c[2]) for y in YEARS}}
        _log(f"{name} {c}: 2022-24 roi={report['designs'][name]['pooled_2022_2024']['roi']} 2025 roi={report['designs'][name]['2025']['roi']}")
    for b in BASELINES:
        out = {}
        for key, ys in (("pooled_2022_2024", SEL_YEARS), ("2025", [2025])):
            out[key] = summarize(baseline([r for y in ys for r in by_year[y]], b), 5.0)
        report["baselines"][b] = {"label": BASELINES[b], **out}
        _log(f"{b}: 2022-24 roi={out['pooled_2022_2024']['roi']} 2025 roi={out['2025']['roi']}")
    # M の感度（2022〜2024 の格子から、M ごとの k × 買い目数別の概況）
    sens = df[df["typeset"].isin(["ALL", "EXOTIC", "UMAREN", "SANRENPUKU"])].copy()
    sens["feasible_rate"] = sens["bet_races"] / sum(len(by_year[y]) for y in SEL_YEARS)
    report["sensitivity"] = sens.sort_values(["mult", "k"]).to_dict("records")
    (OUT / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("final_report.json を保存")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--years", default="")
    sp.add_parser("grid"); sp.add_parser("final"); sp.add_parser("all")
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        if a.cmd == "all":
            a.years = ""
        cmd_build(a)
    if a.cmd in ("grid", "all"):
        cmd_grid(a)
    if a.cmd in ("final", "all"):
        cmd_final(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
