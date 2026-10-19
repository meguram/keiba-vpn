"""3連単の想定オッズ（単勝市場の確率からの割引 Harville）と実際の払戻の較正。

(A) 2020〜2025: 勝った組合せ(1着→2着→3着)の払戻/100 と、想定オッズ 0.725 / P_harville(組合せ) の比。
    単勝の確定オッズ(race_result_flat.odds)を 1/odds で正規化して市場確率にし、2・3着の確率は p^γ で割り引く（γ<1 で本命寄りを弱める）。
    γ ごとに log(実払戻/想定) の平均（偏り）・標準偏差・MAE を出す。同着や払戻が読めないレースは除く。
(B) 取得済みの全組合せオッズ(race_trifecta_odds)がある年は、全組合せの想定と実際の log 比（人気順位帯別）。

python -m src.scripts.data.trifecta_est_calibration [--years 2020 2021 ...] [--out /tmp/trifecta_est_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
RETURN_RATE = 0.725
GAMMAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def harville3(p: np.ndarray, a: int, b: int, c: int, g: float) -> float:
    """P(a→b→c)。1着は p、2着以降は p^g を残りで正規化する（g=1 が通常の Harville）。"""
    q = p ** g
    pb = q[b] / (q.sum() - q[a])
    pc = q[c] / (q.sum() - q[a] - q[b])
    return float(p[a] * pb * pc)


MAX_YEN = 20_000_000       # 3連単の払戻の上限の目安。これを超える値は複数行の連結（払戻修復の対象）


def win_odds_from_store(race_id: str) -> dict[int, float]:
    global _ST
    if _ST is None:
        from src.scraper.storage import HybridStorage

        _ST = HybridStorage()
    o = _ST.load("race_odds", race_id) or {}
    return {int(e["horse_number"]): float(e["win_odds"]) for e in o.get("entries") or [] if e.get("win_odds") and e["win_odds"] > 0}


_ST = None


def winning_triples(year: int) -> list[dict]:
    import polars as pl

    f = TABLES / str(year) / "race_result_flat.parquet"
    if not f.exists():
        return []
    d = pl.read_parquet(f, columns=["race_id", "horse_number", "odds", "payoff", "finish_position"])
    out = []
    for rid, g in d.group_by("race_id"):
        rid = rid[0]
        pay = g["payoff"][0]
        if not pay:
            continue
        try:
            x = json.loads(pay).get("三連単")
        except Exception:
            continue
        if not isinstance(x, dict):
            continue
        m = re.fullmatch(r"\s*(\d+)\s*→\s*(\d+)\s*→\s*(\d+)\s*", str(x.get("numbers", "")))
        try:
            yen = float(str(x.get("payout", "")).replace(",", ""))
        except ValueError:
            continue
        if not m or yen <= 0 or yen > MAX_YEN:
            continue
        nums = {int(r["horse_number"]): r["odds"] for r in g.iter_rows(named=True) if r["odds"] and r["odds"] > 0}
        if not 1.05 < sum(1.0 / o for o in nums.values()) < 1.6:      # 2025 年の race_result_flat.odds は単勝オッズではない（Σ1/odds≈0.21）
            nums = win_odds_from_store(rid)
        tri = tuple(int(v) for v in m.groups())
        if len(nums) < 5 or any(t not in nums for t in tri):
            continue
        out.append({"race_id": rid, "year": year, "odds": nums, "tri": tri, "payout": yen / 100.0})
    return out


def summarize(rows: list[dict]) -> dict:
    res: dict = {"n_races": len(rows), "by_gamma": {}}
    for g in GAMMAS:
        lr = []
        for r in rows:
            nums = sorted(r["odds"])
            inv = np.array([1.0 / r["odds"][n] for n in nums])
            p = inv / inv.sum()
            idx = {n: i for i, n in enumerate(nums)}
            a, b, c = (idx[t] for t in r["tri"])
            est = RETURN_RATE / harville3(p, a, b, c, g)
            lr.append(np.log(r["payout"] / est))
        lr = np.array(lr)
        res["by_gamma"][str(g)] = {"mean_log_ratio": round(float(lr.mean()), 4), "std": round(float(lr.std()), 4),
                                   "mae": round(float(np.abs(lr).mean()), 4), "median_ratio": round(float(np.exp(np.median(lr))), 3),
                                   "mean_ratio": round(float(np.exp(lr).mean()), 3)}
    best = min(res["by_gamma"].items(), key=lambda kv: abs(kv[1]["mean_log_ratio"]))
    res["best_gamma_by_bias"] = best[0]
    return res


def full_grid_compare(years: list[int], max_races: int = 3000) -> dict:
    """取得済みの全組合せオッズと想定オッズ（γ別）の比較。人気順位帯別の log(実/想定) の平均。"""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.scraper.storage import HybridStorage

    import polars as pl

    st = HybridStorage()
    out: dict = {"races": 0, "bands": {}}
    acc: dict[float, dict[str, list[float]]] = {g: {} for g in GAMMAS}
    bands = [(1, 10), (11, 50), (51, 200), (201, 1000), (1001, 10**9)]
    n = 0
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        win = pl.read_parquet(f, columns=["race_id", "horse_number", "odds"])
        for rid, g in win.group_by("race_id"):
            rid = rid[0]
            if n >= max_races:
                break
            if not st.exists("race_trifecta_odds", rid):
                continue
            doc = st.load("race_trifecta_odds", rid)
            odds = {r["horse_number"]: r["odds"] for r in g.iter_rows(named=True) if r["odds"] and r["odds"] > 0}
            if not 1.05 < sum(1.0 / o for o in odds.values()) < 1.6:
                odds = win_odds_from_store(rid)
            rows = doc.get("trifecta") or []
            if len(odds) < 5 or not rows:
                continue
            nums = sorted(odds)
            inv = np.array([1.0 / odds[k] for k in nums])
            p = inv / inv.sum()
            idx = {k: i for i, k in enumerate(nums)}
            n += 1
            for gm in GAMMAS:
                for a, b, c, o, pop in rows:
                    if a not in idx or b not in idx or c not in idx or o <= 0 or pop <= 0:
                        continue
                    est = RETURN_RATE / harville3(p, idx[a], idx[b], idx[c], gm)
                    for lo, hi in bands:
                        if lo <= pop <= hi:
                            acc[gm].setdefault(f"{lo}-{hi}", []).append(float(np.log(o / est)))
                            break
    out["races"] = n
    for gm in GAMMAS:
        out["bands"][str(gm)] = {k: {"n": len(v), "mean_log_ratio": round(float(np.mean(v)), 4), "mae": round(float(np.mean(np.abs(v))), 4)} for k, v in acc[gm].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--grid-years", type=int, nargs="*", default=[2026, 2025, 2024])
    ap.add_argument("--out", default="/tmp/trifecta_est_calibration.json")
    a = ap.parse_args()
    res: dict = {"winning_combo": {}}
    allrows = []
    for y in a.years:
        rows = winning_triples(y)
        allrows += rows
        res["winning_combo"][str(y)] = summarize(rows) if rows else {}
        print(y, len(rows), flush=True)
    res["winning_combo"]["all"] = summarize(allrows)
    if a.grid_years:
        res["full_grid"] = full_grid_compare(a.grid_years)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["winning_combo"]["all"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
