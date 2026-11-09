"""ペルソナ 1: 的中率が高くぶれの小さい券種（馬連・ワイド）で、回収率 100% 超の戦略があるかを、persona1_search と同じ手順で検証する。

python -m src.scripts.data.persona1_exotic [--frame PATH] [--first-test 2023] [--last-test 2024] [--out DIR]

入力は persona1_search と同じ表（1 頭 1 行: race_id, date, horse_number, win_odds, p_win, p_top2, p_top3, ret_win など）。
組合せ券の確率は、モデルの p_win・p_top2・p_top3 から順位の分布（fit_rank_distribution）で作る（モデルのみ）。
Benter 混合は、単勝の混合確率（p ∝ p_model^a · q_market^b）で p_win・p_top2・p_top3 を拡縮してから同じ分布を作る（frozen_tracking.umaren_candidates と同じ）。
オッズは確定オッズ（馬連 = odds、ワイド = 下限 odds_min で保守的に）。払戻は payoff.parquet の確定払戻。

候補の格子（事前に固定）: 券種{馬連, ワイド} × 確率{モデルのみ, Benter 混合} × 期待値の下限{1.0, 1.1, 1.2, 1.3, 1.5} × 1 レースの最大点数{1, 2, 3} × 的中確率の下限{なし, 3%, 8%} = 180。
的中確率の下限は『的中率が高くぶれの小さい買い方』のため（下限なしだと、確率の裾の過大評価で平均 800 倍の超大穴の組ばかり選び、数本の超高配当に回収率が左右される）。結果には『上位 3 的中を除いた回収率』も出す。
手順は persona1_search と同じ（前の年だけで a, b と候補を選ぶ・片側 99% の下限・開催日ブロックの Reality Check）。2022 年は a, b を選ぶ前の年が無いので (0.3, 0.8)（tp-36 の値）を使う。
留保: 確定オッズで期待値を計算するため、購入時点では分からない値を使っている（組合せ券の購入時点のオッズの実測はまだ無い）。
GCS は読まない。費用 0 円。
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import fit_rank_distribution
from src.scripts.data import persona1_search as P
from src.scripts.data import tp36_single as t36

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
OUT = ROOT / "data/local/research/persona1"
DEFAULT_AB = (0.3, 0.8)
BETS = ("umaren", "wide")
BLENDS = ("model", "benter")
THETAS = (1.0, 1.1, 1.2, 1.3, 1.5)
KMAX = (1, 2, 3)
PMIN = (0.0, 0.03, 0.08)
MIN_ODDS = 1.0


def _pair_odds_2026(years: set[int]) -> dict:
    """odds_pair.parquet は 2020〜2025 年。それ以降の年は race_pair_odds_flat（確定オッズ）から読む。"""
    out: dict = {}
    for y in sorted(years):
        f = ROOT / f"data/page_reference/tables/{y}/race_pair_odds_flat.parquet"
        if y <= 2025 or not f.exists():
            continue
        o = pd.read_parquet(f, columns=["race_id", "odds_type", "pair", "odds", "odds_min"])
        o = o[o["odds_type"].isin(BETS)].copy()
        o["o"] = np.where(o["odds_type"] == "wide", o["odds_min"], o["odds"])
        o = o[o["o"] > 0]
        for (rid, bt), g in o.groupby(["race_id", "odds_type"], sort=False):
            out.setdefault(str(rid), {})[bt] = {tuple(sorted(int(x) for x in re.findall(r"\d+", str(pr)))): float(v) for pr, v in zip(g["pair"], g["o"])}
    return out


def _pay_flat(years: set[int]) -> dict:
    """payoff.parquet は 2020〜2025 年。それ以降の年は race_result_flat の payoff（JSON）から読む。"""
    import json as _json

    out: dict = {}
    key = {"umaren": "馬連", "wide": "ワイド"}
    for y in sorted(years):
        f = ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet"
        if y <= 2025 or not f.exists():
            continue
        r = pd.read_parquet(f, columns=["race_id", "payoff"]).drop_duplicates("race_id")
        for rid, js in zip(r["race_id"], r["payoff"]):
            try:
                doc = _json.loads(js) if isinstance(js, str) else {}
            except ValueError:
                continue
            for bt, jp in key.items():
                v = doc.get(jp)
                v = [v] if isinstance(v, dict) else (v or [])
                for x in v:
                    try:
                        c = tuple(sorted(int(t) for t in str(x["numbers"]).replace(" ", "").split("-")))
                        out.setdefault((str(rid), bt), {})[c] = float(str(x["payout"]).replace(",", "")) / 100.0
                    except (KeyError, ValueError):
                        pass
    return out


def load_pair_odds(years: set[int]) -> dict:
    """race_id → bet_type → {(a, b): odds}。ワイドは下限。"""
    o = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min"])
    o = o[o["bet_type"].isin(BETS) & o["race_id"].str[:4].astype(int).isin(years)].copy()
    o["o"] = np.where(o["bet_type"] == "wide", o["odds_min"], o["odds"])
    o = o[o["o"] > 0]
    out: dict = {}
    for (rid, bt), g in o.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {t36.parse_combo(c): float(v) for c, v in zip(g["combo"], g["o"])}
    out.update(_pair_odds_2026(years))
    return out


def load_pay(years: set[int]) -> dict:
    """(race_id, bet_type) → {(a, b): 払戻倍率}。"""
    p = pd.read_parquet(BET / "payoff.parquet", columns=["race_id", "bet_type", "combo", "payout", "ambiguous", "parsed"])
    p = p[p["bet_type"].isin(BETS) & p["race_id"].str[:4].astype(int).isin(years) & p["parsed"] & ~p["ambiguous"]]
    out: dict = {}
    for rid, bt, c, v in zip(p["race_id"], p["bet_type"], p["combo"], p["payout"]):
        out.setdefault((rid, bt), {})[tuple(sorted(t36.parse_combo(c)))] = float(v) / 100.0
    out.update(_pay_flat(years))
    return out


def tickets_for_year(d: pd.DataFrame, year: int, ab: tuple[float, float], pair: dict, pay: dict) -> pd.DataFrame:
    """その年のレースごとに、券種 × 確率の種類ごとの期待値上位 max(KMAX) 点を表にする。"""
    x = P.add_blend(d[d["year"] == year], ab)
    rows = []
    for rid, g in x.groupby("race_id", sort=False):
        if len(g) < 4 or rid not in pair:
            continue
        g = g.sort_values("horse_number")
        nums = g["horse_number"].astype(int).to_numpy()
        w, t2, t3 = g["p_win"].to_numpy(float), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float)
        r = (g["pw_b"] / g["p_win"].clip(lower=1e-6)).clip(0.2, 5.0).to_numpy()
        dists = {"model": fit_rank_distribution(w, t2, t3)[0],
                 "benter": fit_rank_distribution(g["pw_b"].to_numpy(float), np.clip(t2 * r, 1e-6, 0.99), np.clip(t3 * r, 1e-6, 0.99))[0]}
        for blend, dist in dists.items():
            cp = t36.combo_probs(dist)
            for bt in BETS:
                if bt not in cp or bt not in pair[rid]:
                    continue
                idx, pr = cp[bt]
                lab = [tuple(sorted((int(nums[i]), int(nums[j])))) for i, j in idx]
                od = np.array([pair[rid][bt].get(c, np.nan) for c in lab])
                ev = np.where(np.isfinite(od), pr * od, -np.inf)
                top = np.argsort(-ev, kind="stable")[: max(KMAX)]
                hits = pay.get((rid, bt), {})
                for k, i in enumerate(top, 1):
                    if not np.isfinite(ev[i]):
                        break
                    rows.append({"race_id": rid, "date": g["date"].iloc[0], "year": year, "bet": bt, "blend": blend, "rank": k, "ev": float(ev[i]),
                                 "odds": float(od[i]), "p": float(pr[i]), "ret": float(hits.get(lab[i], 0.0)), "known": (rid, bt) in pay})
    return pd.DataFrame(rows)


def cand_list() -> list[tuple[str, str, float, int, float]]:
    return list(itertools.product(BETS, BLENDS, THETAS, KMAX, PMIN))


def race_profit(t: pd.DataFrame, c: tuple[str, str, float, int, float]) -> pd.DataFrame:
    bet, blend, theta, kmax, pmin = c
    s = t[(t["bet"] == bet) & (t["blend"] == blend) & (t["ev"] >= theta) & (t["rank"] <= kmax) & (t["p"] >= pmin) & t["known"]]
    x = pd.DataFrame({"race_id": s["race_id"].to_numpy(), "pl": s["ret"].to_numpy() - 1.0, "n": 1})
    return x.groupby("race_id", sort=False).sum()


def ex_top3(t: pd.DataFrame, c) -> float | None:
    """上位 3 的中を除いた回収率（超高配当 1〜3 本に左右されていないか）。"""
    bet, blend, theta, kmax, pmin = c
    s = t[(t["bet"] == bet) & (t["blend"] == blend) & (t["ev"] >= theta) & (t["rank"] <= kmax) & (t["p"] >= pmin) & t["known"]]["ret"].to_numpy()
    if len(s) <= 3:
        return None
    return round(float(np.sort(s)[::-1][3:].sum() / len(s)), 4)


def run(frame: Path, first_test: int, last_test: int, out: Path) -> dict:
    d = pd.read_parquet(frame)
    d = d[(d["win_odds"] > 0)].copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d["race_id"] = d["race_id"].astype(str)
    q = 1.0 / d["win_odds"]
    d["q_win"] = q / q.groupby(d["race_id"]).transform("sum")
    d["win"] = (d["ret_win"] > 0).astype(int)
    years = sorted(set(d["year"]) & set(range(2020, last_test + 1)))
    pair, pay = load_pair_odds(set(years)), load_pay(set(years))
    tickets = []
    for y in years:
        prior = d[d["year"] < y]
        ab = P.fit_ab(prior) if len(prior) else DEFAULT_AB
        print(f"{y}: a, b = {ab}  チケット作成", flush=True)
        tickets.append(tickets_for_year(d, y, ab, pair, pay))
    T = pd.concat(tickets, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    T.to_parquet(out / f"tickets_{first_test}_{last_test}.parquet", index=False)
    rng = np.random.default_rng(20261109)
    cands = cand_list()
    rc_series: dict[int, list[pd.DataFrame]] = {}
    picks, oos_parts, grid = [], [], []
    for Y in range(first_test, last_test + 1):
        tr, te = T[T["year"] < Y], T[T["year"] == Y]
        best, best_lb, best_row = None, 1.0, None
        for k, c in enumerate(cands):
            roi, lb, n = P.roi_lower(race_profit(tr, c))
            tp = race_profit(te, c)
            grid.append({"test_year": Y, "cand": str(c), "train_roi": None if roi != roi else round(roi, 4), "train_lb99": round(lb, 4), "train_n": n,
                         "test_roi": round(float(1 + tp["pl"].sum() / max(tp["n"].sum(), 1)), 4), "test_n": int(tp["n"].sum()), "test_roi_ex3": ex_top3(te, c)})
            rc_series.setdefault(k, []).append(tp)
            if roi == roi and lb > best_lb:
                best, best_lb, best_row = c, lb, (roi, lb, n)
        if best is None:
            picks.append({"test_year": Y, "pick": None, "note": "前の年の下側 99% 限界が 100% を超える候補が無い → 買わない"})
            continue
        tp = race_profit(te, best)
        oos_parts.append(tp)
        picks.append({"test_year": Y, "pick": str(best), "train_roi": round(best_row[0], 4), "train_lb99": round(best_row[1], 4), "train_n": best_row[2],
                      "test_roi": round(float(1 + tp["pl"].sum() / max(tp["n"].sum(), 1)), 4), "test_n": int(tp["n"].sum())})
    result: dict = {"frame": str(frame), "K": len(cands), "picks": picks}
    if oos_parts:
        allp = pd.concat(oos_parts)
        roi, lo, hi = P.boot_ci(allp, rng)
        result["oos"] = {"roi": round(roi, 4), "ci95": [round(lo, 4), round(hi, 4)], "bets": int(allp["n"].sum())}
    else:
        result["oos"] = {"roi": None, "note": "どの年も買わなかった"}
    day_of = d.drop_duplicates("race_id").set_index("race_id")["date"].astype(str)
    result["reality_check"] = P.reality_check(rc_series, day_of, rng)
    result["caveat"] = "確定オッズ（馬連 = 確定、ワイド = 下限）で期待値を計算しているため、購入時点では分からない値を使っている。組合せ券の購入時点のオッズの実測はまだ無い。"
    (out / f"exotic_{first_test}_{last_test}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(grid).to_csv(out / f"exotic_grid_{first_test}_{last_test}.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(OUT / "frame_v6_expanding.parquet"))
    ap.add_argument("--first-test", type=int, default=2023)
    ap.add_argument("--last-test", type=int, default=2024)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    print(json.dumps(run(Path(a.frame), a.first_test, a.last_test, Path(a.out)), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
