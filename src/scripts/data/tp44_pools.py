"""tp-44 (B) プール間の不整合: 単勝オッズだけから作る順位分布で複勝・馬連・ワイド・馬単の確率を出し、確定オッズの EV と比べて買う。

python -m src.scripts.data.tp44_pools --out data/local/modeling/betting/tp44/pools.json
- 市場だけで作る確率（モデル不使用）。q_i = (1/単勝オッズ)/Σ。1 着 ∝ q^γ、2 着|1 着 ∝ q^λ2、3 着|1・2 着 ∝ q^λ3（Lo–Bacon-Shone 型）。
  γ・λ は『検証年より前の年』だけで当てはめる（walk-forward、2022〜2025 年を検証）。
- 候補（Dr. Z の複勝、Hausch–Ziemba のプール間の差）: EV 推定 = 確率 × 確定オッズ（複勝・ワイドは下限と中央の 2 通り）。しきい値以上を全点均等買い。
- 精算は確定払戻（曖昧なレースは除外）。回収率の区間はレース単位ブートストラップ、多重比較は White の reality check（tp43_marks.reality_check）。
注意: 確定オッズは締切後の値で、購入時点の値ではない（tp-36 と同じ限界）。2026 年は組合せ券の確定オッズが無いので評価できない。
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TEST_YEARS = (2022, 2023, 2024, 2025)
THRESHOLDS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
PMIN = (0.0, 0.02)
MIN_BETS = 150


# ---------------------------------------------------------------- 順位分布
def race_probs(q: np.ndarray, gamma: float, lam2: float, lam3: float) -> dict[str, np.ndarray]:
    """q: 市場の勝率（合計 1）。戻り値は馬の添字（q の並び）ごとの確率。
    top3: (n,)、umaren・wide・umatan: (n, n)（umaren・wide は対称、対角 0）。"""
    n = len(q)
    r1 = q ** gamma
    p1 = r1 / r1.sum()
    r2 = q ** lam2
    r3 = q ** lam3
    s2, s3 = r2.sum(), r3.sum()
    p12 = p1[:, None] * r2[None, :] / np.maximum(s2 - r2[:, None], 1e-12)       # [i, j] = P(1着 i, 2着 j)
    np.fill_diagonal(p12, 0.0)
    den3 = np.maximum(s3 - r3[:, None] - r3[None, :], 1e-12)                       # [i, j] 以外の合計
    p123 = p12[:, :, None] * r3[None, None, :] / den3[:, :, None]                  # [i, j, k]
    idx = np.arange(n)
    p123[idx, idx, :] = 0.0
    p123[idx, :, idx] = 0.0
    p123[:, idx, idx] = 0.0
    top3 = p123.sum((1, 2)) + p123.sum((0, 2)) + p123.sum((0, 1))
    umaren = p12 + p12.T
    pair3 = p123.sum(2) + p123.sum(1) + p123.sum(0)                                # 「i と j が上位 3 着以内」 (i, j の 1・2 着、1・3 着、2・3 着)
    wide = pair3 + pair3.T
    return {"top3": top3, "umaren": umaren, "wide": wide, "umatan": p12}


def _nll(params: np.ndarray, races: list[tuple[np.ndarray, int, int, int]]) -> float:
    gamma, lam2, lam3 = (float(np.clip(p, 0.2, 3.0)) for p in params)
    ll = 0.0
    for q, a, b, c in races:
        r1, r2, r3 = q ** gamma, q ** lam2, q ** lam3
        p = r1[a] / r1.sum() * r2[b] / max(r2.sum() - r2[a], 1e-12) * r3[c] / max(r3.sum() - r3[a] - r3[b], 1e-12)
        ll += np.log(max(p, 1e-300))
    return -ll / len(races)


def fit_params(races: list[tuple[np.ndarray, int, int, int]]) -> tuple[float, float, float]:
    from scipy.optimize import minimize

    res = minimize(_nll, x0=np.array([1.0, 1.0, 1.0]), args=(races,), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 200})
    g, l2, l3 = (float(np.clip(p, 0.2, 3.0)) for p in res.x)
    return g, l2, l3


# ---------------------------------------------------------------- データ
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tp36_census.load と同じ絞り込み（曖昧・解析不能の払戻のレースを除く）。組合せ券は odds_min / odds_max（ワイド）も読む。"""
    w = pd.read_parquet(BET / "odds_win_place.parquet")
    pair = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"])
    pay = pd.read_parquet(BET / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    good = set(pay["race_id"]) - bad
    w, pair, pay = (d[d["race_id"].isin(good)].copy() for d in (w, pair, pay))
    for d in (w, pair, pay):
        d["year"] = d["race_id"].str[:4].astype(int)
    return w, pair, pay


def race_orders(pay: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    s = pay[pay["bet_type"] == "sanrentan"][["race_id", "combo"]].drop_duplicates("race_id")
    out = {}
    for rid, c in zip(s["race_id"], s["combo"]):
        try:
            a, b, k = (int(x) for x in str(c).split(">"))
        except ValueError:
            continue
        out[rid] = (a, b, k)
    return out


def fit_frame(w: pd.DataFrame, orders: dict[str, tuple[int, int, int]], years: list[int]) -> list[tuple[np.ndarray, int, int, int]]:
    races = []
    for rid, g in w[w["year"].isin(years)].groupby("race_id", sort=False):
        if rid not in orders:
            continue
        g = g.sort_values("horse_number")
        o = g["win_odds"].to_numpy(float)
        if len(g) < 4 or not np.all(o >= 1.0):
            continue
        s = float(np.sum(1.0 / o))
        if not 1.0 <= s <= 1.6:
            continue
        nums = g["horse_number"].astype(int).tolist()
        a, b, c = orders[rid]
        if a not in nums or b not in nums or c not in nums:
            continue
        q = (1.0 / o) / s
        races.append((q, nums.index(a), nums.index(b), nums.index(c)))
    return races


def build_year(w: pd.DataFrame, pair: pd.DataFrame, pay: pd.DataFrame, year: int, params: tuple[float, float, float]) -> pd.DataFrame:
    """検証年の全候補（券種×組合せ）を 1 行 1 点で返す。列: race_id, kind, key, p, odds_lo, odds_mid, ret。"""
    wy = w[w["year"] == year]
    py = pair[pair["year"] == year]
    pay_y = pay[pay["year"] == year]
    gamma, lam2, lam3 = params
    rows: list[pd.DataFrame] = []
    meta = []
    for rid, g in wy.groupby("race_id", sort=False):
        g = g.sort_values("horse_number")
        o = g["win_odds"].to_numpy(float)
        if len(g) < 4 or not np.all(o >= 1.0):
            continue
        s = float(np.sum(1.0 / o))
        if not 1.0 <= s <= 1.6:
            continue                                                 # 壊れた単勝オッズのレースは除く
        nums = g["horse_number"].astype(int).to_numpy()
        pr = race_probs((1.0 / o) / s, gamma, lam2, lam3)
        n = len(nums)
        iu, ju = np.triu_indices(n, 1)
        a, b = nums[iu], nums[ju]
        ia, ja = np.where(~np.eye(n, dtype=bool))
        meta.append(pd.DataFrame({"race_id": rid, "kind": "fukusho", "key": nums.astype(str), "p": pr["top3"],
                                  "odds_lo": g["place_odds_min"].to_numpy(float),
                                  "odds_mid": ((g["place_odds_min"] + g["place_odds_max"]) / 2).to_numpy(float)}))
        meta.append(pd.DataFrame({"race_id": rid, "kind": "umaren", "key": [f"{x}-{y}" for x, y in zip(a, b)], "p": pr["umaren"][iu, ju]}))
        meta.append(pd.DataFrame({"race_id": rid, "kind": "wide", "key": [f"{x}-{y}" for x, y in zip(a, b)], "p": pr["wide"][iu, ju]}))
        meta.append(pd.DataFrame({"race_id": rid, "kind": "umatan", "key": [f"{nums[i]}>{nums[j]}" for i, j in zip(ia, ja)], "p": pr["umatan"][ia, ja]}))
    df = pd.concat(meta, ignore_index=True)
    # オッズを付ける（組合せ券は odds_pair、複勝は place_odds）
    pp = py.rename(columns={"bet_type": "kind", "combo": "key"})
    pp["odds_lo"] = pp["odds"].where(pp["kind"] != "wide", pp["odds_min"])
    pp["odds_mid"] = np.where(pp["kind"] == "wide", (pp["odds_min"] + pp["odds_max"]) / 2, pp["odds"])
    df = df.merge(pp[["race_id", "kind", "key", "odds_lo", "odds_mid"]], on=["race_id", "kind", "key"], how="left", suffixes=("", "_pair"))
    for c in ("odds_lo", "odds_mid"):
        df[c] = df[c].where(df["kind"] == "fukusho", df[c + "_pair"]) if f"{c}_pair" in df.columns else df[c]
    df = df.drop(columns=[c for c in df.columns if c.endswith("_pair")])
    # 払戻を付ける
    pay_hit = pay_y[pay_y["bet_type"].isin(["fukusho", "umaren", "wide", "umatan"])][["race_id", "bet_type", "combo", "payout"]].rename(columns={"bet_type": "kind", "combo": "key"})
    pay_hit = pay_hit.drop_duplicates(["race_id", "kind", "key"])
    df = df.merge(pay_hit, on=["race_id", "kind", "key"], how="left")
    df["ret"] = df["payout"].fillna(0.0)
    df["year"] = year
    return df.drop(columns=["payout"])


# ---------------------------------------------------------------- 戦略
def strategies() -> list[tuple]:
    """(券種, オッズの取り方, しきい値, 確率の下限)。"""
    out = []
    for kind, which in (("fukusho", "odds_lo"), ("fukusho", "odds_mid"), ("umaren", "odds_lo"), ("wide", "odds_lo"), ("wide", "odds_mid"), ("umatan", "odds_lo")):
        for t, pm in product(THRESHOLDS, PMIN):
            out.append((kind, which, t, pm))
    return out


def per_race_profit(df: pd.DataFrame, strat: tuple, race_index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    kind, which, t, pm = strat
    d = df[(df["kind"] == kind) & df[which].notna()]
    ev = d["p"] * d[which]
    d = d[(ev >= t) & (d["p"] >= pm)]
    if d.empty:
        z = np.zeros(len(race_index))
        return z, z
    g = d.groupby("race_id").agg(n=("ret", "size"), ret=("ret", "sum"))
    g = g.reindex(race_index).fillna(0.0)
    return 100.0 * g["n"].to_numpy(), g["ret"].to_numpy()


def boot_roi(stake: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    n = len(stake)
    if stake.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ss, rr = stake[idx].sum(1), ret[idx].sum(1)
    ok = ss > 0
    b = rr[ok] / ss[ok]
    return float(ret.sum() / stake.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def ev_bins(df: pd.DataFrame, kind: str, which: str) -> list[dict]:
    d = df[(df["kind"] == kind) & df[which].notna()].copy()
    d["ev"] = d["p"] * d[which]
    edges = [0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 5.0, 1e9]
    d["bin"] = pd.cut(d["ev"], edges, right=False)
    out = []
    for b, g in d.groupby("bin", observed=True):
        out.append({"ev_bin": str(b), "n": int(len(g)), "roi": float(g["ret"].sum() / (100.0 * len(g))), "mean_ev": float(g["ev"].mean())})
    return out


def main() -> None:
    from src.scripts.data.tp43_marks import reality_check

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BET / "tp44" / "pools.json")
    ap.add_argument("--years", type=int, nargs="*", default=list(TEST_YEARS))
    args = ap.parse_args()
    w, pair, pay = load()
    orders = race_orders(pay)
    frames, params_by_year = [], {}
    for y in args.years:
        train = [yy for yy in range(2020, y)]
        params = fit_params(fit_frame(w, orders, train))
        params_by_year[y] = params
        print(f"[{y}] 学習 {train[0]}-{train[-1]} γ={params[0]:.3f} λ2={params[1]:.3f} λ3={params[2]:.3f}", flush=True)
        frames.append(build_year(w, pair, pay, y, params))
    df = pd.concat(frames, ignore_index=True)
    races = pd.Index(sorted(df["race_id"].unique()))
    rid_year = pd.Series({r: int(r[:4]) for r in races})
    S = strategies()
    X = np.zeros((len(races), len(S)))
    stakes = np.zeros_like(X)
    rets = np.zeros_like(X)
    for k, s in enumerate(S):
        st, rt = per_race_profit(df, s, races)
        stakes[:, k], rets[:, k] = st, rt
        X[:, k] = rt - st
    rc = reality_check(X, min_bets=MIN_BETS)
    rows = []
    for k, s in enumerate(S):
        roi, lo, hi = boot_roi(stakes[:, k], rets[:, k])
        by_year = {}
        for y in args.years:
            m = (rid_year.reindex(races).to_numpy() == y)
            by_year[str(y)] = float(rets[m, k].sum() / stakes[m, k].sum()) if stakes[m, k].sum() > 0 else None
        rows.append({"kind": s[0], "odds": s[1], "threshold": s[2], "pmin": s[3], "n_bets": int(round(stakes[:, k].sum() / 100)),
                     "roi": roi, "lo": lo, "hi": hi, "by_year": by_year})
    bins = {f"{k}:{o}": ev_bins(df, k, o) for k, o in (("fukusho", "odds_lo"), ("umaren", "odds_lo"), ("wide", "odds_lo"), ("umatan", "odds_lo"))}
    out = {"params_by_year": {str(y): list(p) for y, p in params_by_year.items()}, "n_races": int(len(races)), "n_strategies": len(S),
           "reality_check": {k: v for k, v in rc.items() if k not in ("kept", "mean", "t")}, "strategies": rows, "ev_bins": bins}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    best = max((r for r in rows if r["n_bets"] >= MIN_BETS), key=lambda r: r["roi"] if r["roi"] == r["roi"] else -1)
    print("戦略数", len(S), "reality check p=", rc["p"], "最良", best["kind"], best["odds"], best["threshold"], best["pmin"], f"ROI={best['roi']:.3f} [{best['lo']:.3f},{best['hi']:.3f}] n={best['n_bets']}")


if __name__ == "__main__":
    main()
