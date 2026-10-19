"""tp-36 追加: 各レース・各券種で EV 上位 k 点だけを買う（k=1 が『1 点買い』）。全 7 券種。

python -m src.scripts.data.tp36_single collect --out data/local/modeling/betting/tp36/single_raw.pkl
python -m src.scripts.data.tp36_single summarize --raw ...single_raw.pkl --out ...single.json

選択に使うオッズ: 'real' = 確定オッズ（単勝・複勝・馬連・ワイド・馬単のみ。複勝・ワイドは (最小+最大)/2）、
'est' = 推定オッズ = 還元率 / 市場確率（単勝の確定オッズから作る市場確率に、市場側の割引 Harville を当てはめる）。3連複・3連単は est のみ。
評価は『選んだ組合せが当たったときの実際の払戻』（払戻表の当たり組合せ）。確率: 'model'（市場なし: v4 の勝率・連対率・複勝率+割引 Harville）と
'mixed'（単勝の Benter 混合 p_blend を勝率とし、連対率・複勝率は p_blend/p_win で拡縮）。
"""

from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP = BET / "tp36"
PAYBACK = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}
BTS = list(PAYBACK)
KMAX = 50
KS = (1, 2, 3, 5, 10, 50)
REAL = {"tansho", "fukusho", "umaren", "wide", "umatan"}


@lru_cache(maxsize=None)
def idx_pairs(n: int) -> np.ndarray:
    return np.array(list(combinations(range(n), 2)), dtype=np.int16).reshape(-1, 2)


@lru_cache(maxsize=None)
def idx_ordered2(n: int) -> np.ndarray:
    return np.array(list(permutations(range(n), 2)), dtype=np.int16).reshape(-1, 2)


@lru_cache(maxsize=None)
def idx_triples(n: int) -> np.ndarray:
    return np.array(list(combinations(range(n), 3)), dtype=np.int16).reshape(-1, 3)


@lru_cache(maxsize=None)
def idx_ordered3(n: int) -> np.ndarray:
    return np.array(list(permutations(range(n), 3)), dtype=np.int16).reshape(-1, 3)


def combo_probs(dist: RankDistribution) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """券種ごとに (組合せの index 配列, 的中確率) を返す。index は出走順位置（馬番昇順）。"""
    n = dist.n
    p2, p3 = dist._p2, dist._p3
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ip = idx_pairs(n)
    out["umaren"] = (ip, (p2 + p2.T)[ip[:, 0], ip[:, 1]])
    if n >= 3:
        w = p3.sum(2) + p3.sum(1) + p3.sum(0)
        out["wide"] = (ip, (w + w.T)[ip[:, 0], ip[:, 1]])
    io = idx_ordered2(n)
    out["umatan"] = (io, p2[io[:, 0], io[:, 1]])
    if n >= 3:
        it = idx_triples(n)
        a, b, c = it[:, 0], it[:, 1], it[:, 2]
        out["sanrenpuku"] = (it, p3[a, b, c] + p3[a, c, b] + p3[b, a, c] + p3[b, c, a] + p3[c, a, b] + p3[c, b, a])
        io3 = idx_ordered3(n)
        out["sanrentan"] = (io3, p3[io3[:, 0], io3[:, 1], io3[:, 2]])
    return out


def parse_combo(s: str) -> tuple:
    return tuple(int(x) for x in str(s).replace(">", "-").replace("→", "-").replace(" ", "").split("-"))


def top_k(ev: np.ndarray, k: int) -> np.ndarray:
    ev = np.where(np.isfinite(ev), ev, -np.inf)
    if len(ev) > 4 * k:
        part = np.argpartition(-ev, k)[:k]
        return part[np.argsort(-ev[part], kind="stable")]
    return np.argsort(-ev, kind="stable")[:k]


def fit_market_alpha(frame: pd.DataFrame, pair: dict, n_races: int = 400) -> tuple[float, float]:
    """市場側の割引 Harville の α2, α3 を、実際の馬連・ワイドの市場確率（1/オッズ を正規化）に当てはめる（評価に使わない年のレースで）。"""
    rids = [r for r in frame["race_id"].unique() if r in pair][:n_races]
    data = []
    for rid in rids:
        g = frame[frame["race_id"] == rid].sort_values("horse_number")
        nums = g["horse_number"].astype(int).to_numpy()
        q = g["q_win"].to_numpy()
        um = pair[rid].get("umaren")
        wd = pair[rid].get("wide")
        if um is None or wd is None:
            continue
        ip = idx_pairs(len(nums))
        ou = np.array([um.get((int(nums[i]), int(nums[j])), np.nan) for i, j in ip])
        ow = np.array([wd.get((int(nums[i]), int(nums[j])), np.nan) for i, j in ip])
        if np.isnan(ou).any() or np.isnan(ow).any():
            continue
        pu, pw = 1 / ou, 1 / ow
        data.append((q, ip, pu / pu.sum(), 3 * pw / pw.sum()))

    def loss2(a2):
        t = 0.0
        for q, ip, pu, _ in data:
            d = RankDistribution(q, a2, 1.0)
            t += float(np.sum(((d._p2 + d._p2.T)[ip[:, 0], ip[:, 1]] - pu) ** 2))
        return t

    a2 = float(minimize_scalar(loss2, bounds=(0.3, 1.3), method="bounded", options={"xatol": 1e-3}).x)

    def loss3(a3):
        t = 0.0
        for q, ip, _, pw in data:
            cp = combo_probs(RankDistribution(q, a2, a3))
            if "wide" in cp:
                t += float(np.sum((cp["wide"][1] - pw) ** 2))
        return t

    a3 = float(minimize_scalar(loss3, bounds=(0.3, 1.3), method="bounded", options={"xatol": 1e-3}).x)
    return a2, a3


def schemes_probs(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    w, t2, t3 = g["p_win"].to_numpy(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy()
    out = {"model": (w, t2, t3)}
    if g["p_blend"].notna().all():
        r = (g["p_blend"] / g["p_win"].clip(lower=1e-6)).clip(0.2, 5.0).to_numpy()
        out["mixed"] = (g["p_blend"].to_numpy(), np.clip(t2 * r, 1e-6, 0.99), np.clip(t3 * r, 1e-6, 0.99))
    return out


def process_race(g: pd.DataFrame, hits: dict, pair_real: dict | None, alpha_m: tuple[float, float], horse_ret: bool = True) -> dict:
    """1 レース分の記録。戻り値: {(bt, scheme, source): (ev[K], odds[K], payout[K], n_available)} と all 買い {('all', bt): (n_combo, ret)}。"""
    g = g.sort_values("horse_number")
    nums = g["horse_number"].astype(int).to_numpy()
    n = len(nums)
    rec: dict = {}
    sp = schemes_probs(g)
    q = g["q_win"].to_numpy()
    mkt = combo_probs(RankDistribution(q, alpha_m[0], alpha_m[1])) if n >= 3 else {}
    # 単勝・複勝（確定オッズ）
    for sch, (w, t2, t3) in sp.items():
        for bt, p, odds, pay in (("tansho", w, g["win_odds"].to_numpy(), g["ret_win"].to_numpy()), ("fukusho", t3, g["place_mid"].to_numpy() if "place_mid" in g else None, g["ret_place"].to_numpy() if "ret_place" in g else None)):
            if odds is None or not np.isfinite(odds).any():
                continue
            ev = p * odds
            sel = top_k(ev, min(KMAX, n))
            rec[(bt, sch, "real")] = (ev[sel], odds[sel], pay[sel], n)
        # 市場確率からの推定オッズ（単勝は確定オッズ = 推定オッズ）
    # 組合せ券
    for sch, (w, t2, t3) in sp.items():
        if n < 3:
            continue
        dist, _ = fit_rank_distribution(w, t2, t3)
        cp = combo_probs(dist)
        for bt, (idx, p) in cp.items():
            lab = [tuple(int(nums[i]) for i in row) for row in idx]
            hb = hits.get(bt, {})
            pm = mkt[bt][1]
            est_odds = PAYBACK[bt] / np.maximum(pm, 1e-12)
            sources = ["est"] + (["real"] if (pair_real is not None and bt in pair_real) else [])
            for src in sources:
                if src == "est":
                    odds = est_odds
                else:
                    od = pair_real[bt]
                    odds = np.array([od.get(t, np.nan) for t in lab])
                ev = p * odds
                sel = top_k(ev, min(KMAX, len(ev)))
                pay = np.array([hb.get(lab[i], 0.0) for i in sel])
                rec[(bt, sch, src)] = (ev[sel], odds[sel], pay, len(ev))
    # 全買い（参考: 全組合せを 1 点ずつ）
    for bt in ("umaren", "wide", "umatan", "sanrenpuku", "sanrentan"):
        if n < 3:
            continue
        ncomb = {"umaren": n * (n - 1) // 2, "wide": n * (n - 1) // 2, "umatan": n * (n - 1), "sanrenpuku": n * (n - 1) * (n - 2) // 6, "sanrentan": n * (n - 1) * (n - 2)}[bt]
        rec[("all", bt)] = (ncomb, float(sum(hits.get(bt, {}).values())))
    return rec


PAY_CAP = {"sanrentan": 2e7}      # 3 連単の 2 千万円超は払戻の取り込み異常（桁連結など）。そのレースのその券種は評価から外す


def load_hits(pay: pd.DataFrame) -> dict:
    out: dict = {}
    for r in pay.itertuples(index=False):
        h = out.setdefault(r.race_id, {})
        if float(r.payout) >= PAY_CAP.get(r.bet_type, float("inf")):
            h.setdefault("_broken", set()).add(r.bet_type)
            continue
        h.setdefault(r.bet_type, {})[parse_combo(r.combo)] = float(r.payout)
    return out


def drop_broken(rec: dict, broken: set) -> dict:
    return {k: v for k, v in rec.items() if not ((k[0] == "all" and k[1] in broken) or (k[0] != "all" and k[0] in broken))}


def load_pair_real(race_ids: set[str], year: int) -> dict:
    pair = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"], filters=[("race_id", ">=", f"{year}"), ("race_id", "<", f"{year + 1}")])
    pair = pair[pair["race_id"].isin(race_ids)]
    pair["o"] = np.where(pair["bet_type"] == "wide", (pair["odds_min"] + pair["odds_max"]) / 2, pair["odds"])
    pair = pair[pair["o"] > 0]
    out: dict = {}
    for (rid, bt), g in pair.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {parse_combo(c): float(o) for c, o in zip(g["combo"], g["o"])}
    return out


def collect(a) -> None:
    d = pd.read_parquet(TP / "edge_frame.parquet")
    d = d[d["year"] >= 2022].copy()
    pay = pd.read_parquet(BET / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    d = d[~d["race_id"].isin(bad)]
    pay = pay[pay["bet_type"].isin(BTS)]
    hits = load_hits(pay)
    pair22 = load_pair_real(set(d.loc[d["year"] == 2022, "race_id"]), 2022)
    alpha_m = fit_market_alpha(d[d["year"] == 2022], pair22)
    del pair22
    print("market alpha", alpha_m, flush=True)
    recs, meta = [], []
    for year in (2023, 2024, 2025):
        dy = d[d["year"] == year]
        if a.max_races:
            dy = dy[dy["race_id"].isin(set(dy["race_id"].unique()[: a.max_races]))]
        pair_real = load_pair_real(set(dy["race_id"]), year)
        for k, (rid, g) in enumerate(dy.groupby("race_id", sort=False)):
            h = hits.get(rid, {})
            recs.append(drop_broken(process_race(g, {k: v for k, v in h.items() if k != "_broken"}, pair_real.get(rid), alpha_m), h.get("_broken", set())))
            meta.append((rid, str(g["date"].iloc[0]), year))
            if k % 1000 == 0:
                print(year, "race", k, flush=True)
        del pair_real
    Path(a.out).write_bytes(pickle.dumps({"recs": recs, "meta": meta, "alpha_market": alpha_m}))
    print("saved", a.out)


# ───────────────────────── 集計 ─────────────────────────
def boot_ci(stake: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    m = stake > 0
    s, r = stake[m], ret[m]
    if s.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    ss, rr = s[idx].sum(1), r[idx].sum(1)
    ok = ss > 0
    b = rr[ok] / ss[ok]
    return float(r.sum() / s.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def path_stats(stake: np.ndarray, ret: np.ndarray, years: np.ndarray, seed: int = 0) -> dict:
    """均等買い（毎回同額）の資金曲線の指標。単位 = 買ったレースの平均購入額。"""
    m = stake > 0
    s, r, y = stake[m], ret[m], years[m]
    if len(s) == 0:
        return {}
    u = float(s.mean())
    prof = (r - s) / u
    cum = np.cumsum(prof)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    max_dd = float((peak - cum).max())
    streak = best = 0
    for p in prof:
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
    rng = np.random.default_rng(seed)
    N = max(1, int(len(prof) / max(1, len(set(y.tolist())))))
    sims = prof[rng.integers(0, len(prof), size=(2000, N))].cumsum(1)
    mn = sims.min(1)
    return {"max_drawdown_units": max_dd, "longest_losing_streak": int(best), "ruin_prob_30u": float((mn <= -30).mean()), "ruin_prob_100u": float((mn <= -100).mean()),
            "year_profit_units": float(prof.sum() / max(1, len(set(y.tolist())))), "profit_units_p5_p95_per_year": [float(np.quantile(sims[:, -1], 0.05)), float(np.quantile(sims[:, -1], 0.95))]}


def kelly_path(ev: np.ndarray, odds: np.ndarray, pay: np.ndarray, years: np.ndarray, frac: float, cap: float, min_ev: float) -> dict:
    """k=1 の Kelly（資金に対する割合・逐次複利）。f = frac × max(0,(EV−1)/(odds−1))、上限 cap。年ごとに資金 1 から。"""
    out = {}
    for y in sorted(set(years.tolist())):
        m = years == y
        W, peak, mdd, nb, wmin = 1.0, 1.0, 0.0, 0, 1.0
        for e, o, p in zip(ev[m], odds[m], pay[m]):
            if not (np.isfinite(e) and np.isfinite(o)) or e < min_ev or o <= 1:
                continue
            f = min(cap, frac * max(0.0, (e - 1) / (o - 1)))
            if f <= 0:
                continue
            nb += 1
            W *= 1.0 + f * (p / 100.0 - 1.0)
            peak = max(peak, W)
            mdd = max(mdd, 1 - W / peak)
            wmin = min(wmin, W)
        out[str(y)] = {"final_wealth": float(W), "max_drawdown": float(mdd), "n_bets": nb, "min_wealth": float(wmin)}
    return out


def key_arrays(recs: list[dict], key: tuple, R: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = np.full((R, KMAX), np.nan)
    od = np.full((R, KMAX), np.nan)
    py = np.zeros((R, KMAX))
    for i, r in enumerate(recs):
        if key in r:
            e, o, p, _ = r[key]
            kk = min(KMAX, len(e))
            ev[i, :kk], od[i, :kk], py[i, :kk] = e[:kk], o[:kk], p[:kk]
    return ev, od, py


def summarize(a) -> None:
    raw = pickle.loads(Path(a.raw).read_bytes())
    recs, meta = raw["recs"], raw["meta"]
    order = sorted(range(len(meta)), key=lambda i: (meta[i][1], meta[i][0]))
    recs = [recs[i] for i in order]
    meta = [meta[i] for i in order]
    years = np.array([m[2] for m in meta])
    R = len(recs)
    keys = sorted({k for r in recs for k in r if k[0] != "all"})
    out: dict = {"races": R, "alpha_market": raw["alpha_market"], "tables": {}, "n_policies_evaluated": 0}
    tries = 0
    for key in keys:
        bt, sch, src = key
        ev, od, py = key_arrays(recs, key, R)
        has = np.isfinite(ev[:, 0])
        res: dict = {"sweep": [], "threshold_k1": [], "threshold_k10": [], "kelly_k1": {}}
        for k in KS:
            bought = np.isfinite(ev[:, :k]) & has[:, None]
            stake = 100.0 * bought.sum(1)
            ret = (py[:, :k] * bought).sum(1)
            roi, lo, hi = boot_ci(stake, ret)
            m = stake > 0
            mean_ev = float((np.where(bought, ev[:, :k], 0.0) * 100.0).sum() / max(stake.sum(), 1e-9)) if m.any() else float("nan")
            row = {"k": k, "roi": roi, "lo": lo, "hi": hi, "race_hit": float((ret[m] > 0).mean()) if m.any() else None,
                   "stake_per_race": float(stake.mean()), "bet_races": int(m.sum()), "mean_ev_bought": mean_ev,
                   "by_year": {str(y): float(ret[(years == y) & m].sum() / max(stake[(years == y) & m].sum(), 1e-9)) for y in sorted(set(years.tolist()))}}
            row.update(path_stats(stake, ret, years))
            res["sweep"].append(row)
            tries += 1
        for tag, kk in (("threshold_k1", 1), ("threshold_k10", 10)):
            for x in (0.0, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0):
                bought = np.isfinite(ev[:, :kk]) & (ev[:, :kk] >= x)
                stake = 100.0 * bought.sum(1)
                ret = (py[:, :kk] * bought).sum(1)
                if stake.sum() <= 0:
                    continue
                roi, lo, hi = boot_ci(stake, ret)
                m = stake > 0
                row = {"min_ev": x, "roi": roi, "lo": lo, "hi": hi, "race_hit": float((ret[m] > 0).mean()), "bet_races": int(m.sum()), "stake_per_race": float(stake.mean()),
                       "by_year": {str(y): float(ret[(years == y) & m].sum() / max(stake[(years == y) & m].sum(), 1e-9)) for y in sorted(set(years.tolist())) if ((years == y) & m).any()}}
                row.update({k: v for k, v in path_stats(stake, ret, years).items() if k in ("max_drawdown_units", "longest_losing_streak", "ruin_prob_30u", "ruin_prob_100u")})
                res[tag].append(row)
                tries += 1
        if bt in ("tansho", "fukusho", "umaren", "wide", "umatan") or src == "est":
            for fr, cap, mev in ((0.25, 0.05, 1.0), (0.25, 0.05, 1.2)):
                res["kelly_k1"][f"f{fr}_cap{cap}_ev{mev}"] = kelly_path(ev[:, 0], od[:, 0], py[:, 0], years, fr, cap, mev)
                tries += 1
        out["tables"]["|".join(key)] = res
    # 全買い
    allb = {}
    for bt in ("umaren", "wide", "umatan", "sanrenpuku", "sanrentan"):
        st = np.array([100.0 * r[("all", bt)][0] if ("all", bt) in r else 0.0 for r in recs])
        rt = np.array([r[("all", bt)][1] if ("all", bt) in r else 0.0 for r in recs])
        roi, lo, hi = boot_ci(st, rt, n_boot=300)
        allb[bt] = {"roi": roi, "lo": lo, "hi": hi, "stake_per_race": float(st.mean())}
    out["buy_all"] = allb
    out["n_policies_evaluated"] = tries
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", a.out, "policies", tries)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default=str(TP / "single_raw.pkl"))
    c.add_argument("--max-races", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--raw", default=str(TP / "single_raw.pkl"))
    s.add_argument("--out", default=str(TP / "single.json"))
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else summarize(a)


if __name__ == "__main__":
    main()
