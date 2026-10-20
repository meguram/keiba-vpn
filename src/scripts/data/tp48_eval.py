"""tp-48: 3連単の実オッズ（2024〜2026）で、tp-36 が保留した買い方を再評価する。

python -m src.scripts.data.tp48_eval collect  --out <dir>/raw.pkl [--max-races N]   # レースごとの記録（ホストで実行）
python -m src.scripts.data.tp48_eval summarize --raw <dir>/raw.pkl --out <dir>/eval.json

評価する買い方（すべて均等額、払戻は確定払戻。レース単位の母数）:
  S  3連単: 実オッズ × 確率（市場なし model / 混合 mixed）の EV 上位 k 点、EV のしきい値、縮約後 EV のしきい値、プール間の不整合（実オッズ ÷ 単勝由来の推定価格）
  T  3連複: 3連単の実オッズから作った『3連単プール由来の価格』（取引できる価格ではない。探索）
  M  印（◎○▲★△、新ルール）の組合せ: 3連単の流し・3連複・馬連・ワイド・単複
  E  的中率が高い券種（複勝・ワイド・馬連）の EV しきい値（実オッズは 2023〜2025 の odds_pair、複勝は 2026 も）
オッズは確定後の値。購入時点との差は対数正規の誤差で感度を見る。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution
from src.scripts.data import tp36_single as t36

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP48 = Path(os.environ.get("TP48_OUT", str(BET / "tp48")))
RATE = {"sanrentan": 0.725, "sanrenpuku": 0.75, "umaren": 0.775, "wide": 0.775, "fukusho": 0.80}
SHRINK_K = {"fukusho": 0.24, "umaren": 0.11, "wide": 0.10, "sanrentan": 0.10, "sanrenpuku": 0.10}      # tp-47 の縮約係数（3連系は未測定のため馬連・ワイド並みの 0.10 を仮置き）
KTOP = 300
YEARS = (2024, 2025, 2026)
THETA = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
KS = (1, 2, 3, 5, 10, 20, 50)


# ───────────────────────── 部品 ─────────────────────────
def place_adj(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """複勝の払戻の目安: 下限 + 0.3 ×（上限 − 下限）（tp-47 の修正）。"""
    return lo + 0.3 * (hi - lo)


def odds_cube(nums: np.ndarray, tri: pd.DataFrame) -> np.ndarray:
    """出走順位置（馬番昇順）の n×n×n の 3連単オッズ。無い組合せは NaN。"""
    n = len(nums)
    cube = np.full((n, n, n), np.nan, dtype=np.float64)
    if tri is None or len(tri) == 0:
        return cube
    pos = np.full(int(max(nums.max(), tri[["a", "b", "c"]].to_numpy().max())) + 1, -1, dtype=np.int64)
    pos[nums] = np.arange(n)
    ai, bi, ci = pos[tri["a"].to_numpy()], pos[tri["b"].to_numpy()], pos[tri["c"].to_numpy()]
    ok = (ai >= 0) & (bi >= 0) & (ci >= 0)
    cube[ai[ok], bi[ok], ci[ok]] = tri["odds"].to_numpy()[ok]
    return cube


def trio_implied(cube: np.ndarray, idx3: np.ndarray) -> np.ndarray:
    """3連単プールの逆数（購入確率の目安）から作る 3連複の市場確率（6 通りの和 ÷ 全体）。"""
    inv = np.where(np.isfinite(cube), 1.0 / np.maximum(cube, 1.0), 0.0)
    tot = inv.sum()
    if tot <= 0:
        return np.full(len(idx3), np.nan)
    a, b, c = idx3[:, 0], idx3[:, 1], idx3[:, 2]
    s = inv[a, b, c] + inv[a, c, b] + inv[b, a, c] + inv[b, c, a] + inv[c, a, b] + inv[c, b, a]
    return s / tot


def stake_ret(combos: list[tuple], hits: dict, ordered: bool) -> tuple[int, float]:
    seen, ret = set(), 0.0
    for c in combos:
        key = tuple(c) if ordered else tuple(sorted(c))
        if key in seen:
            continue
        seen.add(key)
        ret += hits.get(key, 0.0)
    return len(seen), ret


def mark_tickets(marks: dict[int, str]) -> dict[str, tuple[str, list[tuple], bool]]:
    """印から作る買い目。戻り値: 名前 → (券種, 組合せ, 順序あり)。印は honmei(◎) pair(○) show_val(▲) star(★) anchor(△)。"""
    inv = {t: [] for t in ("honmei", "pair", "show_val", "star", "anchor")}
    for hn, t in marks.items():
        if t in inv:
            inv[t].append(hn)
    h = inv["honmei"][0] if inv["honmei"] else None
    o = inv["pair"][0] if inv["pair"] else None
    others = [x for t in ("pair", "show_val", "star", "anchor") for x in inv[t]]
    out: dict[str, tuple[str, list[tuple], bool]] = {}
    if h is None:
        return out
    out["tansho_honmei"] = ("tansho", [(h,)], False)
    out["fukusho_honmei"] = ("fukusho", [(h,)], False)
    out["umaren_honmei_all"] = ("umaren", [(h, x) for x in others], False)
    out["wide_honmei_all"] = ("wide", [(h, x) for x in others], False)
    if o is not None:
        out["umaren_honmei_pair"] = ("umaren", [(h, o)], False)
        out["wide_honmei_pair"] = ("wide", [(h, o)], False)
        rest = [x for x in others if x != o]
        out["sanrenpuku_honmei_pair_flow"] = ("sanrenpuku", [(h, o, x) for x in rest], False)
        out["sanrentan_honmei_pair_12_flow3"] = ("sanrentan", [(a, b, x) for a, b in ((h, o), (o, h)) for x in rest], True)
    if len(others) >= 2:
        out["sanrenpuku_honmei_flow"] = ("sanrenpuku", [(h, x, y) for x, y in combinations(others, 2)], False)
        out["sanrentan_honmei_first_flow"] = ("sanrentan", [(h, x, y) for x, y in permutations(others, 2)], True)
    box = [h] + others[:3]
    if len(box) >= 3:
        out["sanrenpuku_box4"] = ("sanrenpuku", list(combinations(box, 3)), False)
    return out


# ───────────────────────── 収集 ─────────────────────────
def load_hits_all() -> tuple[dict, set]:
    pay = pd.read_parquet(BET / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    hits = t36.load_hits(pay[pay["bet_type"].isin(t36.BTS)])
    from src.scraper.payoff_repair import load_fix
    from src.scripts.data.tp36_final2026 import parse_payoff_json

    pj = pd.read_parquet(ROOT / "data/page_reference/tables/2026/race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
    fix = load_fix(2026)
    for r in pj.itertuples(index=False):
        h = parse_payoff_json(r.payoff, fix.get(str(r.race_id)))
        broken = set(h.pop("_broken", set()))
        if h.get("sanrentan") and max(h["sanrentan"].values()) >= t36.PAY_CAP["sanrentan"]:
            broken.add("sanrentan")
        h["_broken"] = broken
        hits[str(r.race_id)] = h
    return hits, bad


def load_pair_real2(race_ids: set, year: int) -> dict:
    pair = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"],
                           filters=[("race_id", ">=", f"{year}"), ("race_id", "<", f"{year + 1}")])
    pair = pair[pair["race_id"].isin(race_ids)]
    pair["o"] = np.where(pair["bet_type"] == "wide", place_adj(pair["odds_min"], pair["odds_max"]), pair["odds"])
    pair = pair[pair["o"] > 0]
    out: dict = {}
    for (rid, bt), g in pair.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {t36.parse_combo(c): float(o) for c, o in zip(g["combo"], g["o"])}
    return out


def marks_for_race(g: pd.DataFrame) -> dict[int, str]:
    from src.utils.race_probabilities import assign_mece_marks

    entries = []
    def f0(x) -> float:
        x = float(x) if x is not None else 0.0
        return x if np.isfinite(x) else 0.0

    for r in g.itertuples(index=False):
        lo, hi = f0(r.place_odds_min), f0(r.place_odds_max)
        pa = (lo + 0.3 * (hi - lo)) if (lo and hi) else 0.0
        wo = f0(r.win_odds)
        entries.append({"horse_number": int(r.horse_number), "win_prob": float(r.p_win), "top2_prob": float(r.p_top2), "top3_prob": float(r.p_top3),
                        "win_odds": wo, "ev_win": float(r.p_win) * wo if wo else 0.0, "ev_place": float(r.p_top3) * pa if pa else 0.0})
    assign_mece_marks(entries)
    return {e["horse_number"]: e["mark_type"] for e in entries}


def process_race(g: pd.DataFrame, hits: dict, tri: pd.DataFrame | None, pair_real: dict | None, alpha_m: tuple[float, float], use_tri: bool) -> dict:
    g = g.sort_values("horse_number")
    nums = g["horse_number"].astype(int).to_numpy()
    n = len(nums)
    broken = hits.get("_broken", set())
    rec: dict = {"n": n}
    sp = t36.schemes_probs(g)
    # M: 印の組合せ
    marks = marks_for_race(g)
    rec["marks"] = {}
    for name, (bt, combos, ordered) in mark_tickets(marks).items():
        if bt in broken or not hits.get(bt):
            continue
        rec["marks"][name] = stake_ret(combos, hits[bt], ordered)
    rec["n_marks"] = sum(1 for v in marks.values() if v != "none")
    # E: 複勝（単独）
    lo, hi = g["place_odds_min"].to_numpy(float), g["place_odds_max"].to_numpy(float)
    pa = place_adj(lo, hi)
    ok_pl = np.isfinite(pa) & (pa > 0)
    rec["E"] = {}
    for sch, (w, t2, t3) in sp.items():
        if "fukusho" not in broken:
            ev = np.where(ok_pl, t3 * pa, np.nan)
            sel = t36.top_k(ev, min(t36.KMAX, n))
            rec["E"][("fukusho", sch)] = (ev[sel], pa[sel], g["ret_place"].to_numpy(float)[sel], n)
    # 組合せ券
    q = g["q_win"].to_numpy()
    mkt = t36.combo_probs(RankDistribution(q, alpha_m[0], alpha_m[1])) if n >= 3 else {}
    rec["S"], rec["T"] = {}, {}
    if n < 4:
        return rec
    cube = odds_cube(nums, tri) if use_tri else None
    for sch, (w, t2, t3) in sp.items():
        dist, _ = fit_rank_distribution(w, t2, t3)
        cp = t36.combo_probs(dist)
        # E: 馬連・ワイド（実オッズ）
        for bt in ("umaren", "wide"):
            if pair_real is not None and bt in pair_real and bt not in broken:
                idx, p = cp[bt]
                lab = [tuple(int(nums[i]) for i in row) for row in idx]
                od = np.array([pair_real[bt].get(tuple(sorted(c)), np.nan) for c in lab])
                ev = p * od
                sel = t36.top_k(ev, min(t36.KMAX, len(ev)))
                hb = hits.get(bt, {})
                rec["E"][(bt, sch)] = (ev[sel], od[sel], np.array([hb.get(tuple(sorted(lab[i])), 0.0) for i in sel]), len(ev))
        # S: 3連単
        if use_tri and "sanrentan" not in broken and "sanrentan" in cp:
            idx, p = cp["sanrentan"]
            od = cube[idx[:, 0], idx[:, 1], idx[:, 2]]
            ev = p * od
            est = RATE["sanrentan"] / np.maximum(mkt["sanrentan"][1], 1e-12)
            sel = t36.top_k(ev, min(KTOP, int(np.isfinite(ev).sum())))
            lab = [tuple(int(nums[i]) for i in idx[j]) for j in sel]
            hb = hits.get("sanrentan", {})
            rec["S"][sch] = {"ev": ev[sel], "odds": od[sel], "p": p[sel], "ratio": od[sel] / est[sel], "pay": np.array([hb.get(c, 0.0) for c in lab]), "n_avail": int(np.isfinite(ev).sum())}
            # プール間の不整合: 実オッズ ÷ 推定価格（モデル確率に依存しない）
            ratio_all = od / est
            sel2 = t36.top_k(np.where(np.isfinite(ratio_all) & (p > 1e-4), ratio_all, np.nan), 50)
            lab2 = [tuple(int(nums[i]) for i in idx[j]) for j in sel2]
            rec["S"][sch]["incons"] = {"ratio": ratio_all[sel2], "ev": ev[sel2], "pay": np.array([hb.get(c, 0.0) for c in lab2])}
        # T: 3連複（3連単プール由来の価格）
        if use_tri and "sanrenpuku" not in broken and "sanrenpuku" in cp:
            idx3, p3 = cp["sanrenpuku"]
            pm = trio_implied(cube, idx3)
            od = RATE["sanrenpuku"] / np.maximum(pm, 1e-12)
            ev = p3 * od
            sel = t36.top_k(ev, min(t36.KMAX, len(ev)))
            lab = [tuple(sorted(int(nums[i]) for i in idx3[j])) for j in sel]
            hb = hits.get("sanrenpuku", {})
            rec["T"][sch] = (ev[sel], od[sel], np.array([hb.get(c, 0.0) for c in lab]), len(ev))
    return rec


def collect(a) -> None:
    frame = pd.read_parquet(BET / "tp36/frame_all.parquet")
    frame = frame[frame["year"] >= 2022].copy()
    inv = (1.0 / frame["win_odds"].where(frame["win_odds"] >= 1.0)).groupby(frame["race_id"]).sum()
    bad_odds = set(inv[~inv.between(1.0, 1.6)].index) | set(frame.loc[~(frame["win_odds"] >= 1.0), "race_id"])
    frame = frame[~frame["race_id"].isin(bad_odds)]
    hits, bad_pay = load_hits_all()
    frame = frame[~frame["race_id"].isin(bad_pay)]
    excl = json.loads((TP48 / "exclude.json").read_text(encoding="utf-8"))
    pair22 = t36.load_pair_real(set(frame.loc[frame["year"] == 2022, "race_id"]), 2022)
    alpha_m = t36.fit_market_alpha(frame[frame["year"] == 2022], pair22)
    del pair22
    print("market alpha", alpha_m, flush=True)
    recs, meta = [], []
    for year in (2023, 2024, 2025, 2026):
        dy = frame[frame["year"] == year]
        if a.max_races:
            dy = dy[dy["race_id"].isin(set(dy["race_id"].unique()[: a.max_races]))]
        pair_real = load_pair_real2(set(dy["race_id"]), year) if year <= 2025 else {}
        tri_all = pd.read_parquet(TP48 / f"trifecta_{year}.parquet") if year >= 2024 else None
        tri_by = {rid: x for rid, x in tri_all.groupby("race_id")} if tri_all is not None else {}
        ex = set(excl.get(str(year), {}).get("missing", [])) | set(excl.get(str(year), {}).get("inconsistent", [])) if year >= 2024 else set()
        for k, (rid, g) in enumerate(dy.groupby("race_id", sort=False)):
            use_tri = year >= 2024 and rid not in ex and rid in tri_by
            r = process_race(g, hits.get(rid, {}), tri_by.get(rid), pair_real.get(rid), alpha_m, use_tri)
            r["use_tri"] = bool(use_tri)
            recs.append(r)
            meta.append((rid, str(g["date"].iloc[0]), year))
            if k % 500 == 0:
                print(year, k, flush=True)
        del tri_all, tri_by, pair_real
    Path(a.out).write_bytes(pickle.dumps({"recs": recs, "meta": meta, "alpha_market": alpha_m}))
    print("saved", a.out, len(recs))


# ───────────────────────── 集計 ─────────────────────────
def boot(stake: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    return t36.boot_ci(stake, ret, n_boot, seed)


def reality_check(profit: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """White の Reality Check（スチューデント化した最大統計量のレース単位ブートストラップ）。profit: [レース数, 方策数]。"""
    R, K = profit.shape
    mu = profit.mean(0)
    sd = profit.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    t = np.sqrt(R) * mu / sd
    rng = np.random.default_rng(seed)
    tmax = np.empty(n_boot)
    for b in range(n_boot):
        pb = profit[rng.integers(0, R, R)]
        sdb = pb.std(0, ddof=1)
        sdb = np.where(sdb > 1e-12, sdb, np.inf)
        tmax[b] = (np.sqrt(R) * (pb.mean(0) - mu) / sdb).max()
    return {"t_max": float(t.max()), "best": int(t.argmax()), "p_value": float((tmax >= t.max()).mean()), "n_strategies": int(K)}


def fam_arrays(recs: list[dict], getter, K: int):
    ev = np.full((len(recs), K), np.nan)
    od = np.full((len(recs), K), np.nan)
    py = np.zeros((len(recs), K))
    for i, r in enumerate(recs):
        v = getter(r)
        if v is None:
            continue
        e, o, p = v[0], v[1], v[2]
        kk = min(K, len(e))
        ev[i, :kk], od[i, :kk], py[i, :kk] = e[:kk], o[:kk], p[:kk]
    return ev, od, py


def policy_rows(name: str, ev: np.ndarray, py: np.ndarray, years: np.ndarray, mask_ok: np.ndarray, bt: str, store: dict, profit_cols: list, profit_names: list) -> list[dict]:
    """k 掃引・しきい値・縮約後しきい値の方策を評価。"""
    rows = []
    R = len(ev)
    kshr = SHRINK_K.get(bt, 0.1)
    rate = RATE[bt]
    pols = []
    for k in KS:
        b = np.zeros_like(ev, dtype=bool)
        kk = min(k, ev.shape[1])
        b[:, :kk] = np.isfinite(ev[:, :kk])
        pols.append((f"top{k}", b & mask_ok[:, None]))
    for th in THETA:
        for k in (1, 5, 50):
            bought = np.zeros_like(ev, dtype=bool)
            bought[:, :k] = np.isfinite(ev[:, :k]) & (ev[:, :k] >= th)
            pols.append((f"ev>={th}|top{k}", bought & mask_ok[:, None]))
    for th in (1.0, 1.05, 1.2):
        raw_th = rate + (th - rate) / kshr
        bought = np.zeros_like(ev, dtype=bool)
        bought[:, :50] = np.isfinite(ev[:, :50]) & (ev[:, :50] >= raw_th)
        pols.append((f"shrunk{th}(raw>={raw_th:.2f})|top50", bought & mask_ok[:, None]))
    for pn, bought in pols:
        stake = bought.sum(1).astype(float) * 100.0
        ret = (py * bought).sum(1)
        profit_cols.append((ret - stake).astype(np.float32))
        profit_names.append(f"{name}|{pn}")
        row = {"policy": f"{name}|{pn}", "bets": int(bought.sum()), "races_bet": int((stake > 0).sum())}
        for lab, m in (("all", np.ones(R, bool)), ("y2024", years == 2024), ("y2025", years == 2025), ("y2026", years == 2026), ("y2023", years == 2023)):
            s, r_ = stake[m], ret[m]
            row[lab] = (float(r_.sum() / s.sum()) if s.sum() > 0 else None, int(s.sum() / 100))
        rows.append(row)
        store[f"{name}|{pn}"] = (stake, ret)
    return rows


def summarize(a) -> None:
    raw = pickle.loads(Path(a.raw).read_bytes())
    order = sorted(range(len(raw["meta"])), key=lambda i: (raw["meta"][i][1], raw["meta"][i][0]))
    recs = [raw["recs"][i] for i in order]
    meta = [raw["meta"][i] for i in order]
    years = np.array([m[2] for m in meta])
    R = len(recs)
    use_tri = np.array([r.get("use_tri", False) for r in recs])
    out: dict = {"races": R, "by_year": {str(y): int((years == y).sum()) for y in sorted(set(years.tolist()))}, "tri_races": {str(y): int(((years == y) & use_tri).sum()) for y in YEARS}}
    store: dict = {}
    pc: list = []
    pn: list = []
    tables: dict = {}
    # S: 3連単
    for sch in ("model", "mixed"):
        def get(r, sch=sch):
            v = r.get("S", {}).get(sch)
            return (v["ev"], v["odds"], v["pay"]) if v else None
        ev, od, py = fam_arrays(recs, get, KTOP)
        tables[f"S_sanrentan|{sch}"] = policy_rows(f"sanrentan|{sch}", ev[:, :50], py[:, :50], years, use_tri, "sanrentan", store, pc, pn)
        # プール間の不整合（上位 k = 実オッズ ÷ 推定価格が大きい順）
        iev, iod, ipy = np.full((R, 50), np.nan), np.full((R, 50), np.nan), np.zeros((R, 50))
        irat = np.full((R, 50), np.nan)
        for i, r in enumerate(recs):
            v = r.get("S", {}).get(sch)
            if v and "incons" in v:
                x = v["incons"]
                kk = len(x["ratio"])
                irat[i, :kk], iev[i, :kk], ipy[i, :kk] = x["ratio"], x["ev"], x["pay"]
        rows = []
        for rt in (1.0, 1.25, 1.5, 2.0):
            for evmin in (0.0, 1.0, 1.2):
                bought = np.isfinite(irat) & (irat >= rt) & (np.nan_to_num(iev, nan=-1) >= evmin) & use_tri[:, None]
                stake = bought.sum(1).astype(float) * 100
                ret = (ipy * bought).sum(1)
                pc.append((ret - stake).astype(np.float32))
                nm = f"incons|{sch}|ratio>={rt}|ev>={evmin}"
                pn.append(nm)
                store[nm] = (stake, ret)
                row = {"policy": nm, "bets": int(bought.sum()), "races_bet": int((stake > 0).sum())}
                for lab, m in (("all", np.ones(R, bool)), ("y2024", years == 2024), ("y2025", years == 2025), ("y2026", years == 2026)):
                    s, r_ = stake[m], ret[m]
                    row[lab] = (float(r_.sum() / s.sum()) if s.sum() > 0 else None, int(s.sum() / 100))
                rows.append(row)
        tables[f"S_incons|{sch}"] = rows
    # T: 3連複（3連単プール由来の価格）
    for sch in ("model", "mixed"):
        ev, od, py = fam_arrays(recs, lambda r, sch=sch: r.get("T", {}).get(sch), t36.KMAX)
        tables[f"T_sanrenpuku_implied|{sch}"] = policy_rows(f"sanrenpuku_implied|{sch}", ev, py, years, use_tri, "sanrenpuku", store, pc, pn)
    # E: 複勝・ワイド・馬連（実オッズ）
    for bt in ("fukusho", "wide", "umaren"):
        for sch in ("model", "mixed"):
            ev, od, py = fam_arrays(recs, lambda r, bt=bt, sch=sch: r.get("E", {}).get((bt, sch)), t36.KMAX)
            okm = np.isfinite(ev[:, :1]).any(1)
            tables[f"E_{bt}|{sch}"] = policy_rows(f"{bt}|{sch}", ev, py, years, okm, bt, store, pc, pn)
    # M: 印の組合せ
    names = sorted({k for r in recs for k in r.get("marks", {})})
    rows = []
    for nm in names:
        stake = np.array([r.get("marks", {}).get(nm, (0, 0.0))[0] * 100.0 for r in recs])
        ret = np.array([r.get("marks", {}).get(nm, (0, 0.0))[1] for r in recs])
        pc.append((ret - stake).astype(np.float32))
        pn.append(f"marks|{nm}")
        store[f"marks|{nm}"] = (stake, ret)
        row = {"policy": f"marks|{nm}", "bets": int(stake.sum() / 100), "races_bet": int((stake > 0).sum())}
        for lab, m in (("all", np.ones(R, bool)), ("y2023", years == 2023), ("y2024", years == 2024), ("y2025", years == 2025), ("y2026", years == 2026)):
            s, r_ = stake[m], ret[m]
            row[lab] = (float(r_.sum() / s.sum()) if s.sum() > 0 else None, int(s.sum() / 100))
        rows.append(row)
    tables["M_marks"] = rows
    out["marks_per_race"] = {"mean": float(np.mean([r.get("n_marks", 0) for r in recs])), "max": int(max(r.get("n_marks", 0) for r in recs))}
    # 区間（2024〜2025 のプール）と Reality Check
    design = (years == 2024) | (years == 2025)
    P = np.stack(pc, 1)
    ok = P[design].std(0) > 0
    out["n_policies"] = len(pn)
    out["n_policies_testable"] = int(ok.sum())
    rc = reality_check(P[design][:, ok], n_boot=1000)
    names_ok = [n for n, o in zip(pn, ok) if o]
    out["reality_check_2024_25"] = {"t_max": rc["t_max"], "p_value": rc["p_value"], "best": names_ok[rc["best"]], "n_strategies": rc["n_strategies"]}
    cand = []
    for nm in names_ok:
        stake, ret = store[nm]
        m = design
        if stake[m].sum() / 100 < 300:
            continue
        r0, lo, hi = boot(stake[m], ret[m], 400)
        cand.append((lo, r0, hi, nm, int(stake[m].sum() / 100)))
    cand.sort(reverse=True)
    out["top_by_lower_bound_2024_25"] = [{"policy": n, "roi": r0, "lo": lo, "hi": hi, "bets": b} for lo, r0, hi, n, b in cand[:15]]
    cand2 = sorted(cand, key=lambda x: -x[1])
    out["top_by_roi_2024_25"] = [{"policy": n, "roi": r0, "lo": lo, "hi": hi, "bets": b} for lo, r0, hi, n, b in cand2[:15]]
    # 2026 は最終評価（事前に決めた規則: 2024〜25 の下限上位 5 + ROI 上位 5 を 1 回だけ）
    fin = list(dict.fromkeys([n for _, _, _, n, _ in cand[:5]] + [n for _, _, _, n, _ in cand2[:5]]))
    m26 = years == 2026
    out["final_2026"] = []
    for nm in fin:
        stake, ret = store[nm]
        r0, lo, hi = boot(stake[m26], ret[m26], 1000)
        out["final_2026"].append({"policy": nm, "roi": r0, "lo": lo, "hi": hi, "bets": int(stake[m26].sum() / 100)})
    # 全方策の区間（2024〜25）を表に追記
    for tn, rows in tables.items():
        for row in rows:
            st = store.get(row["policy"])
            if st is None or st[0][design].sum() == 0:
                row["design"] = None
                continue
            r0, lo, hi = boot(st[0][design], st[1][design], 300)
            row["design"] = [r0, lo, hi, int(st[0][design].sum() / 100)]
    # 主な方策の詳細: 的中レース率、上位 3 レースの的中を除いた回収率、資金曲線（破産確率）
    keyp = ["umaren|model|top1", "wide|model|top1", "fukusho|model|shrunk1.05(raw>=1.84)|top50", "fukusho|model|ev>=2.0|top5", "sanrentan|model|top5",
            "sanrentan|model|top1", "sanrenpuku_implied|model|top1", "marks|sanrenpuku_honmei_pair_flow", "marks|umaren_honmei_pair", "marks|tansho_honmei"]
    out["detail"] = {}
    for nm in keyp:
        if nm not in store:
            continue
        stake, ret = store[nm]
        m = design & (stake > 0)
        if m.sum() == 0:
            continue
        s_, r_ = stake[m], ret[m]
        top3 = np.argsort(-r_)[:3]
        keep = np.ones(len(r_), bool)
        keep[top3] = False
        ps = t36.path_stats(stake[stake > 0], ret[stake > 0], years[stake > 0])
        out["detail"][nm] = {"races_bet": int(m.sum()), "hit_race_rate": float((r_ > 0).mean()), "roi": float(r_.sum() / s_.sum()),
                             "roi_without_top3_hits": float(r_[keep].sum() / s_[keep].sum()), "top3_hit_payouts": [float(x) for x in np.sort(r_)[::-1][:3]],
                             "ruin_prob_30u": ps.get("ruin_prob_30u"), "ruin_prob_100u": ps.get("ruin_prob_100u"), "longest_losing_streak": ps.get("longest_losing_streak")}
    out["tables"] = tables
    # 購入時点オッズの感度（3連単 model/mixed の top-k、EV 上位 k 点を対数正規の誤差つきで選び直す）
    sens: dict = {}
    rng = np.random.default_rng(0)
    for sch in ("model", "mixed"):
        for sigma in (0.0, 0.05, 0.1, 0.2, 0.3):
            for k in (1, 5):
                stake = np.zeros(R)
                ret = np.zeros(R)
                for i, r in enumerate(recs):
                    v = r.get("S", {}).get(sch)
                    if not v or not use_tri[i]:
                        continue
                    ev = v["ev"] * np.exp(rng.normal(0, sigma, len(v["ev"]))) if sigma > 0 else v["ev"]
                    sel = np.argsort(-ev)[:k]
                    stake[i] = len(sel) * 100.0
                    ret[i] = v["pay"][sel].sum()
                m = design
                r0, lo, hi = boot(stake[m], ret[m], 300)
                sens[f"{sch}|top{k}|sigma{sigma}"] = [r0, lo, hi]
    out["noise_sensitivity"] = sens
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("saved", a.out, "policies", len(pn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="")
    ap.add_argument("--max-races", type=int, default=0)
    a = ap.parse_args()
    {"collect": collect, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
