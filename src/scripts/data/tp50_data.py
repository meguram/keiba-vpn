"""tp-50 データ: レース × 候補買い目（チケット）の表を作る。方策（ルール・教師あり・強化学習）の共通の入力。

python -m src.scripts.data.tp50_data build --year 2024 [--max-races N]     # ホストで実行。出力 <TP50_OUT>/tickets_<year>.parquet
python -m src.scripts.data.tp50_data check                                 # 年別の件数・還元率の簡易確認

1 行 = 1 レースの 1 買い目。券種は 0 単勝 / 1 複勝 / 2 馬連 / 3 ワイド / 4 馬単 / 5 3連単（3連単は 2024 年以降のみ・各レース上位のみ）。
全馬の単複、組合せ券は『EV（モデル・混合）の上位 K 点 ∪ 確率の上位』に絞る。ret は 100 円あたりの払戻倍率（的中しなければ 0）で、確定払戻から作る。
odds は確定オッズ（複勝・ワイドは 下限 + 0.3 ×（上限 − 下限）の目安）。購入時点のオッズとは違う。
p_model: V4 の勝率・連対率・複勝率 + 割引 Harville（市場なし）。p_mix: 単勝の Benter 混合（p_blend）で拡縮した確率。q: 市場確率（単勝の確定オッズから、市場側の割引 Harville）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution
from src.scripts.data import tp36_single as t36

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
B26 = ROOT / "data/local/modeling/betting_2026"
TP48 = BET / "tp48"
OUT = Path(os.environ.get("TP50_OUT", str(BET / "tp50")))
BT_CODE = {"tansho": 0, "fukusho": 1, "umaren": 2, "wide": 3, "umatan": 4, "sanrentan": 5}
BT_NAME = {v: k for k, v in BT_CODE.items()}
RATE = {"tansho": 0.8, "fukusho": 0.8, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrentan": 0.725}
K_EV = 40            # 組合せ券: EV 上位
K_P = 20             # 組合せ券: 確率上位
K_TRI_EV, K_TRI_P = 60, 30


def place_adj(lo, hi):
    return lo + 0.3 * (hi - lo)


def load_pair_all(race_ids: set, path: Path, year: int | None) -> dict:
    """馬連・ワイド・馬単の確定オッズ。戻り値: race_id → bet_type → {組合せ: オッズ}（ワイドは下限 + 0.3 × 幅）。"""
    pair = pd.read_parquet(path, columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"])
    if year is not None:
        pair = pair[(pair["race_id"] >= f"{year}") & (pair["race_id"] < f"{year + 1}")]
    pair = pair[pair["race_id"].isin(race_ids) & pair["bet_type"].isin(["umaren", "wide", "umatan"])].copy()
    pair["o"] = np.where(pair["bet_type"] == "wide", place_adj(pair["odds_min"], pair["odds_max"]), pair["odds"])
    pair = pair[pair["o"] > 0]
    out: dict = {}
    for (rid, bt), g in pair.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {t36.parse_combo(c): float(o) for c, o in zip(g["combo"], g["o"])}
    return out


def entropy(p: np.ndarray) -> float:
    p = np.clip(p / p.sum(), 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def race_context(w: np.ndarray, q: np.ndarray, ev_win: np.ndarray) -> dict:
    s = np.sort(w)[::-1]
    n = len(w)
    dominant = bool(s[0] >= 2.0 / n and s[0] >= 1.6 * s[1])
    even = bool(s[0] < 1.15 * s[1])
    upset = bool(np.argmax(ev_win) != np.argmax(w) and np.nanmax(ev_win) >= 1.15)
    kl = float(np.sum(w * (np.log(np.clip(w, 1e-9, 1)) - np.log(np.clip(q, 1e-9, 1)))))
    return {"r_p1": float(s[0]), "r_ratio12": float(s[0] / max(s[1], 1e-9)), "r_ent": entropy(w), "r_q1": float(np.max(q)), "r_kl": kl,
            "r_dom": int(dominant), "r_even": int(even), "r_upset": int(upset)}


def top_idx(score: np.ndarray, k: int) -> np.ndarray:
    return t36.top_k(np.where(np.isfinite(score), score, -np.inf), min(k, len(score)))


def build_race(g: pd.DataFrame, hits: dict, pair_real: dict | None, tri: pd.DataFrame | None, alpha_m: tuple[float, float]) -> list[dict] | None:
    from src.scripts.data.tp48_eval import odds_cube

    g = g.sort_values("horse_number")
    nums = g["horse_number"].astype(int).to_numpy()
    n = len(nums)
    if n < 4:
        return None
    broken = hits.get("_broken", set())
    sp = t36.schemes_probs(g)
    wm, t2m, t3m = sp["model"]
    wx, t2x, t3x = sp.get("mixed", sp["model"])
    q = g["q_win"].to_numpy()
    pop = g["win_odds"].rank(method="min").to_numpy()
    lo, hi = g["place_odds_min"].to_numpy(float), g["place_odds_max"].to_numpy(float)
    pa = place_adj(lo, hi)
    wo = g["win_odds"].to_numpy(float)
    ctx = race_context(wm, q, wm * wo)
    base = {"race_id": str(g["race_id"].iloc[0]), "n": n, **ctx}
    rows: list[dict] = []

    def add(bt, idx_rows, odds, p_model, p_mix, qq, ret):
        m = len(odds)
        if m == 0:
            return
        comp = np.asarray(idx_rows)
        if comp.ndim == 1:
            comp = comp[:, None]
        pops = pop[comp]
        d = {**base, "bt": BT_CODE[bt], "odds": odds, "p_model": p_model, "p_mix": p_mix, "q": qq, "ret": ret,
             "pop_min": pops.min(1), "pop_max": pops.max(1), "sum_pw": wm[comp].sum(1), "sum_q": q[comp].sum(1),
             "h1": nums[comp[:, 0]], "h2": nums[comp[:, 1]] if comp.shape[1] > 1 else np.zeros(m, int),
             "h3": nums[comp[:, 2]] if comp.shape[1] > 2 else np.zeros(m, int)}
        rows.append(d)

    # 単勝・複勝
    mk = RankDistribution(q, *alpha_m)
    p2q, p3q = mk._p2, mk._p3
    q_place = p2q.sum(1) + p2q.sum(0) + p3q.sum((0, 1))
    ok_w = np.isfinite(wo) & (wo >= 1.0)
    if "tansho" not in broken:
        sel = np.where(ok_w)[0]
        add("tansho", sel, wo[sel], wm[sel], wx[sel], q[sel], g["ret_win"].to_numpy(float)[sel] / 100.0)
    ok_p = np.isfinite(pa) & (pa > 0)
    if "fukusho" not in broken:
        sel = np.where(ok_p)[0]
        add("fukusho", sel, pa[sel], t3m[sel], t3x[sel], q_place[sel], g["ret_place"].to_numpy(float)[sel] / 100.0)
    # 組合せ券
    dist_m, _ = fit_rank_distribution(wm, t2m, t3m)
    dist_x, _ = fit_rank_distribution(wx, t2x, t3x) if "mixed" in sp else (dist_m, None)
    cpm, cpx = t36.combo_probs(dist_m), t36.combo_probs(dist_x)
    mkt = t36.combo_probs(mk)
    for bt in ("umaren", "wide", "umatan"):
        if bt in broken or pair_real is None or bt not in pair_real or bt not in cpm:
            continue
        idx, pm = cpm[bt]
        px = cpx[bt][1]
        qq = mkt[bt][1]
        lab = [tuple(int(nums[i]) for i in row) for row in idx]
        key = (lambda c: c) if bt == "umatan" else (lambda c: tuple(sorted(c)))
        od = np.array([pair_real[bt].get(key(c), np.nan) for c in lab])
        hb = hits.get(bt, {})
        ret = np.array([hb.get(key(c), 0.0) for c in lab]) / 100.0
        ev = np.maximum(pm * od, px * od)
        keep = np.unique(np.concatenate([top_idx(ev, K_EV), top_idx(np.where(np.isfinite(od), pm, np.nan), K_P)]))
        keep = keep[np.isfinite(od[keep])]
        add(bt, idx[keep], od[keep], pm[keep], px[keep], qq[keep], ret[keep])
    if tri is not None and "sanrentan" not in broken and "sanrentan" in cpm:
        cube = odds_cube(nums, tri)
        idx, pm = cpm["sanrentan"]
        px = cpx["sanrentan"][1]
        qq = mkt["sanrentan"][1]
        od = cube[idx[:, 0], idx[:, 1], idx[:, 2]]
        ev = np.maximum(pm * od, px * od)
        keep = np.unique(np.concatenate([top_idx(ev, K_TRI_EV), top_idx(np.where(np.isfinite(od), pm, np.nan), K_TRI_P)]))
        keep = keep[np.isfinite(od[keep])]
        lab = [tuple(int(nums[i]) for i in idx[j]) for j in keep]
        hb = hits.get("sanrentan", {})
        ret = np.array([hb.get(c, 0.0) for c in lab]) / 100.0
        add("sanrentan", idx[keep], od[keep], pm[keep], px[keep], qq[keep], ret)
    return rows


def concat_rows(all_rows: list[dict]) -> pd.DataFrame:
    parts = []
    for d in all_rows:
        m = len(d["odds"])
        parts.append(pd.DataFrame({k: (np.full(m, v) if np.ndim(v) == 0 else v) for k, v in d.items()}))
    return pd.concat(parts, ignore_index=True)


def cmd_build(a) -> None:
    from src.scripts.data import tp48_eval as te
    from src.scripts.data import tp48_2026 as te26
    import json

    year = int(a.year)
    frame = pd.read_parquet(BET / "tp36/frame_all.parquet")
    frame = frame[frame["year"] >= 2022].copy()
    inv = (1.0 / frame["win_odds"].where(frame["win_odds"] >= 1.0)).groupby(frame["race_id"]).sum()
    bad_odds = set(inv[~inv.between(1.0, 1.6)].index) | set(frame.loc[~(frame["win_odds"] >= 1.0), "race_id"])
    frame = frame[~frame["race_id"].isin(bad_odds)]
    hits, bad_pay = te.load_hits_all()
    pay26 = pd.read_parquet(B26 / "payoff.parquet")
    bad_pay = set(bad_pay) | set(pay26.loc[pay26["ambiguous"] | ~pay26["parsed"], "race_id"])
    frame = frame[~frame["race_id"].isin(bad_pay)]
    frame = te26.attach_2026_place(frame, hits)
    pair22 = t36.load_pair_real(set(frame.loc[frame["year"] == 2022, "race_id"]), 2022)
    alpha_m = t36.fit_market_alpha(frame[frame["year"] == 2022], pair22)
    del pair22
    print("market alpha", alpha_m, flush=True)
    dy = frame[frame["year"] == year]
    if a.max_races:
        dy = dy[dy["race_id"].isin(set(dy["race_id"].unique()[: a.max_races]))]
    ids = set(dy["race_id"])
    pair = load_pair_all(ids, (BET / "odds_pair.parquet") if year <= 2025 else (B26 / "odds_pair.parquet"), year if year <= 2025 else None)
    tri_by: dict = {}
    excl: set = set()
    if year >= 2024:
        ex = json.loads((TP48 / "exclude.json").read_text(encoding="utf-8")).get(str(year), {})
        excl = set(ex.get("missing", [])) | set(ex.get("inconsistent", []))
        tri_all = pd.read_parquet(TP48 / f"trifecta_{year}.parquet")
        tri_by = {rid: x for rid, x in tri_all.groupby("race_id") if rid in ids}
        del tri_all
    OUT.mkdir(parents=True, exist_ok=True)
    allrows: list[dict] = []
    dates: dict = {}
    for k, (rid, g) in enumerate(dy.groupby("race_id", sort=False)):
        r = build_race(g, hits.get(rid, {}), pair.get(rid), tri_by.get(rid) if rid not in excl else None, alpha_m)
        if r:
            allrows.extend(r)
            dates[rid] = str(g["date"].iloc[0])
        if k % 500 == 0:
            print(year, k, flush=True)
    df = concat_rows(allrows)
    df["date"] = df["race_id"].map(dates)
    df["year"] = year
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df["bt"] = df["bt"].astype(np.int8)
    f = OUT / f"tickets_{year}.parquet"
    df.to_parquet(f, index=False)
    print("saved", f, df.shape, flush=True)


def cmd_check(a) -> None:
    for y in range(2022, 2027):
        f = OUT / f"tickets_{y}.parquet"
        if not f.exists():
            print(y, "none")
            continue
        d = pd.read_parquet(f, columns=["race_id", "bt", "odds", "ret", "p_model", "q"])
        g = d.groupby("bt").agg(rows=("ret", "size"), ret_mean=("ret", "mean"), races=("race_id", "nunique"))
        print(y, len(d), d["race_id"].nunique())
        print(g.rename(index=BT_NAME).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--max-races", type=int, default=0)
    sp.add_parser("check")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
