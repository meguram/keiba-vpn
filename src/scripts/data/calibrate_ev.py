"""期待値（EV）の較正の検証: 表示する EV が、そのまま買い続けたときの実現回収率に収束するか。

python -m src.scripts.data.calibrate_ev build_horse                      # 全馬の単勝・複勝・連対の EV と実現（score_entries と同じ式）
python -m src.scripts.data.calibrate_ev build_portfolio --years 2022 2023 # 推奨ポートフォリオ（BettingOptimizer）の候補とレースごとの期待回収率・実現
python -m src.scripts.data.calibrate_ev build_picks                      # 画面の買い目提案（suggest_race_bets）の EV と実現
python -m src.scripts.data.calibrate_ev fit                              # 較正方式の walk-forward 比較（年 Y は Y より前の年で当てはめ）
出力は CAL_OUT（既定 data/local/modeling/betting/ev_calibration/）。ホストの出力先が書けないときは CAL_OUT で指定。

較正は『選ばれた後の EV』（winner's curse を含む）で当てはめる。画面に出る EV は EV の高い買い目を選んだ後の値だから、全馬ではなく
推奨ポートフォリオの候補・買い目提案・レース単位の期待回収率そのものを較正対象にする。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import audit_ev_calibration as au
from src.utils.ev_calibration import apply_spec

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("CAL_OUT", ROOT / "data/local/modeling/betting/ev_calibration"))
BASE = {"win": 0.80, "place": 0.80, "top2": 0.775, "tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "trio": 0.75,
        "tierce": 0.725, "portfolio": 0.80}
KIND_OF_BET = {"tansho": "win", "fukusho": "place", "umaren": "umaren", "wide": "wide", "umatan": "umatan", "trio": "trio", "tierce": "tierce"}


# ───────────────────────────────────────── 較正の部品（テスト対象）
def pava(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """単調増加の回帰（isotonic）。x の昇順に並べた (x, y) を重み w でプールし、ブロックの平均 x と平均 y を返す。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    o = np.argsort(x, kind="mergesort")
    x, y, w = x[o], y[o], w[o]
    bx, by, bw = [], [], []
    for xi, yi, wi in zip(x, y, w):
        bx.append(xi * wi)
        by.append(yi * wi)
        bw.append(wi)
        while len(by) > 1 and by[-2] / bw[-2] > by[-1] / bw[-1]:
            lx, ly, lw = bx.pop(), by.pop(), bw.pop()
            bx[-1] += lx
            by[-1] += ly
            bw[-1] += lw
    return np.array([a / c for a, c in zip(bx, bw)]), np.array([b / c for b, c in zip(by, bw)])


def fit_isotonic_knots(ev: np.ndarray, ret: np.ndarray, w: np.ndarray | None = None, n_bins: int = 20, min_per_bin: int = 300) -> dict:
    """EV → 期待される実現回収率の単調な折れ線。点ごとの PAVA はノイズの大きい端で暴れるので、EV の分位で等数の帯にまとめた
    帯の（重み付き）平均 EV と平均実現を PAVA にかけ、帯の中心を結び目にする。端は外挿しない（最小/最大の結び目の値で打ち切り）。"""
    ev, ret = np.asarray(ev, float), np.asarray(ret, float)
    m = np.isfinite(ev) & np.isfinite(ret)
    ev, ret = ev[m], ret[m]
    w = np.ones_like(ret) if w is None else np.asarray(w, float)[m]
    nb = int(max(2, min(n_bins, len(ev) // min_per_bin)))
    qs = np.unique(np.quantile(ev, np.linspace(0, 1, nb + 1)))
    bi = np.clip(np.searchsorted(qs, ev, side="right") - 1, 0, len(qs) - 2)
    bx, by, bw = [], [], []
    for b in range(len(qs) - 1):
        k = bi == b
        if k.sum() == 0:
            continue
        bx.append(float(np.average(ev[k], weights=w[k])))
        by.append(float(np.average(ret[k], weights=w[k])))
        bw.append(float(w[k].sum()))
    px, py = pava(np.array(bx), np.array(by), np.array(bw))
    return {"method": "isotonic", "x": [round(float(a), 5) for a in px], "y": [round(float(b), 5) for b in py]}


def fit_linear(ev: np.ndarray, ret: np.ndarray, w: np.ndarray | None = None) -> dict:
    """実現 = a + b × EV の重み付き最小二乗（切片あり）。b を [0, 1] に、a は平均が合うように調整。"""
    ev, ret = np.asarray(ev, float), np.asarray(ret, float)
    m = np.isfinite(ev) & np.isfinite(ret)
    ev, ret = ev[m], ret[m]
    w = np.ones_like(ret) if w is None else np.asarray(w, float)[m]
    xm, ym = np.average(ev, weights=w), np.average(ret, weights=w)
    sxx = float(np.average((ev - xm) ** 2, weights=w))
    b = 0.0 if sxx <= 0 else float(np.average((ev - xm) * (ret - ym), weights=w) / sxx)
    b = float(np.clip(b, 0.0, 1.0))
    return {"method": "linear", "a": round(ym - b * xm, 5), "b": round(b, 5)}


def fit_shrink(ev: np.ndarray, ret: np.ndarray, base: float, w: np.ndarray | None = None) -> dict:
    """現行の縮約 EV_adj = base + k × (EV − base)。k は原点（base）を通る最小二乗を [0, 1] に丸めた値。"""
    ev, ret = np.asarray(ev, float), np.asarray(ret, float)
    m = np.isfinite(ev) & np.isfinite(ret)
    x, y = ev[m] - base, ret[m] - base
    w = np.ones_like(x) if w is None else np.asarray(w, float)[m]
    d = float((w * x * x).sum())
    k = 0.0 if d <= 0 else float(np.clip((w * x * y).sum() / d, 0.0, 1.0))
    return {"method": "shrink", "base": base, "k": round(k, 5)}


def fit_const(ret: np.ndarray, w: np.ndarray | None = None) -> dict:
    """定数: EV の高低に関係なく、当てはめ期間の実現回収率の（重み付き）平均を返す。EV で見分けられない券種の正直な表示。"""
    ret = np.asarray(ret, float)
    m = np.isfinite(ret)
    w = np.ones_like(ret) if w is None else np.asarray(w, float)
    return {"method": "const", "value": round(float(np.average(ret[m], weights=w[m])), 5)}


def conservative(base: float) -> dict:
    """保守案: 常に払戻率（市場並み）。モデルに優位がないとみなす。"""
    return {"method": "market", "base": base}


def calib_metrics(pred: np.ndarray, ret: np.ndarray, race: np.ndarray, w: np.ndarray | None = None, n_bins: int = 8, n_boot: int = 300, seed: int = 0) -> dict:
    """予測（表示する EV）と実現の収束の指標。帯は予測の分位。
    gap = 平均予測 − 平均実現（全体のずれ）、ece = 帯ごとの |平均予測 − 平均実現| の重み付き平均、slope/intercept = 実現 = a + b×予測、
    cover = 帯のうち、実現のレース単位ブートストラップ 95% 区間が帯の平均予測を含む割合。"""
    p, r = np.asarray(pred, float), np.asarray(ret, float)
    w = np.ones_like(r) if w is None else np.asarray(w, float)
    codes = pd.factorize(np.asarray(race))[0]
    n = len(p)
    if n < 50:
        return {"n": int(n)}
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bi = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, max(len(qs) - 2, 0))
    rng = np.random.default_rng(seed)
    bins, ece, covered = [], 0.0, 0
    for b in range(int(bi.max()) + 1):
        m = bi == b
        if m.sum() < 20:
            continue
        mp, mr = float(np.average(p[m], weights=w[m])), float(np.average(r[m], weights=w[m]))
        g = pd.DataFrame({"c": codes[m], "s": r[m] * w[m], "w": w[m]}).groupby("c").agg(s=("s", "sum"), w=("w", "sum"))
        s, ww = g["s"].to_numpy(), g["w"].to_numpy()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = s[idx].sum(axis=1) / np.maximum(ww[idx].sum(axis=1), 1e-9)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ece += abs(mp - mr) * w[m].sum()
        covered += int(lo <= mp <= hi)
        bins.append({"pred": round(mp, 4), "real": round(mr, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": int(m.sum())})
    sl = fit_linear(p, r, w)
    mse = float(np.average((p - r) ** 2, weights=w))
    return {"n": int(n), "mse": round(mse, 5), "gap": round(float(np.average(p, weights=w) - np.average(r, weights=w)), 4), "ece": round(float(ece / w.sum()), 4),
            "slope": sl["b"], "intercept": sl["a"], "cover": f"{covered}/{len(bins)}", "bins": bins}


# ───────────────────────────────────────── データ作り

BET26 = ROOT / "data/local/modeling/betting_2026"


def load_all_2026():
    """2026 年（未使用）: V4（bundle_v4_2025）の予測 × 確定オッズ・確定払戻。複勝・馬連・ワイド・馬単の確定オッズは betting_2026/ にある。"""
    from src.scripts.data.tp36_final2026 import TP

    w = pd.read_parquet(BET26 / "odds_win_place.parquet")
    pay = pd.read_parquet(BET26 / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"]) if {"ambiguous", "parsed"} <= set(pay.columns) else set()
    bad_odds = set(w.loc[~(w["win_odds"] >= 1.0), "race_id"])
    w = w[~w["race_id"].isin(bad | bad_odds)].copy()
    inv = (1 / w["win_odds"]).groupby(w["race_id"]).sum()
    w = w[~w["race_id"].isin(set(inv[~inv.between(1.0, 1.6)].index))].copy()          # 単勝オッズが壊れているレース
    pred = pd.concat([pd.read_parquet(p_) for p_ in sorted((TP / "pred2026").glob("*.parquet"))], ignore_index=True)
    pred = pred[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]].astype({"race_id": str, "horse_id": str})
    w["horse_id"] = w["horse_id"].astype(str)
    d = w.merge(pred, on=["race_id", "horse_id"], how="inner")
    for bt, col in (("tansho", "ret_win"), ("fukusho", "ret_place")):
        q = pay[pay["bet_type"] == bt][["race_id", "combo", "payout"]].copy()
        q["horse_number"] = q["combo"].astype(int)
        d = d.merge(q[["race_id", "horse_number", "payout"]].rename(columns={"payout": col}), on=["race_id", "horse_number"], how="left")
        d[col] = d[col].fillna(0.0)
    d["year"] = 2026
    d["place_mid"] = (d["place_odds_min"] + d["place_odds_max"]) / 2
    s_ = d.groupby("race_id")["p_win"].transform("sum")
    d = d[(s_ > 0.97) & (s_ < 1.03)].copy()
    pairs = pd.read_parquet(BET26 / "odds_pair.parquet")
    pairs = pairs[pairs["race_id"].isin(set(d["race_id"]))]
    pay = pay[pay["race_id"].isin(set(d["race_id"]))]
    return d, pay, pairs


def _source(a):
    return load_all_2026() if getattr(a, "src", "") == "2026" else au.load_all()

def _hit_top2(pay: pd.DataFrame) -> dict[str, set[int]]:
    """レースごとの 1・2 着の馬番（払戻の馬連の的中組合せから）。"""
    out: dict[str, set[int]] = {}
    for r, c in zip(pay.loc[pay["bet_type"] == "umaren", "race_id"], pay.loc[pay["bet_type"] == "umaren", "combo"]):
        out.setdefault(str(r), set()).update(int(x) for x in str(c).split("-"))
    return out


def cmd_build_horse(a) -> None:
    """全馬の EV（local_predictions.score_entries と同じ式）と実現。連対の実現は『市場の公正オッズで買ったとき』の仮想の払戻。"""
    from src.utils.race_probabilities import market_top2_fair_odds, place_effective_odds

    d, pay, _ = _source(a)
    d = d[d["year"].isin(au.YEARS + (2026,))].copy()
    top2 = _hit_top2(pay)
    rows = []
    for r, g in d.groupby("race_id"):
        fair = market_top2_fair_odds({int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"])})
        hit2 = top2.get(str(r), set())
        for h, wo, pmin, pmax, pw, p2, p3, rw, rp in zip(g["horse_number"], g["win_odds"], g["place_odds_min"], g["place_odds_max"], g["p_win"], g["p_top2"],
                                                           g["p_top3"], g["ret_win"], g["ret_place"]):
            h = int(h)
            pe = place_effective_odds(pmin, pmax) or 0.0
            f2 = fair.get(h)
            rows.append({"race_id": r, "year": int(r[:4]), "horse_number": h, "win_odds": float(wo), "place_eff": pe, "top2_fair": f2,
                         "ev_win": pw * wo, "ev_place": p3 * pe if pe else np.nan, "ev_top2": p2 * f2 if f2 else np.nan,
                         "ret_win": rw / 100, "ret_place": rp / 100, "ret_top2": (f2 if (h in hit2 and f2) else 0.0)})
    OUT.mkdir(parents=True, exist_ok=True)
    name = "horse2026_full.parquet" if getattr(a, "src", "") == "2026" else "horse.parquet"
    pd.DataFrame(rows).to_parquet(OUT / name, index=False)
    print("保存:", OUT / name, len(rows))


def cmd_build_portfolio(a) -> None:
    from src.api.v1.delegates import value_level  # noqa: F401  (import チェック)
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer

    d, pay, pairs = _source(a)
    d = d[d["year"].isin(a.years)]
    hit = au._payoff_dicts(pay)
    pair_by_race = {r: g for r, g in pairs.groupby("race_id")}
    cfg = BettingConfig(max_total_bet=2000, bet_types=["tansho", "fukusho", "umaren", "wide"])
    opt = BettingOptimizer(cfg)
    races, cand = [], []
    ids = sorted(d["race_id"].unique())
    if a.shard is not None:
        ids = ids[a.shard[0]::a.shard[1]]
    if a.limit:
        ids = ids[: a.limit]
    dg = {r: g for r, g in d.groupby("race_id")}
    for n, r in enumerate(ids):
        if r not in pair_by_race:
            continue
        g = dg[r]
        ent, odds = au._entries_for(g)
        preds = [{**e, "win_odds": odds[e["horse_number"]]["win_odds"], "place_odds_min": odds[e["horse_number"]]["place_odds_min"],
                  "place_odds_max": odds[e["horse_number"]]["place_odds_max"], "pred_score": math.log(max(e["win_prob"], 1e-9))} for e in ent]
        single = {"entries": [{"horse_number": hn, **o} for hn, o in odds.items()]}
        port = opt.optimize(preds, au._pair_odds_docs(pair_by_race[r]), 100000, single_odds=single)
        cs = port["candidates"]
        if not cs:
            races.append({"race_id": r, "year": int(r[:4]), "n_bets": 0, "claimed": np.nan, "realized": np.nan, "stake": 0, "ret": 0.0})
            continue
        stake = float(sum(c.bet_amount for c in cs))
        ret = 0.0
        for c in cs:
            key = str(c.pair[0]) if c.bet_type in ("tansho", "fukusho") else (au._key2(*c.pair, ordered=True) if c.bet_type == "umatan" else au._key2(*c.pair))
            bt = {"tansho": "tansho", "fukusho": "fukusho", "umaren": "umaren", "wide": "wide", "umatan": "umatan"}[c.bet_type]
            u = hit[bt].get(r, {}).get(key, 0.0) / 100
            ret += u * c.bet_amount
            cand.append({"race_id": r, "year": int(r[:4]), "bet_type": c.bet_type, "ev": float(c.ev), "odds": float(c.odds), "prob": float(c.prob),
                         "ret": u, "amount": float(c.bet_amount)})
        races.append({"race_id": r, "year": int(r[:4]), "n_bets": len(cs), "claimed": float(port["expected_roi"]), "realized": ret / stake, "stake": stake, "ret": ret})
        if (n + 1) % 500 == 0:
            print(n + 1, "/", len(ids), flush=True)
    tag = a.tag or "_".join(str(y) for y in a.years)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(races).to_parquet(OUT / f"race_{tag}.parquet", index=False)
    pd.DataFrame(cand).to_parquet(OUT / f"cand_{tag}.parquet", index=False)
    print("保存:", tag, len(races), len(cand))


def cmd_build_picks(a) -> None:
    from src.api.local_predictions import score_entries
    from src.utils.race_bet_suggestion import suggest_race_bets

    d, pay, _ = au.load_all()
    d = d[d["year"].isin(au.YEARS)]
    hit = au._payoff_dicts(pay)
    rows = []
    for r, g in d.groupby("race_id"):
        ent, odds = au._entries_for(g)
        sug = suggest_race_bets(score_entries(ent, odds))
        for p in sug.get("picks", []):
            rows.append({"race_id": r, "year": int(r[:4]), "bet_type": p["bet_type"], "ev": float(p["ev"] or 0.0), "confidence": p["confidence"],
                         "ret": au._pick_return(p, hit, r)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "picks.parquet", index=False)
    print("保存:", OUT / "picks.parquet", len(rows))


# ───────────────────────────────────────── 当てはめと評価
METHODS = ("raw", "shrink", "linear", "isotonic", "const", "market")


def fit_method(method: str, ev: np.ndarray, ret: np.ndarray, base: float, w: np.ndarray | None = None) -> dict | None:
    if method == "raw":
        return {"method": "linear", "a": 0.0, "b": 1.0}
    if method == "market":
        return conservative(base)
    if len(ev) < 200:
        return None
    return {"shrink": lambda: fit_shrink(ev, ret, base, w), "linear": lambda: fit_linear(ev, ret, w), "isotonic": lambda: fit_isotonic_knots(ev, ret, w),
            "const": lambda: fit_const(ret, w)}[method]()


def walk_forward(df: pd.DataFrame, ev_col: str, ret_col: str, base: float, w_col: str | None = None, years=(2023, 2024, 2025), methods=METHODS) -> dict:
    """年 Y を Y より前の年で当てはめて Y で評価。methods ごとに年別の指標と、全年をまとめた指標を返す。"""
    out: dict = {}
    for m in methods:
        per_year, preds_all = {}, []
        for y in years:
            tr, te = df[df["year"] < y], df[df["year"] == y]
            if te.empty:
                continue
            spec = fit_method(m, tr[ev_col].to_numpy(), tr[ret_col].to_numpy(), base, None if w_col is None else tr[w_col].to_numpy())
            if spec is None:
                continue
            pred = apply_spec(spec, te[ev_col].to_numpy())
            w = None if w_col is None else te[w_col].to_numpy()
            per_year[str(y)] = calib_metrics(pred, te[ret_col].to_numpy(), te["race_id"].to_numpy(), w)
            preds_all.append(pd.DataFrame({"race_id": te["race_id"].to_numpy(), "pred": pred, "ret": te[ret_col].to_numpy(), "w": 1.0 if w is None else w}))
        if preds_all:
            pa = pd.concat(preds_all)
            per_year["pooled"] = calib_metrics(pa["pred"].to_numpy(), pa["ret"].to_numpy(), pa["race_id"].to_numpy(), pa["w"].to_numpy())
        out[m] = per_year
    return out


def _brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "bins"}


def cmd_fit(a) -> None:
    res: dict = {"horse": {}, "cand": {}, "race": {}, "picks": {}}
    hp = OUT / "horse.parquet"
    if hp.exists():
        h = pd.read_parquet(hp)
        for kind, evc, rc in (("win", "ev_win", "ret_win"), ("place", "ev_place", "ret_place"), ("top2", "ev_top2", "ret_top2")):
            x = h.dropna(subset=[evc])
            res["horse"][kind] = walk_forward(x, evc, rc, BASE[kind])
    cands = sorted(OUT.glob("cand_*.parquet"))
    if cands:
        c = pd.concat([pd.read_parquet(p) for p in cands], ignore_index=True)
        c = c[c["year"] <= 2025]                                  # 2026 年は当てはめ・選択に使わない
        for bt, g in c.groupby("bet_type"):
            res["cand"][bt] = walk_forward(g, "ev", "ret", BASE[KIND_OF_BET.get(bt, bt)], "amount")
        res["cand"]["__all_unweighted"] = None
    rcs = sorted(OUT.glob("race_*.parquet"))
    if rcs:
        r = pd.concat([pd.read_parquet(p) for p in rcs], ignore_index=True)
        r = r[(r["n_bets"] > 0) & (r["year"] <= 2025)].copy()
        res["race"] = walk_forward(r, "claimed", "realized", BASE["portfolio"], "stake")
    pk = OUT / "picks.parquet"
    if pk.exists():
        p = pd.read_parquet(pk)
        for bt, g in p.groupby("bet_type"):
            res["picks"][bt] = walk_forward(g, "ev", "ret", BASE[KIND_OF_BET.get(bt, bt)])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    print("保存:", OUT / "fit.json")


CAND_KINDS = ("tansho", "fukusho", "umaren", "wide")
HORSE_KINDS = (("win", "ev_win", "ret_win"), ("place", "ev_place", "ret_place"), ("top2", "ev_top2", "ret_top2"))
CAL_METHODS = ("shrink", "linear", "isotonic", "const")
# 『収束している』と認める基準（事前に固定）。プール（2023〜2025 の walk-forward）と、最終 2 年の年別の両方で見る。
CRITERIA = {"max_abs_gap": 0.08, "max_ece": 0.10, "min_cover": 0.75, "max_abs_gap_year": 0.12}
FALLBACK = "const"


def choose_method(by_method: dict, criteria: dict = CRITERIA) -> tuple[str | None, dict]:
    """shrink / linear / isotonic のうち基準を満たすものの中から、プールの較正誤差 ece が最小の方式を選ぶ。
    どれも満たさなければ const（EV の高低では見分けられない。実現回収率の平均だけを返す）にする。const は ece が自明に小さいので比較には入れない。"""
    def check(m: str):
        yrs = by_method.get(m) or {}
        r = yrs.get("pooled")
        if not r or "ece" not in r:
            return None
        n_ok, n_all = (int(x) for x in str(r["cover"]).split("/"))
        cover = n_ok / max(n_all, 1)
        gaps = [abs(yrs[y]["gap"]) for y in ("2024", "2025") if y in yrs and "gap" in yrs[y]]
        checks = {"gap": abs(r["gap"]) <= criteria["max_abs_gap"], "ece": r["ece"] <= criteria["max_ece"], "cover": cover >= criteria["min_cover"],
                  "year_gap": bool(gaps) and all(g <= criteria["max_abs_gap_year"] for g in gaps)}
        return {"ok": all(checks.values()), "checks": checks, "pooled": {k: r[k] for k in ("n", "mse", "gap", "ece", "slope", "intercept", "cover") if k in r}}
    cands = {m: check(m) for m in ("shrink", "linear", "isotonic")}
    cands = {m: c for m, c in cands.items() if c}
    passed = {m: c for m, c in cands.items() if c["ok"]}
    if passed:
        best = min(passed, key=lambda m: passed[m]["pooled"]["ece"])
        return best, {**passed[best], "all": {m: c["pooled"] for m, c in cands.items()}}
    cc = check("const")
    return ("const", {"ok": bool(cc and cc["ok"]), "note": "EV で見分けられない（基準を満たす方式なし）", **(cc or {}),
                      "all": {m: c["pooled"] for m, c in cands.items()}}) if cc else (None, {"ok": False, "reason": "データ不足"})


def cmd_build_2026(a) -> None:
    """2026 年（未使用）の単勝: 予測 × 確定単勝オッズ。複勝・組合せ券の確定オッズが無いので、評価できるのは単勝だけ。"""
    from src.scripts.data.tp36_final2026 import TP, build_frame2026

    d = build_frame2026(TP / "pred2026", ROOT / "data/page_reference/tables/2026/race_result_flat.parquet")
    d = d.assign(ev_win=d["p_win"] * d["win_odds"], ret_win=d["ret_win"] / 100)
    OUT.mkdir(parents=True, exist_ok=True)
    d[["race_id", "horse_number", "win_odds", "ev_win", "ret_win"]].assign(year=2026).to_parquet(OUT / "horse2026.parquet", index=False)
    print("保存:", OUT / "horse2026.parquet", len(d))


def _cal_by_kind(tr: pd.DataFrame, te: pd.DataFrame, methods: dict[str, str]) -> np.ndarray:
    out = np.full(len(te), np.nan)
    for bt, m in methods.items():
        g = tr[tr["bet_type"] == bt]
        spec = fit_method(m, g["ev"].to_numpy(), g["ret"].to_numpy(), BASE[KIND_OF_BET.get(bt, bt)], g["amount"].to_numpy())
        mask = (te["bet_type"] == bt).to_numpy()
        if spec is not None and mask.any():
            out[mask] = apply_spec(spec, te.loc[mask, "ev"].to_numpy())
    return out


def _level_table(pred: np.ndarray, ret: np.ndarray, stake: np.ndarray, race: np.ndarray, thresholds, n_boot: int = 300) -> list[dict]:
    """予測（較正後）が閾値以上のレースの実現回収率（購入額加重）と、レース単位ブートストラップ 95% 区間。"""
    rng = np.random.default_rng(0)
    out = []
    for t in thresholds:
        m = pred >= t
        if m.sum() < 30:
            out.append({"min": t, "n": int(m.sum())})
            continue
        a, s = ret[m], stake[m]
        idx = rng.integers(0, m.sum(), size=(n_boot, m.sum()))
        boot = a[idx].sum(axis=1) / s[idx].sum(axis=1)
        out.append({"min": t, "n": int(m.sum()), "share": round(float(m.mean()), 4), "pred": round(float(np.average(pred[m], weights=s)), 4),
                    "real": round(float(a.sum() / s.sum()), 4), "lo": round(float(np.percentile(boot, 2.5)), 4), "hi": round(float(np.percentile(boot, 97.5)), 4)})
    return out


def cmd_final(a) -> None:
    """選んだ方式で、(1) 候補の EV 較正 → レースの期待回収率（購入額加重）の walk-forward、(2) 閾値ごとの実現、(3) 2026 年の単勝、(4) 設定ファイルの出力。"""
    fit = json.loads((OUT / "fit.json").read_text(encoding="utf-8"))
    sel: dict = {"horse": {}, "cand": {}}
    for kind, _, _ in HORSE_KINDS:
        sel["horse"][kind] = choose_method(fit["horse"][kind])
    for bt in CAND_KINDS:
        sel["cand"][bt] = choose_method(fit["cand"][bt])
    c = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("cand_*.parquet"))], ignore_index=True)
    c = c[c["year"] <= 2025]                                      # 2026 年は当てはめ・選択に使わない
    r = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("race_*.parquet"))], ignore_index=True)
    r = r[(r["n_bets"] > 0) & (r["year"] <= 2025)]
    methods = {bt: (sel["cand"][bt][0] or "const") for bt in CAND_KINDS}
    rows = []
    for y in (2023, 2024, 2025):
        tr, te = c[c["year"] < y], c[c["year"] == y].copy()
        te["cal"] = _cal_by_kind(tr, te, methods)
        te = te.dropna(subset=["cal"])
        te["w"] = te["cal"] * te["amount"]
        g = te.groupby("race_id").agg(w=("w", "sum"), a=("amount", "sum")).reset_index()
        g["pred"] = g["w"] / g["a"]
        g["year"] = y
        rows.append(g)
    P = pd.concat(rows).merge(r[["race_id", "realized", "stake", "ret", "claimed"]], on="race_id")
    res: dict = {"selection": {k: {kk: [vv[0], vv[1]] for kk, vv in v.items()} for k, v in sel.items()}, "cand_methods": methods, "criteria": CRITERIA}
    res["race_composed"] = {"pooled": calib_metrics(P["pred"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()),
                            **{str(y): calib_metrics(g["pred"].to_numpy(), g["realized"].to_numpy(), g["race_id"].to_numpy(), g["stake"].to_numpy()) for y, g in P.groupby("year")}}
    res["race_raw_claimed"] = calib_metrics(P["claimed"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy())
    grid = [round(x, 2) for x in np.arange(0.85, 1.25, 0.03)]
    res["race_levels_pooled"] = _level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(), grid)
    res["race_levels_by_year"] = {str(y): _level_table(g["pred"].to_numpy(), g["ret"].to_numpy(), g["stake"].to_numpy(), g["race_id"].to_numpy(), grid)
                                  for y, g in P.groupby("year")}
    res["race_realized_overall"] = _level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(), [0.0])
    # 馬ごとの買い推奨度: 較正後の EV（単勝・連対・複勝）の最大とその券種の実現
    h = pd.read_parquet(OUT / "horse.parquet")
    parts = []
    for y in (2023, 2024, 2025):
        tr, te = h[h["year"] < y], h[h["year"] == y].copy()
        cal = {}
        for kind, evc, rc in HORSE_KINDS:
            x = tr.dropna(subset=[evc])
            spec = fit_method(sel["horse"][kind][0] or "const", x[evc].to_numpy(), x[rc].to_numpy(), BASE[kind])
            cal[kind] = apply_spec(spec, te[evc].fillna(0).to_numpy())
            te[f"cal_{kind}"] = np.where(te[evc].notna(), cal[kind], np.nan)
        parts.append(te)
    H = pd.concat(parts)
    cc = H[["cal_win", "cal_top2", "cal_place"]].to_numpy()
    arg = np.nanargmax(np.where(np.isnan(cc), -9, cc), axis=1)
    H["best_cal"] = np.nanmax(np.where(np.isnan(cc), -9, cc), axis=1)
    H["best_ret"] = np.choose(arg, [H["ret_win"].to_numpy(), H["ret_top2"].to_numpy(), H["ret_place"].to_numpy()])
    one = np.ones(len(H))
    hgrid = [round(x, 2) for x in np.arange(0.70, 1.20, 0.05)]
    res["horse_levels_pooled"] = _level_table(H["best_cal"].to_numpy(), H["best_ret"].to_numpy(), one, H["race_id"].to_numpy(), hgrid)
    res["horse_best_calibration"] = calib_metrics(H["best_cal"].to_numpy(), H["best_ret"].to_numpy(), H["race_id"].to_numpy())
    # 2026 年の単勝（未使用）
    p26 = OUT / "horse2026.parquet"
    if p26.exists():
        h26 = pd.read_parquet(p26)
        x = h[h["year"] <= 2025].dropna(subset=["ev_win"])
        spec = fit_method(sel["horse"]["win"][0] or "const", x["ev_win"].to_numpy(), x["ret_win"].to_numpy(), BASE["win"])
        pred = apply_spec(spec, h26["ev_win"].to_numpy())
        res["win_2026"] = {"raw": calib_metrics(h26["ev_win"].to_numpy(), h26["ret_win"].to_numpy(), h26["race_id"].to_numpy()),
                           "calibrated": calib_metrics(pred, h26["ret_win"].to_numpy(), h26["race_id"].to_numpy()), "method": sel["horse"]["win"][0]}
    # 設定（全期間 2022〜2025 で当てはめ）
    cfg_kinds: dict = {}
    for kind, evc, rc in HORSE_KINDS:
        x = h.dropna(subset=[evc])
        cfg_kinds[kind] = {**fit_method(sel["horse"][kind][0] or "const", x[evc].to_numpy(), x[rc].to_numpy(), BASE[kind]), "validated": bool(sel["horse"][kind][1]["ok"]), "n": int(len(x))}
    for bt in CAND_KINDS:
        g = c[c["bet_type"] == bt]
        cfg_kinds[f"cand_{bt}"] = {**fit_method(methods[bt], g["ev"].to_numpy(), g["ret"].to_numpy(), BASE[KIND_OF_BET.get(bt, bt)], g["amount"].to_numpy()),
                                   "validated": bool(sel["cand"][bt][1]["ok"]), "n": int(len(g))}
    res["config_kinds"] = cfg_kinds
    (OUT / "final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    print("保存:", OUT / "final.json")



TIERS = {"strong": 1.10, "recommend": 1.00, "watch": 0.90}      # 馬ごとの買い推奨度（較正後 EV の最大）
LEVELS = {"hot": 1.03, "good": 1.00, "ok": 0.97}               # レース価値（較正後の期待回収率。2022〜2026 年で当てはめ直した後の分布に対する閾値）
CAPS = {"ev": 1.30, "roi": 1.15}                                # 検証で裏づけのある範囲を超える値は打ち切る


def cmd_config(a) -> None:
    """設定ファイルを作る。馬ごとの EV の較正（win / top2 / place）は 2022〜2025 年で当てはめて固定した方式（2026 年の評価で確認済み）。
    推奨ポートフォリオの買い目・レースの期待回収率は、2026 年の評価で基準を満たさなかった（年によって実現回収率が 78〜105% と動く）ので、
    2022〜2026 年を合わせて当てはめ直し、『目安』として年別の範囲を併記する。"""
    f = json.loads((OUT / "final.json").read_text(encoding="utf-8"))
    e = json.loads((OUT / "eval2026.json").read_text(encoding="utf-8"))
    kinds: dict = {}
    for kind in ("win", "top2", "place"):
        k26 = e["horse"][kind]["calibrated"]
        ok26 = abs(k26["gap"]) <= CRITERIA["max_abs_gap_year"] and k26["ece"] <= CRITERIA["max_ece"]
        kinds[kind] = {**{kk: vv for kk, vv in f["config_kinds"][kind].items() if kk not in ("validated", "n")}, "validated": bool(f["config_kinds"][kind]["validated"] and ok26),
                       "n": f["config_kinds"][kind].get("n"), "gap_2026": k26["gap"], "ece_2026": k26["ece"]}
    c = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("cand_*.parquet"))], ignore_index=True)
    r = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("race_*.parquet"))], ignore_index=True)
    r = r[r["n_bets"] > 0]
    methods = f["cand_methods"]
    c["cal"] = np.nan
    for bt in CAND_KINDS:
        g = c[c["bet_type"] == bt]
        spec = fit_method(methods[bt], g["ev"].to_numpy(), g["ret"].to_numpy(), BASE[KIND_OF_BET.get(bt, bt)], g["amount"].to_numpy())
        c.loc[g.index, "cal"] = apply_spec(spec, g["ev"].to_numpy())
        r26 = e["cand"].get(bt, {}).get("calibrated", {})
        kinds[f"cand_{bt}"] = {**spec, "validated": False, "n": int(len(g)), "gap_2026": r26.get("gap"),
                               "note": "2026 年の評価で基準（年のずれ 0.12 以内・ece 0.10 以下）を満たさない。2022〜2026 年で当てはめ直した目安"}
    cw = c.assign(w=c["cal"] * c["amount"]).groupby("race_id").agg(w=("w", "sum"), a=("amount", "sum")).reset_index()
    cw["pred"] = (cw["w"] / cw["a"]).clip(upper=CAPS["roi"])
    P = r.merge(cw[["race_id", "pred"]], on="race_id")
    yearly = {int(y): round(float(g["ret"].sum() / g["stake"].sum()), 4) for y, g in P.groupby("year")}
    grid = [0.90, 0.93, 0.95, LEVELS["ok"], LEVELS["good"], LEVELS["hot"]]
    prod = {"yearly_realized": yearly, "pooled_realized": round(float(P["ret"].sum() / P["stake"].sum()), 4),
            "pred_quantiles": {str(q): round(float(P["pred"].quantile(q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "levels_pooled": _level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(), grid),
            "levels_by_year": {str(y): _level_table(g["pred"].to_numpy(), g["ret"].to_numpy(), g["stake"].to_numpy(), g["race_id"].to_numpy(), grid) for y, g in P.groupby("year")}}
    (OUT / "production.json").write_text(json.dumps(prod, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    lo, hi = min(yearly.values()), max(yearly.values())
    cfg = {
        "version": 2, "enabled": True,
        "label": "期待回収率（較正後）",
        "note": "表示する EV を、過去の実績（確定オッズ・確定払戻）に合わせて較正する。検証と手順: docs/html/modeling/chapters/ev_calibration_adopt.html。"
                "モデルや推奨ロジックを更新したら src/scripts/data/calibrate_ev.py で当てはめ直すこと。",
        "fit": {"horse": "2022-2025（V4 の leave-one-year-out / bundle_v4_2024）", "portfolio": "2022-2026（確定オッズ。max_total_bet=2000, min_ev=1.05, 15 点, 単勝・複勝・馬連・ワイド）"},
        "criteria": f["criteria"], "kinds": kinds, "tiers": TIERS, "levels": LEVELS, "cap": CAPS,
        "range": {"lo": lo, "hi": hi, "basis": "2022〜2026 年の年別の実現回収率（購入額加重）", "yearly": yearly},
        "validation": {"horse_best_2022_2025": {k: v for k, v in f["horse_best_calibration"].items() if k != "bins"},
                       "horse_best_2026": {k: v for k, v in e["horse_best"].items() if k != "bins"},
                       "race_composed_2023_2025": {k: v for k, v in f["race_composed"]["pooled"].items() if k != "bins"},
                       "race_2026": {k: v for k, v in e["race"]["calibrated"].items() if k != "bins"}},
    }
    path = Path(a.out) if a.out else ROOT / "config/ev_calibration.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print("保存:", path)


def cmd_eval2026(a) -> None:
    """2026 年（未使用）で、2022〜2025 年で当てはめて固定した較正（final.json の config_kinds）を評価する。方式と基準は 2026 を見る前に確定済み。"""
    f = json.loads((OUT / "final.json").read_text(encoding="utf-8"))
    ck = f["config_kinds"]
    res: dict = {"criteria_year_gap": CRITERIA["max_abs_gap_year"]}
    h = pd.read_parquet(OUT / "horse2026_full.parquet")
    hres, cols = {}, {}
    for kind, evc, rc in HORSE_KINDS:
        x = h.dropna(subset=[evc])
        pred = apply_spec(ck[kind], x[evc].to_numpy())
        cols[kind] = pd.Series(np.nan, index=h.index)
        cols[kind].loc[x.index] = pred
        hres[kind] = {"raw": calib_metrics(x[evc].to_numpy(), x[rc].to_numpy(), x["race_id"].to_numpy()),
                      "calibrated": calib_metrics(pred, x[rc].to_numpy(), x["race_id"].to_numpy())}
    res["horse"] = hres
    cc = np.column_stack([cols["win"], cols["top2"], cols["place"]])
    best = np.nanmax(np.where(np.isnan(cc), -9, cc), axis=1)
    arg = np.nanargmax(np.where(np.isnan(cc), -9, cc), axis=1)
    bret = np.choose(arg, [h["ret_win"].to_numpy(), h["ret_top2"].to_numpy(), h["ret_place"].to_numpy()])
    res["horse_best"] = calib_metrics(best, bret, h["race_id"].to_numpy())
    res["horse_tiers"] = _level_table(best, bret, np.ones(len(h)), h["race_id"].to_numpy(), [TIERS["watch"], TIERS["recommend"], TIERS["strong"]])
    c = pd.read_parquet(OUT / "cand_2026.parquet")
    c["cal"] = np.nan
    cres = {}
    for bt in CAND_KINDS:
        m = c["bet_type"] == bt
        if not m.any():
            continue
        c.loc[m, "cal"] = apply_spec(ck[f"cand_{bt}"], c.loc[m, "ev"].to_numpy())
        g = c[m]
        cres[bt] = {"raw": calib_metrics(g["ev"].to_numpy(), g["ret"].to_numpy(), g["race_id"].to_numpy(), g["amount"].to_numpy()),
                    "calibrated": calib_metrics(g["cal"].to_numpy(), g["ret"].to_numpy(), g["race_id"].to_numpy(), g["amount"].to_numpy())}
    res["cand"] = cres
    r = pd.read_parquet(OUT / "race_2026.parquet")
    r = r[r["n_bets"] > 0]
    cw = c.dropna(subset=["cal"]).assign(w=lambda x: x["cal"] * x["amount"]).groupby("race_id").agg(w=("w", "sum"), a=("amount", "sum")).reset_index()
    cw["pred"] = (cw["w"] / cw["a"]).clip(upper=CAPS["roi"])
    P = r.merge(cw[["race_id", "pred"]], on="race_id")
    res["race"] = {"n": int(len(P)), "raw": calib_metrics(P["claimed"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()),
                   "calibrated": calib_metrics(P["pred"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()),
                   "overall": _level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(), [0.0]),
                   "levels": _level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(),
                                          [LEVELS["ok"], LEVELS["good"], LEVELS["hot"]])}
    (OUT / "eval2026.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    print("保存:", OUT / "eval2026.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build_horse", "build_portfolio", "build_picks", "build_2026", "fit", "final", "config", "eval2026"])
    ap.add_argument("--years", type=int, nargs="+", default=list(au.YEARS))
    ap.add_argument("--shard", type=int, nargs=2, default=None, help="i n: レース一覧の i 番目から n 件おき")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--src", default="", help="2026 を指定すると betting_2026/ の入力を使う")
    ap.add_argument("--out", default="", help="config の出力先")
    a = ap.parse_args()
    {"build_horse": cmd_build_horse, "build_portfolio": cmd_build_portfolio, "build_picks": cmd_build_picks, "build_2026": cmd_build_2026, "fit": cmd_fit,
     "final": cmd_final, "config": cmd_config, "eval2026": cmd_eval2026}[a.cmd](a)


if __name__ == "__main__":
    main()
