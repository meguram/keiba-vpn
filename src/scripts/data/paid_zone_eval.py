"""有料ゾーンの推奨度の評価（docs/operations/paid_zone_tiers.md）。

python -m src.scripts.data.paid_zone_eval odds                      # 案B の想定オッズ（全券種）の精度と較正の当てはめ
python -m src.scripts.data.paid_zone_eval build --mode C|Bo|B|A [--years 2022 2023 2024 2025] [--shard i n]
python -m src.scripts.data.paid_zone_eval report                     # レース単位の推奨度ごとの回収率（確定払戻で精算）

モード: C=実オッズ（単勝・複勝・馬連・ワイド・馬単。3連複の過去オッズは無い）／Bo=実単勝・複勝から全券種の想定オッズを導く（案Bの上限）／
B=想定単勝・複勝（final_odds_prediction。2025 年だけ out-of-sample）から導く／A=確率のみ。
設計年は 2022〜23、確認年は 2024〜25。2026 は触らない。出力先は PZ_OUT（既定 data/local/modeling/betting/paid_zone）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PZ_OUT", ROOT / "data/local/modeling/betting/paid_zone"))
BET = ROOT / "data/local/modeling/betting"
DESIGN, CONFIRM = (2022, 2023), (2024, 2025)
BT_MAP = {"tansho": "tansho", "fukusho": "fukusho", "umaren": "umaren", "wide": "wide", "umatan": "umatan", "sanrenpuku": "sanrenpuku"}


# ───────────────────────── 部品 ─────────────────────────

def _combo(bt: str, pair) -> str:
    pair = [int(x) for x in pair]
    if bt in ("tansho", "fukusho"):
        return str(pair[0])
    if bt == "umatan":
        return ">".join(str(x) for x in pair)
    return "-".join(str(x) for x in sorted(pair))


class _Au:
    """audit_ev_calibration の必要な 2 関数だけ（本体はホストに無いので写す）。"""

    @staticmethod
    def _entries_for(g: pd.DataFrame):
        ent = [{"horse_number": int(h), "horse_name": f"#{int(h)}", "win_prob": float(p), "top2_prob": float(p2), "top3_prob": float(p3)}
               for h, p, p2, p3 in zip(g["horse_number"], g["p_win"], g["p_top2"], g["p_top3"])]
        odds = {int(h): {"win_odds": float(w_), "place_odds_min": float(a_), "place_odds_max": float(b_), "popularity": int(po) if po == po else None}
                for h, w_, a_, b_, po in zip(g["horse_number"], g["win_odds"], g["place_odds_min"], g["place_odds_max"], g["popularity"])}
        return ent, odds

    @staticmethod
    def _pair_odds_docs(pairs_r: pd.DataFrame) -> dict:
        out: dict = {"umaren": [], "wide": [], "umatan": []}
        for bt, c, o, omin, omax in zip(pairs_r["bet_type"], pairs_r["combo"], pairs_r["odds"], pairs_r["odds_min"], pairs_r["odds_max"]):
            if bt == "wide":
                out["wide"].append({"pair": [int(x) for x in c.split("-")], "odds_min": float(omin), "odds_max": float(omax)})
            elif bt == "umaren":
                out["umaren"].append({"pair": [int(x) for x in c.split("-")], "odds": float(o)})
            elif bt == "umatan":
                out["umatan"].append({"pair": [int(x) for x in c.split(">")], "odds": float(o)})
        return out


def load_inputs():
    from src.scripts.data.tp36_edge import load_frame

    au = _Au
    d = load_frame()
    pairs = pd.read_parquet(BET / "odds_pair.parquet")
    pairs = pairs[pairs["race_id"].isin(set(d["race_id"]))]
    return au, d, pairs, load_payoffs()


def _norm_combo(bt: str, combo: str) -> str:
    import re

    parts = [x for x in re.split(r"[>→-]", str(combo)) if x != ""]
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return str(combo)
    if bt in ("umatan", "sanrentan"):
        return ">".join(str(x) for x in nums)
    return "-".join(str(x) for x in sorted(nums))


def load_payoffs() -> dict[str, dict[str, dict[str, float]]]:
    """race_id → 券種 → 組み合わせ → 払戻（100 円あたり）。2022〜2025 年（2026 は読まない）。"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    d = pd.read_parquet(BET / "payoff.parquet", columns=["race_id", "bet_type", "combo", "payout"])
    d = d[d["bet_type"].isin(list(BT_MAP) + ["sanrentan"])]
    for rid, bt, combo, pay in zip(d["race_id"].astype(str), d["bet_type"], d["combo"], d["payout"]):
        out.setdefault(rid, {}).setdefault(bt, {})[_norm_combo(bt, combo)] = float(pay)
    return out


def _calibrate(items: list[dict]) -> dict:
    """推奨ポートフォリオの候補に較正後の EV を付ける（delegates._calibrate_candidates と同じ。本体を import しない）。"""
    from src.utils.ev_calibration import active_config, calibrate_portfolio

    cal = calibrate_portfolio(items, active_config(False))
    if cal is None:
        return {}
    for item, e in zip(items, cal["evs"]):
        item["ev_calibrated"] = round(e, 3)
    return {"expected_roi_calibrated": cal["expected_roi"]}


def predicted_2025() -> pd.DataFrame:
    """2025 年の想定単勝・複勝（final_odds_prediction。2020〜24 年で学習、2025 年は out-of-sample）。"""
    from src.pipeline.models.final_odds_predictor import FinalOddsPredictor

    f = pd.read_parquet(ROOT / "data/local/modeling/final_odds_eval_cache/eval_2025.parquet")
    res = FinalOddsPredictor().predict(f)
    assert len(res) == len(f)
    out = pd.DataFrame({"race_id": f["race_id"].astype(str).to_numpy(), "horse_number": f["horse_number"].astype(int).to_numpy(),
                        "pw": [r.predicted_win_odds for r in res], "pmin": [r.predicted_place_odds_min for r in res],
                        "pmax": [r.predicted_place_odds_max for r in res]})
    assert (out["horse_number"].to_numpy() == np.array([r.horse_number for r in res])).all()
    return out


# ───────────────────────── 想定オッズの精度 ─────────────────────────

def _fair_tables(win_odds: dict[int, float]):
    from src.pipeline.inference.exotic_probabilities import RankDistribution
    from src.utils import paid_zone as PZ

    items = sorted(win_odds.items())
    hns = [h for h, _ in items]
    q = np.array([1.0 / o for _, o in items], dtype=float)
    dist = RankDistribution(q / q.sum())
    pb = PZ.config().get("payback") or {}
    out = {"umaren": {}, "wide": {}, "umatan": {}}
    for i in range(len(hns)):
        for j in range(len(hns)):
            if i == j:
                continue
            out["umatan"][(hns[i], hns[j])] = float(pb.get("umatan", 0.75)) / max(dist.umatan(i, j), 1e-6)
            if i < j:
                out["umaren"][(hns[i], hns[j])] = float(pb.get("umaren", 0.775)) / max(dist.umaren(i, j), 1e-6)
                out["wide"][(hns[i], hns[j])] = float(pb.get("wide", 0.775)) / max(dist.wide(i, j), 1e-6)
    return out


def cmd_odds(a) -> None:
    from scipy.stats import spearmanr

    au, d, pairs, _ = load_inputs()
    pred = predicted_2025()
    pr = {(r, h): (w, mn, mx) for r, h, w, mn, mx in pred.itertuples(index=False)}
    rows, tick = [], 0
    pg = {r: g for r, g in pairs.groupby("race_id")}
    for rid, g in d.groupby("race_id"):
        if rid not in pg or len(g) < 5:
            continue
        year = int(rid[:4])
        sources = {"actual": {int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"])}}
        if year == 2025:
            p_ = {int(h): pr[(rid, int(h))][0] for h in g["horse_number"] if (rid, int(h)) in pr}
            if len(p_) == len(g):
                sources["predicted"] = p_
        P = pg[rid]
        for src, wo in sources.items():
            fair = _fair_tables(wo)
            for bt, c, o, omin, omax in zip(P["bet_type"], P["combo"], P["odds"], P["odds_min"], P["odds_max"]):
                if bt not in fair:
                    continue
                key = tuple(int(x) for x in c.replace(">", "-").split("-"))
                key = key if bt == "umatan" else tuple(sorted(key))
                act = float(omin) if bt == "wide" else float(o)
                if act > 0 and key in fair[bt]:
                    rows.append((rid, year, src, bt, math.log(fair[bt][key]), math.log(act)))
        tick += 1
        if tick % 2000 == 0:
            print(tick, flush=True)
    df = pd.DataFrame(rows, columns=["race_id", "year", "src", "bt", "lf", "la"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "odds_accuracy.parquet", index=False)
    res: dict = {"fit": {}, "table": []}
    for bt in ("umaren", "wide", "umatan"):
        t = df[(df["bt"] == bt) & (df["src"] == "actual") & (df["year"].isin(DESIGN))]
        b, a0 = np.polyfit(t["lf"].to_numpy(), t["la"].to_numpy(), 1)
        res["fit"][bt] = {"a": round(float(a0), 4), "b": round(float(b), 4), "n": int(len(t))}
    for (src, bt, yr), t in df.groupby(["src", "bt", "year"]):
        f = res["fit"].get(bt, {"a": 0.0, "b": 1.0})
        err_raw = t["lf"] - t["la"]
        err_cal = (f["a"] + f["b"] * t["lf"]) - t["la"]
        sp = []
        for _, tt in t.groupby("race_id"):
            if len(tt) > 5:
                s = spearmanr(tt["lf"], tt["la"])[0]
                if s == s:
                    sp.append(s)
        res["table"].append({"src": src, "bet_type": bt, "year": int(yr), "n": int(len(t)), "bias_raw": round(float(err_raw.mean()), 4),
                             "rmse_raw": round(float(np.sqrt((err_raw ** 2).mean())), 4), "mae_raw": round(float(err_raw.abs().mean()), 4),
                             "bias_cal": round(float(err_cal.mean()), 4), "rmse_cal": round(float(np.sqrt((err_cal ** 2).mean())), 4),
                             "spearman": round(float(np.mean(sp)), 4) if sp else None})
    res.update(_odds_extra(d, pg, pr))
    (OUT / "odds_accuracy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["fit"], ensure_ascii=False))
    print("3連複(勝ち馬券の払戻から):", json.dumps(res.get("trio_check"), ensure_ascii=False))
    print("想定単勝の較正(2025 の偶数番目のレースで当てはめ→奇数番目で確認):", json.dumps(res.get("single_fit"), ensure_ascii=False))
    for r in res["table"]:
        print(r)


def _odds_extra(d: pd.DataFrame, pg: dict, pr: dict) -> dict:
    """(1) 3連複: 過去の確定オッズが無いので、勝ち馬券の払戻と公正オッズの比（確率で重み付けした偏り）から a を決める。
    (2) 想定単勝・複勝の較正: 2025 年（out-of-sample）の偶数番目のレースで log 実 = a + b·log 想定 を当てはめ、奇数番目で確認。"""
    from src.pipeline.inference.exotic_probabilities import RankDistribution

    payoff = load_payoffs()
    rows = []
    for rid, g in d.groupby("race_id"):
        if len(g) < 5 or rid not in payoff:
            continue
        trio = (payoff[rid].get("sanrenpuku") or {})
        if not trio:
            continue
        hns = sorted(int(h) for h in g["horse_number"])
        wo = {int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"])}
        q = np.array([1.0 / wo[h] for h in hns])
        dist = RankDistribution(q / q.sum())
        idx = {h: i for i, h in enumerate(hns)}
        for combo, pay in trio.items():
            try:
                a_, b_, c_ = (idx[int(x)] for x in combo.split("-"))
            except (KeyError, ValueError):   # 取消・除外などで表に無い馬番を含む払戻は飛ばす
                continue
            fair = 0.75 / max(dist.sanrenpuku(a_, b_, c_), 1e-6)
            rows.append((int(rid[:4]), math.log(fair), math.log(pay / 100.0)))
    t = pd.DataFrame(rows, columns=["year", "lf", "la"])
    des = t[t["year"].isin(DESIGN)]
    a_trio = float((des["la"] - des["lf"]).mean())
    trio_check = {"a_design": round(a_trio, 4), "n_design": int(len(des))}
    for y in (2024, 2025):
        ty = t[t["year"] == y]
        trio_check[f"bias_{y}_raw"] = round(float((ty["lf"] - ty["la"]).mean()), 4)
        trio_check[f"bias_{y}_cal"] = round(float((ty["lf"] + a_trio - ty["la"]).mean()), 4)
        trio_check[f"n_{y}"] = int(len(ty))
    # 想定単勝の較正
    ids = sorted({r for r, _ in pr})
    fit_ids, test_ids = set(ids[0::2]), set(ids[1::2])
    q = d[d["year"] == 2025]
    pw = np.array([pr.get((r, int(h)), (np.nan,))[0] for r, h in zip(q["race_id"], q["horse_number"])], dtype=float)
    la = np.log(q["win_odds"].to_numpy(float))
    ok = np.isfinite(pw)
    is_fit = q["race_id"].isin(fit_ids).to_numpy() & ok
    is_test = q["race_id"].isin(test_ids).to_numpy() & ok
    b, a0 = np.polyfit(np.log(pw[is_fit]), la[is_fit], 1)
    err = lambda m: (a0 + b * np.log(pw[m])) - la[m]  # noqa: E731
    single = {"win": {"a": round(float(a0), 4), "b": round(float(b), 4), "n_fit": int(is_fit.sum())},
              "test_rmse_raw": round(float(np.sqrt(np.mean((np.log(pw[is_test]) - la[is_test]) ** 2))), 4),
              "test_rmse_cal": round(float(np.sqrt(np.mean(err(is_test) ** 2))), 4),
              "test_corr": round(float(np.corrcoef(np.log(pw[is_test]), la[is_test])[0, 1]), 4)}
    # 複勝（最小）
    pm = np.array([pr.get((r, int(h)), (np.nan, np.nan))[1] for r, h in zip(q["race_id"], q["horse_number"])], dtype=float)
    lp = np.log(q["place_odds_min"].to_numpy(float))
    okp = np.isfinite(pm) & (pm > 0) & np.isfinite(lp)
    fp = q["race_id"].isin(fit_ids).to_numpy() & okp
    bp, ap = np.polyfit(np.log(pm[fp]), lp[fp], 1)
    single["place_min"] = {"a": round(float(ap), 4), "b": round(float(bp), 4), "n_fit": int(fp.sum())}
    return {"trio_check": trio_check, "single_fit": single}


# ───────────────────────── 推奨度の評価 ─────────────────────────

def _items(cands, bets_cfg=None) -> list[dict]:
    return [{"bet_type": c.bet_type, "pair": list(c.pair), "odds": c.odds, "prob": c.prob, "ev": float(c.ev), "kelly_fraction": c.kelly_fraction,
             "bet_amount": c.bet_amount, "expected_return": c.expected_return} for c in cands]


def cmd_build(a) -> None:
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
    from src.utils import paid_zone as PZ

    au, d, pairs, payoff = load_inputs()
    d = d[d["year"].isin(a.years)]
    pg = {r: g for r, g in pairs.groupby("race_id")}
    pred = None
    test_ids: set[str] | None = None
    if a.mode == "B":
        pred = predicted_2025()
        pr = {(r, h): (w, mn, mx) for r, h, w, mn, mx in pred.itertuples(index=False)}
        test_ids = set(sorted({r for r, _ in pr})[1::2])   # 想定単勝の較正を当てはめていない奇数番目のレースだけで評価
    ids = sorted(r for r in d["race_id"].unique() if r in pg and (test_ids is None or r in test_ids))
    if a.shard:
        ids = ids[a.shard[0]::a.shard[1]]
    if a.limit:
        ids = ids[: a.limit]
    dg = {r: g for r, g in d.groupby("race_id")}
    bts_all = PZ.bet_types()
    bts_c = [b for b in bts_all if b != "sanrenpuku"]
    out = []
    for n, rid in enumerate(ids):
        g = dg[rid]
        if len(g) < 5:
            continue
        ent, odds = au._entries_for(g)
        preds = [{**e, "pred_score": math.log(max(e["win_prob"], 1e-9))} for e in ent]
        score, cands_out, mode_used = None, [], a.mode
        if a.mode == "C":
            single = {"entries": [{"horse_number": hn, **o} for hn, o in odds.items()]}
            pp = [{**p, "win_odds": odds[p["horse_number"]]["win_odds"], "place_odds_min": odds[p["horse_number"]]["place_odds_min"],
                   "place_odds_max": odds[p["horse_number"]]["place_odds_max"]} for p in preds]
            port = BettingOptimizer(BettingConfig(max_total_bet=None, bet_types=bts_c)).optimize(pp, au._pair_odds_docs(pg[rid]), 100000, single_odds=single)
            cands_out = _items(port["candidates"])
            cal = _calibrate(cands_out) if cands_out else {}
            tot = float(port["total_bet"] or 0)
            score = cal.get("expected_roi_calibrated") if cal else None
            if score is None and tot:
                score = float(port["expected_return"]) / tot
        elif a.mode in ("Bo", "B"):
            if a.mode == "Bo":
                win = {hn: o["win_odds"] for hn, o in odds.items()}
                single = {"entries": [{"horse_number": hn, **o} for hn, o in odds.items()]}
            else:
                rows = {int(h): pr.get((rid, int(h))) for h in g["horse_number"]}
                if any(v is None for v in rows.values()):
                    continue
                single = PZ.calibrate_single({"entries": [{"horse_number": h, "win_odds": v[0], "place_odds_min": v[1], "place_odds_max": v[2]} for h, v in rows.items()]})
                win = {e["horse_number"]: e["win_odds"] for e in single["entries"]}
            pairs_syn = PZ.synth_pair_odds(win)
            port = BettingOptimizer(BettingConfig(max_total_bet=None, bet_types=bts_all)).optimize(preds, pairs_syn, 100000, single_odds=single)
            cands_out = _items(port["candidates"])
            tot = float(port["total_bet"] or 0)
            score = float(port["expected_return"]) / tot if tot else None
        else:  # A
            cands_out = PZ.prob_only_candidates(preds)
            score = PZ.confidence_index(preds)
        m_for_roles = "C" if a.mode == "C" else ("B" if a.mode in ("B", "Bo") else "A")
        payload = PZ.build_paid_payload(cands_out, preds, race_id=rid, roi=99.0, m=m_for_roles, max_total_bet=2000) if cands_out else {"bets": []}
        pay_r = payoff.get(rid, {})
        bets = []
        for b in payload["bets"]:
            got = float((pay_r.get(BT_MAP[b["bet_type"]]) or {}).get(_combo(b["bet_type"], b["combo"]), 0.0))
            bets.append({"bt": b["bet_type"], "role": b["role"], "units": b["units"], "pay100": got})
        out.append({"race_id": rid, "year": int(rid[:4]), "score": None if score is None else float(score), "n_runners": len(g), "bets": json.dumps(bets)})
        if (n + 1) % 500 == 0:
            print(a.mode, n + 1, "/", len(ids), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}_" + ("all" if not a.shard else f"s{a.shard[0]}of{a.shard[1]}") + (f"_{a.tag}" if a.tag else "")
    pd.DataFrame(out).to_parquet(OUT / f"tiers_{tag}.parquet", index=False)
    print("保存:", tag, len(out))


def _race_table(df: pd.DataFrame, level: np.ndarray, unit_yen: int = 100) -> pd.DataFrame:
    stake = np.zeros(len(df))
    ret = np.zeros(len(df))
    for i, (bs, lv) in enumerate(zip(df["bets"], level)):
        if lv < 1:
            continue
        for b in json.loads(bs):
            stake[i] += b["units"] * unit_yen
            ret[i] += b["units"] * b["pay100"] / 100 * unit_yen
    t = df[["race_id", "year", "score"]].copy()
    t["level"] = level
    t["stake"], t["ret"] = stake, ret
    return t


def _ci(stake: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    if stake.sum() <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(stake)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        s = stake[ix].sum()
        vals.append(ret[ix].sum() / s if s > 0 else np.nan)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def _levels(score: np.ndarray, th: list[float]) -> np.ndarray:
    lv = np.zeros(len(score), dtype=int)
    for i, t in enumerate(th, 1):
        lv[np.nan_to_num(score, nan=-1e9) >= t] = i
    return lv


def _design_thresholds(c_scores: np.ndarray, m_scores: np.ndarray, c_th: list[float]) -> list[float]:
    """C の設計年の段階の割合に合わせて、他モードのしきい値を設計年の分位で決める（ROI を見て決めない）。"""
    share = np.array([(c_scores >= t).mean() for t in c_th])      # 各段階の下限以上の割合（累積）
    ms = m_scores[~np.isnan(m_scores)]
    return [float(np.quantile(ms, 1 - s)) if s > 0 else float("inf") for s in share]


def cmd_report(a) -> None:
    from src.utils import paid_zone as PZ

    def load(mode):
        fs = sorted(OUT.glob(f"tiers_{mode}_*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None

    data = {m: load(m) for m in ("C", "Bo", "B", "A")}
    c_th = PZ.thresholds("C")
    c_design = data["C"][data["C"]["year"].isin(DESIGN)]["score"].to_numpy(float) if data["C"] is not None else None
    ths = {"C": c_th}
    for m in ("Bo", "B", "A"):
        if data[m] is None or c_design is None:
            continue
        src = data["Bo" if m == "B" else m]
        ths[m] = _design_thresholds(c_design, src[src["year"].isin(DESIGN)]["score"].to_numpy(float), c_th)
    res: dict = {"thresholds": ths, "tables": {}}
    for m, df in data.items():
        if df is None or m not in ths:
            continue
        b_ids = set(data["B"]["race_id"]) if data["B"] is not None else set()
        for label, years in (("設計年(2022-23)", DESIGN), ("確認年(2024-25)", CONFIRM), ("2025のみ", (2025,)), ("B比較用(2025の奇数番目)", (2025,))):
            sub = df[df["year"].isin(years)]
            if label.startswith("B比較用"):
                sub = sub[sub["race_id"].isin(b_ids)]
            if sub.empty:
                continue
            lv = _levels(sub["score"].to_numpy(float), ths[m])
            t = _race_table(sub.reset_index(drop=True), lv)
            rows = []
            for k in range(0, 5):
                s = t[t["level"] == k]
                rows.append({"level": k, "n_races": int(len(s)), "stake": float(s["stake"].sum()), "ret": float(s["ret"].sum()),
                             "roi": float(s["ret"].sum() / s["stake"].sum()) if s["stake"].sum() > 0 else None,
                             "ci": _ci(s["stake"].to_numpy(), s["ret"].to_numpy()) if s["stake"].sum() > 0 else None,
                             "hit_rate": float((s["ret"] > 0).mean()) if len(s) and s["stake"].sum() > 0 else None,
                             "roi_ex_top3": float((s["ret"].sum() - s["ret"].nlargest(3).sum()) / s["stake"].sum()) if s["stake"].sum() > 0 else None})
            bet = t[t["level"] >= 1]
            rho = None
            ok = [r for r in rows if r["roi"] is not None and r["level"] >= 1]
            if len(ok) >= 3:
                from scipy.stats import spearmanr

                rho = float(spearmanr([r["level"] for r in ok], [r["roi"] for r in ok])[0])
            # 段階に依らず『スコア（期待回収率や確信度）が高いレースほど回収率が高いか』: 買い目のあるレースでの順位相関（スコア × レースの回収率）
            lv_all = np.full(len(sub), 4, dtype=int)
            ta = _race_table(sub.reset_index(drop=True), lv_all)
            ta = ta[(ta["stake"] > 0) & ta["score"].notna()]
            rho_all = ci_rho = None
            if len(ta) > 30:
                from scipy.stats import spearmanr

                roi_r = (ta["ret"] / ta["stake"]).to_numpy()
                sc = ta["score"].to_numpy(float)
                rho_all = float(spearmanr(sc, roi_r)[0])
                rng = np.random.default_rng(0)
                vals = []
                for _ in range(300):
                    ix = rng.integers(0, len(ta), len(ta))
                    vals.append(spearmanr(sc[ix], roi_r[ix])[0])
                ci_rho = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
            res["tables"][f"{m}|{label}"] = {"score_roi_rho": rho_all, "score_roi_rho_ci": ci_rho,
                                              "all_bets_roi": float(ta["ret"].sum() / ta["stake"].sum()) if len(ta) else None, "n_bet_races": int(len(ta)),
                                              "rows": rows, "portfolio_roi": float(bet["ret"].sum() / bet["stake"].sum()) if bet["stake"].sum() > 0 else None,
                                              "portfolio_ci": _ci(bet["stake"].to_numpy(), bet["ret"].to_numpy()), "n_all": int(len(t)), "monotone_rho": rho}
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["thresholds"], ensure_ascii=False))
    for k, v in res["tables"].items():
        print("==", k, "全体", v["n_all"], "R / 推奨度1以上の回収率", None if v["portfolio_roi"] is None else round(v["portfolio_roi"], 3), v["portfolio_ci"], "単調性ρ", v["monotone_rho"], "| スコア×回収率の順位相関", None if v["score_roi_rho"] is None else round(v["score_roi_rho"], 3), None if not v["score_roi_rho_ci"] else [round(x, 3) for x in v["score_roi_rho_ci"]], "| 買い目のある全レースの回収率", None if v["all_bets_roi"] is None else round(v["all_bets_roi"], 3), v["n_bet_races"], "R")
        for r in v["rows"]:
            print("  ", r["level"], r["n_races"], None if r["roi"] is None else round(r["roi"], 3), None if not r["ci"] else [round(x, 3) for x in r["ci"]],
                  "的中", None if r["hit_rate"] is None else round(r["hit_rate"], 3), "上位3除外", None if r["roi_ex_top3"] is None else round(r["roi_ex_top3"], 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("odds")
    b = sp.add_parser("build")
    b.add_argument("--mode", required=True, choices=["C", "Bo", "B", "A"])
    b.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    b.add_argument("--shard", type=int, nargs=2, default=None)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--tag", default="")
    sp.add_parser("report")
    a = ap.parse_args()
    {"odds": cmd_odds, "build": cmd_build, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
