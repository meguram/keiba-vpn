"""めぐAI印 v2（ユーザー定義）と現行 v1 の比較評価。

python -m src.scripts.data.mark_v2_eval build --out DIR            # ホストで実行: レースごとの記録（確率・確定オッズ・払戻・着順）を作る
python -m src.scripts.data.mark_v2_eval design --data DIR/races.pkl --out DIR   # 設計年(2022〜23)で各印の定義を決め、確認年(2024〜25)で確かめる
python -m src.scripts.data.mark_v2_eval final  --data DIR/races.pkl --out DIR   # 決めた設定の最終比較（2026 は最後に 1 回だけ）

規則: 閾値・定義は 2022〜2023 年だけで決め、2024〜2025 年は確認、2026 年（テストセット②）は選択に使わない。
市場オッズは特徴量ではなく、期待値（印の選び方）と精算にだけ使う。回収率は 1 頭 100 円の均等買い、確定オッズ・確定払戻。
レース単位の母数: 的中率は『その印が付いたレースのうち、いずれかが当たったレース』も併記し、区間はレースの再標本化で出す。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import os

ROOT = Path(os.environ.get("MARKV2_DATA_ROOT") or Path(__file__).resolve().parents[3])
FRAME = ROOT / "data/local/modeling/betting/tp36/frame_all.parquet"
RANKS = ROOT / "data/local/features/target/rank_tbl"
MARKS = ("honmei", "pair", "show_val", "star", "anchor")
SYM = {"honmei": "◎", "pair": "○", "show_val": "▲", "star": "★", "anchor": "△"}
DESIGN, CONFIRM, FINAL = (2022, 2023), (2024, 2025), (2026,)
BOOT = 500


# ───────────────────────── データ ─────────────────────────
def _arr(series, dtype=float):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=dtype, na_value=np.nan) if dtype is float else pd.to_numeric(series, errors="coerce").to_numpy(dtype=dtype)


def build(args) -> None:
    from src.utils.prob_calibration import calibrate_arrays, load_config

    assert load_config().get("targets"), "config/prob_calibration.json が必要です"
    d = pd.read_parquet(FRAME)
    rk = pd.concat([pd.read_parquet(p) for p in sorted(RANKS.glob("*/rank.parquet"))], ignore_index=True)
    rk["race_id"] = rk["race_id"].astype(str)
    rk["horse_id"] = rk["horse_id"].astype(str)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    d = d.merge(rk[["race_id", "horse_id", "rank"]], on=["race_id", "horse_id"], how="left")
    pay = load_payoffs()
    races = []
    for rid, g in d.groupby("race_id", sort=True):
        g = g.sort_values("horse_number")
        if len(g) < 5 or g["win_odds"].isna().any() or (g["win_odds"] <= 0).any() or g[["p_win", "p_top2", "p_top3"]].isna().any().any():
            continue
        rank = _arr(g["rank"])
        if np.isnan(rank).sum() > len(g) - 3:
            continue
        rank = np.where(np.isnan(rank), 99.0, rank)
        r = {
            "race_id": rid, "year": int(g["year"].iloc[0]), "month": int(str(g["date"].iloc[0])[5:7]), "n": len(g),
            "hn": _arr(g["horse_number"], int), "win_odds": _arr(g["win_odds"]),
            "pmin": _arr(g["place_odds_min"]), "pmax": _arr(g["place_odds_max"]),
            "raw": np.stack([_arr(g["p_win"]), _arr(g["p_top2"]), _arr(g["p_top3"])]),
            "ret_win": _arr(g["ret_win"]), "ret_place": _arr(g["ret_place"]), "rank": rank,
            "pop": _arr(g["popularity"]), "pay": pay.get(rid, {}),
        }
        r["cal"] = None
        if len(g) >= 8:
            cal = calibrate_arrays({"win": r["raw"][0], "top2": r["raw"][1], "top3": r["raw"][2]}, r["win_odds"])
            if cal is not None:
                r["cal"] = np.stack([cal["win"], cal["top2"], cal["top3"]]).astype(float)
        races.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(races, open(out / "races.pkl", "wb"))
    print("races", len(races), pd.Series([r["year"] for r in races]).value_counts().sort_index().to_dict())


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
    """race_id → bet_type → combo → 払戻（100 円あたり）。2022〜2025 は betting/payoff.parquet、2026 は betting_2026/payoff.parquet。"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for p in (ROOT / "data/local/modeling/betting/payoff.parquet", ROOT / "data/local/modeling/betting_2026/payoff.parquet"):
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["race_id", "bet_type", "combo", "payout"])
        d = d[d["bet_type"].isin(["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"])]
        for rid, bt, combo, pay in zip(d["race_id"].astype(str), d["bet_type"], d["combo"], d["payout"]):
            out.setdefault(rid, {}).setdefault(bt, {})[_norm_combo(bt, combo)] = float(pay)
    return out


def entries_for(r: dict, path: str) -> list[dict]:
    from src.utils.race_probabilities import place_effective_odds

    p = r["cal"] if (path == "cal" and r["cal"] is not None) else r["raw"]
    calibrated = path == "cal" and r["cal"] is not None
    from src.utils.race_probabilities import market_top2_fair_odds

    es = []
    fair = market_top2_fair_odds({int(h): float(o) for h, o in zip(r["hn"], r["win_odds"])})
    for i in range(r["n"]):
        pe = place_effective_odds(r["pmin"][i], r["pmax"][i]) if np.isfinite(r["pmin"][i]) else None
        pw, p2, p3 = float(p[0][i]), float(p[1][i]), float(p[2][i])
        evp = p3 * pe if pe else 0.0
        t2 = fair.get(int(r["hn"][i]))
        es.append({"horse_number": int(r["hn"][i]), "win_prob": pw, "top2_prob": p2, "top3_prob": p3, "win_odds": float(r["win_odds"][i]),
                   "ev_win": pw * float(r["win_odds"][i]), "ev_place": evp, "ev_top2": (p2 * t2) if t2 else None, "top2_odds_est": t2,
                   "place_odds_eff": pe, "expected_value": evp, "popularity": float(r["pop"][i]) if np.isfinite(r["pop"][i]) else None,
                   "prob_calibrated": calibrated, "composite_score": (p3 ** 0.55) * (evp ** 0.45) if evp > 0 and p3 > 0 else p3, "_i": i})
    es.sort(key=lambda x: x["composite_score"], reverse=True)
    return es


def mark_all(races: list[dict], path: str, rule) -> list[np.ndarray]:
    """レースごとの印（長さ n の文字列配列。馬番昇順）。rule(entries) が in-place で mark_type を付ける。"""
    out = []
    for r in races:
        es = entries_for(r, path)
        rule(es)
        m = np.array(["none"] * r["n"], dtype=object)
        for e in es:
            m[e["_i"]] = e["mark_type"]
        out.append(m)
    return out


# ───────────────────────── 集計 ─────────────────────────
def per_race_table(races: list[dict], marks: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """印ごとに、レースごとのベクトル（頭数・単勝/複勝の払戻・1着/2着内/3着内の数・当たったか）。"""
    tab = {k: {f: np.zeros(len(races)) for f in ("n", "ret_win", "ret_place", "h1", "h2", "h3", "any3", "star_odds_gt50", "star_odds_gt100")} for k in MARKS}
    tab["cover"] = {"top3": np.zeros(len(races)), "top3_marked": np.zeros(len(races))}
    max_ret = {k: [] for k in MARKS}
    star_odds = []
    for j, (r, m) in enumerate(zip(races, marks)):
        top3 = r["rank"] <= 3
        tab["cover"]["top3"][j] = top3.sum()
        tab["cover"]["top3_marked"][j] = (top3 & np.isin(m, MARKS)).sum()
        for k in MARKS:
            sel = m == k
            if not sel.any():
                continue
            t = tab[k]
            t["n"][j] = sel.sum()
            t["ret_win"][j] = np.nansum(r["ret_win"][sel])
            t["ret_place"][j] = np.nansum(r["ret_place"][sel]) if np.isfinite(r["ret_place"][sel]).any() else np.nan
            t["h1"][j] = (r["rank"][sel] == 1).sum()
            t["h2"][j] = (r["rank"][sel] <= 2).sum()
            t["h3"][j] = (r["rank"][sel] <= 3).sum()
            t["any3"][j] = float(t["h3"][j] > 0)
            if k == "star":
                t["star_odds_gt50"][j] = (r["win_odds"][sel] > 50).sum()
                t["star_odds_gt100"][j] = (r["win_odds"][sel] > 100).sum()
                star_odds.extend(r["win_odds"][sel].tolist())
    tab["_star_odds"] = {"v": np.array(star_odds)}
    return tab


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def summarize(races: list[dict], marks: list[np.ndarray], boot: bool = False, seed: int = 0) -> dict:
    tab = per_race_table(races, marks)
    nr = len(races)
    out: dict = {"races": nr}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nr, size=(BOOT, nr)) if boot and nr else None
    for k in MARKS:
        t = tab[k]
        n = t["n"]
        s = {"n_horses": int(n.sum()), "per_race": float(n.sum() / nr), "races_with": int((n > 0).sum()),
             "win": _ratio(t["h1"], n), "top2": _ratio(t["h2"], n), "top3": _ratio(t["h3"], n),
             "race_hit3": _ratio(t["any3"], (n > 0).astype(float)),
             "roi_win": _ratio(t["ret_win"], 100 * n) * 1.0, "roi_place": _ratio(np.nan_to_num(t["ret_place"]), 100 * n * np.isfinite(t["ret_place"]))}
        s["roi_win"] = _ratio(t["ret_win"], 100 * n)
        if boot and idx is not None:
            num_w, num_p, den = t["ret_win"], np.nan_to_num(t["ret_place"]), 100 * n
            okp = np.isfinite(t["ret_place"]) * den
            rw = num_w[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
            rp = num_p[idx].sum(1) / np.maximum(okp[idx].sum(1), 1e-9)
            h3 = t["h3"][idx].sum(1) / np.maximum(n[idx].sum(1), 1e-9)
            s["roi_win_ci"] = [float(np.percentile(rw, 2.5)), float(np.percentile(rw, 97.5))]
            s["roi_place_ci"] = [float(np.percentile(rp, 2.5)), float(np.percentile(rp, 97.5))]
            s["top3_ci"] = [float(np.percentile(h3, 2.5)), float(np.percentile(h3, 97.5))]
        if k == "star":
            s["odds_gt50_share"] = _ratio(t["star_odds_gt50"], n)
            s["odds_gt100_share"] = _ratio(t["star_odds_gt100"], n)
            so = tab["_star_odds"]["v"]
            s["odds_median"] = float(np.median(so)) if len(so) else float("nan")
        out[k] = s
    # 上位的中除外（単勝の払戻の大きい順に k 件を、賭け金ごと除く）
    for k in ("honmei", "star", "pair"):
        rets = []
        for r, m in zip(races, marks):
            sel = m == k
            rets.extend(r["ret_win"][sel].tolist())
        rets = np.sort(np.array(rets))[::-1]
        tot, cnt = rets.sum(), len(rets)
        out[k]["roi_win_ex_top3"] = float((tot - rets[:3].sum()) / (100 * (cnt - 3))) if cnt > 3 else float("nan")
        out[k]["roi_win_ex_top10"] = float((tot - rets[:10].sum()) / (100 * (cnt - 10))) if cnt > 10 else float("nan")
    cov = tab["cover"]
    out["coverage_top3"] = _ratio(cov["top3_marked"], cov["top3"])
    out["marks_per_race"] = float(sum(tab[k]["n"].sum() for k in MARKS) / nr)
    # ◎○の組
    both_q = both3 = 0
    pairs = 0
    for r, m in zip(races, marks):
        h, p = np.where(m == "honmei")[0], np.where(m == "pair")[0]
        if len(h) and len(p):
            pairs += 1
            both_q += int(r["rank"][h[0]] <= 2 and r["rank"][p[0]] <= 2)
            both3 += int(r["rank"][h[0]] <= 3 and r["rank"][p[0]] <= 3)
    out["honmei_pair"] = {"races": pairs, "both_top2": both_q / pairs if pairs else float("nan"), "both_top3": both3 / pairs if pairs else float("nan")}
    return out


def by_years(races: list[dict], years: tuple[int, ...]) -> list[dict]:
    return [r for r in races if r["year"] in years]


# ───────────────────────── 設計 ─────────────────────────
def v2_rule(params: dict):
    from src.utils.race_probabilities import assign_mece_marks_v2

    return lambda es: assign_mece_marks_v2(es, params)


def v1_rule(es: list[dict]) -> None:
    from src.utils.race_probabilities import assign_mece_marks_v1

    assign_mece_marks_v1(es)


def design(args) -> None:
    races = pickle.load(open(args.data, "rb"))
    dz, cf = by_years(races, DESIGN), by_years(races, CONFIRM)
    log: dict = {"design_races": len(dz), "confirm_races": len(cf)}
    res: dict = {}
    best: dict = {"honmei": {"win_mult": 1.0, "topk": 3}, "pair": {"rule": "win"}, "show_val": {"rule": "top3"},
                  "star": {"odds_min": 10.0, "odds_max": None, "top3_floor_mult": 1.0, "ev": "win", "ev_min": 0.0},
                  "anchor": {"limits": [[13, 3], [8, 3], [5, 2], [0, 1]], "min_top3_mult": 0.0}}
    for path in ("raw", "cal"):
        v1 = summarize(dz, mark_all(dz, path, v1_rule))
        log[f"v1_{path}_design"] = {k: v1[k] for k in ("honmei", "pair", "show_val", "star", "anchor")}
        # ◎
        rows = []
        for wm in (0.8, 1.0, 1.2, 1.5, 2.0):
            for tk in (2, 3, 4, 18):
                p = {**best, "honmei": {"win_mult": wm, "topk": tk}}
                s = summarize(dz, mark_all(dz, path, v2_rule(p)))["honmei"]
                rows.append({"win_mult": wm, "topk": tk, "win": s["win"], "top3": s["top3"], "roi_win": s["roi_win"], "roi_win_ex_top3": s["roi_win_ex_top3"]})
        res[f"{path}_honmei"] = rows
        # ○
        rows = []
        for rule in ("win", "top2", "blend"):
            p = {**best, "pair": {"rule": rule}}
            sm = summarize(dz, mark_all(dz, path, v2_rule(p)))
            rows.append({"rule": rule, "top2": sm["pair"]["top2"], "top3": sm["pair"]["top3"], "both_top2": sm["honmei_pair"]["both_top2"], "both_top3": sm["honmei_pair"]["both_top3"], "roi_place": sm["pair"]["roi_place"]})
        res[f"{path}_pair"] = rows
        # ★
        rows = []
        for om in (7.0, 10.0, 15.0, 20.0):
            for fm in (0.6, 0.8, 1.0, 1.2, 1.5):
                for ev in ("win", "place", "max"):
                    p = {**best, "star": {"odds_min": om, "odds_max": None, "top3_floor_mult": fm, "ev": ev, "ev_min": 0.0}}
                    sm = summarize(dz, mark_all(dz, path, v2_rule(p)))["star"]
                    rows.append({"odds_min": om, "top3_floor_mult": fm, "ev": ev, "per_race": sm["per_race"], "top3": sm["top3"], "roi_win": sm["roi_win"],
                                 "roi_place": sm["roi_place"], "roi_win_ex_top3": sm["roi_win_ex_top3"], "gt50": sm["odds_gt50_share"], "gt100": sm["odds_gt100_share"], "median_odds": sm["odds_median"]})
        res[f"{path}_star"] = rows
        # ▲
        rows = []
        for rule in ("top3", "composite"):
            p = {**best, "show_val": {"rule": rule}}
            s = summarize(dz, mark_all(dz, path, v2_rule(p)))["show_val"]
            rows.append({"rule": rule, "top3": s["top3"], "roi_place": s["roi_place"], "per_race": s["per_race"]})
        res[f"{path}_show_val"] = rows
        # △
        rows = []
        for name, lim in (("fixed2", [[0, 2]]), ("fixed3", [[0, 3]]), ("v1_like", [[13, 4], [8, 3], [5, 2], [0, 1]]), ("3-3-2-1", [[13, 3], [8, 3], [5, 2], [0, 1]]),
                          ("3-2-1", [[13, 3], [8, 2], [0, 1]]), ("2-3頭(12頭以上3)", [[12, 3], [0, 2]])):
            for mm in (0.0, 1.0):
                p = {**best, "anchor": {"limits": lim, "min_top3_mult": mm}}
                sm = summarize(dz, mark_all(dz, path, v2_rule(p)))
                rows.append({"limits": name, "min_top3_mult": mm, "per_race": sm["anchor"]["per_race"], "top3": sm["anchor"]["top3"], "roi_place": sm["anchor"]["roi_place"], "coverage_top3": sm["coverage_top3"]})
        res[f"{path}_anchor"] = rows
    json.dump({"log": log, "grid": res}, open(Path(args.out) / "design_grid.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("saved design_grid.json")


# ───────────────────────── 最終比較 ─────────────────────────
def final(args) -> None:
    races = pickle.load(open(args.data, "rb"))
    params = json.load(open(args.params))["v2"]
    out: dict = {"params": params}
    for path in ("raw", "cal"):
        for name, ys in (("design_2022-23", DESIGN), ("confirm_2024-25", CONFIRM)) + ((("final_2026", FINAL),) if args.with_2026 else ()):
            sub = by_years(races, ys)
            out[f"{path}_{name}"] = {"v1": summarize(sub, mark_all(sub, path, v1_rule), boot=True), "v2": summarize(sub, mark_all(sub, path, v2_rule(params)), boot=True)}
        for y in (2022, 2023, 2024, 2025):
            sub = by_years(races, (y,))
            a, b = summarize(sub, mark_all(sub, path, v1_rule)), summarize(sub, mark_all(sub, path, v2_rule(params)))
            out[f"{path}_year_{y}"] = {m: {"v1": {k: a[m][k] for k in ("win", "top3", "roi_win", "roi_place")}, "v2": {k: b[m][k] for k in ("win", "top3", "roi_win", "roi_place")}} for m in MARKS}
    json.dump(out, open(Path(args.out) / "final.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("saved final.json")


# ───────────────────────── C: 期待値追求型 ─────────────────────────
C_DEFAULT = {"honmei_win_mult": 0.8, "pair_top2_mult": 0.8, "star_odds_min": 10.0, "star_top3_mult": 0.8, "place_top3_mult": 1.0, "anchor_limits": [[13, 3], [8, 3], [5, 2], [0, 1]]}


def assign_marks_ev(es: list[dict], c: dict | None = None) -> None:
    """各印を『期待値の順』に選ぶ（最低限の確率条件つき）。◎=単勝EV、○=連対EV、★=人気薄の単勝EV、▲=複勝EV、△=複勝EVの次点。"""
    c = {**C_DEFAULT, **(c or {})}
    n = len(es)
    base = 1.0 / max(n, 1)
    for e in es:
        e["mark_type"] = "none"

    def f(e, k):
        v = e.get(k)
        return float(v) if v is not None else 0.0

    def free():
        return [e for e in es if e["mark_type"] == "none"]

    pool = [e for e in es if f(e, "win_prob") >= base * c["honmei_win_mult"]] or sorted(es, key=lambda x: -f(x, "win_prob"))[:1]
    max(pool, key=lambda x: (f(x, "ev_win"), f(x, "win_prob")))["mark_type"] = "honmei"
    pool = [e for e in free() if f(e, "top2_prob") >= base * 2.0 * c["pair_top2_mult"]]
    if pool:
        max(pool, key=lambda x: (f(x, "ev_top2"), f(x, "top2_prob")))["mark_type"] = "pair"
    pool = [e for e in free() if f(e, "win_odds") >= c["star_odds_min"] and f(e, "top3_prob") >= base * 3.0 * c["star_top3_mult"] and f(e, "ev_win") > 0]
    if pool:
        max(pool, key=lambda x: (f(x, "ev_win"), f(x, "top3_prob")))["mark_type"] = "star"
    pool = [e for e in free() if f(e, "top3_prob") >= base * 3.0 * c["place_top3_mult"] and f(e, "ev_place") > 0]
    if pool:
        max(pool, key=lambda x: (f(x, "ev_place"), f(x, "top3_prob")))["mark_type"] = "show_val"
    lim = next((int(k) for mn, k in c["anchor_limits"] if n >= int(mn)), 0)
    pool = sorted([e for e in free() if f(e, "top3_prob") >= base * 3.0 * c["place_top3_mult"] and f(e, "ev_place") > 0], key=lambda x: (f(x, "ev_place"), f(x, "top3_prob")), reverse=True)
    for e in pool[:lim]:
        e["mark_type"] = "anchor"


# ───────────────────────── 買い目（race_bet_suggestion）の精算 ─────────────────────────
BT = {"tansho": "tansho", "fukusho": "fukusho", "umaren": "umaren", "wide": "wide", "umatan": "umatan", "trio": "sanrenpuku", "tierce": "sanrentan"}


def settle(picks: list[dict], pay: dict) -> tuple[int, float, dict]:
    stake, ret, by = 0, 0.0, {}
    for p in picks:
        bt = BT.get(p.get("bet_type"))
        if not bt:
            continue
        horses = p.get("horses") or []
        combo = str(horses[0]) if bt in ("tansho", "fukusho") else _norm_combo(bt, "-".join(str(h) for h in horses))
        got = float((pay.get(bt) or {}).get(combo, 0.0))
        stake += 100
        ret += got
        b = by.setdefault(bt, [0, 0.0])
        b[0] += 100
        b[1] += got
    return stake, ret, by


def run_rule(races: list[dict], path: str, rule) -> dict:
    """ルールを適用 → 消印 → 較正後 EV → 買い目提案 → 精算。印の配列とレースごとの精算を返す。"""
    from src.utils.race_bet_suggestion import suggest_race_bets
    from src.utils.race_probabilities import assign_kesi_mark, attach_calibrated_ev, race_situation

    marks, stake, ret, situ = [], np.zeros(len(races)), np.zeros(len(races)), []
    by_type: dict[str, list[float]] = {}
    for j, r in enumerate(races):
        es = entries_for(r, path)
        rule(es)
        assign_kesi_mark(es)
        for e in es:
            attach_calibrated_ev(e)
        m = np.array(["none"] * r["n"], dtype=object)
        for e in es:
            m[e["_i"]] = e["mark_type"]
        marks.append(m)
        situ.append(race_situation(es))
        picks = (suggest_race_bets(es) or {}).get("picks") or []
        st, rt, by = settle(picks, r["pay"])
        stake[j], ret[j] = st, rt
        for bt, (a, b) in by.items():
            x = by_type.setdefault(bt, [0.0, 0.0])
            x[0] += a
            x[1] += b
    return {"marks": marks, "stake": stake, "ret": ret, "situation": situ, "by_type": by_type}


def roi_ci(stake: np.ndarray, ret: np.ndarray, seed: int = 1) -> dict:
    nr = len(stake)
    if nr == 0 or stake.sum() == 0:
        return {"roi": float("nan"), "ci": [float("nan")] * 2, "races_with_bets": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nr, size=(BOOT, nr))
    rr = ret[idx].sum(1) / np.maximum(stake[idx].sum(1), 1e-9)
    # 上位的中除外: レース単位の払戻が大きい順に 3 レース分（賭け金ごと）を除く
    order = np.argsort(-ret)[:3]
    keep = np.ones(nr, bool)
    keep[order] = False
    return {"roi": float(ret.sum() / stake.sum()), "ci": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
            "roi_ex_top3_races": float(ret[keep].sum() / max(stake[keep].sum(), 1e-9)), "races_with_bets": int((stake > 0).sum()),
            "bets_per_race": float(stake.sum() / 100 / nr), "stake": float(stake.sum())}


def paired_diff(sa: np.ndarray, ra: np.ndarray, sx: np.ndarray, rx: np.ndarray, seed: int = 2) -> dict:
    """A に対する X の回収率の差（X − A）。同じレースを再標本化した 95% 区間。"""
    nr = len(sa)
    if nr == 0 or sa.sum() == 0 or sx.sum() == 0:
        return {"diff": float("nan"), "ci": [float("nan")] * 2}
    idx = np.random.default_rng(seed).integers(0, nr, size=(BOOT, nr))
    d = rx[idx].sum(1) / np.maximum(sx[idx].sum(1), 1e-9) - ra[idx].sum(1) / np.maximum(sa[idx].sum(1), 1e-9)
    return {"diff": float(rx.sum() / sx.sum() - ra.sum() / sa.sum()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_better": float((d > 0).mean())}


def idx_of(races: list[dict], years: tuple[int, ...]) -> np.ndarray:
    return np.array([i for i, r in enumerate(races) if r["year"] in years], dtype=int)


def size_bucket(n: int) -> str:
    return "<=10頭" if n <= 10 else "11-14頭" if n <= 14 else ">=15頭"


def compare(args) -> None:
    races = pickle.load(open(args.data, "rb"))
    cfg = json.load(open(args.params))
    rules = {"A_v1": v1_rule, "B_user_v2": v2_rule(cfg["v2"]), "C_ev": (lambda es, c=cfg.get("c"): assign_marks_ev(es, c))}
    years_sets = {"design_2022-23": DESIGN, "confirm_2024-25": CONFIRM}
    if args.with_2026:
        years_sets["final_2026"] = FINAL
    out: dict = {"candidates": list(rules), "n_candidates_note": "A,B,C の 3 候補 + D（切替の割当 2 方式 × 状況ごとに 3 候補から選択）"}
    for path in ("raw", "cal"):
        runs = {name: run_rule(races, path, rule) for name, rule in rules.items()}
        res: dict = {}
        for name, rr in runs.items():
            res[name] = {}
            for yname, ys in years_sets.items():
                ix = idx_of(races, ys)
                sub = [races[i] for i in ix]
                summ = summarize(sub, [rr["marks"][i] for i in ix], boot=True)
                res[name][yname] = {"tickets": roi_ci(rr["stake"][ix], rr["ret"][ix]), "marks": {k: summ[k] for k in MARKS}, "coverage_top3": summ["coverage_top3"],
                                    "marks_per_race": summ["marks_per_race"], "honmei_pair": summ["honmei_pair"]}
            for y in (2022, 2023, 2024, 2025):
                ix = idx_of(races, (y,))
                res[name][f"year_{y}"] = roi_ci(runs[name]["stake"][ix], runs[name]["ret"][ix])
            for mth in range(1, 13):
                ix = np.array([i for i in idx_of(races, CONFIRM) if races[i]["month"] == mth], dtype=int)
                if len(ix):
                    res[name].setdefault("confirm_by_month", {})[str(mth)] = roi_ci(runs[name]["stake"][ix], runs[name]["ret"][ix])
            res[name]["by_bet_type"] = {bt: {"stake": v[0], "roi": v[1] / v[0] if v[0] else None} for bt, v in runs[name]["by_type"].items()}
        res["paired_vs_A"] = {}
        for nm in ("B_user_v2", "C_ev"):
            res["paired_vs_A"][nm] = {}
            for yname, ys in years_sets.items():
                ix = idx_of(races, ys)
                res["paired_vs_A"][nm][yname] = paired_diff(runs["A_v1"]["stake"][ix], runs["A_v1"]["ret"][ix], runs[nm]["stake"][ix], runs[nm]["ret"][ix])
        # D: 状況ごとに最良の規則を設計年で選び、確認年で再現するかを見る
        names = list(rules)
        design_ix = idx_of(races, DESIGN)
        for scheme in ("situation", "size"):
            key = [rr for rr in [runs["A_v1"]["situation"]]][0] if scheme == "situation" else [size_bucket(r["n"]) for r in races]
            groups = sorted(set(key))
            assign: dict[str, str] = {}
            detail = {}
            for g in groups:
                gi = np.array([i for i in design_ix if key[i] == g], dtype=int)
                scores = {}
                for nm in names:
                    st, rt = runs[nm]["stake"][gi], runs[nm]["ret"][gi]
                    ci = roi_ci(st, rt)
                    scores[nm] = ci["ci"][0] if st.sum() > 0 else -1
                best = max(scores, key=scores.get) if len(gi) >= 300 else "A_v1"
                assign[g] = best
                detail[g] = {"design_races": int(len(gi)), "ci_lower": {k: round(v, 4) for k, v in scores.items()}, "chosen": best}
            d_stake = np.array([runs[assign[key[i]]]["stake"][i] for i in range(len(races))])
            d_ret = np.array([runs[assign[key[i]]]["ret"][i] for i in range(len(races))])
            res[f"D_switch_by_{scheme}"] = {"assignment": detail}
            for yname, ys in years_sets.items():
                ix = idx_of(races, ys)
                res[f"D_switch_by_{scheme}"][yname] = {"tickets": roi_ci(d_stake[ix], d_ret[ix]),
                                                       "paired_vs_A": paired_diff(runs["A_v1"]["stake"][ix], runs["A_v1"]["ret"][ix], d_stake[ix], d_ret[ix])}
        out[path] = res
    json.dump(out, open(Path(args.out) / "compare.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("saved compare.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=build)
    d = sp.add_parser("design")
    d.add_argument("--data", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=design)
    f = sp.add_parser("final")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--params", required=True)
    f.add_argument("--with-2026", action="store_true")
    f.set_defaults(fn=final)
    c = sp.add_parser("compare")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--params", required=True)
    c.add_argument("--with-2026", action="store_true")
    c.set_defaults(fn=compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
