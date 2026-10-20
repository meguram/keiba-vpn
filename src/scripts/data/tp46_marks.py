"""tp-46 印（◎○▲★△）の付け方をレースの状況で切り替えるかの評価。

python -m src.scripts.data.tp46_marks run            # 2022〜2025 年の現行ルールと状況別ルールを比較（軽い。設計=2022〜23・確認=2024・参考=2025）
python -m src.scripts.data.tp46_marks final2026      # 2026 年を一度だけ（run が出した採否の規則で）

入力: tp-36 の frame_all.parquet（V4 の予測 p_win/p_top2/p_top3 と確定の単勝・複勝オッズ）と payoff.parquet（確定払戻）。環境変数 TP46_DATA に
その置き場（tp36/frame_all.parquet と payoff.parquet があるディレクトリ）を指定できる。評価する印は本番の ``assign_mece_marks`` そのもの。
期待値は確定オッズから計算した値（本番では確定前は予測オッズ。その揺れは安定性の評価で別に見る）。市場オッズは印の選び方と精算にだけ使う。
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp43_marks as tm
from src.utils.race_probabilities import assign_mece_marks, race_situation

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP46_OUT") or ROOT / "data/local/modeling/betting/tp46")
DATA = Path(os.environ.get("TP46_DATA") or ROOT / "data/local/modeling/betting")
DESIGN_YEARS = (2022, 2023)
CONFIRM_YEAR = 2024
REF_YEAR = 2025
MIN_HORSES = 6
NO_VALUE = {"value_ev": 99.0}
NO_DOM = {"dom_mult": 99.0}

# 評価する方策: (situational, cfg)
POLICIES: dict[str, tuple[bool, dict]] = {
    "P0_current": (False, {}),
    "P1_switch": (True, {}),
    "P1a_close_only": (True, {**NO_VALUE, **NO_DOM}),
    "P1b_value_only": (True, {"close_ratio": 0.0}),
    "P1c_close_1.10": (True, {"close_ratio": 1.10, **NO_VALUE, **NO_DOM}),
    "P1d_close_1.25": (True, {"close_ratio": 1.25, **NO_VALUE, **NO_DOM}),
    "P1e_value_1.30": (True, {"value_ev": 1.30, "close_ratio": 0.0}),
    "P1f_value_1.05": (True, {"value_ev": 1.05, "close_ratio": 0.0}),
    "P2_maybe_composite": (False, {"maybe_by_composite": 1.0}),
}
METRICS = ["honmei_win", "honmei_top3", "top2cov", "umaren_hit", "cover3", "all3", "win_cover", "n_marks", "n_maybe",
           "roi_tan", "roi_fuku", "roi_umaren", "roi_wide", "roi_trio_box", "roi_trio_nagashi"]


def load(years: tuple[int, ...]) -> pd.DataFrame:
    tm.TP = DATA / "tp36"
    tm.BET = DATA
    return tm.load_frame(years)


def entries_of(g: pd.DataFrame, odds_noise: np.ndarray | None = None, no_odds: bool = False) -> list[dict]:
    pw, p2, p3 = (g[c].to_numpy(float) for c in ("p_win", "p_top2", "p_top3"))
    odds = g["win_odds"].to_numpy(float)
    if odds_noise is not None:
        odds = odds * odds_noise
    pm = g["place_mid"].to_numpy(float) if "place_mid" in g else np.full(len(g), np.nan)
    out = []
    for i, hn in enumerate(g["horse_number"].to_numpy()):
        e = {"horse_number": int(hn), "win_prob": float(pw[i]), "top2_prob": float(p2[i]), "top3_prob": float(p3[i])}
        if not no_odds and np.isfinite(odds[i]) and odds[i] > 0:
            e["win_odds"] = float(odds[i])
            e["ev_win"] = float(pw[i] * odds[i])
            if np.isfinite(pm[i]):
                e["ev_place"] = float(p3[i] * pm[i])
        out.append(e)
    return out


def marks_of(entries: list[dict], pol: str) -> dict[int, str]:
    sit, cfg = POLICIES[pol]
    es = [dict(e) for e in entries]
    assign_mece_marks(es, situational=sit, cfg=cfg)
    return {e["horse_number"]: e["mark_type"] for e in es}


def race_metrics(mk: dict[int, str], pay: dict) -> dict[str, float]:
    win_set = {k for k in pay.get("tansho", {})}
    top2 = next(iter(pay.get("umaren", {})), frozenset())
    top3 = next(iter(pay.get("sanrenpuku", {})), frozenset())
    honmei = next((h for h, m in mk.items() if m == "honmei"), None)
    pair = next((h for h, m in mk.items() if m == "pair"), None)
    maybes = [h for h, m in mk.items() if m == "anchor"]
    marked = {h for h, m in mk.items() if m != "none"}
    nan = float("nan")
    r = {k: nan for k in METRICS}
    r["n_marks"], r["n_maybe"] = float(len(marked)), float(len(maybes))
    r["cover3"] = len(marked & set(top3)) / 3 if top3 else nan
    r["all3"] = float(bool(top3) and set(top3) <= marked)
    r["win_cover"] = float(bool(win_set) and bool(win_set & marked))
    if honmei is None:
        return r
    r["honmei_win"] = float(honmei in win_set)
    r["honmei_top3"] = float(honmei in top3)
    got = [x for x in (honmei, pair) if x is not None]
    r["top2cov"] = len(set(got) & set(top2)) / 2 if top2 else nan
    r["umaren_hit"] = float(pair is not None and frozenset((honmei, pair)) == top2)
    r["roi_tan"] = pay.get("tansho", {}).get(honmei, 0.0) / 100.0
    r["roi_fuku"] = pay.get("fukusho", {}).get(honmei, 0.0) / 100.0
    if pair is not None:
        r["roi_umaren"] = pay.get("umaren", {}).get(frozenset((honmei, pair)), 0.0) / 100.0
        r["roi_wide"] = pay.get("wide", {}).get(frozenset((honmei, pair)), 0.0) / 100.0
        if maybes:
            keys = [frozenset((honmei, pair, a)) for a in maybes]
            r["roi_trio_nagashi"] = sum(pay.get("sanrenpuku", {}).get(k, 0.0) for k in keys) / 100.0 / len(keys)
    box = [honmei] + maybes
    if len(box) >= 3:
        keys = [frozenset(c) for c in combinations(box, 3)]
        r["roi_trio_box"] = sum(pay.get("sanrenpuku", {}).get(k, 0.0) for k in keys) / 100.0 / len(keys)
    return r


def evaluate(frame: pd.DataFrame, pays: dict, bad: set[str], policies=None, seed: int = 0) -> pd.DataFrame:
    policies = policies or list(POLICIES)
    rng = np.random.default_rng(seed)
    rows = []
    for rid, g in frame.groupby("race_id", sort=False):
        if rid in bad or rid not in pays or len(g) < MIN_HORSES:
            continue
        pay = pays[rid]
        if not (pay.get("tansho") and pay.get("umaren") and pay.get("sanrenpuku")):
            continue
        es = entries_of(g)
        sit = race_situation(es)
        # 安定性: T-60 → T-15 のオッズ変動を模した対数正規ノイズ（σ=0.07）を2回かけて印を付け、印が変わる程度を見る
        n1 = np.exp(rng.normal(0, 0.07, len(g)))
        n2 = np.exp(rng.normal(0, 0.07, len(g)))
        es1, es2 = entries_of(g, n1), entries_of(g, n2)
        es0 = entries_of(g, no_odds=True)
        for pol in policies:
            mk = marks_of(es, pol)
            row = {"race_id": rid, "year": int(g["year"].iloc[0]), "policy": pol, "situation": sit, "n": len(g), **race_metrics(mk, pay)}
            m1, m2 = marks_of(es1, pol), marks_of(es2, pol)
            h1 = next((h for h, m in m1.items() if m == "honmei"), None)
            h2 = next((h for h, m in m2.items() if m == "honmei"), None)
            row["honmei_flip"] = float(h1 != h2)
            row["marks_changed"] = float(sum(1 for h in m1 if m1[h] != m2[h]))
            m0 = marks_of(es0, pol)
            row["noodds_honmei_same"] = float(next((h for h, m in m0.items() if m == "honmei"), None) == next((h for h, m in mk.items() if m == "honmei"), None))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """a−b の平均と95%区間（レース単位の対ブートストラップ）。欠損は両方有効な行だけ。"""
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df: pd.DataFrame, years: tuple[int, ...]) -> dict:
    d = df[df["year"].isin(years)]
    base = d[d["policy"] == "P0_current"].set_index("race_id")
    out: dict = {}
    for pol, g in d.groupby("policy"):
        g = g.set_index("race_id")
        row = {"races": int(len(g))}
        for m in METRICS + ["honmei_flip", "marks_changed", "noodds_honmei_same"]:
            row[m] = float(np.nanmean(g[m])) if g[m].notna().any() else None
        if pol != "P0_current":
            idx = g.index.intersection(base.index)
            for m in ("honmei_win", "honmei_top3", "top2cov", "umaren_hit", "cover3", "win_cover", "roi_tan", "roi_fuku", "roi_umaren", "roi_wide", "roi_trio_box", "roi_trio_nagashi"):
                row[f"d_{m}"] = paired_boot(g.loc[idx, m].to_numpy(float), base.loc[idx, m].to_numpy(float))
            row["changed_honmei_share"] = float((g.loc[idx, "honmei_win"].notna() & (g.loc[idx, "honmei_win"] != base.loc[idx, "honmei_win"])).mean())
        out[pol] = row
    return out


def situation_shares(df: pd.DataFrame) -> dict:
    d = df[df["policy"] == "P0_current"]
    return {str(y): g["situation"].value_counts(normalize=True).round(3).to_dict() for y, g in d.groupby("year")}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load((2022, 2023, 2024, 2025))
    pays, bad = tm.load_payoffs(set(frame["race_id"]))
    df = evaluate(frame, pays, bad)
    df.to_parquet(OUT / "race_metrics.parquet", index=False)
    res = {"races": int(df["race_id"].nunique()), "policies": list(POLICIES), "situation_shares": situation_shares(df),
           "design_2022_23": summarize(df, DESIGN_YEARS), "confirm_2024": summarize(df, (CONFIRM_YEAR,)), "ref_2025": summarize(df, (REF_YEAR,))}
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT / "result.json", "races", res["races"])


def final2026(rule: str) -> None:
    from src.scraper.payoff_repair import load_fix
    from src.scripts.data.tp36_final2026 import parse_payoff_json

    frame = load((2026,))
    pj = pd.read_parquet(ROOT / "data/page_reference/tables/2026/race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
    fix = load_fix(2026)
    ids = set(frame["race_id"])
    pays, bad = {}, set()
    for r in pj.itertuples(index=False):
        rid = str(r.race_id)
        if rid not in ids:
            continue
        p, broken = tm.payoffs_from_json(parse_payoff_json(r.payoff, fix.get(rid)))
        if broken:
            bad.add(rid)
        pays[rid] = p
    df = evaluate(frame, pays, bad, policies=["P0_current", rule])
    res = {"races": int(df["race_id"].nunique()), "rule": rule, "excluded_broken_payoff": len(bad), "summary": summarize(df, (2026,))}
    (OUT / "final2026.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT / "final2026.json", "races", res["races"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "final2026"])
    ap.add_argument("--rule", default="P1_switch")
    a = ap.parse_args()
    run() if a.cmd == "run" else final2026(a.rule)


if __name__ == "__main__":
    main()
