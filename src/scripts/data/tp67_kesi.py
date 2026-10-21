"""tp-67 消印（上位5人気のうち好走率がもっとも低そうな1頭）の定義の比較。

python -m src.scripts.data.tp67_kesi run --data <frame_all.parquet> --out <出力ディレクトリ>   # 2022〜2025 年（設計=2022〜23・確認=2024・参考=2025）
python -m src.scripts.data.tp67_kesi robust --data ... --out ...     # 人気順をオッズの揺れで作り直した頑健性
python -m src.scripts.data.tp67_kesi final2026 --data ... --out ... --rule <run が選んだ定義>     # 2026 年を一度だけ

入力: tp-36 の frame_all.parquet（V4 の OOF 予測 p_win/p_top2/p_top3 と確定の単勝・複勝オッズ・人気・払戻 ret_win/ret_place）。
印は本番の ``assign_mece_marks``。市場オッズ・人気は消印の選び方と精算にだけ使う（特徴量にはしない）。
好走＝複勝の払戻が付いた（ret_place>0）。8 頭未満の複勝は 2 着以内だが、この表ではそのまま「複勝圏」として数える。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.race_probabilities import assign_mece_marks, harville_top3_prob

DESIGN_YEARS = (2022, 2023)
CONFIRM_YEAR = 2024
REF_YEAR = 2025
MIN_FIELD = 8
TOP_POP = 5
MAIN_MARKS = {"honmei", "pair", "show_val", "star"}
BOOT = 2000

# 定義名 → 説明
RULES = {
    "A_min_top3": "複勝確率が最小",
    "B_min_win": "勝率が最小",
    "C1_mkt_gap": "市場の複勝見込み（単勝オッズの Harville）−モデルの複勝確率が最大",
    "C2_ratio": "モデルの複勝確率 ÷ 市場の複勝見込みが最小",
    "D1_min_ev_place": "複勝期待値（複勝確率×複勝オッズ）が最小",
    "D2_min_ev_win": "単勝期待値（勝率×単勝オッズ）が最小",
    "E_rank_avg": "A と C2 の順位平均が最小",
    "F_min_top3_pop": "複勝確率が最小（人気 4・5 位に限る）",
    "Z_pop5": "（比較用）5 番人気の馬",
}
POLICIES = {
    "P_main": "主印（◎○▲★）の付いた馬は消印にしない（△は消印に上書き可）",
    "P_any": "印（△を含む）が付いた馬は消印にしない",
    "P_none": "印に関係なく選ぶ（主印の馬も候補）",
}


def load(path: str, years: tuple[int, ...]) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[d["year"].isin(years)].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce").astype(int)
    return d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def _entries(g: pd.DataFrame) -> list[dict]:
    pw, p2, p3 = (g[c].to_numpy(float) for c in ("p_win", "p_top2", "p_top3"))
    odds = g["win_odds"].to_numpy(float)
    pm = g["place_mid"].to_numpy(float)
    out = []
    for i, hn in enumerate(g["horse_number"].to_numpy()):
        e = {"horse_number": int(hn), "win_prob": float(pw[i]), "top2_prob": float(p2[i]), "top3_prob": float(p3[i])}
        if np.isfinite(odds[i]) and odds[i] > 0:
            e["win_odds"] = float(odds[i])
            e["ev_win"] = float(pw[i] * odds[i])
            if np.isfinite(pm[i]):
                e["ev_place"] = float(p3[i] * pm[i])
        out.append(e)
    return out


def race_rows(g: pd.DataFrame, rng: np.random.Generator | None = None, sigma: float = 0.0) -> pd.DataFrame | None:
    """1 レース分の候補表（上位 5 人気）。8 頭未満・オッズ欠損のレースは None。sigma>0 なら人気順だけを、オッズに対数正規の揺れを足して決める（予測オッズで人気を作る場合の頑健性の確認用）。"""
    if len(g) < MIN_FIELD or not np.isfinite(g["win_odds"]).all() or (g["win_odds"] <= 0).any():
        return None
    es = _entries(g)
    assign_mece_marks(es)
    g = g.copy()
    g["mark"] = [e["mark_type"] for e in es]
    q = (1.0 / g["win_odds"].to_numpy(float))
    q = q / q.sum()
    g["mkt_top3"] = harville_top3_prob(q)
    if sigma > 0 and rng is not None:
        noisy = g["win_odds"].to_numpy(float) * np.exp(rng.normal(0.0, sigma, len(g)))
        pop = pd.Series(noisy).rank(method="first").to_numpy(float)
    else:
        pop = g["popularity"].to_numpy(float)
    key = pd.DataFrame({"pop": np.where(np.isfinite(pop), pop, 99.0), "odds": g["win_odds"].to_numpy(float), "hn": g["horse_number"].to_numpy()})
    order = key.sort_values(["pop", "odds", "hn"]).index.to_numpy()
    top = g.iloc[order[:TOP_POP]].copy()
    top["pop_rank"] = np.arange(1, len(top) + 1)
    top["hit3"] = (top["ret_place"].to_numpy(float) > 0).astype(float)
    top["won"] = top["win"].to_numpy(float)
    top["ev_place_m"] = top["p_top3"] * top["place_mid"]
    top["ev_win_m"] = top["p_win"] * top["win_odds"]
    top["gap"] = top["mkt_top3"] - top["p_top3"]
    top["ratio"] = top["p_top3"] / top["mkt_top3"].clip(lower=1e-6)
    top["rank_a"] = top["p_top3"].rank(method="first")
    top["rank_c"] = top["ratio"].rank(method="first")
    top["e_score"] = top["rank_a"] + top["rank_c"]
    return top


def pick(top: pd.DataFrame, rule: str, policy: str) -> int | None:
    t = top
    if policy == "P_main":
        t = t[~t["mark"].isin(MAIN_MARKS)]
    elif policy == "P_any":
        t = t[t["mark"] == "none"]
    if rule == "F_min_top3_pop":
        t = t[t["pop_rank"] >= 4]
    if len(t) == 0:
        return None
    if rule in ("A_min_top3", "F_min_top3_pop"):
        return int(t.sort_values(["p_top3", "pop_rank"], ascending=[True, False]).index[0])
    if rule == "Z_pop5":
        return int(t.sort_values("pop_rank", ascending=False).index[0])
    if rule == "B_min_win":
        return int(t.sort_values(["p_win", "pop_rank"], ascending=[True, False]).index[0])
    if rule == "C1_mkt_gap":
        return int(t.sort_values(["gap", "pop_rank"], ascending=[False, False]).index[0])
    if rule == "C2_ratio":
        return int(t.sort_values(["ratio", "pop_rank"], ascending=[True, False]).index[0])
    if rule == "D1_min_ev_place":
        return int(t.sort_values(["ev_place_m", "pop_rank"], ascending=[True, False]).index[0])
    if rule == "D2_min_ev_win":
        return int(t.sort_values(["ev_win_m", "pop_rank"], ascending=[True, False]).index[0])
    if rule == "E_rank_avg":
        return int(t.sort_values(["e_score", "pop_rank"], ascending=[True, False]).index[0])
    raise ValueError(rule)


def build_races(df: pd.DataFrame, sigma: float = 0.0, seed: int = 0) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    for rid, g in df.groupby("race_id", sort=False):
        t = race_rows(g.reset_index(drop=True), rng, sigma)
        if t is not None and len(t) == TOP_POP:
            t = t.reset_index(drop=True)
            t["year"] = int(g["year"].iloc[0])
            out[rid] = t
    return out


def evaluate(races: dict[str, pd.DataFrame], rule: str, policy: str) -> pd.DataFrame:
    """1 レース 1 行: 消印馬の成績と、同レースの上位 5 人気の他の 4 頭の平均。"""
    rows = []
    for rid, t in races.items():
        i = pick(t, rule, policy)
        if i is None:
            continue
        k = t.loc[i]
        oth = t.drop(index=i)
        rows.append({
            "race_id": rid, "year": int(k["year"]), "pop_rank": int(k["pop_rank"]), "mark": k["mark"],
            "k_hit3": float(k["hit3"]), "k_win": float(k["won"]), "k_fuku": float(k["ret_place"]) / 100.0, "k_tan": float(k["ret_win"]) / 100.0,
            "o_hit3": float(oth["hit3"].mean()), "o_win": float(oth["won"].mean()),
            "o_fuku": float(oth["ret_place"].mean()) / 100.0, "o_tan": float(oth["ret_win"].mean()) / 100.0,
            "t_hit3": float(t["hit3"].mean()), "t_win": float(t["won"].mean()),
        })
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, n: int = BOOT, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(ev: pd.DataFrame) -> dict:
    if len(ev) == 0:
        return {"n": 0}
    d_hit = (ev["k_hit3"] - ev["o_hit3"]).to_numpy()
    d_win = (ev["k_win"] - ev["o_win"]).to_numpy()
    r = {"n": int(len(ev)), "kesi_hit3": float(ev["k_hit3"].mean()), "others_hit3": float(ev["o_hit3"].mean()), "top5_hit3": float(ev["t_hit3"].mean()),
         "kesi_win": float(ev["k_win"].mean()), "others_win": float(ev["o_win"].mean()),
         "kesi_fuku_roi": float(ev["k_fuku"].mean()), "others_fuku_roi": float(ev["o_fuku"].mean()),
         "kesi_tan_roi": float(ev["k_tan"].mean()), "others_tan_roi": float(ev["o_tan"].mean()),
         "main_mark_rate": float((ev["mark"].isin(MAIN_MARKS)).mean())}
    for name, d in (("d_hit3", d_hit), ("d_win", d_win)):
        m, lo, hi = boot_ci(d)
        r[name] = m
        r[name + "_lo"], r[name + "_hi"] = lo, hi
    return r


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.data, (*DESIGN_YEARS, CONFIRM_YEAR, REF_YEAR))
    races = build_races(df)
    print(f"対象 {len(races)} レース（8 頭以上・単勝オッズあり）", flush=True)
    res: dict = {"n_races": {int(y): int(sum(1 for t in races.values() if t['year'].iloc[0] == y)) for y in sorted(df['year'].unique())}, "rules": RULES, "policies": POLICIES, "table": {}}
    for policy in POLICIES:
        for rule in RULES:
            ev = evaluate(races, rule, policy)
            row = {"design": summarize(ev[ev["year"].isin(DESIGN_YEARS)]), "confirm": summarize(ev[ev["year"] == CONFIRM_YEAR]), "ref2025": summarize(ev[ev["year"] == REF_YEAR])}
            res["table"][f"{policy}|{rule}"] = row
            d, c = row["design"], row["confirm"]
            print(f"{policy:7s} {rule:16s} 設計 n={d.get('n')} 消印3着内{d.get('kesi_hit3', float('nan')):.3f} 他4頭{d.get('others_hit3', float('nan')):.3f} 差{d.get('d_hit3', float('nan')):+.3f}"
                  f" | 確認 差{c.get('d_hit3', float('nan')):+.3f}[{c.get('d_hit3_lo', float('nan')):+.3f},{c.get('d_hit3_hi', float('nan')):+.3f}]", flush=True)
    (out / "tp67_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def robust(args) -> None:
    """人気順をオッズの揺れで作り直したときの A_min_top3（P_main）の効果。"""
    out = Path(args.out)
    df = load(args.data, (CONFIRM_YEAR, REF_YEAR))
    res = {}
    for sigma in (0.0, 0.15, 0.30, 0.50):
        races = build_races(df, sigma, seed=1)
        ev = evaluate(races, "A_min_top3", "P_main")
        res[str(sigma)] = summarize(ev)
        s = res[str(sigma)]
        print(f"sigma={sigma}: n={s['n']} 消印3着内{s['kesi_hit3']:.3f} 他4頭{s['others_hit3']:.3f} 差{s['d_hit3']:+.3f}[{s['d_hit3_lo']:+.3f},{s['d_hit3_hi']:+.3f}]", flush=True)
    (out / "tp67_robust.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def final2026(args) -> None:
    out = Path(args.out)
    policy, rule = args.policy, args.rule
    df = load(args.data, (2026,))
    if args.rank:
        # 2026 年の frame には複勝の払戻が無いので、確定着順（rank_tbl）から複勝圏（3 着以内）を作る。複勝回収率は出さない
        rk = pd.read_parquet(args.rank)
        rk = rk.assign(race_id=rk["race_id"].astype(str), horse_id=rk["horse_id"].astype(str))[["race_id", "horse_id", "rank"]]
        df = df.assign(horse_id=df["horse_id"].astype(str)).merge(rk, on=["race_id", "horse_id"], how="left")
        df["ret_place"] = np.where(df["rank"].to_numpy(float) <= 3, 100.0, 0.0)
    races = build_races(df)
    ev = evaluate(races, rule, policy)
    s = summarize(ev)
    s["n_races"] = len(races)
    (out / "tp67_final2026.json").write_text(json.dumps({"policy": policy, "rule": rule, "result": s}, ensure_ascii=False, indent=1))
    print(json.dumps(s, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    c = sub.add_parser("robust")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    b = sub.add_parser("final2026")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--rule", required=True)
    b.add_argument("--policy", default="P_main")
    b.add_argument("--rank", default=None, help="rank_tbl/2026/rank.parquet（race_id, horse_id, rank）。2026 年の複勝圏の判定に使う")
    args = ap.parse_args()
    {"run": run, "final2026": final2026, "robust": robust}[args.cmd](args)


if __name__ == "__main__":
    main()
