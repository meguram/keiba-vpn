"""段 1 (b): 組の市場どうしの比べと、AI の組の期待値の分布（着順・払戻は読まない）。

    KEIBA_SESSION=pair-bets nice -n 19 python -m src.research.pair_bets.market

- 年: 2022〜24（R0-15 決定 4・R0-17 決定 3 の組の符号の確かめと同じ年）。c13 と同じ平場（重賞の cls 6・7・8 を除く。頭数 5 以上）。
- 市場の組の確率 m: 馬連は確定の odds の逆数を、馬を突き合わせた後にレースの中で和 1 に直した値。
  ワイドは odds_min・odds_max の中点の逆数を、レースの中で和 3 に直した値（ワイドは 1 レースに 3 組が当たる）。
- 単勝からの組の確率: 確定の q から Harville・LBS（place_w.ordered_probs）。
- AI の組の確率: c13 の凍結の組み合わせ p ∝ p_rich^a · q^b（a, b は config/megu_cycles/candidates/c13_flat.json の combo_ab_t60）。
  ここでは q を確定の値で置く（判定のオッズを確定にした楽観の上限）。組は同じ Harville・LBS で伸ばす。
- 出すもの（どれも着順なし）:
  1. 組の人気の帯 × 型ごとの log(m / 単勝からの組) の平均（組の票が単勝の票に比べて、どの帯を高く・安く付けるか）。
  2. AI の組の期待値（AI の組の確率 × 組のオッズ。ワイドは下限と中点）の線ごとの組の数・1 レースあたり・その組の AI ÷ m の比。
  3. 単勝の期待値（同じ p × 確定の単勝オッズ）の線ごとの数と比（比べの基準）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council.place_w import LBS, ordered_probs

ROOT = Path(__file__).resolve().parents[3]
TABLES = Path(os.environ.get("PAIR_TABLES") or "/home/jovyan/work/keiba-vpn/data/page_reference/tables")
RUNNERS = Path("/home/jovyan/work/megu_cycles_out/runners.parquet")
RICH = Path("/home/jovyan/work/megu_cycles_out/rich_pred_v5bag.parquet")
C13 = ROOT / "config/megu_cycles/candidates/c13_flat.json"
OUT = ROOT / "docs/operations/pair_bets/market.json"
YEARS = (2022, 2023, 2024)
MODELS = {"harville": (1.0, 1.0, 1.0), "lbs": LBS}
EV_LINES = (1.0, 1.2, 1.3, 1.5, 2.0)
GRADED_CLS = (6, 7, 8)
PAIR_POP_BANDS = [(1, 3), (4, 10), (11, 30), (31, 80), (81, 999)]


def horses() -> pd.DataFrame:
    d = pd.read_parquet(RUNNERS, columns=["race_id", "horse_id", "year", "hn", "odds_final", "field", "cls"],
                        filters=[("year", ">=", YEARS[0]), ("year", "<=", YEARS[-1])])
    assert int(d["year"].max()) <= 2024
    assert not set(d.columns) & {"fin", "y1", "y3", "ret_tan", "ret_fuku"}, "着順・払戻の列は読まない"
    r = pd.read_parquet(RICH, columns=["race_id", "horse_id", "pr_win"], filters=[("year", ">=", YEARS[0]), ("year", "<=", YEARS[-1])])
    r = r.drop_duplicates(["race_id", "horse_id"])
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d = d.merge(r, on=["race_id", "horse_id"], how="left")
    d = d[(d["odds_final"] > 0) & ~pd.to_numeric(d["cls"], errors="coerce").isin(GRADED_CLS)].copy()     # c13 の segment flat（mc_eval.seg_mask）
    a, b = json.loads(C13.read_text())["combo_ab_t60"]
    inv = 1 / d["odds_final"]
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    pr = d["pr_win"].fillna(1e-4).clip(1e-6, 1)
    d["pr"] = pr / pr.groupby(d["race_id"]).transform("sum")
    lp = a * np.log(d["pr"]) + b * np.log(d["q"])
    e = np.exp(lp - lp.groupby(d["race_id"]).transform("max"))
    d["p"] = e / e.groupby(d["race_id"]).transform("sum")
    d["hn"] = pd.to_numeric(d["hn"], errors="coerce")
    return d.sort_values(["race_id", "hn"]).reset_index(drop=True)


def pairs(ty: str) -> pd.DataFrame:
    cols = ["race_id", "odds_type", "pair", "odds", "odds_min", "odds_max", "popularity"]
    F = pd.concat([pd.read_parquet(TABLES / str(y) / "race_pair_odds_flat.parquet", columns=cols, filters=[("odds_type", "==", ty)]) for y in YEARS],
                  ignore_index=True)
    F["race_id"] = F["race_id"].astype(str)
    pr = F["pair"].astype(str).str.strip("[]").str.split(",", expand=True)
    F["a"], F["b"] = pd.to_numeric(pr[0], errors="coerce"), pd.to_numeric(pr[1], errors="coerce")
    if ty == "wide":
        F["o"] = pd.to_numeric(F["odds_min"], errors="coerce")
        F["o_mid"] = (F["o"] + pd.to_numeric(F["odds_max"], errors="coerce")) / 2
    else:
        F["o"] = pd.to_numeric(F["odds"], errors="coerce")
        F["o_mid"] = F["o"]
    return F[(F["o"] > 0)][["race_id", "a", "b", "o", "o_mid", "popularity"]]


def pair_probs(p: np.ndarray, lam) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        P3 = np.nan_to_num(ordered_probs(p, 3, lam))
    P2 = P3.sum(axis=2)
    w = P3.sum(axis=2) + P3.sum(axis=1) + P3.sum(axis=0)
    return P2 + P2.T, w + w.T


def main() -> None:
    from src.ops import data_ledger as DL
    with DL.guard(tool="src/research/pair_bets/market.py", family="rules_council", version="pair_bets_stage1b", period=("2022-01-01", "2024-12-31"),
                   purpose="select", reads=("確定オッズ", "AI の予測"),
                   reason="組の市場と単勝からの組の確率・AI の組の期待値の分布。着順・払戻は読まない（critic の後に平場へ絞って流し直し）") as tok:
        _run(tok)


def _run(tok) -> None:
    H = horses()
    res = {"years": list(YEARS), "models": list(MODELS), "ev_lines": list(EV_LINES), "pair_pop_bands": PAIR_POP_BANDS}
    # 単勝の基準（着順なし）
    ev_w = H["p"] * H["odds_final"]
    res["win"] = {str(t): {"n": int((ev_w >= t).sum()), "per_race": round(float((ev_w >= t).sum() / H["race_id"].nunique()), 3),
                           "p_over_q": round(float(H.loc[ev_w >= t, "p"].sum() / H.loc[ev_w >= t, "q"].sum()), 4)} for t in EV_LINES}
    c13 = (ev_w >= 1.3) & (H["odds_final"] <= 50)                      # c13 の線（t 1.3・omax 50）。オッズは確定で置いた
    res["win_c13_line"] = {"n": int(c13.sum()), "p_over_q": round(float(H.loc[c13, "p"].sum() / H.loc[c13, "q"].sum()), 4)}
    hg = {rid: g for rid, g in H.groupby("race_id", sort=False)}
    for ty in ("umaren", "wide"):
        F = pairs(ty)
        F = F[F["race_id"].isin(hg.keys())]
        rows = []
        for rid, g in F.groupby("race_id", sort=False):
            h = hg[rid]
            if len(h) < 5:
                continue
            pos = {int(x): i for i, x in enumerate(h["hn"].to_numpy())}
            ia = g["a"].map(pos); ib = g["b"].map(pos)
            ok = ia.notna() & ib.notna()
            if ok.sum() < len(g) * 0.9:
                continue
            g = g[ok]; ia = ia[ok].astype(int).to_numpy(); ib = ib[ok].astype(int).to_numpy()
            q = h["q"].to_numpy(float); p = h["p"].to_numpy(float)
            inv = 1 / g["o_mid"].to_numpy(float)
            m = inv / inv.sum() * (3.0 if ty == "wide" else 1.0)
            out = {"race_id": rid, "n": len(h), "o": g["o"].to_numpy(), "o_mid": g["o_mid"].to_numpy(), "pop": g["popularity"].to_numpy(), "m": m}
            for name, lam in MODELS.items():
                uq, wq = pair_probs(q, lam)
                up, wp = pair_probs(p, lam)
                Q, P = (uq, up) if ty == "umaren" else (wq, wp)
                out[f"Hq_{name}"] = Q[ia, ib]
                out[f"Hp_{name}"] = P[ia, ib]
            rows.append(pd.DataFrame(out))
        R = pd.concat(rows, ignore_index=True)
        nr = R["race_id"].nunique()
        band = np.full(len(R), -1)
        for i, (lo, hi) in enumerate(PAIR_POP_BANDS):
            band[(R["pop"] >= lo) & (R["pop"] <= hi)] = i
        R["band"] = band
        rt = {"races": int(nr), "pairs": int(len(R))}
        for name in MODELS:
            lr = np.log(R["m"] / R[f"Hq_{name}"].clip(1e-9))
            rt[f"log_m_over_H_{name}"] = {f"{lo}〜{hi}": round(float(lr[R["band"] == i].mean()), 4) for i, (lo, hi) in enumerate(PAIR_POP_BANDS)}
            rt[f"m_over_H_sum_{name}"] = {f"{lo}〜{hi}": round(float(R.loc[R["band"] == i, "m"].sum() / R.loc[R["band"] == i, f"Hq_{name}"].sum()), 4)
                                          for i, (lo, hi) in enumerate(PAIR_POP_BANDS)}
            for oname in ("o", "o_mid"):
                ev = R[f"Hp_{name}"] * R[oname]
                rt[f"ev_{name}_{oname}"] = {str(t): {"pairs": int((ev >= t).sum()), "per_race": round(float((ev >= t).sum() / nr), 3),
                                                     "races_with_any": round(float(R.loc[ev >= t, "race_id"].nunique() / nr), 3),
                                                     "ai_over_m": round(float(R.loc[ev >= t, f"Hp_{name}"].sum() / max(R.loc[ev >= t, "m"].sum(), 1e-12)), 4),
                                                     "ai_over_Hq": round(float(R.loc[ev >= t, f"Hp_{name}"].sum() / max(R.loc[ev >= t, f"Hq_{name}"].sum(), 1e-12)), 4),
                                                     "m_over_Hq": round(float(R.loc[ev >= t, "m"].sum() / max(R.loc[ev >= t, f"Hq_{name}"].sum(), 1e-12)), 4),
                                                     "median_odds": float(np.median(R.loc[ev >= t, oname])) if (ev >= t).any() else None}
                                            for t in EV_LINES}
        res[ty] = rt
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tok.finish(result_path=OUT)
    print(json.dumps(res, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
