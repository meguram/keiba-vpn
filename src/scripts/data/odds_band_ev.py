"""人気帯・単勝オッズ帯ごとの単勝／複勝の回収率（確定オッズ＝発走直前の最終オッズで代用）と、★評価。調査・分析。

python -m src.scripts.data.odds_band_ev            # odds_band_ev.json と odds_band_ev.md を書く
回収率は 100 円ずつ全頭に賭けたときの払戻の平均（確定払戻）。払戻の解釈ができなかったレースは除く（0 円扱いにしない）。
区間はレース単位のクラスタブートストラップ（同じレースの馬は一緒に抜く）。2020–2025 年で評価し、2026 年は選択に使わず確認だけに使う。
★: 『真の回収率が 85% 以上である確率』（回収率の事前を 80% とした縮約）と、年別の安定で付ける。
"""

from __future__ import annotations

import json
import os
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.tp28_workout_stats import parse_payoff

ROOT = Path(__file__).resolve().parents[3]
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
EDGES = [0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 1e9]
LABELS = ["~1.4", "1.5~1.9", "2.0~2.9", "3.0~3.9", "4.0~4.9", "5.0~6.9", "7.0~9.9", "10~14.9", "15~19.9", "20~29.9", "30~49.9", "50~99.9", "100~"]
PRIOR_MEAN, THRESH = 0.80, 0.85


def _use_race_odds(d: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """race_result_flat の odds・popularity は 2025 年に壊れたレースがある（払戻と合わない）。race_odds_flat の確定単勝オッズ・人気があれば、それを優先する。"""
    parts = []
    for y in years:
        p = TABLES / str(y) / "race_odds_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_number", "win_odds", "popularity"]))
    if not parts:
        return d
    o = pd.concat(parts, ignore_index=True)
    o["race_id"] = o["race_id"].astype(str)
    for c in ("horse_number", "win_odds", "popularity"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = o.dropna(subset=["win_odds"]).drop_duplicates(["race_id", "horse_number"]).rename(columns={"win_odds": "odds_ro", "popularity": "pop_ro"})
    d = d.merge(o, on=["race_id", "horse_number"], how="left")
    ok = d["odds_ro"].notna() & (d["odds_ro"] > 0)
    d["odds"] = np.where(ok, d["odds_ro"], d["odds"])
    d["popularity"] = np.where(ok & d["pop_ro"].notna(), d["pop_ro"], d["popularity"])
    return d.drop(columns=["odds_ro", "pop_ro"])


def load(years: list[int]) -> pd.DataFrame:
    cols = ["race_id", "date", "horse_number", "finish_position", "odds", "popularity", "payoff", "venue", "surface", "field_size"]
    d = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    for c in ("horse_number", "finish_position", "odds", "popularity", "field_size"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["surface"].isin(["芝", "ダート"])].dropna(subset=["horse_number", "finish_position", "odds", "popularity"])
    d = _use_race_odds(d, years)
    d = d[(d["popularity"] >= 1) & (d["odds"] > 0)]                   # 人気・オッズのない馬（取消・除外など）は除く
    d["year"] = pd.to_datetime(d["date"]).dt.year
    rw, rp, okw, okp = [], [], [], []
    for _, g in d.groupby("race_id", sort=False):
        fin = {int(h): int(f) for h, f in zip(g["horse_number"], g["finish_position"])}
        pay = parse_payoff(g["payoff"].iloc[0], fin)
        n_top = sum(1 for f in fin.values() if f <= 3)
        has_w = len(pay["win"]) >= 1 and all(h in pay["win"] for h, f in fin.items() if f == 1)
        has_p = len(pay["place"]) >= min(n_top, 3) and len(pay["place"]) > 0
        for h in g["horse_number"].astype(int):
            rw.append(pay["win"].get(h, 0.0)); rp.append(pay["place"].get(h, 0.0)); okw.append(has_w); okp.append(has_p)
    d["ret_win"], d["ret_place"], d["ok_win"], d["ok_place"] = rw, rp, okw, okp
    d["band"] = pd.cut(d["odds"], EDGES, labels=LABELS, right=False)
    d["pop"] = d["popularity"].clip(upper=18).astype(int)
    return d


def band_stats(d: pd.DataFrame, key: str, ret: str, ok: str, n_boot: int = 300, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = d[d[ok]]
    rows = []
    for k, g in x.groupby(key, observed=True):
        if len(g) < 200:
            continue
        r = g[ret].to_numpy(); rid = g["race_id"].to_numpy()
        codes, uniq = pd.factorize(rid)
        s = np.bincount(codes, weights=r); n = np.bincount(codes)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(uniq), len(uniq))
            boots.append(s[i].sum() / n[i].sum())
        se = max(float(np.std(boots)), 1e-3)
        mean = float(r.mean())
        # 縮約: 事前 N(0.80, 0.08²)（市場全体の控除後の水準と、帯ごとのばらつき）
        pv = 0.08**2
        post_var = 1 / (1 / pv + 1 / se**2); post_mean = post_var * (PRIOR_MEAN / pv + mean / se**2)
        p85 = 0.5 * (1 + erf(((post_mean - THRESH) / sqrt(post_var)) / sqrt(2)))
        yr = g.groupby("year")[ret].mean()
        rows.append({"band": str(k), "n": int(len(g)), "n_races": int(len(uniq)), "hit": float((g[ret] > 0).mean()), "return": mean, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     "shrunk": float(post_mean), "p_ge_85": float(p85), "by_year": {int(a): round(float(b), 3) for a, b in yr.items()}, "years_ge_85": int((yr >= THRESH).sum()), "n_years": int(len(yr))})
    return rows


def stars(r: dict) -> str:
    p, y = r["p_ge_85"], r["years_ge_85"] / max(r["n_years"], 1)
    if p >= 0.8 and y >= 0.67:
        return "★★★★★"
    if p >= 0.6 and y >= 0.5:
        return "★★★★"
    if p >= 0.4:
        return "★★★"
    if p >= 0.15:
        return "★★"
    return "★"


def main() -> None:
    d = load(list(range(2020, 2027)))
    chk = d[(d["finish_position"] == 1) & d["ok_win"]]
    implied = chk["ret_win"] * 1.0
    cons = float(((implied - chk["odds"]).abs() < 0.2).mean())          # 払戻と記録されたオッズの整合
    train = d[d["year"] <= 2025]
    rep: dict = {"odds_payoff_consistency_winners": round(cons, 3), "rows": int(len(d)), "market_return_win": float(train[train["ok_win"]]["ret_win"].mean()), "market_return_place": float(train[train["ok_place"]]["ret_place"].mean())}
    for key in ("pop", "band"):
        for ret, ok, nm in (("ret_win", "ok_win", "単勝"), ("ret_place", "ok_place", "複勝")):
            rows = band_stats(train, key, ret, ok)
            hold = band_stats(d[d["year"] == 2026], key, ret, ok, n_boot=100) if (d["year"] == 2026).any() else []
            h = {r["band"]: r for r in hold}
            for r in rows:
                r["stars"] = stars(r)
                r["y2026"] = round(h[r["band"]]["return"], 3) if r["band"] in h else None
            rep[f"{key}_{nm}"] = rows
    (OUT / "odds_band_ev.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"全体の回収率: 単勝 {rep['market_return_win']:.3f}、複勝 {rep['market_return_place']:.3f}（2020–2025）。払戻とオッズの整合 {cons:.3f}"]
    for key, nm in (("pop", "人気"), ("band", "単勝オッズ帯")):
        for kind in ("単勝", "複勝"):
            L += ["", f"## {nm}別 {kind}の回収率", "| 帯 | 出走 | 的中率 | 回収率 | 95%区間 | 縮約後 | 85%以上の確率 | 年別85%以上 | 2026年 | ★ |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in rep[f"{key}_{kind}"]:
                L.append(f"| {r['band']} | {r['n']} | {r['hit']:.3f} | {r['return']:.3f} | {r['ci'][0]:.3f}~{r['ci'][1]:.3f} | {r['shrunk']:.3f} | {r['p_ge_85']:.2f} | {r['years_ge_85']}/{r['n_years']} | {r['y2026']} | {r['stars']} |")
    (OUT / "odds_band_ev.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:1]))


if __name__ == "__main__":
    main()
