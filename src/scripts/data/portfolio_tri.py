"""3連単を候補に加えた『全券種・縛りなし』の検証（portfolio_target_eval の候補表に 3連単の行を足す）。

python -m src.scripts.data.portfolio_tri build            # cands_tri_<年>.parquet（3連単の候補。EV上位40＋確率上位25に刈り込み）
python -m src.scripts.data.portfolio_tri validate         # 想定3連単オッズの写像を、取得済みの実オッズ（2024〜2025）で確認
python -m src.scripts.data.portfolio_tri grid             # 全券種(単勝〜3連単)・縛りなしの格子（M=5）を評価
的中確率は MB60（自前の勝率・連対率・複勝率）、想定オッズは odds_pred の構成（想定単複の周辺確率 → MB60 の確率 → 2 次多項式）。
券種・馬の組合せに共有の縛りは入れない（買い目は 1 件ずつ独立）。
"""

from __future__ import annotations

import json
import sys
from itertools import permutations

import numpy as np
import pandas as pd

from src.scripts.data import odds_pred_eval as OP_E
from src.scripts.data import portfolio_target_eval as E

BT_TRI = 6
E.TYPESETS["ALL7"] = {0, 1, 2, 3, 4, 5, 6}
E.TYPESETS["EXOTIC7"] = {2, 3, 4, 5, 6}
E.TYPESETS["SANRENTAN"] = {6}
N_KEEP_EV, N_KEEP_P = 40, 25


def _coef() -> list[float]:
    cfg = json.loads((E.ROOT / "config" / "odds_pred" / "odds_pred_config.json").read_text(encoding="utf-8"))
    return cfg["poly2"]["sanrentan"]


def build_year(year: int) -> dict:
    co = _coef()
    base = pd.read_parquet(E.OUT / f"cands_{year}.parquet", columns=["race_id", "date"]).drop_duplicates("race_id")
    dates = dict(zip(base["race_id"], base["date"]))
    fr = pd.read_parquet(E.FRAME, columns=["race_id", "horse_number", "p_win", "p_top2", "p_top3", "year", "horse_id"])
    fr = fr[(fr["year"] == year) & fr["race_id"].astype(str).isin(dates)].copy()
    fr["race_id"], fr["horse_id"] = fr["race_id"].astype(str), fr["horse_id"].astype(str)
    wp = pd.read_parquet(E.OP / "wp_oof_pred.parquet", columns=["race_id", "horse_id", "win_B", "place_B"])
    wp["race_id"], wp["horse_id"] = wp["race_id"].astype(str), wp["horse_id"].astype(str)
    d = fr.merge(wp, on=["race_id", "horse_id"], how="inner")
    pay = pd.read_parquet(E.BET / "payoff.parquet")
    pay["race_id"] = pay["race_id"].astype(str)
    pay = pay[pay["bet_type"] == "sanrentan"].drop_duplicates(["race_id", "combo"], keep="last")
    pm = {(r, str(c)): v for r, c, v in zip(pay["race_id"], pay["combo"], pay["payout"])}
    parts = []
    for rid, g in d.groupby("race_id", sort=True):
        g = g.sort_values("horse_number")
        n = len(g)
        if n < 5:
            continue
        hn = g["horse_number"].to_numpy().astype(int)
        w_o = g["p_win"].to_numpy(float)
        Do = OP_E._dist(w_o / w_o.sum(), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float))
        w_h, p2h, p3h = OP_E._marginals(1.0 / g["win_B"].to_numpy(float), g["place_B"].to_numpy(float))
        Dh = OP_E._dist(w_h, p2h, p3h)
        idx = np.array(list(permutations(range(n), 3)))
        a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
        p_own, p_hat = Do._p3[a, b, c], Dh._p3[a, b, c]
        odds = np.clip(np.exp(np.polyval(co, np.log(1.0 / np.clip(p_hat, 1e-12, None)))), 1.0, None)
        ev = p_own * odds
        r_ev = (-ev).argsort(kind="stable").argsort(kind="stable") + 1
        r_p = (-p_own).argsort(kind="stable").argsort(kind="stable") + 1
        keep = (r_ev <= N_KEEP_EV) | (r_p <= N_KEEP_P)
        keys = [f"{hn[i]}>{hn[j]}>{hn[k]}" for i, j, k in idx[keep]]
        parts.append(pd.DataFrame({"race_id": rid, "bt": BT_TRI, "combo": keys, "p": p_own[keep], "o": odds[keep],
                                   "pay100": [pm.get((rid, k), 0.0) for k in keys], "n": n, "year": year, "date": dates[rid]}))
    out = pd.concat(parts, ignore_index=True)
    for col in ("hn", "pop", "p_rank", "wB_rank"):
        out[col] = np.nan
    out.to_parquet(E.OUT / f"cands_tri_{year}.parquet", index=False)
    return {"year": year, "races": int(out["race_id"].nunique()), "rows": len(out), "hits": int((out["pay100"] > 0).sum())}


def load_races(years: list[int]) -> list[dict]:
    """元の候補（単勝〜3連複）に 3連単の行を足したレース配列。"""
    races = E.load_races(years)
    tri = pd.concat([pd.read_parquet(E.OUT / f"cands_tri_{y}.parquet") for y in years], ignore_index=True)
    tri_by = {rid: g for rid, g in tri.groupby("race_id")}
    for R in races:
        g = tri_by.get(R["race_id"])
        if g is None:
            continue
        R["bt"] = np.concatenate([R["bt"], g["bt"].to_numpy().astype(int)])
        for key, col in (("combo", "combo"), ("p", "p"), ("o", "o"), ("pay", "pay100"), ("hn", "hn"), ("pop", "pop"),
                         ("p_rank", "p_rank"), ("wB_rank", "wB_rank")):
            R[key] = np.concatenate([R[key], g[col].to_numpy()])
    return races


def validate(n_races: int = 300) -> None:
    """想定3連単オッズ（全組合せ）を、取得済みの実オッズ（race_trifecta_odds、キャップ値除く）と比べる。"""
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    co = _coef()
    rows = []
    for year in (2024, 2025):
        base = pd.read_parquet(E.OUT / f"cands_{year}.parquet", columns=["race_id"]).drop_duplicates("race_id")["race_id"].tolist()
        rng = np.random.default_rng(year)
        pick = list(rng.choice(base, size=min(n_races // 2, len(base)), replace=False))
        fr = pd.read_parquet(E.FRAME, columns=["race_id", "horse_number", "horse_id", "year"])
        fr = fr[fr["year"] == year]
        fr["race_id"], fr["horse_id"] = fr["race_id"].astype(str), fr["horse_id"].astype(str)
        wp = pd.read_parquet(E.OP / "wp_oof_pred.parquet", columns=["race_id", "horse_id", "win_B", "place_B"])
        wp["race_id"], wp["horse_id"] = wp["race_id"].astype(str), wp["horse_id"].astype(str)
        d = fr.merge(wp, on=["race_id", "horse_id"])
        for rid in pick:
            doc = st.load("race_trifecta_odds", rid)
            g = d[d["race_id"] == rid].sort_values("horse_number")
            if not doc or len(g) < 5:
                continue
            hn = g["horse_number"].to_numpy().astype(int)
            w_h, p2h, p3h = OP_E._marginals(1.0 / g["win_B"].to_numpy(float), g["place_B"].to_numpy(float))
            Dh = OP_E._dist(w_h, p2h, p3h)
            pos = {int(h): i for i, h in enumerate(hn)}
            for a, b, c, o, _pop in doc.get("trifecta", []):
                if o >= 999999 or o <= 0 or int(a) not in pos or int(b) not in pos or int(c) not in pos:
                    continue
                p = Dh._p3[pos[int(a)], pos[int(b)], pos[int(c)]]
                rows.append((year, rid, float(o), float(np.exp(np.polyval(co, np.log(1.0 / max(p, 1e-12)))))))
    v = pd.DataFrame(rows, columns=["year", "race_id", "act", "hat"])
    v["err"] = np.log(v["hat"]) - np.log(v["act"])
    v["bucket"] = pd.cut(v["hat"], [0, 50, 200, 1000, 5000, 1e9], labels=["<50", "50-200", "200-1000", "1000-5000", ">5000"])
    print("組合せ数", len(v), "対数RMSE", round(float(np.sqrt((v["err"] ** 2).mean())), 3), "偏り(対数平均)", round(float(v["err"].mean()), 3))
    print(v.groupby("bucket", observed=True)["err"].agg(["count", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(3).to_string())
    v.groupby(["year", "bucket"], observed=True)["err"].agg(["count", "mean"]).round(3).to_csv(E.OUT / "tri_validate.csv")


def grid() -> None:
    races = load_races(E.YEARS)
    by_year = {y: [r for r in races if r["year"] == y] for y in E.YEARS}
    pooled = [r for y in E.SEL_YEARS for r in by_year[y]]
    rows = []
    for ts in ("ALL7", "EXOTIC7", "ALL", "SANRENTAN"):
        for mb in (1, 3, 5, None):
            for k in (1.25, 1.5, 2.0, 2.5, 3.0):
                for tau in (0.0, 1.0):
                    c = (ts, mb, 5.0, k, tau)
                    s, v = E.summarize(E.run_design(pooled, c), 5.0), E.summarize(E.run_design(by_year[2025], c), 5.0)
                    rows.append({"typeset": ts, "max_bets": mb, "k": k, "tau": tau, **{f"a_{x}": s[x] for x in s}, **{f"b_{x}": v[x] for x in v}})
        print(ts, "done", flush=True)
    pd.DataFrame(rows).to_parquet(E.OUT / "grid_tri.parquet", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        for y in E.YEARS:
            print(build_year(y), flush=True)
    elif cmd == "validate":
        validate()
    elif cmd == "grid":
        grid()
