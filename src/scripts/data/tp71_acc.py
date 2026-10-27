"""tp-71 準備: 券種別の『想定オッズ』の精度（確定オッズとの差・順位相関・較正・人気帯別・穴馬の偏り）。読み取りのみ（馬券のシミュレーションはしない）。

python -m src.scripts.data.tp71_acc run --out DIR [--years 2022 2023 2024 2025] [--sample-tri 0.15]

想定の作り方（本番の案 B と同じ構造）: 想定単勝（V8＝各年より前だけで学習した po_<年>）→ レース内で 1/オッズを正規化 → Harville の着順分布 → 公正オッズ = 払戻率 ÷ 確率
→ 券種別に log(実) = a + b·log(公正) で較正。較正の係数は 2 通り: (本番) config/paid_zone.json の odds_calibration（実単勝・2022〜23 年で当てはめ）／
(WF) 想定単勝を入力にして『その年より前の年』で当てはめ直した係数（walk-forward。2022 年は当てはめ専用）。
正解: 単勝・複勝＝確定オッズ、馬連・ワイド・馬単＝全組合せの確定オッズ（ワイドは下限と上限の中央）、3連単＝2024〜26 年は全組合せ（tp48）、
3連複＝全組合せの確定オッズは過去に無いので勝ち馬券の払戻のみ（的中した組合せだけの比較。偏りが出やすい点に注意）。2026 年は参照のみ（較正・選択に使わない）。
指標: 対数誤差の偏り・RMSE・MAE、レース内スピアマン順位相関、想定オッズの帯ごとの『実オッズ平均 ÷ 想定オッズ平均』、実際の人気帯別の偏り、穴（実オッズ 50 倍以上）の偏り。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
BET26 = ROOT / "data/local/modeling/betting_2026"
TP48 = BET / "tp48"
OUT = Path(os.environ.get("TP71_OUT", "/home/jovyan/work/tp71_out"))
PAYBACK = {"umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}
PROD_CAL = {"umaren": (0.8852, 0.7773), "wide": (0.6939, 0.7283), "umatan": (0.9633, 0.7971), "sanrenpuku": (-0.0608, 1.0)}
POP_BANDS = [(1, 3, "人気1-3"), (4, 6, "人気4-6"), (7, 9, "人気7-9"), (10, 99, "人気10以下")]
ODDS_BANDS = [(0, 5, "~5倍"), (5, 15, "5-15倍"), (15, 50, "15-50倍"), (50, 200, "50-200倍"), (200, 1e9, "200倍~")]


def _combos(n: int, k: int, ordered: bool) -> np.ndarray:
    it = permutations(range(n), k) if ordered else combinations(range(n), k)
    return np.array(list(it), dtype=np.int16).reshape(-1, k)


_CACHE: dict = {}


def idx(n: int, k: int, ordered: bool) -> np.ndarray:
    key = (n, k, ordered)
    if key not in _CACHE:
        _CACHE[key] = _combos(n, k, ordered)
    return _CACHE[key]


def load_win_place(years) -> pd.DataFrame:
    a = pd.read_parquet(BET / "odds_win_place.parquet")
    b = pd.read_parquet(BET26 / "odds_win_place.parquet")
    d = pd.concat([a, b], ignore_index=True)
    d["year"] = d["race_id"].str[:4].astype(int)
    d = d[d["year"].isin(list(years)) & (d["win_odds"] > 0)]
    return d[["race_id", "year", "horse_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity"]].drop_duplicates(["race_id", "horse_number"])


def load_pairs(years) -> pd.DataFrame:
    parts = []
    for y in years:
        src = BET26 if y == 2026 else BET
        p = pd.read_parquet(src / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max", "popularity"])
        parts.append(p[p["race_id"].str[:4] == str(y)])
    p = pd.concat(parts, ignore_index=True)
    p["o"] = np.where(p["bet_type"] == "wide", (p["odds_min"] + p["odds_max"]) / 2, p["odds"])
    p = p[p["o"] > 0]
    s = p["combo"].str.replace(">", "-", regex=False).str.split("-", expand=True).astype(float)
    p["a"], p["b"] = s[0].astype(int), s[1].astype(int)
    return p[["race_id", "bet_type", "a", "b", "o", "popularity"]]


def load_payoffs() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(BET / "payoff.parquet"), pd.read_parquet(BET26 / "payoff.parquet")], ignore_index=True)
    d = d[d["bet_type"].isin(["sanrenpuku", "sanrentan"]) & ~d["ambiguous"].astype(bool) & d["parsed"].astype(bool)]
    s = d["combo"].astype(str).str.replace(">", "-", regex=False).str.split("-", expand=True)
    d = d[s[2].notna()].copy()
    s = s[s[2].notna()].astype(float)
    d["a"], d["b"], d["c"] = s[0].astype(int), s[1].astype(int), s[2].astype(int)
    d["o"] = d["payout"].astype(float) / 100.0
    return d[["race_id", "bet_type", "a", "b", "c", "o", "popularity"]]


def load_trifecta(years) -> pd.DataFrame:
    parts = []
    for y in years:
        f = TP48 / f"trifecta_{y}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame(columns=["race_id", "a", "b", "c", "odds", "pop"])
    t = pd.concat(parts, ignore_index=True)
    t["race_id"] = t["race_id"].astype(str)
    return t


def load_po(po_dir: str = "po", prefix: str = "po") -> pd.DataFrame:
    fs = sorted((OUT / po_dir).glob(f"{prefix}_*.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=["race_id", "horse_id", "win_pred", "place_pred"]) for f in fs], ignore_index=True)
    p["race_id"] = p["race_id"].astype(str)
    return p


def race_tables(g: pd.DataFrame, win_col: str):
    """レース g（馬番昇順）から Harville の公正オッズを組合せごとに返す（キー配列つき）。"""
    from src.pipeline.inference.exotic_probabilities import RankDistribution

    g = g.sort_values("horse_number")
    hn = g["horse_number"].to_numpy(int)
    q = 1.0 / g[win_col].to_numpy(float)
    dist = RankDistribution(q / q.sum())
    n = len(hn)
    out = {}
    p2, p3 = dist._p2, dist._p3
    ip = idx(n, 2, False)
    io = idx(n, 2, True)
    out["umaren"] = (hn[ip[:, 0]], hn[ip[:, 1]], None, (p2 + p2.T)[ip[:, 0], ip[:, 1]])
    w = p3.sum(2) + p3.sum(1) + p3.sum(0)
    out["wide"] = (hn[ip[:, 0]], hn[ip[:, 1]], None, (w + w.T)[ip[:, 0], ip[:, 1]])
    out["umatan"] = (hn[io[:, 0]], hn[io[:, 1]], None, p2[io[:, 0], io[:, 1]])
    return out, dist, hn


def fair_of(bt: str, prob: np.ndarray) -> np.ndarray:
    return PAYBACK[bt] / np.clip(prob, 1e-9, None)


def run(a) -> None:
    from src.pipeline.inference.exotic_probabilities import RankDistribution  # noqa: F401

    years = list(a.years)
    wp = load_win_place(years + ([2026] if 2026 not in years else []))
    po = load_po(a.po_dir, a.po_prefix)
    wp = wp.merge(po, on=["race_id", "horse_id"], how="left")
    pairs = load_pairs(years + [2026])
    pay = load_payoffs()
    tri = load_trifecta([2024, 2025, 2026])
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("umaren", "wide", "umatan", "sanrenpuku", "sanrentan")}
    single_rows = []
    pair_by_race = {k: g for k, g in pairs.groupby(["race_id", "bet_type"], sort=False)}
    pay_by_race = {k: g for k, g in pay.groupby(["race_id", "bet_type"], sort=False)}
    tri_by_race = {k: g for k, g in tri.groupby("race_id", sort=False)} if len(tri) else {}
    done = 0
    for rid, g in wp.groupby("race_id", sort=False):
        year = int(rid[:4])
        if len(g) < 5:
            continue
        has_po = g["win_pred"].notna().all()
        srcs = []
        if has_po:
            srcs.append(("pred", "win_pred"))
        srcs.append(("actual", "win_odds"))
        if has_po:
            la = np.log(g["win_odds"].to_numpy(float))
            single_rows.append(pd.DataFrame({"year": year, "race_id": rid, "bt": "tansho", "pred": g["win_pred"].to_numpy(float), "act": g["win_odds"].to_numpy(float),
                                             "pop": g["popularity"].to_numpy(float)}))
            pm = g["place_odds_min"].to_numpy(float)
            ok = pm > 0
            if ok.any():
                single_rows.append(pd.DataFrame({"year": year, "race_id": rid, "bt": "fukusho", "pred": g["place_pred"].to_numpy(float)[ok], "act": pm[ok],
                                                 "pop": g["popularity"].to_numpy(float)[ok]}))
        for src, col in srcs:
            gg = g.sort_values("horse_number")
            tabs, dist, hn = race_tables(gg, col)
            for bt in ("umaren", "wide", "umatan"):
                pa = pair_by_race.get((rid, bt))
                if pa is None:
                    continue
                x, y_, _, prob = tabs[bt]
                df = pd.DataFrame({"a": x, "b": y_, "lf": np.log(fair_of(bt, prob))})
                m = df.merge(pa[["a", "b", "o", "popularity"]], on=["a", "b"], how="inner")
                if len(m):
                    rows[bt].append(pd.DataFrame({"year": year, "race_id": rid, "src": src, "lf": m["lf"].to_numpy(), "la": np.log(m["o"].to_numpy(float)), "pop": m["popularity"].to_numpy(float)}))
            # 3連複・3連単: 勝ち馬券（払戻）
            idx_of = {int(h): i for i, h in enumerate(hn)}
            for bt in ("sanrenpuku", "sanrentan"):
                pw = pay_by_race.get((rid, bt))
                if pw is None:
                    continue
                lf, la, pp = [], [], []
                for r in pw.itertuples(index=False):
                    try:
                        i, j, k = idx_of[r.a], idx_of[r.b], idx_of[r.c]
                    except KeyError:
                        continue
                    pr = dist.sanrenpuku(i, j, k) if bt == "sanrenpuku" else dist.sanrentan(i, j, k)
                    lf.append(math.log(fair_of(bt, np.array([pr]))[0]))
                    la.append(math.log(r.o))
                    pp.append(r.popularity)
                if lf:
                    rows[bt].append(pd.DataFrame({"year": year, "race_id": rid, "src": src, "lf": lf, "la": la, "pop": pp, "winner": True}))
            # 3連単の全組合せ（2024〜26）
            tr = tri_by_race.get(rid)
            if tr is not None and year >= 2024 and a.sample_tri > 0:
                n = len(hn)
                io3 = idx(n, 3, True)
                sel = rng.random(len(io3)) < a.sample_tri
                io3 = io3[sel]
                pr = dist._p3[io3[:, 0], io3[:, 1], io3[:, 2]]
                df = pd.DataFrame({"a": hn[io3[:, 0]], "b": hn[io3[:, 1]], "c": hn[io3[:, 2]], "lf": np.log(fair_of("sanrentan", pr))})
                m = df.merge(tr.rename(columns={"odds": "o"})[["a", "b", "c", "o", "pop"]], on=["a", "b", "c"], how="inner")
                if len(m):
                    rows["sanrentan"].append(pd.DataFrame({"year": year, "race_id": rid, "src": src, "lf": m["lf"].to_numpy(), "la": np.log(m["o"].to_numpy(float)), "pop": m["pop"].to_numpy(float), "winner": False}))
        done += 1
        if done % 500 == 0:
            print(done, flush=True)
    (OUT / a.acc).mkdir(parents=True, exist_ok=True)
    for k, v in rows.items():
        if v:
            d = pd.concat(v, ignore_index=True)
            if "winner" not in d:
                d["winner"] = False
            d["winner"] = d["winner"].fillna(False)
            d.to_parquet(OUT / a.acc / f"rows_{k}.parquet", index=False)
    if single_rows:
        pd.concat(single_rows, ignore_index=True).to_parquet(OUT / a.acc / "rows_single.parquet", index=False)
    print("rows 保存", {k: sum(len(x) for x in v) for k, v in rows.items()}, flush=True)


def _fit(t: pd.DataFrame):
    if len(t) < 50:
        return None
    b, a0 = np.polyfit(t["lf"].to_numpy(), t["la"].to_numpy(), 1)
    return float(a0), float(b)


def _metrics(t: pd.DataFrame, a0: float, b: float) -> dict:
    p = a0 + b * t["lf"].to_numpy()
    e = p - t["la"].to_numpy()
    sp = []
    for _, tt in t.groupby("race_id"):
        if len(tt) >= 6:
            s = spearmanr(tt["lf"], tt["la"])[0]
            if s == s:
                sp.append(s)
    out = {"n": int(len(t)), "races": int(t["race_id"].nunique()), "bias": round(float(e.mean()), 3), "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
           "mae": round(float(np.abs(e).mean()), 3), "spearman": round(float(np.mean(sp)), 3) if sp else None}
    pred_o, act_o = np.exp(p), np.exp(t["la"].to_numpy())
    out["ratio_mean_actual_over_pred"] = round(float(act_o.mean() / pred_o.mean()), 3)
    bands = {}
    for lo, hi, nm in ODDS_BANDS:
        m = (pred_o >= lo) & (pred_o < hi)
        if m.sum() >= 30:
            bands[nm] = {"n": int(m.sum()), "ratio": round(float(act_o[m].mean() / pred_o[m].mean()), 3), "log_bias": round(float(e[m].mean()), 3)}
    out["by_pred_band"] = bands
    pop = t["pop"].to_numpy(float)
    pb = {}
    for lo, hi, nm in POP_BANDS:
        m = (pop >= lo) & (pop <= hi)
        if m.sum() >= 30:
            pb[nm] = {"n": int(m.sum()), "log_bias": round(float(e[m].mean()), 3), "ratio": round(float(act_o[m].mean() / pred_o[m].mean()), 3)}
    out["by_pop"] = pb
    m = act_o >= 50
    if m.sum() >= 30:
        out["longshot_ge50"] = {"n": int(m.sum()), "log_bias": round(float(e[m].mean()), 3), "share": round(float(m.mean()), 3)}
    return out


def report(a) -> None:
    acc = OUT / a.acc
    res: dict = {"table": [], "fit": {}}
    for bt in ("umaren", "wide", "umatan", "sanrenpuku", "sanrentan"):
        f = acc / f"rows_{bt}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        for src in ("pred", "actual"):
            s = d[d["src"] == src]
            for scope in ("full", "winner"):
                ss = s[s["winner"]] if scope == "winner" else s[~s["winner"]]
                if bt in ("sanrenpuku",) and scope == "full":
                    continue
                if bt in ("umaren", "wide", "umatan") and scope == "winner":
                    continue
                if not len(ss):
                    continue
                for y in sorted(ss["year"].unique()):
                    ty = ss[ss["year"] == y]
                    prior = ss[ss["year"] < y]
                    # (本番) config の係数
                    if bt in PROD_CAL:
                        a0, b0 = PROD_CAL[bt]
                        res["table"].append({"bt": bt, "src": src, "scope": scope, "year": int(y), "cal": "prod_config", **_metrics(ty, a0, b0)})
                    # 無較正（公正オッズそのもの）
                    res["table"].append({"bt": bt, "src": src, "scope": scope, "year": int(y), "cal": "raw_fair", **_metrics(ty, 0.0, 1.0)})
                    # walk-forward（想定単勝を入力にして前年までで当てはめ）
                    if bt == "sanrenpuku" or bt == "sanrentan":
                        base = d[(d["src"] == src) & d["winner"]]
                        pa = base[(base["year"] < y) & (base["year"] >= 2022)]
                        fit = (float((pa["la"] - pa["lf"]).mean()), 1.0) if len(pa) >= 50 else None
                    else:
                        pa = prior[prior["year"] >= 2022]
                        fit = _fit(pa)
                    if fit:
                        res["fit"][f"{bt}/{src}/{scope}/{int(y)}"] = {"a": round(fit[0], 4), "b": round(fit[1], 4), "n": int(len(pa))}
                        res["table"].append({"bt": bt, "src": src, "scope": scope, "year": int(y), "cal": "walk_forward", **_metrics(ty, fit[0], fit[1])})
    # 単勝・複勝
    f = acc / "rows_single.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        d["lf"], d["la"] = np.log(d["pred"]), np.log(d["act"])
        for bt in ("tansho", "fukusho"):
            s = d[d["bt"] == bt]
            for y in sorted(s["year"].unique()):
                ty = s[s["year"] == y]
                pa = s[s["year"] < y]
                res["table"].append({"bt": bt, "src": "pred", "scope": "full", "year": int(y), "cal": "raw_V8", **_metrics(ty, 0.0, 1.0)})
                fit = _fit(pa)
                if fit:
                    res["fit"][f"{bt}/pred/full/{int(y)}"] = {"a": round(fit[0], 4), "b": round(fit[1], 4), "n": int(len(pa))}
                    res["table"].append({"bt": bt, "src": "pred", "scope": "full", "year": int(y), "cal": "walk_forward", **_metrics(ty, fit[0], fit[1])})
    (acc / "accuracy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res["table"]:
        if r["cal"] in ("walk_forward", "prod_config") and r["src"] == "pred" or (r["src"] == "actual" and r["cal"] == "prod_config"):
            print(r["bt"], r["src"], r["scope"], r["year"], r["cal"], "n", r["n"], "bias", r["bias"], "rmse", r["rmse"], "sp", r["spearman"], "ratio", r["ratio_mean_actual_over_pred"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--sample-tri", type=float, default=0.15)
    p.add_argument("--po-dir", default="po")
    p.add_argument("--po-prefix", default="po")
    p.add_argument("--acc", default="acc")
    p.set_defaults(fn=run)
    q = sub.add_parser("report")
    q.add_argument("--acc", default="acc")
    q.set_defaults(fn=report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
