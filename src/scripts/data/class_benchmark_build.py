"""クラス別の基準めぐ指数（突破ライン）の表 config/class_benchmark.json を作る。

指数 = 絶対指数 megu_v3a_abs_robust（status ok・低信頼を除く）。『○歳XXX』のクラスごとに、勝ち馬の指数の
下位25%（突破ライン）・中央値・上位75%を出す。面（芝/ダ）・距離帯は、親との中央値の差が DELTA 点以上、かつ
n が MIN_N 以上のときだけ分ける（足りない所は親にフォールバック）。

実行（データのあるホストで）:
  python -m src.scripts.data.class_benchmark_build --years 2023,2024,2025 --out config/class_benchmark.json
  python -m src.scripts.data.class_benchmark_build --evaluate --build-years 2023,2024 --eval-year 2025 --report /tmp/class_benchmark_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.inference import class_benchmark as cb

MIN_N = 30
DELTA = 2.0          # 親との中央値の差がこの点（=0.2秒）以上のときだけ、面・距離帯で分ける
LOOKBACK = 5         # 直近何走の最高指数を見るか
BASES = {"win": lambda f: f == 1, "second": lambda f: f == 2, "top2": lambda f: f <= 2, "top3": lambda f: f <= 3}     # 基準にする馬
VALUE = "megu_index_v3a_abs_robust"
BASIS_JA = {"win": "勝ち馬", "second": "2着馬", "top2": "2着以内の馬", "top3": "3着以内の馬"}


# ── 純粋関数 ─────────────────────────────────────────────────

def _stat(s: pd.Series) -> dict:
    return {"n": int(len(s)), "median": round(float(s.median()), 1), "p25": round(float(s.quantile(0.25)), 1), "p75": round(float(s.quantile(0.75)), 1)}


def build_cells(w: pd.DataFrame, min_n: int = MIN_N, delta: float = DELTA) -> dict[str, dict]:
    """w: 基準にする走（勝ち馬など）。列 age, tier, surf, band, idx。キーは class_benchmark.keys_for と同じ形式。"""
    cells: dict[str, dict] = {}
    w = w.dropna(subset=["tier", "idx"])
    # クラスのみ（年齢不問）— 最後の親
    for tier, g in w.groupby("tier"):
        if len(g) >= min_n:
            cells[f"*|{tier}"] = {**_stat(g.idx), "level": "tier"}
    for (age, tier), g in w[w.age.notna()].groupby(["age", "tier"]):
        if len(g) < min_n:
            continue
        cells[f"{age}|{tier}"] = {**_stat(g.idx), "level": "cell"}
        parent = cells[f"{age}|{tier}"]
        for sf, gs in g[g.surf.notna()].groupby("surf"):
            if len(gs) < min_n:
                continue
            st = _stat(gs.idx)
            sp = parent
            if abs(st["median"] - parent["median"]) >= delta:
                cells[f"{age}|{tier}|{sf}"] = {**st, "level": "surface"}
                sp = cells[f"{age}|{tier}|{sf}"]
            for bd, gb in gs[gs.band.notna()].groupby("band"):
                if len(gb) < min_n:
                    continue
                sb = _stat(gb.idx)
                if abs(sb["median"] - sp["median"]) >= delta:
                    cells[f"{age}|{tier}|{sf}|{bd}"] = {**sb, "level": "distance"}
    return cells


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """race_class / grade / race_name / surface / distance から age, tier, surf, band を足す。"""
    ct = [cb.cell_of(a, b, c) for a, b, c in zip(df["race_class"], df["grade"], df["race_name"])]
    out = df.copy()
    out["age"] = [c[0] for c in ct]
    out["tier"] = [c[1] for c in ct]
    out["surf"] = out["surface"].map(cb.surface_key)
    out["band"] = out["distance"].map(cb.dist_band)
    return out


def prior_max(df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """各行について、その馬のそれ以前の有効な走（ok かつ低信頼でない）の直近 lookback 走の最高指数。無ければ NaN。"""
    d = df[df["ok"]].sort_values(["horse_id", "date", "race_id"])
    pm = d.groupby("horse_id")["idx"].transform(lambda s: s.shift(1).rolling(lookback, min_periods=1).max())
    return pm.reindex(df.index)


def lift_table(g: pd.DataFrame) -> dict:
    """g: 列 win, top3, exp_win(=1/頭数), exp_top3(=3/頭数)。ランダムに比べた倍率。"""
    n = len(g)
    if not n:
        return {"n": 0}
    return {"n": int(n), "win_rate": round(float(g.win.mean()), 4), "top3_rate": round(float(g.top3.mean()), 4),
            "win_lift": round(float(g.win.sum() / g.exp_win.sum()), 3), "top3_lift": round(float(g.top3.sum() / g.exp_top3.sum()), 3)}


def cluster_bootstrap_diff(g: pd.DataFrame, flag: str, metric: str, expected: str, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    """レース単位の再標本化で、(flag=True の倍率) − (flag=False の倍率) の 95% 区間。倍率 = 実績の合計 ÷ ランダムな期待の合計。"""
    f = g[flag].astype(bool)
    agg = pd.DataFrame({
        "race_id": g["race_id"], "am": g[metric].where(f, 0.0), "ae": g[expected].where(f, 0.0),
        "bm": g[metric].where(~f, 0.0), "be": g[expected].where(~f, 0.0),
    }).groupby("race_id").sum()
    if len(agg) < 30:
        return None
    rng = np.random.default_rng(seed)
    a = agg.to_numpy()
    cnt = rng.multinomial(len(a), np.full(len(a), 1.0 / len(a)), size=n_boot)         # 各回の、各レースが何回選ばれたか
    s = cnt @ a
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    if not ok.any():
        return None
    d = s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


# ── データ読み込み（ホスト用） ──────────────────────────────────

def load(tables: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        m = pd.read_parquet(tables / str(y) / "megu_index_flat.parquet", columns=["race_id", "horse_id", VALUE, "megu_v3a_status", "megu_v3a_low_confidence"])
        r = pd.read_parquet(tables / str(y) / "race_result_flat.parquet",
                            columns=["race_id", "horse_id", "race_class", "grade", "race_name", "surface", "distance", "date", "finish_position", "field_size"])
        frames.append(m.merge(r, on=["race_id", "horse_id"], how="inner"))
    df = pd.concat(frames, ignore_index=True)
    df["idx"] = df[VALUE]
    df["ok"] = (df["megu_v3a_status"] == "ok") & (~df["megu_v3a_low_confidence"].fillna(False).astype(bool)) & df["idx"].notna()
    df["year"] = df["race_id"].str[:4].astype(int)
    df["fin"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["date"] = df["date"].astype(str).str.replace("/", "-")
    return add_context(df)


def basis_rows(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    return df[df["ok"] & df["tier"].notna() & df["fin"].notna() & BASES[basis](df["fin"])]


def stability(df: pd.DataFrame, years: list[int], basis: str, min_year_n: int = 15) -> dict:
    """基準の安定性。(1) 四分位幅（p75−p25）の n 加重平均、(2) 年ごとの中央値が全期間の中央値からぶれる大きさ（各年 min_year_n 以上のセル）、
    (3) セルの p25 のブートストラップ標準誤差（n 加重）。小さいほど安定。"""
    w = basis_rows(df[df["year"].isin(years)], basis)
    w = w[w["age"].notna()]
    iqr, wobble, se, ns = [], [], [], []
    rng = np.random.default_rng(0)
    for (age, tier), g in w.groupby(["age", "tier"]):
        if len(g) < MIN_N:
            continue
        iqr.append(float(g.idx.quantile(0.75) - g.idx.quantile(0.25)))
        ns.append(len(g))
        full = float(g.idx.median())
        devs = [abs(float(gy.idx.median()) - full) for _, gy in g.groupby("year") if len(gy) >= min_year_n]
        wobble.append(float(np.mean(devs)) if devs else np.nan)
        v = g.idx.to_numpy()
        se.append(float(np.std([np.quantile(rng.choice(v, len(v)), 0.25) for _ in range(200)])))
    ns = np.array(ns, dtype=float)
    wm = lambda xs: round(float(np.nansum(np.array(xs) * ns) / ns[~np.isnan(np.array(xs))].sum()), 2) if len(xs) else None
    return {"cells": len(ns), "iqr": wm(iqr), "year_median_wobble": wm(wobble), "p25_boot_se": wm(se)}


def evaluate(df: pd.DataFrame, build_years: list[int], eval_year: int, basis: str = "win") -> dict:
    """基準を build_years で作り、eval_year の出走馬を『直近5走の最高指数 ≥ 突破ライン』で分けて勝率・3着内率を比べる。"""
    df = df.copy()
    df["prior"] = prior_max(df)
    base = basis_rows(df[df["year"].isin(build_years)], basis)
    cells = build_cells(base)
    doc = {"cells": cells}
    ev = df[(df["year"] == eval_year) & df["tier"].notna() & df["fin"].notna()].copy()
    lines = []
    for r in ev.itertuples():
        hit = cb.lookup(r.race_class, r.grade, r.race_name, r.surface, r.distance, doc)
        lines.append(hit["line"] if hit else np.nan)
    ev["line"] = lines
    ev["win"] = (ev["fin"] == 1).astype(int)
    ev["top3"] = (ev["fin"] <= 3).astype(int)
    fs = pd.to_numeric(ev["field_size"], errors="coerce").fillna(ev.groupby("race_id")["horse_id"].transform("count"))
    ev["exp_win"] = 1.0 / fs
    ev["exp_top3"] = np.minimum(3.0 / fs, 1.0)
    has = ev[ev["line"].notna()]
    known = has[has["prior"].notna()].copy()
    known["above"] = known["prior"] >= known["line"]
    out = {
        "basis": basis, "n_cells": len(cells), "eval_runners": int(len(ev)), "with_line": int(len(has)), "with_history": int(len(known)),
        "above": lift_table(known[known["above"]]), "below": lift_table(known[~known["above"]]),
        "no_history": lift_table(has[has["prior"].isna()]),
        "diff_win_lift_ci": cluster_bootstrap_diff(known, "above", "win", "exp_win"),
        "diff_top3_lift_ci": cluster_bootstrap_diff(known, "above", "top3", "exp_top3"),
        "share_above": round(float(known["above"].mean()), 3) if len(known) else None,
        "share_above_per_race_median": round(float(known.groupby("race_id")["above"].mean().median()), 3) if len(known) else None,
        "races_above": int(known[known["above"]]["race_id"].nunique()), "races_below": int(known[~known["above"]]["race_id"].nunique()),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/page_reference/tables")
    ap.add_argument("--years", default="2023,2024,2025", help="基準を作る年")
    ap.add_argument("--out", default="")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--build-years", default="2023,2024")
    ap.add_argument("--eval-year", type=int, default=2025)
    ap.add_argument("--report", default="")
    ap.add_argument("--basis", default="second", choices=list(BASES), help="基準にする馬（win=勝ち馬 / second=2着馬 / top2 / top3）")
    a = ap.parse_args(argv)
    tables = Path(a.tables)
    if a.evaluate:
        yrs = sorted({int(a.eval_year) - 4, *[int(y) for y in a.build_years.split(",")], int(a.eval_year)})
        yrs = [y for y in range(min(yrs), int(a.eval_year) + 1) if (tables / str(y) / "megu_index_flat.parquet").exists()]
        df = load(tables, yrs)
        by = [int(y) for y in a.build_years.split(",")]
        rep = {"build_years": a.build_years, "eval_year": a.eval_year,
               "stability_2023_2025": {b: stability(df, [2023, 2024, 2025], b) for b in BASES},
               **{b: evaluate(df, by, a.eval_year, basis=b) for b in BASES}}
        txt = json.dumps(rep, ensure_ascii=False, indent=1)
        (Path(a.report).write_text(txt, encoding="utf-8") if a.report else print(txt))
        return 0
    years = [int(y) for y in a.years.split(",")]
    df = load(tables, years)
    w = basis_rows(df[df["year"].isin(years)], a.basis)
    cells = build_cells(w)
    unread = df[df["tier"].isna()]
    doc = {
        "version": 1, "basis": a.basis, "index": VALUE, "period": f"{min(years)}-{max(years)}", "min_n": MIN_N, "split_delta": DELTA,
        "definition": f"突破ライン(line)=そのクラスの{BASIS_JA[a.basis]}の指数の下位25%（p25）。median/p75 は{BASIS_JA[a.basis]}の中央値・上位75%。面・距離帯は親との中央値の差が2点以上のときだけ分ける",
        "n_basis": int(len(w)), "n_races_unread_class": int(unread["race_id"].nunique()), "cells": cells,
    }
    txt = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(f"{a.out}: {len(cells)} cells, {len(w)} {a.basis}, 読めないクラス {unread['race_id'].nunique()} レース")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
