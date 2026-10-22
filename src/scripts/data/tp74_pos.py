"""tp-74: 追走位置 × コース × 馬場 × ペースで好走率の条件が変わるか（変わる分を現行モデルと市場が織り込んでいるか）。

python tp74_pos.py prereg   # 型の定義と数だけを出力（結果を見る前に実行してコミットする）
python tp74_pos.py eval     # 検定・交互作用・標本外の評価
python tp74_pos.py figs     # 図

入力（ホスト）: tp72_out/mining.parquet（1 出走 1 行。p_top3 等は V4 の OOF、2022〜2026）、
  tp54_out/horse_ease.parquet（pf_early = 位置取り予想。馬柱だけで決まる、枠補正込み、2020〜2025）、
  race_quality.parquet（実測のペース・隊列と、事前のペース確率 p_joint）。
位置取りの 2 つの形：
  P（事前に使える形）= pf_early の帯（逃げ ≤0.15 / 先行 ≤0.35 / 差し ≤0.65 / 追込）。ペースは事前確率 P(ハイ)−P(スロー) の三分位。
  O（オラクル）= 今走の実際の初角・4 角の帯と、実測のペース。レース後にしか分からない。仕組みの確認用で、買い目の根拠にはしない。
市場オッズ・人気は特徴量にしない（回収率の計算にだけ払戻を使う）。発見 2022〜2024、確認 2025 を 1 回、2026 は記述のみ。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("TP74_OUT", "/home/jovyan/work/tp74_out"))
MINING = Path(os.environ.get("TP72_OUT", "/home/jovyan/work/tp72_out")) / "mining.parquet"
EASE = Path(os.environ.get("TP54_OUT", "/home/jovyan/work/tp54_out")) / "horse_ease.parquet"
RQ = Path(os.environ.get("TP74_RQ", "/home/jovyan/work/keiba-vpn/data/local/modeling/shared_cache/race_quality/race_quality.parquet"))

DISC = (2022, 2023, 2024)
MIN_RUNS = 600            # 型として検定する最小の出走数（発見期間）。結果は見ずに、件数だけで決める
BAND_NAMES = ["逃げ", "先行", "差し", "追込"]
K_SHRINK = 200.0          # 階層縮約の強さ（出走数）。固定
PACE_NAMES = {"slow": "スロー", "mid": "ミドル", "high": "ハイ"}

# ───────────────────────── 型の定義（結果を見る前に固定）
# (family, band 列, 条件の列のタプル)
SPECS: list[tuple[str, str, tuple[str, ...]]] = []
for c in ((), ("c5",), ("cv",), ("g3",), ("s3",), ("pp3",), ("fs3",), ("c5", "pp3"), ("c5", "g3"), ("sf", "g3", "pp3"), ("sf", "s3", "pp3"), ("c5", "s3"), ("cv", "pp3")):
    SPECS.append(("P", "bp", c))
for c in ((), ("c5",), ("g3",), ("s3",), ("pace3",), ("shape3",), ("joint9",), ("c5", "pace3"), ("sf", "g3", "pace3"), ("sf", "s3", "pace3"), ("c5", "g3")):
    SPECS.append(("O1", "ba1", c))
for c in ((), ("c5",), ("g3",), ("pace3",), ("c5", "pace3"), ("sf", "g3", "pace3")):
    SPECS.append(("O4", "ba4", c))
for b, c in (("chg3", ()), ("chg3", ("pace3",)), ("chg3", ("c5",)), ("chg3", ("g3",)), ("bp", ("chg3",)), ("bp", ("chg3", "pace3"))):
    SPECS.append(("M", b, c))

LADDER_P = [("0 要素（位置の帯のみ）", ()), ("+コース", ("c5",)), ("+コース+馬場", ("c5", "g3")), ("+コース+ペース予想", ("c5", "pp3")),
            ("+コース+馬場+ペース予想", ("c5", "g3", "pp3")), ("+コース+馬場+ペース予想+硬さ", ("c5", "g3", "pp3", "s3")), ("ペース予想のみ", ("pp3",))]
LADDER_O = [("0 要素（位置の帯のみ）", ()), ("+コース", ("c5",)), ("+コース+馬場", ("c5", "g3")), ("+コース+ペース実測", ("c5", "pace3")),
            ("+コース+馬場+ペース実測", ("c5", "g3", "pace3")), ("+コース+馬場+ペース実測+硬さ", ("c5", "g3", "pace3", "s3")), ("ペース実測のみ", ("pace3",))]

HET_P = [("c5", "コース（面×距離帯）"), ("cv", "競馬場×面"), ("g3", "馬場状態"), ("s3f", "硬さ（面別）"), ("pp3", "ペース予想"), ("fs3", "頭数"), ("sfg3pp3", "面×馬場×ペース予想")]
HET_O = [("c5", "コース（面×距離帯）"), ("g3", "馬場状態"), ("pace3", "ペース実測"), ("shape3", "隊列実測"), ("joint9", "ペース×隊列"), ("sfg3pace3", "面×馬場×ペース実測")]


def band4(x: pd.Series) -> pd.Series:
    v = np.select([x <= 0.15, x <= 0.35, x <= 0.65], [0, 1, 2], 3).astype(float)
    return pd.Series(np.where(x.isna(), np.nan, v), index=x.index)


def load() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "year", "venue_code", "surface", "dist_band", "track_condition", "fs", "cush_bin", "dmoist_bin", "top3", "win", "p_top3", "p_win",
            "ret_place", "ret_win", "pos1_frac", "posL_frac", "style5"]
    m = pd.read_parquet(MINING, columns=cols)
    m = m[m["year"] >= 2022].copy()
    m["race_id"] = m["race_id"].astype(str)
    m["horse_id"] = m["horse_id"].astype(str)
    e = pd.read_parquet(EASE, columns=["race_id", "horse_id", "pf_early"])
    e["race_id"] = e["race_id"].astype(str)
    e["horse_id"] = e["horse_id"].astype(str)
    m = m.merge(e.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    m["pf_src"] = np.where(m["pf_early"].notna(), "pf", np.where(m["year"] == 2026, "style5", "none"))
    m["pf_use"] = m["pf_early"].where(m["pf_early"].notna(), np.where(m["year"] == 2026, m["style5"], np.nan))
    q = pd.read_parquet(RQ)
    q["race_id"] = q["race_id"].astype(str)
    ph = q[[c for c in q.columns if c.startswith("p_joint_high")]].sum(axis=1, min_count=1)
    ps = q[[c for c in q.columns if c.startswith("p_joint_slow")]].sum(axis=1, min_count=1)
    q["pp"] = ph - ps
    ref = q.loc[q["year"] == 2021, "pp"].dropna()
    lo, hi = ref.quantile([1 / 3, 2 / 3]).to_numpy()
    q["pp3"] = np.where(q["pp"].isna(), np.nan, np.where(q["pp"] >= hi, 2, np.where(q["pp"] <= lo, 0, 1)))
    q["pace3"] = q["pace_cls"].map({"slow": 0, "mid": 1, "high": 2})
    q["shape3"] = q["shape_cls"].map({"compact": 0, "normal": 1, "stretched": 2})
    m = m.merge(q[["race_id", "pp3", "pace3", "shape3"]].drop_duplicates("race_id"), on="race_id", how="left")
    m["joint9"] = m["pace3"] * 3 + m["shape3"]
    m["bp"] = band4(m["pf_use"])
    m["ba1"] = band4(m["pos1_frac"])
    m["ba4"] = band4(m["posL_frac"])
    d = m["pos1_frac"] - m["pf_use"]
    m["chg3"] = np.where(d.isna(), np.nan, np.where(d <= -0.10, 0, np.where(d >= 0.10, 2, 1)))
    m["g3"] = m["track_condition"].map({"良": 0, "稍重": 1, "重": 2, "不良": 2})
    m["s3"] = np.where(m["surface"] == "芝", m["cush_bin"], m["dmoist_bin"])
    m["fs3"] = np.where(m["fs"] <= 12, 0, np.where(m["fs"] <= 15, 1, 2))
    m["c5"] = m["surface"] + "|" + m["dist_band"].astype(str)
    m["cv"] = m["venue_code"].astype(str) + "|" + m["surface"]
    m["sf"] = m["surface"]
    m["s3f"] = np.where(pd.isna(m["s3"]), np.nan, m["surface"] + "|" + pd.Series(m["s3"]).astype("Float64").astype(str))
    m["sfg3pp3"] = np.where(m[["g3", "pp3"]].isna().any(axis=1), np.nan, m["surface"] + "|" + m["g3"].astype("Float64").astype(str) + "|" + m["pp3"].astype("Float64").astype(str))
    m["sfg3pace3"] = np.where(m[["g3", "pace3"]].isna().any(axis=1), np.nan, m["surface"] + "|" + m["g3"].astype("Float64").astype(str) + "|" + m["pace3"].astype("Float64").astype(str))
    m["t3"] = m["top3"].astype(float)
    m["w"] = m["win"].astype(float)
    m["r3"] = m["t3"] - m["p_top3"]
    m["rw"] = m["w"] - m["p_win"]
    m["rp"] = m["ret_place"].astype(float)
    m["rwin"] = m["ret_win"].astype(float)
    m["rid"] = pd.factorize(m["race_id"])[0]
    return m.dropna(subset=["p_top3"]).reset_index(drop=True)


def cell_id(row_or_df, band: str, conds: tuple[str, ...]) -> list[str]:
    return [band, *conds]


def cell_stats(d: pd.DataFrame, keys: list[str], val: str, baseline: float = 0.0) -> pd.DataFrame:
    """セルごとの平均（val − baseline）と、レース単位のクラスタ頑健な標準誤差。"""
    x = d.dropna(subset=keys + [val]).copy()
    x["_v"] = x[val] - baseline
    g = x.groupby(keys + ["rid"], observed=True)["_v"].agg(["sum", "count"]).reset_index()
    t = g.groupby(keys, observed=True).agg(S=("sum", "sum"), N=("count", "sum"), races=("rid", "nunique")).reset_index()
    g = g.merge(t[keys + ["S", "N"]], on=keys)
    g["dev"] = (g["sum"] - g["S"] / g["N"] * g["count"]) ** 2
    se = g.groupby(keys, observed=True)["dev"].sum().pow(0.5).reset_index(name="num")
    t = t.merge(se, on=keys)
    t["mean"] = t["S"] / t["N"]
    t["se"] = t["num"] / t["N"]
    return t[keys + ["N", "races", "mean", "se"]]


def norm_p2(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def norm_p1(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def bh(ps: list[float], q: float = 0.05) -> list[bool]:
    n = len(ps)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: ps[i])
    k = 0
    for r, i in enumerate(order, 1):
        if ps[i] <= q * r / n:
            k = r
    keep = set(order[:k])
    return [i in keep for i in range(n)]


def spec_label(band: str, conds: tuple[str, ...]) -> str:
    return "×".join([band, *conds])


def enumerate_cells(m: pd.DataFrame) -> list[dict]:
    """型 = 位置の帯 × 条件の水準の組。発見期間の出走数が MIN_RUNS 以上のものだけ（結果は見ない）。"""
    out = []
    disc = m[m["year"].isin(DISC)]
    for fam, band, conds in SPECS:
        keys = [band, *conds]
        x = disc.dropna(subset=keys)
        if "bp" in keys:
            x = x[x["pf_src"] == "pf"]
        g = x.groupby(keys, observed=True)["rid"].agg(["size", "nunique"]).reset_index()
        for _, r in g.iterrows():
            if r["size"] >= MIN_RUNS:
                out.append({"family": fam, "spec": spec_label(band, conds), "band": band, "conds": list(conds), "key": [r[k] if not isinstance(r[k], (np.floating, float)) else float(r[k]) for k in keys],
                            "n_disc": int(r["size"]), "races_disc": int(r["nunique"])})
    return out


def prereg() -> None:
    m = load()
    cells = enumerate_cells(m)
    fam = pd.Series([c["family"] for c in cells]).value_counts().to_dict()
    spec_counts = pd.Series([c["spec"] for c in cells]).value_counts().to_dict()
    pre = {"note": "結果を見る前に件数だけで定義した型。位置の帯 × 条件の水準の組で、発見期間(2022-2024)の出走が 600 以上のものを検定する。BH は family ごと。",
           "min_runs": MIN_RUNS, "specs": [{"family": f, "band": b, "conds": list(c)} for f, b, c in SPECS], "n_cells_by_family": fam, "n_cells_by_spec": spec_counts,
           "n_cells_total": len(cells), "tests_per_cell": ["top3 残差(両側)", "複勝回収率>100%(片側)"],
           "ladder_P": [(a, list(b)) for a, b in LADDER_P], "ladder_O": [(a, list(b)) for a, b in LADDER_O],
           "heterogeneity_P": HET_P, "heterogeneity_O": HET_O,
           "n_tests_primary": 2 * len(cells), "rows": int(len(m)), "disc_rows": int(m["year"].isin(DISC).sum()),
           "cuts": {"band": [0.15, 0.35, 0.65], "chg3": [-0.10, 0.10], "pp3": "2021 年のレース単位の P(ハイ)−P(スロー) の三分位", "k_shrink": K_SHRINK}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg.json").write_text(json.dumps(pre, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: pre[k] for k in ("n_cells_by_family", "n_cells_total", "n_tests_primary", "rows", "disc_rows")}, ensure_ascii=False))


# ───────────────────────── 評価
def _kt(vals) -> tuple:
    return tuple(float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) for v in vals)


def eval_cells(m: pd.DataFrame, cells: list[dict]) -> list[dict]:
    res = []
    by_spec: dict[tuple, list[dict]] = {}
    for c in cells:
        by_spec.setdefault((c["family"], c["band"], tuple(c["conds"])), []).append(c)
    periods = {"disc": list(DISC), "y2022": [2022], "y2023": [2023], "y2024": [2024], "conf2025": [2025], "d2026": [2026]}
    mets = {"r3": ("r3", 0.0), "rw": ("rw", 0.0), "rp": ("rp", 100.0), "t3": ("t3", 0.0), "rwin": ("rwin", 100.0)}
    for (fam, band, conds), cs in by_spec.items():
        keys = [band, *conds]
        needs_pf = "bp" in keys
        lookup: dict[str, dict[str, dict]] = {}
        for pn, ys in periods.items():
            d = m[m["year"].isin(ys)]
            if needs_pf:
                d = d[d["bp"].notna()] if pn == "d2026" else d[d["pf_src"] == "pf"]
            if len(d) == 0:
                continue
            lookup[pn] = {}
            for met, (col, bl) in mets.items():
                t = cell_stats(d, keys, col, baseline=bl)
                lookup[pn][met] = {_kt(r[keys].tolist()): [int(r["N"]), float(r["mean"]), float(r["se"])] for _, r in t.iterrows()}
        for c in cs:
            row = {k: c[k] for k in ("family", "spec", "band", "conds", "key", "n_disc", "races_disc")}
            kk = _kt(c["key"])
            for pn in lookup:
                for met in mets:
                    v = lookup[pn][met].get(kk)
                    if v is not None:
                        row[f"{pn}_{met}"] = v
            res.append(row)
    return res


def add_tests(res: list[dict]) -> None:
    for fam in sorted({r["family"] for r in res}):
        rows = [r for r in res if r["family"] == fam and "disc_r3" in r]
        p3 = [norm_p2(r["disc_r3"][1] / r["disc_r3"][2]) if r["disc_r3"][2] > 0 else 1.0 for r in rows]
        pr = [norm_p1(r["disc_rp"][1] / r["disc_rp"][2]) if r["disc_rp"][2] > 0 else 1.0 for r in rows]    # 回収率 > 100% の片側検定
        b3 = bh(p3)
        br = bh(pr)
        for r, a, b, c1, c2 in zip(rows, p3, pr, b3, br):
            r["p_r3"], r["p_roi"], r["bh_r3"], r["bh_roi"] = a, b, bool(c1), bool(c2)
            same = [r.get(f"y{y}_r3", [0, 0, 1])[1] for y in (2022, 2023, 2024)]
            r["loyo_same_sign"] = bool(all(x > 0 for x in same) or all(x < 0 for x in same))


# 階層縮約の標本外評価
def fit_chain(train: pd.DataFrame, band: str, conds: tuple[str, ...], val: str, base: pd.Series | float, k: float = K_SHRINK):
    keys_levels = [[band]] + [[band, *conds[:j]] for j in range(1, len(conds) + 1)]
    tr = train.copy()
    tr["_res"] = tr[val] - (base if np.isscalar(base) else base.loc[tr.index])
    est_prev = None
    tables = []
    for keys in keys_levels:
        x = tr.dropna(subset=keys)
        g = x.groupby(keys, observed=True)["_res"].agg(["sum", "count"]).reset_index()
        if est_prev is None:
            g["est"] = g["sum"] / (g["count"] + k)
        else:
            pk = keys_levels[len(tables) - 1]
            g = g.merge(est_prev.rename(columns={"est": "_parent"}), on=pk, how="left")
            g["_parent"] = g["_parent"].fillna(0.0)
            g["est"] = (g["sum"] + k * g["_parent"]) / (g["count"] + k)
        est_prev = g[keys + ["est"]]
        tables.append(est_prev.copy())
    return keys_levels, tables


def apply_chain(test: pd.DataFrame, keys_levels, tables) -> np.ndarray:
    off = np.zeros(len(test))
    for keys, t in zip(keys_levels, tables):
        mg = test[keys].merge(t, on=keys, how="left")
        e = mg["est"].to_numpy()
        ok = ~np.isnan(e)
        off = np.where(ok, e, off)
    return off


def ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def oos(m: pd.DataFrame, band: str, conds: tuple[str, ...], train_years: list[int], test_year: int, mode: str) -> dict:
    """mode='inc': 現行モデルの確率 p_top3 に足す補正の効果。mode='raw': 全体の平均に対するセルの説明力（モデルを使わない）。"""
    d = m[m["bp"].notna()] if band == "bp" else m[m[band].notna()]
    train = d[d["year"].isin(train_years)].reset_index(drop=True)
    test = d[d["year"] == test_year].reset_index(drop=True)
    if len(test) == 0 or len(train) == 0:
        return {}
    if mode == "inc":
        base_tr = train["p_top3"]
        kl, tb = fit_chain(train, band, conds, "t3", base_tr)
        off = apply_chain(test, kl, tb)
        p_old = test["p_top3"].to_numpy()
        p_new = np.clip(p_old + off, 1e-4, 1 - 1e-4)
    else:
        mu = float(train["t3"].mean())
        kl, tb = fit_chain(train, band, conds, "t3", mu)
        off = apply_chain(test, kl, tb)
        p_old = np.full(len(test), mu)
        p_new = np.clip(mu + off, 1e-4, 1 - 1e-4)
    y = test["t3"].to_numpy()
    diff = ll(y, p_new) - ll(y, p_old)
    s = pd.DataFrame({"rid": test["rid"], "d": diff}).groupby("rid")["d"].agg(["sum", "count"])
    n = len(test)
    mean = float(diff.mean())
    se = float(np.sqrt(((s["sum"] - mean * s["count"]) ** 2).sum()) / n)
    return {"n": n, "d_logloss": mean, "se": se, "ll_old": float(ll(y, p_old).mean())}


def heterogeneity(m: pd.DataFrame, which: str, items: list[tuple[str, str]]) -> list[dict]:
    """位置の帯（逃げ・先行 vs 差し・追込）の効きが条件の水準で違うか。生の好走率と、現行モデルの残差の両方で（レース単位クラスタのワルド検定）。"""
    import statsmodels.formula.api as smf

    bcol = "bp" if which == "P" else "ba1"
    d = m[m["year"].isin(DISC) & m[bcol].notna()].copy()
    if which == "P":
        d = d[d["pf_src"] == "pf"]
    d["front"] = (d[bcol] <= 1).astype(float)
    out = []
    for col, label in items:
        x = d.dropna(subset=[col]).copy()
        x["X"] = x[col].astype(str)
        if x["X"].nunique() < 2:
            continue
        row = {"cond": col, "label": label, "levels": int(x["X"].nunique()), "n": int(len(x))}
        gaps = {}
        for lv, g in x.groupby("X"):
            a, b = g[g["front"] == 1], g[g["front"] == 0]
            if len(a) >= 200 and len(b) >= 200:
                gaps[lv] = {"t3_gap": float(a["t3"].mean() - b["t3"].mean()), "r3_gap": float(a["r3"].mean() - b["r3"].mean()), "n": int(len(g))}
        row["gaps"] = gaps
        for yv in ("t3", "r3"):
            try:
                fit = smf.ols(f"{yv} ~ front * C(X)", data=x).fit(cov_type="cluster", cov_kwds={"groups": x["rid"].to_numpy()})
                inter = [n for n in fit.params.index if n.startswith("front:")]
                w = fit.wald_test(" , ".join(f"{n} = 0" for n in inter), scalar=True)
                row[f"wald_{yv}"] = {"stat": float(w.statistic), "p": float(w.pvalue), "df": len(inter)}
            except Exception as exc:  # noqa: BLE001
                row[f"wald_{yv}"] = {"error": str(exc)[:80]}
        out.append(row)
    return out


def run_eval() -> None:
    m = load()
    pre = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    cells = enumerate_cells(m)
    assert len(cells) == pre["n_cells_total"], (len(cells), pre["n_cells_total"])
    res = eval_cells(m, cells)
    add_tests(res)
    (OUT / "cells.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    out: dict = {"n_cells": len(res)}
    # 標本外
    ladder = {}
    for which, lad, band in (("P", LADDER_P, "bp"), ("O1", LADDER_O, "ba1"), ("O4", LADDER_O, "ba4")):
        rows = []
        for lab, conds in lad:
            r = {"label": lab, "conds": list(conds)}
            for mode in ("inc", "raw"):
                r[mode] = {"y2024": oos(m, band, conds, [2022, 2023], 2024, mode), "y2025": oos(m, band, conds, [2022, 2023, 2024], 2025, mode), "y2026": oos(m, band, conds, [2022, 2023, 2024], 2026, mode)}
            rows.append(r)
        ladder[which] = rows
    out["ladder"] = ladder
    out["het_P"] = heterogeneity(m, "P", HET_P)
    out["het_O"] = heterogeneity(m, "O", HET_O)
    # 帯ごとの全体の表
    def band_table(col: str, sub: pd.DataFrame) -> list[dict]:
        g = sub.dropna(subset=[col]).groupby(col).agg(n=("t3", "size"), t3=("t3", "mean"), p=("p_top3", "mean"), w=("w", "mean"), pw=("p_win", "mean"), rp=("rp", "mean"), rwin=("rwin", "mean")).reset_index()
        return g.to_dict("records")
    out["bands"] = {"P_disc": band_table("bp", m[m["year"].isin(DISC) & (m["pf_src"] == "pf")]), "P_2025": band_table("bp", m[(m["year"] == 2025) & (m["pf_src"] == "pf")]),
                    "O1_disc": band_table("ba1", m[m["year"].isin(DISC)]), "O4_disc": band_table("ba4", m[m["year"].isin(DISC)]), "P_2026": band_table("bp", m[m["year"] == 2026])}
    out["coverage"] = {"pf_cov": float((m[m["year"].between(2022, 2025)]["pf_src"] == "pf").mean()), "pp3_cov": float(m["pp3"].notna().mean()), "pace3_cov": float(m["pace3"].notna().mean()),
                       "s3_cov": float(m["s3"].notna().mean()), "rows": int(len(m))}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("done", out["coverage"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prereg"
    {"prereg": prereg, "eval": run_eval}[cmd]()
