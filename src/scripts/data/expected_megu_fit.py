"""想定めぐ指数の『条件変化の補正』を学習し、config/expected_megu/model.json に書く（推論は src/pipeline/inference/expected_megu.py）。

目的変数 = その走の実際の絶対指数、基準 = 直前の過去走（直近 5 走）の加重平均。条件変化の特徴（面の切り替え、距離差、クラス差、休み明け、馬齢、
競馬場、斤量差）でリッジ回帰を作る。2020〜2024 年で学習（leave-one-year-out で係数の強さと特徴を選ぶ）、2025 年で一度だけ検証する。
データ（馬×レースの指数とレースの条件）は読み取りのみ。重みの候補（面の違い・距離の遠さで過去走を軽くする）も同じ手順で選ぶ。

  KEIBA_ENV=stg python -m src.scripts.data.expected_megu_fit --tables data/page_reference/tables --out config/expected_megu/model.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from src.pipeline.inference import expected_megu as em

FEATURES = ["to_dirt", "to_turf", "dist_up", "dist_down", "class_up", "class_down", "rest_long", "rest_short", "age", "venue_new", "weight_diff"]
WEIGHT_GRID = [(1.0, 1e9), (0.6, 1e9), (0.35, 1e9), (1.0, 1200.0), (0.6, 1200.0), (0.35, 600.0), (0.6, 600.0)]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
CAP = 8.0


def load_runs(tables: Path, years: list[int]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for y in years:
        m, _ = em._load_year(y, tables)
        for hid, rs in m.items():
            out.setdefault(hid, []).extend(rs)
    for rs in out.values():
        rs.sort(key=lambda r: r[0])
    return out


def build(runs_by_horse: dict[str, list[tuple]], target_years: set[int], wcfg: dict | None) -> dict:
    """目標の走ごとに、基準（加重平均）と特徴を作る。リーク無し（その走より前の走だけ）。"""
    rows = []
    for hid, rs in runs_by_horse.items():
        for t in range(1, len(rs)):
            cur = rs[t]
            y = int(cur[0][:4])
            if y not in target_years:
                continue
            prior = sorted((em._pad(r) for r in rs[:t]), key=lambda r: r[0], reverse=True)[: em.N_RUNS]
            today = {"surface": cur[3], "distance": cur[4], "venue": cur[5], "level": cur[6], "weight": cur[7], "age": em._pad(cur)[8]}
            w = em.run_weights(prior, today, wcfg)
            vals = [r[1] for r in prior]
            base = sum(wi * v for wi, v in zip(w, vals)) / sum(w)
            f = em.condition_features(prior, w, today, cur[0])
            if f is None:
                continue
            rows.append((y, cur[1], base, len(prior), f, em.change_score(f)))
    return {
        "year": np.array([r[0] for r in rows]), "y": np.array([r[1] for r in rows]), "base": np.array([r[2] for r in rows]),
        "n": np.array([r[3] for r in rows]), "score": np.array([r[5] for r in rows]),
        "X": np.array([[r[4][k] for k in FEATURES] for r in rows]) if rows else np.zeros((0, len(FEATURES))),
    }


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def ridge(X: np.ndarray, r: np.ndarray, alpha: float) -> tuple[np.ndarray, float, np.ndarray]:
    c = X.mean(axis=0)
    Xc = X - c
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(A, Xc.T @ (r - r.mean()))
    return coef, float(r.mean()), c


def loyo(d: dict, cols: list[int], alpha: float, years: list[int]) -> tuple[float, np.ndarray]:
    """leave-one-year-out の MAE と、その予測残差。"""
    resid = d["y"] - d["base"]
    pred = np.zeros(len(resid))
    for y in years:
        te = d["year"] == y
        tr = np.isin(d["year"], [yy for yy in years if yy != y])
        coef, icpt, c = ridge(d["X"][tr][:, cols], resid[tr], alpha)
        pred[te] = np.clip(icpt + (d["X"][te][:, cols] - c) @ coef, -CAP, CAP)
    m = np.isin(d["year"], years)
    return mae(d["y"][m], d["base"][m] + pred[m]), (resid - pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(em.hg.tables_dir()))
    ap.add_argument("--out", default=str(em.MODEL_PATH))
    ap.add_argument("--report", default="")
    ap.add_argument("--fit-years", default="2020-2024")
    ap.add_argument("--valid-year", type=int, default=2025)
    a = ap.parse_args()
    fy0, fy1 = (int(x) for x in a.fit_years.split("-"))
    fit_years = list(range(fy0, fy1 + 1))
    tables = Path(a.tables)
    t0 = time.time()
    runs = load_runs(tables, list(range(2020, a.valid_year + 1)))
    print(f"読み込み {sum(len(v) for v in runs.values())} 走 / {len(runs)} 頭 ({time.time() - t0:.0f}s)", flush=True)

    # 1) 重みの候補を選ぶ（学習年の基準の MAE。2025 年は使わない）
    scores = []
    for s, dsc in WEIGHT_GRID:
        wcfg = None if (s == 1.0 and dsc >= 1e8) else {"surface_mismatch": s, "distance_scale": dsc}
        d = build(runs, set(fit_years), wcfg)
        scores.append(((s, dsc), mae(d["y"], d["base"]), len(d["y"])))
        print(f"  重み s={s} d={dsc}: 基準 MAE {scores[-1][1]:.4f} (n={scores[-1][2]})", flush=True)
    best_w = min(scores, key=lambda x: x[1])[0]
    wcfg = None if (best_w[0] == 1.0 and best_w[1] >= 1e8) else {"surface_mismatch": best_w[0], "distance_scale": best_w[1]}

    d_all = build(runs, set(fit_years) | {a.valid_year}, wcfg)
    d0_all = build(runs, set(fit_years) | {a.valid_year}, None)
    fit = d_all["year"] <= fy1
    # 2) 係数の強さ（alpha）を LOYO で選び、特徴を後ろ向きに減らす（MAE が 0.001 以上悪化しない限り削る）
    cols = list(range(len(FEATURES)))
    best_a = min(ALPHAS, key=lambda al: loyo(d_all, cols, al, fit_years)[0])
    cur = loyo(d_all, cols, best_a, fit_years)[0]
    improved = True
    while improved and len(cols) > 1:
        improved = False
        for j in list(cols):
            trial = [c for c in cols if c != j]
            m = loyo(d_all, trial, best_a, fit_years)[0]
            if m <= cur + 0.001:
                cols, cur, improved = trial, min(cur, m), True
                break
    base_loyo = mae(d_all["y"][fit], d_all["base"][fit])
    print(f"alpha={best_a} 採用特徴={[FEATURES[c] for c in cols]} LOYO MAE {cur:.4f} (基準 {base_loyo:.4f})", flush=True)
    # 3) 学習年全体で最終係数、LOYO 残差から条件変化の群ごとの誤差 SD
    resid = d_all["y"] - d_all["base"]
    coef, icpt, c = ridge(d_all["X"][fit][:, cols], resid[fit], best_a)
    _, loyo_resid = loyo(d_all, cols, best_a, fit_years)
    th = {"some": 1.0, "large": 3.0}
    grp = np.where(d_all["score"] >= th["large"], 2, np.where(d_all["score"] >= th["some"], 1, 0))
    resid_sd = {name: float(np.std(loyo_resid[fit & (grp == i)])) for i, name in enumerate(["none", "some", "large"]) if (fit & (grp == i)).sum() > 50}
    # 4) 2025 年での検証（一度だけ）
    v = d_all["year"] == a.valid_year
    pred = np.clip(icpt + (d_all["X"][v][:, cols] - c) @ coef, -CAP, CAP)
    y, b0, b1 = d_all["y"][v], d0_all["base"][v] if len(d0_all["base"]) == len(d_all["base"]) else d_all["base"][v], d_all["base"][v]
    report = {"n_valid": int(v.sum()), "overall": {"plain_mae": mae(y, b0), "weighted_mae": mae(y, b1), "adjusted_mae": mae(y, b1 + pred),
                                                    "plain_corr": corr(y, b0), "adjusted_corr": corr(y, b1 + pred)}, "by_group": {}, "by_case": {}}
    gv = grp[v]
    for i, name in enumerate(["none", "some", "large"]):
        m = gv == i
        if m.sum() > 30:
            report["by_group"][name] = {"n": int(m.sum()), "plain_mae": mae(y[m], b0[m]), "adjusted_mae": mae(y[m], (b1 + pred)[m]),
                                        "adjusted_corr": corr(y[m], (b1 + pred)[m]),
                                        "within_range_rate": float(np.mean(np.abs(y[m] - (b1 + pred)[m]) <= resid_sd.get(name, 6.0)))}
    Xv = d_all["X"][v]
    for label, mask in (("ダート替わり(初)", Xv[:, FEATURES.index("to_dirt")] >= 0.99), ("芝替わり(初)", Xv[:, FEATURES.index("to_turf")] >= 0.99),
                        ("距離±400m以上", (Xv[:, FEATURES.index("dist_up")] >= 4) | (Xv[:, FEATURES.index("dist_down")] >= 4)),
                        ("昇級1段以上", Xv[:, FEATURES.index("class_up")] >= 1), ("降級1段以上", Xv[:, FEATURES.index("class_down")] >= 1),
                        ("休み明け120日以上", Xv[:, FEATURES.index("rest_long")] >= 1), ("初めての競馬場", Xv[:, FEATURES.index("venue_new")] >= 1)):
        if mask.sum() > 30:
            report["by_case"][label] = {"n": int(mask.sum()), "plain_mae": mae(y[mask], b0[mask]), "adjusted_mae": mae(y[mask], (b1 + pred)[mask])}
    model = {
        "version": "v1", "trained_on": f"{fy0}-{fy1}", "validated_on": str(a.valid_year),
        "weights": wcfg or {}, "features": [FEATURES[c] for c in cols],
        "coef": {FEATURES[c]: float(k) for c, k in zip(cols, coef)}, "center": {FEATURES[c]: float(x) for c, x in zip(cols, c)},
        "intercept": icpt, "cap": CAP, "alpha": best_a, "group_thresholds": th, "resid_sd": resid_sd,
        "eval": {"n_valid": report["n_valid"], **{k: round(x, 4) for k, x in report["overall"].items()}},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(model, ensure_ascii=False, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"完了 {time.time() - t0:.0f}s → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
