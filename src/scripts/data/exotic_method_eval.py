"""券種別の的中確率の計算法を、実着順で比べる（馬単・馬連・ワイド・3連複・3連単・複勝）。

入力: 既存の OOF（race_id, horse_id, year, p_win, p_top2, p_top3。2022〜2024、2026 年は使わない）と実着順。
方法:
  H   Harville（勝率だけ。α=1）
  DG  割引 Harville・全レース共通 α（学習年の最尤。LOYO）
  D1  割引 Harville・レースごとに連対率・複勝率へ α を当てはめ（現行）
  M3  周辺確率一致の Plackett–Luce（2・3 着目の強さを馬ごとに連対率・複勝率へ合わせる）
  M3c M3 の入力（勝率・連対率・複勝率）を LOYO で Platt 較正してから使う
  T   Thurstone（正規）型の着順分布（Monte Carlo、部分標本のみ。参考）
評価: 排他的な結果の対数スコア（−log P(実際の組合せ)）と、全組合せの Brier / 二値対数損失、較正曲線（予測確率の帯ごとの実測）。
同着（1〜3 着の着順が 1,2,3 と重複なく並ばないレース）は除く。複勝は 8 頭以上で 3 着以内、7 頭以下は 2 着以内。

    python -m src.scripts.data.exotic_method_eval run
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.pipeline.inference.exotic_probabilities import (  # noqa: E402
    RankDistribution,
    fit_marginal_distribution,
    fit_rank_distribution,
    normalize_marginal,
)

OOF = ROOT / "data/local/modeling/training_pipeline_v4/v6/eval_rev/oof_z:RB@e.parquet"
OUT = Path(os.environ.get("EXOTIC_EVAL_OUT", ROOT / "data/local/modeling/exotic_eval"))
EPS = 1e-12
BINS = np.concatenate([[0.0], np.logspace(-4, 0, 41)])  # 予測確率の帯（対数刻み）
BETS = ["fukusho", "umatan", "umaren", "wide", "sanrenpuku", "sanrentan"]
METHODS = ["H", "DG", "D1", "M3", "M3c", "MB30", "MB50", "MB60", "MB70"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_COMB: dict[tuple[int, int], np.ndarray] = {}


def _combs(n: int, k: int) -> np.ndarray:
    if (n, k) not in _COMB:
        _COMB[(n, k)] = np.array(list(combinations(range(n), k)), dtype=int)
    return _COMB[(n, k)]


def load_races() -> list[dict]:
    from src.scripts.data import tp27_eval as t

    _, _, meta, _, _ = t.setup()
    meta = meta[["race_id", "horse_id", "finish_position"]].copy()
    o = pd.read_parquet(OOF)
    for d in (o, meta):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    m = o.merge(meta, on=["race_id", "horse_id"], how="left")
    m["finish_position"] = pd.to_numeric(m["finish_position"], errors="coerce")
    races, skipped = [], {"dead_heat_or_missing": 0, "small": 0}
    for rid, g in m.groupby("race_id", sort=True):
        n = len(g)
        if n < 4:
            skipped["small"] += 1
            continue
        rk = g["finish_position"].to_numpy(dtype=float)
        order = np.argsort(np.where(np.isnan(rk) | (rk <= 0), 1e9, rk), kind="stable")
        top = rk[order[:3]]
        if not np.array_equal(top, [1.0, 2.0, 3.0]) or np.sum(np.isin(rk, [1.0, 2.0, 3.0])) != 3:
            skipped["dead_heat_or_missing"] += 1
            continue
        races.append({
            "race_id": rid, "year": int(g["year"].iloc[0]), "n": n,
            "win": g["p_win"].to_numpy(float), "top2": g["p_top2"].to_numpy(float), "top3": g["p_top3"].to_numpy(float),
            "abc": order[:3].copy(), "rank": np.where(np.isnan(rk) | (rk <= 0), 99.0, rk),
        })
    _log(f"レース {len(races)}（除外: {skipped}）")
    cap = int(os.environ.get("EXOTIC_EVAL_MAX", "0") or 0)
    return races[::max(len(races) // cap, 1)][:cap] if cap else races


# ── 周辺確率の Platt 較正（帯集計でロジスティック回帰） ──
def _platt_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize

    p = np.clip(p, 1e-6, 1 - 1e-6)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, 101)))
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    n = np.bincount(idx, minlength=len(qs) - 1).astype(float)
    k = np.bincount(idx, weights=y, minlength=len(qs) - 1)
    pm = np.bincount(idx, weights=p, minlength=len(qs) - 1) / np.maximum(n, 1)
    ok = n > 0
    x = np.log(pm[ok] / (1 - pm[ok]))
    nn, kk = n[ok], k[ok]

    def nll(ab):
        z = ab[0] + ab[1] * x
        return float(np.sum(nn * np.logaddexp(0, z) - kk * z))

    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9})
    return float(r.x[0]), float(r.x[1])


def _platt_apply(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = ab[0] + ab[1] * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z))


def marginal_calibrators(races: list[dict], years: list[int]) -> dict[int, dict[str, tuple[float, float]]]:
    """各年を除いた年で、勝率・連対率・複勝率の Platt 係数を当てはめる（LOYO）。"""
    out = {}
    for y in years:
        tr = [r for r in races if r["year"] != y]
        cal = {}
        for key, kmax in (("win", 1), ("top2", 2), ("top3", 3)):
            p = np.concatenate([r[key] for r in tr])
            yy = np.concatenate([(r["rank"] <= kmax).astype(float) for r in tr])
            cal[key] = _platt_fit(p, yy)
        out[y] = cal
    return out


def global_alpha(races: list[dict]) -> tuple[float, float]:
    """共通 α2, α3 の最尤（Benter 型。勝率を worth とし、実際の 2・3 着の条件付き確率を最大化）。"""
    from scipy.optimize import minimize_scalar

    def ll2(a):
        s = 0.0
        for r in races:
            w = r["win"] / r["win"].sum()
            i, j = r["abc"][0], r["abc"][1]
            l = w ** a
            s += np.log(l[j]) - np.log(l.sum() - l[i])
        return -s

    def ll3(a):
        s = 0.0
        for r in races:
            w = r["win"] / r["win"].sum()
            i, j, k = r["abc"]
            l = w ** a
            s += np.log(l[k]) - np.log(l.sum() - l[i] - l[j])
        return -s

    a2 = minimize_scalar(ll2, bounds=(0.2, 1.5), method="bounded").x
    a3 = minimize_scalar(ll3, bounds=(0.2, 1.5), method="bounded").x
    return float(a2), float(a3)


def make_dist(method: str, r: dict, ga: tuple[float, float] | None, cal: dict | None):
    win, t2, t3 = r["win"], r["top2"], r["top3"]
    if method == "H":
        return RankDistribution(win, 1.0, 1.0)
    if method == "DG":
        return RankDistribution(win, ga[0], ga[1])
    if method == "D1":
        return fit_rank_distribution(win, t2, t3)[0]
    if method == "M3":
        return fit_marginal_distribution(win, t2, t3)[0]
    if method == "M3c":
        w = _platt_apply(win, cal["win"])
        a = _platt_apply(t2, cal["top2"])
        b = _platt_apply(t3, cal["top3"])
        return fit_marginal_distribution(w, a, b)[0]
    if method.startswith("MB"):  # M3 と共通 α の対数線形ブレンド（θ=MBxx/100。θ=1 で M3、0 で DG）
        th = int(method[2:]) / 100.0
        d3 = fit_marginal_distribution(win, t2, t3)[0]
        w = d3.win
        l2 = np.exp(th * np.log(np.clip(d3.lam2, 1e-12, None)) + (1 - th) * ga[0] * np.log(w))
        l3 = np.exp(th * np.log(np.clip(d3.lam3, 1e-12, None)) + (1 - th) * ga[1] * np.log(w))
        return RankDistribution(w, 1.0, 1.0, lam2=l2, lam3=l3)
    raise KeyError(method)


def race_tables(d: RankDistribution, r: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """券種ごとの (予測確率, 的中 0/1) の組合せ配列と、排他的な的中確率を返す。"""
    n = r["n"]
    a, b, c = (int(x) for x in r["abc"])
    rank = r["rank"]
    p2, p3 = d._p2, d._p3
    out: dict[str, tuple] = {}
    # 複勝（馬ごと）
    k = 3 if n >= 8 else 2
    pf = d.top3() if k == 3 else d.top2()
    out["fukusho"] = (pf, (rank <= k).astype(float))
    # 馬単（順序付きペア）
    ii, jj = np.where(~np.eye(n, dtype=bool))
    y = ((ii == a) & (jj == b)).astype(float)
    out["umatan"] = (p2[ii, jj], y)
    # 馬連・ワイド
    pr = _combs(n, 2)
    ui, uj = pr[:, 0], pr[:, 1]
    um = p2[ui, uj] + p2[uj, ui]
    yu = (((ui == a) & (uj == b)) | ((ui == b) & (uj == a))).astype(float)
    out["umaren"] = (um, yu)
    tr = _combs(n, 3)
    ti, tj, tk = tr[:, 0], tr[:, 1], tr[:, 2]
    tri = p3[ti, tj, tk] + p3[ti, tk, tj] + p3[tj, ti, tk] + p3[tj, tk, ti] + p3[tk, ti, tj] + p3[tk, tj, ti]
    ytri = (np.isin(ti, [a, b, c]) & np.isin(tj, [a, b, c]) & np.isin(tk, [a, b, c])).astype(float)
    out["sanrenpuku"] = (tri, ytri)
    # ワイド: 各ペアについて、その 2 頭を含む 3 連複の和
    full = np.zeros((n, n))
    np.add.at(full, (ti, tj), tri)
    np.add.at(full, (ti, tk), tri)
    np.add.at(full, (tj, tk), tri)
    pw = full[ui, uj]
    top3set = {a, b, c}
    yw = (np.isin(ui, list(top3set)) & np.isin(uj, list(top3set))).astype(float)
    out["wide"] = (pw, yw)
    # 3連単
    if n >= 3:
        pt = _combs(n, 3)
        perms = np.concatenate([pt[:, [0, 1, 2]], pt[:, [0, 2, 1]], pt[:, [1, 0, 2]], pt[:, [1, 2, 0]], pt[:, [2, 0, 1]], pt[:, [2, 1, 0]]])
        ps = p3[perms[:, 0], perms[:, 1], perms[:, 2]]
        ys = ((perms[:, 0] == a) & (perms[:, 1] == b) & (perms[:, 2] == c)).astype(float)
        out["sanrentan"] = (ps, ys)
    return out


def score_race(tabs: dict) -> dict[str, float]:
    s: dict[str, float] = {}
    for bet, (p, y) in tabs.items():
        p = np.clip(p, EPS, 1 - EPS)
        s[f"brier_{bet}"] = float(np.sum((p - y) ** 2))
        s[f"bll_{bet}"] = float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if bet in ("umatan", "umaren", "sanrenpuku", "sanrentan"):
            s[f"ls_{bet}"] = float(-np.sum(y * np.log(p)))                # 排他的な結果の −log P
        elif bet == "wide":
            s["ls_wide"] = float(-np.sum(y * np.log(p)) / 3.0)             # 3 組の平均
        else:
            s["ls_fukusho"] = float(-np.sum(y * np.log(p)) / max(y.sum(), 1))
    return s


def bin_stats(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(BINS, p, side="right") - 1, 0, len(BINS) - 2)
    n = np.bincount(idx, minlength=len(BINS) - 1)
    k = np.bincount(idx, weights=y, minlength=len(BINS) - 1)
    sp = np.bincount(idx, weights=p, minlength=len(BINS) - 1)
    return np.vstack([n, k, sp])


def bootstrap(a: np.ndarray, b: np.ndarray, rng, B: int = 400) -> tuple[float, float, float]:
    d = a - b
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    races = load_races()
    years = sorted({r["year"] for r in races})
    cals = marginal_calibrators(races, years)
    gal = {y: global_alpha([r for r in races if r["year"] != y]) for y in years}
    _log(f"共通 α（年を除いた最尤）: {gal}")
    scores = {m: [] for m in METHODS}
    bins = {m: {b: np.zeros((3, len(BINS) - 1)) for b in BETS} for m in METHODS}
    bins_year = {m: {y: {b: np.zeros((3, len(BINS) - 1)) for b in BETS} for y in years} for m in METHODS}
    t0 = time.time()
    for ri, r in enumerate(races):
        for m in METHODS:
            d = make_dist(m, r, gal[r["year"]], cals[r["year"]])
            tabs = race_tables(d, r)
            scores[m].append(score_race(tabs))
            for b, (p, y) in tabs.items():
                st = bin_stats(p, y)
                bins[m][b] += st
                bins_year[m][r["year"]][b] += st
        if (ri + 1) % 1000 == 0:
            _log(f"{ri + 1}/{len(races)} レース（{time.time() - t0:.0f}s）")
    rng = np.random.default_rng(0)
    yr = np.array([r["year"] for r in races])
    keys = list(scores["D1"][0].keys())
    arr = {m: {k: np.array([s[k] for s in scores[m]]) for k in keys} for m in METHODS}
    res: dict = {"n_races": len(races), "years": years, "global_alpha_loyo": {str(k): v for k, v in gal.items()}, "means": {}, "vs_D1": {}, "vs_H": {}, "by_year_vs_D1": {}}
    for m in METHODS:
        res["means"][m] = {k: float(arr[m][k].mean()) for k in keys}
    for base, name in (("D1", "vs_D1"), ("H", "vs_H")):
        for m in METHODS:
            if m == base:
                continue
            res[name][m] = {k: dict(zip(("diff", "lo", "hi"), bootstrap(arr[m][k], arr[base][k], rng))) for k in keys}
    for m in METHODS:
        if m == "D1":
            continue
        res["by_year_vs_D1"][m] = {
            str(y): {k: float((arr[m][k][yr == y] - arr["D1"][k][yr == y]).mean()) for k in keys if k.startswith(("ls_", "bll_"))} for y in years
        }
    res["calibration"] = {m: {b: bins[m][b].tolist() for b in BETS} for m in METHODS}
    res["calibration_year"] = {m: {str(y): {b: bins_year[m][y][b].tolist() for b in BETS} for y in years} for m in METHODS}
    res["bins"] = BINS.tolist()
    (OUT / "method_eval.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "per_race_scores.npz", **{f"{m}|{k}": arr[m][k] for m in METHODS for k in keys}, year=yr)
    _log("完了")


def _fit_platt_bins(n: np.ndarray, k: np.ndarray, sp: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize

    ok = n > 0
    pm = np.clip(sp[ok] / n[ok], 1e-7, 1 - 1e-7)
    x = np.log(pm / (1 - pm))
    nn, kk = n[ok], k[ok]

    def nll(ab):
        z = ab[0] + ab[1] * x
        return float(np.sum(nn * np.logaddexp(0, z) - kk * z))

    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
    return float(r.x[0]), float(r.x[1])


def _iso_knots(n: np.ndarray, k: np.ndarray, sp: np.ndarray, min_n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """帯集計から単調な較正曲線（等張回帰）の節点 (p, p') を作る。観測が少ない帯（n<min_n）は隣と併合しない代わりに除く。"""
    from sklearn.isotonic import IsotonicRegression

    ok = n >= min_n
    x = sp[ok] / n[ok]
    y = k[ok] / n[ok]
    iso = IsotonicRegression(y_min=1e-9, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y, sample_weight=n[ok])
    return x, np.asarray(iso.predict(x))


def calib(method: str) -> None:
    """選んだ方法の組合せ確率を、年を除いた年の帯集計で Platt 較正し（LOYO）、除いた年で改善幅を測る。"""
    res = json.loads((OUT / "method_eval.json").read_text(encoding="utf-8"))
    races = load_races()
    years = sorted({r["year"] for r in races})
    gal = {y: global_alpha([r for r in races if r["year"] != y]) for y in years}
    cals_m = marginal_calibrators(races, years)
    coef = {}
    for y in years:
        coef[y] = {}
        for b in BETS:
            agg = sum(np.array(res["calibration_year"][method][str(yy)][b]) for yy in years if yy != y)
            coef[y][b] = _fit_platt_bins(agg[0], agg[1], agg[2])
    knots = {y: {} for y in years}
    for y in years:
        for b in BETS:
            agg = sum(np.array(res["calibration_year"][method][str(yy)][b]) for yy in years if yy != y)
            knots[y][b] = _iso_knots(agg[0], agg[1], agg[2])
    _log("Platt 係数（LOYO）: " + json.dumps({str(y): {b: [round(v, 3) for v in c] for b, c in coef[y].items()} for y in years}))
    rows = []
    for r in races:
        d = make_dist(method, r, gal[r["year"]], cals_m[r["year"]])
        tabs = race_tables(d, r)
        row = {"year": r["year"]}
        for b, (p, y) in tabs.items():
            pc = _platt_apply(p, coef[r["year"]][b])
            pr = np.clip(p, EPS, 1 - EPS)
            pc = np.clip(pc, EPS, 1 - EPS)
            row[f"bll_raw_{b}"] = float(-np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
            row[f"bll_cal_{b}"] = float(-np.sum(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
            row[f"brier_raw_{b}"] = float(np.sum((pr - y) ** 2))
            row[f"brier_cal_{b}"] = float(np.sum((pc - y) ** 2))
            kx, ky = knots[r["year"]][b]
            pi = np.clip(np.interp(p, kx, ky), EPS, 1 - EPS)
            row[f"bll_iso_{b}"] = float(-np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi)))
            row[f"brier_iso_{b}"] = float(np.sum((pi - y) ** 2))
        rows.append(row)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(1)
    out = {"method": method, "coef": {str(y): {b: list(c) for b, c in coef[y].items()} for y in years}, "effect": {}}
    for b in BETS:
        for metric in ("bll", "brier"):
            a, c = df[f"{metric}_cal_{b}"].to_numpy(), df[f"{metric}_raw_{b}"].to_numpy()
            dm, lo, hi = bootstrap(a, c, rng)
            by_year = {str(y): float((df.loc[df.year == y, f"{metric}_cal_{b}"] - df.loc[df.year == y, f"{metric}_raw_{b}"]).mean()) for y in years}
            out["effect"][f"{metric}_{b}"] = {"diff": dm, "lo": lo, "hi": hi, "by_year": by_year}
            ai = df[f"{metric}_iso_{b}"].to_numpy()
            di, loi, hii = bootstrap(ai, c, rng)
            by_year_i = {str(y): float((df.loc[df.year == y, f"{metric}_iso_{b}"] - df.loc[df.year == y, f"{metric}_raw_{b}"]).mean()) for y in years}
            out["effect"][f"{metric}_iso_{b}"] = {"diff": di, "lo": loi, "hi": hii, "by_year": by_year_i}
    (OUT / f"calib_{method}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    _log("完了")


def _thurstone_ls(r: dict, rng: np.random.Generator, n_fit: int = 30000, n_eval: int = 150000) -> dict[str, float]:
    """Thurstone（正規）型: 能力 μ_i + N(0,1) の大小で着順が決まる。μ は勝率に合わせ（共通乱数で反復）、組合せ確率は Monte Carlo。"""
    w = r["win"] / r["win"].sum()
    n = r["n"]
    mu = 0.9 * np.log(np.clip(w, 1e-6, None))
    z = rng.standard_normal((n_fit, n))
    for _ in range(10):
        pw = np.bincount(np.argmax(mu + z, axis=1), minlength=n) / n_fit
        mu = mu + 0.6 * (np.log(np.clip(w, 1e-6, None)) - np.log(np.clip(pw, 0.5 / n_fit, None)))
    top = np.argsort(-(mu + rng.standard_normal((n_eval, n))), axis=1)[:, :3]
    a, b, c = (int(x) for x in r["abc"])
    fl = 0.5 / n_eval
    t1, t2, t3 = top[:, 0], top[:, 1], top[:, 2]
    p_um = max(np.mean((t1 == a) & (t2 == b)), fl)
    p_ur = max(np.mean(((t1 == a) & (t2 == b)) | ((t1 == b) & (t2 == a))), fl)
    s3 = np.sort(top, axis=1)
    p_sp = max(np.mean((s3 == np.sort([a, b, c])).all(axis=1)), fl)
    p_st = max(np.mean((t1 == a) & (t2 == b) & (t3 == c)), fl)
    inn = lambda x, y: np.mean(((t1 == x) | (t2 == x) | (t3 == x)) & ((t1 == y) | (t2 == y) | (t3 == y)))  # noqa: E731
    p_w = [max(inn(x, y), fl) for x, y in ((a, b), (a, c), (b, c))]
    return {"ls_umatan": -np.log(p_um), "ls_umaren": -np.log(p_ur), "ls_sanrenpuku": -np.log(p_sp), "ls_sanrentan": -np.log(p_st), "ls_wide": float(np.mean(-np.log(p_w)))}


def thurstone(n_races: int = 700) -> None:
    races = load_races()
    sub = races[:: max(len(races) // n_races, 1)][:n_races]
    years = sorted({r["year"] for r in races})
    gal = {y: global_alpha([r for r in races if r["year"] != y]) for y in years}
    cals = marginal_calibrators(races, years)
    rng = np.random.default_rng(3)
    out = {"H": [], "T": [], "D1": [], "MB60": []}
    t0 = time.time()
    for i, r in enumerate(sub):
        for m in ("H", "D1", "MB60"):
            sc = score_race(race_tables(make_dist(m, r, gal[r["year"]], cals[r["year"]]), r))
            out[m].append({k: sc[k] for k in ("ls_umatan", "ls_umaren", "ls_wide", "ls_sanrenpuku", "ls_sanrentan")})
        out["T"].append(_thurstone_ls(r, rng))
        if (i + 1) % 100 == 0:
            _log(f"{i + 1}/{len(sub)}（{time.time() - t0:.0f}s）")
    keys = list(out["H"][0].keys())
    arr = {m: {k: np.array([x[k] for x in out[m]]) for k in keys} for m in out}
    rb = np.random.default_rng(5)
    res = {"n_races": len(sub), "means": {m: {k: float(arr[m][k].mean()) for k in keys} for m in arr}, "T_vs_H": {k: dict(zip(("diff", "lo", "hi"), bootstrap(arr["T"][k], arr["H"][k], rb))) for k in keys}, "T_vs_MB60": {k: dict(zip(("diff", "lo", "hi"), bootstrap(arr["T"][k], arr["MB60"][k], rb))) for k in keys}}
    (OUT / "thurstone_ref.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    _log("完了")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "thurstone":
        thurstone()
    elif cmd == "calib":
        calib(sys.argv[2] if len(sys.argv) > 2 else "M3")
    else:
        raise SystemExit(f"unknown command: {cmd}")
