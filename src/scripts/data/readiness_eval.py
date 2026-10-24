"""導入可否の確認（readiness-audit）: v4/v5 の勝率・連対・複勝を、2026 年（独立データ）で基準に照らして判定する集計。

基準は readiness_criteria.json（2026 年の結果を見る前に固定）。ホストで実行する:

  python -m src.scripts.data.readiness_eval --out /home/jovyan/work/readiness_out/ensemble.json \
      --v4-dir data/local/inference/predictions_backfill/bundle_v4_2025 --v4-shadow data/local/inference/predictions_v3 \
      [--v5-dir data/local/inference/predictions_backfill/bundle_v5_2025]

指標は馬単位の二値 LogLoss（勝率=1着、連対=2着以内、複勝=3着以内）。中止・取消などで着順が無い馬は除く。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REF_TEST2025 = {
    "v4": {"win": 0.20739587725510406, "top2": 0.3235967155948879, "top3": 0.406398901843765},
    "v5": {"win": 0.20714416274338798, "top2": 0.32298768203793715, "top3": 0.40559974611403815},
}
TOL_C1 = 0.010
ECE_MAX = {"win": 0.006, "top2": 0.010, "top3": 0.015}
SLOPE_RANGE = (0.85, 1.15)
STRATUM_MIN_ROWS = 500
SUM_TOL = 1e-3
TARGETS = {"win": 1, "top2": 2, "top3": 3}
PCOL = {"win": "p_win", "top2": "p_top2", "top3": "p_top3"}


def bin_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"p": p, "y": y, "q": q})
    g = df.groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float((np.abs(g["p"] - g["y"]) * g["n"]).sum() / len(df))


def calib_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y ~ sigmoid(a + b*logit(p)) の (a, b)。ニュートン法。"""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.c_[np.ones_like(z), z]
    w = np.array([0.0, 1.0])
    for _ in range(50):
        m = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - m)
        H = (X * (m * (1 - m))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def load_results(tables: Path, year: int = 2026) -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "field_size", "horse_id", "horse_number", "finish_position"]
    df = pd.read_parquet(tables / str(year) / "race_result_flat.parquet", columns=cols)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["race_id"].str[4:6].isin([f"{i:02d}" for i in range(1, 11)])]
    return df


def prior_runs(tables: Path, df: pd.DataFrame) -> pd.Series:
    """各馬の、そのレース日より前の出走数（2020 年以降のデータ内）。履歴の浅い馬の層分けに使う。"""
    parts = []
    for y in range(2020, 2027):
        p = tables / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["horse_id", "date"]))
    h = pd.concat(parts, ignore_index=True)
    h["horse_id"] = h["horse_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna().sort_values(["horse_id", "date"])
    h["n_prior"] = h.groupby("horse_id").cumcount()
    h = h.drop_duplicates(["horse_id", "date"], keep="first")
    m = df[["horse_id", "date"]].merge(h, on=["horse_id", "date"], how="left")
    return m["n_prior"].fillna(0).astype(int).to_numpy()


def load_preds(dirs: list[Path]) -> pd.DataFrame:
    parts = []
    for d in dirs:
        for f in sorted(d.glob("*/*.parquet")):
            x = pd.read_parquet(f)
            x["race_id"] = x["race_id"].astype(str)
            x["horse_id"] = x["horse_id"].astype(str)
            x["pred_day"] = f.parent.name
            parts.append(x)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "horse_id"], keep="first") if parts else pd.DataFrame()


def attach_targets(res: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    m = pred.merge(res, on=["race_id", "horse_id"], how="inner")
    m = m[m["finish_position"] >= 1].copy()
    for k, n in TARGETS.items():
        m[f"y_{k}"] = (m["finish_position"] <= n).astype(float)
    n = m.groupby("race_id")["horse_id"].transform("size")
    m["n_in_race"] = n
    for k, kk in TARGETS.items():
        m[f"prior_{k}"] = np.minimum(kk / n, 1.0)
    return m


def metrics(m: pd.DataFrame) -> dict:
    out = {"rows": int(len(m)), "races": int(m["race_id"].nunique())}
    for k in TARGETS:
        y, p = m[f"y_{k}"].to_numpy(), m[PCOL[k]].to_numpy()
        a, b = calib_slope(p, y)
        out[k] = {"logloss": float(bin_ll(y, p).mean()), "prior_logloss": float(bin_ll(y, m[f"prior_{k}"].to_numpy()).mean()),
                  "ece": ece(p, y), "calib_intercept": a, "calib_slope": b, "mean_p": float(p.mean()), "mean_y": float(y.mean())}
    return out


def strata(m: pd.DataFrame, n_prior: pd.Series | None) -> dict:
    d = m.copy()
    d["f_size"] = pd.cut(d["field_size"].astype(float), [0, 9, 13, 16, 99], labels=["~9頭", "10-13頭", "14-16頭", "17頭~"])
    d["dist_band"] = pd.cut(d["distance"].astype(float), [0, 1300, 1700, 2100, 9999], labels=["~1200", "1400-1600", "1800-2000", "2200~"])
    d["month"] = d["date"].dt.month.astype(str) + "月"
    d["hist"] = pd.cut(d["n_prior"], [-1, 0, 2, 999], labels=["初出走", "1-2走", "3走~"]) if "n_prior" in d else "NA"
    d["cls"] = d["race_class"].astype(str)
    out = {}
    for col in ["f_size", "surface", "dist_band", "cls", "venue", "month", "hist"]:
        rows = []
        for key, g in d.groupby(col, observed=True):
            if len(g) < STRATUM_MIN_ROWS:
                continue
            rec = {"stratum": str(key), "rows": int(len(g))}
            worse = []
            for k in TARGETS:
                y = g[f"y_{k}"].to_numpy()
                lm, lp = bin_ll(y, g[PCOL[k]].to_numpy()).mean(), bin_ll(y, g[f"prior_{k}"].to_numpy()).mean()
                rec[k] = {"model": float(lm), "prior": float(lp)}
                if lm >= lp:
                    worse.append(k)
            rec["worse_than_prior"] = worse
            rows.append(rec)
        out[col] = rows
    return out


def top_pick(m: pd.DataFrame) -> dict:
    idx = m.groupby("race_id")["p_win"].idxmax()
    t = m.loc[idx]
    return {"races": int(len(t)), "win_rate": float((t["finish_position"] == 1).mean()), "top3_rate": float((t["finish_position"] <= 3).mean())}


def sums_check(pred: pd.DataFrame) -> dict:
    g = pred.groupby("race_id")[list(PCOL.values())].sum()
    bad = {k: int(((g[c] - TARGETS[k]).abs() > SUM_TOL).sum()) for k, c in PCOL.items()}
    return {"races": int(len(g)), "bad": bad, "nan_rows": int(pred[list(PCOL.values())].isna().any(axis=1).sum())}


def paired_boot(a: pd.DataFrame, b: pd.DataFrame, key: str = "win", n_boot: int = 2000, seed: int = 0) -> dict:
    """b - a の LogLoss 差（レース単位ブートストラップ）。同じ (race_id, horse_id) だけ。"""
    m = a.merge(b, on=["race_id", "horse_id"], suffixes=("_a", "_b"))
    y = m[f"y_{key}_a"].to_numpy()
    d = bin_ll(y, m[f"{PCOL[key]}_b"].to_numpy()) - bin_ll(y, m[f"{PCOL[key]}_a"].to_numpy())
    g = pd.DataFrame({"race_id": m["race_id"], "d": d}).groupby("race_id")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return {"races": int(len(g)), "delta": float(s.sum() / n.sum()), "lo": float(np.quantile(est, 0.025)), "hi": float(np.quantile(est, 0.975))}


def market_ll(tables: Path, year: int, with_probs: bool = False):
    """市場（確定単勝オッズの逆数を正規化）の勝率 LogLoss。年の難しさ（モデルの劣化か、年が難しいか）の切り分け用。"""
    res = load_results(tables, year)
    res = res[res["finish_position"] >= 1]
    import pyarrow.parquet as pq

    fo = tables / str(year) / "race_odds_flat.parquet"
    names = pq.read_schema(fo).names
    cols = [c for c in ["race_id", "horse_number", "win_odds", "snapshot_at"] if c in names]
    o = pd.read_parquet(fo, columns=cols)
    o["race_id"] = o["race_id"].astype(str)
    if "snapshot_at" in o.columns:
        o = o.sort_values("snapshot_at")
    o = o.drop_duplicates(["race_id", "horse_number"], keep="last")
    m = res.merge(o, on=["race_id", "horse_number"], how="inner")
    m["win_odds"] = pd.to_numeric(m["win_odds"], errors="coerce")
    m = m[m["win_odds"] >= 1.0]
    inv = 1.0 / m["win_odds"]
    s_ = inv.groupby(m["race_id"]).transform("sum")
    tot = (1.0 / m["win_odds"]).groupby(m["race_id"]).sum()
    ok = tot[(tot > 1.0) & (tot < 1.6)].index                      # 壊れた単勝オッズのレースを除く（Σ1/odds が 1.0〜1.6 の外）
    m = m[m["race_id"].isin(ok)].copy()
    m["p"] = (1.0 / m["win_odds"]) / (1.0 / m["win_odds"]).groupby(m["race_id"]).transform("sum")
    y = (m["finish_position"] == 1).astype(float).to_numpy()
    out = {"year": year, "rows": int(len(m)), "races": int(m["race_id"].nunique()), "market_win_logloss": float(bin_ll(y, m["p"].to_numpy()).mean())}
    return (out, m[["race_id", "horse_id", "p"]].rename(columns={"p": "p_mkt"})) if with_probs else out


def verdict(model: str, mt: dict, cov: dict, sc: dict, stra: dict, top: dict) -> dict:
    ref = REF_TEST2025[model]
    c1 = {k: (mt[k]["logloss"] - ref[k]) for k in TARGETS}
    c1_ok = all(v <= TOL_C1 for v in c1.values())
    c2 = {k: {"ece": mt[k]["ece"], "slope": mt[k]["calib_slope"]} for k in TARGETS}
    c2_ok = all(mt[k]["ece"] <= ECE_MAX[k] and SLOPE_RANGE[0] <= mt[k]["calib_slope"] <= SLOPE_RANGE[1] for k in TARGETS)
    worse = [(col, r["stratum"], r["worse_than_prior"]) for col, rows in stra.items() for r in rows if r["worse_than_prior"]]
    c3_ok = len(worse) == 0
    c4_ok = cov["coverage"] >= 0.999 and sc["nan_rows"] == 0 and all(v == 0 for v in sc["bad"].values())
    return {"C1_delta_vs_test2025": c1, "C1_ok": c1_ok, "C2": c2, "C2_ok": c2_ok, "C3_worse_strata": worse, "C3_ok": c3_ok,
            "C4_coverage": cov, "C4_sums": sc, "C4_ok": c4_ok, "S2_top_pick": top,
            "S2_ok": bool(top["win_rate"] >= 0.26 and top["top3_rate"] >= 0.55)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/page_reference/tables")
    ap.add_argument("--v4-dir", default="data/local/inference/predictions_backfill/bundle_v4_2025")
    ap.add_argument("--v4-shadow", default="data/local/inference/predictions_v3")
    ap.add_argument("--v5-dir", default="", help="v5 の確率のフォルダ（カンマ区切りで複数可）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tables = Path(a.tables)
    res = load_results(tables)
    res["n_prior"] = prior_runs(tables, res)
    races_all = res[res["finish_position"] >= 1].groupby("race_id").size()
    races_all = set(races_all[races_all >= 5].index)

    out: dict = {"results_races": len(races_all), "market": {str(y): market_ll(tables, y) for y in (2025, 2026)}}
    v4_dirs = [Path(a.v4_dir)] + ([Path(a.v4_shadow)] if a.v4_shadow else [])
    p4 = load_preds(v4_dirs)
    m4 = attach_targets(res, p4)
    cov4 = {"pred_races": int(p4["race_id"].nunique()), "covered": int(len(set(p4["race_id"]) & races_all)), "coverage": len(set(p4["race_id"]) & races_all) / max(len(races_all), 1)}
    m4s = m4.copy()
    out["v4"] = {"metrics": metrics(m4), "strata": strata(m4s, res["n_prior"]), "top_pick": top_pick(m4)}
    mk, mp = market_ll(tables, 2026, with_probs=True)
    mm = m4.merge(mp, on=["race_id", "horse_id"], how="inner")
    yv = mm["y_win"].to_numpy()
    out["v4_vs_market_2026"] = {"rows": int(len(mm)), "races": int(mm["race_id"].nunique()), "model_win_ll": float(bin_ll(yv, mm["p_win"].to_numpy()).mean()),
                                "market_win_ll": float(bin_ll(yv, mm["p_mkt"].to_numpy()).mean())}
    out["v4"]["verdict"] = verdict("v4", out["v4"]["metrics"], cov4, sums_check(p4), out["v4"]["strata"], out["v4"]["top_pick"])
    v5_dirs = [Path(x) for x in a.v5_dir.split(",") if x and Path(x).exists()]
    if v5_dirs:
        p5 = load_preds(v5_dirs)
        if len(p5):
            m5 = attach_targets(res, p5)
            same = m4.merge(m5[["race_id", "horse_id"]], on=["race_id", "horse_id"])
            out["v5_sample"] = {"metrics_v5": metrics(m5), "metrics_v4_same_rows": metrics(same), "paired_v5_minus_v4": {k: paired_boot(m4, m5, k) for k in TARGETS},
                                "sums": sums_check(p5), "top_pick_v5": top_pick(m5), "top_pick_v4_same": top_pick(same)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "v4" else {"metrics": v["metrics"], "verdict": v["verdict"]}) for k, v in out.items()}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
