"""最終オッズ予測の bundle 同士を、同じ評価行（既定 2025 年）で比べる（学習はしない・本番の bundle は変更しない）。

python -m src.scripts.data.eval_final_odds_bundles --bundles models/final_odds_bundle.json models/final_odds_bundle.candidate.json --out /tmp/final_odds_eval.json
評価行は final_odds_dataset と同じ作り方（確定オッズがラベル）。作るのに時間がかかるので parquet にキャッシュする。
指標: MAPE（学習時と同じ定義）、対数の平均絶対誤差、レース内の順位相関（Spearman）、予測÷確定オッズの中央値と p10/p90。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HEADS = (("win", "target_win_odds"), ("place_min", "target_place_min"), ("place_max", "target_place_max"))
DEFAULT_CACHE = "data/local/modeling/final_odds_eval_cache/eval_{years}.parquet"


def build_eval_frame(years: tuple[str, ...], cache: Path, max_races: int | None = None) -> pd.DataFrame:
    if cache.exists() and max_races is None:
        return pd.read_parquet(cache)
    from src.pipeline.models.final_odds_dataset import build_final_odds_dataset
    from src.scraper.storage import HybridStorage

    df = build_final_odds_dataset(HybridStorage(), train_years=(), eval_years=years, max_races=max_races)
    if max_races is None and not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def predict_bundle(path: Path, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """final_odds_predictor と同じ手順（欠けた列は 0、欠損は 0 埋め、予測は exp で最低 1）。"""
    import lightgbm as lgb

    b = json.loads(path.read_text(encoding="utf-8"))
    cols = b["feature_cols"]
    X = df.reindex(columns=cols).fillna(0).astype(float)
    out = {}
    for h, v in b["heads"].items():
        out[h] = np.maximum(np.exp(lgb.Booster(model_str=v["model_txt"]).predict(X)), 1.0)
    out["_n_missing_cols"] = np.array([len([c for c in cols if c not in df.columns])])
    return out


def within_race_spearman(race: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    d = pd.DataFrame({"r": race, "p": pred, "a": actual})
    d["pr"] = d.groupby("r")["p"].rank()
    d["ar"] = d.groupby("r")["a"].rank()
    cs = []
    for _, g in d.groupby("r"):
        if len(g) >= 3 and g["pr"].std() > 0 and g["ar"].std() > 0:
            cs.append(np.corrcoef(g["pr"], g["ar"])[0, 1])
    return float(np.mean(cs)) if cs else float("nan")


def metrics(df: pd.DataFrame, pred: dict[str, np.ndarray]) -> dict:
    res: dict = {"missing_feature_cols": int(pred["_n_missing_cols"][0])}
    race = df["race_id"].astype(str).to_numpy()
    for h, col in HEADS:
        if h not in pred:
            continue
        p, a = pred[h], df[col].astype(float).to_numpy()
        ratio = p / np.maximum(a, 1e-9)
        la = np.log(np.maximum(a, 1.0))
        res[h] = {
            "mape_pct": round(float(np.mean(np.abs(p - a) / np.maximum(a, 1.0))) * 100, 2),
            "log_mae": round(float(np.mean(np.abs(np.log(p) - la))), 4),
            "rmse_log": round(float(np.sqrt(np.mean((np.log(p) - la) ** 2))), 4),
            "spearman_within_race": round(within_race_spearman(race, p, a), 4),
            "ratio_median": round(float(np.median(ratio)), 3),
            "ratio_p10": round(float(np.quantile(ratio, 0.10)), 3),
            "ratio_p90": round(float(np.quantile(ratio, 0.90)), 3),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--years", default="2025")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--max-races", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    years = tuple(a.years.split(","))
    cache = Path(a.cache or DEFAULT_CACHE.format(years="_".join(years)))
    t0 = time.time()
    df = build_eval_frame(years, cache, a.max_races)
    print(f"評価行 {len(df):,} 行 / {df['race_id'].nunique():,} レース  {time.time() - t0:.0f}s", flush=True)
    rep: dict = {"rows": int(len(df)), "races": int(df["race_id"].nunique()), "years": list(years), "bundles": {}}
    for bp in a.bundles:
        rep["bundles"][bp] = metrics(df, predict_bundle(Path(bp), df))
        print(bp, json.dumps(rep["bundles"][bp], ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
