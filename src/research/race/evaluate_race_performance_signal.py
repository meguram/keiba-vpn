from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

from src.pipeline.features.feature_store import FeatureStore

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

OUT_JSON = Path("data/local/meta/modeling/race_performance_eval.json")


def _distance_band(distance: int) -> str:
    if distance <= 1400:
        return "sprint"
    if distance <= 1800:
        return "mile"
    if distance <= 2200:
        return "intermediate"
    if distance <= 2800:
        return "long"
    return "extended"


def _ndcg_at_k(y_true: list[int], y_score: list[float], k: int = 3) -> float:
    if not y_true:
        return 0.0
    order = np.argsort(-np.asarray(y_score))
    gains = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        gains += (2 ** int(y_true[idx]) - 1) / np.log2(rank + 1)
    ideal = sorted(y_true, reverse=True)
    ideal_gain = 0.0
    for rank, rel in enumerate(ideal[:k], start=1):
        ideal_gain += (2 ** int(rel) - 1) / np.log2(rank + 1)
    if ideal_gain <= 0:
        return 0.0
    return float(gains / ideal_gain)


def _race_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    top1_hit = []
    top1_top3 = []
    ndcg3_win = []
    ndcg3_top3 = []
    mrr = []

    for _, g in df.groupby("race_id", sort=False):
        gs = g.sort_values(score_col, ascending=False)
        top = gs.iloc[0]
        top1_hit.append(1.0 if int(top["is_win"]) == 1 else 0.0)
        top1_top3.append(1.0 if int(top["is_top3"]) == 1 else 0.0)
        ndcg3_win.append(_ndcg_at_k(g["is_win"].tolist(), g[score_col].tolist(), k=3))
        ndcg3_top3.append(_ndcg_at_k(g["is_top3"].tolist(), g[score_col].tolist(), k=3))

        win_positions = gs.index[gs["is_win"].values == 1].tolist()
        if win_positions:
            rank = gs.index.get_loc(win_positions[0]) + 1
            mrr.append(1.0 / rank)
        else:
            mrr.append(0.0)

    return {
        "top1_hit_rate": round(float(np.mean(top1_hit)), 4),
        "top1_top3_rate": round(float(np.mean(top1_top3)), 4),
        "ndcg3_win": round(float(np.mean(ndcg3_win)), 4),
        "ndcg3_top3": round(float(np.mean(ndcg3_top3)), 4),
        "mrr_win": round(float(np.mean(mrr)), 4),
    }


def _build_dataset() -> pd.DataFrame:
    perf = pd.read_parquet("data/local/features/snapshots/race_performance_2020_2025.parquet")

    store = FeatureStore()
    shutuba = store.load_source(
        "race_shutuba",
        years=["2020", "2021", "2022", "2023", "2024", "2025"],
        columns=["distance", "surface", "field_size", "grade", "race_class"],
    )
    shutuba = shutuba.drop_duplicates(subset=["race_id", "horse_number"])

    df = perf.merge(
        shutuba[["race_id", "horse_number", "distance", "surface", "field_size", "grade", "race_class"]],
        on=["race_id", "horse_number"],
        how="left",
        suffixes=("", "_shutuba"),
    )

    if "distance_shutuba" in df.columns:
        df["distance"] = df["distance_shutuba"].fillna(df["distance"])
    if "surface_shutuba" in df.columns:
        df["surface"] = df["surface_shutuba"].fillna(df["surface"])
    if "field_size_shutuba" in df.columns:
        df["field_size"] = df["field_size_shutuba"].fillna(df["field_size"])
    df = df.drop(columns=[c for c in df.columns if c.endswith("_shutuba")])
    df["distance"] = df["distance"].fillna(0).astype(int)
    df["distance_band"] = df["distance"].map(_distance_band)
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df["year"] = df["race_id"].str[:4]
    df["is_win"] = (df["finish_position"] == 1).astype(int)
    df["is_top3"] = (df["finish_position"] <= 3).astype(int)

    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    def _lag_roll(s: pd.Series, n: int, fn: str) -> pd.Series:
        shifted = s.shift(1)
        if fn == "mean":
            return shifted.rolling(n, min_periods=1).mean()
        if fn == "max":
            return shifted.rolling(n, min_periods=1).max()
        if fn == "std":
            return shifted.rolling(n, min_periods=2).std()
        raise ValueError(fn)

    grp = df.groupby("horse_id", group_keys=False)
    df["perf_last1"] = grp["run_performance_final"].shift(1)
    df["perf_last2"] = grp["run_performance_final"].shift(2)
    df["perf_avg3"] = grp["run_performance_final"].apply(lambda s: _lag_roll(s, 3, "mean"))
    df["perf_avg5"] = grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["perf_avg10"] = grp["run_performance_final"].apply(lambda s: _lag_roll(s, 10, "mean"))
    df["perf_best5"] = grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "max"))
    df["perf_std5"] = grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "std"))
    df["perf_stdscore_last1"] = grp["run_performance_final_std"].shift(1)
    df["perf_stdscore_avg5"] = grp["run_performance_final_std"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["perf_pct_last1"] = grp["run_performance_final_pct"].shift(1)
    df["perf_pct_avg5"] = grp["run_performance_final_pct"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["time_fig_avg5"] = grp["time_figure"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["margin_fig_avg5"] = grp["margin_adjusted_figure"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["pace_fig_avg5"] = grp["pace_adjusted_figure"].apply(lambda s: _lag_roll(s, 5, "mean"))
    df["n_prev_runs"] = grp.cumcount()
    df["perf_trend_last1_vs_avg5"] = df["perf_last1"] - df["perf_avg5"]

    surf_grp = df.sort_values(["horse_id", "surface", "race_date_dt", "race_id"]).groupby(["horse_id", "surface"], group_keys=False)
    df["perf_same_surface_avg5"] = surf_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()

    dist_grp = df.sort_values(["horse_id", "distance_band", "race_date_dt", "race_id"]).groupby(["horse_id", "distance_band"], group_keys=False)
    df["perf_same_distance_avg5"] = dist_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()
    surf_dist_grp = df.sort_values(["horse_id", "surface", "distance_band", "race_date_dt", "race_id"]).groupby(["horse_id", "surface", "distance_band"], group_keys=False)
    df["perf_same_surface_distance_avg5"] = surf_dist_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()

    class_grp = df.sort_values(["horse_id", "class_group", "race_date_dt", "race_id"]).groupby(["horse_id", "class_group"], group_keys=False)
    df["perf_same_class_avg5"] = class_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()

    venue_grp = df.sort_values(["horse_id", "venue_code", "race_date_dt", "race_id"]).groupby(["horse_id", "venue_code"], group_keys=False)
    df["perf_same_venue_avg5"] = venue_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()
    cond_grp = df.sort_values(["horse_id", "track_condition", "race_date_dt", "race_id"]).groupby(["horse_id", "track_condition"], group_keys=False)
    df["perf_same_track_cond_avg5"] = cond_grp["run_performance_final"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()
    df["pace_same_distance_avg5"] = dist_grp["pace_adjusted_figure"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()
    df["margin_same_distance_avg5"] = dist_grp["margin_adjusted_figure"].apply(lambda s: _lag_roll(s, 5, "mean")).sort_index()

    df["race_level_gap_avg5"] = df["perf_avg5"] - df["race_level_pre_race"]
    df["perf_momentum_3_10"] = df["perf_avg3"] - df["perf_avg10"]
    df["perf_ceiling_gap"] = df["perf_best5"] - df["perf_avg5"]
    df["perf_surface_distance_gap"] = df["perf_same_surface_avg5"] - df["perf_same_distance_avg5"]
    grp_u = df.groupby("horse_id", group_keys=False)
    df["unc_adj_perf_last1"] = (grp_u["run_performance_final"].shift(1) * (1 - grp_u["race_uncertainty"].shift(1))).fillna(0.0)
    df["unc_adj_perf_avg5"] = grp_u.apply(lambda g: (g["run_performance_final"].shift(1) * (1 - g["race_uncertainty"].shift(1))).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    for col, default in [
        ("field_relative_pre_rating", 0.0),
        ("field_top3_gap", 0.0),
        ("horse_pre_rating_pre_race", 0.0),
        ("horse_pre_rating_neutral", 0.0),
        ("current_weight_advantage_points", 0.0),
        ("reference_weight_equivalent", 0.0),
        ("normalized_weight_carried", 0.0),
        ("perf_stdscore_last1", 0.0),
        ("perf_stdscore_avg5", 0.0),
        ("perf_pct_last1", 0.5),
        ("perf_pct_avg5", 0.5),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    df["perf_missing_last1"] = df["perf_last1"].isna().astype(int)
    df["perf_missing_same_surface"] = df["perf_same_surface_avg5"].isna().astype(int)
    df["perf_missing_same_distance"] = df["perf_same_distance_avg5"].isna().astype(int)

    fill_zero_cols = [
        "perf_last1", "perf_last2", "perf_avg3", "perf_avg5", "perf_avg10",
        "perf_best5", "perf_std5", "time_fig_avg5", "margin_fig_avg5", "pace_fig_avg5",
        "perf_trend_last1_vs_avg5", "perf_same_surface_avg5", "perf_same_distance_avg5",
        "perf_same_surface_distance_avg5", "perf_same_class_avg5", "perf_same_venue_avg5", "pace_same_distance_avg5",
        "perf_same_track_cond_avg5", "margin_same_distance_avg5", "race_level_gap_avg5",
        "perf_momentum_3_10", "perf_ceiling_gap", "perf_surface_distance_gap",
        "unc_adj_perf_last1", "unc_adj_perf_avg5",
    ]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0.0)

    return df


def run_evaluation() -> dict[str, Any]:
    df = _build_dataset()
    train = df[df["year"].isin(["2020", "2021", "2022", "2023", "2024"])].copy()
    test = df[df["year"] == "2025"].copy()

    feature_cols = [
        "perf_last1",
        "perf_last2",
        "perf_avg3",
        "perf_avg5",
        "perf_avg10",
        "perf_best5",
        "perf_std5",
        "time_fig_avg5",
        "margin_fig_avg5",
        "pace_fig_avg5",
        "perf_trend_last1_vs_avg5",
        "perf_same_surface_avg5",
        "perf_same_distance_avg5",
        "perf_same_surface_distance_avg5",
        "perf_same_class_avg5",
        "perf_same_venue_avg5",
        "perf_same_track_cond_avg5",
        "pace_same_distance_avg5",
        "margin_same_distance_avg5",
        "race_level_gap_avg5",
        "perf_momentum_3_10",
        "perf_ceiling_gap",
        "perf_surface_distance_gap",
        "unc_adj_perf_last1",
        "unc_adj_perf_avg5",
        "horse_pre_rating_neutral",
        "horse_pre_rating_pre_race",
        "field_relative_pre_rating",
        "field_top3_gap",
        "current_weight_advantage_points",
        "reference_weight_equivalent",
        "normalized_weight_carried",
        "perf_stdscore_last1",
        "perf_stdscore_avg5",
        "perf_pct_last1",
        "perf_pct_avg5",
        "n_prev_runs",
        "perf_missing_last1",
        "perf_missing_same_surface",
        "perf_missing_same_distance",
    ]

    heuristic_score = (
        0.30 * test["perf_avg5"]
        + 0.20 * test["perf_last1"]
        + 0.15 * test["perf_same_surface_avg5"]
        + 0.15 * test["perf_same_distance_avg5"]
        + 0.10 * test["field_relative_pre_rating"]
        + 0.10 * test["perf_stdscore_avg5"]
        + 0.08 * test["current_weight_advantage_points"]
        + 0.07 * test["horse_pre_rating_neutral"]
    )
    test["heuristic_score"] = heuristic_score

    # win model
    lgb_win = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb_win.fit(train[feature_cols], train["is_win"])
    test["pred_win"] = lgb_win.predict_proba(test[feature_cols])[:, 1]

    # top3 model
    lgb_top3 = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb_top3.fit(train[feature_cols], train["is_top3"])
    test["pred_top3"] = lgb_top3.predict_proba(test[feature_cols])[:, 1]

    result = {
        "dataset": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_races": int(train["race_id"].nunique()),
            "test_races": int(test["race_id"].nunique()),
            "feature_cols": feature_cols,
        },
        "heuristic_ranking": _race_metrics(test, "heuristic_score"),
        "lgbm_win": {
            "auc": round(float(roc_auc_score(test["is_win"], test["pred_win"])), 4),
            "logloss": round(float(log_loss(test["is_win"], np.clip(test["pred_win"], 1e-7, 1 - 1e-7))), 4),
            **_race_metrics(test, "pred_win"),
        },
        "lgbm_top3": {
            "auc": round(float(roc_auc_score(test["is_top3"], test["pred_top3"])), 4),
            "logloss": round(float(log_loss(test["is_top3"], np.clip(test["pred_top3"], 1e-7, 1 - 1e-7))), 4),
            **_race_metrics(test.assign(pred_top3_rank=test["pred_top3"]), "pred_top3_rank"),
        },
        "feature_importance_top10": [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(
                zip(feature_cols, lgb_win.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    script_basic_config()
    res = run_evaluation()
    print(json.dumps(res, ensure_ascii=False, indent=2))
