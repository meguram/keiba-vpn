"""クロス・相対特徴量生成（AREA-07 §4-3）。"""

from __future__ import annotations

import pandas as pd


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """脚質×コース形状・同レース内相対化・ペース事前シナリオを追加。"""
    out = df.copy()
    if "running_style_score" in out.columns and "final_straight_length" in out.columns:
        out["style_x_straight"] = out["running_style_score"] * out["final_straight_length"]
    if "running_style_score" in out.columns and "distance" in out.columns:
        out["style_x_distance"] = out["running_style_score"] * out["distance"]

    if "running_style_score" in out.columns and "race_id" in out.columns:
        out["front_runner_count"] = out.groupby("race_id")["running_style_score"].transform(
            lambda x: (x < -2).sum()
        )

    if "speed_index_avg" in out.columns and "race_id" in out.columns:
        mean_by_race = out.groupby("race_id")["speed_index_avg"].transform("mean")
        out["rel_speed_index"] = out["speed_index_avg"] / mean_by_race.replace(0, pd.NA)

    if "days_since_last" in out.columns and "race_id" in out.columns:
        out["rel_days_since_last"] = out["days_since_last"] - out.groupby("race_id")[
            "days_since_last"
        ].transform("mean")

    if "odds_value" in out.columns and "race_id" in out.columns:
        out["rel_odds_rank"] = out.groupby("race_id")["odds_value"].rank(ascending=True)

    if "front_runner_count" in out.columns and "field_size" in out.columns:
        ratio = out["front_runner_count"] / out["field_size"].replace(0, pd.NA)

        def _pace_prior(r: float) -> str:
            if pd.isna(r):
                return "MIDDLE"
            if r > 0.3:
                return "HIGH"
            if r < 0.1:
                return "SLOW"
            return "MIDDLE"

        out["pace_scenario_prior"] = ratio.apply(_pace_prior)

    return out


def running_style_label(score: float | None) -> str:
    """脚質スコア → FRONT/STALKER/MID/CLOSER。"""
    if score is None:
        return "MID"
    if score <= -3:
        return "FRONT"
    if score <= -1:
        return "STALKER"
    if score >= 2:
        return "CLOSER"
    return "MID"
