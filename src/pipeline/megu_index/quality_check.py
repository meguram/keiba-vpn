"""めぐ指数ローカル品質チェック（短期データ検証用）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.pipeline.megu_index.career_prize import classify_grade
from src.pipeline.megu_index.common import MEGU_BASE, MEGU_POINTS_PER_SEC, adjusted_time_to_megu
from src.pipeline.megu_index.flat_metadata import is_obstacle_race_name
from src.pipeline.megu_index.track_speed import assign_class_rank


def _finish_pos_col(df: pd.DataFrame) -> str:
    return "finish_position" if "finish_position" in df.columns else "finish_pos"


def summarize_megu_quality(
    df: pd.DataFrame,
    *,
    label: str = "",
    megu_col: str = "megu_index",
) -> dict[str, Any]:
    """
    品質サマリを返す。

    df は race_result メタ（grade, race_name, finish_position 等）を含む想定。
    """
    out: dict[str, Any] = {"label": label, "rows": len(df)}
    if df.empty:
        return out

    if megu_col not in df.columns and {"adjusted_time_sec", "par_time_final", "corrected_time_sec"} <= set(df.columns):
        if "corrected_time_sec" in df.columns:
            df = df.copy()
            df["_megu"] = df.apply(
                lambda r: adjusted_time_to_megu(r["corrected_time_sec"], r["par_time_final"]),
                axis=1,
            )
        else:
            df = df.copy()
            df["_megu"] = df.apply(
                lambda r: adjusted_time_to_megu(r["adjusted_time_sec"], r["par_time_final"]),
                axis=1,
            )
        megu_col = "_megu"

    status_col = "computation_status" if "computation_status" in df.columns else None
    if status_col:
        valid = df[df[status_col] == "valid"].copy()
        out["status_counts"] = df[status_col].value_counts().to_dict()
    else:
        valid = df[df[megu_col].notna()].copy()

    out["valid_rows"] = len(valid)
    if valid.empty:
        return out

    if "race_name" in valid.columns:
        obs = valid["race_name"].apply(is_obstacle_race_name)
        out["obstacle_valid_rows"] = int(obs.sum())

    megu = pd.to_numeric(valid[megu_col], errors="coerce")
    out["megu_abs_gt_150"] = int((megu.abs() > 150).sum())
    out["megu_median"] = float(megu.median())
    out["megu_mean"] = float(megu.mean())

    fp = _finish_pos_col(valid)
    if fp in valid.columns:
        valid = valid.copy()
        valid["finish_pos"] = pd.to_numeric(valid[fp], errors="coerce")
        valid["class_rank"] = valid.apply(
            lambda r: assign_class_rank(r.get("grade"), r.get("race_class")),
            axis=1,
        )
        s2 = valid[valid["finish_pos"] == 2].dropna(subset=["class_rank"])
        if len(s2) >= 20:
            by_cr = s2.groupby("class_rank")[megu_col].median()
            out["2nd_median_by_class_rank"] = {int(k): round(v, 1) for k, v in by_cr.items()}
            if len(by_cr) >= 3:
                r, _ = stats.spearmanr(by_cr.index.astype(float), by_cr.values)
                out["spearman_class_rank_2nd_median"] = float(r)

        w = valid[(valid["finish_pos"] == 1)].dropna(subset=["class_rank"])
        if len(w) >= 20:
            w["gn"] = w.apply(
                lambda r: classify_grade(r.get("grade"), r.get("race_name"), r.get("race_class")),
                axis=1,
            )
            g1 = w[w["gn"] == "G1"][megu_col]
            if len(g1) >= 5:
                out["g1_winner_megu_median"] = float(g1.median())

    # 公式一致（corrected_time がある場合）
    if {"corrected_time_sec", "par_time_final", megu_col} <= set(valid.columns):
        recalc = valid.apply(
            lambda r: adjusted_time_to_megu(r["corrected_time_sec"], r["par_time_final"]),
            axis=1,
        )
        diff = (valid[megu_col] - recalc).abs()
        out["formula_max_diff"] = float(diff.max()) if len(diff) else 0.0
        out["formula_mismatch_gt_1e4"] = int((diff > 1e-4).sum())

    # レース内 1点=0.1秒
    time_col = "corrected_time_sec" if "corrected_time_sec" in valid.columns else "adjusted_time_sec"
    if time_col in valid.columns and "race_id" in valid.columns:
        slopes = []
        for _, g in valid.groupby("race_id"):
            if len(g) < 3:
                continue
            x = g[time_col].values
            y = g[megu_col].values
            if np.std(x) < 1e-9:
                continue
            s, *_ = stats.linregress(x, y)
            slopes.append(s)
        if slopes:
            out["race_slope_median"] = float(np.median(slopes))

    return out


def compare_quality(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """before/after サマリの差分。"""
    keys = [
        "obstacle_valid_rows",
        "megu_abs_gt_150",
        "spearman_class_rank_2nd_median",
        "g1_winner_megu_median",
        "formula_mismatch_gt_1e4",
        "race_slope_median",
    ]
    delta = {}
    for k in keys:
        if k in before or k in after:
            delta[k] = {
                "before": before.get(k),
                "after": after.get(k),
            }
    return delta


def passes_quality_gates(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """短期検証の合格基準。"""
    failures: list[str] = []
    if summary.get("obstacle_valid_rows", 0) > 0:
        failures.append(f"obstacle_valid_rows={summary['obstacle_valid_rows']}")
    if summary.get("megu_abs_gt_150", 999) > 0:
        failures.append(f"megu_abs_gt_150={summary.get('megu_abs_gt_150')}")
    sp = summary.get("spearman_class_rank_2nd_median")
    if sp is not None and sp < 0:
        failures.append(f"spearman_2nd={sp:.3f} (expected >= 0)")
    slope = summary.get("race_slope_median")
    if slope is not None and abs(slope + MEGU_POINTS_PER_SEC) > 0.01:
        failures.append(f"race_slope_median={slope:.4f} (expected -10)")
    g1 = summary.get("g1_winner_megu_median")
    if g1 is not None and g1 < MEGU_BASE:
        failures.append(f"g1_winner_median={g1:.1f} (expected >= {MEGU_BASE})")
    return len(failures) == 0, failures
