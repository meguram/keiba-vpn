"""フィールド質（FQ）計算 — AREA-11 §5。"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline.megu_index.career_prize import classify_grade, estimate_prize

logger = logging.getLogger(__name__)

FQ_FLOOR_YEN = 1_000_000  # 100万円
MANYEN_TO_YEN = 10_000


def _finish_pos_col(df: pd.DataFrame) -> str:
    return "finish_position" if "finish_position" in df.columns else "finish_pos"


def _is_shinba_race(row: pd.Series) -> bool:
    for col in ("race_type", "race_name", "race_class", "grade"):
        val = row.get(col)
        if val and "新馬" in str(val):
            return True
    return False


def build_career_yen_before_race(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    各 (horse_id, race_id) 行に、そのレースより前の累積獲得賞金（円）を付与。

    Returns:
        DataFrame [horse_id, race_id, race_dt, career_yen_before]
    """
    if df_results.empty:
        return pd.DataFrame(columns=["horse_id", "race_id", "race_dt", "career_yen_before"])

    df = df_results.copy()
    fp = _finish_pos_col(df)
    df["race_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["finish_pos_num"] = pd.to_numeric(df.get(fp), errors="coerce")
    df["grade_norm"] = df.apply(
        lambda r: classify_grade(r.get("grade"), r.get("race_name"), r.get("race_class")),
        axis=1,
    )
    df["prize_yen"] = df.apply(
        lambda r: estimate_prize(r["grade_norm"], r["finish_pos_num"]) * MANYEN_TO_YEN,
        axis=1,
    )
    df = df.sort_values(["horse_id", "race_dt", "race_id"])

    records = []
    for horse_id, grp in df.groupby("horse_id", sort=False):
        cum = 0.0
        for _, row in grp.iterrows():
            records.append({
                "horse_id": str(row["horse_id"]),
                "race_id": str(row["race_id"]),
                "race_dt": row["race_dt"],
                "career_yen_before": cum,
            })
            cum += float(row["prize_yen"] or 0.0)

    return pd.DataFrame(records)


def compute_par_log_fq(fq_by_race: pd.Series) -> float:
    """全レース FQ の幾何平均 → log(par_FQ)。"""
    valid = fq_by_race.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return math.log(FQ_FLOOR_YEN)
    return float(np.log(valid).mean())


def compute_race_fq_yen(
    race_rows: pd.DataFrame,
    career_before: pd.DataFrame,
    race_id: str,
) -> Optional[float]:
    """
    レースの FQ（円）= 上位5着以内の馬の事前累積賞金平均。
    算出不能時は None。
    """
    fp = _finish_pos_col(race_rows)
    top = race_rows.copy()
    top["finish_pos_num"] = pd.to_numeric(top.get(fp), errors="coerce")
    top = top[(top["finish_pos_num"] >= 1) & (top["finish_pos_num"] <= 5)]
    if top.empty:
        return None

    cb = career_before[career_before["race_id"] == str(race_id)].set_index("horse_id")
    prizes = []
    for _, row in top.iterrows():
        hid = str(row["horse_id"])
        if hid in cb.index:
            val = cb.loc[hid, "career_yen_before"]
            prizes.append(float(val.iloc[0] if isinstance(val, pd.Series) else val))
        else:
            prizes.append(0.0)

    if not prizes:
        return None
    if all(p <= 0 for p in prizes):
        return None

    active = [max(p, FQ_FLOOR_YEN) if p > 0 else FQ_FLOOR_YEN for p in prizes if p > 0]
    if not active:
        return None
    return float(np.mean(active))


def attach_fq_and_delta_level(
    df: pd.DataFrame,
    df_hist: Optional[pd.DataFrame],
    beta_level: float,
    par_log_fq: Optional[float] = None,
) -> pd.DataFrame:
    """
    df に field_quality（円）, log_fq_dev, delta_level_sec を付与。

    FQ はレース単位。同一 race_id の全馬が同じ delta_level_sec。
    """
    out = df.copy()
    out["field_quality"] = np.nan
    out["delta_level_sec"] = 0.0

    if beta_level == 0.0:
        return out

    combined = (
        pd.concat([df_hist, df], ignore_index=True)
        if df_hist is not None and not df_hist.empty
        else df.copy()
    )
    career_before = build_career_yen_before_race(combined)

    fq_records: list[dict] = []
    race_meta = combined.drop_duplicates("race_id")
    for race_id, grp in combined.groupby("race_id"):
        meta = race_meta[race_meta["race_id"] == race_id]
        if meta.empty:
            continue
        row0 = meta.iloc[0]
        if _is_shinba_race(row0):
            fq_records.append({"race_id": str(race_id), "field_quality": None, "log_fq_dev": 0.0})
            continue
        fq = compute_race_fq_yen(grp, career_before, str(race_id))
        if fq is None:
            fq_records.append({"race_id": str(race_id), "field_quality": None, "log_fq_dev": 0.0})
        else:
            fq_clip = max(float(fq), FQ_FLOOR_YEN)
            fq_records.append({
                "race_id": str(race_id),
                "field_quality": fq_clip,
                "log_fq_dev": math.log(fq_clip),
            })

    fq_df = pd.DataFrame(fq_records)
    if fq_df.empty:
        return out

    if par_log_fq is None:
        valid_fq = fq_df["field_quality"].dropna()
        par_log_fq = compute_par_log_fq(valid_fq)

    fq_df["delta_level_sec"] = beta_level * (fq_df["log_fq_dev"] - par_log_fq)
    fq_df.loc[fq_df["field_quality"].isna(), "delta_level_sec"] = 0.0

    fq_idx = fq_df.set_index("race_id")
    out = df.copy()
    out["field_quality"] = out["race_id"].astype(str).map(
        lambda rid: fq_idx.loc[rid, "field_quality"] if rid in fq_idx.index else np.nan
    )
    out["delta_level_sec"] = out["race_id"].astype(str).map(
        lambda rid: float(fq_idx.loc[rid, "delta_level_sec"]) if rid in fq_idx.index else 0.0
    )

    logger.info(
        "FQ: par_log_fq=%.4f, races_with_fq=%d / %d",
        par_log_fq,
        fq_df["field_quality"].notna().sum(),
        len(fq_df),
    )
    return out
