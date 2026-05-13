"""
race_result 履歴から、当該レース「直前」までの累計統計を付与（リークなし）。

raw → preprocess: 本モジュールでテーブル横断の時系列累計を計算し、
LayerA 母表に merge（modeling 前段）。

同一 id（horse / jockey / trainer）を日付・race_id 順に並べ、
cumsum(当行を含む勝ち) − 当行の勝ち = 直前までの勝数（同様に top3）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _norm_id(s: pd.Series) -> pd.Series:
    """欠損は '-1'、数値は int 文字列に揃える。"""
    def one(v: Any) -> str:
        if pd.isna(v):
            return "-1"
        try:
            if float(v) == int(float(v)):
                return str(int(float(v)))
        except (TypeError, ValueError):
            pass
        return str(v)

    return s.map(one)


def _prior_rates_for_id(
    df: pd.DataFrame,
    id_col: str,
    prefix: str,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    cols = [c for c in ("race_id", "horse_number", id_col, date_col, "finish_position") if c in df.columns]
    if id_col not in cols:
        return pd.DataFrame()

    work = df[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work["_gid"] = _norm_id(work[id_col])
    work = work.sort_values(["_gid", date_col, "race_id", "horse_number"], kind="mergesort")

    fp = pd.to_numeric(work["finish_position"], errors="coerce")
    work["_win"] = (fp == 1).astype(np.float64)
    work["_t3"] = ((fp >= 1) & (fp <= 3)).astype(np.float64)

    g = work.groupby("_gid", sort=False)
    cs_win = g["_win"].cumsum()
    cs_t3 = g["_t3"].cumsum()
    prior_wins = cs_win - work["_win"]
    prior_top3 = cs_t3 - work["_t3"]
    prior_races = g.cumcount()

    out = work[["race_id", "horse_number"]].copy()
    out[f"{prefix}_prior_races"] = prior_races.to_numpy()
    out[f"{prefix}_prior_wins"] = prior_wins.to_numpy()
    out[f"{prefix}_prior_top3"] = prior_top3.to_numpy()
    pr = prior_races.astype(float).replace(0, np.nan).to_numpy()
    pw = prior_wins.to_numpy()
    pt = prior_top3.to_numpy()
    out[f"{prefix}_prior_win_rate"] = np.where(np.isfinite(pr) & (pr > 0), pw / pr, np.nan)
    out[f"{prefix}_prior_top3_rate"] = np.where(np.isfinite(pr) & (pr > 0), pt / pr, np.nan)

    out = out.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
    return out


def _prior_avg_finish(df: pd.DataFrame) -> pd.DataFrame:
    if "horse_id" not in df.columns:
        return pd.DataFrame()

    work = df[["race_id", "horse_number", "horse_id", "date", "finish_position"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["_hid"] = _norm_id(work["horse_id"])
    fp = pd.to_numeric(work["finish_position"], errors="coerce")
    work["_fp"] = fp
    work = work.sort_values(["_hid", "date", "race_id"], kind="mergesort")

    g = work.groupby("_hid", sort=False)
    prior_sum = g["_fp"].cumsum() - work["_fp"].fillna(0)
    prior_cnt = g.cumcount()
    avg = np.where(prior_cnt > 0, prior_sum / prior_cnt, np.nan)

    out = work[["race_id", "horse_number"]].copy()
    out["horse_prior_avg_finish"] = avg
    out = out.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
    return out


def build_history_feature_table(race_result_df: pd.DataFrame) -> pd.DataFrame:
    """race_result 全行から race_id+horse_number 単位の履歴特徴1表。"""
    if race_result_df.empty:
        return pd.DataFrame()

    rr = race_result_df.copy()
    j = _prior_rates_for_id(rr, "jockey_id", "jockey")
    t = _prior_rates_for_id(rr, "trainer_id", "trainer")
    h = _prior_rates_for_id(rr, "horse_id", "horse")
    af = _prior_avg_finish(rr)

    base = rr[["race_id", "horse_number"]].drop_duplicates()
    m = base.merge(j, on=["race_id", "horse_number"], how="left")
    m = m.merge(t, on=["race_id", "horse_number"], how="left")
    m = m.merge(h, on=["race_id", "horse_number"], how="left")
    m = m.merge(af, on=["race_id", "horse_number"], how="left")

    feat_cols = [c for c in m.columns if c not in ("race_id", "horse_number")]
    logger.info("履歴特徴付与: %d 行, features=%s", len(m), feat_cols)
    return m


def merge_history_into_layer_a(
    layer_df: pd.DataFrame,
    race_result_full: pd.DataFrame,
) -> pd.DataFrame:
    hist = build_history_feature_table(race_result_full)
    if hist.empty:
        return layer_df
    return layer_df.merge(hist, on=["race_id", "horse_number"], how="left")
