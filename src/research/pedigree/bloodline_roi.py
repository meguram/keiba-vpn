"""血統研究用の回収率計算（単勝・複勝オッズ上限付き）。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 大穴上振れ回避: この倍率以下の単勝オッズのみ回収率集計に含める
MAX_ROI_ODDS = 30.0

SLIM_PATH = Path("data/page_reference/pedigree_race_index/race_result_slim.parquet")
TABLES_DIR = Path("data/page_reference/tables")


def roi_eligible_mask(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return o.gt(0) & o.le(MAX_ROI_ODDS)


def _load_odds_from_tables(years: list[str] | None = None) -> pd.DataFrame:
    if not TABLES_DIR.exists():
        return pd.DataFrame(columns=["race_id", "horse_id", "odds"])
    if years is None:
        years = sorted(d.name for d in TABLES_DIR.iterdir() if d.is_dir() and d.name.isdigit())
    parts: list[pd.DataFrame] = []
    for yr in years:
        p = TABLES_DIR / yr / "race_result_flat.parquet"
        if not p.exists():
            continue
        try:
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "odds"]))
        except Exception as e:  # pragma: no cover
            logger.warning("odds 読込スキップ %s: %s", p, e)
    if not parts:
        return pd.DataFrame(columns=["race_id", "horse_id", "odds"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out["horse_id"].astype(str)
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out.loc[~((out["odds"] > 0) & (out["odds"] <= 999.9)), "odds"] = np.nan
    return out


def attach_payoffs_and_odds(df: pd.DataFrame, years: list[str] | None = None) -> pd.DataFrame:
    """race_result_slim の払戻 + race_result_flat の単勝オッズをマージ。"""
    if df.empty or "race_id" not in df.columns or "horse_id" not in df.columns:
        return df

    d = df.copy()
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)

    slim_cols = ["race_id", "horse_id", "win_payout", "place_payout"]
    if SLIM_PATH.exists():
        try:
            slim = pd.read_parquet(SLIM_PATH, columns=slim_cols)
            slim["race_id"] = slim["race_id"].astype(str)
            slim["horse_id"] = slim["horse_id"].astype(str)
            d = d.drop(columns=[c for c in slim_cols[2:] if c in d.columns], errors="ignore")
            d = d.merge(slim, on=["race_id", "horse_id"], how="left")
        except Exception as e:  # pragma: no cover
            logger.warning("race_result_slim 読込失敗: %s", e)

    for c in ("win_payout", "place_payout"):
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(np.int64)

    if years is None and "date" in d.columns:
        years = sorted(d["date"].astype(str).str.slice(0, 4).dropna().unique())
    odds_df = _load_odds_from_tables(years)
    if not odds_df.empty:
        d = d.drop(columns=["odds"], errors="ignore")
        d = d.merge(odds_df, on=["race_id", "horse_id"], how="left")
    elif "odds" not in d.columns:
        d["odds"] = np.nan

    return d


def prepare_roi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """オッズ <= MAX_ROI_ODDS の出走のみ払戻を回収率集計用に残す。"""
    d = df.copy()
    elig = roi_eligible_mask(d.get("odds", pd.Series(np.nan, index=d.index)))
    d["roi_eligible"] = elig.astype(np.int8)
    wp = pd.to_numeric(d.get("win_payout", 0), errors="coerce").fillna(0)
    pp = pd.to_numeric(d.get("place_payout", 0), errors="coerce").fillna(0)
    d["win_payout_roi"] = np.where(elig, wp, 0).astype(np.int64)
    d["place_payout_roi"] = np.where(elig, pp, 0).astype(np.int64)
    return d


def sire_roi_totals(sub: pd.DataFrame) -> tuple[int, float, float]:
    """種牡馬サブセットの (roi対象出走数, 単回収%, 複回収%)。"""
    if sub.empty:
        return 0, 0.0, 0.0
    if "roi_eligible" not in sub.columns:
        sub = prepare_roi_columns(sub)
    n = int(sub["roi_eligible"].sum())
    if n <= 0:
        return 0, 0.0, 0.0
    wr = float(sub["win_payout_roi"].sum()) / n
    pr = float(sub["place_payout_roi"].sum()) / n
    return n, round(wr, 1), round(pr, 1)
