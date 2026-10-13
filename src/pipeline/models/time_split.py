"""時系列 CV 分割（暦年 expanding window + purge/embargo）と学習対象年の制限。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from src.pipeline.models.dataset_split import race_calendar_year

# 2026 年以降はデータが不完全なため学習に使わない（確実に揃っている年のみ）。
TRAINING_YEARS: tuple[str, ...] = tuple(str(y) for y in range(2020, 2026))


def restrict_to_training_years(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    years: Sequence[str] = TRAINING_YEARS,
) -> pd.DataFrame:
    """race_id の開催年が学習対象年のものだけを残す。"""
    keep = df[race_id_col].map(race_calendar_year).isin(set(years))
    return df.loc[keep].copy()


@dataclass(frozen=True)
class Fold:
    fold_id: str
    valid_year: str
    train_index: np.ndarray
    valid_index: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp


def yearly_expanding_folds(
    df: pd.DataFrame,
    *,
    valid_years: Sequence[str] = ("2022", "2023", "2024"),
    first_train_year: str = "2020",
    purge_days: int = 7,
    embargo_days: int = 3,
    race_id_col: str = "race_id",
    date_col: str = "date",
) -> Iterator[Fold]:
    """
    検証を 1 暦年（季節を揃える）、学習をそれ以前の全期間とする expanding window。
    学習は valid 開始日の (purge_days + embargo_days) 日前より前に限る。
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} に欠損または変換不能な値が {int(dates.isna().sum())} 件あります")

    years = df[race_id_col].map(race_calendar_year)
    train_pool = years.isin(set(TRAINING_YEARS)) & (years >= first_train_year)
    gap = pd.Timedelta(days=purge_days + embargo_days)

    for vy in valid_years:
        if vy not in TRAINING_YEARS:
            raise ValueError(f"valid 年 {vy} は学習対象年 {TRAINING_YEARS} の外です")
        valid_start = pd.Timestamp(f"{vy}-01-01")
        train_end = valid_start - gap
        valid_mask = years == vy
        train_mask = train_pool & (dates < train_end)
        if not valid_mask.any() or not train_mask.any():
            raise ValueError(f"fold {vy}: 学習または検証の行が空です")
        yield Fold(
            fold_id=f"valid_{vy}",
            valid_year=vy,
            train_index=df.index[train_mask].to_numpy(),
            valid_index=df.index[valid_mask].to_numpy(),
            train_end=train_end,
            valid_start=valid_start,
        )
