"""A-4 馬場の偏り（前残り度）: 上位 3 頭の 4 角の相対位置が、そのレースのペース差から期待される位置より前か後ろか。

前傾なのに逃げ・先行が粘れば「前有利」、道中が緩んで前に残っただけなら偏りとは読まない（ペースで切り分ける）。
期待位置は、前半 3F−後半 3F と頭数の 1 次式で、そのレースの年より前の年だけで当てる。
前の年のデータが足りない年（最初の年）は欠損にする（同じ年の後のレースを係数に入れない）。
bias_front = 期待 − 実際（正なら前が有利）。出力は env_bias_*（前日まで _prior、同日の前の R まで _day）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import prior_and_day

BIAS_COLS = ["bias_front"]


def _fit(g: pd.DataFrame) -> tuple[float, float, float]:
    g = g.dropna(subset=["front_idx", "pace_diff"])
    X = np.c_[np.ones(len(g)), g["pace_diff"].to_numpy(), g["field_size"].to_numpy()]
    beta, *_ = np.linalg.lstsq(X, g["front_idx"].to_numpy(), rcond=None)
    return tuple(beta)


def bias_race(rs: pd.DataFrame) -> pd.DataFrame:
    rs = rs.copy()
    rs["year"] = rs["date"].dt.year
    rs["bias_front"] = np.nan
    for (surf, year), g in rs.groupby(["surface", rs["year"]]):
        past = rs[(rs["surface"] == surf) & (rs["year"] < year)].dropna(subset=["front_idx", "pace_diff"])
        if len(past) < 500:
            continue
        b0, b1, b2 = _fit(past)
        idx = g.index
        exp = b0 + b1 * rs.loc[idx, "pace_diff"] + b2 * rs.loc[idx, "field_size"]
        rs.loc[idx, "bias_front"] = exp - rs.loc[idx, "front_idx"]
    return rs


def bias_features(rs_bias: pd.DataFrame) -> pd.DataFrame:
    return prior_and_day(rs_bias, BIAS_COLS, "env_bias")
