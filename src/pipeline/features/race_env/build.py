"""レース環境 ENV（層 A）をレース単位の表にする。学習（2016〜2025）と推論（1 日分）で同じ関数を使う。"""

from __future__ import annotations

import pandas as pd

from .asof import load_results, race_summary
from .bias import bias_features, bias_race
from .going_level import going_level_features, going_level_race


def build_env_race(rs: pd.DataFrame) -> pd.DataFrame:
    """rs: race_summary（過去を含む全期間）。race_id と env_* 列を返す。"""
    dev = going_level_race(rs)
    dev = bias_race(dev)
    gl = going_level_features(dev)
    bs = bias_features(dev)
    return gl.merge(bs.drop(columns=[c for c in bs.columns if c.endswith("_prior_gap") or c.endswith("_day_n")]), on="race_id", how="left")


def build_from_tables(years) -> pd.DataFrame:
    return build_env_race(race_summary(load_results(years)))
