"""A-1 馬場の水準: 勝ち馬と 2 着の時計を、同じ場・面・距離・クラスの「それより前の日」の基準と比べた差を、日・場・面の単位で持つ。

差は秒（負なら基準より速い）。時計全体・前半・上がり 3F の 3 本。勝ち馬だけで決めず 2 着も見る（予想家の観点）。
出力は env_gl_*（前日まで _prior、同日の前の R まで _day）。そのレース自身の結果は入らない。
"""

from __future__ import annotations

import pandas as pd

from .asof import prior_and_day, with_par

GL_COLS = ["dev_time", "dev_front", "dev_l3f"]


def going_level_race(rs: pd.DataFrame) -> pd.DataFrame:
    """rs: race_summary。レースごとの基準との差（このレース自身の値。特徴量にはせず、集計の材料にだけ使う）。"""
    rs = with_par(rs, "top2_time", "time")
    rs = with_par(rs, "top2_front", "front")
    rs = with_par(rs, "top2_l3f", "l3f")
    return rs


def going_level_features(rs_dev: pd.DataFrame) -> pd.DataFrame:
    return prior_and_day(rs_dev, GL_COLS, "env_gl")
