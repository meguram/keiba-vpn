"""推論用: 出走馬の馬別調教（horse_training の記録）から、学習時と同じ追い切り特徴 TL_A を作る。

学習時（tp-19）は全馬の追い切り表から (種別, 所属, 強度, 月) ごとの参照時計を作っていた。推論では、その参照表を保存したもの
（`build_reference_table` で作る小さな表）を使う。参照表に無い月は、その種別で直近の月の値で補う。
馬 ID（horse_id）だけをキーにする。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features import training_laps as tl
from src.pipeline.features import training_patterns as tp

A_PREFIXES = ("f_", "w1_", "w2_")
A_EXTRA = ["f_untimed_later", "n_work_14", "n_hard_14", "mean_t4z_14", "best_t1z_14", "n_work_28", "slope_share_28", "best_t1z_28", "has_training"]


def tl_a_columns(columns) -> list[str]:
    """学習時（tp-19 notebook）の TL_A の列の決め方と同じ。"""
    cols = [c for c in columns if c.startswith(A_PREFIXES) and c != "f_untimed_later"] + A_EXTRA
    return [c for c in dict.fromkeys(cols) if c in set(columns) and c not in ("horse_id", "race_date")]


def build_reference_table(laps: pd.DataFrame, window_months: int = 12, min_n: int = 200) -> pd.DataFrame:
    """追い切り表（training_laps.parquet）から参照表を作る。毎月作り直せばよい小さな表。"""
    return tp.build_reference(tp.prepare(laps), window_months, min_n)


def records_to_laps(records: dict[str, dict[str, Any] | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hid, rec in records.items():
        if rec:
            rec = dict(rec, horse_id=rec.get("horse_id") or hid)
            rows += tl.record_to_rows(rec)
    if not rows:
        return pd.DataFrame()
    return tl.add_lap_features(pd.DataFrame(rows))


def tl_a_for_race(records: dict[str, dict[str, Any] | None], race_date: str, ref: pd.DataFrame) -> pd.DataFrame:
    """race_date のレースに出る馬（records のキー）の TL_A。index = horse_id。調教の記録が無い馬の行は作らない（欠損扱い）。
    records は馬の全履歴（過去レースの調教を含む）を持つことが望ましい（馬自身との差が過去の最終追い切りの平均を使うため）。"""
    laps = records_to_laps(records)
    if laps.empty:
        return pd.DataFrame()
    laps = laps[laps["race_date"] <= race_date]                        # 当該レース以前の記録だけ
    t = tp.apply_reference(tp.prepare(laps), ref, asof=True)
    feat = tp.add_self_relative(tp.race_features(t))
    f = feat[feat["race_date"] == race_date]
    cols = tl_a_columns(f.columns)
    return f.set_index("horse_id")[cols].astype("float32")
