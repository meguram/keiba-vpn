"""v5 の推論用: 追い切りサイクルの推移 CY（tp-40、列名は wc_*）を、当日の出走馬に付ける。

学習側（tp40_condition.cycle_group）の関数をそのまま使う。入力は TL_A（serving_stage_rel.tl_columns_for_races）と同じ『馬別調教ページの記録』と
『(種別, 所属, 強度) の直近 12 か月の分布の基準表（tl_reference）』だけで、馬ごとの履歴状態は要らない。標準化は TL_A と同じ asof 版（調教日より前の月の分布）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.pipeline.features import tp40_condition as C
from src.pipeline.features import training_patterns as TP
from src.pipeline.features import training_serving as ts

CY_METRICS = {"g": "cc", "g_t1": "c1", "g_acc": "ca"}
PREFIX = C.GROUP_PREFIX["CY"]


def cy_for_race(records: dict[str, dict[str, Any] | None], race_date: str, ref: pd.DataFrame) -> pd.DataFrame:
    """race_date のレースに出る馬の CY（index = horse_id、列名は wc_*）。調教の記録が無い馬の行は作らない（欠損扱い）。"""
    laps = ts.records_to_laps(records)
    if laps.empty:
        return pd.DataFrame()
    laps = laps[laps["race_date"] <= race_date]
    if laps.empty:
        return pd.DataFrame()
    t = TP.apply_reference(TP.prepare(laps), ref, asof=True)
    T = C.standardized_workouts(t)
    if T.empty:
        return pd.DataFrame()
    cy = C.cycle_group(T, CY_METRICS)
    f = cy[cy["race_date"].astype(str) == race_date]
    cols = [c for c in f.columns if c not in C.KEYC]
    return f.set_index("horse_id")[cols].add_prefix(PREFIX)


def cy_columns_for_races(races: pd.DataFrame, records: dict[str, dict[str, Any] | None], ref: pd.DataFrame) -> pd.DataFrame:
    """races: race_id・horse_id・date。戻り値: race_id・horse_id + wc_* 列。記録の無い馬は欠損の行。"""
    r = races[["race_id", "horse_id", "date"]].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    parts = []
    for d, g in r.groupby("date", sort=False):
        got = cy_for_race({h: records.get(h) for h in g["horse_id"].unique()}, d, ref)
        m = g[["race_id", "horse_id"]].merge(got.reset_index(), on="horse_id", how="left") if len(got) else g[["race_id", "horse_id"]].copy()
        parts.append(m)
    return pd.concat(parts, ignore_index=True) if parts else r[["race_id", "horse_id"]]
