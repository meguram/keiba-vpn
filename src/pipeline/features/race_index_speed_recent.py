"""
race_index の ``speed_recent``（近3走のスピード指数リスト）を数値列に展開する。

スクレイパは ``[sk__index1, sk__index2, sk__index3]`` をリスト（またはその JSON 文字列）で保存する。
欠損セルはパーサ上 ``0`` だが、特徴量・学習用には **0 を null（Float64 の NA）** にして「指数0」と「データ無し」を混同しない。

展開後の列名（フラット結合・LayerA）::

    speed_recent_1  # 直近（1走前相当のセル = sk__index1）
    speed_recent_2
    speed_recent_3

特徴量ストア用の stem は ``idx_speed_recent_1`` など（``pipeline/raw_table_features.py``）。
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any

import pandas as pd


def parse_speed_recent_list(val: Any) -> list[float | int]:
    """セル値を長さ不定のリストに正規化（最大3要素を想定）。"""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        raw = list(val)
    elif isinstance(val, str):
        s = val.strip()
        if not s or s == "[]":
            return []
        try:
            raw = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            try:
                raw = json.loads(s)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, (list, tuple)):
            return []
        raw = list(raw)
    else:
        return []

    out: list[float | int] = []
    for x in raw[:3]:
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                out.append(0)
            else:
                out.append(int(float(x)))
        except (TypeError, ValueError):
            out.append(0)
    return out


def _slot_value(lst: list[float | int], index: int) -> float | None:
    if index >= len(lst):
        return None
    v = lst[index]
    if v == 0:
        return None
    return float(v)


def expand_speed_recent_column(
    series: pd.Series,
    *,
    dest_prefix: str = "speed_recent",
    n_slots: int = 3,
) -> dict[str, pd.Series]:
    """1列 ``speed_recent`` から ``{dest_prefix}_1`` … の Series 辞書を返す（元列は削除側で対応）。"""
    parsed = series.map(parse_speed_recent_list)
    out: dict[str, pd.Series] = {}
    for i in range(n_slots):
        name = f"{dest_prefix}_{i + 1}"
        out[name] = parsed.map(lambda L, idx=i: _slot_value(L, idx)).astype("Float64")
    return out


def expand_speed_recent_in_dataframe(
    df: pd.DataFrame,
    col: str = "speed_recent",
    *,
    dest_prefix: str = "speed_recent",
) -> pd.DataFrame:
    """``col`` を展開列に置き換えた DataFrame を返す（``col`` は落とす）。"""
    if col not in df.columns:
        return df
    extra = expand_speed_recent_column(df[col], dest_prefix=dest_prefix)
    out = df.drop(columns=[col])
    for k, s in extra.items():
        out[k] = s
    return out
