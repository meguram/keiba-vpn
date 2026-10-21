"""v6 の候補: 新しい知見（追走難度 v2・位置取り予想・ラップ予想）を v5 の列セットに足す群の部品（純粋関数）。

どの群も出走表の段階で分かる値（予測）だけから作る。入力の EASE2（追走難度 v2）と LAP（ラップ予想）は、年ごとの
leave-one-year-out の予測（その年のラベルは学習に使わない）。レース内の相対化は同じレースの出走馬の予測値どうしの比較で、結果は使わない。

- ``race_rel``: 列ごとのレース内の順位（0〜1）と、レース平均との差。
- ``lap_entropy``: ラップ予想のシナリオ確率（スロー/ミドル/ハイ）の不確実性（エントロピー、最大確率、ハイ−スロー）。
- ``interactions``: 位置取り予想 × シナリオ、追走難度 × シナリオの交互作用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
PROB = ["lp_p_slow", "lp_p_mid", "lp_p_high"]


def race_rel(df: pd.DataFrame, cols: list[str], prefix: str = "v6r_") -> pd.DataFrame:
    """cols の各列について、レース内の順位（昇順で 0〜1、欠損は NaN）と、レース内の平均との差を返す。"""
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n = s.groupby(df["race_id"], sort=False).transform("count")
        r = s.groupby(df["race_id"], sort=False).rank(method="average")
        out[f"{prefix}{c}_rk"] = ((r - 1) / (n - 1).replace(0, np.nan)).astype("float32")
        out[f"{prefix}{c}_gap"] = (s - s.groupby(df["race_id"], sort=False).transform("mean")).astype("float32")
    return pd.DataFrame(out, index=df.index)


def lap_entropy(lap: pd.DataFrame) -> pd.DataFrame:
    """シナリオ確率（3 列）から、エントロピー・最大確率・ハイ−スロー。"""
    P = lap[PROB].to_numpy(dtype=float)
    P = np.clip(P, 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    ent = -(P * np.log(P)).sum(axis=1)
    return pd.DataFrame({"v6_lp_entropy": ent.astype("float32"), "v6_lp_pmax": P.max(axis=1).astype("float32"),
                         "v6_lp_hi_minus_slow": (P[:, 2] - P[:, 0]).astype("float32")}, index=lap.index)


def interactions(df: pd.DataFrame) -> pd.DataFrame:
    """df は pf_early（位置取り予想: 0=先頭〜1=後方）、EASE2_drain / EASE2_capacity、lp_* を持つ。"""
    hs = (pd.to_numeric(df["lp_p_high"], errors="coerce") - pd.to_numeric(df["lp_p_slow"], errors="coerce")).astype(float)
    pf = pd.to_numeric(df["pf_early"], errors="coerce")
    pf_rk = pf.groupby(df["race_id"], sort=False).rank(method="average", pct=True)
    out = pd.DataFrame(index=df.index)
    out["v6_front_x_pace"] = ((0.5 - pf_rk) * hs).astype("float32")                 # 前に行く馬ほど、ハイ寄りで不利（負）
    out["v6_pf_x_f3"] = (pf * pd.to_numeric(df["lp_exp_f3_dev"], errors="coerce")).astype("float32")
    out["v6_drain_x_hs"] = (pd.to_numeric(df["EASE2_drain"], errors="coerce") * hs).astype("float32")
    out["v6_cap_x_hs"] = (pd.to_numeric(df["EASE2_capacity"], errors="coerce") * hs).astype("float32")
    out["v6_drain_x_l3"] = (pd.to_numeric(df["EASE2_drain"], errors="coerce") * pd.to_numeric(df["lp_exp_l3_dev"], errors="coerce")).astype("float32")
    return out
