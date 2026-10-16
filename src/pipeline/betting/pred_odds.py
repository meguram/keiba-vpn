"""予測オッズで馬券を決めるための部品（そのレースのオッズ関連情報を一切使わない）。

- オッズ予測: 検証年より前の年だけで学習（ローリング）。特徴量は履歴・血統・コース・調教などで、オッズ・人気を含まない（``assert_no_odds_features``）。
- レース内の整合: 予測した単勝オッズを、Σ(1/オッズ) = 払戻率になるよう比例調整（市場の構造だけを使い、実際のオッズは見ない）。
- 馬連・ワイド・馬単の予測オッズ: 予測した単勝の暗黙確率から割引 Harville で組の確率を作り、券種ごとの定数 c_type（過去年の 実際のオッズ × 市場の暗黙確率 の中央値 = 控除と歪みを含む）で割る。
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

ODDS_PATTERN = re.compile(r"odds|オッズ|popular|ninki|人気|pop_|payout|払戻|tansho|fukusho|market|win_prob", re.I)
PAYBACK_WIN = 0.797


def assert_no_odds_features(cols: Sequence[str]) -> None:
    bad = [c for c in cols if ODDS_PATTERN.search(c)]
    if bad:
        raise ValueError(f"オッズ・人気に関わる特徴量が含まれています: {bad}")


def normalize_win_odds(race_ids: np.ndarray, odds: np.ndarray, payback: float = PAYBACK_WIN) -> np.ndarray:
    """レースごとに Σ(1/オッズ) = payback になるよう、予測した単勝オッズを比例調整（下限 1.0）。"""
    inv = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.0, None)
    s = pd.Series(inv).groupby(np.asarray(race_ids)).transform("sum").to_numpy()
    return np.clip(1.0 / (inv / s * payback), 1.0, None)


def pair_constants(tickets: pd.DataFrame, types=("umaren", "wide", "umatan")) -> dict[str, float]:
    """過去年の買い目の表（実際のオッズと、実際の単勝から作った市場確率 p_mkt）から、券種ごとの c_type = 中央値(オッズ × p_mkt)。予測オッズ = c_type / P(組)。"""
    out = {}
    for t in types:
        d = tickets[tickets["type"] == t]
        out[t] = float((d["odds"] * d["p_mkt"]).median()) if len(d) else {"umaren": 0.775, "wide": 0.775, "umatan": 0.75}[t]
    return out
