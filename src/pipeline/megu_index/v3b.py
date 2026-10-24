"""めぐ指数 v3b: 履歴の集約層（UI・めぐ印・成長曲線・v6 以降の特徴量候補用）。

v3a の値そのものは変えない（V5 は v3a で学習済みで、値を変えると学習時と推論時がずれる）。
v3b は『v3a の走ごとの指数を、履歴として集めるときの扱い』だけを変える追加層。
  (a) 大差負けの下限: 同じレースの勝ち馬の指数 − FLOOR 点（約 4 秒）より下の値は引き上げる
  (b) 低信頼セル（基準タイムの薄いセル）の走は、集約での重みを下げる（信頼度つきの重み）
  (c) 新しい走ほど重い（減衰 DECAY）加重 Huber で集約する
  (d) excusable（不利などで一過性と判定された走）の扱いは引数（既定は除かない。tp-69 の結論）
根拠と検証: docs/html/modeling/chapters/index_variance.html（tp-65）、megu_excusable.html（tp-69）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOOR = 40.0        # 勝ち馬の指数 − 40 点（約 4 秒）を下限にする（tp-65: めぐ・netkeiba とも +0.006）
DECAY = 0.65        # 新しい走ほど重い（直近 5 走）
N_RUNS = 5
HUBER_C = 1.345
LOW_CONF_WEIGHT = 0.5   # 低信頼セルの走の重み（tp-69: 0.0 は悪化、0.5 が最良、1.0 = 重みなし。次走予測の対数尤度 +0.0036）
EXCUSABLE_WEIGHT = 1.0  # excusable と判定された走の重み。既定は 1.0（除かない）: tp-69 で、除外すると次走予測が悪化した（0.0〜0.5 は −0.0003〜−0.007）。判定は表示用
SCALE = 11.5            # 指数の散らばり（2020〜2021 年の相対指数の MAD×1.4826）


def floor_by_winner(x: pd.Series, race_id: pd.Series, finish_position: pd.Series, floor: float = FLOOR) -> pd.Series:
    """同じレースの勝ち馬（着順 1）の指数 − floor を下限にする。勝ち馬の指数が無いレースは下限なし。"""
    w = x.where(finish_position == 1).groupby(race_id).transform("max")
    return pd.Series(np.where(w.notna(), np.maximum(x, w - floor), x), index=x.index)


def weighted_huber(M: np.ndarray, w0: np.ndarray, scale: float, c: float = HUBER_C, iters: int = 8) -> np.ndarray:
    """行ごとの加重 Huber の位置推定。M: (n, k) 新しい順、欠損は NaN。w0: (n, k) の事前の重み（減衰×信頼度×excusable）。"""
    ok = ~np.isnan(M) & (w0 > 0)
    Mz = np.nan_to_num(M)
    wz = np.where(ok, w0, 0.0)
    den = wz.sum(axis=1)
    mu = np.where(den > 0, (wz * Mz).sum(axis=1) / np.maximum(den, 1e-12), np.nan)
    for _ in range(iters):
        r = (Mz - np.nan_to_num(mu)[:, None]) / scale
        h = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        w = wz * h
        d = w.sum(axis=1)
        mu = np.where(d > 0, (w * Mz).sum(axis=1) / np.maximum(d, 1e-12), np.nan)
    return mu


def run_weights(k: int = N_RUNS, decay: float = DECAY) -> np.ndarray:
    return decay ** np.arange(k)


def aggregate(values: list[float | None], low_conf: list[bool | None] | None = None, excusable: list[bool | None] | None = None,
              scale: float = SCALE, low_conf_weight: float = LOW_CONF_WEIGHT, excusable_weight: float = EXCUSABLE_WEIGHT) -> float | None:
    """1 頭の直近の走（新しい順）の指数から v3b の集約値を返す。values は下限（floor_by_winner）を適用済みの値。
    scale は指数の散らばりの目安（MAD×1.4826。2020〜2021 年の相対指数で約 11.5）。全部が除外・欠損なら None。"""
    k = min(len(values), N_RUNS)
    if k == 0:
        return None
    M = np.array([[np.nan if v is None else float(v) for v in values[:N_RUNS]] + [np.nan] * (N_RUNS - k)], dtype=float)
    w = run_weights()[None, :].copy()
    if low_conf:
        for i, lc in enumerate(low_conf[:N_RUNS]):
            if lc:
                w[0, i] *= low_conf_weight
    if excusable:
        for i, ex in enumerate(excusable[:N_RUNS]):
            if ex:
                w[0, i] *= excusable_weight
    out = weighted_huber(M, w, scale)[0]
    return None if np.isnan(out) else float(out)


def asof_matrix(x: pd.Series, groups: pd.Series, k: int = N_RUNS) -> np.ndarray:
    """各行の『その走より前の k 走』（新しい順）。x と groups は 馬×日付 で並べ替え済みであること。"""
    return np.column_stack([x.groupby(groups).shift(i).to_numpy(dtype=float) for i in range(1, k + 1)])
