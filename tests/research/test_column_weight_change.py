"""めぐコラム『馬体重の大幅変化』の、体重の履歴と大幅増の 3 分類（src/research/columns/weight_change.py）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.columns import weight_change as wc


def test_history_uses_only_past_runs():
    c = pd.DataFrame({"horse_id": "h", "race_id": [f"r{i}" for i in range(5)],
                      "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
                      "weight": [470, 480, np.nan, 460, 490], "jra_fin": [1, 5, 2, 3, 1]})
    h = wc.history(c).set_index("race_id")
    assert np.isnan(h.loc["r0", "best"]) and np.isnan(h.loc["r0", "prev_w"])
    assert h.loc["r1", "best"] == 470 and h.loc["r1", "max_prev"] == 470
    assert h.loc["r3", "prev_w"] == 480                    # 体重の無い出走は飛ばして直前の体重
    assert h.loc["r3", "best"] == 470                      # 2 着の出走は体重が無いので数えない
    assert h.loc["r4", "best"] == 465 and h.loc["r4", "n_best"] == 2 and h.loc["r4", "max_prev"] == 480


def test_classify():
    d = pd.DataFrame([
        # w, gain, best, max_prev, prev_w, rest, age
        (480, 12, 482, 490, 468, 28, 5),        # ベスト ±4 に戻り、前走は軽かった
        (500, 14, 480, 490, 486, 90, 5),        # 休み明けで過去最高を更新 → 成長
        (470, 10, np.nan, 466, 460, 28, 3),     # 3 歳で過去最高を更新 → 成長
        (500, 12, 480, 496, 488, 28, 5),        # 休み明けでも若駒でもなく過去最高を更新 → 太め
        (470, 10, np.nan, 480, 460, 28, 5),     # ベスト無し・過去最高も超えない → その他
        (470, 1, 470, 480, 469, 28, 5),         # 変わらず
        (470, 6, 470, 480, 464, 28, 5),         # 対象外
    ], columns=["w", "gain", "best", "max_prev", "prev_w", "rest", "age"])
    got = wc.classify(d).tolist()
    assert got[:6] == [wc.BACK, wc.GROW, wc.GROW, wc.FAT, wc.OTHER, wc.SAME]
    assert pd.isna(got[6])
    # 閾値を変えると区分が変わる（+12 以上なら 3 歳の +10 は対象外）
    assert pd.isna(wc.classify(d, gain_min=12).iloc[2])
