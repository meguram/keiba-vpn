"""めぐコラム『輸送は成績を落とす？』の判定と距離（src/research/columns/transport_vs_stay.py）。"""

from __future__ import annotations

import pandas as pd

from src.research.columns import transport_vs_stay as tv


def test_great_circle_km_from_training_centers():
    # 大圏距離（道のりではない）。美浦→中山は 50km 弱、栗東→京都は 30km 前後、栗東→札幌は 1,000km 前後
    assert 40 < tv.KM[("MI", "中山")] < 55
    assert 20 < tv.KM[("KU", "京都")] < 40
    assert 950 < tv.KM[("KU", "札幌")] < 1050
    assert tv.KM[("MI", "阪神")] > tv.KM[("MI", "東京")]


def test_km_band_edges_are_left_closed():
    got = tv.km_band(pd.Series([0.0, 99.9, 100.0, 299.9, 300.0, 500.0, 800.0, 1200.0])).tolist()
    assert got == [tv.KM_BANDS[0], tv.KM_BANDS[0], tv.KM_BANDS[1], tv.KM_BANDS[1], tv.KM_BANDS[2], tv.KM_BANDS[3], tv.KM_BANDS[4], tv.KM_BANDS[4]]


def test_stay_class_uses_last_workout_location():
    last = pd.Series(["MI", "02", "KU", None, "10"])
    home = pd.Series(["MI", "MI", "MI", "KU", "KU"])
    # 所属トレセン → 輸送、競馬場・もう一方のトレセン → 滞在、記録なし → 判定不能
    assert tv.stay_class(last, home).tolist() == ["輸送", "滞在", "滞在", "判定不能", "滞在"]


def test_rotation_stay_needs_same_far_region_and_short_gap():
    d = pd.DataFrame({
        "horse_id": ["a", "a", "a", "b", "b"],
        "date": ["2024-07-01", "2024-07-20", "2024-09-30", "2024-07-01", "2024-07-15"],
        "venue": ["函館", "札幌", "札幌", "東京", "中山"],
        "km": [640.0, 790.0, 790.0, 85.0, 46.0],
    })
    # a: 函館→札幌（同じ北海道・19 日）は滞在の目安、その次は間隔が長い。b: 地元（300km 未満）は対象外
    assert tv.rotation_stay(d).tolist() == [False, True, False, False, False]
