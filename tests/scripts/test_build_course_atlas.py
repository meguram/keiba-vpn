"""コース概形アトラスの生成のテスト（座標の写像・HTML の構造）。"""

from __future__ import annotations

import math
import re

from src.scripts.docs import build_course_atlas as A


def test_layout_maps_goal_and_start():
    b = [524.0, 771.0, 1036.0, 1479.0, 1740.0, 2005.0]
    left = A.Layout(2083.1, b, clockwise=False)
    right = A.Layout(2083.1, b, clockwise=True)
    # ゴールは下の直線上。左回りは右寄り・右回りは左寄り（走る向きの前方）
    gx_l, gy_l = left.xy(0.0)
    gx_r, gy_r = right.xy(0.0)
    assert gy_l == A.YB and gy_r == A.YB
    assert abs((gx_l + gx_r) - (A.CXL + A.CXR)) < 0.2          # 左右対称
    # ゴール前の直線の始まり（q = b1）は直線の端、ゴール後の直線の終わり（q = L）はゴールのすぐ先
    x_start, _ = left.xy(524.0)
    assert abs(x_start - A.CXL) < 0.2
    # 走る向き（q が減る方向）に x が増える（左回りの下の直線）
    assert left.xy(300.0)[0] < left.xy(100.0)[0] < left.xy(0.0)[0]
    # 向正面は上の直線、弧の頂点は円の側面
    assert left.xy(1200.0)[1] == A.YT
    assert abs(left.xy(1740.0)[0] - (A.CXR + A.R * math.cos(math.radians(0)))) < 1.0           # 1C と 2C の境 = 弧の真横（右端）
    assert 0 <= left.xy(2083.0)[0] - gx_l < 80                                                  # ゴール後の直線は短い


def test_polyline_and_bounds_fallback():
    lay = A.Layout(1600.0, A.bounds_from_profile(None, 1600.0, 292.0), clockwise=True)
    pts = lay.polyline(1000.0, 0.0).split()
    assert len(pts) >= 2 and all(re.match(r"^-?\d+\.\d,-?\d+\.\d$", p) for p in pts)
    b = A.bounds_from_profile(None, 1600.0, 292.0)
    assert b == sorted(b) and b[-1] < 1600.0


def test_html_has_all_venues_and_switch_rules():
    h = A.build_html()
    for name in ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"):
        assert f">{name}</label>" in h
    assert h.count('name="atl_v"') == 10
    assert ":has(input.v0:checked)" in h and ".vp0:has(input.t0_0:checked)" in h
    assert "<script" not in h                                  # JavaScript を使わない
    assert h.count("<svg") >= 24                               # コース（芝/ダート × 内外）ごとに 1 つ
    assert "訂正" in h                                           # 第 2 ラウンドの訂正が表に出る
