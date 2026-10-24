"""コース概形アトラスの生成のテスト（実形状の座標変換・経路・HTML の構造）。"""

from __future__ import annotations

import json
import re

import numpy as np

from src.scripts.docs import build_course_atlas as A


def _routes():
    return json.loads(A.ROUTES.read_text(encoding="utf-8"))


def test_frame_fits_map_and_puts_stand_below_goal():
    R = _routes()
    for vc, v in R["venues"].items():
        fr = A.Frame(v)
        for c in v["circuits"].values():
            xy = fr.xy(np.array(c["pts_q"]))
            assert xy[:, 0].min() >= -1 and xy[:, 0].max() <= A.MAP_W + 1, vc
            assert xy[:, 1].min() >= -1 and xy[:, 1].max() <= A.MAP_H + 1, vc
            if c.get("stand_pts"):
                sp = fr.xy(np.array(c["stand_pts"]))
                goal = fr.xy(np.array(c["goal_xy"]))[0]
                near = sp[int(np.argmin(np.hypot(*(sp - goal).T)))]
                assert near[1] > goal[1] - 5, (vc, "ゴールにいちばん近いスタンドは、ゴールより下（画面の下側）にそろえる")
            break


def test_course_route_and_confidence_for_known_courses():
    R = _routes()
    v = R["venues"]["05"]
    d1600 = v["courses"]["ダート||1600"]
    route = A.course_route(v["circuits"]["ダート|"], d1600)
    # 引込み線の先端から始まり、ゴール（pts_q[0]）で終わる
    assert np.allclose(route[0], d1600["stub"][0]) and np.allclose(route[-1], v["circuits"]["ダート|"]["pts_q"][0])
    conf, why = A.course_confidence(d1600)
    assert conf == "実形状" and "引込み線" in why
    t2000 = v["courses"]["芝||2000"]                  # 東京芝2000m: 1コーナー奥のポケット（独立資料で修正）。位置の裏づけはあるが曲線は補間なので『推定』
    assert t2000["start"] == "pocket" and A.course_confidence(t2000)[0] == "推定" and t2000.get("override")
    assert A.course_confidence({"flags": ["x"], "start": "main"})[0] == "推定"


def test_html_has_all_venues_and_switch_rules():
    h = A.build_html()
    for name in ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"):
        assert f">{name}</label>" in h
    assert h.count('name="atl_v"') == 10
    assert ":has(input.v0:checked)" in h and ".vp0:has(input.t0_0:checked)" in h
    assert "<script" not in h                                  # JavaScript を使わない
    assert h.count("<svg") >= 25                               # 背景の定義 1 つ + コース（芝/ダート × 内外）ごとに 1 つ（地図の入れ子の svg を含む）
    assert h.count('<use href="#bg-') >= 24                    # 背景は競馬場ごとに 1 つの定義を <use> で呼ぶ
    assert re.search(r'<g id="bg-05">', h)
    assert "OpenStreetMap" in h
    assert "訂正" in h                                           # 第 2 ラウンドの訂正が表に出る
    assert "実形状" in h and "推定" in h
