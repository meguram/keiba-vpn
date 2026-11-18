"""走法軸の点数別 勝率・複勝率の集計（期待率の表・区分）と、HTML 側の高−低の差の確認。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import axis_score_curves as asc
from src.scripts.docs import build_axis_score_curves_html as page


def test_dist_band_and_class():
    assert asc.dist_band(1200) == "〜1200m" and asc.dist_band(1600) == "1500〜1700m" and asc.dist_band(2500) == "2500m〜"
    assert asc.class_of("G1") == "OP以上" and asc.class_of("2勝") == "2・3勝" and asc.class_of("障害") is None


def test_place_cut_by_field_size():
    assert list(asc.place_cut(np.array([5, 7, 8, 18]))) == [2, 2, 3, 3]


def test_expected_tables_follow_popularity_and_sum_near_one():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(3000):
        n = 12
        order = rng.permutation(n)             # 人気順と無関係なら、どの人気順も同じ率になる
        for pop in range(1, n + 1):
            rows.append((r, pop, n, order[pop - 1] + 1))
    t = pd.DataFrame(rows, columns=["race_id", "popularity", "field_size", "finish_position"])
    tw, tp = asc.expected_tables(t)
    assert abs(tw[12, 1:13].sum() - 1.0) < 0.05
    assert abs(tp[12, 1:13].mean() - 3 / 12) < 0.02
    assert tw[12, 5] > 0 and tw[4, 1] == 0           # 5 頭未満は表に無い


def test_side_stats_adjusted_difference():
    n = [100] * 10
    p = [20] * 4 + [30] + [40] * 5        # 複勝数（低 20%・高 40%）
    arr = [n, [0] * 10, p, [0] * 10, p]       # 期待 = 実績 → 補正後 0
    s = page.side_stats(arr)
    assert abs(s["raw"] - 0.20) < 1e-9 and abs(s["adj"]) < 1e-9
    assert page.side_stats([[10] * 10] + [[1] * 10] * 4) is None     # 走数が少なければ出さない


def test_side_stats_excludes_middle_and_signs_z():
    n = [100] * 10
    p = [20] * 4 + [99] + [40] * 5        # 5 点台の区間は高低どちらにも入らない
    arr = [n, [0] * 10, p, [0] * 10, [30] * 10]   # 期待は一律 30% → 補正後 = 実績の差と同じ
    s = page.side_stats(arr)
    assert abs(s["raw"] - 0.20) < 1e-9 and s["z"] > 0
    arr2 = [n, [0] * 10, p[::-1], [0] * 10, [30] * 10]
    assert page.side_stats(arr2)["z"] < 0


def test_build_embeds_constants_and_escapes_script_end():
    c = {"A": [[200] * 10, [20] * 10, [60] * 10, [20.0] * 10, [60.0] * 10], "B": [[50] * 10, [5] * 10, [15] * 10, [5.0] * 10, [15.0] * 10]}
    d = {"meta": {"built": "2026-11-18", "exp_years": [2016, 2023], "periods": {"A": ["2024-01-01", "2025-12-31"], "B": ["2026-01-01", "2026-12-31"]},
                  "runs": {"A": 2000, "B": 500}, "axis_runs": {"ts": {"A": 2000, "B": 500}}, "course_min_runs": 400, "history_mtime": "2026-11-12T00:00:00"},
         "axes": [{"key": "ts", "label": "</script>TS", "high": "高", "low": "低"}],
         "dims": {"all": {"label": "全体", "groups": [{"key": "全体", "label": "全体", "data": {"ts": c}}]}}}
    h = page.build(d)
    assert "__MINN__" not in h and f"SIDE={page.MIN_SIDE}" in h and f"MINN={page.MIN_N}" in h
    assert h.count("</script>") == 1
