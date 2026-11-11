"""V2 枠×脚質の前週バイアス: 当日の週以降の結果は特徴量に入らない。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features.article_check import wkbias as wb


def _day(date, n=6, win_inner=True, tag="a"):
    rows = []
    for i in range(n):
        rows.append({"race_id": f"{date}{tag}", "horse_id": f"{tag}{date}{i}", "date": pd.Timestamp(date), "venue_code": "05", "surface": "芝", "distance": 1600,
                     "field_size": n, "horse_number": i + 1, "finish_position": (i + 1) if win_inner else (n - i), "a_style": 0.2})
    return rows


def _frame(dates, **kw):
    rows = []
    for dt in dates:
        rows += _day(dt, **kw)
    return wb.prepare(pd.DataFrame(rows))


def _val(f, d, col, date, gate):
    m = d.index[(d["date"] == pd.Timestamp(date)) & (d["g3"] == gate)][0]
    return f.loc[m, col]


class WkBiasLeakTest(unittest.TestCase):
    def test_same_week_and_future_not_used(self):
        # 2022-01-08(土)・01-09(日) は同じ週。前週 01-01(土)・01-02(日)
        d = _frame(["2022-01-01", "2022-01-02", "2022-01-08", "2022-01-09", "2022-01-15"])
        f = wb.build(d)
        # 01-09 の行は、同じ週の 01-08 を使わず 01-01/01-02 だけ
        v = _val(f, d, "wkb_gate", "2022-01-09", 0)
        # 内枠(g3=0)の 2 頭 × 2 日 = 4 件が 3 着内かつ期待を上回る
        self.assertGreater(v, 0)
        d2 = d.copy()
        d2.loc[d2["date"] == pd.Timestamp("2022-01-08"), "finish_position"] = d2.loc[d2["date"] == pd.Timestamp("2022-01-08"), "field_size"] + 0   # 同週の結果を壊す
        d2.loc[d2["date"] == pd.Timestamp("2022-01-15"), "finish_position"] = 99
        d2 = wb.prepare(d2.drop(columns=["g3", "s3", "exc", "qdate"]))
        f2 = wb.build(d2)
        self.assertAlmostEqual(_val(f, d, "wkb_gate", "2022-01-09", 0), _val(f2, d2, "wkb_gate", "2022-01-09", 0))
        self.assertAlmostEqual(_val(f, d, "wkb_c_gate", "2022-01-09", 0), _val(f2, d2, "wkb_c_gate", "2022-01-09", 0))

    def test_previous_week_is_used(self):
        d = _frame(["2022-01-01", "2022-01-02", "2022-01-08"])
        f = wb.build(d)
        # 前週の結果は内枠が有利 → 翌週の内枠は正、外枠は負
        self.assertGreater(_val(f, d, "wkb_gate", "2022-01-08", 0), 0)
        self.assertLess(_val(f, d, "wkb_gate", "2022-01-08", 2), 0)
        self.assertEqual(_val(f, d, "wkb_gate", "2022-01-01", 0), 0.0)   # 前週が無い → 0

    def test_window_drops_old_results(self):
        d = _frame(["2022-01-01", "2022-03-05"])   # 63 日後は短い窓の外
        f = wb.build(d)
        self.assertEqual(_val(f, d, "wkb_gate", "2022-03-05", 0), 0.0)
        self.assertGreater(_val(f, d, "wkb_c_gate", "2022-03-05", 0), 0)   # 長い窓には入る


if __name__ == "__main__":
    unittest.main()
