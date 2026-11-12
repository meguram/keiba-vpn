"""V1 前走の再採点: 相手の後走が今回の日以降なら、特徴量に入らない（先読みしない）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features.article_check import prev_rescore as pr


def _runs(extra=None):
    """H が R1(1/1)→R3(3/1) に出る。R1 の相手 O は R2(2/1)・R4(4/1) に走る。"""
    rows = []

    def add(rid, date, hid, fp, t, fs=2):
        rows.append({"race_id": rid, "date": pd.Timestamp(date), "venue_code": "05", "surface": "芝", "distance": 1600, "field_size": fs,
                     "finish_position": fp, "horse_number": 1, "horse_id": hid, "time_sec": t})

    add("R1", "2022-01-01", "H", 2, 96.5)
    add("R1", "2022-01-01", "O", 1, 96.0)
    add("R2", "2022-01-08", "O", 1, 95.0)
    add("R2", "2022-01-08", "X", 2, 96.0)
    add("R3", "2022-01-15", "H", 1, 95.0)
    add("R3", "2022-01-15", "Y", 2, 96.0)
    add("R4", "2022-01-22", "O", 2, 95.0)
    add("R4", "2022-01-22", "Y", 1, 94.5)
    for e in extra or []:
        add(*e)
    return pd.DataFrame(rows)


def _row(f, rid, hid):
    return f[(f["race_id"] == rid) & (f["horse_id"] == hid)].iloc[0]


class PrevRescoreLeakTest(unittest.TestCase):
    def test_uses_opponent_run_before_today_only(self):
        f = pr.build(_runs())
        h3 = _row(f, "R3", "H")
        # R3 の日 (1/15) より前の O の後走は R2 (1/8) の 1 着だけ（R4 は 1/22 で未来）
        self.assertAlmostEqual(h3["pr_aft1"], (1.0 + pr.SHRINK_K * pr.PRIOR) / (1 + pr.SHRINK_K))
        # 次走以降 3 走まで（J=3）でも、R4 (1/22、未来) は入らない
        self.assertAlmostEqual(h3["pr_aft1_n3"], (1.0 + pr.SHRINK_K * pr.PRIOR) / (1 + pr.SHRINK_K))
        self.assertTrue(np.isnan(_row(f, "R1", "H")["pr_aft1"]))   # 前走が無い

    def test_future_results_do_not_change_past_row(self):
        base = pr.build(_runs())
        # O の R4 の結果を入れ替えても、R3 の H の特徴量は同じ
        alt = _runs()
        alt.loc[(alt["race_id"] == "R4") & (alt["horse_id"] == "O"), ["finish_position", "time_sec"]] = [1, 90.0]
        alt.loc[(alt["race_id"] == "R4") & (alt["horse_id"] == "Y"), ["finish_position", "time_sec"]] = [2, 95.0]
        b = pr.build(alt)
        cols = pr.OUT_COLS
        pd.testing.assert_series_equal(_row(base, "R3", "H")[cols], _row(b, "R3", "H")[cols])

    def test_same_day_opponent_run_not_used(self):
        # O が H の今回と同じ日 (R3 の日) に別のレースに走っても使わない
        base = pr.build(_runs())
        same = pr.build(_runs([("R3b", "2022-01-15", "O", 2, 99.0), ("R3b", "2022-01-15", "Z", 1, 98.0)]))
        # R2 を O が走らない設定にして、同日走が j=1 になるようにしても入らない
        no_r2 = _runs()
        no_r2 = no_r2[no_r2["race_id"] != "R2"]
        a = pr.build(no_r2)
        b = pr.build(pd.concat([no_r2, _runs([("R3b", "2022-01-15", "O", 2, 99.0), ("R3b", "2022-01-15", "Z", 1, 98.0)]).iloc[-2:]], ignore_index=True))
        for c in ("pr_aft1", "pr_aft1_n3"):
            self.assertAlmostEqual(_row(a, "R3", "H")[c], _row(b, "R3", "H")[c])
            self.assertAlmostEqual(_row(base, "R3", "H")[c], _row(same, "R3", "H")[c])

    def test_own_current_result_not_used(self):
        base = pr.build(_runs())
        alt = _runs()
        alt.loc[(alt["race_id"] == "R3") & (alt["horse_id"] == "H"), ["finish_position", "time_sec"]] = [2, 99.0]
        b = pr.build(alt)
        pd.testing.assert_series_equal(_row(base, "R3", "H")[pr.OUT_COLS], _row(b, "R3", "H")[pr.OUT_COLS])

    def test_lag2_excludes_self(self):
        # H の 2 走前 (R1) の相手として H 自身の後走 (R2) は数えない
        rows = [("R0", "2021-12-25", "H", 1, 90.0), ("R0", "2021-12-25", "O", 2, 91.0), ("R1", "2022-01-01", "H", 1, 90.0), ("R1", "2022-01-01", "P", 2, 91.0)]
        f = pr.build(_runs(rows))
        h3 = _row(f, "R3", "H")
        # L=2 は R1。相手 O の後走 R2 (1.0) だけ（H 自身の R3 以降・R2 は含まない）
        self.assertFalse(np.isnan(h3["pr_aft_l3"]))

    def test_margin_and_aftermath_ordering(self):
        f = pr.build(_runs())
        h3 = _row(f, "R3", "H")
        # H は R1 で 0.5 秒差の 2 着 → own = exp(-1)
        self.assertAlmostEqual(h3["pr_own1"], float(np.exp(-1.0)))
        self.assertAlmostEqual(h3["pr_rescore1"], h3["pr_own1"] * h3["pr_aft1"] / pr.PRIOR)


if __name__ == "__main__":
    unittest.main()
