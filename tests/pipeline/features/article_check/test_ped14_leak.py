"""V3 血統14項目: 産駒の成績は同日以降を含まず、出走数で減衰する。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features.article_check import ped14 as p


def _runs():
    rows = []

    def add(rid, date, hid, fp, surf="芝", dist=1600, fs=4):
        rows.append({"race_id": rid, "date": pd.Timestamp(date), "venue_code": "05", "surface": surf, "distance": dist, "field_size": fs,
                     "finish_position": fp, "horse_number": 1, "horse_id": hid, "time_sec": 90.0})

    # 兄 B1(同じ父 S) が 1/1 に 1 着、今回の A が 1/8 と 1/15 に走る。弟 B2 は 1/15(A と同日)に最下位
    add("R1", "2022-01-01", "B1", 1)
    add("R1", "2022-01-01", "Z1", 4)
    add("R2", "2022-01-08", "A", 2)
    add("R2", "2022-01-08", "Z2", 1)
    add("R3", "2022-01-15", "A", 2)
    add("R3", "2022-01-15", "B2", 4)
    add("R3", "2022-01-15", "Z3", 1)
    return pd.DataFrame(rows)


PARENTS = pd.DataFrame({"horse_id": ["A", "B1", "B2", "Z1", "Z2", "Z3"], "sire": ["S", "S", "S", "T", "T", "T"], "dsire": ["D"] * 6, "dam": ["M1", "M2", "M3", "M4", "M5", "M6"]})


def _row(f, rid, hid):
    return f[(f["race_id"] == rid) & (f["horse_id"] == hid)].iloc[0]


class Ped14Test(unittest.TestCase):
    def test_same_day_progeny_not_used_and_first_run_is_neutral(self):
        f = p.build(_runs(), PARENTS)
        # 1/8 の A: 父 S の産駒は B1 の 1 着だけ → pow > 0
        self.assertGreater(_row(f, "R2", "A")["ped14_sire_pow"], 0)
        # 1/15 の A: B2 の同日の最下位は使わない → 1/8 の A 自身の 2 着(0.667) と B1(1.0) の平均は 0.5 超
        self.assertGreater(_row(f, "R3", "A")["ped14_sire_pow"], 0)
        # 1/1 の B1: 過去の産駒なし → 0
        self.assertAlmostEqual(_row(f, "R1", "B1")["ped14_sire_pow"], 0.0)

    def test_future_results_do_not_change_past_row(self):
        base = p.build(_runs(), PARENTS)
        alt = _runs()
        alt.loc[(alt["race_id"] == "R3") & (alt["horse_id"] == "B2"), "finish_position"] = 1
        b = p.build(alt, PARENTS)
        cols = p.raw_cols() + p.gated_cols()
        pd.testing.assert_series_equal(_row(base, "R3", "A")[cols], _row(b, "R3", "A")[cols])

    def test_gate_decays_with_runs(self):
        f = p.build(_runs(), PARENTS)
        a3 = _row(f, "R3", "A")   # 2 走目
        self.assertAlmostEqual(a3["ped14_sire_pow_g"], a3["ped14_sire_pow"] / (1 + 1 / p.TAU))
        self.assertEqual(a3["ped14_nruns"], 1)

    def test_surface_pref_sign(self):
        r = _runs()
        r = pd.concat([r, pd.DataFrame([{"race_id": "R0", "date": pd.Timestamp("2021-12-25"), "venue_code": "05", "surface": "ダート", "distance": 1600, "field_size": 4,
                                           "finish_position": 4, "horse_number": 1, "horse_id": "B0", "time_sec": 90.0}])], ignore_index=True)
        pt = pd.concat([PARENTS, pd.DataFrame({"horse_id": ["B0"], "sire": ["S"], "dsire": ["D"], "dam": ["M0"]})], ignore_index=True)
        f = p.build(r, pt)
        self.assertGreater(_row(f, "R2", "A")["ped14_sire_pref"], 0)   # 芝 1 着・ダート最下位 → 芝向き


if __name__ == "__main__":
    unittest.main()
