"""tp-34 休養明け・レース間隔の特徴量: 先読みが無いこと（将来の結果を変えても過去行の特徴量が変わらない）と、定義の確認。"""

from __future__ import annotations

import warnings
import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs
from src.pipeline.features import tp34_rest as tr
from tests.pipeline.test_factory_self import _late_modified, _toy

warnings.filterwarnings("ignore")


def _laps(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for (h, d), _ in df.groupby(["horse_id", "date"]):
        for k in range(int(rng.integers(0, 6))):
            db = int(rng.integers(1, 30))
            rows.append({"horse_id": h, "train_date": (d - pd.Timedelta(days=db)).strftime("%Y-%m-%d"), "race_date": d.strftime("%Y-%m-%d"), "kind": ["slope", "wood", "poly"][k % 3],
                         "leg_color": ["一杯", "強め", "馬也"][k % 3], "days_before": db})
    return pd.DataFrame(rows)


class NoLeakTest(unittest.TestCase):
    def test_every_group_is_as_of(self):
        df = _toy()
        laps = _laps(df)
        cut = "2021-06-01"
        a = tr.build_all(df, laps)
        b = tr.build_all(_late_modified(df, cut), laps)
        early = (df["date"] < pd.Timestamp(cut)).to_numpy()
        self.assertEqual(set(a), set(tr.GROUPS))
        for g in a:
            x, y = a[g].loc[early].drop(columns=tr.KEY).reset_index(drop=True), b[g].loc[early].drop(columns=tr.KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(x, y, check_exact=False, atol=1e-5, obj=g)

    def test_same_day_results_are_not_used(self):
        df = _toy()
        day = df["date"].sort_values().iloc[len(df) // 2]
        d2 = df.copy()
        same = d2["date"] == day
        d2.loc[same, "finish_position"] = (d2.loc[same, "finish_position"] % 8) + 1
        a, b = tr.build_all(df), tr.build_all(d2)
        for g in a:
            x = a[g].loc[same.to_numpy()].drop(columns=tr.KEY).reset_index(drop=True)
            y = b[g].loc[same.to_numpy()].drop(columns=tr.KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(x, y, check_exact=True, obj=g)                 # 厳密一致（端数への痕跡も不可）


class FloatingPointLeakTest(unittest.TestCase):
    """『累積和 − 自分』の端数に今回の結果の痕跡が残らない（tp-34 で、前走 1 回の馬で今回勝ったときだけ厳密に 0 になる値を木モデルが拾った）。"""

    def test_excl_cum_is_bitwise_independent_of_the_current_value(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b = float(rng.random()), float(rng.random())
            for cur in (0.0, b, 1.0 / 3.0, 0.1, 7.0 / 9.0):
                d = pd.DataFrame({"horse_id": ["h", "h"], "fp": [a, cur]})
                self.assertEqual(float(fs.excl_cum(d, ["horse_id"], "fp")[1]), a)               # 厳密に前走の値そのもの（端数なし）
                d3 = pd.DataFrame({"horse_id": ["h", "h", "h"], "fp": [a, b, cur]})
                base = float(fs.excl_cum(d3, ["horse_id"], "fp")[2])
                d3b = d3.assign(fp=[a, b, 0.123456789])
                self.assertEqual(float(fs.excl_cum(d3b, ["horse_id"], "fp")[2]), base)           # 今回の値を変えても、ビットまで同じ

    def test_prev_fp_vs_mean_is_exactly_zero_for_second_start_regardless_of_result(self):
        df = _toy()
        f = tr.build_all(df, None, only=["R6_prev"])["R6_prev"]
        d = tr.prepare(df)
        second = (d.groupby("horse_id").cumcount() == 1).to_numpy()
        v = f.loc[second, "rs6_prev_fp_vs_mean"].to_numpy(dtype="float64")
        self.assertTrue(np.all(v == 0.0))                                                       # 今回の結果にかかわらず常に 0（端数なし）


class DefinitionTest(unittest.TestCase):
    def _runs(self, dates: list[str]) -> pd.DataFrame:
        df = _toy(n_races=4, n_horses=8)
        h = df["horse_id"].iloc[0]
        base = df[df["horse_id"] == h].iloc[:1]
        rows = []
        for i, dt in enumerate(dates):
            r = base.copy()
            r["date"] = pd.Timestamp(dt)
            r["race_id"] = f"2021050{i:04d}"[:12]
            r["finish_position"] = 1 + (i % 5)
            rows.append(r)
        return pd.concat(rows, ignore_index=True)

    def test_band_and_tataki(self):
        runs = self._runs(["2021-01-09", "2021-01-23", "2021-02-06", "2021-07-10", "2021-07-24", "2021-08-07", "2021-08-14"])
        d = tr.prepare(runs)
        self.assertTrue(np.isnan(d["gap"].iloc[0]))
        self.assertEqual(list(d["band"].iloc[1:].astype(int)), [1, 1, 5, 1, 1, 0])           # 14,14 日=2 週(1)、154 日=14〜26 週(5)、14,14 日、7 日=連闘〜1 週(0)
        self.assertEqual(list(d["tataki"].iloc[1:]), [2.0, 3.0, 1.0, 2.0, 3.0, 4.0])         # 154 日空いた 4 走目が休み明け初戦（叩き 1 戦目）

    def test_horse_gap_vs_own_history(self):
        runs = self._runs(["2021-01-09", "2021-01-23", "2021-02-06", "2021-03-06"])
        f = tr.build_all(runs, only=["R1_band"])["R1_band"]
        self.assertAlmostEqual(float(f["rs1_gap_vs_mean"].iloc[3]), 28 - 14.0)                  # 過去の間隔 14,14 の平均との差
        self.assertTrue(np.isnan(f["rs1_gap_vs_mean"].iloc[1]))                                 # 2 走目は『過去の間隔』が無い

    def test_work_table_excludes_workouts_on_or_after_race(self):
        laps = pd.DataFrame({"horse_id": ["h"] * 3, "train_date": ["2021-01-01", "2021-01-05", "2021-01-10"], "race_date": ["2021-01-10"] * 3, "kind": ["slope", "wood", "poly"],
                             "leg_color": ["一杯", "馬也", "強め"], "days_before": [9, 5, 0]})
        w = tr.work_table(laps)
        self.assertEqual(int(w["n_work"].iloc[0]), 2)                                           # レース当日（train_date == race_date）は含めない


class ServingTest(unittest.TestCase):
    def test_last_day_rows_with_fast_asof_equal_training_values(self):
        """推論と同じ形（当日の行は結果なし、全馬横断の as-of は当日専用の高速版）で作った当日の値が、学習側（結果ありの全期間）の値と一致する。"""
        from src.pipeline.features import serving_v4 as sv

        df = _toy()
        day = df["date"].max()
        today = (df["date"] == day).to_numpy()
        full = tr.build_all(df, None, only=["R1_band", "R2_horse_pref", "R4_trainer", "R6_prev", "R8_season"])
        srv_in = df.copy()
        for c in ("finish_position", "margin", "passing_order", "last_3f", "time_sec"):
            srv_in.loc[today, c] = np.nan
        with sv.fast_asof(day):
            srv = tr.build_all(srv_in, None, only=["R1_band", "R2_horse_pref", "R4_trainer", "R6_prev", "R8_season"])
        for g in full:
            a = full[g].loc[today].drop(columns=tr.KEY).reset_index(drop=True)
            b = srv[g].loc[today].drop(columns=tr.KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-4, rtol=1e-4, obj=g)


if __name__ == "__main__":
    unittest.main()
