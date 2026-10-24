"""推論用の追い切り特徴（学習時との一致、未来を見ないこと）のテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import training_laps as tl
from src.pipeline.features import training_patterns as tp
from src.pipeline.features import training_serving as ts


def _record(hid: str, races: list[str], seed: int):
    rng = np.random.default_rng(seed)
    entries = []
    for rd in races:
        d0 = pd.Timestamp(rd)
        for days, leg in ((3, "一杯"), (10, "強め"), (17, "馬也")):
            t4 = 52.5 + rng.normal(0, 0.6); t3 = 38.0 + rng.normal(0, 0.4); t1 = 12.4 + rng.normal(0, 0.25)
            entries.append({"date": (d0 - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), "course": "栗坂", "track_condition": "良", "rider": "助手",
                            "lap_times": [round(t4, 1), round(t3, 1), round(t3 / 1.5 + 12.5, 1) if False else round(25.4, 1), round(t1, 1)], "position": "", "leg_color": leg, "rank": "B",
                            "race_info": f"{d0:%Y/%m/%d}  東京11R テスト 結果 ： 1着", "evaluation": ""})
    return {"horse_id": hid, "entries": entries}


class ParityTest(unittest.TestCase):
    def setUp(self):
        self.races = ["2022-03-06", "2022-04-03", "2022-05-01", "2022-06-05"]
        self.recs = {f"h{i}": _record(f"h{i}", self.races, i) for i in range(60)}
        self.laps = ts.records_to_laps(self.recs)
        # 学習時の計算（全馬・全期間を一度に）
        t = tp.add_reference_z(tp.prepare(self.laps), min_n=1)
        self.train_feat = tp.add_self_relative(tp.race_features(t))
        self.ref = tp.build_reference(tp.prepare(self.laps), min_n=1)

    def test_serving_matches_training(self):
        rd = "2022-06-05"
        got = ts.tl_a_for_race(self.recs, rd, self.ref)
        want = self.train_feat[self.train_feat["race_date"] == rd].set_index("horse_id")
        cols = ts.tl_a_columns(want.reset_index().columns)
        self.assertGreater(len(cols), 20)
        pd.testing.assert_frame_equal(got[cols].sort_index(), want[cols].astype("float32").sort_index(), check_exact=False, atol=1e-4, rtol=1e-4)

    def test_future_races_do_not_change_features(self):
        early = self.races[2]
        a = ts.tl_a_for_race(self.recs, early, self.ref)
        trimmed = {h: dict(r, entries=[e for e in r["entries"] if e["race_info"][:10].replace("/", "-") <= early]) for h, r in self.recs.items()}
        b = ts.tl_a_for_race(trimmed, early, self.ref)
        pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_exact=False, atol=1e-6)

    def test_missing_month_uses_latest_reference(self):
        ref = self.ref[self.ref["ym"] <= self.ref["ym"].max() - 2]               # 直近 2 か月分を落とした参照表
        got = ts.tl_a_for_race(self.recs, "2022-06-05", ref)
        self.assertTrue(got["f_t4z"].notna().mean() > 0.9)

    def test_no_record_returns_empty(self):
        self.assertTrue(ts.tl_a_for_race({"x": None}, "2022-06-05", self.ref).empty)


if __name__ == "__main__":
    unittest.main()
