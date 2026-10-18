"""tp-33 輸送・滞在の特徴量: 場所の読み取り、輸送区分、先読みなし（将来の結果を変えても過去行の特徴量が変わらない）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp33_transport as tt


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, laps = [], []
    horses = [f"h{i}" for i in range(10)]
    trainers = {h: ("t_ku" if i % 2 == 0 else "t_mi") for i, h in enumerate(horses)}
    venues = ["05", "06", "08", "09", "01", "10"]
    for r in range(40):
        date = (pd.Timestamp("2022-01-08") + pd.Timedelta(days=7 * r)).strftime("%Y-%m-%d")
        for h in rng.choice(horses, size=6, replace=False):
            v = venues[(r + int(h[1:])) % len(venues)]
            rows.append({"race_id": f"R{r:03d}{h}", "horse_id": h, "date": date, "venue_code": v, "trainer_id": trainers[h], "finish_position": int(rng.integers(1, 12))})
            home = "栗坂" if trainers[h] == "t_ku" else "美坂"
            away = {"01": "札ダ", "10": "小ダ"}.get(v, home)
            for db, crs in ((3, home), (10, away if rng.random() < 0.5 else home)):
                laps.append({"horse_id": h, "race_date": date, "train_date": (pd.Timestamp(date) - pd.Timedelta(days=db)).strftime("%Y-%m-%d"), "course": crs, "days_before": db})
    return pd.DataFrame(rows), pd.DataFrame(laps)


class LocTest(unittest.TestCase):
    def test_course_loc(self):
        self.assertEqual(tt.course_loc("栗坂"), "KU")
        self.assertEqual(tt.course_loc("CW"), "KU")
        self.assertEqual(tt.course_loc("南W"), "MI")
        self.assertEqual(tt.course_loc("函W"), "02")
        self.assertEqual(tt.course_loc("札ダ"), "01")
        self.assertEqual(tt.course_loc("小ダ"), "10")
        self.assertEqual(tt.course_loc("栗坂一番時計"), "KU")
        self.assertEqual(tt.course_loc("船橋"), "NAR")
        self.assertEqual(tt.course_loc(""), "UNK")

    def test_tier(self):
        self.assertEqual(tt.tier_of_km(tt.KM_HOME[("MI", "05")], "05"), 0)           # 美浦→東京はホーム
        self.assertEqual(tt.tier_of_km(tt.KM_HOME[("KU", "08")], "08"), 0)           # 栗東→京都
        self.assertEqual(tt.tier_of_km(tt.KM_HOME[("KU", "05")], "05"), 2)           # 栗東→東京は長距離
        self.assertEqual(tt.tier_of_km(tt.KM_HOME[("MI", "09")], "09"), 2)           # 美浦→阪神
        self.assertEqual(tt.tier_of_km(tt.KM_HOME[("MI", "01")], "01"), 3)           # 北海道は遠方
        self.assertEqual(tt.tier_of_km(float("nan"), "05"), -1)


class NoLeakTest(unittest.TestCase):
    def test_future_results_do_not_change_past_rows(self):
        df, laps = _toy()
        a = tt.build(df.copy(), laps)
        df2 = df.copy()
        late = pd.to_datetime(df2["date"]) >= pd.Timestamp("2022-06-01")
        df2.loc[late, "finish_position"] = (df2.loc[late, "finish_position"] % 11) + 1
        b = tt.build(df2, laps)
        early = (pd.to_datetime(df["date"]) < pd.Timestamp("2022-06-01")).to_numpy()
        for g in a:
            pd.testing.assert_frame_equal(a[g][early].reset_index(drop=True), b[g][early].reset_index(drop=True), check_exact=False, atol=1e-9)

    def test_same_day_result_not_used(self):
        df, laps = _toy()
        a = tt.build(df.copy(), laps)
        d0 = df["date"].iloc[len(df) // 2]
        df2 = df.copy()
        df2.loc[df2["date"] == d0, "finish_position"] = 1
        b = tt.build(df2, laps)
        same = (df["date"] == d0).to_numpy()
        for g in ("TT_rate",):
            pd.testing.assert_frame_equal(a[g][same].reset_index(drop=True), b[g][same].reset_index(drop=True), check_exact=False, atol=1e-9)


class StayTest(unittest.TestCase):
    def test_stay_flag_when_trained_away(self):
        df = pd.DataFrame([{"race_id": "R1", "horse_id": "h", "date": "2022-07-10", "venue_code": "01", "trainer_id": "t_ku", "finish_position": 1}])
        laps = pd.DataFrame([
            {"horse_id": "h", "race_date": "2022-07-10", "train_date": "2022-07-07", "course": "札ダ", "days_before": 3},
            {"horse_id": "h", "race_date": "2022-07-10", "train_date": "2022-06-30", "course": "栗坂", "days_before": 10},
        ])
        # 拠点は他の馬の追い切りから決まる（この調教師の馬の追い切りが栗東の施設）
        home = pd.Series({"t_ku": "KU"})
        g = tt.build(df, laps, home_map=home)
        self.assertEqual(g["TT_stay"]["tt_stay_flag"].iloc[0], 1.0)
        self.assertEqual(g["TT_stay"]["tt_stay_at_venue"].iloc[0], 1.0)
        self.assertEqual(g["TT_tier"]["tt_tier"].iloc[0], 3.0)                     # 栗東→札幌は遠方


if __name__ == "__main__":
    unittest.main()
