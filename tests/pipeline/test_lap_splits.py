"""lap_splits ユーティリティのユニットテスト。"""

import unittest

from src.pipeline.megu_index.lap_splits import (
    lap_segment_end_distances,
    parse_lap_times,
    select_split_point,
)


class TestLapSplits(unittest.TestCase):
    def test_segment_200_multiple(self):
        self.assertEqual(lap_segment_end_distances(1800, 9), [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])

    def test_segment_100m_first(self):
        self.assertEqual(lap_segment_end_distances(1700, 9), [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700])

    def test_segment_150m_first(self):
        self.assertEqual(lap_segment_end_distances(1150, 6), [150, 350, 550, 750, 950, 1150])

    def test_select_closest_to_half(self):
        # 1400m → 中間700m。600と800が同距離 → 中間以下の600を優先
        self.assertEqual(select_split_point(1400, [200, 400, 600, 800, 1000]), 600)
        # 1800m → 中間900m。800と1000が同距離 → 800
        self.assertEqual(select_split_point(1800, [200, 400, 600, 800, 1000, 1200]), 800)

    def test_parse_lap_times_1800(self):
        laps = [12.4, 11.3, 12.4, 12.7, 12.8, 12.2, 12.0, 11.9, 12.0]
        d = parse_lap_times(laps, 1800)
        self.assertEqual(select_split_point(1800, list(d.keys())), 800)
        self.assertAlmostEqual(d[800], 48.8, places=1)


if __name__ == "__main__":
    unittest.main()
