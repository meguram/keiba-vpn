"""lap_pattern 分類のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.lap_pattern import classify_race_lap_pattern


class TestLapPattern(unittest.TestCase):
    def test_slow_burst(self):
        # L5,L4 遅め → L2,L1 速め（終盤加速）
        laps = [11.5, 11.6, 11.7, 11.8, 12.4, 12.5, 11.6, 11.5, 11.4, 11.3]
        out = classify_race_lap_pattern(laps)
        self.assertIsNotNone(out)
        self.assertEqual(out["code"], "slow_burst")

    def test_consuming(self):
        laps = [11.5, 11.6, 11.7, 11.8, 12.0, 12.0, 12.0, 12.0, 12.5]
        out = classify_race_lap_pattern(laps)
        self.assertEqual(out["code"], "consuming")

    def test_sustained(self):
        laps = [11.0, 11.5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]
        out = classify_race_lap_pattern(laps)
        self.assertEqual(out["code"], "sustained")

    def test_standard_fallback(self):
        # 後半5Fにばらつきがあり、L1突出・終盤加速も閾値未満
        laps = [12.0, 11.5, 12.5, 11.4, 12.3, 11.6, 12.2, 11.8, 12.1]
        out = classify_race_lap_pattern(laps)
        self.assertEqual(out["code"], "standard")

    def test_insufficient_data(self):
        self.assertIsNone(classify_race_lap_pattern([12.0, 11.5, 12.0]))


if __name__ == "__main__":
    unittest.main()
