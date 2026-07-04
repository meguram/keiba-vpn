"""final_odds 特徴量拡張のユニットテスト。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline.models.final_odds_features import enrich_final_odds_features, select_model_feature_columns


class TestFinalOddsFeatures(unittest.TestCase):
    def test_enrich_adds_field_relative_columns(self):
        df = pd.DataFrame(
            {
                "horse_number": [1, 2, 3],
                "field_size": [3, 3, 3],
                "speed_max": [110, 105, 100],
                "career_win_rate": [0.2, 0.1, 0.05],
                "distance": [1600, 1600, 1600],
                "surface": ["芝", "芝", "芝"],
                "direction": ["左", "左", "左"],
                "track_condition": ["良", "良", "良"],
                "venue": ["東京", "東京", "東京"],
                "bracket_number": [1, 2, 3],
            }
        )
        out = enrich_final_odds_features(df)
        self.assertIn("fld_z_speed_max", out.columns)
        self.assertIn("fld_rank_speed_max", out.columns)
        self.assertIn("surface_code", out.columns)
        cols = select_model_feature_columns(out)
        self.assertIn("fld_z_speed_max", cols)
        self.assertNotIn("horse_name", cols)


if __name__ == "__main__":
    unittest.main()
