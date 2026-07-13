"""条件不一致重みのユニットテスト。"""

import unittest

from src.pipeline.megu_index.common import WEIGHTS_B
from src.pipeline.megu_index.condition_weights import (
    apply_condition_weights,
    condition_mismatch_multiplier,
)
from src.pipeline.megu_index.predict import predict_megu_scores


class TestConditionWeights(unittest.TestCase):
    def test_multiplier_match(self):
        self.assertEqual(
            condition_mismatch_multiplier("芝", 1600, "芝", 1600),
            1.0,
        )

    def test_multiplier_surface_only(self):
        m = condition_mismatch_multiplier(
            "芝", 1600, "ダート", 1600,
            weights={"w_surface_only": 0.4, "w_match": 1.0, "w_distance_only": 0.5, "w_both": 0.2},
        )
        self.assertEqual(m, 0.4)

    def test_multiplier_distance_only(self):
        m = condition_mismatch_multiplier(
            "芝", 1600, "芝", 2400,
            weights={"w_distance_only": 0.6, "w_match": 1.0, "w_surface_only": 0.4, "w_both": 0.2},
        )
        self.assertEqual(m, 0.6)

    def test_multiplier_both(self):
        m = condition_mismatch_multiplier(
            "芝", 1600, "ダート", 2400,
            weights={"w_both": 0.15, "w_match": 1.0, "w_surface_only": 0.4, "w_distance_only": 0.6},
        )
        self.assertEqual(m, 0.15)

    def test_apply_reduces_turf_weight_for_dirt_target(self):
        hist = [
            {"surface": "ダート", "distance": 1600},
            {"surface": "芝", "distance": 1600},
        ]
        nw, mults = apply_condition_weights(
            WEIGHTS_B[:2], hist,
            surface_target="ダート", distance_target=1600,
            condition_weights={
                "w_match": 1.0, "w_surface_only": 0.2,
                "w_distance_only": 0.5, "w_both": 0.1,
            },
        )
        self.assertAlmostEqual(sum(nw), 1.0, places=6)
        self.assertGreater(nw[0], nw[1])

    def test_predict_surface_mismatch_lowers_final(self):
        """芝のみの過去走 → ダート予測では指数が下がる（重み割引）。"""
        hist = [{"megu_index": 110.0, "par_time_sec": 96.0, "surface": "芝", "distance": 1600}]
        no_disc = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="ダート", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            condition_weights={
                "w_match": 1.0, "w_surface_only": 1.0,
                "w_distance_only": 1.0, "w_both": 1.0,
            },
        )
        discounted = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="ダート", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            condition_weights={
                "w_match": 1.0, "w_surface_only": 0.3,
                "w_distance_only": 0.5, "w_both": 0.1,
            },
        )
        # 1走のみなら重み割引しても値は同じ（正規化で100%がその1走）
        self.assertEqual(no_disc["megu_final"], 110.0)

        hist2 = [
            {"megu_index": 100.0, "par_time_sec": 96.0, "surface": "ダート", "distance": 1600},
            {"megu_index": 115.0, "par_time_sec": 96.0, "surface": "芝", "distance": 1600},
        ]
        p0 = predict_megu_scores(
            hist2, par_time_target=96.0, surface_target="ダート", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            condition_weights={
                "w_match": 1.0, "w_surface_only": 1.0,
                "w_distance_only": 1.0, "w_both": 1.0,
            },
        )
        p1 = predict_megu_scores(
            hist2, par_time_target=96.0, surface_target="ダート", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            condition_weights={
                "w_match": 1.0, "w_surface_only": 0.2,
                "w_distance_only": 0.5, "w_both": 0.1,
            },
        )
        # 芝の高めぐが混ざると過大評価 → 割引でダート寄りに下がる
        self.assertGreater(p0["megu_final"] or 0, p1["megu_final"] or 0)
        self.assertEqual(p1["condition_weight_multipliers"], [1.0, 0.2])


if __name__ == "__main__":
    unittest.main()
