"""推論パイプラインの連対率・複勝率が Harville 式で導出されること（T-013 / DEC-022）。

外部接続はしない: 環境変数を空にし、ラップ予測はスタブに差し替える。
"""
import os
import unittest
from unittest.mock import patch

import numpy as np


def _load_module():
    with patch.dict(os.environ, {}, clear=True):
        from src.pipeline.inference import inference_pipeline

    return inference_pipeline


def _raw(scores):
    preds = [
        {"horse_id": f"h{i}", "horse_number": i + 1, "pred_score": s, "pred_rank": i + 1}
        for i, s in enumerate(scores)
    ]
    return {"race_id": "202606010101", "distance": 1600, "total_horses": len(preds), "predictions": preds}


class InferencePipelineProbabilityTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.stub = {"pace_category": "M", "lap_times": []}

    def _map(self, scores):
        with patch.object(self.mod, "predict_pace_and_laps", return_value=self.stub):
            return self.mod._map_stage1_to_spec(_raw(scores), model_version="t")

    def test_place_and_show_sum_to_harville_totals(self):
        horses = self._map([2.0, 1.2, 0.7, 0.1, -0.4, -1.0, -1.5, -2.0])["horses"]
        self.assertAlmostEqual(sum(h["win_prob"] for h in horses), 1.0, delta=1e-3)
        self.assertAlmostEqual(sum(h["place_prob"] for h in horses), 2.0, delta=5e-3)
        self.assertAlmostEqual(sum(h["show_prob"] for h in horses), 3.0, delta=5e-3)

    def test_ordering_and_bounds(self):
        for h in self._map([3.0, 1.0, 0.0, -1.0, -2.0])["horses"]:
            self.assertLessEqual(h["win_prob"], h["place_prob"])
            self.assertLessEqual(h["place_prob"], h["show_prob"])
            self.assertLessEqual(h["show_prob"], 1.0)

    def test_fixed_multipliers_are_gone(self):
        # 固定倍率なら本命の複勝率は min(1, win*3.0)。Harville では win の 3 倍にならない
        top = self._map([4.0, 0.0, -0.5, -1.0, -1.5, -2.0])["horses"][0]
        self.assertNotAlmostEqual(top["show_prob"], min(1.0, top["win_prob"] * 3.0), places=2)
        self.assertNotAlmostEqual(top["place_prob"], min(1.0, top["win_prob"] * 2.2), places=2)

    def test_matches_shared_harville_functions(self):
        from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

        out = self._map([1.5, 0.3, -0.2, -1.1])["horses"]
        win = np.array([h["win_prob"] for h in out])
        win = win / win.sum()
        np.testing.assert_allclose([h["place_prob"] for h in out], harville_top2_prob(win), atol=2e-3)
        np.testing.assert_allclose([h["show_prob"] for h in out], harville_top3_prob(win), atol=2e-3)

    def test_empty_field(self):
        self.assertEqual(self._map([])["horses"], [])


if __name__ == "__main__":
    unittest.main()
