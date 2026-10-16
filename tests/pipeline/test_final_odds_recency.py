"""最終オッズ予測の新しい年を重くする学習重みのテスト。"""

from __future__ import annotations

import unittest

import numpy as np

from src.pipeline.models.final_odds_trainer import recency_weights


class RecencyTest(unittest.TestCase):
    def test_none_means_no_weights(self):
        self.assertIsNone(recency_weights(["202401010101"], None))
        self.assertIsNone(recency_weights(["202401010101"], 0))

    def test_halflife(self):
        w = recency_weights(["202001010101", "202301010101", "202401010101"], 2.0)
        np.testing.assert_allclose(w, [0.5 ** 2, 0.5 ** 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
