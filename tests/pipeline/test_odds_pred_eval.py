import unittest

import numpy as np

from src.scripts.data import odds_pred_eval as m


class TestOddsPredHelpers(unittest.TestCase):
    def test_share_normalization_and_odds(self):
        rid = np.array(["a"] * 4 + ["b"] * 3)
        sh = m._norm_share(rid, np.log(np.array([0.4, 0.3, 0.2, 0.1, 0.6, 0.3, 0.1])))
        self.assertAlmostEqual(sh[:4].sum(), 1.0)
        self.assertAlmostEqual(sh[4:].sum(), 1.0)
        odds = m._odds_from_share(rid, sh)
        self.assertAlmostEqual(float((1.0 / odds[:4]).sum()), 1.0 / m.PAYBACK_WIN, places=3)

    def test_arrays_probabilities_sum(self):
        w = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
        pl = 1.0 / np.array([1.5, 2.0, 2.6, 4.5, 7.0])
        w, p2, p3 = m._marginals(w, 1.0 / pl)
        hn = np.array([3, 5, 1, 8, 2])
        arr = m._arrays(m._dist(w, p2, p3), hn)
        self.assertAlmostEqual(float(arr["umaren"][1].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(arr["umatan"][1].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(arr["sanrenpuku"][1].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(arr["wide"][1].sum()), 3.0, places=6)
        self.assertIn("1-3", set(arr["umaren"][0]))          # 馬番は小さい順
        self.assertIn("5>3", set(arr["umatan"][0]))


if __name__ == "__main__":
    unittest.main()
