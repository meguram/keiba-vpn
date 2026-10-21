"""tp-70 確率の較正（src/utils/prob_calibration.py）と、評価用部品（tp70_lib / tp70_fit）のテスト。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.scripts.data import tp70_fit as F
from src.scripts.data import tp70_lib as L
from src.utils import prob_calibration as P


def race(n=12, seed=0):
    rng = np.random.default_rng(seed)
    odds = np.sort(rng.uniform(1.8, 120, n))
    q = (1 / odds) / (1 / odds).sum()
    w = rng.dirichlet(np.ones(n) * 1.5)
    from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

    return odds, {"win": w, "top2": harville_top2_prob(w), "top3": harville_top3_prob(w)}


class CalibrationTest(unittest.TestCase):
    def test_sums_and_range(self):
        odds, pm = race()
        out = P.calibrate_arrays(pm, odds)
        self.assertIsNotNone(out)
        for t, k in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
            self.assertAlmostEqual(float(out[t].sum()), k, places=3)
            self.assertTrue(np.all(out[t] > 0) and np.all(out[t] < 1))

    def test_pulls_toward_market(self):
        odds, pm = race()
        out = P.calibrate_arrays(pm, odds)
        mk = P.market_probs(odds)
        d_raw = np.abs(pm["top3"] - mk["top3"]).sum()
        d_cal = np.abs(out["top3"] - mk["top3"]).sum()
        self.assertLess(d_cal, d_raw)       # 較正後は市場に近づく（モデルの自信過剰を縮める）

    def test_fallbacks(self):
        odds, pm = race()
        self.assertIsNone(P.calibrate_arrays(pm, odds[:5]))                  # 8 頭未満
        bad = odds.copy()
        bad[2] = 0.0
        self.assertIsNone(P.calibrate_arrays(pm, bad))                       # オッズ欠損
        entries = [{"horse_number": i + 1, "win_prob": pm["win"][i], "top2_prob": pm["top2"][i], "top3_prob": pm["top3"][i]} for i in range(len(odds))]
        om = {i + 1: float(odds[i]) for i in range(len(odds))}
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": ""}):
            self.assertIsNone(P.calibrate_entries(entries, om))              # 既定オフ
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": "1"}):
            self.assertIsNone(P.calibrate_entries(entries, om, market_is_real=False))   # 予測オッズは使わない
            res = P.calibrate_entries(entries, om)
            self.assertEqual(len(res), len(entries))
            self.assertAlmostEqual(sum(r["win_prob"] for r in res), 1.0, places=2)

    def test_methods_agree_in_shape(self):
        odds, pm = race(seed=3)
        a = P.calibrate_arrays(pm, odds, {"top3": "blend"})
        b = P.calibrate_arrays(pm, odds, {"top3": "blend_pop"})
        self.assertLess(np.abs(a["top3"] - b["top3"]).max(), 0.15)           # 人気区分別は同じ方向の小さな調整


class LibTest(unittest.TestCase):
    def test_scale_to_sum_matches_utils(self):
        p = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01])
        a = L.scale_to_sum(p, 3.0)
        b = P.scale_to_sum(p, 3.0)
        np.testing.assert_allclose(a, b, atol=1e-9)
        self.assertAlmostEqual(float(a.sum()), 3.0, places=4)

    def test_ece_and_logloss(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1] * 20)
        p = np.full(len(y), y.mean())
        self.assertAlmostEqual(L.ece(y, p), 0.0, places=6)
        self.assertGreater(L.logloss(y, np.full(len(y), 0.05)), L.logloss(y, p))

    def test_fit_predict_improves_over_raw(self):
        rng = np.random.default_rng(0)
        n_race, n = 400, 10
        rows = []
        for r in range(n_race):
            q = rng.dirichlet(np.ones(n))
            truth = 0.5 * np.log(q) + 0.5 * rng.normal(0, 0.5, n)       # 実力は市場と無関係な成分を半分持つ
            w = np.exp(truth) / np.exp(truth).sum()
            winner = rng.choice(n, p=w)
            for i in range(n):
                rows.append({"race_id": f"r{r}", "year": 2022 + (r % 3), "n": n, "p_win": float(np.clip(q[i] * 1.8 if i == 0 else q[i] * 0.9, 1e-4, 0.9)), "m_win": q[i], "h1": int(i == winner), "pop": i + 1})
        df = pd.DataFrame(rows)
        for c in ("p_top2", "p_top3", "m_top2", "m_top3"):
            df[c] = df["p_win"] if c.startswith("p") else df["m_win"]
        df["h2"] = df["h1"]
        df["h3"] = df["h1"]
        tr, te = df[df["year"] < 2024], df[df["year"] == 2024]
        p = F.fit_predict(tr, te, "win", "C3")
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertLessEqual(L.logloss(te["h1"], p), L.logloss(te["h1"], te["p_win"]) + 1e-9)


if __name__ == "__main__":
    unittest.main()
