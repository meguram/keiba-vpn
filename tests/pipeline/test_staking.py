"""購入額の段階づけ（見送り / 500 / 1000 / 2000 円）の単体テスト。"""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import staking as sk
from src.pipeline.inference.exotic_probabilities import RankDistribution


class StakingTest(unittest.TestCase):
    def test_round_units(self):
        u = sk.round_units(np.array([0.5, 0.3, 0.2]), 1000)
        self.assertEqual(int(u.sum()), 10)
        self.assertEqual(list(u), [5, 3, 2])
        u2 = sk.round_units(np.array([1, 1, 1]), 500)
        self.assertEqual(int(u2.sum()), 5)
        self.assertEqual(int(sk.round_units(np.array([0.0, 0.0]), 500).sum()), 0)
        self.assertEqual(int(sk.round_units(np.array([1.0]), 0).sum()), 0)

    def test_portfolio_outcomes_matches_brute_force(self):
        rng = np.random.default_rng(0)
        n = 6
        win = rng.dirichlet(np.ones(n) * 2)
        hn = np.arange(1, n + 1)
        tickets = [("tansho", "2"), ("fukusho", "3"), ("umaren", "1-4"), ("wide", "2-5"), ("umatan", "4>1")]
        w = np.array([0.3, 0.3, 0.1, 0.2, 0.1]); odds = np.array([5.0, 2.0, 30.0, 8.0, 60.0])
        o = sk.portfolio_outcomes(hn, win, 0.8, 0.7, tickets, w, odds)
        d = RankDistribution(win, 0.8, 0.7)
        hit = ppos = ev = 0.0
        for i, j, k in itertools.permutations(range(n), 3):
            first, second, third = hn[i], hn[j], hn[k]
            r = 0.0
            r += w[0] * odds[0] * (first == 2)
            r += w[1] * odds[1] * (3 in (first, second, third))
            r += w[2] * odds[2] * ({first, second} == {1, 4})
            r += w[3] * odds[3] * (2 in (first, second, third) and 5 in (first, second, third))
            r += w[4] * odds[4] * (first == 4 and second == 1)
            p = d._p3[i, j, k]
            hit += p * (r > 0); ppos += p * (r >= 1 - 1e-9); ev += p * r
        self.assertAlmostEqual(o["hit"], hit, places=9); self.assertAlmostEqual(o["ppos"], ppos, places=9); self.assertAlmostEqual(o["ev"], ev, places=9)

    def test_tier_rule_and_quantiles(self):
        rng = np.random.default_rng(1)
        f = pd.DataFrame({"race_id": [f"r{i}" for i in range(1000)], "ev_cal": rng.uniform(0.5, 1.2, 1000), "hit": rng.uniform(0.05, 0.9, 1000)})
        f["ev_mix"] = f["ev_cal"]
        rule = sk.quantile_rule(f, h0=0.0, h1=0.0, h2=0.0)
        s = rule.assign(f)
        shares = [(s == v).mean() for v in sk.STAKES]
        np.testing.assert_allclose(shares, [0.4, 0.3, 0.2, 0.1], atol=0.02)
        strict = sk.TierRule(0.0, 0.0, 0.0, h0=0.5, h1=0.5, h2=0.5)
        s2 = strict.assign(f)
        self.assertTrue(((s2 == 0) == (f["hit"] < 0.5)).all())            # 当たらなさすぎるレースは見送り

    def test_quantile_rule_survives_ties(self):
        rng = np.random.default_rng(2)
        step = np.round(rng.uniform(0.7, 0.85, 1000), 2)                      # 階段状の ev_cal（同値が多い）
        f = pd.DataFrame({"race_id": [f"r{i}" for i in range(1000)], "ev_cal": step, "ev_raw": rng.uniform(0.5, 1.0, 1000), "hit": 0.9})
        f["ev_mix"] = f["ev_cal"] + 1e-3 * f["ev_raw"]
        s = sk.quantile_rule(f, h0=0.0, h1=0.0, h2=0.0).assign(f)
        shares = [(s == v).mean() for v in sk.STAKES]
        np.testing.assert_allclose(shares, [0.4, 0.3, 0.2, 0.1], atol=0.06)    # 4 段階がすべて使われる

    def test_settle_tiered(self):
        t = pd.DataFrame({"race_id": ["a", "a", "b"], "type": ["tansho", "fukusho", "tansho"], "combo": ["1", "1", "2"], "mult": [5.0, 0.0, 0.0], "w": [0.6, 0.4, 1.0]})
        f = pd.DataFrame({"race_id": ["a", "b"], "ev_cal": [1.0, 1.0], "hit": [0.5, 0.5]})
        d = sk.settle_tiered(t, f, np.array([1000, 0]))
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(float(d["stake"].iloc[0]), 1000.0)
        self.assertAlmostEqual(float(d["ret"].iloc[0]), 600 / 100 * 5.0 * 100)   # 600 円分 × 5 倍
        self.assertAlmostEqual(float(d["profit"].iloc[0]), 2000.0)


if __name__ == "__main__":
    unittest.main()
