import unittest

import numpy as np

from src.pipeline.inference import portfolio_target as PT


class RequiredStakeTest(unittest.TestCase):
    def test_ten_times_needs_half_of_budget_for_500pct(self):
        s = PT.required_stake(np.array([10.0]), 5.0, 10_000)
        self.assertEqual(s[0], 5_000)

    def test_rounds_up_to_100_yen(self):
        s = PT.required_stake(np.array([30.0]), 5.0, 10_000)  # 1666.67 → 1700
        self.assertEqual(s[0], 1_700)

    def test_safety_factor_scales_stake(self):
        a = PT.required_stake(np.array([20.0]), 5.0, 10_000, k=1.0)[0]
        b = PT.required_stake(np.array([20.0]), 5.0, 10_000, k=2.0)[0]
        self.assertEqual((a, b), (2_500, 5_000))

    def test_invalid_odds_are_infinite(self):
        s = PT.required_stake(np.array([0.0, np.nan, -1.0]), 5.0, 10_000)
        self.assertTrue(np.all(np.isinf(s)))


class SelectPortfolioTest(unittest.TestCase):
    def setUp(self):
        self.p = np.array([0.30, 0.10, 0.05, 0.02, 0.01])
        self.o = np.array([3.0, 12.0, 30.0, 80.0, 200.0])

    def test_every_chosen_bet_reaches_target_when_it_hits(self):
        params = PT.TargetParams(mult=5.0, k=1.0)
        idx, st = PT.select_portfolio(self.p, self.o, params)
        self.assertGreater(len(idx), 0)
        self.assertLessEqual(st.sum(), params.budget)
        for j, s in zip(idx, st):
            self.assertGreaterEqual(s * self.o[j], 5.0 * params.budget - 1e-6)

    def test_infeasible_race_is_skipped(self):
        idx, st = PT.select_portfolio(np.array([0.5, 0.3]), np.array([1.5, 2.0]), PT.TargetParams(mult=5.0))
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(st), 0)

    def test_single_bet_picks_highest_probability_among_feasible(self):
        idx, st = PT.select_portfolio(self.p, self.o, PT.TargetParams(mult=5.0, max_bets=1))
        self.assertEqual(list(idx), [1])  # 3.0 倍は予算内に収まらない。12 倍が最大の p
        self.assertEqual(st[0], 4_200)

    def test_tau_filters_by_expected_value(self):
        idx, _ = PT.select_portfolio(self.p, self.o, PT.TargetParams(mult=1.0, tau=1.0))
        for j in idx:
            self.assertGreaterEqual(self.p[j] * self.o[j], 1.0)

    def test_allowed_mask_limits_candidates(self):
        allowed = np.array([False, False, True, True, True])
        idx, _ = PT.select_portfolio(self.p, self.o, PT.TargetParams(mult=5.0), allowed)
        self.assertTrue(all(allowed[j] for j in idx))

    def test_max_bets_caps_count(self):
        idx, _ = PT.select_portfolio(self.p, self.o, PT.TargetParams(mult=1.0, max_bets=2))
        self.assertLessEqual(len(idx), 2)


class OutcomeTest(unittest.TestCase):
    def test_no_torigami_when_multiplier_at_least_one(self):
        o = np.array([10.0, 40.0])
        params = PT.TargetParams(mult=2.0, k=1.0)
        idx, st = PT.select_portfolio(np.array([0.1, 0.03]), o, params)
        for hit_j in range(len(idx)):
            pay100 = np.zeros(len(idx))
            pay100[hit_j] = o[idx[hit_j]] * 100  # 100 円あたりの払戻
            oc = PT.race_outcome(st, pay100, params.mult, params.budget)
            self.assertTrue(oc["hit"])
            self.assertFalse(oc["torigami"])
            self.assertTrue(oc["reach"])

    def test_miss_has_zero_payout(self):
        oc = PT.race_outcome(np.array([1_000.0]), np.array([0.0]), 5.0, 10_000)
        self.assertFalse(oc["hit"])
        self.assertEqual(oc["pay"], 0.0)

    def test_torigami_detected(self):
        oc = PT.race_outcome(np.array([5_000.0, 5_000.0]), np.array([0.0, 150.0]), 1.0, 10_000)
        self.assertTrue(oc["hit"])
        self.assertTrue(oc["torigami"])  # 払戻 7,500 < 購入 10,000


class BaselineHelpersTest(unittest.TestCase):
    def test_kelly_stakes_within_budget(self):
        idx, st = PT.kelly_stakes(np.array([0.3, 0.2, 0.05]), np.array([4.0, 6.0, 30.0]))
        self.assertLessEqual(st.sum(), PT.DEFAULT_BUDGET)
        self.assertTrue(np.all(st % 100 == 0))

    def test_equal_split(self):
        idx, st = PT.equal_split(np.array([1.5, 0.9, 1.2, 1.1]), 2)
        self.assertEqual(list(idx), [0, 2])
        self.assertEqual(list(st), [5_000.0, 5_000.0])


if __name__ == "__main__":
    unittest.main()
