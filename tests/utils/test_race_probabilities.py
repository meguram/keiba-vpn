"""race_probabilities のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.race_probabilities import (
    assign_mece_marks,
    buy_recommendation_tier,
    derive_race_probabilities,
    qualifies_star_mark,
)


class TestRaceProbabilities(unittest.TestCase):
    def test_win_probs_sum_to_one(self):
        probs = derive_race_probabilities([10.0, 8.0, 6.0, 4.0])
        total = sum(p["win_prob"] for p in probs)
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_top3_ge_top2_ge_win(self):
        probs = derive_race_probabilities([12.0, 9.0, 7.0, 5.0, 3.0])
        for p in probs:
            self.assertGreaterEqual(p["top3_prob"], p["top2_prob"])
            self.assertGreaterEqual(p["top2_prob"], p["win_prob"])

    def test_mece_honmei_and_star_at_most_one(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.28, "top2_prob": 0.45, "top3_prob": 0.6,
             "win_odds": 15.0, "ev_win": 1.2, "ev_place": 1.0},
            {"horse_number": 2, "win_prob": 0.18, "top2_prob": 0.4, "top3_prob": 0.55,
             "win_odds": 8.0, "ev_win": 0.8, "ev_place": 0.9},
            {"horse_number": 3, "win_prob": 0.05, "top2_prob": 0.28, "top3_prob": 0.42},
            {"horse_number": 4, "win_prob": 0.04, "top2_prob": 0.26, "top3_prob": 0.38},
        ]
        assign_mece_marks(entries)
        types = [e["mark_type"] for e in entries]
        self.assertEqual(types.count("honmei"), 1)
        self.assertLessEqual(types.count("star"), 1)

    def _race(self, n=14, seed=0):
        import random

        rng = random.Random(seed)
        raw = [rng.random() ** 2 + 0.02 for _ in range(n)]
        t = sum(raw)
        win = sorted((r / t for r in raw), reverse=True)
        out = []
        for i, w in enumerate(win):
            out.append({"horse_number": i + 1, "win_prob": w, "top2_prob": min(1.0, w * 2.1), "top3_prob": min(1.0, w * 3.0)})
        return out

    def test_each_mark_except_maybe_is_single(self):
        for seed in range(20):
            entries = self._race(14, seed)
            assign_mece_marks(entries)
            c = {k: [e["mark_type"] for e in entries].count(k) for k in ("honmei", "pair", "show_val", "star")}
            self.assertTrue(all(v <= 1 for v in c.values()), c)

    def test_pair_is_highest_top2_after_honmei(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.3, "top2_prob": 0.5, "top3_prob": 0.6},
            {"horse_number": 2, "win_prob": 0.1, "top2_prob": 0.45, "top3_prob": 0.5},
            {"horse_number": 3, "win_prob": 0.08, "top2_prob": 0.4, "top3_prob": 0.48},
            {"horse_number": 4, "win_prob": 0.05, "top2_prob": 0.15, "top3_prob": 0.3},
        ]
        assign_mece_marks(entries)
        types = {e["horse_number"]: e["mark_type"] for e in entries}
        self.assertEqual(types[1], "honmei")
        self.assertEqual(types[2], "pair")
        self.assertNotEqual(types[3], "pair")

    def test_pair_not_assigned_below_threshold(self):
        entries = [{"horse_number": i + 1, "win_prob": 1 / 12, "top2_prob": 1 / 12 * 1.1, "top3_prob": 0.25} for i in range(12)]
        assign_mece_marks(entries)
        self.assertEqual([e["mark_type"] for e in entries].count("pair"), 0)
        self.assertEqual([e["mark_type"] for e in entries].count("honmei"), 0)

    def test_maybe_limit_by_runner_count(self):
        from src.utils.race_probabilities import maybe_limit

        self.assertEqual([maybe_limit(n) for n in (3, 5, 7, 8, 12, 13, 18)], [1, 2, 2, 3, 3, 4, 4])

    def test_maybe_marks_capped_and_ordered_by_top3(self):
        entries = self._race(16, 3)
        assign_mece_marks(entries)
        maybes = [e for e in entries if e["mark_type"] == "anchor"]
        self.assertLessEqual(len(maybes), 4)
        others = [e for e in entries if e["mark_type"] == "none" and e["top3_prob"] >= 1 / 16 * 1.35]
        if maybes and others:
            self.assertGreaterEqual(min(m["top3_prob"] for m in maybes), max(o["top3_prob"] for o in others) - 1e-12)

    def test_total_marks_stay_small(self):
        for n in (6, 10, 14, 18):
            for seed in range(10):
                entries = self._race(n, seed)
                assign_mece_marks(entries)
                marked = sum(1 for e in entries if e["mark_type"] != "none")
                self.assertLessEqual(marked, 3 + 4 + 1)

    def test_one_mark_per_horse_and_empty(self):
        assign_mece_marks([])
        entries = self._race(10, 1)
        assign_mece_marks(entries)
        self.assertTrue(all(isinstance(e["mark_type"], str) for e in entries))

    def test_qualifies_star_odds_band(self):
        self.assertTrue(
            qualifies_star_mark(
                {"win_odds": 15.0, "ev_win": 1.1, "ev_place": 0.5, "top3_prob": 0.2},
                0.1,
            )
        )
        self.assertFalse(
            qualifies_star_mark(
                {"win_odds": 5.0, "ev_win": 1.2, "ev_place": 0.5, "top3_prob": 0.2},
                0.1,
            )
        )

    def test_assign_star_mark(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.2, "top2_prob": 0.35, "top3_prob": 0.5,
             "win_odds": 15.0, "ev_win": 1.2, "ev_place": 1.0},
            {"horse_number": 2, "win_prob": 0.15, "top2_prob": 0.3, "top3_prob": 0.45,
             "win_odds": 8.0, "ev_win": 0.8, "ev_place": 0.9},
        ]
        assign_mece_marks(entries)
        stars = [e for e in entries if e["mark_type"] == "star"]
        self.assertEqual(len(stars), 1)
        self.assertEqual(stars[0]["horse_number"], 1)

    def test_buy_recommendation_tier(self):
        self.assertEqual(buy_recommendation_tier({"ev_win": 1.3}), "強推奨")
        self.assertEqual(buy_recommendation_tier({"ev_place": 0.95}), "様子見")


if __name__ == "__main__":
    unittest.main()


def _loop_top2(w):
    """旧実装（三重ループ）。ベクトル化版との数値一致を固定するための参照。"""
    import numpy as np

    n = len(w)
    top2 = np.copy(w)
    for i in range(n):
        for j in range(n):
            if j != i and 1.0 - w[j] > 1e-9:
                top2[i] += w[j] * w[i] / (1.0 - w[j])
    return np.minimum(top2, 1.0)


def _loop_top3(w):
    import numpy as np

    n = len(w)
    top3 = np.copy(_loop_top2(w))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            d1 = 1.0 - w[j]
            if d1 < 1e-9:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                d2 = 1.0 - w[j] - w[k]
                if d2 > 1e-9:
                    top3[i] += w[j] * (w[k] / d1) * (w[i] / d2)
    return np.minimum(top3, 1.0)


class TestVectorizedHarville(unittest.TestCase):
    def test_matches_loop_reference(self):
        import numpy as np

        from src.utils.race_probabilities import (
            _softmax_win_probs,
            harville_top2_prob,
            harville_top3_prob,
        )

        rng = np.random.default_rng(42)
        for n in (1, 2, 3, 5, 12, 18):
            for scale in (0.3, 1.0, 3.0):
                w = _softmax_win_probs(rng.normal(scale=scale, size=n))
                with self.subTest(n=n, scale=scale):
                    np.testing.assert_allclose(harville_top2_prob(w), _loop_top2(w), atol=1e-12)
                    np.testing.assert_allclose(harville_top3_prob(w), _loop_top3(w), atol=1e-12)

    def test_sums_match_harville_identities(self):
        import numpy as np

        from src.utils.race_probabilities import _softmax_win_probs, harville_top2_prob, harville_top3_prob

        w = _softmax_win_probs(np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0]))
        self.assertAlmostEqual(float(harville_top2_prob(w).sum()), 2.0, places=9)
        self.assertAlmostEqual(float(harville_top3_prob(w).sum()), 3.0, places=9)

    def test_degenerate_inputs(self):
        import numpy as np

        from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

        one = np.array([1.0])
        np.testing.assert_allclose(harville_top3_prob(one), [1.0])
        self.assertEqual(len(harville_top3_prob(np.array([]))), 0)
