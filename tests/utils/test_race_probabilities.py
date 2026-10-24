"""race_probabilities のユニットテスト。"""
from __future__ import annotations

import unittest
from unittest import mock

import pytest

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

    @mock.patch.dict("os.environ", {"KEIBA_MARK_RULE": "v1"})
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

    # ---- レースの状況による切替（situational。既定はオフ）----
    def _ents(self, rows):
        return [{"horse_number": i + 1, "win_prob": w, "top2_prob": t2, "top3_prob": t3, **({"ev_win": ev} if ev is not None else {})}
                for i, (w, t2, t3, ev) in enumerate(rows)]

    def test_situation_dom_close_value_normal(self):
        from src.utils.race_probabilities import race_situation

        n = 10
        dom = self._ents([(0.4, 0.6, 0.8, 1.0)] + [(0.0667, 0.15, 0.3, 0.8)] * (n - 1))
        self.assertEqual(race_situation(dom), "dom")
        close = self._ents([(0.2, 0.4, 0.6, 0.9), (0.19, 0.38, 0.55, 0.9)] + [(0.0762, 0.15, 0.3, 0.8)] * (n - 2))
        self.assertEqual(race_situation(close), "close")
        value = self._ents([(0.2, 0.4, 0.6, 0.9), (0.15, 0.35, 0.5, 1.4)] + [(0.0813, 0.15, 0.3, 0.8)] * (n - 2))
        self.assertEqual(race_situation(value), "value")
        normal = self._ents([(0.2, 0.4, 0.6, 0.9), (0.15, 0.35, 0.5, 0.95)] + [(0.0813, 0.15, 0.3, 0.8)] * (n - 2))
        self.assertEqual(race_situation(normal), "normal")

    def test_situation_without_odds_never_value(self):
        from src.utils.race_probabilities import race_situation

        es = self._ents([(0.2, 0.4, 0.6, None), (0.15, 0.35, 0.5, None)] + [(0.0813, 0.15, 0.3, None)] * 8)
        self.assertEqual(race_situation(es), "normal")

    def test_situational_value_moves_honmei_to_high_ev_horse(self):
        n = 10
        es = self._ents([(0.2, 0.4, 0.6, 0.9), (0.15, 0.35, 0.5, 1.4)] + [(0.0813, 0.15, 0.3, 0.8)] * (n - 2))
        assign_mece_marks(es, situational=True)
        types = {e["horse_number"]: e["mark_type"] for e in es}
        self.assertEqual((types[2], types[1]), ("honmei", "pair"))
        es2 = self._ents([(0.2, 0.4, 0.6, 0.9), (0.15, 0.35, 0.5, 1.4)] + [(0.0813, 0.15, 0.3, 0.8)] * (n - 2))
        assign_mece_marks(es2)  # 既定はオフ: ◎は勝率最上位
        self.assertEqual(es2[0]["mark_type"], "honmei")

    def test_situational_close_picks_higher_ev_of_top2(self):
        n = 10
        es = self._ents([(0.2, 0.4, 0.6, 0.9), (0.19, 0.38, 0.55, 1.05)] + [(0.0762, 0.15, 0.3, 0.8)] * (n - 2))
        assign_mece_marks(es, situational=True, cfg={"value_ev": 99.0})
        self.assertEqual((es[1]["mark_type"], es[0]["mark_type"]), ("honmei", "pair"))

    def test_situational_marks_counts_unchanged(self):
        for seed in range(15):
            es = self._race(14, seed)
            for i, e in enumerate(es):
                e["ev_win"] = 0.6 + (i * 7 % 11) / 10
            assign_mece_marks(es, situational=True)
            ts = [e["mark_type"] for e in es]
            self.assertTrue(all(ts.count(k) <= 1 for k in ("honmei", "pair", "show_val", "star")))
            self.assertLessEqual(ts.count("anchor"), 4)

    @mock.patch("src.utils.race_probabilities.star_v1_config", return_value={})
    def test_qualifies_star_odds_band(self, _m):
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

    @mock.patch.dict("os.environ", {"KEIBA_MARK_RULE": "v1"})
    @mock.patch("src.utils.race_probabilities.star_v1_config", return_value={})
    def test_assign_star_mark(self, _m):
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
        from unittest import mock

        with mock.patch("src.utils.ev_calibration.load_config", return_value={"enabled": False}):          # 較正が無効のときは従来の閾値
            self.assertEqual(buy_recommendation_tier({"ev_win": 1.3}), "強推奨")
            self.assertEqual(buy_recommendation_tier({"ev_place": 0.95}), "様子見")

    def test_buy_recommendation_tier_calibrated(self):
        from unittest import mock

        cfg = {"enabled": True, "kinds": {"win": {"method": "const", "value": 0.7}, "top2": {"method": "const", "value": 0.7},
                                          "place": {"method": "isotonic", "x": [0.5, 1.5], "y": [0.8, 1.2]}},
               "tiers": {"strong": 1.1, "recommend": 1.0, "watch": 0.9}}
        with mock.patch("src.utils.ev_calibration.load_config", return_value=cfg):
            self.assertEqual(buy_recommendation_tier({"ev_win": 3.0}), "—")                        # 単勝の EV は高くても較正後は 0.7
            self.assertEqual(buy_recommendation_tier({"ev_place": 1.5}), "強推奨")                  # 較正後 1.2
            self.assertEqual(buy_recommendation_tier({"ev_place": 0.9}), "様子見")                  # 較正後 0.96
            self.assertEqual(buy_recommendation_tier({"ev_win": 0}), "—")                           # EV が無い馬は較正しない


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


class TestStarCalibrated(unittest.TestCase):
    """★は較正後の単勝期待値で判定する（config/mark_rule.json の star_v1）。"""

    CFG = {"calibrated": True, "ev_win_cal_min": 0.95, "odds_min": 10.0, "odds_max": 30.0}

    def test_field_size_and_ev_ceiling(self):
        cfg = {**self.CFG, "min_field": 13, "ev_win_cal_max": 1.1}
        e = {"win_odds": 15.0, "ev_win": 1.0, "top3_prob": 0.1}
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=cfg), mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=1.0):
            self.assertFalse(qualifies_star_mark(e, 1 / 12))       # 12 頭以下は対象外
            self.assertTrue(qualifies_star_mark(e, 1 / 14))
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=cfg), mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=1.15):
            self.assertFalse(qualifies_star_mark(e, 1 / 14))       # 1.1 以上は対象外

    def test_uses_calibrated_ev_not_raw(self):
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=self.CFG), \
                mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=0.80):
            self.assertFalse(qualifies_star_mark({"win_odds": 15.0, "ev_win": 1.6, "top3_prob": 0.3}, 0.07))     # 生の EV が高くても較正後が低ければ★にしない
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=self.CFG), \
                mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=0.96):
            self.assertTrue(qualifies_star_mark({"win_odds": 15.0, "ev_win": 1.0, "top3_prob": 0.1}, 0.07))
            self.assertFalse(qualifies_star_mark({"win_odds": 35.0, "ev_win": 1.0, "top3_prob": 0.1}, 0.07))    # 30 倍超は対象外
            self.assertFalse(qualifies_star_mark({"win_odds": 8.0, "ev_win": 1.0, "top3_prob": 0.1}, 0.07))     # 10 倍未満は対象外

    def test_falls_back_when_calibration_unavailable(self):
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=self.CFG), \
                mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=None):
            self.assertTrue(qualifies_star_mark({"win_odds": 15.0, "ev_win": 1.2, "top3_prob": 0.1}, 0.07))     # 較正が無効のときは従来の判定


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


def test_market_top2_fair_odds_makes_market_ev_equal_the_payout_rate():
    """p2 を市場の連対確率に置けば EV は払戻率（馬連 0.775）になる。以前の『単勝×1.65』は平均で数倍高かった。"""
    import numpy as np

    from src.utils.race_probabilities import (
        RETURN_RATE_UMAREN,
        TOP2_MARKET_LAMBDA,
        harville_top2_prob,
        market_top2_fair_odds,
    )

    win_odds = {1: 2.5, 2: 4.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 40.0, 7: 80.0, 8: 120.0}
    fair = market_top2_fair_odds(win_odds)
    q = np.array([1 / o for o in win_odds.values()])
    q /= q.sum()
    w = q ** TOP2_MARKET_LAMBDA
    p2 = harville_top2_prob(w / w.sum())
    for (h, o), p in zip(fair.items(), p2):
        assert p * o == pytest.approx(RETURN_RATE_UMAREN, rel=0.02)
    # 旧換算（×1.65）は人気薄で桁違いに高い
    assert fair[8] < win_odds[8] * 1.65 / 2
    # オッズが揃わないレースは計算しない
    assert market_top2_fair_odds({1: 2.0, 2: 0.0, 3: 5.0, 4: 9.0}) == {}
    assert market_top2_fair_odds({1: 2.0, 2: 3.0}) == {}


def test_place_effective_odds_is_between_min_and_max_near_the_low_end():
    from src.utils.race_probabilities import PLACE_RANGE_WEIGHT, place_effective_odds

    assert place_effective_odds(1.2, 1.6) == pytest.approx(1.2 + PLACE_RANGE_WEIGHT * 0.4)
    assert place_effective_odds(1.5, 1.5) == pytest.approx(1.5)
    assert place_effective_odds(None, 2.0) is None and place_effective_odds(0, 2.0) is None
    assert place_effective_odds(2.0, None) == pytest.approx(2.0)
    assert place_effective_odds(1.2, 1.6) < (1.2 + 1.6) / 2


def test_buy_tier_uses_calibrated_ev_only_when_enabled(monkeypatch):
    from src.utils import ev_calibration
    from src.utils.race_probabilities import buy_recommendation_tier

    e = {"ev_win": 1.5, "ev_top2": 0.5, "ev_place": 0.5}
    monkeypatch.setattr(ev_calibration, "load_config", lambda: {})
    assert buy_recommendation_tier(e) == "強推奨"
    monkeypatch.setattr(ev_calibration, "load_config", lambda: {"enabled": True, "k": {"win": 0.2}, "base": {"win": 0.8}})
    assert buy_recommendation_tier(e) == "様子見"      # 0.8 + 0.2×(1.5−0.8) = 0.94 → 様子見（0.9 以上）


class TestStarV2Calibrated(unittest.TestCase):
    CFG = {"calibrated": True, "ev_win_cal_min": 0.95, "odds_min": 10.0, "odds_max": 30.0}

    def _entries(self):
        n = 8
        return [{"horse_number": i + 1, "win_prob": 0.2 if i == 0 else 0.1, "top2_prob": 0.3, "top3_prob": 0.4, "win_odds": 5.0 if i == 0 else 15.0 + i, "ev_win": 1.5, "ev_place": 1.2} for i in range(n)]

    def test_v2_star_uses_calibrated_ev(self):
        from src.utils.race_probabilities import assign_mece_marks_v2

        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=self.CFG), \
                mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=0.8):
            es = self._entries()
            assign_mece_marks_v2(es)
            self.assertFalse(any(e["mark_type"] == "star" for e in es))       # 較正後が低ければ、生の期待値が高くても★は付かない
        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value=self.CFG), \
                mock.patch("src.utils.race_probabilities._cal_horse_ev", return_value=1.0):
            es = self._entries()
            assign_mece_marks_v2(es)
            self.assertEqual(sum(e["mark_type"] == "star" for e in es), 1)
