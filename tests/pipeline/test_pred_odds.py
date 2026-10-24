"""予測オッズ（そのレースのオッズを使わない）での馬券の表の生成の単体テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import pred_odds as po
from src.pipeline.betting import ticket_env as te
from tests.pipeline.test_betting_policy import _race


class PredOddsTest(unittest.TestCase):
    def test_assert_no_odds_features(self):
        po.assert_no_odds_features(["speed_max", "b_aff_course", "a_trend"])
        with self.assertRaises(ValueError):
            po.assert_no_odds_features(["speed_max", "prev1_odds"])
        with self.assertRaises(ValueError):
            po.assert_no_odds_features(["popularity_rank"])

    def test_normalize_win_odds(self):
        race = np.array(["a"] * 4 + ["b"] * 3)
        o = np.array([2.0, 4.0, 8.0, 16.0, 1.5, 3.0, 6.0])
        n = po.normalize_win_odds(race, o, 0.8)
        for r in ("a", "b"):
            self.assertAlmostEqual(float((1 / n[race == r]).sum()), 0.8, places=6)

    def test_pair_constants(self):
        t = pd.DataFrame({"type": ["umaren"] * 4 + ["wide"] * 2, "odds": [10, 20, 30, 40, 5, 6], "p_mkt": [0.08, 0.04, 0.026, 0.02, 0.15, 0.13]})
        c = po.pair_constants(t)
        self.assertAlmostEqual(c["umaren"], float(np.median([0.8, 0.8, 0.78, 0.8])), places=2)
        self.assertIn("umatan", c)

    def test_tickets_use_predicted_odds_only(self):
        pred, wp, pair, pay = _race("r1")
        # 実際のオッズを大きく変えても、判断に使う列（odds・p_mkt・候補の選び方）は変わらない
        podds = pd.DataFrame({"race_id": "r1", "horse_id": pred["horse_id"], "win_pred": 0.8 / pred["p_win"].to_numpy() * 1.1, "place_pred": np.maximum(0.8 / pred["p_win"].to_numpy() / 3, 1.1)})
        consts = {"umaren": 0.78, "wide": 0.8, "umatan": 0.75}
        a = te.build_tickets_predicted(pred, wp, pay, podds, consts, (0.8, 0.7), K=6)
        wp2 = wp.copy(); wp2["win_odds"] = wp2["win_odds"] * 3.0; wp2["place_odds_min"] = 99.0
        b = te.build_tickets_predicted(pred, wp2, pay, podds, consts, (0.8, 0.7), K=6)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        self.assertEqual(set(a["type"]), set(te.TYPES))
        # 精算用の mult だけが実際の払戻
        hit = a[(a["type"] == "tansho") & (a["combo"] == "1")]
        if len(hit):
            self.assertAlmostEqual(float(hit["mult"].iloc[0]), 5.0)
        # 予測の単勝オッズは、判断用の odds にそのまま入る
        t1 = a[(a["type"] == "tansho") & (a["combo"] == "1")]
        if len(t1):
            self.assertAlmostEqual(float(t1["odds"].iloc[0]), float(podds["win_pred"].iloc[0]))

    def test_race_with_runner_not_in_prediction_is_skipped(self):
        pred, wp, pair, pay = _race("r1")
        podds = pd.DataFrame({"race_id": "r1", "horse_id": pred["horse_id"], "win_pred": 0.8 / pred["p_win"].to_numpy(), "place_pred": 2.0})
        out = te.build_tickets_predicted(pred.iloc[:-1], wp, pay, podds, {"umaren": 0.78, "wide": 0.8, "umatan": 0.75}, (0.8, 0.7))
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
