"""馬券戦略の探索（規則ベースの買い目・払戻の精算・reality check）の単体テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import lab


class LabTest(unittest.TestCase):
    def test_combos(self):
        top = [3, 7, 1, 9, 5, 2]
        self.assertEqual(lab.combos(lab.Rule("tansho", "single"), top), ["3"])
        self.assertEqual(lab.combos(lab.Rule("fukusho", "box", 2), top), ["3", "7"])
        self.assertEqual(sorted(lab.combos(lab.Rule("umaren", "box", 3), top)), ["1-3", "1-7", "3-7"])
        self.assertEqual(len(lab.combos(lab.Rule("sanrenpuku", "box", 5), top)), 10)
        self.assertEqual(len(lab.combos(lab.Rule("sanrentan", "box", 4), top)), 24)
        self.assertEqual(lab.combos(lab.Rule("umatan", "nagashi", 2), top), ["3>7", "3>1"])
        self.assertEqual(sorted(lab.combos(lab.Rule("sanrenpuku", "nagashi", 3), top)), sorted(["1-3-7", "3-7-9", "1-3-9"]))
        self.assertEqual(len(lab.combos(lab.Rule("sanrentan", "nagashi", 3), top)), 6)
        f = lab.combos(lab.Rule("sanrentan", "formation", 1, 2, 3), top)       # 1 着=[3] 2 着=[3,7,1] 3 着=[3,7,1,9,5,2] の重複なし
        self.assertEqual(f[0].split(">")[0], "3")
        self.assertTrue(all(len(set(x.split(">"))) == 3 for x in f))
        self.assertEqual(lab.combos(lab.Rule("sanrentan", "box", 5), [1, 2, 3]), [])         # 足りなければ空

    def test_evaluate_and_orders(self):
        pred = pd.DataFrame({"race_id": ["r1"] * 4, "horse_id": list("abcd"), "p_win": [0.1, 0.5, 0.3, 0.1], "p_top3": [0.3, 0.9, 0.7, 0.4]})
        hn = pd.DataFrame({"race_id": ["r1"] * 4, "horse_id": list("abcd"), "horse_number": [1, 2, 3, 4]})
        o = lab.build_orders(pred, hn)
        self.assertEqual(o["r1"]["win"][:2], [2, 3]); self.assertAlmostEqual(o["r1"]["p1"], 0.5)
        payoff = pd.DataFrame({"race_id": ["r1", "r1", "r1"], "bet_type": ["tansho", "umaren", "umaren"], "combo": ["2", "2-3", "1-4"], "payout": [300, 800, 5000], "parsed": True, "ambiguous": False})
        pay = lab.payoff_lookup(payoff)
        d = lab.evaluate_rule(lab.Rule("umaren", "box", 2), o, pay, ["r1"])
        self.assertEqual(float(d["stake"].iloc[0]), 100.0); self.assertEqual(float(d["ret"].iloc[0]), 800.0)
        self.assertTrue(lab.evaluate_rule(lab.Rule("tansho", "single"), o, pay, ["r1"], min_p1=0.6).empty)
        # ランダム基準は再現性がある
        a = lab.evaluate_rule(lab.Rule("umaren", "box", 2), o, pay, ["r1"], random_seed=1)
        b = lab.evaluate_rule(lab.Rule("umaren", "box", 2), o, pay, ["r1"], random_seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_reality_check_detects_real_edge_and_not_noise(self):
        rng = np.random.default_rng(0)
        n, m = 3000, 40
        noise = rng.normal(-0.2, 1.0, (n, m))                       # すべて期待損益 −0.2（控除率の損）
        r0 = lab.reality_check(noise, n_boot=300)
        self.assertGreater(r0["p_value"], 0.2)
        edge = noise.copy(); edge[:, 7] += 0.35                        # 1 つだけ本当に +0.15 の戦略
        r1 = lab.reality_check(edge, n_boot=300)
        self.assertEqual(r1["best"], 7); self.assertLess(r1["p_value"], 0.05)

    def test_reality_check_not_fooled_by_many_strategies(self):
        rng = np.random.default_rng(1)
        n, m = 800, 300                                              # 期待損益 0 の戦略が 300 個: 最良は偶然 +3σ 近く見えても p 値は小さくならない
        prof = rng.normal(0.0, 1.0, (n, m))
        r = lab.reality_check(prof, n_boot=300)
        self.assertGreater(r["best_mean"], 0.05)                      # 最良は +に見える
        self.assertGreater(r["p_value"], 0.05)                        # しかし偶然の範囲


class Level1Test(unittest.TestCase):
    def _tickets(self):
        return pd.DataFrame({"race_id": ["a", "a", "a", "b", "b"], "type": ["umaren", "umaren", "wide", "umaren", "umaren"], "p_comb": [0.2, 0.1, 0.4, 0.3, 0.05], "p_model": [0.2, 0.1, 0.4, 0.3, 0.05],
                             "odds": [6.0, 15.0, 2.0, 4.0, 40.0], "mult": [6.0, 0.0, 0.0, 0.0, 0.0]})

    def test_noisy_odds(self):
        o = np.array([2.0, 10.0, 50.0])
        np.testing.assert_allclose(lab.noisy_odds(o, 0.0), o)
        n = lab.noisy_odds(np.full(2000, 10.0), 0.2, seed=1)
        self.assertAlmostEqual(float(np.log(n / 10.0).std()), 0.2, delta=0.02)
        self.assertTrue((lab.noisy_odds(o, 1.0) >= 1.0).all())

    def test_select_portfolio(self):
        d = lab.prepare_tickets(self._tickets(), 0.0)
        port = lab.select_portfolio(d, "umaren", tau=1.0, pmin=0.0, topk=2)           # EV: a=1.2, 1.5; b=1.2, 2.0
        self.assertEqual(sorted(port["race_id"]), ["a", "b"])
        a = port[port["race_id"] == "a"].iloc[0]
        self.assertEqual(int(a["n"]), 2); self.assertAlmostEqual(a["mean_mult"], 3.0); self.assertAlmostEqual(a["ev_mean"], (1.2 + 1.5) / 2)
        self.assertAlmostEqual(a["hit"], 1 - 0.8 * 0.9)
        self.assertEqual(len(lab.select_portfolio(d, "umaren", tau=1.0, pmin=0.25, topk=2)), 1)            # p 下限でレース b の 1 点だけ
        self.assertEqual(len(lab.select_portfolio(d, "umaren", tau=1.6, pmin=0.0, topk=1)), 1)             # b の 2.0 だけ
        self.assertTrue(lab.select_portfolio(d, "umaren", tau=9.0, pmin=0.0, topk=1).empty)

    def test_tier_and_settle(self):
        port = pd.DataFrame({"race_id": ["a", "b", "c", "d"], "n": 1, "mean_mult": [2.0, 0.0, 1.0, 0.0], "ev_mean": [0.95, 1.12, 1.30, 1.30], "hit": [0.9, 0.2, 0.5, 0.35]})
        self.assertEqual(list(lab.tier_stake(port, "ev")), [500, 1000, 2000, 2000])
        self.assertEqual(list(lab.tier_stake(port, "evhit")), [500, 500, 2000, 1000])                           # hit 0.2 → 500 に抑える / 0.35 → 1000 に抑える
        s = lab.settle_portfolio(port, lab.tier_stake(port, "flat"))
        self.assertEqual(float(s["ret"].sum()), 1000 * (2.0 + 0.0 + 1.0 + 0.0))


if __name__ == "__main__":
    unittest.main()
