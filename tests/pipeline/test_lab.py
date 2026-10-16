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


if __name__ == "__main__":
    unittest.main()
