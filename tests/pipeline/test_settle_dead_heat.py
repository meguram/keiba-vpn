import unittest

from src.pipeline.inference import prediction_results as PR


def _result(finish, payoff, n=10):
    return {"finish": finish, "n_runners": n, "payoff": PR.parse_payoff(payoff)}


# 3 着同着（2 と 3）。実データ 202405010309 と同じ形（ワイドは 5 組、3 連複・3 連単は 2 組）
TIE3 = _result(
    {7: 1, 4: 2, 2: 3, 3: 3, 1: 5, 5: 6, 8: 7, 6: 8},
    {
        "単勝": {"numbers": "7", "payout": "370"},
        "複勝": [{"numbers": "7", "payout": "110"}, {"numbers": "4", "payout": "110"}, {"numbers": "2", "payout": "110"}, {"numbers": "3", "payout": "200"}],
        "馬連": {"numbers": "4 - 7", "payout": "390"},
        "ワイド": [{"numbers": "4 - 7", "payout": "120"}, {"numbers": "2 - 7", "payout": "150"}, {"numbers": "3 - 7", "payout": "460"}, {"numbers": "2 - 4", "payout": "120"}, {"numbers": "3 - 4", "payout": "340"}],
        "馬単": {"numbers": "7 → 4", "payout": "760"},
        "三連複": [{"numbers": "2 - 4 - 7", "payout": "210"}, {"numbers": "3 - 4 - 7", "payout": "790"}],
        "三連単": [{"numbers": "7 → 4 → 2", "payout": "1,060"}, {"numbers": "7 → 4 → 3", "payout": "4,050"}],
    },
    n=8,
)


class SettleDeadHeatTest(unittest.TestCase):
    def test_third_place_tie(self):
        s = PR.settle_candidate
        self.assertEqual(s("wide", (3, 2), 1000, 5.0, TIE3), (False, 0.0, False))  # 3 着を分け合う 2 頭の組は当たらない
        self.assertEqual(s("wide", (2, 4), 1000, 5.0, TIE3), (True, 1200.0, False))
        self.assertEqual(s("sanrenpuku", (2, 4, 7), 100, 5.0, TIE3), (True, 210.0, False))
        self.assertEqual(s("sanrenpuku", (7, 3, 4), 100, 5.0, TIE3), (True, 790.0, False))
        self.assertEqual(s("sanrenpuku", (2, 3, 7), 100, 5.0, TIE3), (False, 0.0, False))
        self.assertEqual(s("fukusho", (3,), 100, 2.0, TIE3), (True, 200.0, False))
        self.assertEqual(s("sanrentan", (7, 4, 3), 100, 5.0, TIE3), (True, 4050.0, False))
        self.assertEqual(s("sanrentan", (7, 3, 4), 100, 5.0, TIE3), (False, 0.0, False))
        self.assertEqual(s("umatan", (7, 4), 100, 5.0, TIE3), (True, 760.0, False))

    def test_second_place_tie_umaren(self):
        res = _result({1: 1, 2: 2, 3: 2, 4: 4}, {"馬連": [{"numbers": "1 - 2", "payout": "500"}, {"numbers": "1 - 3", "payout": "600"}]}, n=4)
        self.assertEqual(PR.settle_candidate("umaren", (3, 1), 100, 5.0, res), (True, 600.0, False))
        self.assertEqual(PR.settle_candidate("umaren", (2, 3), 100, 5.0, res), (False, 0.0, False))

    def test_first_place_tie(self):
        res = _result({3: 1, 5: 1, 1: 3, 2: 4}, {"単勝": [{"numbers": "3", "payout": "250"}, {"numbers": "5", "payout": "400"}]}, n=10)
        self.assertEqual(PR.settle_candidate("tansho", (5,), 100, 4.0, res), (True, 400.0, False))
        self.assertEqual(PR.settle_candidate("umaren", (5, 3), 100, 4.0, res), (True, 400.0, True))  # 払戻が無ければオッズから概算
        self.assertEqual(PR.settle_candidate("tansho", (1,), 100, 4.0, res), (False, 0.0, False))


if __name__ == "__main__":
    unittest.main()
