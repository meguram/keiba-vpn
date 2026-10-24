import unittest

from src.pipeline.betting import winning_combos as W


class WinningCombosTest(unittest.TestCase):
    def test_no_tie(self):
        g = W.finish_groups({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
        self.assertEqual(W.winning_combos(g, "tansho"), {(1,)})
        self.assertEqual(W.winning_combos(g, "fukusho"), {(1,), (2,), (3,)})
        self.assertEqual(W.winning_combos(g, "umaren"), {(1, 2)})
        self.assertEqual(W.winning_combos(g, "umatan"), {(1, 2)})
        self.assertEqual(W.winning_combos(g, "wide"), {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(W.winning_combos(g, "sanrenpuku"), {(1, 2, 3)})
        self.assertEqual(W.winning_combos(g, "sanrentan"), {(1, 2, 3)})

    def test_third_place_dead_heat(self):
        # 202405010309: 7,4 の次に 2 と 3 が 3 着同着
        g = W.finish_groups({7: 1, 4: 2, 2: 3, 3: 3, 1: 5, 5: 6, 8: 7, 6: 8})
        self.assertEqual(W.winning_combos(g, "tansho"), {(7,)})
        self.assertEqual(W.winning_combos(g, "fukusho"), {(7,), (4,), (2,), (3,)})
        self.assertEqual(W.winning_combos(g, "umaren"), {(4, 7)})
        self.assertEqual(W.winning_combos(g, "wide"), {(4, 7), (2, 7), (3, 7), (2, 4), (3, 4)})  # 3 着を分け合う 2-3 は当たらない
        self.assertEqual(W.winning_combos(g, "umatan"), {(7, 4)})
        self.assertEqual(W.winning_combos(g, "sanrenpuku"), {(2, 4, 7), (3, 4, 7)})
        self.assertEqual(W.winning_combos(g, "sanrentan"), {(7, 4, 2), (7, 4, 3)})

    def test_first_place_dead_heat(self):
        g = W.finish_groups({3: 1, 5: 1, 1: 3, 2: 4})
        self.assertEqual(W.winning_combos(g, "tansho"), {(3,), (5,)})
        self.assertEqual(W.winning_combos(g, "umaren"), {(3, 5)})
        self.assertEqual(W.winning_combos(g, "umatan"), {(3, 5), (5, 3)})
        self.assertEqual(W.winning_combos(g, "sanrenpuku"), {(1, 3, 5)})
        self.assertEqual(W.winning_combos(g, "sanrentan"), {(3, 5, 1), (5, 3, 1)})
        self.assertEqual(W.winning_combos(g, "wide"), {(1, 3), (1, 5), (3, 5)})
        self.assertEqual(W.winning_combos(g, "fukusho", n_runners=10), {(1,), (3,), (5,)})

    def test_second_place_dead_heat(self):
        g = W.finish_groups({1: 1, 2: 2, 3: 2, 4: 4})
        self.assertEqual(W.winning_combos(g, "umaren"), {(1, 2), (1, 3)})
        self.assertEqual(W.winning_combos(g, "umatan"), {(1, 2), (1, 3)})
        self.assertEqual(W.winning_combos(g, "sanrenpuku"), {(1, 2, 3)})
        self.assertEqual(W.winning_combos(g, "sanrentan"), {(1, 2, 3), (1, 3, 2)})

    def test_wide_second_place_tie_and_three_way_third(self):
        g = W.finish_groups({1: 1, 2: 2, 3: 2, 4: 4})
        self.assertEqual(W.winning_combos(g, "wide"), {(1, 2), (1, 3), (2, 3)})
        g = W.finish_groups({1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 6})
        self.assertEqual(W.winning_combos(g, "wide"), {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)})

    def test_three_way_first(self):
        g = W.finish_groups({4: 1, 6: 1, 9: 1, 2: 4})
        self.assertEqual(len(W.winning_combos(g, "umaren")), 3)
        self.assertEqual(len(W.winning_combos(g, "umatan")), 6)
        self.assertEqual(W.winning_combos(g, "sanrenpuku"), {(4, 6, 9)})
        self.assertEqual(len(W.winning_combos(g, "sanrentan")), 6)

    def test_fukusho_by_field_size(self):
        g = W.finish_groups({i: i for i in range(1, 8)})
        self.assertEqual(W.winning_combos(g, "fukusho", n_runners=7), {(1,), (2,)})
        self.assertEqual(W.winning_combos(g, "fukusho", n_runners=4), set())
        self.assertEqual(W.winning_combos(g, "fukusho", n_runners=8), {(1,), (2,), (3,)})

    def test_excludes_nonfinishers(self):
        g = W.finish_groups({1: 1, 2: 2, 3: 3, 4: -1, 5: 0})
        self.assertEqual(g, [[1], [2], [3]])

    def test_wakuren_same_bracket(self):
        g = W.finish_groups({1: 1, 2: 2, 3: 3})
        self.assertEqual(W.winning_combos(g, "wakuren", brackets={1: 1, 2: 1, 3: 2}), {(1, 1)})
        self.assertEqual(W.winning_combos(g, "wakuren"), set())

    def test_parse_and_payoff(self):
        self.assertEqual(W.parse_combo("wide", "7 - 4"), (4, 7))
        self.assertEqual(W.parse_combo("umatan", "7 → 4"), (7, 4))
        self.assertEqual(W.parse_combo("sanrentan", "7>4>2"), (7, 4, 2))
        self.assertIsNone(W.parse_combo("wide", "x - y"))
        p = W.normalize_payoff({"ワイド": [{"numbers": "4 - 7", "payout": "1,120"}, {"numbers": "2 - 7", "payout": "150"}], "馬単": {"numbers": "7 → 4", "payout": "760"}})
        self.assertEqual(p["wide"], {(4, 7): 1120, (2, 7): 150})
        self.assertEqual(W.hit_payout("wide", (7, 4), p["wide"]), 1120)
        self.assertEqual(W.hit_payout("umatan", (4, 7), p["umatan"]), 0)
        self.assertEqual(W.hit_payout("umatan", (7, 4), p["umatan"]), 760)

    def test_diff_expected(self):
        g = W.finish_groups({7: 1, 4: 2, 2: 3, 3: 3, 1: 5})
        d = W.diff_expected(g, "wide", [(4, 7), (2, 7), (3, 7), (2, 4), (3, 4)])
        self.assertEqual(d["missing"], [])
        self.assertEqual(d["extra"], [])
        d = W.diff_expected(g, "wide", [(4, 7), (2, 7), (3, 7), (2, 4), (3, 4), (2, 3)])
        self.assertEqual(d["extra"], [(2, 3)])


if __name__ == "__main__":
    unittest.main()
