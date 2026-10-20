import numpy as np
import pandas as pd

from src.scripts.data import tp48_eval as t


def test_odds_cube_and_trio_implied_sum_to_one():
    nums = np.array([2, 5, 7, 9])
    rows = []
    from itertools import permutations
    for a, b, c in permutations(nums, 3):
        rows.append((int(a), int(b), int(c), 50.0 + a + b * 2 + c * 3))
    tri = pd.DataFrame(rows, columns=["a", "b", "c", "odds"])
    cube = t.odds_cube(nums, tri)
    assert np.isfinite(cube).sum() == 24 and np.isnan(cube[0, 0, 0])
    from src.scripts.data.tp36_single import idx_triples
    pm = t.trio_implied(cube, idx_triples(4))
    assert abs(pm.sum() - 1.0) < 1e-9


def test_odds_cube_ignores_missing_and_unknown_horses():
    nums = np.array([1, 2, 3])
    tri = pd.DataFrame({"a": [1, 9], "b": [2, 2], "c": [3, 3], "odds": [10.0, 5.0]})
    cube = t.odds_cube(nums, tri)
    assert cube[0, 1, 2] == 10.0 and np.isfinite(cube).sum() == 1


def test_mark_tickets_counts_and_unordered_keys():
    marks = {1: "honmei", 2: "pair", 3: "show_val", 4: "star", 5: "anchor", 6: "anchor", 7: "none"}
    tk = t.mark_tickets(marks)
    assert len(tk["umaren_honmei_all"][1]) == 5
    assert len(tk["sanrenpuku_honmei_pair_flow"][1]) == 4
    assert len(tk["sanrentan_honmei_first_flow"][1]) == 20
    assert len(tk["sanrenpuku_box4"][1]) == 4
    n, ret = t.stake_ret([(2, 1), (1, 2)], {(1, 2): 570.0}, ordered=False)
    assert (n, ret) == (1, 570.0)
    n, ret = t.stake_ret([(2, 1), (1, 2)], {(1, 2): 570.0}, ordered=True)
    assert (n, ret) == (2, 570.0)


def test_place_adj_between_bounds():
    assert abs(t.place_adj(np.array([1.0]), np.array([2.0]))[0] - 1.3) < 1e-12
