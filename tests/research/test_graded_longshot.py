"""うまうまー式の穴馬の分析（src/research/graded/longshot.py）のテスト。"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.research.graded import longshot as LS


def _data(seed=0, n=50, T=4, lv=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, lv, size=(n, T)).astype(np.int32)
    X[rng.random((n, T)) < 0.1] = -1
    top = (rng.random(n) < 0.2).astype(float)
    Y = np.column_stack([np.ones(n), top, top, np.zeros(n), np.zeros(n), np.ones(n), top * 5, top * 2])
    race = np.array([f"r{i // 10}" for i in range(n)])
    return X, Y, race


def test_cells_match_brute_force():
    X, Y, race = _data()
    C = LS.cells(X, Y, race, 2)
    got = {(tuple(r.ts), tuple(r.cs)): (r.n, r.k, r.edk) for r in C.itertuples()}
    for d in (1, 2):
        for ts in itertools.combinations(range(X.shape[1]), d):
            sub = X[:, ts]
            ok = (sub >= 0).all(1)
            for cs in {tuple(map(int, r)) for r in sub[ok]}:
                m = ok & (sub == np.array(cs)).all(1)
                edk = len(set(race[m & (Y[:, 1] > 0)]))
                assert got[(ts, cs)] == (m.sum(), Y[m, 1].sum(), edk)
    assert len(got) == sum(len({tuple(r) for r in X[:, ts][(X[:, ts] >= 0).all(1)]}) for d in (1, 2) for ts in itertools.combinations(range(4), d))


def test_score_and_pass_rule():
    C = pd.DataFrame({"n": [10, 40, 10], "k": [5, 8, 2], "edk": [3, 5, 2], "w1": 0, "w2": 0, "w3": 0, "np": 1, "rt": 0, "rf": 0, "ts": [(0,)] * 3, "cs": [(1,), (2,), (3,)]})
    S = LS.score(C, 100, 10)     # 分母の複勝率 10%
    assert S["recall"].tolist() == [0.5, 0.8, 0.2] and S["lift"].round(2).tolist() == [5.0, 2.0, 2.0]
    assert S["pass"].tolist() == [False, True, False]   # 再現率 60% 未満・好走 3 頭未満は採らない


def test_bh():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    assert LS.bh(p, 4, 0.10).tolist() == [True, True, True, False]
    assert LS.bh(p, 400, 0.10).tolist() == [False, False, False, False]


def test_permute_within_race():
    y = pd.DataFrame({"w1": [1, 0, 0, 1, 0, 0], "w2": 0, "w3": 0, "top3": [1, 0, 0, 1, 0, 0], "np": 1.0, "rt": [5, 0, 0, 7, 0, 0], "rf": [2, 0, 0, 3, 0, 0]})
    race = np.array(["a", "a", "a", "b", "b", "b"])
    for sd in range(5):
        p = LS.permute(y, race, sd)
        assert p.groupby(race)["rt"].sum().tolist() == [5, 7] and ((p["w1"] == 1) == (p["rt"] > 0)).all()


def test_example_stats_counts_recall_and_lift():
    R = pd.DataFrame({"rkey": "根岸S", "main": True, "year": 2020, "race_id": ["x"] * 4 + ["y"] * 4, "pop": [6, 7, 8, 9, 6, 7, 8, 9],
                      "age": [6, 7, 4, 5, 6, 4, 4, 4], "fin": [1, 5, 2, 9, 3, 4, 6, 7], "top3": [1, 0, 1, 0, 1, 0, 0, 0],
                      "pay_ok": True, "ret_tan": [9.0, 0, 0, 0, 0, 0, 0, 0], "ret_fuku": [3.0, 0, 2.0, 0, 1.5, 0, 0, 0]})
    for c in ("wt", "interval", "p_fin", "p_tdiff", "p_pos4", "p_l3rank"):
        R[c] = np.nan
    R["p_grade"], R["p_vc"] = None, None
    e = next(x for x in LS.example_stats(R, (2016, 2024)) if x["condition"] == "6 歳以上")
    assert (e["k"], e["K"], e["N"]) == (2, 3, 8) and e["hit"]["rec"] == [1, 0, 1, 1] and abs(e["lift"] - (2 / 3) / (3 / 8)) < 1e-9


def test_dedupe_and_matches_and_modal_courses():
    X = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32)
    C = pd.DataFrame({"ts": [(0, 1), (0,), (1,)], "cs": [(0, 1), (0,), (1,)], "n": 2})
    D = LS.dedupe(C, X, np.array([10, 11, 12]))
    assert D["ts"].tolist() == [(0,)]                         # 同じ 2 頭になる 3 つの条件は、型の数が少ない 1 つに
    assert LS.matches(X, (0, 1), (0, 1)).tolist() == [True, True, False]
    from src.research.graded.run import _modal_courses
    g = pd.DataFrame({"rkey": ["宝塚記念"] * 3 + ["X"] * 2, "course": ["阪神", "阪神", "京都", "A", "B"], "race_id": ["1", "2", "3", "4", "5"],
                      "dt": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01", "2023-01-01", "2024-01-01"])})
    assert _modal_courses(g, 2024) == {"宝塚記念": "阪神", "X": "B"}   # 同数なら新しい方
