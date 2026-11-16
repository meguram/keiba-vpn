"""重賞のルールベース予想家の「買い」の組み合わせ探し（src/research/graded/buy.py）のテスト。"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.research.graded import buy as BY


def _data(seed=0, n=60, T=6, lv=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, lv, size=(n, T)).astype(np.int32)
    X[rng.random((n, T)) < 0.1] = -1            # 不明の水準
    top3 = (rng.random(n) < 0.35).astype(float)
    w1 = top3 * (rng.random(n) < 0.4)
    w2 = (top3 - w1) * (rng.random(n) < 0.5)
    w3 = top3 - w1 - w2
    npay = (rng.random(n) < 0.95).astype(float)
    rt = w1 * rng.uniform(1, 9, n) * npay
    rf = top3 * rng.uniform(1, 4, n) * npay
    e3 = np.full(n, 0.2)
    Y = np.column_stack([np.ones(n), top3, w1, w2, w3, npay, rt, rf, e3])
    return X, Y


def _brute(X, Y, max_depth):
    found = set()
    n, T = X.shape
    for d in range(1, max_depth + 1):
        for ts in itertools.combinations(range(T), d):
            sub = X[:, ts]
            ok = (sub >= 0).all(1)
            for cs in {tuple(r) for r in sub[ok]}:
                m = ok & (sub == np.array(cs)).all(1)
                s = Y[m].sum(0)
                if s[0] >= BY.N_FLOOR and s[1] >= BY.RATE_MIN * s[0] - 1e-9 and s[5] > 0 and (s[6] / s[5] > 1 or s[7] / s[5] > 1):
                    found.add((ts, tuple(int(c) for c in cs)))
    return found


def test_search_matches_brute_force_including_pruning():
    for seed in range(4):
        X, Y = _data(seed)
        got = {(tuple(ts), tuple(cs)) for ts, cs in BY.search(X, Y, 3)}
        assert got == _brute(X, Y, 3)
        got2 = {(tuple(ts), tuple(cs)) for ts, cs in BY.search(X, Y, 2)}
        assert got2 == _brute(X, Y, 2)


def test_describe_dedupes_same_rows_and_counts_races_horses():
    X = np.array([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=np.int32)
    Y = np.array([[1, 1, 1, 0, 0, 1, 5, 2, 0.2]] * 4, dtype=float)
    found = [((0,), (0,)), ((0, 1), (0, 0))]     # 同じ 2 走（行 0・1）になる 2 つの条件
    c = BY.describe(X, Y, np.array(["r1", "r2", "r1", "r2"]), np.array(["h1", "h1", "h2", "h3"]), np.array([10, 11, 12, 13]), found)
    assert len(c) == 1 and c[0]["ts"] == (0,) and c[0]["n_equiv"] == 2
    assert c[0]["ed"] == 2 and c[0]["nh"] == 1 and c[0]["rec"] == [2, 0, 0, 0] and c[0]["roi_t"] == 5


def test_permute_keeps_race_membership_and_moves_outcomes_together():
    b = pd.DataFrame({"race_id": ["a"] * 5 + ["b"] * 5, "w1": [1, 0, 0, 0, 0] * 2, "w2": [0, 1, 0, 0, 0] * 2, "w3": [0, 0, 1, 0, 0] * 2,
                      "top3": [1, 1, 1, 0, 0] * 2, "np": 1.0, "rt": [5, 0, 0, 0, 0] * 2, "rf": [2, 3, 4, 0, 0] * 2, "e3": 0.6, "x": range(10)})
    p = BY.permute_within_race(b, 3)
    for r in ("a", "b"):
        assert sorted(p[p.race_id == r]["rf"]) == sorted(b[b.race_id == r]["rf"])
    assert (p["x"] == b["x"]).all() and (p["e3"] == b["e3"]).all()
    assert ((p["w1"] == 1) == (p["rt"] == 5)).all()          # 着順と払戻は一緒に動く


def test_cells_ok_matches_cell_ok():
    rng = np.random.default_rng(0)
    cells = pd.DataFrame({"layer": rng.choice(["主", "副"], 200), "depth": rng.integers(1, 4, 200), "n": rng.integers(2, 12, 200),
                          "np": rng.integers(2, 12, 200), "ed": rng.integers(1, 4, 200), "nh": rng.integers(1, 4, 200),
                          "roi_t": rng.uniform(0, 2, 200), "roi_f": rng.uniform(0, 2, 200)})
    for v in BY.all_versions():
        assert BY.cells_ok(cells, v).tolist() == [BY.cell_ok(c, v) for c in cells.to_dict("records")]


def test_hits_and_buy_metrics_count_horses_once():
    cells = pd.DataFrame({"rkey": ["R", "R"], "course": ["C", "C"], "ts": [(0,), (1,)], "cs": [(1,), (0,)]})
    rows = pd.DataFrame({"race_id": ["x"] * 3, "rkey": "R", "course": "C", "hn": [1, 2, 3], "fin": [1, 5, 2],
                         "pay_ok": [True, True, None], "ret_tan": [4.0, 0, 0], "ret_fuku": [1.5, 0, 1.2]})
    X = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
    H = BY.hits_for(cells, rows, X)
    cnt = np.bincount(H["i"], minlength=3)
    assert cnt.tolist() == [2, 1, 0]
    m = BY.buy_metrics(rows, cnt >= 1, np.array([0.1, 0.9, 0.5]))
    assert m["buy"]["rec"] == [1, 0, 0, 1] and m["buy"]["top3_rate"] == 0.5 and m["buy"]["roi_t"] == 2.0
    assert m["honmei_buy"]["rec"] == [0, 0, 0, 1]           # 買いの馬のうち前日版の点が最大の 2 番
    assert m["honmei_preday_same_races"]["rec"] == [0, 0, 0, 1]
    assert m["all_same_races"]["n"] == 3 and m["buy_share"] == 2 / 3


def test_hits_skip_other_race_or_course_and_permute_interleaved():
    cells = pd.DataFrame({"rkey": ["R"], "course": ["C"], "ts": [(0,)], "cs": [(1,)]})
    rows = pd.DataFrame({"rkey": ["R", "R", "Q"], "course": ["C", "D", "C"]})
    X = np.array([[1], [1], [1]], dtype=np.int32)
    assert BY.hits_for(cells, rows, X)["i"].tolist() == [0]          # 別のコース・別の重賞には当てない
    b = pd.DataFrame({"race_id": list("abababab"), "w1": [1, 0] * 4, "w2": 0, "w3": 0, "top3": [1, 0] * 4, "np": 1.0,
                      "rt": [9, 0] * 4, "rf": [3, 0] * 4, "e3": 0.2})
    for sd in range(5):
        p = BY.permute_within_race(b, sd)
        assert p.groupby("race_id")["w1"].sum().tolist() == [4, 0]   # レースの中だけで入れ替わる
