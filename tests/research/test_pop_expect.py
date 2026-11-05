import numpy as np
import pandas as pd

from src.research.race import pop_expect as pe
from src.research.race import running_map_course as rmc


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(400):
        n = 12
        order = np.argsort(rng.normal(size=n) - np.linspace(1.5, -1.5, n))   # 人気上位ほど着順が良い
        fp = np.empty(n, int); fp[order] = np.arange(1, n + 1)
        rows += [{"race_id": f"r{r}", "horse_id": f"h{r}_{k}", "popularity": k + 1, "finish_position": int(fp[k]), "n_field": n} for k in range(n)]
    return pd.DataFrame(rows)


def test_table_is_monotone_and_favourite_higher():
    t = pe.build_table(_frame())["12"]
    assert len(t) == 12 and t[0] > t[-1]
    assert all(a >= b - 1e-9 for a, b in zip(t, t[1:]))


def test_expect_falls_back_without_popularity():
    t = pe.build_table(_frame())
    e_n = rmc.expected_score(np.array([12, 12]))
    out = pe.expect(t, np.array([12, 12]), np.array([1, np.nan]), e_n)
    assert out[0] == t["12"][0] and out[1] == e_n[1]
    assert (pe.expect(None, np.array([12]), np.array([1]), e_n[:1]) == e_n[:1]).all()


def test_pop_switch_changes_expectation_only_when_table_set():
    n, pop, e_n = np.array([12]), np.array([1.0]), rmc.expected_score(np.array([12]))
    assert (rmc.expected_by_pop(n, pop, e_n) == e_n).all()
    rmc.POP_TABLE = pe.build_table(_frame())
    try:
        assert rmc.expected_by_pop(n, pop, e_n)[0] > e_n[0]
    finally:
        rmc.POP_TABLE = None
