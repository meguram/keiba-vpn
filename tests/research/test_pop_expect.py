import numpy as np
import pytest
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


def test_ability_proxy_axis_loses_importance_with_popularity_expectation():
    """人気に織り込まれた実力と相関する軸（実力の代理）は、頭数だけの期待では寄与が大きく、人気込みの期待では小さくなる。"""
    rng = np.random.default_rng(1)
    rows, Z = [], []
    for r in range(1500):
        n = 12
        ability, own = rng.normal(size=n), rng.normal(size=n)       # own = 市場が知らない、コース固有の効く軸
        z_proxy, z_own = ability + 0.5 * rng.normal(size=n), own
        perf = ability + 0.4 * own + 0.8 * rng.normal(size=n)
        market = ability + 0.3 * rng.normal(size=n)
        pop = np.empty(n, int); pop[np.argsort(-market)] = np.arange(1, n + 1)
        fp = np.empty(n, int); fp[np.argsort(-perf)] = np.arange(1, n + 1)
        rows += [{"race_id": f"r{r}", "horse_id": f"h{r}_{k}", "popularity": int(pop[k]), "finish_position": int(fp[k]), "n_field": n} for k in range(n)]
        Z += list(zip(z_proxy, z_own))
    f, Z = pd.DataFrame(rows), np.array(Z)
    y = rmc.score_of(f["finish_position"].to_numpy())
    n = f["n_field"].to_numpy()
    e_n = rmc.expected_score(n)
    e_pop = pe.expect(pe.build_table(f), n, f["popularity"].to_numpy(), e_n)
    share = lambda r: (lambda d: d / d.sum())(rmc._importance(Z, r)[0])
    s_n, s_pop = share(y - e_n), share(y - e_pop)
    assert s_n[0] > 0.8
    assert s_pop[0] < s_n[0] - 0.3 and s_pop[1] > s_n[1]


def test_course_maps_build_uses_popularity_table_and_restores_switch(monkeypatch):
    """基準の作成は人気込みの期待で行い、表を course_maps に同梱し、作成後に POP_TABLE を元（None）へ戻す。"""
    from src.research.race import running_map as rm

    rng = np.random.default_rng(2)
    fr = _frame()
    ids = fr[["race_id", "horse_id"]].copy()
    ids["date"] = pd.Timestamp("2025-03-01")
    ids["venue"], ids["surface"], ids["distance"], ids["track_condition"] = "東京", "芝", 1600, "良"
    ids["finish_position"] = fr["finish_position"].to_numpy()
    table = ids[["race_id", "horse_id", "date"]].copy()
    for _, col, *_ in rm.AXES:
        table[col] = rng.normal(size=len(table))
    q = {key: list(np.linspace(-3, 3, 1001)) for key, *_ in rm.AXES}
    ptab = pe.build_table(fr)
    monkeypatch.setattr(pe, "default_table", lambda: ptab)
    monkeypatch.setattr(pe, "popularity_frame", lambda years: fr[["race_id", "horse_id", "popularity"]])
    monkeypatch.setattr(rmc, "POP_TABLE", None)
    out = rm._build_course_maps(ids, table, q, [1.0] * len(rm.AXES))
    assert out["pop_table"] == ptab and rmc.POP_TABLE is None
    assert out["courses"]
    monkeypatch.setattr(pe, "default_table", lambda: None)
    with pytest.raises(RuntimeError):
        rm._build_course_maps(ids, table, q, [1.0] * len(rm.AXES))
