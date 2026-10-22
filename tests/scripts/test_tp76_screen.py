"""tp-76: スクリーニングの統計部品（ロジスティック、レース単位の標準誤差、経験ベイズ、シャドウの並べ替え）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import tp76_screen as S


def test_fit_logit_recovers_coefficient():
    rng = np.random.default_rng(0)
    n = 40000
    x = rng.normal(size=n)
    off = rng.normal(-1.0, 0.5, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(off + 0.3 + 0.5 * x)))).astype(float)
    X = np.column_stack([np.ones(n), x])
    b = S.fit_logit(X, y, off, 1e-3)
    assert abs(b[0] - 0.3) < 0.05 and abs(b[1] - 0.5) < 0.05


def test_ratio_stats_cluster_se_larger_than_naive_when_races_correlate():
    rng = np.random.default_rng(1)
    race = np.repeat(np.arange(400), 10)
    shock = np.repeat(rng.normal(0, 1, 400), 10)
    d = shock + rng.normal(0, 0.1, race.size)
    m, se, nr = S.ratio_stats(d, race, np.ones(race.size, bool))
    assert nr == 400 and se > d.std() / np.sqrt(d.size) * 2


def test_eb_posterior_shrinks_and_orders():
    d = np.array([-3.0, -1.0, 0.0, 1.0]) * 1e-5
    se = np.full(4, 2e-5)
    post, mu, tau2 = S.eb_posterior(d, se, d, se)
    assert post[0] > post[1] > post[2] > post[3]


def test_shadow_permutations_stay_within_date_or_race():
    class D:  # noqa: D401
        pass
    d = D()
    d.n = 60
    d.date = np.repeat(np.arange(6), 10)
    d.race = np.repeat(np.arange(12), 5)
    perms = S.make_perms(d, np.random.default_rng(0))
    assert len(perms) == 3
    assert all((d.date[p] == d.date).all() for p in (perms[0], perms[2]))
    assert (d.race[perms[1]] == d.race).all() and (perms[1] != np.arange(60)).all()


def test_unit_definitions_are_consistent():
    cols = {c for v in S.UNITS.values() for c in v[0]}
    assert len(S.UNITS) == 27 and len(cols) == 32
    assert all(v[2] in ("低", "中", "高") for v in S.UNITS.values())
