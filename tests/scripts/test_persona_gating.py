import numpy as np
from scipy.optimize import check_grad

from src.scripts.data import persona_gating as pg


def _toy(seed=0, n_races=40, k=6, d=3, c=3):
    rng = np.random.default_rng(seed)
    ridx = np.repeat(np.arange(n_races), k)
    X = rng.normal(size=(n_races * k, d))
    win = np.zeros(n_races * k, dtype=int)
    win[np.arange(n_races) * k + rng.integers(0, k, n_races)] = 1
    cell = np.repeat(rng.integers(0, c, n_races), k)
    return X, ridx, win, cell


def test_race_logprob_normalizes_within_race():
    ridx = np.repeat(np.arange(3), 4)
    p = np.random.default_rng(1).uniform(0.01, 0.5, 12)
    lp = pg._race_logprob(p, ridx)
    for r in range(3):
        assert abs(np.exp(lp[ridx == r]).sum() - 1.0) < 1e-9


def test_gradient_matches_numeric():
    X, ridx, win, cell = _toy()
    m = pg.CL(X, ridx, win, cell, lam=2.0)
    th = np.random.default_rng(2).normal(size=m.d + m.C * m.d) * 0.3
    err = check_grad(lambda t: m.nll_grad(t)[0], lambda t: m.nll_grad(t)[1], th)
    assert err < 1e-5


def test_strong_shrinkage_collapses_to_global_weights():
    X, ridx, win, cell = _toy(seed=3)
    m = pg.CL(X, ridx, win, cell, lam=1e8)
    th = m.fit()
    assert np.abs(th[m.d:]).max() < 1e-4
    X2, ridx2, win2, _ = _toy(seed=4)
    g = pg.CL(X, ridx, win, None, 0.0).fit()
    assert np.allclose(th[: m.d], g, atol=2e-2)
