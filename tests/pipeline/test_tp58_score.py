import numpy as np
import pytest

from src.pipeline.features.tp58_score import ScoreTest


def _race_data(seed=0, n_races=60, n_h=8):
    rng = np.random.default_rng(seed)
    races = np.repeat(np.arange(n_races), n_h)
    p = rng.uniform(0.02, 0.4, n_races * n_h)
    y = np.zeros(n_races * n_h)
    for r in range(n_races):
        y[r * n_h + rng.integers(n_h)] = 1
    z = rng.normal(size=n_races * n_h)
    periods = {"a": races < n_races // 2, "b": races >= n_races // 2}
    return races, p, y, z, periods


def _win_ll(beta, races, p, y, z, mask):
    ll = 0.0
    for r in np.unique(races[mask]):
        i = races == r
        s = np.log(p[i] / p[i].sum()) + beta * z[i]
        ll += (y[i] * s).sum() - np.log(np.exp(s).sum())
    return ll


def test_win_score_matches_numeric_derivatives():
    races, p, y, z, periods = _race_data()
    st = ScoreTest(y, p, races, periods, "win")
    g, I, _ = st.gi(z[:, None])
    for i, k in enumerate(["a", "b"]):
        m = periods[k]
        h = 1e-4
        f = lambda b: _win_ll(b, races, p, y, z, m)  # noqa: E731
        g_num = (f(h) - f(-h)) / (2 * h)
        I_num = -(f(h) - 2 * f(0) + f(-h)) / h**2
        assert g[i, 0] == pytest.approx(g_num, rel=2e-3, abs=2e-3)
        assert I[i, 0] == pytest.approx(I_num, rel=2e-3)


def test_binary_score_matches_numeric_derivatives():
    races, p, y, z, periods = _race_data(1)
    st = ScoreTest(y, p, races, periods, "binary")
    g, I, _ = st.gi(z[:, None])
    logit = np.log(p / (1 - p))
    for i, k in enumerate(["a", "b"]):
        m = periods[k]

        def f(b):
            s = logit[m] + b * z[m]
            return (y[m] * s - np.log1p(np.exp(s))).sum()

        h = 1e-4
        assert g[i, 0] == pytest.approx((f(h) - f(-h)) / (2 * h), rel=2e-3, abs=2e-3)
        assert I[i, 0] == pytest.approx(-(f(h) - 2 * f(0) + f(-h)) / h**2, rel=2e-3)


def test_efficient_score_removes_nuisance_direction():
    races, p, y, z, periods = _race_data(2)
    for kind in ("win", "binary"):
        st = ScoreTest(y, p, races, periods, kind)
        raw = st.efficient_z(z, np.zeros((len(z), 0)))
        assert raw["a"] == pytest.approx(st.z(z[:, None])["a"][0], rel=1e-4)
        # z 自身を nuisance にすれば有効情報はゼロ（0 を返す）
        assert abs(st.efficient_z(z, z[:, None])["a"]) < 1e-3


def test_non_contiguous_races_rejected():
    with pytest.raises(ValueError):
        ScoreTest(np.zeros(4), np.full(4, 0.2), np.array([0, 1, 0, 1]), {"a": np.ones(4, bool)}, "win")


def test_race_constant_column_gives_zero_for_win():
    races, p, y, z, periods = _race_data(3)
    zc = np.repeat(np.random.default_rng(0).normal(size=len(np.unique(races))), 8) * 1000.0
    st = ScoreTest(y, p, races, periods, "win")
    assert st.z(zc[:, None])["a"][0] == 0.0
