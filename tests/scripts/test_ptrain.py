import numpy as np
import pandas as pd

from src.scripts.data import ptrain as P


def _toy(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    race = np.repeat(np.arange(n // 10), 10).astype(str)
    year = np.repeat([2022, 2023, 2024], n // 3)
    y = (rng.random(n) < 0.1).astype(float)
    lg = np.log(0.1 / 0.9)
    good = lg + (y - 0.1) * 0.8 + rng.normal(0, 1, n)
    bad = lg + rng.normal(0, 1, n)
    key = pd.DataFrame({"race_id": race, "horse_id": np.arange(n).astype(str), "year": year})
    p = lambda z: 1 / (1 + np.exp(-z))
    pp = {"a": {tg: p(good) for tg in P.TARGETS}, "b": {tg: p(bad) for tg in P.TARGETS}}
    return key, pp, {tg: y for tg in P.TARGETS}


def test_stack_forward_prefers_informative_learner():
    key, pp, y = _toy()
    out = P.stack_forward(key, pp, y, ["a", "b"])
    yr = key["year"].to_numpy() == 2024
    ll_stack = P._ll(y["win"][yr], out["win"][yr])
    ll_bad = P._ll(y["win"][yr], pp["b"]["win"][yr])
    assert ll_stack < ll_bad


def test_stack_first_year_uses_equal_weights():
    key, pp, y = _toy()
    out = P.stack_forward(key, pp, y, ["a", "b"])
    yr = key["year"].to_numpy() == 2022
    eq = P._norm(key["race_id"].to_numpy(), P._sigmoid((P._logit(pp["a"]["win"]) + P._logit(pp["b"]["win"])) / 2), 1.0)[yr]
    # 最初の年は結合の係数を使わず等重み
    assert np.allclose(out["win"][yr], eq)


def test_fit_gate_shrinks_regime_deviation_with_large_lambda():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.15).astype(float)
    Z = np.column_stack([np.log(0.15 / 0.85) + (y - 0.15) * k + rng.normal(0, 1, n) for k in (0.8, 0.3)])
    reg = np.where(rng.random(n) < 0.5, "a", "b")
    tr = np.arange(n) < 3000
    _, beta_small, _ = P.fit_gate(Z, y, reg, tr, 1.0)
    _, beta_large, _ = P.fit_gate(Z, y, reg, tr, 1e6)
    k = Z.shape[1]
    assert np.abs(beta_large[1 + k:]).max() < np.abs(beta_small[1 + k:]).max() + 1e-9
    assert np.abs(beta_large[1 + k:]).max() < 1e-3


def test_persona_cols_core_plus_own_prefix():
    cfg = {"core": {"exact": ["distance"], "prefix": ["fe_"]}, "personas": {"x": {"prefix": ["o_"], "exact": ["z"]}}}
    cols = ["distance", "fe_sin1", "o_win", "z", "zz", "pd_a"]
    assert P.persona_cols(cols, "x", cfg) == ["distance", "fe_sin1", "o_win", "z"]
