from itertools import permutations

import numpy as np
import pandas as pd

from src.pipeline.models import zoo


def _brute(p, g2, g3):
    """全順列を数え上げる（2 着以降は p^γ に比例）。"""
    n = len(p)
    q2, q3 = p ** g2, p ** g3
    t2, t3 = np.zeros(n), np.zeros(n)
    for a in range(n):
        t2[a] += p[a]
        t3[a] += p[a]
        for b in range(n):
            if b == a:
                continue
            pb = p[a] * q2[b] / (q2.sum() - q2[a])
            t2[b] += pb
            pb3 = p[a] * q3[b] / (q3.sum() - q3[a])
            t3[b] += pb3
            for c in range(n):
                if c in (a, b):
                    continue
                t3[c] += pb3 * q3[c] / (q3.sum() - q3[a] - q3[b])
    return t2, t3


def test_harville_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(7))
    race = np.array(["a"] * 7)
    for g2, g3 in [(1.0, 1.0), (0.8, 0.7)]:
        t2, t3 = _brute(p, g2, g3)
        h2, h3 = zoo.harville_topk(race, p, g2, g3)
        assert np.allclose(h2, t2, atol=1e-9) and np.allclose(h3, t3, atol=1e-9)
        assert abs(h2.sum() - 2) < 1e-9 and abs(h3.sum() - 3) < 1e-9


def test_small_field_top3_is_one():
    race = np.array(["a"] * 3)
    h2, h3 = zoo.harville_topk(race, np.array([0.5, 0.3, 0.2]))
    assert np.allclose(h3, 1.0) and abs(h2.sum() - 2) < 1e-9


def test_finalize_sums_and_uses_harville_when_missing():
    race = np.array(["a"] * 5 + ["b"] * 6)
    out = zoo.finalize(race, {"win": np.random.default_rng(1).uniform(0.05, 0.4, 11)})
    for r in ("a", "b"):
        m = race == r
        assert abs(out["win"][m].sum() - 1) < 1e-6 and abs(out["top2"][m].sum() - 2) < 1e-6 and abs(out["top3"][m].sum() - 3) < 1e-6


def test_nn_and_linear_learn_a_simple_signal_on_cpu():
    rng = np.random.default_rng(0)
    n_r, n_h = 120, 8
    race = np.repeat([f"r{i:04d}" for i in range(n_r)], n_h)
    X = pd.DataFrame({"a": rng.normal(size=n_r * n_h), "b": rng.normal(size=n_r * n_h), "c": pd.Categorical(rng.choice(["x", "y"], n_r * n_h))})
    finish = np.concatenate([np.argsort(np.argsort(-(X["a"].to_numpy()[i * n_h:(i + 1) * n_h] + rng.normal(0, 0.4, n_h)))) + 1 for i in range(n_r)])
    y = {"win": (finish == 1).astype(int), "top2": (finish <= 2).astype(int), "top3": (finish <= 3).astype(int), "finish": finish}
    for rec in ("linear", "nn"):
        o = zoo.fit_predict(rec, X, y, race, X, race, threads=2, params={"epochs": 40, "hidden": (16,), "lr": 2e-2, "batch_races": 32})
        p = zoo.finalize(race, o)
        from src.pipeline.models import win_models as wm
        assert wm.top1_hit_rate(race, p["win"], y["win"].astype(float)) > 0.3, rec
