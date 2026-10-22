import numpy as np
import pandas as pd

from src.scripts.data import tp73b_multi as M


def test_add_flags_uses_discovery_quantile_and_counts():
    n = 100
    Pp = pd.DataFrame({"year": [2023] * n, "d_str": np.linspace(0, 1, n), "d_sty": np.linspace(0, 1, n), "d_gate": np.linspace(0, 1, n), "d_apt": np.linspace(0, 1, n),
                       "both_front": [1.0] * 10 + [0.0] * 90, "same_type": [0.0] * n})
    F, cuts = M.add_flags(Pp)
    assert abs(F["str"].mean() - 0.2) < 0.02 and abs(cuts["str"] - 0.2) < 0.02
    assert F["k"].max() == 5 and F["type"].sum() == 0


def test_soft_nll_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    npair = 6
    races = [{"pos": (0, npair), "um_p": np.arange(npair), "um_lp": np.log(np.full(npair, 1 / npair)), "um_act": 2}]
    F = rng.normal(size=(npair, 3))
    g = rng.normal(size=3) * 0.1
    v, gr = M.soft_nll(g, races, F, "um")
    num = np.zeros(3)
    for i in range(3):
        d = np.zeros(3); d[i] = 1e-6
        num[i] = (M.soft_nll(g + d, races, F, "um")[0] - M.soft_nll(g - d, races, F, "um")[0]) / 2e-6
    assert np.allclose(gr, num, atol=1e-5)


def test_oe_boot_ratio():
    race = np.array(["a", "a", "b", "b"])
    o, e = np.array([1.0, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])
    r = M.oe_boot(race, o, e, np.ones(4, bool), B=50)
    assert abs(r["oe"] - 1.0) < 1e-9 and r["lo"] <= r["oe"] <= r["hi"] + 1e-9
