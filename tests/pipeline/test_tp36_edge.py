import numpy as np
import pandas as pd

from src.scripts.data.tp36_census import boot_roi
from src.scripts.data.tp36_edge import blend, boot, fit_benter


def _races(n=200, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n):
        k = 10
        q = rng.dirichlet(np.ones(k) * 2)
        w = int(rng.choice(k, p=q))
        for h in range(k):
            rows.append({"race_id": f"r{r}", "p_win": q[h], "q_win": q[h], "win": float(h == w)})
    return pd.DataFrame(rows)


def test_blend_sums_to_one_per_race():
    d = _races()
    p = blend(d, 0.4, 0.8)
    assert np.allclose(p.groupby(d["race_id"]).sum(), 1.0)


def test_fit_benter_recovers_pure_market_when_model_equals_market():
    d = _races(400)
    a, b = fit_benter(d)
    assert abs((a + b) - 1.0) < 0.25


def test_bootstrap_roi_interval_contains_point_and_is_deterministic():
    g = pd.DataFrame({"race_id": range(300), "stake": 100.0, "ret": np.where(np.arange(300) % 5 == 0, 450.0, 0.0)})
    roi, lo, hi, n = boot_roi(g)
    assert lo <= roi <= hi and n == 300
    assert boot_roi(g)[1:3] == (lo, hi)
    roi2, lo2, hi2, n2, hit = boot(g)
    assert abs(roi - roi2) < 1e-12 and abs(hit - 0.2) < 1e-9


def test_exotic_pair_probabilities_sum_constraints():
    from src.scripts.data.tp36_exotic import pair_probs

    w = np.array([0.4, 0.3, 0.2, 0.1])
    r = pair_probs(w, np.array([0.6, 0.5, 0.3, 0.2]), np.array([0.8, 0.8, 0.7, 0.5]))
    assert abs(r["umaren"].sum() / 2 - 1.0) < 1e-9
    assert abs(r["wide"].sum() / 2 - 3.0) < 1e-9
    assert abs(r["umatan"].sum() - 1.0) < 1e-9


def test_reality_check_null_and_signal():
    from src.scripts.data.tp36_rc import reality_check

    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, size=(3000, 20))
    assert reality_check(null, n_boot=400)["p_value"] > 0.05            # 利益ゼロ（帰無）では最良でも有意にならない
    sig = null.copy()
    sig[:, 3] += 0.2
    r = reality_check(sig, n_boot=400)
    assert r["best"] == 3 and r["p_value"] < 0.05
