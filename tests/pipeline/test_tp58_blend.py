import numpy as np
import pandas as pd

from src.scripts.data.tp58_nn_eval import _blend_frame


def _frames():
    rng = np.random.default_rng(0)
    n_r, n_h = 5, 6
    ids = pd.DataFrame({"race_id": np.repeat([f"r{i}" for i in range(n_r)], n_h), "horse_id": [f"h{j}" for j in range(n_r * n_h)], "year": 2022})

    def probs(seed):
        r = np.random.default_rng(seed)
        w = r.uniform(0.1, 1, n_r * n_h)
        w = w / pd.Series(w).groupby(ids["race_id"].to_numpy()).transform("sum").to_numpy()
        return w, r.uniform(0.05, 0.9, n_r * n_h), r.uniform(0.05, 0.9, n_r * n_h)

    b, n = ids.copy(), ids.copy()
    b["p_win"], b["p_top2"], b["p_top3"] = probs(1)
    n["p_win"], n["p_top2"], n["p_top3"] = probs(2)
    return b, n[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]


def test_blend_endpoints_and_normalisation():
    b, n = _frames()
    out0 = _blend_frame(b, n, {"win": 0.0, "top2": 0.0, "top3": 0.0})
    np.testing.assert_allclose(out0["p_win"], b["p_win"], atol=1e-9)
    np.testing.assert_allclose(out0["p_top2"], b["p_top2"], atol=1e-9)
    out1 = _blend_frame(b, n, {"win": 1.0, "top2": 1.0, "top3": 1.0})
    np.testing.assert_allclose(out1["p_win"], n["p_win"], atol=1e-9)
    np.testing.assert_allclose(out1["p_top3"], n["p_top3"], atol=1e-9)
    mid = _blend_frame(b, n, {"win": 0.4, "top2": 0.1, "top3": 0.2})
    np.testing.assert_allclose(mid.groupby("race_id")["p_win"].sum().to_numpy(), 1.0, atol=1e-9)
