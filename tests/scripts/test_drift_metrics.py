"""ドリフト診断・監視の純関数のテスト（合成データ）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M


def test_psi_same_distribution_is_small_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert M.psi_numeric(a, rng.normal(0, 1, 5000)) < 0.05
    assert M.psi_numeric(a, rng.normal(1.0, 1, 5000)) > 0.25
    assert M.ks_stat(a, a) == 0.0


def test_psi_counts_missing_as_its_own_bin():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 4000)
    b = a.copy()
    b[:1500] = np.nan
    assert M.psi_numeric(a, b) > 0.25


def test_feature_shift_handles_categorical_and_missing_columns():
    ref = pd.DataFrame({"x": np.arange(100.0), "s": ["a", "b"] * 50})
    cur = pd.DataFrame({"x": np.arange(100.0) + 50, "s": ["a"] * 100})
    out = M.feature_shift(ref, cur, ["x", "s", "nope"])
    assert set(out["feature"]) == {"x", "s"}
    assert (out["psi"] > 0.1).all()


def test_calibration_and_ece_zero_for_calibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.6, 40000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert M.ece(p, y) < 0.01
    a, b = M.calib_slope(p, y)
    assert abs(b - 1) < 0.1 and abs(a) < 0.1
    assert M.ece(np.clip(p * 2, 0, 1), y) > 0.05


def test_gap_with_ci_and_decompose_sum_to_total():
    rng = np.random.default_rng(3)
    rows = []
    for yr, shift in ((2024, 0.0), (2026, 0.4)):
        for r in range(300):
            n = 10
            pm = rng.dirichlet(np.ones(n) * 2)
            logit = np.log(pm) + rng.normal(0, shift, n)
            pw = np.exp(logit) / np.exp(logit).sum()
            w = rng.choice(n, p=pm)
            for i in range(n):
                rows.append({"race_id": f"{yr}{r:04d}", "year": yr, "p_win": pw[i], "p_mkt": pm[i], "y_win": float(i == w), "grp": "A" if i < 5 else "B"})
    df = pd.DataFrame(rows)
    base, cur = df[df.year == 2024], df[df.year == 2026]
    cb, cc = M.gap_with_ci(base), M.gap_with_ci(cur)
    assert cc["gap"] > cb["gap"]
    assert cc["lo"] <= cc["gap"] <= cc["hi"]
    dec = M.decompose_gap(base, cur, "grp")
    assert abs(dec["mix"] + dec["within"] - dec["total"]) < 1e-9
    assert abs(dec["total"] - (cc["gap"] - cb["gap"])) < 1e-9


def test_control_limits_run_length_and_consecutive():
    v = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95])
    cl = M.control_limits(v, 3.0)
    assert cl["lcl"] < cl["mean"] < cl["ucl"]
    assert M.run_length([False, True, True]) == 2
    assert M.run_length([True, False]) == 0
    assert M.consecutive_over([False, True, True, True], 3)
    assert not M.consecutive_over([True, True], 3)


def test_cusum_detects_sustained_shift_but_not_noise():
    rng = np.random.default_rng(4)
    calm = rng.normal(0, 1, 60).tolist()
    assert not M.cusum(calm, 0.0, 1.0)["alarm_now"]
    shifted = calm[:20] + (rng.normal(1.5, 1, 20)).tolist()
    r = M.cusum(shifted, 0.0, 1.0)
    assert r["first_alarm_index"] is not None and r["first_alarm_index"] >= 20
    down = M.cusum((-np.array(shifted)).tolist(), 0.0, 1.0, direction="down")
    assert down["first_alarm_index"] == r["first_alarm_index"]


def test_iso_week_is_monday():
    s = pd.Series(pd.to_datetime(["2026-10-17", "2026-10-18", "2026-10-19"]))
    assert M.iso_week_of(s).tolist() == ["2026-10-12", "2026-10-12", "2026-10-19"]


def test_new_helpers_brier_rankcorr_fdr_ewma_edges():
    assert abs(M.brier(np.array([1, 0]), np.array([0.8, 0.2])) - 0.04) < 1e-12
    q = M.bh_fdr([0.001, 0.02, 0.04, 0.5])
    assert q[0] < q[1] <= q[2] < q[3] and all(0 <= x <= 1 for x in q)
    ew = M.ewma([0.0] * 10 + [3.0] * 10, 0.2, 0.0, 1.0)
    assert not any(ew["flags"][:10]) and any(ew["flags"][10:])
    x = np.random.default_rng(0).normal(size=2000)
    e = M.quantile_edges(x, 10)
    assert M.psi_with_edges(e, x) < 0.02 and M.psi_with_edges(e, x + 1.5) > 0.25
    df = pd.DataFrame({"race_id": ["a"] * 4 + ["b"] * 4, "p": [0.4, 0.3, 0.2, 0.1] * 2, "finish_position": [1, 2, 3, 4, 4, 3, 2, 1]})
    rc = M.race_rank_corr_by_race(df, "p")
    assert abs(rc["a"] - 1.0) < 1e-9 and abs(rc["b"] + 1.0) < 1e-9
    se = M.cluster_gap_se(pd.DataFrame({"race_id": ["a", "a", "b", "b"], "d": [0.1, 0.1, 0.3, 0.3]}))
    assert abs(se[0] - 0.2) < 1e-12 and se[1] > 0
    m = pd.DataFrame({"race_id": ["a"] * 4, "p_mkt": [0.4, 0.3, 0.2, 0.1]})
    p2, p3 = M.market_top_probs(m)
    assert abs(p2.sum() - 2.0) < 0.05 and abs(p3.sum() - 3.0) < 0.1
