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
