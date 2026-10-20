import numpy as np
import pytest

from src.scripts.data import audit_ev_calibration as au


def test_bin_table_assigns_bins_and_reports_roi():
    ev = np.array([0.5, 0.9, 1.1, 1.3, 1.3, 2.5])
    ret = np.array([0.4, 0.8, 0.9, 1.0, 0.0, 3.0])
    race = np.array(["a", "a", "b", "b", "c", "c"])
    t = au.bin_table(ev, ret, race, n_boot=10)
    by = {r["bin"]: r for r in t}
    assert by["[0,0.8)"]["n"] == 1 and by["[0,0.8)"]["roi"] == pytest.approx(0.4)
    assert by["[1.2,1.5)"]["n"] == 2 and by["[1.2,1.5)"]["roi"] == pytest.approx(0.5)
    assert by["[2,3)"]["n"] == 1 and sum(r["n"] for r in t) == 6


def test_fit_shrink_k_recovers_slope_and_clips():
    rng = np.random.default_rng(0)
    ev = 0.8 + rng.normal(0, 1.0, 5000)
    ret = 0.8 + 0.2 * (ev - 0.8) + rng.normal(0, 0.3, 5000)
    assert au.fit_shrink_k(ev, ret, 0.8) == pytest.approx(0.2, abs=0.05)
    assert au.fit_shrink_k(ev, 0.8 - (ev - 0.8), 0.8) == 0.0                  # 逆向きは 0 に丸める
    assert au.fit_shrink_k(ev[:10], ret[:10], 0.8) is None


def test_identity_summary_and_labels():
    s = au.identity_summary(np.array([1.25, 1.26, 1.24]), 1.25)
    assert s["n"] == 3 and s["within_tol"] == 1.0 and s["ratio_to_expected"] == pytest.approx(1.0, abs=0.01)
    assert au.identity_summary(np.array([]), 1.0) == {"n": 0}
    assert len(au.ev_bin_labels()) == len(au.EV_BINS) - 1


def test_market_top2_probs_sum_to_two():
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert au.market_top2_probs(q, 0.9).sum() == pytest.approx(2.0, abs=1e-6)
