import numpy as np

from src.scripts.data.tp44_pools import boot_roi, race_probs


def test_race_probs_are_proper_distributions():
    q = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    pr = race_probs(q, 1.1, 0.9, 0.8)
    assert abs(pr["top3"].sum() - 3.0) < 1e-9
    assert abs(pr["umatan"].sum() - 1.0) < 1e-9
    assert abs(np.triu(pr["umaren"], 1).sum() - 1.0) < 1e-9
    assert abs(np.triu(pr["wide"], 1).sum() - 3.0) < 1e-9          # 上位 3 頭から 2 頭を選ぶ組合せは 3 通り
    assert np.allclose(pr["umaren"], pr["umaren"].T) and np.allclose(pr["wide"], pr["wide"].T)
    assert np.all(np.diag(pr["umatan"]) == 0)


def test_gamma_one_matches_plain_harville_for_win_prob():
    q = np.array([0.5, 0.3, 0.2])
    pr = race_probs(q, 1.0, 1.0, 1.0)
    assert abs(pr["umatan"][0, 1] - 0.5 * 0.3 / 0.5) < 1e-12      # P(1着0)·P(2着1|1着0)=0.5·0.3/(0.3+0.2)


def test_boot_roi_handles_no_bets():
    roi, lo, hi = boot_roi(np.zeros(5), np.zeros(5))
    assert roi != roi and lo != lo and hi != hi
