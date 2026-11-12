"""tp-83: 出走馬の能力分布から作る同時確率・差/形の特徴量の検査（確率の和・対称性・先読みなし）。"""

import numpy as np
import pandas as pd

from src.pipeline.features import tp83_fieldsim as fsim
from tests.pipeline.test_tp42_racelevel import _synth

CONSTS = {"R2": 25.0, "tau2": 40.0, "kappa": 2.0, "m_all": 45.0, "mu0": 40.0, "sd0": 9.0}


def test_win_probabilities_sum_to_one_and_follow_mu():
    rng = np.random.default_rng(0)
    mu = np.array([60.0, 55.0, 50.0, 50.0, 40.0])
    sd = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    b = fsim._race_block(mu, sd, np.full(5, 5.0), rng)
    assert abs(b["p_win"].sum() - 1.0) < 1e-9
    assert b["p_win"][0] > b["p_win"][1] > b["p_win"][2] > b["p_win"][4]
    assert abs(b["p_win"][2] - b["p_win"][3]) < 1e-9, "同じ分布の 2 頭は同じ確率"
    assert 2.7 < b["p_top3"].sum() < 3.3


def test_two_horse_win_probability_matches_closed_form():
    from scipy.special import ndtr

    mu, sd = np.array([55.0, 50.0]), np.array([4.0, 6.0])
    b = fsim._race_block(mu, sd, np.array([5.0, 5.0]), np.random.default_rng(0))
    assert abs(b["p_win"][0] - ndtr(5.0 / np.hypot(4.0, 6.0))) < 2e-3


def test_larger_uncertainty_pulls_a_favourite_down():
    mu = np.array([60.0, 50.0, 50.0, 50.0])
    p_small = fsim._race_block(mu, np.full(4, 3.0), np.full(4, 5.0), np.random.default_rng(0))["p_win"][0]
    p_big = fsim._race_block(mu, np.full(4, 12.0), np.full(4, 5.0), np.random.default_rng(0))["p_win"][0]
    assert p_small > p_big


def test_no_lookahead_features_ignore_results_on_or_after_the_race_day():
    d = _synth(3)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    base = fsim.build(d, CONSTS)
    cut = float(np.sort(d["dnum"].unique())[len(d["dnum"].unique()) // 2])
    d2 = d.copy()
    late = d2["dnum"] >= cut
    d2.loc[late, "a"] = np.random.default_rng(5).normal(80, 30, late.sum())
    alt = fsim.build(d2, CONSTS)
    keep = (d["dnum"] <= cut).to_numpy()
    a, b = base.loc[keep].to_numpy(float), alt.loc[keep].to_numpy(float)
    both = np.isnan(a) & np.isnan(b)
    assert np.allclose(np.where(both, 0, a), np.where(both, 0, b), atol=1e-9, rtol=0)


def test_debut_horse_uses_prior_and_widest_sd():
    d = _synth(4)
    mu, sd, n = fsim.ability(d, CONSTS)
    first = n == 0
    assert first.any()
    assert np.allclose(mu[first], CONSTS["mu0"]) and np.allclose(sd[first], CONSTS["sd0"])
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))


def test_mean_is_shrunk_toward_population_for_short_histories():
    d = _synth(5)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    mu, sd, n = fsim.ability(d, CONSTS)
    raw = fsim._stats(fsim.rl.lags_m(d, "a", fsim.K, 0), fsim._weights(d, False))["E"]
    one = (n == 1)
    assert one.any()
    assert np.all(np.abs(mu[one] - CONSTS["m_all"]) < np.abs(raw[one] - CONSTS["m_all"]) + 1e-9)


def test_huber_downweights_an_outlier_run_in_mu_and_sd():
    from tests.pipeline.test_tp42_racelevel import _synth

    d = _synth(6)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    d = d.sort_values(["horse_id", "dnum"]).reset_index(drop=True)
    d["a"] = 45.0
    last = d.groupby("horse_id").cumcount(ascending=False) == 1       # 各馬の「今回の1つ前」だけ外れ値
    d.loc[last, "a"] = 95.0
    mu0, sd0, n = fsim.ability(d, CONSTS)
    mu1, sd1, _ = fsim.ability(d, {**CONSTS, "huber_k": 1.5})
    m = n >= 3
    assert m.any()
    assert np.all(mu1[m] <= mu0[m] + 1e-9) and np.any(mu1[m] < mu0[m] - 0.5)
    assert np.all(sd1[m] <= sd0[m] + 1e-9)


def test_huber_off_matches_plain_shrunk_mean_and_no_change_without_outliers():
    d = _synth(5)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    d["a"] = 45.0
    mu0, _, _ = fsim.ability(d, CONSTS)
    mu1, _, _ = fsim.ability(d, {**CONSTS, "huber_k": 1.5})
    assert np.allclose(mu0, mu1)


def test_t_distribution_sums_to_one_and_gives_far_longshots_more_chance():
    mu = np.array([60.0, 50.0, 50.0, 38.0])
    sd = np.full(4, 4.0)
    nn = np.full(4, 5.0)
    bn = fsim._race_block(mu, sd, nn, np.random.default_rng(0))
    bt = fsim._race_block(mu, sd, nn, np.random.default_rng(0), df=4.0)
    assert abs(bt["p_win"].sum() - 1.0) < 1e-9
    assert bt["p_win"][0] > bt["p_win"][1] > bt["p_win"][3]
    assert bt["p_win"][3] > bn["p_win"][3], "裾が重いぶん、大きく離れた馬の一発が増える"
    assert 2.7 < bt["p_top3"].sum() < 3.3


def test_sd_by_n_multiplier_is_applied_and_fit_uses_2021_only():
    d = _synth(6)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    m = fsim.fit_sd_by_n(d, CONSTS, min_rows=1)
    assert len(m) == fsim.K + 1 and all(np.isfinite(m))
    _, sd0, n = fsim.ability(d, CONSTS)
    _, sd1, _ = fsim.ability(d, {**CONSTS, "sd_n_mult": [1.0, 2.0, 1, 1, 1, 1]})
    k = n == 1
    assert k.any() and np.allclose(sd1[k], 2.0 * sd0[k])
    d2 = d.copy()
    d2["date"] = d2["date"] + pd.Timedelta(days=3650)
    assert fsim.fit_sd_by_n(d2, CONSTS, min_rows=1) == [1.0] * (fsim.K + 1)
