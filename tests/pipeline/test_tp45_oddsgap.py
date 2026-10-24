import numpy as np
import pandas as pd

from src.scripts.data import tp45_oddsgap as t


def _race(n_races=3, n=5, seed=0):
    rng = np.random.default_rng(seed)
    race = np.repeat([f"R{i}" for i in range(n_races)], n)
    odds = rng.uniform(2, 40, n_races * n)
    return race, odds


def test_gap_zero_when_prediction_equals_final():
    race, odds = _race()
    assert np.allclose(t.gap_g(race, odds, odds), 0.0)


def test_gap_sign_market_bought_more_than_predicted_is_negative():
    race = np.array(["A"] * 3)
    final = np.array([2.0, 5.0, 10.0])
    pred = np.array([4.0, 5.0, 10.0])      # 馬 0 は予測より確定オッズが低い（買われた）
    g = t.gap_g(race, final, pred)
    assert g[0] < 0 and g[1] > 0 and g[2] > 0


def test_gap_is_scale_free_within_race():
    race, odds = _race()
    pred = odds * 1.7
    assert np.allclose(t.gap_g(race, odds, pred), 0.0)    # 予測が一定倍でも正規化で 0


def test_norm_prob_sums_to_one():
    race, odds = _race()
    p = t.norm_prob(race, 1 / odds)
    assert np.allclose(pd.Series(p).groupby(race).sum().to_numpy(), 1.0)


def test_boot_ratio_point_estimate():
    race = np.array(["a", "a", "b", "b"])
    stake = np.array([100.0, 100.0, 100.0, 100.0])
    ret = np.array([0.0, 300.0, 0.0, 100.0])
    r, lo, hi = t.boot_ratio(stake, ret, race, n_boot=200)
    assert abs(r - 1.0) < 1e-9 and lo <= r <= hi


def test_cluster_ols_recovers_slope_and_clusters():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(200), 5)
    x = rng.normal(size=1000)
    y = 0.5 + 2.0 * x + rng.normal(size=1000)
    r = t.cluster_ols(y, x[:, None], groups)
    assert abs(r["coef"][1] - 2.0) < 0.1 and r["clusters"] == 200 and r["se"][1] > 0


def test_reality_check_null_is_not_significant():
    rng = np.random.default_rng(2)
    prof = rng.normal(size=(500, 12))
    rc = t.reality_check(prof, n_boot=300)
    assert rc["p"] > 0.05 and len(rc["t"]) == 12


def test_reality_check_detects_a_real_edge():
    rng = np.random.default_rng(3)
    prof = rng.normal(size=(800, 6))
    prof[:, 2] += 0.4
    assert t.reality_check(prof, n_boot=300)["p"] < 0.05


def test_top_overlap_identical_ranking():
    race = np.repeat(["a", "b"], 6)
    v = np.tile(np.arange(6, dtype=float), 2)
    o = t.top_overlap(race, -v, v)                # a の小さい方 = b の大きい方
    assert o["jaccard_top3"] == 1.0 and o["top1_same"] == 1.0


def test_thresholds_and_strategy_count():
    g = np.linspace(-1, 1, 1001)
    thr = t.thresholds(g)
    assert thr[("low", 0.10)] < thr[("low", 0.30)] < thr[("high", 0.70)] < thr[("high", 0.90)]
    assert len(t.strategies()) == 36


def test_insert_section_is_idempotent():
    base = "<h1>x</h1>\n<h2>6. 取り入れる価値のある手法と実装案</h2>\n<p>後</p>"
    blk = f"{t.START}\n<h2>5b</h2>\n{t.END}"
    once = t.insert_section(base, blk)
    assert once.index(t.START) < once.index("<h2>6.") and once.count(t.START) == 1
    twice = t.insert_section(once, blk.replace("5b", "5b-改"))
    assert twice.count(t.START) == 1 and "5b-改" in twice and "<p>後</p>" in twice
