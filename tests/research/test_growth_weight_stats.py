import numpy as np

from src.research.pedigree import growth_weight_stats as gw


def test_age_and_stage():
    assert gw.age_from_id("2020100123", 2022) == 2
    assert gw.age_from_id("", 2022) is None and gw.age_from_id("abcd1", 2022) is None
    assert gw.stage_of(2, 6) == "2歳前半" and gw.stage_of(2, 11) == "2歳後半"
    assert gw.stage_of(3, 5) == "3歳前半" and gw.stage_of(3, 7) == "3歳後半"
    assert gw.stage_of(4, 1) == "4歳" and gw.stage_of(7, 1) == "5歳以上" and gw.stage_of(None, 1) is None


def test_ols_cluster_recovers_slope_and_cluster_se_is_wider_with_shared_shocks():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(200), 10)
    shock = np.repeat(rng.normal(0, 1, 200), 10)       # 同じクラスタで共有される誤差
    x = rng.normal(size=2000) + np.repeat(rng.normal(0, 1, 200), 10)   # x もクラスタ共通の成分を持つ（誤差とは独立）
    y = 2.0 * x + shock + rng.normal(0, 1, 2000)
    X = np.column_stack([np.ones(2000), x])
    r = gw.ols_cluster(X, y, g)
    assert abs(r["beta"][1] - 2.0) < 0.2 and r["clusters"] == 200
    # クラスタを無視した（各行を別クラスタとした）SE より大きい
    naive = gw.ols_cluster(X, y, np.arange(2000))
    assert r["se"][1] > naive["se"][1]


def test_design_with_dummies_drops_reference():
    X, names = gw.design_with_dummies({"a": np.array([1.0, 2, 3])}, {"s": ["x", "y", "x"]}, {"s": "x"})
    assert names == ["const", "a", "s=y"] and X.shape == (3, 3) and list(X[:, 2]) == [0, 1, 0]


def test_spearman_ties_and_nan():
    assert abs(gw.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(gw.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert np.isnan(gw.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert np.isnan(gw.spearman([1, 2], [1, 2]))


def test_eb_shrink_no_heterogeneity_gives_tau2_zero_and_full_shrink():
    rng = np.random.default_rng(1)
    se = np.full(300, 1.0)
    est = rng.normal(0.5, 1.0, 300)            # 真の個体差なし（ばらつきは誤差だけ）
    r = gw.eb_shrink(est, se)
    assert r["tau2"] < 0.15 and np.nanmax(r["weight"]) < 0.2
    # 個体差ありなら tau2 が立つ
    est2 = rng.normal(0.5, 1.0, 300) + rng.normal(0, 2.0, 300)
    assert gw.eb_shrink(est2, se)["tau2"] > 2.0


def test_slope_se_and_bins():
    x = np.arange(10.0)
    b, se, n = gw.slope_se(x, 3 * x + 1)
    assert abs(b - 3) < 1e-9 and n == 10
    assert np.isnan(gw.slope_se(np.ones(8), np.arange(8.0))[0])
    labs = gw.bin_labels(np.array([-12, -3, 0, 4, 20, np.nan]), [-10, -4, -1, 2, 5, 10])
    assert labs[0] == "≤-11" and labs[2] == "-1〜1" and labs[4] == "≥10" and labs[5] is None


def test_group_slopes_matches_slope_se():
    rng = np.random.default_rng(2)
    gid = np.repeat(np.arange(5), [10, 6, 3, 8, 12])
    x = rng.normal(size=len(gid))
    y = 1.5 * x + rng.normal(size=len(gid))
    r = gw.group_slopes(gid, x, y)
    for g in range(5):
        m = gid == g
        b, se, n = gw.slope_se(x[m], y[m])
        if np.isnan(b):
            assert np.isnan(r["slope"][g])
        else:
            assert abs(r["slope"][g] - b) < 1e-9 and abs(r["se"][g] - se) < 1e-9 and r["n"][g] == n
    assert np.isnan(r["slope"][2])          # n=3 < min_n
