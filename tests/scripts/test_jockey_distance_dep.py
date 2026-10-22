import numpy as np
import pandas as pd

from src.scripts.data import jockey_distance_dep as J


def test_band_of():
    d = pd.Series([1000, 1400, 1401, 1800, 2000, 2400, 3000])
    assert J.band_of(d).tolist() == ["b1", "b1", "b2", "b2", "b3", "b4", "b5"]


def test_split_cov_recovers_jockey_variance():
    rng = np.random.default_rng(0)
    nj, nh, rides = 40, 3000, 6
    jt = rng.normal(0, 0.5, nj)                     # 騎手の真の効果（分散 0.25）
    ht = rng.normal(0, 1.0, nh)
    h = np.repeat(np.arange(nh), rides)
    # 馬は 3 人の騎手に乗られる（つながりを作る）
    j = np.concatenate([rng.choice(nj, rides, replace=True) for _ in range(nh)])
    y = ht[h] + jt[j] + rng.normal(0, 1.0, len(h))
    bA, bB, W = J.split_fits(h, j, y, nj, seed=1)
    cov = J.cov_boot(bA, bB, W, np.ones((1, nj)))[0]
    assert 0.12 < cov < 0.35                         # 真値 0.25 に近い（ノイズが独立なので不偏、弱い縮約で少し小さい）


def test_split_cov_is_near_zero_without_jockey_effect():
    rng = np.random.default_rng(1)
    nj, nh, rides = 40, 3000, 6
    ht = rng.normal(0, 1.0, nh)
    h = np.repeat(np.arange(nh), rides)
    j = np.concatenate([rng.choice(nj, rides, replace=True) for _ in range(nh)])
    y = ht[h] + rng.normal(0, 1.0, len(h))
    bA, bB, W = J.split_fits(h, j, y, nj, seed=2)
    assert abs(J.cov_boot(bA, bB, W, np.ones((1, nj)))[0]) < 0.02


def test_slope_and_pval():
    x = np.arange(4, dtype=float)
    assert np.allclose(J.slope(x, np.array([[1.0, 2.0, 3.0, 4.0]])), 1.0)
    assert J.pval2(np.array([1.0, 2.0, 3.0])) <= 0.0 + 1e-9
    assert J.pval2(np.array([-1.0, 1.0])) == 1.0
