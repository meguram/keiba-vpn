import numpy as np

from src.scripts.data import tp73_side as T


def test_ancestor_entries_maternal_and_paternal_sides():
    # 節点: 0=馬, 1=父, 2=母, 3=父父, 4=父母, 5=母父, 6=母母
    sire = np.array([1, 3, 5, -1, -1, -1, -1])
    dam = np.array([2, 4, 6, -1, -1, -1, -1])
    mat = T.ancestor_entries(sire, dam, np.array([2]), "mat")
    assert sorted(zip(mat[1].tolist(), mat[2].tolist())) == [(5, 2)]            # 母系の男の祖先は母父（世代 2）だけ
    pat = T.ancestor_entries(sire, dam, np.array([1]), "pat")
    assert sorted(zip(pat[1].tolist(), pat[2].tolist())) == [(1, 1), (3, 2)]     # 父系は父（世代 1）と父父（世代 2）


def test_ancestor_entries_counts_crosses_separately():
    # 母の父と母母の父が同じ節点 9（クロス）→ 2 回の登場
    sire = np.array([-1, -1, 9, 9, -1, -1, -1, -1, -1, -1])
    dam = np.array([-1, -1, 3, -1, -1, -1, -1, -1, -1, -1])
    H, N, G = T.ancestor_entries(sire, dam, np.array([2]), "mat")
    assert sorted(zip(N.tolist(), G.tolist())) == [(9, 2), (9, 3)]


def test_harville_pairs_sums_and_symmetry():
    q = np.array([0.4, 0.3, 0.2, 0.1])
    P3, P2 = T.harville_pairs(q)
    assert np.allclose(P3, P3.T) and np.allclose(P2, P2.T)
    assert abs(P3.sum() / 2 - 3.0) < 1e-9          # 3 着以内の 3 頭からできる対は 3 つ
    assert abs(P2.sum() / 2 - 1.0) < 1e-9          # 1-2 着の対は 1 つ


def test_bh_monotone_and_bounds():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    q = T.bh(p)
    assert (q >= p).all() and (q <= 1).all() and q[0] <= q[1] <= q[2] <= q[3]


def test_clu_ols_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = 0.5 * x + rng.normal(size=2000)
    g = np.repeat(np.arange(500), 4)
    b, se, _, _ = T.clu_ols(np.c_[np.ones(2000), x], y, g)
    assert abs(b[1] - 0.5) < 0.1 and se[1] > 0
