"""pair-bets 段 1 の算数の部品（植え方・組の確率の和）。"""
import numpy as np

from src.research.pair_bets.arith import MODELS, pair_tables, plant


def test_plant_ratio_after_normalization():
    q = np.array([0.4, 0.3, 0.2, 0.1])
    p = plant(q, [1], 1.28)
    assert abs(p.sum() - 1) < 1e-12
    assert abs(p[1] / q[1] - 1.28) < 1e-12
    p2 = plant(q, [1, 3], 1.15)
    assert abs(p2.sum() - 1) < 1e-12
    assert np.allclose(p2[[1, 3]] / q[[1, 3]], 1.15)


def test_pair_tables_sum_and_symmetry():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(9))
    for lam in MODELS.values():
        u, w = pair_tables(q, lam)
        assert np.allclose(u, u.T) and np.allclose(w, w.T)
        assert abs(np.triu(u, 1).sum() - 1) < 1e-9          # 馬連は 1 組が当たる
        assert abs(np.triu(w, 1).sum() - 3) < 1e-9          # ワイドは 3 組が当たる


def test_pair_tables_match_brute_force_harville():
    import itertools
    q = np.array([0.33, 0.25, 0.2, 0.14, 0.08])
    u, w = pair_tables(q, (1.0, 1.0, 1.0))
    U = np.zeros((5, 5))
    W = np.zeros((5, 5))
    for i, j, k in itertools.permutations(range(5), 3):
        pr = q[i] * q[j] / (1 - q[i]) * q[k] / (1 - q[i] - q[j])
        U[i, j] += pr
        U[j, i] += pr
        for a, b in ((i, j), (i, k), (j, k)):
            W[a, b] += pr
            W[b, a] += pr
    assert np.allclose(u, U) and np.allclose(w, W)


def test_no_edge_gives_ratio_one():
    q = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    for lam in MODELS.values():
        u0, w0 = pair_tables(q, lam)
        u1, w1 = pair_tables(plant(q, [0], 1.0), lam)
        assert np.allclose(u0, u1) and np.allclose(w0, w1)
