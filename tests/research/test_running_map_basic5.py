"""基本 5 軸の地図部品の検算（合成データ）。距離・近い馬・配置・血統側の距離分布。"""
import numpy as np
import pandas as pd

from src.research.race import running_map_basic5 as b5


def _Z(n: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 5))


def test_pair_dist_is_symmetric_and_rms_of_shared_axes() -> None:
    Z = np.array([[0.0, 0, 0, 0, 0], [3.0, 4.0, np.nan, 0, 0], [1.0, np.nan, np.nan, np.nan, 0]])
    D = b5.pair_dist(Z)
    assert np.allclose(D, D.T, equal_nan=True) and np.allclose(np.diag(D), 0)
    assert abs(D[0, 1] - np.sqrt((9 + 16 + 0 + 0) / 4)) < 1e-12        # 共通の 4 軸の二乗平均の平方根
    assert np.isnan(D[0, 2]) and np.isnan(D[1, 2])                     # 共通が 3 軸未満は測らない


def test_similar_returns_k_nearest_in_order() -> None:
    Z = _Z()
    D = b5.pair_dist(Z)
    s = b5.similar(D, 5)
    assert all(len(x) == 5 for x in s)
    for i, row in enumerate(s):
        ds = [d for _, d in row]
        assert ds == sorted(ds) and i not in [j for j, _ in row]
        assert abs(ds[0] - np.nanmin(np.where(np.eye(len(D), dtype=bool), np.nan, D)[i])) < 1e-12


def test_axis_diffs_none_for_missing() -> None:
    assert b5.axis_diffs(np.array([0.0, np.nan, 1.0]), np.array([0.5, 1.0, np.nan])) == [0.5, None, None]


def test_radial_keeps_true_distance_from_center() -> None:
    Z = _Z()
    D = b5.pair_dist(Z)
    Y = b5.layout_pca(Z)
    R = b5.radial(D[7], Y, 7)
    assert np.allclose(np.hypot(R[:, 0], R[:, 1]), D[7], atol=1e-9)
    assert np.allclose(R[7], 0)


def test_mds_beats_or_matches_pca_on_distance_rank() -> None:
    Z = _Z(60, 1)
    D = b5.pair_dist(Z)
    assert b5.dist_rho(D, b5.layout_mds(Z)) >= b5.dist_rho(D, b5.layout_pca(Z)) - 0.02


def test_recall_is_one_when_layout_is_the_space() -> None:
    rng = np.random.default_rng(2)
    Z = np.zeros((30, 5))
    Z[:, :2] = rng.normal(size=(30, 2))                   # 実質 2 次元
    D = b5.pair_dist(Z)
    Y = Z[:, :2]
    assert b5.knn_recall(D, Y) == 1.0
    assert abs(b5.dist_rho(D, Y) - 1.0) < 1e-9
    assert b5.knn_recall(D[:4, :4], Y[:4]) is None


def test_ped_hist_shrinks_small_sires_to_global() -> None:
    P = np.zeros((6, len(b5.BINS)))
    P[:, 5] = 10                                          # 全体は 1500m に集中
    P[0, 20] = 3                                          # 馬 0 だけ 3000m
    parent = np.array(["big", "big", "big", "big", "small", ""])
    tabs, g = b5.parent_tables(P, parent, kappa=30.0)
    assert tabs["small"][5] > 0.3 and abs(g[5] - (5 + 10 / 13) / 6) < 1e-9
    assert abs(g.sum() - 1) < 1e-12 and set(tabs) == {"big", "small"}
    assert all(abs(v.sum() - 1) < 1e-12 for v in tabs.values())
    assert tabs["small"].argmax() == g.argmax()           # 産駒 1 頭は全体の形に引かれる
    assert tabs["big"][20] > tabs["small"][20]            # 3000m の産駒がいる種牡馬ほど 3000m 側に寄る
    mixed = b5.ped_hist(tabs["big"], None, g)
    assert np.allclose(mixed, tabs["big"])
    assert np.allclose(b5.ped_hist(None, None, g), g)
    both = b5.ped_hist(tabs["big"], tabs["small"], g)
    assert abs(both.sum() - 1) < 1e-12


def test_hist_summary_and_dict() -> None:
    h = np.zeros(len(b5.BINS))
    h[[5, 6, 7]] = [0.2, 0.5, 0.3]                        # 1500・1600・1700m
    s = b5.hist_summary(h)
    assert s == {"short_m": 1500, "long_m": 1700, "best_m": 1600}
    d = b5.hist_dict(h)
    assert d == {1500: 0.2, 1600: 0.5, 1700: 0.3}


def test_horse_points_bins_and_clips() -> None:
    p = pd.DataFrame({"horse_id": ["a", "a", "b"], "dist100": [1200, 400, 5000], "points": [3.0, 1.0, 5.0]})
    ids, P = b5.horse_points(p)
    assert ids == ["a", "b"]
    assert P[0, 2] == 3.0 and P[0, 0] == 1.0 and P[1, -1] == 5.0     # 範囲外は端の階級へ
