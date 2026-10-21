"""tp-60: 構成埋め込みが『ID でなく祖先の構成』で近さを決めること（全兄弟＝距離 0、父だけ違う産駒、同系の近縁）を、小さな架空の血統で確かめる。"""

import numpy as np

from src.scripts.data import tp60_embed as te


def _toy():
    """節点: 0..5 は創始者、6=父X(0,1)、7=母M(2,3)、8=DI(6,7)、9=BT(6,7)（DI と BT は全兄弟）、10=別の母N(4,5)、
    11=Mare(7,3)、12=Mare×DI の産駒（父=8,母=11）、13=Mare×BT の産駒（父=9,母=11）、14=無関係（10, 10 の別親 3,4 を使う）。"""
    sire = np.array([-1, -1, -1, -1, -1, -1, 0, 2, 6, 6, 4, 7, 8, 9, 10])
    dam = np.array([-1, -1, -1, -1, -1, -1, 1, 3, 7, 7, 5, 3, 11, 11, 4])
    return sire, dam


def _cos(E, a, b):
    return float(E[a] @ E[b] / (np.linalg.norm(E[a]) * np.linalg.norm(E[b])))


def test_full_siblings_identical_when_parents_id_excluded():
    sire, dam = _toy()
    E = te.compute_E(sire, dam, d=64, tmin=2, T=6, gamma=1.0, seed=1)
    assert _cos(E, 8, 9) > 0.999999            # DI と BT: 親が同じ → 祖先の構成が同一


def test_father_differs_children_identical_when_fathers_are_full_brothers():
    sire, dam = _toy()
    E = te.compute_E(sire, dam, d=64, tmin=2, T=6, gamma=1.0, seed=1)
    assert _cos(E, 12, 13) > 0.999999          # 同じ母 × DI／BT の産駒


def test_id_bearing_variant_breaks_the_identity():
    sire, dam = _toy()
    E1 = te.compute_E(sire, dam, d=256, tmin=1, T=6, gamma=1.0, seed=1)
    assert _cos(E1, 12, 13) < 0.999            # tmin=1 は父の ID（DI／BT 固有）を持ち込むので同一にならない
    assert _cos(E1, 12, 13) > _cos(E1, 12, 14)


def test_related_closer_than_unrelated():
    sire, dam = _toy()
    E = te.compute_E(sire, dam, d=256, tmin=2, T=6, gamma=1.0, seed=2)
    assert _cos(E, 12, 8) > _cos(E, 12, 14)    # 祖先を共有する DI は、無関係な馬より近い


def test_partner_rank_and_auc():
    sire, dam = _toy()
    E = te.compute_E(sire, dam, d=64, tmin=2, T=6, gamma=1.0, seed=1)
    pool = np.arange(6, 15)
    rk, frac = te.partner_rank(E, pool, 8, 9)
    assert rk == 1 and frac <= 1 / 8 + 1e-9
    pos = np.array([[8, 9], [12, 13]])
    neg = np.array([[8, 14], [9, 14], [12, 14], [13, 14]])
    assert te.pair_auc(E, pos, neg) == 1.0


def test_serving_identity_child_from_parents_shifted_window():
    """未見の馬（親だけ分かる）の埋め込みは、親の『窓を 1 世代ずらした』埋め込みの平均で作れる（推論時は親の 32 次元だけ持てばよい）。"""
    sire, dam = _toy()
    full = te.compute_E(sire, dam, d=64, tmin=2, T=8, gamma=1.0, seed=3)
    shifted = te.compute_E(sire, dam, d=64, tmin=1, T=7, gamma=1.0, seed=3)
    child = 12
    pred = 0.5 * (shifted[sire[child]] + shifted[dam[child]])
    assert np.allclose(full[child], pred, atol=1e-5)
