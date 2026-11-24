"""穴馬 × コースのルール探しの道具（src/research/course/longshot_rules.py・longshot_synth.py）の小さな検算。データは読まない。"""

import numpy as np
import pandas as pd

from src.research.course import longshot_rules as LR
from src.research.course import longshot_synth as SY
from src.scripts.docs import build_longshot_rules as BL


def test_race_noise_win_is_exact_for_exclusive_wins():
    # 1 レースに条件の馬 2 頭（q 0.1・0.2）。1 着は 1 頭なので W は 0/1、分散は P(1−P)、P = 0.3
    ridx = np.array([0, 0])
    o = np.ones((2, 1))
    vw, _ = LR.race_noise(ridx, o, np.array([0.1, 0.2]), np.array([0.3, 0.3]), np.array([3.0]), np.array([16.0]),
                          np.array([1.0]), np.array([1.0]))
    assert np.isclose(vw[0], 0.3 * 0.7)


def test_race_noise_place_matches_hypergeometric_for_equal_horses():
    # 16 頭・3 枠・条件の馬 4 頭（どれも 3/16）。非復元の抽出の分散 m·(T/N)(1−T/N)(N−m)/(N−1)
    N, T, m = 16.0, 3.0, 4
    ridx = np.zeros(m, int)
    e = np.full(m, T / N)
    _, vp = LR.race_noise(ridx, np.ones((m, 1)), np.full(m, 0.05), e, np.array([T]), np.array([N]), np.array([1.0]), np.array([1.0]))
    assert np.isclose(vp[0], m * (T / N) * (1 - T / N) * (N - m) / (N - 1))


def test_tau_is_zero_without_dispersion_and_noise():
    E = np.array([5.0, 10.0, 20.0])
    p = np.ones(3)
    assert LR._tau(p * E, E, np.zeros(3), p) == 0.0
    # 雑音だけのばらつきは引かれて 0 になる: (W − pE)²/E の和 = V/E の和
    W = np.array([6.0, 9.0, 20.0])
    V = (W - p * E) ** 2
    assert np.isclose(LR._tau(W, E, V, p), 0.0)


def test_shrink_falls_back_to_prior_when_tau_not_positive():
    out = LR._shrink(np.array([10.0]), np.array([4.0]), np.array([1.0]), 0.0)
    assert np.allclose(out, 1.0)
    out = LR._shrink(np.array([10.0]), np.array([4.0]), np.array([1.0]), 0.1)
    assert 1.0 < out[0] < 2.5


def test_assign_folds_keeps_kaisai_together_and_is_reproducible():
    d = pd.DataFrame({"kaisai": [f"2016050{i % 7}" for i in range(50)]})
    f1, s1 = LR.assign_folds(d, 20261124)
    f2, s2 = LR.assign_folds(d, 20261124)
    assert (f1 == f2).all() and s1 == s2
    for _, g in pd.Series(f1).groupby(d["kaisai"]):
        assert g.nunique() == 1


def test_sim_draw_win_marginals_follow_weights():
    # 2 レース。穴馬の行は各レースの 1 頭目（pop_row 0・1）
    R = pd.DataFrame({"race_id": ["a", "a", "a", "b", "b"], "q": [0.2, 0.5, 0.3, 0.4, 0.6],
                      "odds_final": [5.0, 2.0, 3.3, 2.5, 1.7], "field": [3, 3, 3, 2, 2], "pop_row": [0, -1, -1, 1, -1]})
    d = pd.DataFrame({"obin": [0, 0]})
    sim = SY.Sim(d, R, np.ones(LR.NB))
    assert np.allclose(sim.win_prob(None), [0.2, 0.4])
    rng = np.random.default_rng(0)
    wins = np.mean([sim.draw(None, rng)[0] for _ in range(4000)], axis=0)
    assert np.allclose(wins, [0.2, 0.4], atol=0.03)


def test_doc_labels_keep_open_ended_bands_and_venue_order():
    edges = [10.0, 12.2, 15.0, 50.0]
    assert "10 倍未満" in BL.obin_label({"obin": 0}, edges)
    assert "50 倍超" in BL.obin_label({"obin": 2}, edges)
    units = ["小倉芝1200", "札幌ダ1700", "東京芝1600", "東京ダ1400"]
    assert sorted(units, key=BL.unit_key) == ["札幌ダ1700", "東京ダ1400", "東京芝1600", "小倉芝1200"]


def test_sim_draw_place_slots_follow_field_size():
    # 3 頭立ては払戻の枠が 2。全頭を穴馬の行にすると、1 着は 1 頭・枠の中は 2 頭
    R = pd.DataFrame({"race_id": ["a"] * 3, "q": [0.2, 0.3, 0.5], "odds_final": [5.0, 3.3, 2.0], "field": [3, 3, 3], "pop_row": [0, 1, 2]})
    d = pd.DataFrame({"obin": [0, 0, 0]})
    sim = SY.Sim(d, R, np.ones(LR.NB))
    w, t = sim.draw(None, np.random.default_rng(1))
    assert w.sum() == 1 and t.sum() == 2


def test_permute_keeps_strata_and_race_level_types_consistent():
    rng = np.random.default_rng(0)
    n = 60
    d = pd.DataFrame({"unit": ["東京芝1600"] * n, "cgrp": 0, "era": 0, "obin": rng.integers(0, 2, n),
                      "race_id": np.repeat([f"r{i}" for i in range(12)], 5)})
    L = pd.DataFrame({"A01": rng.choice(["x", "y", "z"], n), "H06": np.repeat([f"h{i % 3}" for i in range(12)], 5)})
    Lp = LR.permute(d, L, {"cuts": {"race_level_types": ["H06"]}}, 7)
    # 馬の型は帯の中で入れ替わるだけ（帯ごとの水準の数は同じ）
    for b in (0, 1):
        m = d["obin"].to_numpy() == b
        assert sorted(L.loc[m, "A01"]) == sorted(Lp.loc[m, "A01"])
    # レース単位の型は、レースの中で揃ったまま
    assert (Lp.groupby(d["race_id"])["H06"].nunique() == 1).all()


def test_subset_2021_remaps_pop_rows():
    d = pd.DataFrame({"year": [2020, 2023, 2021], "race_id": ["a", "b", "c"]})
    L = pd.DataFrame({"A01": ["x", "y", "z"]})
    R = pd.DataFrame({"race_id": ["a", "a", "b", "c"], "q": [0.5, 0.5, 1.0, 1.0], "odds_final": 2.0, "field": 2, "pop_row": [0, -1, 1, 2]})
    d6, L6, R6 = SY.subset_2021(d, L, R)
    assert list(d6["race_id"]) == ["a", "c"] and list(L6["A01"]) == ["x", "z"]
    assert list(R6["race_id"]) == ["a", "a", "c"] and list(R6["pop_row"]) == [0, -1, 1]


def test_page_numbers_come_from_json():
    import json
    D = json.loads(BL.SRC.read_text(encoding="utf-8"))
    page = BL.build()
    assert f"{D['all'][0]['n']:,} 走" in page
    assert f"{len(D['types'])} 型" in page
    assert "2025・26 年は載せない" in page
