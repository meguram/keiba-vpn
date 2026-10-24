import numpy as np
import pandas as pd

from src.scripts.data import tp43_marks as tm


def _pay():
    return {
        "tansho": {1: 300.0},
        "fukusho": {1: 120.0, 2: 150.0, 3: 200.0},
        "umaren": {frozenset((1, 2)): 500.0},
        "wide": {frozenset((1, 2)): 200.0, frozenset((1, 3)): 400.0, frozenset((2, 3)): 900.0},
        "umatan": {(1, 3): 1200.0},
        "sanrenpuku": {frozenset((1, 2, 3)): 1500.0},
    }


def test_race_returns_counts_tickets_and_pays_per_stake():
    r = dict(zip(tm.SCHEME_NAMES, tm.race_returns([1, 2, 3, 4, 5, 6], _pay())))
    assert r["tan1"] == 3.0 and abs(r["fuku1"] - 1.2) < 1e-6
    assert abs(r["umaren_n3"] - 2.5) < 1e-6                 # 500/100 を 2 点で等分
    assert abs(r["umaren_b3"] - 5.0 / 3) < 1e-6
    assert abs(r["wide_n3"] - (2.0 + 4.0) / 2) < 1e-6
    assert abs(r["wide_b3"] - (2.0 + 4.0 + 9.0) / 3) < 1e-6
    assert abs(r["umatan_n3"] - 12.0 / 2) < 1e-6            # 1→3 だけ的中
    assert abs(r["sanpuku_a4"] - 15.0 / 3) < 1e-6           # ◎+2頭（○▲△から）: 3 点
    assert abs(r["sanpuku_b4"] - 15.0 / 4) < 1e-6 and abs(r["sanpuku_b5"] - 15.0 / 10) < 1e-6
    assert r["sanpuku_b3"] == 15.0


def test_order_of_marks_matters_for_umatan_and_axis():
    r = dict(zip(tm.SCHEME_NAMES, tm.race_returns([3, 1, 2, 4, 5, 6], _pay())))
    assert r["umatan_n3"] == 0.0                             # 3→1, 3→2 は外れ
    assert abs(r["umaren_n3"] - 0.0) < 1e-6                  # 3-1, 3-2: 馬連は 1-2 のみ
    assert abs(r["wide_n3"] - (4.0 + 9.0) / 2) < 1e-6


def _race(n=8, seed=0):
    rng = np.random.default_rng(seed)
    pw = rng.dirichlet(np.ones(n) * 2)
    pt3 = np.clip(pw * 3 + 0.05, 0.02, 0.95)
    odds = 0.8 / pw * rng.uniform(0.8, 1.2, n)
    q = (1 / odds) / (1 / odds).sum()
    return pd.DataFrame({"race_id": "r", "year": 2024, "horse_number": np.arange(1, n + 1), "p_win": pw, "p_top2": pt3 * 0.7, "p_top3": pt3,
                         "win_odds": odds, "q_win": q, "place_mid": 1 + odds / 4, "p_blend": 0.5 * pw + 0.5 * q})


def test_mark_scores_threshold_priority_and_missing_blend():
    g = _race()
    s = tm.mark_scores(g)
    top = int(np.argmax(s["M5b_t35"]))
    assert g["p_top3"].iloc[top] >= 0.35                    # 複勝確率の条件を満たす馬が優先される
    assert int(np.argmax(s["M1_pwin"])) == int(np.argmax(g["p_win"]))
    g2 = g.drop(columns=["p_blend"])
    s2 = tm.mark_scores(g2)
    assert s2["M3b_evwin_blend"] is None and s2["M5b_t35"] is None and s2["M1_pwin"] is not None


def test_race_features_orientation():
    flat = _race(10)
    flat[["p_win", "q_win"]] = 0.1
    sharp = flat.copy()
    sharp["p_win"] = [0.82] + [0.02] * 9
    sharp["q_win"] = sharp["p_win"]
    sharp["win_odds"] = 0.8 / sharp["p_win"]
    ff, fs = tm.race_features(flat), tm.race_features(sharp)
    assert ff["ent"] > fs["ent"] and ff["max"] > fs["max"] and ff["fav"] > fs["fav"] and ff["mkt"] > fs["mkt"]


def test_build_and_buckets_and_m6():
    frames = []
    for i in range(30):
        g = _race(8 + i % 4, seed=i)
        g["race_id"] = f"2024{i:08d}"
        frames.append(g)
    fr = pd.concat(frames, ignore_index=True)
    pay = {rid: _pay() for rid in fr["race_id"].unique()}
    rf, R = tm.build(fr, pay)
    assert R.shape == (30, len(tm.COMBOS)) and len(rf) == 30
    assert np.isfinite(R).all()
    cuts = tm.cutoffs(rf.assign(year=2022))
    bk = tm.buckets(rf, cuts)
    assert set(np.unique(bk["ent"])) <= {0, 1, 2}
    M6 = tm.derived_m6(R, bk["ent"].to_numpy(), "M5b_t35")
    i1 = tm.COMBO_IDX[("M1_pwin", "tan1")]
    i5 = tm.COMBO_IDX[("M5b_t35", "tan1")]
    sch = tm.SCHEME_NAMES.index("tan1")
    for k in range(30):
        assert M6[k, sch] == (R[k, i5] if bk["ent"].iloc[k] == 2 else R[k, i1])


def test_reality_check_detects_planted_edge_and_not_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(3000, 120)).astype(np.float32)
    null = tm.reality_check(X, n_boot=200, min_bets=10)
    assert null["p"] > 0.05
    X2 = X.copy()
    X2[:, 7] += 0.15
    alt = tm.reality_check(X2, n_boot=200, min_bets=10)
    assert alt["p"] < 0.05 and alt["kept"][alt["best_idx_in_kept"]] == 7


def test_ruin_prob_decreases_with_edge():
    rng = np.random.default_rng(0)
    base = rng.choice([0.0, 3.0], p=[0.7, 0.3], size=2000)         # 平均 0.9
    good = rng.choice([0.0, 3.5], p=[0.7, 0.3], size=2000)         # 平均 1.05
    assert tm.ruin_prob(base, n_sim=800) > tm.ruin_prob(good, n_sim=800)
