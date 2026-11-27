"""穴馬 AI（src/research/longshot_ai/lsai.py）の純粋関数のテスト。データは読まない。"""

from pathlib import Path

import numpy as np
import pytest

from src.research.longshot_ai import lsai as la

ROOT = Path(__file__).resolve().parents[2]


def test_mf_columns_drops_policy_and_rejects_market_names():
    assert la.mf_columns(["a", "oikiri_eval", "oikiri_eval_rank", "oikiri_eval_gap", "course_type", "b"]) == ["a", "b"]
    with pytest.raises(AssertionError):
        la.mf_columns(["a", "lag1_popularity"])


def test_population_threshold_and_missing():
    m = la.population(np.array([9.9, 10.0, 55.0, np.nan]))
    assert m.tolist() == [False, True, True, False]


def test_race_top1_mask_and_tiebreak():
    race = np.array(["r1", "r1", "r1", "r2", "r2", "r3"])
    score = np.array([0.5, 0.9, 0.9, 0.2, 0.1, 0.7])
    mask = np.array([True, True, True, True, True, False])
    hn = np.array([1, 3, 2, 1, 2, 1])
    out = la.race_top1(race, score, mask, hn)
    # r1: 0.9 が 2 頭 → 馬番の小さい 2 番（行 2）。r2: 行 3。r3: 母集団が空なので印なし
    assert out.tolist() == [False, False, True, True, False, False]


def test_q_from_odds_sums_to_one_and_zero_for_missing():
    race = np.array(["a", "a", "a"])
    q = la.q_from_odds(race, np.array([2.0, 4.0, np.nan]))
    assert q.sum() == pytest.approx(1.0)
    assert q[2] == 0.0
    assert q[0] == pytest.approx(2 / 3)


def test_odds_band_edges():
    b = la.odds_band(np.array([9.9, 10.0, 14.9, 15.0, 29.9, 30.0, 50.0, 300.0]))
    assert b.tolist() == [-1, 0, 0, 1, 1, 2, 3, 3]


def _sim_case(n_r=3000, agree=0.5, seed=0):
    rng = np.random.default_rng(seed)
    # レースごとに 2 頭（行 2r と 2r+1）。★ は 2r、比べ先は agree の割合で同じ馬、ほかは 2r+1。再現は 2 通り
    p = np.full(2 * n_r, 0.2)
    star = np.tile(np.arange(n_r) * 2, (2, 1))
    ref = np.where(rng.random((2, n_r)) < agree, star, star + 1)
    cl = np.arange(n_r) // 10
    return p, star, ref, cl


def test_power_sim_lift_hits_target_difference():
    p, star, ref, cl = _sim_case()
    res = la.power_sim(p, star, ref, cl, target=0.01, n_sim=10)
    # 上乗せ L × （差のあるレースの割合）＝ target になる
    differ = (star != ref).mean()
    assert res["L"] * differ == pytest.approx(0.01, rel=1e-6)


def test_power_sim_null_is_about_alpha_and_large_is_high():
    p, star, ref, cl = _sim_case()
    lo = la.power_sim(p, star, ref, cl, target=1e-9, n_sim=300, seed=1)["power"]
    hi = la.power_sim(p, star, ref, cl, target=0.08, n_sim=200, seed=1)["power"]
    assert lo < 0.08
    assert hi > 0.95


def test_power_sim_all_same_has_no_power():
    p, star, ref, cl = _sim_case(agree=1.0)
    res = la.power_sim(p, star, ref, cl, target=0.01, n_sim=20)
    assert res["power"] == 0.0 and res["L"] is None


def test_prereg_is_la2_and_main_arm_only():
    pr = la.prereg()
    assert pr["version"] in ("la-2", "la-3", "la-4", "la-5")
    assert pr["main_arm"].startswith("L1_MF_xgb")
    assert pr["labels"]["train"] == "y_p1"


def test_not_imported_from_serving_path():
    for d in ("src/pipeline", "src/api"):
        for p in (ROOT / d).rglob("*.py"):
            assert "longshot_ai" not in p.read_text(encoding="utf-8", errors="ignore"), p
