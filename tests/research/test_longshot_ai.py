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
    assert res["base_diff"] == pytest.approx(0.0)
    assert res["L"] * differ == pytest.approx(0.01, rel=1e-6)


def test_power_sim_includes_base_difference():
    p, star, ref, cl = _sim_case()
    p = p.copy()
    p[1::2] = 0.25                                 # 比べ先（行 2r+1）がもともと 5pt 高い
    res = la.power_sim(p, star, ref, cl, target=0.01, n_sim=10)
    differ = (star != ref).mean()
    assert res["base_diff"] == pytest.approx(-0.05 * differ, rel=1e-6)
    assert res["base_diff"] + res["L"] * differ == pytest.approx(0.01, rel=1e-6)


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
    assert pr["version"] in ("la-2", "la-3", "la-4", "la-5", "la-6", "la-7")
    assert pr["main_arm"].startswith("L1_MF_xgb")
    assert pr["labels"]["train"] == "y_p1"


def test_not_imported_from_serving_path():
    for d in ("src/pipeline", "src/api"):
        for p in (ROOT / d).rglob("*.py"):
            assert "longshot_ai" not in p.read_text(encoding="utf-8", errors="ignore"), p



def _toy_frame():
    import pandas as pd
    rows = []
    # 2 レース × 4 頭。オッズは再現も確定も同じ。馬 1 は一桁（母集団の外）
    for rid in ("201901010101", "201901010102"):
        for h, (o, pl1, pc0, y) in enumerate([(3.0, 0.9, 0.9, 1), (12.0, 0.2, 0.5, 0), (20.0, 0.6, 0.1, 1), (40.0, 0.1, 0.3, 0)], start=1):
            d = {"race_id": rid, "date": "2019-01-05", "year": 2019, "hn": h, "odds_final": o, "p_L1": pl1, "p_C0": pc0,
                 "dnf": False, "y3": bool(y), "fq3": 1.0 / o * 1.5}
            for k in range(5):
                d[f"odds_pre{k}"] = o
                d[f"cq3_{k}"] = 0.5 / o
            rows.append(d)
    return pd.DataFrame(rows)


def test_select_metrics_ae_uses_final_q3_and_diffs():
    r = _toy_frame()
    m = la.select_metrics(r, "a_drop", n_boot=50)
    b = m["boot"]
    # ★ は 20 倍の馬（当たり）、C0 と市場は 12 倍の馬（外れ）
    assert b["d_star_c0"]["point"] == 1.0 and b["d_star_market"]["point"] == 1.0
    assert b["ae_star"]["point"] == pytest.approx(1.0 / (1.5 / 20))            # 分母は確定の q3（fq3）
    assert b["ae_pre_star"]["point"] == pytest.approx(1.0 / (0.5 / 20))        # 再現の q3 は記述


def test_case_c_worst_star_takes_dnf_horse():
    r = _toy_frame()
    m = (r["race_id"] == "201901010101") & (r["hn"] == 4)
    r.loc[m, "dnf"] = True
    r.loc[m, "y3"] = False
    r.loc[m, ["p_L1", "p_C0"]] = np.nan
    a = la.select_metrics(r, "a_drop", n_boot=20)
    c = la.select_metrics(r, "c_worst", n_boot=20)
    assert a["boot"]["rate_star"]["point"] == 1.0
    assert c["boot"]["rate_star"]["point"] == 0.5             # 1 レースは中止の馬を選んだとして 0


def test_judge_select_needs_all_lines():
    def mk(lo_c, lo_m, ae_lo, ae_pt, ae_c0, pt_c=0.01, pt_m=0.01):
        return {"boot": {"d_star_c0": {"lo95": lo_c, "point": pt_c}, "d_star_market": {"lo95": lo_m, "point": pt_m},
                         "ae_star": {"lo95": ae_lo, "point": ae_pt}, "ae_c0": {"point": ae_c0}}}
    good = mk(0.001, 0.001, 1.01, 1.1, 1.0)
    assert la.judge_select(good, good, good)["pass"]
    assert not la.judge_select(mk(-0.001, 0.001, 1.01, 1.1, 1.0), good, good)["pass"]
    assert not la.judge_select(mk(0.001, -0.001, 1.01, 1.1, 1.0), good, good)["pass"]
    assert not la.judge_select(mk(0.001, 0.001, 0.99, 1.1, 1.0), good, good)["pass"]
    assert not la.judge_select(mk(0.001, 0.001, 1.01, 0.9, 1.0), good, good)["pass"]
    assert not la.judge_select(good, mk(0.001, 0.001, 1.01, 1.1, 1.0, pt_m=-0.01), good)["pass"]
    assert not la.judge_select(good, good, mk(0, 0, 1, 1, 1, pt_c=-0.01))["pass"]
    assert not la.judge_select(good, good, None)["pass"]


def test_chalk_races():
    race = np.array(["a"] * 4 + ["b"] * 4)
    y3 = np.array([1, 1, 1, 0, 1, 1, 1, 0], bool)
    pop = np.array([1, 2, 3, 4, 1, 2, 5, 3], float)
    assert la.chalk_races(race, y3, pop).tolist() == [True] * 4 + [False] * 4


def test_cluster_boot_ratio_and_mean():
    cl = np.array([0, 0, 1, 1])
    b = la.cluster_boot(cl, {"m": np.array([1.0, 0.0, 1.0, 1.0]), "r": np.array([1.0, 1.0, 0.0, 2.0])},
                        den={"r": np.array([2.0, 2.0, 2.0, 2.0])}, n_boot=100)
    assert b["m"]["point"] == pytest.approx(0.75)
    assert b["r"]["point"] == pytest.approx(0.5)



def test_b_zero_market_can_pick_dnf_horse():
    r = _toy_frame()
    m = (r["race_id"] == "201901010101") & (r["hn"] == 2)          # 12 倍の馬（市場の 1 位）が中止
    r.loc[m, "dnf"] = True
    r.loc[m, "y3"] = False
    r.loc[m, ["p_L1", "p_C0"]] = np.nan
    a = la.select_metrics(r, "a_drop", n_boot=20)
    b = la.select_metrics(r, "b_zero", n_boot=20)
    # a: 市場は 20 倍の馬（当たり）を選ぶ。b: 市場は中止の 12 倍の馬（0）を選べる
    assert a["boot"]["rate_market"]["point"] == 0.5
    assert b["boot"]["rate_market"]["point"] == 0.0


def test_positive_control_centered_detects_even_if_observed_negative():
    n = 4000
    rng = np.random.default_rng(3)
    d = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.2, 0.62, 0.18])         # 観測の差は負（−2pt）
    P = {"star": np.zeros((5, n), int), "c0": np.ones((5, n), int)}       # どの再現でも違う馬
    m = {"_use": np.ones(n, bool), "_d_sc": d, "_clus": np.arange(n) // 4, "_ys": np.full(n, 0.2), "_P": P}
    big = la.positive_control(m, lift=0.05, reps=20, n_boot=200)
    none = la.positive_control(m, lift=0.0, reps=20, n_boot=200)
    assert big["reached"] and big["detected"] > 0.9
    assert none["detected"] < 0.2


def test_la_label_single_digit_is_zero():
    import pandas as pd
    j = pd.DataFrame({"has_label": [True] * 4, "odds_final": [3.0, 12.0, 5.0, 30.0], "odds_pre0": [3.0, 12.0, 11.0, 30.0],
                      "y_p1": [True, True, True, False], "y3": [True, True, True, False], "race_id": ["a"] * 4,
                      "pop_final": [1.0, 4.0, 2.0, 8.0]})
    rows, y = la.arm_rows(j, "LA")
    assert rows.all() and y.tolist() == [0, 1, 0, 0]
    rows, y = la.arm_rows(j, "L1T")
    assert rows.tolist() == [False, True, True, True]
    rows, _ = la.arm_rows(j, "L1")
    assert rows.tolist() == [False, True, False, True]



def test_l0_lc_rows():
    import pandas as pd
    j = pd.DataFrame({"has_label": [True] * 8, "odds_final": [3.0, 12.0, 5.0, 30.0, 3.0, 12.0, 5.0, 30.0],
                      "odds_pre0": [3.0, 12.0, 5.0, 30.0, 3.0, 12.0, 5.0, 30.0],
                      "y_p1": [True, True, True, True, True, True, True, False], "y3": [True, False, True, True, True, True, True, False],
                      "race_id": ["a"] * 4 + ["b"] * 4, "pop_final": [1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]})
    rows, y = la.arm_rows(j, "L0")
    assert rows.tolist() == [False, True, False, True] * 2 and y.tolist() == [1, 0, 1, 1, 1, 1, 1, 0]
    rows, _ = la.arm_rows(j, "LC")
    # レース b は 3 着内が 1・3・2 番人気（人気どおり）なので訓練から外す。レース a は 4 番人気が 3 着内
    assert rows.tolist() == [False, True, False, True, False, False, False, False]
