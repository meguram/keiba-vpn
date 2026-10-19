import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import RankDistribution
from src.scripts.data import tp36_single as ts
from src.scripts.data.tp36_final2026 import parse_payoff_json


def test_combo_probs_sum_to_expected_totals():
    d = RankDistribution(np.array([0.3, 0.25, 0.2, 0.15, 0.1]), 0.8, 0.7)
    cp = ts.combo_probs(d)
    assert abs(cp["umaren"][1].sum() - 1.0) < 1e-9
    assert abs(cp["wide"][1].sum() - 3.0) < 1e-9
    assert abs(cp["umatan"][1].sum() - 1.0) < 1e-9
    assert abs(cp["sanrenpuku"][1].sum() - 1.0) < 1e-9
    assert abs(cp["sanrentan"][1].sum() - 1.0) < 1e-9


def test_top_k_orders_by_ev_and_skips_nan():
    ev = np.array([0.5, np.nan, 2.0, 1.0, 3.0])
    assert list(ts.top_k(ev, 3)) == [4, 2, 3]
    big = np.random.default_rng(0).random(500)
    assert list(ts.top_k(big, 5)) == list(np.argsort(-big)[:5])


def test_parse_combo_variants():
    assert ts.parse_combo("2-6") == (2, 6)
    assert ts.parse_combo("6>2>3") == (6, 2, 3)
    assert ts.parse_combo("9 → 10") == (9, 10)


def test_load_hits_drops_anomalous_trifecta_and_marks_race_broken():
    pay = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "bet_type": ["sanrentan", "umaren", "sanrentan"], "combo": ["1>2>3", "1-2", "3>2>1"], "payout": [58_367_060, 1200, 5000]})
    h = ts.load_hits(pay)
    assert h["r1"]["_broken"] == {"sanrentan"} and "sanrentan" not in h["r1"] and h["r1"]["umaren"][(1, 2)] == 1200
    assert h["r2"]["sanrentan"][(3, 2, 1)] == 5000
    rec = {("sanrentan", "model", "est"): 1, ("umaren", "model", "est"): 2, ("all", "sanrentan"): 3}
    assert ts.drop_broken(rec, {"sanrentan"}) == {("umaren", "model", "est"): 2}


def test_process_race_records_hit_payout_and_full_buy():
    n = 6
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(n))
    q = rng.dirichlet(np.ones(n))
    g = pd.DataFrame({"horse_number": np.arange(1, n + 1), "p_win": p, "p_top2": np.clip(p * 2, 0, 0.99), "p_top3": np.clip(p * 3, 0, 0.99), "p_blend": p,
                      "q_win": q, "win_odds": 0.8 / q, "ret_win": 0.0, "place_mid": 2.0, "ret_place": 0.0})
    hits = {"umaren": {(1, 2): 1500.0}, "sanrentan": {(1, 2, 3): 9000.0}}
    rec = ts.process_race(g, hits, None, (0.8, 0.7))
    ev, odds, pay, navail = rec[("umaren", "model", "est")]
    assert navail == 15 and len(ev) == 15 and np.all(np.diff(ev) <= 1e-12)
    assert pay.sum() == 1500.0                                # 全組合せを取れば当たりの払戻が 1 回だけ入る
    assert rec[("all", "umaren")] == (15, 1500.0)
    assert rec[("sanrentan", "model", "est")][3] == 120


def test_boot_ci_and_path_stats():
    stake = np.full(400, 100.0)
    ret = np.where(np.arange(400) % 20 == 0, 2400.0, 0.0)
    roi, lo, hi = ts.boot_ci(stake, ret)
    assert lo <= roi <= hi and abs(roi - 1.2) < 1e-9
    ps = ts.path_stats(stake, ret, np.repeat([2024, 2025], 200))
    assert ps["longest_losing_streak"] == 19 and 0 <= ps["ruin_prob_30u"] <= 1


def test_parse_payoff_json_flags_concatenated_multi_hit_entries():
    ok = '{"単勝": {"numbers": "7", "payout": "380"}, "ワイド": [{"numbers": "5 - 7", "payout": "250"}, {"numbers": "3 - 7", "payout": "210"}, {"numbers": "3 - 5", "payout": "230"}]}'
    bad = '{"単勝": {"numbers": "10", "payout": "210"}, "ワイド": {"numbers": "9 - 106 - 106 - 9", "payout": "380330740"}, "三連単": {"numbers": "2 → 12 → 15", "payout": "58,367,060"}}'
    h = parse_payoff_json(ok)
    assert len(h["wide"]) == 3 and h["tansho"][(7,)] == 380.0 and not h["_broken"]
    hb = parse_payoff_json(bad)
    assert hb["_broken"] == {"wide", "sanrentan"} and hb["tansho"][(10,)] == 210.0
