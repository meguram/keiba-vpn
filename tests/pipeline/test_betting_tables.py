"""馬券シミュ用テーブルの整形（払戻の文字列分割・組み合わせの復元・整合の検証）のテスト。実データで見つけた例を使う。"""

import json

import numpy as np
import pandas as pd

from src.pipeline.betting import build_odds_payoff_tables as bt


def test_split_payouts_concatenated_amounts():
    assert bt.split_payouts("100230150", 3) == [[100, 230, 150]]
    assert bt.split_payouts("4002401,020", 3) == [[400, 240, 1020]]
    assert bt.split_payouts("7402,0501,370", 3) == [[740, 2050, 1370]]
    assert bt.split_payouts("1,090") == [[1090]]
    assert bt.split_payouts("930") == [[930]]
    assert bt.split_payouts("") == []


def test_split_payouts_rejects_invalid_numbers():
    assert bt.split_payouts("12345") == []          # 4 桁以上でカンマ無し・3 桁の連なりに分けられない
    assert bt.split_payouts("105") == []            # 10 円単位でない


PAYOFF_A = {"単勝": {"numbers": "4", "payout": "120"}, "複勝": {"numbers": "458", "payout": "100230150"}, "枠連": {"numbers": "4 - 5", "payout": "1,090"},
            "馬連": {"numbers": "4 - 5", "payout": "1,040"}, "ワイド": {"numbers": "4 - 54 - 85 - 8", "payout": "4002401,020"},
            "馬単": {"numbers": "4 → 5", "payout": "1,200"}, "三連複": {"numbers": "4 - 5 - 8", "payout": "2,060"}, "三連単": {"numbers": "4 → 5 → 8", "payout": "5,930"}}


def _rows(payoff, fin):
    return {(r["bet_type"], r["combo"]): r["payout"] for r in bt.parse_payoff(payoff, fin, list(fin))}


def test_parse_payoff_real_example_a():
    got = _rows(PAYOFF_A, {4: 1, 5: 2, 8: 3, 1: 4, 2: 5})
    assert got[("tansho", "4")] == 120
    assert (got[("fukusho", "4")], got[("fukusho", "5")], got[("fukusho", "8")]) == (100, 230, 150)
    assert got[("wakuren", "4-5")] == 1090 and got[("umaren", "4-5")] == 1040
    assert (got[("wide", "4-5")], got[("wide", "4-8")], got[("wide", "5-8")]) == (400, 240, 1020)
    assert got[("umatan", "4>5")] == 1200 and got[("sanrenpuku", "4-5-8")] == 2060 and got[("sanrentan", "4>5>8")] == 5930


def test_parse_payoff_two_digit_horse_numbers_real_example_b():
    p = {"複勝": {"numbers": "16123", "payout": "290220390"}, "ワイド": {"numbers": "12 - 163 - 163 - 12", "payout": "7402,0501,370"},
         "馬単": {"numbers": "16 → 12", "payout": "4,380"}, "三連複": {"numbers": "3 - 12 - 16", "payout": "9,970"}}
    got = _rows(p, {16: 1, 12: 2, 3: 3, 1: 4})
    assert (got[("fukusho", "16")], got[("fukusho", "12")], got[("fukusho", "3")]) == (290, 220, 390)
    assert (got[("wide", "12-16")], got[("wide", "3-16")], got[("wide", "3-12")]) == (740, 2050, 1370)
    assert got[("umatan", "16>12")] == 4380 and got[("sanrenpuku", "3-12-16")] == 9970


def test_dead_heat_win_has_two_rows():
    got = _rows({"単勝": {"numbers": "58", "payout": "310450"}}, {5: 1, 8: 1, 2: 3})
    assert got == {("tansho", "5"): 310, ("tansho", "8"): 450}


def test_unparseable_is_flagged_not_dropped():
    rows = bt.parse_payoff({"単勝": {"numbers": "99", "payout": "120"}}, {4: 1}, [4])
    assert rows and rows[0]["parsed"] is False


def test_pair_combo_orders():
    assert bt.pair_combo("[2, 1]", "umaren") == "1-2" and bt.pair_combo("[2, 1]", "umatan") == "2>1" and bt.pair_combo("[3, 12]", "wide") == "3-12"


def _result(rid="r1"):
    fin = {4: 1, 5: 2, 8: 3, 1: 4}
    return pd.DataFrame({"race_id": rid, "horse_number": list(fin), "horse_id": [f"h{h}" for h in fin], "finish_position": list(fin.values()),
                         "payoff": json.dumps(PAYOFF_A, ensure_ascii=False), "date": "2025-01-05"})


def test_build_payoff_table_and_consistency():
    res = _result()
    pay = bt.build_payoff_table(res)
    assert pay["parsed"].all() and set(pay["bet_type"]) == set(bt.BET_KEYS.values())
    odds = pd.DataFrame({"race_id": "r1", "date": "2025-01-05", "horse_number": [4, 5, 8, 1], "win_odds": [1.2, 5.0, 9.9, 20.0],
                         "place_odds_min": [1.0, 2.0, 1.4, 3.0], "place_odds_max": [1.1, 2.5, 1.6, 4.0], "popularity": [1, 2, 3, 4]})
    wp = bt.build_win_place_table(odds, res)
    assert wp.loc[wp.horse_number == 4, "horse_id"].iloc[0] == "h4" and not wp["has_snapshot"].any()
    pair = pd.DataFrame({"race_id": "r1", "bet_type": ["umaren", "umatan", "wide"], "combo": ["4-5", "4>5", "4-5"], "odds": [10.4, 12.0, np.nan],
                         "odds_min": [np.nan, np.nan, 3.9], "odds_max": [np.nan, np.nan, 4.1], "popularity": [1, 1, 1]})
    rep = bt.consistency_report(pay, wp, pair, res)
    assert rep["payoff_parse_rate"] == 1.0
    assert rep["tansho_exact_share"] == 1.0 and rep["fukusho_inside_range_share"] == 1.0
    assert rep["umaren_exact_share"] == 1.0 and rep["umatan_exact_share"] == 1.0 and rep["wide_inside_range_share"] == 1.0


def test_takeout_implied_payback_for_fair_odds():
    odds = pd.DataFrame({"race_id": ["202501010101"] * 4, "date": "2025-01-05", "horse_number": [1, 2, 3, 4], "win_odds": [3.2, 3.2, 3.2, 3.2], "place_odds_min": 1.0, "place_odds_max": 1.0,
                         "popularity": 1, "horse_id": "x", "phase": "p", "has_snapshot": False})
    pair = pd.DataFrame({"race_id": [], "bet_type": [], "combo": [], "odds": [], "odds_min": [], "odds_max": [], "popularity": []})
    t = bt.takeout_by_year(odds, pair)
    assert abs(t["tansho"]["2025"]["implied_payback"] - 0.8) < 1e-9


def test_old_list_format_with_popularity():
    p = {"複勝": [{"numbers": "13", "payout": "900", "popularity": "11"}, {"numbers": "2", "payout": "170", "popularity": "1"}, {"numbers": "10", "payout": "640", "popularity": "8"}],
         "ワイド": [{"numbers": "2 - 13", "payout": "2,490", "popularity": "32"}, {"numbers": "10 - 13", "payout": "7,940", "popularity": "59"}],
         "単勝": {"numbers": "13", "payout": "4,220", "popularity": "10"}}
    rows = bt.parse_payoff(p, {13: 1, 2: 2, 10: 3}, [13, 2, 10])
    got = {(r["bet_type"], r["combo"]): (r["payout"], r["popularity"]) for r in rows}
    assert got[("fukusho", "13")] == (900, 11) and got[("fukusho", "2")] == (170, 1)
    assert got[("wide", "2-13")] == (2490, 32) and got[("wide", "10-13")] == (7940, 59)
    assert got[("tansho", "13")] == (4220, 10)
    assert all(r["parsed"] and not r["ambiguous"] for r in rows)
