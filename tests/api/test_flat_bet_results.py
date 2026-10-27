"""パターン別の推奨の結果は『条件（確定オッズ ≥ 想定倍率）を満たしたパターンを 100 円ずつ買った場合』で精算する。結果表示用の API も確認する。"""
from unittest.mock import patch

from src.api import tier
from src.api.v1.delegates import attach_bet_results
from src.pipeline.inference.race_result_view import build_result_view


class _St:
    def load(self, cat, rid):
        if cat != "race_result":
            return None
        return {"race_name": "テスト", "date": "2026-10-25", "entries": [
            {"horse_number": n, "finish_position": p, "horse_name": f"馬{n}", "jockey_name": "騎手", "odds": 3.5, "popularity": p} for n, p in ((5, 1), (1, 2), (2, 3), (3, 4))],
            "payoff": {"単勝": {"numbers": "5", "payout": "450", "popularity": "2"}, "複勝": [{"numbers": "5", "payout": "150"}, {"numbers": "1", "payout": "130"}],
                       "馬連": {"numbers": "1-5", "payout": "1,230", "popularity": "4"}}}


def _payload():
    return {"strategy": "target", "status": "bet", "bets": [
        {"bet_type": "tansho", "combo": [5], "yen": 700, "est_multiple": 4.0},      # 当たり・確定 4.5 倍 ≥ 4.0 で条件を満たす
        {"bet_type": "umaren", "combo": [1, 5], "yen": 300, "est_multiple": 15.0},  # 当たりだが確定 12.3 倍 < 15.0 で条件外
        {"bet_type": "tansho", "combo": [2], "yen": 500, "est_multiple": 6.0},      # 外れ・条件を満たす
    ]}


def test_flat_100_counts_only_patterns_that_pass_the_condition():
    amap = {("tansho", (5,)): 4.5, ("umaren", (1, 5)): 12.3, ("tansho", (2,)): 8.0}
    with patch("src.pipeline.inference.prediction_results.actual_odds_map", return_value=amap):
        out = attach_bet_results(_payload(), "R1", _St())
    b = out["bets"]
    assert [x["result"]["passed"] for x in b] == [True, False, True]
    assert b[0]["result"]["payout"] == 450 and b[1]["result"]["payout"] == 1230 and b[2]["result"]["hit"] is False
    r = out["result"]
    assert (r["invested"], r["returned"], r["hits"], r["n_bets"]) == (200, 450, 1, 2)
    assert r["roi"] == 2.25 and r["stake_yen"] == 100 and r["condition_known"] is True and r["n_patterns"] == 3
    assert r["all"] == {"invested": 300, "returned": 1680, "hits": 2}


def test_flat_100_counts_every_pattern_when_actual_odds_are_unknown():
    with patch("src.pipeline.inference.prediction_results.actual_odds_map", return_value={}):
        out = attach_bet_results(_payload(), "R1", _St())
    assert [x["result"]["passed"] for x in out["bets"]] == [None, None, None]
    r = out["result"]
    assert (r["invested"], r["n_bets"], r["condition_known"]) == (300, 3, False)


def test_result_view_lists_finish_and_payoff_per_bet_type():
    v = build_result_view(_St(), "R1")
    assert v["available"] and v["status"] == "final" and v["official"]
    assert [(f["pos"], f["horse_number"]) for f in v["finish"]] == [(1, 5), (2, 1), (3, 2), (4, 3)]
    assert [p["bet_label"] for p in v["payoff"]] == ["単勝", "複勝", "馬連"]
    assert v["payoff"][1]["rows"] == [{"numbers": "5", "payout": 150, "popularity": None}, {"numbers": "1", "payout": 130, "popularity": None}]


def test_result_view_is_unavailable_without_a_result():
    assert build_result_view(type("S", (), {"load": lambda self, c, r: None})(), "R9") == {"available": False, "race_id": "R9"}


def test_result_path_is_member_only():
    assert tier.member_only_path("/api/race/202604040404/result")
    assert not tier.member_only_path("/api/race/202604040404/result-status")
