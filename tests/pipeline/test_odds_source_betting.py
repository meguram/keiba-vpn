"""オッズの出どころ選択（実/予測）・3連複確率・optimize_betting の odds_* 応答。外部接続なし。"""

from __future__ import annotations

from itertools import combinations
from unittest.mock import patch

from src.api.v1 import delegates
from src.pipeline.inference import odds_source as OS
from src.pipeline.inference.betting import BetSimulator, BettingConfig, BettingOptimizer

RID = "202606010101"

SINGLE_T15 = {
    "race_id": RID, "phase": "t15", "snapshot_at": "2026-06-01T14:35:00+09:00",
    "entries": [{"horse_number": 1, "win_odds": 8.0, "place_odds_min": 2.0, "place_odds_max": 3.0}],
    "snapshots": [{"phase": "t60", "snapshot_at": "2026-06-01T13:50:00+09:00",
                   "entries": [{"horse_number": 1, "win_odds": 6.0, "place_odds_min": 1.8, "place_odds_max": 2.6}]}],
}
PAIR_T15 = {
    "race_id": RID, "phase": "t15", "snapshot_at": "2026-06-01T14:35:00+09:00",
    "umaren": [{"pair": [1, 2], "odds": 9.0}], "sanrenpuku": [{"combo": [3, 2, 1], "odds": 20.0}],
    "snapshots": [{"phase": "t60", "snapshot_at": "2026-06-01T13:50:00+09:00",
                   "umaren": [{"pair": [1, 2], "odds": 7.0}], "sanrenpuku": []}],
}
PREDICTED = {"entries": [{"horse_number": 1, "predicted_win_odds": 5.0,
                          "predicted_place_odds_min": 1.5, "predicted_place_odds_max": 2.2}]}


class _Storage:
    def __init__(self, docs=None):
        self.docs = docs or {}

    def load(self, category, key):
        return self.docs.get(category)


def _resolve(docs, **kw):
    with patch("src.pipeline.inference.final_odds_service.load_cached_response", return_value=kw.pop("pred", None)):
        return OS.resolve_odds(_Storage(docs), RID, **kw)


# ── pick_snapshot / list_phases ──

def test_pick_snapshot_latest_and_history():
    latest = OS.pick_snapshot(SINGLE_T15)
    assert latest["phase"] == "t15" and "snapshots" not in latest
    t60 = OS.pick_snapshot(SINGLE_T15, "t60")
    assert t60["phase"] == "t60" and t60["entries"][0]["win_odds"] == 6.0 and t60["race_id"] == RID
    assert OS.pick_snapshot(SINGLE_T15, "t15")["snapshot_at"].startswith("2026-06-01T14:35")
    assert OS.pick_snapshot(SINGLE_T15, "t30") is None
    assert OS.pick_snapshot(None) is None and OS.pick_snapshot({}) is None


def test_pick_snapshot_legacy_doc_is_latest():
    legacy = {"race_id": RID, "entries": [{"horse_number": 1, "win_odds": 3.0}]}
    assert OS.pick_snapshot(legacy)["entries"][0]["win_odds"] == 3.0
    assert OS.pick_snapshot(legacy, "t60") is None
    assert OS.list_phases(legacy) == [{"phase": "latest", "as_of": None}]


def test_list_phases():
    assert [p["phase"] for p in OS.list_phases(SINGLE_T15)] == ["t15", "t60"]
    assert OS.list_phases(None) == []


# ── resolve_odds ──

def test_auto_prefers_actual():
    r = _resolve({"race_odds": SINGLE_T15, "race_pair_odds": PAIR_T15}, pred=PREDICTED)
    assert r["source"] == "actual" and r["phase"] == "t15" and r["allowed_bet_types"] is None
    assert r["as_of"].startswith("2026-06-01T14:35")
    assert r["pair_odds"]["umaren"][0]["odds"] == 9.0
    assert [p["phase"] for p in r["available"]] == ["t15", "t60"]


def test_auto_phase_t60_uses_matching_pair_snapshot():
    r = _resolve({"race_odds": SINGLE_T15, "race_pair_odds": PAIR_T15}, phase="t60")
    assert r["source"] == "actual" and r["phase"] == "t60"
    assert r["single_odds"]["entries"][0]["win_odds"] == 6.0
    assert r["pair_odds"]["umaren"][0]["odds"] == 7.0


def test_auto_falls_back_to_predicted_without_actual():
    r = _resolve({}, pred=PREDICTED)
    assert r["source"] == "predicted" and r["allowed_bet_types"] == {"tansho", "fukusho"}
    assert r["pair_odds"] == {} and r["single_odds"]["entries"][0]["win_odds"] == 5.0
    assert r["single_odds"]["entries"][0]["place_odds_min"] == 1.5


def test_auto_missing_phase_falls_back_to_predicted_and_notes_it():
    r = _resolve({"race_odds": SINGLE_T15}, phase="t30", pred=PREDICTED)
    assert r["source"] == "predicted" and "t30" in r["note"]


def test_actual_mode_never_falls_back():
    r = _resolve({}, mode="actual", pred=PREDICTED)
    assert r["source"] is None and r["single_odds"] is None and "未取得" in r["note"]


def test_predicted_mode_ignores_actual():
    r = _resolve({"race_odds": SINGLE_T15, "race_pair_odds": PAIR_T15}, mode="predicted", pred=PREDICTED)
    assert r["source"] == "predicted" and r["pair_odds"] == {}


def test_nothing_available():
    r = _resolve({}, pred=None)
    assert r["source"] is None and "予測オッズも未計算" in r["note"]


# ── 3連複 ──

PROBS = {n: p for n, p in zip(range(1, 8), [0.30, 0.22, 0.15, 0.12, 0.10, 0.07, 0.04])}


def test_sanrenpuku_probabilities_sum_to_one():
    opt = BettingOptimizer()
    total = sum(opt._sanrenpuku_prob(PROBS, *c) for c in combinations(sorted(PROBS), 3))
    assert abs(total - 1.0) < 1e-9
    assert opt._sanrenpuku_prob(PROBS, 1, 2, 3) == opt._sanrenpuku_prob(PROBS, 3, 1, 2)
    assert opt._sanrenpuku_prob(PROBS, 1, 1, 2) == 0.0


def test_optimize_sanrenpuku_candidate_shape():
    preds = [{"horse_number": n, "horse_name": f"H{n}", "pred_score": s}
             for n, s in zip(range(1, 8), [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])]
    pair_odds = {"sanrenpuku": [{"combo": [3, 2, 1], "odds": 60.0}, {"combo": [5, 6, 7], "odds": 900.0}]}
    cfg = BettingConfig(bet_types=["sanrenpuku"], min_ev=1.0, min_prob=0.001, top_n_for_pairs=4)
    res = BettingOptimizer(cfg).optimize(preds, pair_odds, 100000)
    assert len(res["candidates"]) == 1  # 5-6-7 は上位4頭の外なので評価しない
    c = res["candidates"][0]
    assert c.pair == (1, 2, 3) and c.pair_label == "1-2-3" and c.horse_names == ("H1", "H2", "H3")
    assert c.bet_type == "sanrenpuku" and c.odds == 60.0


def test_simulator_sanrenpuku_hit():
    preds = [{"horse_number": n, "horse_name": f"H{n}", "pred_score": s}
             for n, s in zip(range(1, 8), [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])]
    sim = BetSimulator(BettingConfig(bet_types=["sanrenpuku"], min_ev=1.0, min_prob=0.001))
    pair_odds = {"sanrenpuku": [{"combo": [1, 2, 3], "odds": 60.0}]}
    assert sim.simulate_race(preds, pair_odds, [3, 1, 2, 4], 100000)["hits"]
    assert not sim.simulate_race(preds, pair_odds, [3, 1, 4, 2], 100000)["hits"]


# ── optimize_betting（合成 storage・実 BettingOptimizer）──

PREDS = {"predictions": [
    {"horse_number": n, "horse_name": f"H{n}", "pred_score": s}
    for n, s in zip(range(1, 7), [3.0, 2.0, 1.5, 1.0, 0.5, 0.0])
]}
ACTUAL_SINGLE = {
    "race_id": RID, "phase": "t15", "snapshot_at": "2026-06-01T14:35:00+09:00",
    "entries": [{"horse_number": 1, "win_odds": 8.0, "place_odds_min": 2.0, "place_odds_max": 3.0}],
}
ACTUAL_PAIR = {"race_id": RID, "phase": "t15", "sanrenpuku": [{"combo": [1, 2, 3], "odds": 40.0}]}


def _optimize(docs, body, pred=PREDICTED):
    storage = _Storage(docs)
    with patch("src.pipeline.inference.race_prediction_service.load_cached", return_value=PREDS), \
         patch("src.scraper.storage.HybridStorage", lambda: storage), \
         patch("src.pipeline.inference.final_odds_service.load_cached_response", return_value=pred):
        return delegates.optimize_betting({"race_id": RID, "min_prob": 0.001, **body})


def test_optimize_actual_reports_source_and_gap():
    payload, code = _optimize({"race_odds": ACTUAL_SINGLE, "race_pair_odds": ACTUAL_PAIR},
                              {"bet_types": ["tansho", "sanrenpuku"], "min_ev": 1.0})
    assert code == 200
    assert payload["odds_source"] == "actual" and payload["odds_phase"] == "t15"
    assert payload["odds_as_of"].startswith("2026-06-01T14:35")
    by_type = {c["bet_type"]: c for c in payload["candidates"]}
    t = by_type["tansho"]
    assert t["predicted_odds"] == 5.0 and t["odds_gap"] == 0.6  # 8.0/5.0 - 1
    s = by_type["sanrenpuku"]
    assert s["pair"] == [1, 2, 3] and s["pair_label"] == "1-2-3" and len(s["horse_names"]) == 3
    assert "predicted_odds" not in s


def test_optimize_predicted_mode_restricts_bet_types():
    payload, code = _optimize({"race_odds": ACTUAL_SINGLE}, {"odds_mode": "predicted", "min_ev": 0.0,
                                                             "bet_types": ["tansho", "umaren", "sanrenpuku"]})
    assert code == 200 and payload["odds_source"] == "predicted"
    assert {c["bet_type"] for c in payload["candidates"]} <= {"tansho"}
    assert all("odds_gap" not in c for c in payload["candidates"])  # 予測オッズ同士の乖離は意味が無い
    code2 = _optimize({}, {"odds_mode": "predicted", "bet_types": ["umaren"]})[1]
    assert code2 == 400


def test_optimize_actual_mode_without_odds_is_404_with_reason():
    payload, code = _optimize({}, {"odds_mode": "actual"})
    assert code == 404 and "実オッズ" in payload["error"]
    payload, code = _optimize({}, {"odds_mode": "predicted"}, pred=None)
    assert code == 404 and "予測オッズ" in payload["error"]


def test_optimize_auto_without_any_odds_keeps_legacy_200():
    payload, code = _optimize({}, {}, pred=None)
    assert code == 200 and payload["odds_source"] is None and payload["candidates"] == []
    assert "未" in payload["odds_note"]


# ── 購入合計の上限（max_total_bet） ──

def _cap_candidates(n=6):
    from src.pipeline.inference.betting import BetCandidate

    return [BetCandidate("tansho", (i + 1,), (f"h{i}", ""), 6.0, 0.25, 1.5, 0.05 - i * 0.006, 0, 0.0) for i in range(n)]


def test_capped_allocation_never_exceeds_budget_and_uses_100_yen_units():
    from src.pipeline.inference.betting import BettingOptimizer

    for budget in (100, 700, 2000, 2050, 9900):
        cs = _cap_candidates()
        out = BettingOptimizer._allocate_capped(cs, budget, sum(c.kelly_fraction for c in cs))
        total = sum(c.bet_amount for c in out)
        assert total <= budget and all(c.bet_amount % 100 == 0 and c.bet_amount >= 100 for c in out)
        assert total == budget // 100 * 100  # 100 円単位で使い切る


def test_capped_allocation_prefers_higher_kelly_when_budget_is_tight():
    from src.pipeline.inference.betting import BettingOptimizer

    cs = _cap_candidates()
    out = BettingOptimizer._allocate_capped(cs, 300, sum(c.kelly_fraction for c in cs))
    assert [c.pair[0] for c in out] == [1, 2, 3] and [c.bet_amount for c in out] == [100, 100, 100]
    assert BettingOptimizer._allocate_capped(_cap_candidates(), 50, 0.2) == []


def test_optimize_betting_default_cap_is_2000_and_overridable():
    p_default, c1 = _optimize({"race_odds": ACTUAL_SINGLE}, {"min_ev": 0.0, "bet_types": ["tansho"]})
    assert c1 == 200 and p_default["max_total_bet"] == 2000 and p_default["total_bet"] <= 2000
    p_500, _ = _optimize({"race_odds": ACTUAL_SINGLE}, {"min_ev": 0.0, "bet_types": ["tansho"], "max_total_bet": 500})
    assert p_500["total_bet"] <= 500
    p_none, _ = _optimize({"race_odds": ACTUAL_SINGLE}, {"min_ev": 0.0, "bet_types": ["tansho"], "max_total_bet": None})
    assert p_none["max_total_bet"] is None
    assert _optimize({"race_odds": ACTUAL_SINGLE}, {"max_total_bet": 50})[1] == 400
    assert _optimize({"race_odds": ACTUAL_SINGLE}, {"max_total_bet": "abc"})[1] == 400


def test_build_odds_map_tolerates_none_odds_in_predictions():
    """バックフィルの予測は win_odds 等が None のことがある。落とさず、値がある馬だけ拾う。"""
    preds = [{"horse_number": 1, "win_odds": None, "place_odds_min": None},
             {"horse_number": 2, "win_odds": 3.5, "place_odds_min": 1.4, "place_odds_max": 1.8}]
    r = BettingOptimizer(BettingConfig())._build_odds_map({}, preds, {})
    assert r["tansho"] == {(2,): 3.5} and r["fukusho"] == {(2,): 1.4}
