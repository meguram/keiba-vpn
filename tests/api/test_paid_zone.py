"""有料ゾーン（会員向け）の推奨馬券: 推奨度の段階と買い目の区分だけを出し、オッズ・期待値・金額の数値は出さない。外部接続なし。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.v1 import delegates
from src.utils import paid_zone as PZ

RID = "202606010101"
WIN = [0.30, 0.20, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]
ACTUAL_WIN_ODDS = [2.9, 4.6, 6.1, 7.8, 10.5, 13.0, 24.0, 33.0]


def _preds():
    out = []
    for i, w in enumerate(WIN, 1):
        out.append({"horse_number": i, "horse_name": f"馬{i}", "horse_id": f"h{i}", "win_prob": w, "top2_prob": min(w * 1.85, 0.9),
                    "top3_prob": min(w * 2.5, 0.95), "pred_score": 0.0, "mark_type": "none", "win_odds": 99.0, "ev_win": 9.9})
    return out


def _single(odds):
    return {"entries": [{"horse_number": i, "win_odds": o, "place_odds_min": max(o * 0.3, 1.1), "place_odds_max": max(o * 0.45, 1.3)} for i, o in enumerate(odds, 1)]}


def _pairs(win_odds):
    return PZ.synth_pair_odds({i: o for i, o in enumerate(win_odds, 1)})


class _Storage:
    def __init__(self, docs=None):
        self.docs = docs or {}
        self.loaded: list[str] = []

    def load(self, category, key):
        self.loaded.append(category)
        return self.docs.get(category)


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.delenv("KEIBA_PAID_ZONE", raising=False)
    monkeypatch.delenv("KEIBA_PAID_ZONE_MODE", raising=False)
    PZ.reload()
    yield
    PZ.reload()


def _run(mode, storage=None, predicted=True, body=None):
    monkey_env = {"KEIBA_PAID_ZONE_MODE": mode}
    st = storage or _Storage({
        "race_odds": {"phase": "t15", "entries": _single(ACTUAL_WIN_ODDS)["entries"]},
        "race_pair_odds": {"phase": "t15", **_pairs(ACTUAL_WIN_ODDS)},
    })
    pso = _single([3.4, 5.0, 6.5, 8.0, 11.0, 14.0, 26.0, 36.0]) if predicted else None
    with patch.dict("os.environ", monkey_env), PZ_reload(), patch.object(PZ, "thresholds", lambda m=None: [0.1, 0.2, 0.3, 0.4]), \
         patch("src.scraper.storage.HybridStorage", lambda: st), \
         patch("src.pipeline.inference.race_prediction_service.load_cached", return_value={"predictions": _preds()}), \
         patch("src.pipeline.inference.odds_source.predicted_single_odds", return_value=pso):
        payload, code = delegates.optimize_betting({"race_id": RID, **(body or {})}, view="paid")
    return payload, code, st


class PZ_reload:
    def __enter__(self):
        PZ.reload()

    def __exit__(self, *a):
        PZ.reload()


def test_level_for_roi_thresholds():
    assert PZ.level_for_roi(None) == 0 and PZ.level_for_roi(0.5, "C") == 0
    assert PZ.level_for_roi(0.97, "C") == 1 and PZ.level_for_roi(1.0, "C") == 2
    assert PZ.level_for_roi(1.03, "C") == 3 and PZ.level_for_roi(1.5, "C") == 4
    assert PZ.thresholds("A")[0] > PZ.thresholds("C")[0]     # A は確信度の尺度（C と水準が違う）
    assert [t["label"] for t in PZ.tiers()][:2] == ["推奨しない", "元返し程度"]


def test_synth_pair_odds_counts_and_floor():
    o = PZ.synth_pair_odds({i: x for i, x in enumerate(ACTUAL_WIN_ODDS, 1)})
    assert len(o["umaren"]) == 28 and len(o["wide"]) == 28 and len(o["umatan"]) == 56 and len(o["sanrenpuku"]) == 56
    assert all(e["odds"] >= 1.0 for e in o["umaren"] + o["umatan"] + o["sanrenpuku"])
    assert PZ.synth_pair_odds({1: 2.0, 2: 3.0}) == {}


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_paid_payload_has_no_numbers_that_reveal_odds(mode):
    payload, code, _ = _run(mode)
    assert code == 200 and payload["view"] == "paid"
    assert PZ.find_forbidden_keys(payload) == []
    assert 0 <= payload["race_tier"]["level"] <= 4
    for b in payload["bets"]:
        assert set(b) >= {"bet_type", "combo", "role", "units", "yen", "legs"}
        assert b["role"] in ("main", "cover") and b["yen"] == b["units"] * 100
        assert {k for k in b} .isdisjoint({"odds", "ev", "prob", "bet_amount", "kelly_fraction"})
    assert payload["total_yen"] == sum(b["yen"] for b in payload["bets"])
    assert payload["disclaimer"] and payload["legend"][0]["label"] == "推奨しない"


def test_mode_a_uses_no_odds_at_all():
    st = _Storage()
    with patch("src.pipeline.inference.odds_source.predicted_single_odds", side_effect=AssertionError("A は想定オッズも使わない")):
        payload, code, _ = _run("A", storage=st, predicted=True)
    assert code == 200 and "race_odds" not in st.loaded and "race_pair_odds" not in st.loaded
    assert payload["bets"], "確率だけで買い目が出る"


def test_mode_b_never_reads_actual_odds():
    st = _Storage({"race_odds": {"entries": _single([1.1] * 8)["entries"]}, "race_pair_odds": {"umaren": []}})
    payload, code, st = _run("B", storage=st)
    assert code == 200 and "race_odds" not in st.loaded and "race_pair_odds" not in st.loaded
    assert PZ.find_forbidden_keys(payload) == []


def test_mode_b_falls_back_to_a_without_predicted_odds():
    payload, code, st = _run("B", predicted=False)
    assert code == 200 and "race_odds" not in st.loaded and payload["bets"]


def test_mode_c_reads_actual_odds_but_does_not_return_them():
    payload, code, st = _run("C")
    assert code == 200 and "race_odds" in st.loaded
    assert PZ.find_forbidden_keys(payload) == []


def test_cap_limits_total_and_drops_cover_first():
    full, _, _ = _run("B", body={"max_total_bet": 100000})
    small, code, _ = _run("B", body={"max_total_bet": 300})
    assert code == 200 and small["total_yen"] <= 300
    assert small["total_yen"] <= full["total_yen"]
    # 上限に収めるとき、本線を残して押さえから外す（本線が減るのは、押さえが無くなってから）
    n_main_full = len([b for b in full["bets"] if b["role"] == "main"])
    n_main_small = len([b for b in small["bets"] if b["role"] == "main"])
    n_cover_small = len([b for b in small["bets"] if b["role"] == "cover"])
    assert n_cover_small == 0 or n_main_small == n_main_full


def test_calibrate_single_expands_and_keeps_shape():
    s = {"entries": [{"horse_number": 1, "win_odds": 10.0, "place_odds_min": 2.0, "place_odds_max": 3.0},
                     {"horse_number": 2, "win_odds": 40.0, "place_odds_min": 5.0, "place_odds_max": 8.0}]}
    out = PZ.calibrate_single(s)
    assert out is not s and out["entries"][0]["horse_number"] == 1
    assert all(e["win_odds"] >= 1.0 and e["place_odds_max"] >= e["place_odds_min"] for e in out["entries"])
    assert PZ.calibrate_single(None) is None


def test_invalid_cap_and_missing_race():
    assert _run("B", body={"max_total_bet": 50})[1] == 400
    with patch("src.scraper.storage.HybridStorage", lambda: _Storage()):
        assert delegates.optimize_betting({}, view="paid")[1] == 400


def test_order_is_by_bet_type_then_probability_not_by_value():
    payload, _, _ = _run("B")
    kinds = [b["bet_type"] for b in payload["bets"]]
    assert kinds == sorted(kinds, key=PZ.BET_ORDER.index)


def test_race_values_paid_view_has_tier_not_roi():
    races = {"races": [{"race_id": RID, "round": 1, "venue": "東京", "race_name": "テスト"}]}
    st = _Storage({"race_lists": races})
    with patch("src.scraper.storage.HybridStorage", lambda: st), \
         patch("src.pipeline.inference.race_prediction_service.load_cached", return_value={"predictions": _preds()}), \
         patch("src.pipeline.inference.odds_source.predicted_single_odds", return_value=_single([3.4, 5.0, 6.5, 8.0, 11.0, 14.0, 26.0, 36.0])):
        delegates._VALUE_CACHE.clear()
        out = delegates.race_value_summary("20260601", view="paid", max_total_bet=2000)
    assert out["view"] == "paid" and "expected_roi" not in out["races"][0] and out["races"][0]["tier"] is not None
    assert PZ.find_forbidden_keys(out) == []
