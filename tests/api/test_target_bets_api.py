"""推奨馬券の既定方策 target の API 層: 開発者・有料会員の応答、実オッズを出さないこと、旧方策への切り戻し、レース一覧の推奨度、結果の付与。外部接続なし。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.api.v1 import delegates
from src.pipeline.inference import target_bets as TB
from src.utils import paid_zone as PZ

RID = "202606010101"


def _preds(n=14, seed=2):
    rng = np.random.default_rng(seed)
    w = np.sort(rng.dirichlet(np.ones(n) * 0.8))[::-1]
    return [{"horse_number": i + 1, "horse_name": f"馬{i + 1}", "horse_id": f"h{i + 1}", "win_prob": float(w[i]), "top2_prob": float(min(w[i] * 1.9, 0.9)),
             "top3_prob": float(min(w[i] * 2.7, 0.95)), "pred_score": 0.0, "mark_type": "none", "win_odds": 99.0, "ev_win": 9.9} for i in range(n)]


def _single(preds, seed=5, noise=0.4):
    rng = np.random.default_rng(seed)
    w = np.array([p["win_prob"] for p in preds])
    pm = w * np.exp(rng.normal(0, noise, len(w)))
    pm /= pm.sum()
    wo = 0.8 / pm
    return {"entries": [{"horse_number": i + 1, "win_odds": float(wo[i]), "place_odds_min": float(max(wo[i] * 0.3, 1.1)), "place_odds_max": float(max(wo[i] * 0.45, 1.3))}
                        for i in range(len(wo))]}


class _Storage:
    def load(self, category, key):
        return None


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "target")
    monkeypatch.delenv("KEIBA_PAID_ZONE", raising=False)
    TB.reload()
    PZ.reload()
    yield
    TB.reload()
    PZ.reload()


def _call(view, body=None, preds=None, single="auto"):
    preds = preds or _preds()
    sg = _single(preds) if single == "auto" else single
    with patch("src.scraper.storage.HybridStorage", lambda: _Storage()), \
         patch("src.pipeline.inference.race_prediction_service.load_cached", return_value={"predictions": preds}), \
         patch.object(delegates, "_predicted_singles", return_value=sg):
        return delegates.optimize_betting({"race_id": RID, **(body or {})}, view=view)


@pytest.mark.parametrize("view", ["paid", "full"])
def test_target_payload_for_paid_and_developer_is_the_same_content(view):
    payload, code = _call(view, {"max_total_bet": 2000})
    assert code == 200 and payload["strategy"] == "target" and payload["view"] == view
    assert payload["budget_yen"] == 2000 and payload["target_multiple"] == 5.0
    assert PZ.find_forbidden_keys(payload) == []                     # 実オッズ・期待値などの禁止キーなし
    from src.api.tier import ACTUAL_ODDS_KEYS

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    assert not (set(keys(payload)) & ACTUAL_ODDS_KEYS)               # tier.py が『実オッズ』として落とすキーも無い
    assert payload["race_tier"]["label"] in ("見送り", "標準", "注目", "高注目", "最注目")
    assert "保証" in payload["disclaimer"] and "想定オッズ" in payload["note"]
    if payload["status"] == "bet":
        assert payload["total_yen"] == sum(b["yen"] for b in payload["bets"]) <= 2000
        assert all(b["bet_type"] == "sanrenpuku" and {"yen", "est_multiple", "est_payout_yen", "hit_rate_pct", "legs"} <= set(b) for b in payload["bets"])


def test_no_cap_plans_with_10000_yen_and_cap_below_100_is_rejected():
    payload, code = _call("paid", {"max_total_bet": None})
    assert code == 200 and payload["budget_yen"] == 10000 and payload["max_total_bet"] is None
    payload, code = _call("paid", {"max_total_bet": 50})
    assert code == 400 and "100 円以上" in payload["error"]


def test_default_cap_is_the_ui_default_2000():
    payload, code = _call("paid")
    assert code == 200 and payload["budget_yen"] == delegates.DEFAULT_MAX_TOTAL_BET == 2000


def test_missing_predictions_and_missing_race_id():
    with patch("src.scraper.storage.HybridStorage", lambda: _Storage()), \
         patch("src.pipeline.inference.race_prediction_service.load_cached", return_value=None), \
         patch("src.api.local_predictions.load_local_prediction_payload", return_value=None):
        p, c = delegates.optimize_betting({"race_id": RID}, view="paid")
    assert c == 404
    assert delegates.optimize_betting({}, view="paid")[1] == 400


def test_legacy_switch_restores_the_old_response(monkeypatch):
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "legacy")
    TB.reload()
    payload, code = _call("paid")
    assert code == 200 and payload.get("strategy") != "target" and "bets" in payload


def test_result_attachment_uses_est_multiple_when_payoff_is_missing():
    payload, _ = _call("paid", {"max_total_bet": 2000})
    if payload["status"] != "bet":
        pytest.skip("この乱数では見送り")
    a, b, c = (payload["bets"][0]["combo"] + [99, 98, 97])[:3]
    res = {"status": "provisional", "official": False, "finish": {a: 1, b: 2, c: 3}, "payoff": {}, "popularity": {}, "odds": {}, "n_runners": 14}
    with patch("src.pipeline.inference.prediction_results.load_race_result", return_value=res):
        out = delegates.attach_bet_results(payload, RID, storage=_Storage())
    assert out["result"]["n_bets"] == len(out["bets"]) and out["result"]["invested"] == payload["total_yen"]
    assert all("result" in x for x in out["bets"])


def test_race_value_rows_are_tiers_for_both_views():
    races = [{"race_id": f"2026060101{i:02d}", "round": i, "venue": "東京", "race_name": f"R{i}"} for i in range(1, 4)]

    class _St(_Storage):
        def load(self, category, key):
            return {"races": races} if category == "race_lists" else None

    delegates._VALUE_CACHE.clear()
    with patch("src.scraper.storage.HybridStorage", lambda: _St()), \
         patch("src.pipeline.inference.race_prediction_service.load_cached", return_value={"predictions": _preds()}), \
         patch.object(delegates, "_predicted_singles", return_value=_single(_preds())):
        for view in ("paid", "full"):
            delegates._VALUE_CACHE.clear()
            out = delegates.race_value_summary("20260601", view=view, max_total_bet=2000)
            assert out["strategy"] == "target" and len(out["races"]) == 3
            for r in out["races"]:
                assert r["basis"] == "target" and r["tier"]["label"] and r["level"] in ("hot", "good", "ok", "pass")
                assert not (set(r) & {"expected_roi", "max_ev", "top_bet"})


# ── 想定オッズの基準(sec77)の固定: 検証と別の想定オッズで推奨を作らない ──
def test_require_sec77_defaults_on_only_for_stg_and_prod(monkeypatch):
    monkeypatch.delenv("KEIBA_TARGET_REQUIRE_SEC77", raising=False)
    for env, want in (("stg", True), ("prod", True), ("dev", False), ("", False)):
        monkeypatch.setenv("KEIBA_ENV", env)
        assert TB.require_sec77() is want
    monkeypatch.setenv("KEIBA_ENV", "prod")
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "0")
    assert TB.require_sec77() is False
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "1")
    assert TB.require_sec77() is True


@pytest.mark.parametrize("view", ["paid", "full"])
@pytest.mark.parametrize("odds_model", ["legacy", None])
def test_without_sec77_odds_no_bets_are_recommended_when_required(monkeypatch, view, odds_model):
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "1")
    preds = _preds()
    sg = None if odds_model is None else {**_single(preds), "odds_model": odds_model}
    payload, code = _call(view, {"max_total_bet": 2000}, preds=preds, single=sg)
    assert code == 200 and payload["status"] == "skip" and payload["level"] == 0 and payload["bets"] == []
    assert payload["odds_pending"] is True and payload["odds_basis"] == "pending" and payload["skip_reason"]


def test_with_sec77_odds_bets_are_recommended_when_required(monkeypatch):
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "1")
    preds = _preds()
    payload, code = _call("full", {"max_total_bet": 2000}, preds=preds, single={**_single(preds), "odds_model": "sec77"})
    assert code == 200 and not payload.get("odds_pending") and payload.get("odds_basis") in (None, "sec77")


def test_predicted_singles_does_not_compute_legacy_when_required(monkeypatch):
    from src.pipeline.inference import final_odds_service

    called = []
    monkeypatch.setattr(final_odds_service, "get_or_compute", lambda *a, **k: called.append(1))
    monkeypatch.setattr("src.pipeline.inference.odds_source.predicted_single_odds", lambda storage, rid: None)
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "1")
    assert delegates._predicted_singles(_Storage(), RID) is None and not called
    monkeypatch.setenv("KEIBA_TARGET_REQUIRE_SEC77", "0")
    delegates._predicted_singles(_Storage(), RID)
    assert called


def test_response_has_only_the_default_plan_with_at_most_five_patterns():
    for seed in range(2, 12):
        preds = _preds(seed=seed)
        payload, code = _call("full", {"max_total_bet": 2000}, preds=preds, single={**_single(preds, seed=seed), "odds_model": "sec77"})
        assert code == 200
        assert [p["id"] for p in payload.get("plans") or []] in ([], [payload["default_plan"]])
        if payload["status"] == "bet":
            assert 1 <= len(payload["bets"]) <= 5 and [b["priority"] for b in payload["bets"]] == list(range(1, len(payload["bets"]) + 1))
            return
    pytest.skip("この乱数では見送り")
