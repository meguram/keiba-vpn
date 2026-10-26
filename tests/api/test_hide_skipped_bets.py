from src.api.v1 import delegates as D


def _target(level, status="bet"):
    plan = {"id": "p", "status": "bet", "bets": [{"combo": [1, 2, 3], "yen": 500}], "level": 3}
    return {"strategy": "target", "view": "full", "status": status, "level": level, "skip_reason": None, "bets": [{"combo": [1, 2, 3], "yen": 500}],
            "total_yen": 500, "any_hit_rate_pct": 8.0, "portfolio_ev": 2.1, "horse_marks": {"1": "honmei"}, "plans": [plan, dict(plan, id="q")],
            "race_tier": {"level": level, "label": "x"}, "max_bets": 1}


def test_target_skip_returns_no_bets_in_any_plan():
    p = D.hide_skipped_bets(_target(0))
    assert p["skipped"] is True and p["status"] == "skip" and p["bets"] == [] and p["plans"] == [] and p["total_yen"] == 0
    assert "any_hit_rate_pct" not in p and "portfolio_ev" not in p and "horse_marks" not in p and p["skip_reason"]
    assert p["race_tier"]["level"] == 0                                   # 推奨度そのものは残す


def test_target_standard_or_above_is_untouched():
    p = D.hide_skipped_bets(_target(1))
    assert "skipped" not in p and len(p["bets"]) == 1 and len(p["plans"]) == 2 and p["total_yen"] == 500


def test_paid_view_skip_by_race_tier():
    p = {"view": "paid", "race_tier": {"level": 0}, "bets": [{"combo": [1, 2]}]}
    assert D.hide_skipped_bets(p)["bets"] == []
    q = {"view": "paid", "race_tier": {"level": 2}, "bets": [{"combo": [1, 2]}]}
    assert D.hide_skipped_bets(q)["bets"] == [{"combo": [1, 2]}]


def test_full_view_skip_from_calibrated_roi():
    low = {"candidates": [{"bet_type": "tansho"}], "total_bet": 1000, "expected_return": 900, "expected_roi_calibrated": 0.9, "roi_pct": 90.0}
    out = D.hide_skipped_bets(dict(low))
    assert out["candidates"] == [] and out["total_bet"] == 0 and out["expected_return"] == 0 and "roi_pct" not in out and out["skipped"] is True
    high = {"candidates": [{"bet_type": "tansho"}], "total_bet": 1000, "expected_return": 1200, "expected_roi_calibrated": 1.2}
    assert "skipped" not in D.hide_skipped_bets(dict(high)) and len(high["candidates"]) == 1


def test_optimize_betting_hides_skips_but_internal_calls_keep_raw(monkeypatch):
    monkeypatch.setattr(D, "_optimize_betting", lambda body, **kw: (_target(0), 200))
    assert D.optimize_betting({"race_id": "r"})[0]["bets"] == []
    assert len(D.optimize_betting({"race_id": "r"}, hide_skipped=False)[0]["bets"]) == 1
    assert len(D.optimize_betting({"race_id": "r"}, _ctx={})[0]["bets"]) == 1


def test_attach_bet_results_does_not_add_result_to_skipped_race():
    p = D.hide_skipped_bets(_target(0))
    assert "result" not in D.attach_bet_results(p, "r")
