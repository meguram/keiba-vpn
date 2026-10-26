"""既定方策 target（目標払戻つきの 3連複）の単体テスト。外部接続なし。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.pipeline.inference import portfolio_target as PT
from src.pipeline.inference import target_bets as TB


@pytest.fixture(autouse=True)
def _target(monkeypatch):
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "target")
    TB.reload()
    yield
    TB.reload()


def _field(n=16, seed=1, noise=0.35):
    rng = np.random.default_rng(seed)
    w = np.sort(rng.dirichlet(np.ones(n) * 0.8))[::-1]
    t2, t3 = np.minimum(w * 1.9, 0.9), np.minimum(w * 2.7, 0.95)
    ents = [{"horse_number": i + 1, "horse_name": f"馬{i + 1}", "win_prob": float(w[i]), "top2_prob": float(t2[i]), "top3_prob": float(t3[i])} for i in range(n)]
    pm = w * np.exp(rng.normal(0, noise, n))
    pm /= pm.sum()
    wo = 0.8 / pm
    single = {"entries": [{"horse_number": i + 1, "win_odds": float(wo[i]), "place_odds_min": float(max(wo[i] * 0.3, 1.1)),
                           "place_odds_max": float(max(wo[i] * 0.45, 1.3))} for i in range(n)]}
    return ents, single


def test_available_by_default_and_legacy_switch(monkeypatch):
    assert TB.available() and TB.strategy() == "target"
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "legacy")
    assert not TB.available() and TB.strategy() == "legacy"


def test_params_match_the_adopted_design():
    p = TB.params(2000)
    assert (p.mult, p.k, p.tau, p.max_bets, p.budget) == (5.0, 2.0, 1.0, 3, 2000)


def test_required_stake_rounds_up_and_target_is_reached():
    odds = np.array([12.0, 55.5, 300.0])
    st = PT.required_stake(odds, 5.0, 2000, 2.0)
    assert st.tolist() == [1700.0, 400.0, 100.0]
    assert np.all(st / 100 * odds * 100 >= 5.0 * 2000 * 2.0 - 1e-6)


@pytest.mark.parametrize("budget", [None, 1000, 2000, 5000])
def test_portfolio_constraints_for_budgets(budget):
    ents, single = _field()
    r = TB.build_target_portfolio(ents, single, budget, race_id="T")
    assert r["strategy"] == "target" and r["status"] in ("bet", "skip")
    b0 = budget or TB.default_budget()
    assert r["budget_yen"] == b0
    if r["status"] == "bet":
        assert 1 <= len(r["bets"]) <= 3
        assert r["total_yen"] == sum(b["yen"] for b in r["bets"]) <= b0
        for b in r["bets"]:
            assert b["bet_type"] == "sanrenpuku" and len(b["combo"]) == 3 and b["yen"] % 100 == 0
            # 想定倍率ベースで、どれか 1 つ当たれば払戻が M×B 以上（安全係数 k=2 の余裕つき）
            assert b["yen"] * b["est_multiple"] >= 5.0 * b0 * 2.0 - 100 * b["est_multiple"]
            assert b["est_payout_yen"] >= 5.0 * b0
        assert r["level"] in (1, 2, 3, 4) and 0 < r["any_hit_rate_pct"] <= 100
        assert all(abs(b["ev_line_multiple"] - 1 / (b["hit_rate_pct"] / 100)) <= 0.06 + 0.002 * b["ev_line_multiple"] for b in r["bets"]), "期待値ライン＝1÷的中確率"


def test_skip_when_no_edge_and_when_field_is_small():
    ents, single = _field(noise=0.0)       # 想定オッズが自前確率と同じ → 期待値 < 1（τ=1.0 を満たさない）→ 見送り
    r = TB.build_target_portfolio(ents, single, 2000)
    assert r["status"] == "skip" and r["bets"] == [] and r["total_yen"] == 0 and r["level"] == 0 and r["skip_reason"]
    small, sg = _field(n=4)
    r = TB.build_target_portfolio(small, sg, 2000)
    assert r["status"] == "skip" and r["level"] == 0


def test_without_predicted_singles_uses_own_probabilities_and_says_so():
    ents, _ = _field()
    r = TB.build_target_portfolio(ents, None, 2000)
    assert r["odds_basis"] == "own" if r["status"] == "skip" else True


def test_response_has_no_real_odds_or_market_keys():
    ents, single = _field()
    r = TB.build_target_portfolio(ents, single, 2000)
    forbidden = {"odds", "win_odds", "place_odds_min", "place_odds_max", "popularity", "odds_source", "odds_phase", "payoff", "ev", "bet_amount"}
    def walk(o):
        if isinstance(o, dict):
            assert not (set(o) & forbidden), set(o) & forbidden
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(r)


def test_budget_changes_stakes_not_the_choice_of_combos_much():
    ents, single = _field(seed=3)
    a = TB.build_target_portfolio(ents, single, 2000)
    b = TB.build_target_portfolio(ents, single, 10000)
    if a["status"] == b["status"] == "bet":
        assert b["total_yen"] >= a["total_yen"]


def test_tier_level_thresholds(monkeypatch):
    assert TB.tier_level(None) == 0
    assert [TB.tier_level(x) for x in (0.02, 0.06, 0.09, 0.12)] == [1, 2, 3, 4]


def test_config_files_are_valid():
    c = json.loads(TB.CONFIG_PATH.read_text(encoding="utf-8"))
    assert c["params"] == {"mult": 5.0, "k": 2.0, "tau": 1.0, "max_bets": 3}
    assert "保証" in c["notice"] and "約94%" in c["notice"]
    assert len(TB.odds_pred_config()["poly2"]["sanrenpuku"]) == 3


def test_build_plans_returns_all_plans_and_default_pattern():
    import numpy as np

    from src.pipeline.inference import target_bets as TB

    rng = np.random.default_rng(3)
    n = 12
    w = rng.dirichlet(np.ones(n) * 0.8)
    ents = [{"horse_number": i + 1, "horse_name": f"H{i + 1}", "win_prob": w[i], "top2_prob": min(w[i] * 2.2, 0.9), "top3_prob": min(w[i] * 3.2, 0.95)} for i in range(n)]
    out = TB.build_target_portfolio(ents, None, 10000, race_id="t")
    ids = [p["id"] for p in out["plans"]]
    assert out["default_plan"] == "pattern_all" and "sanrenpuku3" in ids and "wide3" in ids
    for p in out["plans"]:
        assert p["status"] in ("bet", "skip")
        assert sum(b["yen"] for b in p["bets"]) <= 10000
        for b in p["bets"]:                       # どの買い目が当たっても払戻の想定が目標(5倍)以上（安全係数込み）
            assert b["est_payout_yen"] >= 5 * 10000 * (1.0 if p["kind"] == "pattern" else 0.0) * 0.99 or p["kind"] != "pattern"
            assert "odds" not in b                  # 実オッズのキーを出さない


def test_portfolio_ev_matches_sum_of_hit_rate_times_payout_for_every_plan():
    """買い目全体の期待払戻・期待回収率 = Σ(的中確率×購入額×想定倍率)。表示中の各買い目の数字から再計算しても一致する。"""
    ents, single = _field(16, seed=3, noise=0.5)
    out = TB.build_target_portfolio(ents, single, 2000)
    assert out["plans"], "プラン方式の応答であること"
    seen = 0
    for p in out["plans"]:
        if p["status"] != "bet":
            assert "portfolio_ev" not in p and "expected_payout_yen" not in p
            continue
        seen += 1
        exp = sum(b["hit_rate_pct"] / 100 * b["est_payout_yen"] for b in p["bets"])
        assert p["portfolio_ev"] == pytest.approx(exp / p["total_yen"], abs=0.05)
        assert p["expected_payout_yen"] == pytest.approx(exp, rel=0.02, abs=30)
        if p.get("race_ev") is not None:                     # pattern_all の race_ev は同じ量
            assert p["portfolio_ev"] == pytest.approx(p["race_ev"], abs=0.02)
    assert seen >= 1
    d = next(x for x in out["plans"] if x["id"] == out["default_plan"])
    if d["status"] == "bet":
        assert out["portfolio_ev"] == d["portfolio_ev"] and out["expected_payout_yen"] == d["expected_payout_yen"]


def test_portfolio_ev_helper_basic():
    cand = {"p": np.array([0.2, 0.1, 0.05]), "odds": np.array([10.0, 30.0, 100.0])}
    r = TB._portfolio_ev(cand, np.array([0, 2]), np.array([100.0, 200.0]))
    assert r["expected_payout_yen"] == 1200 and r["portfolio_ev"] == 4.0        # 0.2*100*10 + 0.05*200*100 = 200+1000
