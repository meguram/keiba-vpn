"""レース選択の強調表示用: 開催日の各レースの推奨期待回収率とレベル。"""

from __future__ import annotations

import pytest

from src.api.v1 import delegates as D


def test_value_level_thresholds(monkeypatch):
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: {"enabled": False})      # 較正が無効のときは従来の閾値
    assert D.value_level(1.6, 5) == "hot" and D.value_level(1.5, 1) == "hot"
    assert D.value_level(1.3, 5) == "good" and D.value_level(1.1, 5) == "ok"
    assert D.value_level(1.04, 5) == "pass" and D.value_level(2.0, 0) == "pass" and D.value_level(None, 3) == "pass"


def _payload(roi, n=3, total=2000):
    cands = [{"bet_type": "umaren", "pair_label": f"1-{i + 2}", "ev": 1.2 + 0.1 * i} for i in range(n)]
    return {"candidates": cands, "total_bet": total, "expected_return": roi * total, "odds_source": "actual", "odds_phase": "latest"}, 200


@pytest.fixture()
def fake(monkeypatch):
    rois = {"r01": 2.0, "r02": 1.8, "r03": 1.6, "r04": 1.55, "r05": 1.52, "r06": 1.3, "r07": 1.1, "r08": 0.9, "r09": None, "r10": 1.7}

    class S:
        def load(self, cat, key):
            return {"races": [{"race_id": r, "round": i + 1, "venue": "札幌", "race_name": r} for i, r in enumerate(rois)]}

    def opt(body):
        roi = rois[body["race_id"]]
        if roi is None:
            return {"error": "予測結果がありません"}, 404
        return _payload(roi, n=0 if body["race_id"] == "r08" else 3)

    monkeypatch.setattr("src.scraper.storage.HybridStorage", lambda *a, **k: S())
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: {"enabled": False})      # 従来の閾値・期待回収率で検査
    monkeypatch.setattr(D, "optimize_betting", opt)
    D._VALUE_CACHE.clear()
    return rois


def test_summary_levels_rank_and_hot_is_limited_to_top_fraction(fake):
    out = D.race_value_summary("20260906")
    by = {r["race_id"]: r for r in out["races"]}
    assert by["r09"]["level"] == "none" and by["r09"]["rank"] is None and "予測" in by["r09"]["reason"]
    assert by["r08"]["level"] == "pass"  # 買い目なし
    assert by["r07"]["level"] == "ok" and by["r06"]["level"] == "good"
    # 絶対水準 1.5 以上は 6 レース(r01,r02,r03,r04,r05,r10)。上位 20%（9 レース中 2 件）だけが hot
    hot = sorted(r["race_id"] for r in out["races"] if r["level"] == "hot")
    assert hot == ["r01", "r02"] and by["r03"]["level"] == "good" and by["r10"]["level"] == "good"
    assert by["r01"]["rank"] == 1 and by["r02"]["rank"] == 2 and by["r10"]["rank"] == 3
    assert by["r01"]["top_bet"]["bet_type"] == "umaren" and by["r01"]["max_ev"] == pytest.approx(1.4)
    assert [l["level"] for l in out["levels"]] == ["hot", "good", "ok"]


def test_summary_survives_a_failing_race_and_is_cached(fake, monkeypatch):
    calls = {"n": 0}
    real = D.optimize_betting

    def flaky(body):
        calls["n"] += 1
        if body["race_id"] == "r03":
            raise RuntimeError("boom")
        return real(body)

    monkeypatch.setattr(D, "optimize_betting", flaky)
    out = D.race_value_summary("20260906")
    assert {r["race_id"]: r["level"] for r in out["races"]}["r03"] == "none"
    n = calls["n"]
    assert D.race_value_summary("20260906") is out and calls["n"] == n  # TTL 内は再計算しない


def test_route_validates_params_and_requires_login(fake, monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")  # 自動ログイン(ローカル直アクセス)
    from src.api.flask_app import create_app

    c = create_app().test_client()
    assert c.get("/api/v1/betting/race-values").status_code == 400
    assert c.get("/api/v1/betting/race-values?date=2026").status_code == 400
    assert c.get("/api/v1/betting/race-values?date=20260906&max_total_bet=50").status_code == 400
    ok = c.get("/api/v1/betting/race-values?date=20260906", headers={"Host": "127.0.0.1:23110"})
    assert ok.status_code == 200 and len(ok.get_json()["races"]) == 10
    tunnel = c.get("/api/v1/betting/race-values?date=20260906", headers={"Host": "127.0.0.1:23110", "X-Forwarded-Host": "meguai-dev.tcpexposer.com"})
    assert tunnel.status_code == 401


def test_value_level_thresholds_calibrated(monkeypatch):
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: {"enabled": True, "levels": {"hot": 1.03, "good": 1.0, "ok": 0.97}})
    assert D.value_level(1.05, 3) == "hot" and D.value_level(1.0, 3) == "good" and D.value_level(0.98, 3) == "ok"
    assert D.value_level(0.96, 3) == "pass" and D.value_level(1.2, 0) == "pass" and D.value_level(None, 3) == "pass"


def test_calibrate_candidates_adds_fields(monkeypatch):
    cfg = {"enabled": True, "kinds": {"cand_tansho": {"method": "const", "value": 0.9, "validated": True},
                                      "cand_umaren": {"method": "const", "value": 1.0, "validated": False}}, "cap": {"ev": 1.3, "roi": 1.15}}
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: cfg)
    cands = [{"bet_type": "tansho", "ev": 2.0, "bet_amount": 300}, {"bet_type": "umaren", "ev": 3.0, "bet_amount": 100}]
    out = D._calibrate_candidates(cands)
    assert cands[0]["ev_calibrated"] == 0.9 and cands[0]["ev_calibration_validated"] is True
    assert cands[1]["ev_calibrated"] == 1.0 and cands[1]["ev_calibration_validated"] is False
    assert out["expected_roi_calibrated"] == pytest.approx(0.925, abs=1e-3)           # (0.9×300 + 1.0×100) / 400
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: {"enabled": False})
    assert D._calibrate_candidates([{"bet_type": "tansho", "ev": 2.0, "bet_amount": 300}]) == {}


def test_value_level_prob_calibrated_uses_pc_levels(monkeypatch):
    std = {"enabled": True, "levels": {"hot": 1.03, "good": 1.0, "ok": 0.97}}
    pc = {"enabled": True, "levels": {"hot": 9.99, "good": 0.949, "ok": 0.935}}
    monkeypatch.setattr("src.utils.ev_calibration.load_config", lambda: std)
    monkeypatch.setattr("src.utils.ev_calibration.load_config_pc", lambda: pc)
    assert D.value_level(0.96, 3) == "pass"                              # 従来の閾値（補正しないレース）では見送り
    assert D.value_level(0.96, 3, True) == "good"                        # 補正したレースは補正後の閾値
    assert D.value_level(1.5, 3, True) == "good"                         # hot は補正後は出さない
    assert D.value_level(0.93, 3, True) == "pass" and D.value_level(0.94, 3, True) == "ok"
    monkeypatch.setattr("src.utils.ev_calibration.load_config_pc", lambda: {})      # PC 版が無ければ従来の設定
    assert D.value_level(0.96, 3, True) == "pass"


def test_day_ranking_is_separate_for_calibrated_and_not():
    rows = [{"level": "hot", "expected_roi": 1.2, "prob_calibrated": False}, {"level": "hot", "expected_roi": 1.1, "prob_calibrated": False},
            {"level": "hot", "expected_roi": 0.99, "prob_calibrated": True}, {"level": "good", "expected_roi": 0.95, "prob_calibrated": True},
            {"level": "none", "expected_roi": None}]
    D._apply_day_ranking(rows)
    assert [r["rank"] for r in rows[:4]] == [1, 2, 1, 2]                 # 補正あり・なしで別々に順位を付ける
    assert rows[4]["rank"] is None
    plain = [{"level": "hot", "expected_roi": 1.3 - 0.1 * i} for i in range(5)]
    D._apply_day_ranking(plain)                                          # 補正の機能が無効（フラグ無し）なら従来どおり 1 組
    assert [r["rank"] for r in plain] == [1, 2, 3, 4, 5] and [r["level"] for r in plain].count("hot") == 1


def test_value_row_carries_prob_calibrated_flag():
    payload, code = _payload(0.96)
    payload["prob_calibration"] = {"applied": True}
    row = D._value_row({"race_id": "r", "round": 1, "venue": "x", "race_name": "y"}, payload, code)
    assert row["prob_calibrated"] is True
    plain, code = _payload(0.96)
    assert D._value_row({"race_id": "r"}, plain, code)["prob_calibrated"] is False


def test_value_row_marks_missing_prediction_as_waiting():
    race = {"race_id": "r", "round": 1, "venue": "x", "race_name": "y"}
    assert D._value_row(race, {"error": "予測結果がありません。先に推論を実行してください"}, 404)["reason"] == "予測待ち"
    assert D._value_row_paid(race, {"error": "x"}, 404)["reason"] == "予測待ち"
    assert D._value_row_paid(race, {"error": "x"}, 500)["reason"] == "計算できません"
