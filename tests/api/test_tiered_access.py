"""公開（匿名）と会員の出し分け: 匿名は勝率・複勝率まで。会員限定の項目はサーバ側で落とす。外部接続なし。"""

from __future__ import annotations

import json
import os

os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api import auth
from src.api.tier import LOCKED_GROUPS, PUBLIC_ENTRY_FIELDS, member_only_path, redact_predictions, redact_race_detail

ENTRY = {
    "pred_rank": 3, "horse_number": 5, "horse_name": "テスト", "horse_id": "2020100001", "bracket_number": 3, "sex_age": "牡4",
    "jockey_name": "騎手", "jockey_weight": 56.0, "trainer_name": "調教師", "weight": 480, "weight_change": 2,
    "win_prob": 0.12, "top3_prob": 0.4, "show_prob": 0.4,
    # ↓ 会員限定
    "top2_prob": 0.2, "place_prob": 0.2, "win_prob_model": 0.1, "top3_prob_model": 0.3,
    "win_odds": 8.5, "place_odds_min": 2.0, "place_odds_max": 3.0, "popularity": 4, "top2_odds_est": 20.0,
    "ev_win": 1.2, "ev_place": 1.1, "ev_top2": 1.0, "ev_win_cal": 1.1, "ev_place_cal": 1.0, "expected_value": 1.2,
    "mark_type": "honmei", "recommendation": "◎", "buy_tier": "S", "megu_mark": "x", "kesi_reason": "r",
    "pred_score": 1.0, "normalized_score": 0.5, "composite_score": 0.9, "composite_rank": 1,
    "position_forecast": {"x": 1}, "tracking_v2": {"ease_pct": 50}, "l3f_coherent": {"dev_sec": 0.1},
    "expected_megu": 90, "growth": {"x": 1}, "estimated_prob": 0.1, "prob_calibrated": True,
}
TOP = {
    "race_id": "202608020311", "race_name": "R", "venue": "京都", "round": 11, "surface": "芝", "distance": 2000,
    "track_condition": "良", "status": "success", "has_prediction": True, "total_horses": 2,
    "model_version": "v4", "prediction_source": "shadow", "entries_status": {"phase": "final"},
    "predictions": [ENTRY, {**ENTRY, "horse_number": 6, "horse_name": "B", "win_prob": 0.30, "pred_rank": 9}],
    "bet_suggestion": {"x": 1}, "prob_calibration": {"applied": True}, "pace_forecast": {}, "gate_bias": {}, "pace_type": {},
    "scenario_forecast": {}, "lap_forecast": {}, "expected_megu_meta": {}, "class_benchmark": {}, "growth_meta": {},
    "pace_unified": {}, "tracking_v2_meta": {}, "training_coverage": {}, "_compute_meta": {}, "ev_calibrated": True,
}


def test_redact_predictions_keeps_only_win_and_show_prob():
    out = redact_predictions(TOP)
    for e in out["predictions"]:
        assert set(e) <= set(PUBLIC_ENTRY_FIELDS)
        assert {"win_prob", "show_prob", "top3_prob"} <= set(e)
    locked = {k for ks in LOCKED_GROUPS.values() for k in ks}
    flat = set(out) | {k for e in out["predictions"] for k in e}
    assert not (locked & flat), locked & flat
    assert "bet_suggestion" not in out and "prob_calibration" not in out and "_compute_meta" not in out
    assert out["race_id"] == TOP["race_id"] and out["entries_status"] == {"phase": "final"}
    assert out["access"]["tier"] == "public"


def test_redact_predictions_ranks_by_win_prob_not_composite():
    out = redact_predictions(TOP)
    assert [e["horse_number"] for e in out["predictions"]] == [6, 5]
    assert [e["pred_rank"] for e in out["predictions"]] == [1, 2]


def test_redact_predictions_does_not_mutate_input_and_handles_no_prediction():
    before = json.dumps(TOP, sort_keys=True)
    redact_predictions(TOP)
    assert json.dumps(TOP, sort_keys=True) == before
    out = redact_predictions({"status": "not_found"})
    assert out["status"] == "not_found" and "predictions" not in out
    entries_only = redact_predictions({"has_prediction": False, "predictions": [{"horse_number": 1, "horse_name": "A", "win_odds": 3.0, "popularity": 1}]})
    assert entries_only["predictions"] == [{"horse_number": 1, "horse_name": "A"}]


def test_redact_race_detail_drops_odds_and_side_info_keeps_facts():
    detail = {
        "race_id": "x", "race_name": "R", "race_shutuba": {"entries": [{"horse_number": 1, "horse_name": "A", "odds": 3.0, "popularity": 1}]},
        "race_result": {"entries": [{"horse_number": 1, "odds": 3.0}]}, "race_odds": {"entries": []}, "race_pair_odds": {}, "race_index": {},
        "race_paddock": {}, "race_barometer": {}, "race_trainer_comment": {}, "smartrc_race": {}, "horses": {"a": {}},
        "data_availability": {"race_odds": True, "race_shutuba": True},
    }
    out = redact_race_detail(detail)
    for k in ("race_odds", "race_pair_odds", "race_index", "race_paddock", "race_barometer", "race_trainer_comment", "smartrc_race"):
        assert k not in out
    assert out["race_shutuba"]["entries"] == [{"horse_number": 1, "horse_name": "A"}]
    assert out["race_result"] == detail["race_result"] and out["horses"] == {"a": {}}
    assert out["data_availability"] == {"race_shutuba": True}
    assert detail["race_shutuba"]["entries"][0]["odds"] == 3.0  # 元は変えない


@pytest.mark.parametrize("path,expected", [
    ("/api/race/202608020311/tracking-difficulty", True),
    ("/api/race/202608020311/bundle", True),
    ("/api/race-quality/day", True),
    ("/api/v1/betting/optimize", True),
    ("/api/v1/betting/race-values", True),
    ("/api/v1/races/1/final-odds", True),
    ("/api/v1/races/1/megu-index-predicted", True),
    ("/api/race/202608020311/predictions", False),     # 中身を落として返す
    ("/api/race/202608020311", False),
    ("/api/growth-curve/2020100001", True),
    ("/api/growth-curve/status", False),
    ("/api/race-list/20261024", False),
    ("/api/v1/auth/status", False),
])
def test_member_only_path(path, expected):
    assert member_only_path(path) is expected


# ── 認証: 会員ロール ──

def test_member_token_is_not_developer():
    t = auth._make_token(role="member")
    assert auth.token_role(t) == "member" and auth._verify_token(t) is False
    assert auth.token_role(auth._make_token()) == "dev"
    assert auth.token_role("member:1:abc") is None and auth.token_role("admin:1:abc") is None


def test_test_user_is_off_by_default_and_hashed(monkeypatch):
    monkeypatch.delenv("KEIBA_TEST_USER", raising=False)
    assert auth.verify_member_login("test", "test") is False
    monkeypatch.setenv("KEIBA_TEST_USER", "1")
    assert auth.verify_member_login("test", "test") is True
    assert auth.verify_member_login("test", "wrong") is False
    assert auth.verify_member_login("nobody", "test") is False
    assert auth.verify_member_login("", "test") is False
    text = open(auth._MEMBERS_FILE, encoding="utf-8").read()
    assert '"password"' not in text and '"hash"' in text


# ── FastAPI / Flask の端から端 ──

def _fastapi():
    from src.api import app as appmod

    return appmod, TestClient(appmod.app, raise_server_exceptions=False)


def _cookie(role: str) -> dict:
    return {auth.COOKIE_NAME: auth._make_token(role=role)}


def test_fastapi_predictions_by_tier(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    appmod, c = _fastapi()

    async def fake(_rid):
        return JSONResponse(TOP)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake)
    anon = c.get("/api/race/202608020311/predictions").json()
    assert "ev_win" not in anon["predictions"][0] and "bet_suggestion" not in anon and anon["access"]["tier"] == "public"
    assert anon["predictions"][0]["win_prob"] == 0.30
    for role in ("member", "dev"):
        full = c.get("/api/race/202608020311/predictions", cookies=_cookie(role)).json()
        assert full["predictions"][0]["ev_win"] == 1.2 and "bet_suggestion" in full


def test_fastapi_member_only_paths_401_for_anonymous(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    _, c = _fastapi()
    for path in ("/api/race/202608020311/tracking-difficulty", "/api/race-quality/day?date=20261024", "/api/race/202608020311/bundle"):
        r = c.get(path)
        assert r.status_code == 401 and r.json()["reason"] == "member_required", path
        assert c.get(path, cookies=_cookie("member")).status_code != 401, path


def test_fastapi_auth_status_and_form_login(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    monkeypatch.setenv("KEIBA_TEST_USER", "1")
    _, c = _fastapi()
    assert c.get("/api/auth/status").json()["tier"] == "public"
    assert c.get("/api/auth/status", cookies=_cookie("member")).json()["tier"] == "member"
    assert c.get("/api/auth/status", cookies=_cookie("dev")).json()["is_developer"] is True
    assert c.get("/api/auth/status", cookies=_cookie("member")).json()["is_developer"] is False
    r = c.post("/login", data={"id": "test", "password": "test", "next": "/"}, follow_redirects=False)
    assert r.status_code == 303
    assert auth.token_role(r.cookies.get(auth.COOKIE_NAME)) == "member"
    bad = c.post("/login", data={"id": "test", "password": "nope", "next": "/"}, follow_redirects=False)
    assert bad.status_code == 200 and auth.COOKIE_NAME not in bad.cookies


def test_flask_login_and_member_gate(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    monkeypatch.setenv("KEIBA_TEST_USER", "1")
    monkeypatch.setenv("DEV_PASSWORD", "devpw")
    from src.api.flask_app import create_app

    c = create_app().test_client()
    st = c.get("/api/v1/auth/status").get_json()
    assert st["tier"] == "public" and st["is_member"] is False
    assert c.post("/api/v1/betting/optimize", json={"race_id": "x"}).status_code == 401
    assert c.get("/api/v1/betting/race-values?date=20261024").status_code == 401
    assert c.get("/api/v1/races/202608020311/final-odds").status_code == 401

    assert c.post("/api/v1/auth/login", json={"id": "test", "password": "bad"}).status_code == 401
    assert c.post("/api/v1/auth/login", json={"id": "test", "password": "test"}).status_code == 200
    st = c.get("/api/v1/auth/status").get_json()
    assert st["tier"] == "member" and st["is_member"] and st["is_developer"] is False
    assert c.post("/api/v1/betting/optimize", json={}).status_code == 400            # 認可は通り、race_id 検証で 400
    assert c.post("/api/v1/admin/git-pull").status_code == 401                         # 会員は開発者向け機能に入れない

    d = create_app().test_client()
    assert d.post("/api/v1/auth/login", json={"password": "devpw"}).status_code == 200
    assert d.get("/api/v1/auth/status").get_json()["tier"] == "dev"


def test_test_user_cannot_login_when_flag_off(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    monkeypatch.delenv("KEIBA_TEST_USER", raising=False)
    from src.api.flask_app import create_app

    assert create_app().test_client().post("/api/v1/auth/login", json={"id": "test", "password": "test"}).status_code == 401
