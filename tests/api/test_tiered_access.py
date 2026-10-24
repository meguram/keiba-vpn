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

@pytest.fixture(autouse=True)
def _no_anti_scraping(monkeypatch):
    """区分ごとの出し分けを、スクレイピング対策（config/anti_scraping.json）の 401 と切り離して検証する。"""
    monkeypatch.setenv("KEIBA_ANTI_SCRAPING", "off")


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


# ── ゲスト / 無料ログイン / 有料ログイン / 開発者 の4区分（公開モード・公開許可 true）──
# 端末（PC・iPhone）には依存しない: 区分はセッション Cookie だけで決まる。無料ログインはコンテンツ上はゲストと同じ（ログイン済みの表示だけ違う）。

TIERS = ("guest", "free", "paid", "dev")
_DETAIL_FULL = {
    "race_id": "202608020311", "race_name": "R", "venue": "京都", "round": 11, "date": "20261024", "start_time": "15:40",
    "surface": "芝", "distance": 2000, "grade": "G1", "track_condition": "良", "direction": "右", "weather": "晴",
    "race_shutuba": {"race_id": "x", "entries": [{"horse_number": 1, "horse_name": "A", "bracket_number": 1, "sex_age": "牡4", "jockey_name": "J",
                                                  "jockey_weight": 56.0, "trainer_name": "T", "weight": 480, "weight_change": 2,
                                                  "odds": 3.0, "popularity": 1, "horse_id": "h"}]},
    "race_result": {"race_id": "x", "entries": [{"horse_number": 1, "horse_name": "A", "finish_position": 1, "odds": 3.0, "popularity": 1}]},
    "race_odds": {"entries": [{"horse_number": 1, "win_odds": 3.0}]}, "race_pair_odds": {}, "race_index": {}, "race_paddock": {},
}
_ODDS_KEYS = {"win_odds", "final_odds", "popularity", "ev_win", "odds_sum", "odds_n", "pop_sum", "pop_n", "w_odds_sum", "w_pop_sum",
              "avg_odds", "avg_popularity", "win_avg_odds", "win_avg_popularity", "candidates", "total_bet", "total_return", "odds", "ev"}


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture()
def tier_env(monkeypatch, tmp_path):
    """公開モード＋B・O の公開許可 true、予測成績の保存先を一時ディレクトリへ、free / paid の会員を用意。"""
    from src.pipeline.inference import prediction_results as pr
    from src.utils import public_policy as P
    from tests.pipeline.test_prediction_results import _storage

    monkeypatch.setenv("KEIBA_ENV", "stg")
    monkeypatch.setenv("DEV_SECRET_KEY", "k")
    monkeypatch.setenv("KEIBA_PUBLIC_MODE", "public")
    cfg = json.loads(P.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["release"].update(b_facts_public=True, odds_derived_public=True)
    f = tmp_path / "pf.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(P, "CONFIG_PATH", f)
    P.reload()
    monkeypatch.setenv("KEIBA_PREDICTION_RESULTS_DIR", str(tmp_path))
    recs = {"free1": {"plan": "free"}, "paid1": {"plan": "paid"}}
    monkeypatch.setattr(auth, "member_record", lambda uid: recs.get(uid))
    doc = pr.build_day("20261018", _storage())
    doc["date"] = "20261018"
    pr.write_day(doc, tmp_path)
    from src.api import prediction_results_api as api

    api._CACHE.clear()
    yield
    P.reload()


def _tier_cookies(tier: str) -> dict:
    if tier == "guest":
        return {}
    if tier == "free":
        return {auth.COOKIE_NAME: auth._make_token(role="member", uid="free1")}
    if tier == "paid":
        return {auth.COOKIE_NAME: auth._make_token(role="member", uid="paid1")}
    return {auth.COOKIE_NAME: auth._make_token()}


def _get(c, path, tier):
    return c.get(path, cookies=_tier_cookies(tier))


def test_auth_status_reports_four_tiers(tier_env):
    _, c = _fastapi()
    got = {t: _get(c, "/api/auth/status", t).json() for t in TIERS}
    assert [got[t]["tier"] for t in TIERS] == ["public", "free", "member", "dev"]
    assert [got[t]["logged_in"] for t in TIERS] == [False, True, True, True]
    assert [got[t]["is_member"] for t in TIERS] == [False, False, True, True]
    assert [got[t]["is_developer"] for t in TIERS] == [False, False, False, True]


def test_free_login_sees_the_same_content_as_guest(tier_env, monkeypatch):
    appmod, c = _fastapi()

    async def fake_pred(_rid):
        return JSONResponse(TOP)

    async def fake_detail(_rid):
        return JSONResponse(_DETAIL_FULL)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake_pred)
    monkeypatch.setattr(appmod, "_race_detail_impl", fake_detail)
    for path in ("/api/race/202608020311/predictions", "/api/race/202608020311", "/api/prediction-results?scope=day&date=20261018",
                 "/api/prediction-results/news", "/api/prediction-results/mark-card", "/api/prediction-results/days"):
        g, f = _get(c, path, "guest"), _get(c, path, "free")
        assert g.status_code == f.status_code == 200, path
        g_json, f_json = g.json(), f.json()
        for body in (g_json, f_json):
            for volatile in ("built_at",):
                body.pop(volatile, None)
        assert g_json == f_json, path


def test_predictions_by_four_tiers(tier_env, monkeypatch):
    appmod, c = _fastapi()

    async def fake(_rid):
        return JSONResponse(TOP)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake)
    out = {t: _get(c, "/api/race/202608020311/predictions", t).json() for t in TIERS}
    guest_locked = {k for ks in LOCKED_GROUPS.values() for k in ks}
    for t in ("guest", "free"):
        keys = set(out[t]) | {k for e in out[t]["predictions"] for k in e}
        assert not (guest_locked & keys), (t, guest_locked & keys)
        assert {"win_prob", "top3_prob", "show_prob", "horse_name"} <= keys
    pe = out["paid"]["predictions"][0]
    assert {"ev_win_cal", "mark_type", "top2_prob"} <= set(pe)                       # 有料: 較正後の期待値・印・連対率
    assert not ({"ev_win", "ev_place", "expected_value", "popularity", "top2_odds_est", "odds_source"} & set(pe))     # 実オッズ（と生の期待値）は出さない
    assert pe.get("win_odds") != ENTRY["win_odds"] and "bet_suggestion" not in out["paid"]
    assert "horse_id" not in pe                                                       # 取得元固有（C）は誰にも出さない
    assert out["dev"]["predictions"][0]["ev_win"] == 1.2 and "bet_suggestion" in out["dev"]   # 開発者: 全項目


def test_race_detail_never_exposes_market_odds_except_developer(tier_env, monkeypatch):
    appmod, c = _fastapi()

    async def fake(_rid):
        return JSONResponse(_DETAIL_FULL)

    monkeypatch.setattr(appmod, "_race_detail_impl", fake)
    for t in ("guest", "free", "paid"):
        d = _get(c, "/api/race/202608020311", t).json()
        assert d["race_shutuba"]["entries"][0]["horse_name"] == "A", t                # 事実は全員に出る
        assert "race_odds" not in d and not ({"odds", "popularity", "horse_id"} & set(_walk_keys(d))), t
    assert "race_odds" in _get(c, "/api/race/202608020311", "dev").json()


def test_member_only_paths_by_four_tiers(tier_env):
    _, c = _fastapi()
    for path in ("/api/race/202608020311/tracking-difficulty", "/api/race-quality/day?date=20261024", "/api/prediction-results/bets-summary?date=20261018"):
        for t in ("guest", "free"):
            r = _get(c, path, t)
            assert r.status_code == 401 and r.json()["reason"] == "member_required", (path, t)
        for t in ("paid", "dev"):
            assert _get(c, path, t).status_code != 401, (path, t)


def test_prediction_results_hides_odds_and_bets_from_guest_and_free(tier_env):
    _, c = _fastapi()
    for path in ("/api/prediction-results?scope=day&date=20261018", "/api/prediction-results?scope=week&date=20261018",
                 "/api/prediction-results/news", "/api/prediction-results/mark-card", "/api/prediction-results/days"):
        for t in ("guest", "free"):
            body = _get(c, path, t).json()
            assert not (_ODDS_KEYS & set(_walk_keys(body))), (path, t, _ODDS_KEYS & set(_walk_keys(body)))
    full = _get(c, "/api/prediction-results?scope=day&date=20261018", "dev").json()
    assert full["kpi"]["invested"] > 0 and full["kpi"]["roi"] is not None and "bets" in full["races"][0]
    g = _get(c, "/api/prediction-results?scope=day&date=20261018", "guest").json()
    assert g["kpi"]["roi"] is None and g["kpi"]["invested"] == 0 and g["kpi"]["by_type"] == {} and g["kpi"]["honmei"] == full["kpi"]["honmei"]
    assert "bets" not in g["races"][0] and g["summary"]["bets"] == {"invested": 0, "returned": 0, "races_bought": 0, "races_hit": 0, "estimated_races": 0, "by_type": {}}
    assert g["summary"]["bets"]["invested"] == 0 and all(pt.get("roi") is None for pt in g["series"]) and "roi" not in set(_walk_keys(g["series"]))
    assert {h["kind"] for h in g["highlights"]} <= {"big_miss"} and {h["kind"] for h in full["highlights"]} - {"big_miss"}
    assert g["races"][0]["marks"]["star"][0]["horse_name"] == full["races"][0]["marks"]["star"][0]["horse_name"]   # 予測と着順の突き合わせは出る


def test_prediction_results_news_for_guest_has_only_public_text(tier_env):
    _, c = _fastapi()
    full = _get(c, "/api/prediction-results/news?limit=10", "dev").json()
    assert any("回収率" in i["subtitle"] for i in full["items"]) and any("単勝" in i["subtitle"] for i in full["items"])
    for t in ("guest", "free"):
        items = _get(c, "/api/prediction-results/news?limit=10", t).json()["items"]
        assert items and {i["kind"] for i in items} <= {"day", "week", "big_miss"}
        for i in items:
            assert "roi" not in i and i["subtitle"] == i["subtitle_public"]
            assert not any(w in i["subtitle"] for w in ("回収率", "単勝", "倍", "円")), i["subtitle"]


def test_redact_prediction_results_does_not_mutate_input(tier_env):
    from src.api.tier import redact_prediction_results

    doc = {"scope": "day", "kpi": {"roi": 1.5, "invested": 100, "returned": 150, "by_type": {"tansho": {}}, "honmei": {"n": 1}},
           "summary": {"marks": {"honmei": {"n": 1, "odds_sum": 3.0}}, "bets": {"invested": 100}}, "series": [{"date": "d", "roi": 1.5}],
           "highlights": [{"kind": "ana_hit", "text": "単勝19倍"}], "races": [{"race_id": "r", "bets": {"total_bet": 100}, "top1": {"win_odds": 3, "horse_name": "A"}}]}
    before = json.dumps(doc, sort_keys=True)
    out = redact_prediction_results("/api/prediction-results", doc)
    assert json.dumps(doc, sort_keys=True) == before
    assert out["races"] == [{"race_id": "r", "top1": {"horse_name": "A"}}] and out["highlights"] == [] and out["kpi"]["roi"] is None


def test_flask_betting_requires_paid_not_free(tier_env, monkeypatch):
    monkeypatch.setenv("DEV_PASSWORD", "devpw")
    from src.api.flask_app import create_app

    def client(tier):
        c = create_app().test_client()
        for k, v in _tier_cookies(tier).items():
            try:
                c.set_cookie(k, v)
            except TypeError:
                c.set_cookie("localhost", k, v)
        return c

    for path in ("/api/v1/betting/race-values?date=20261024", "/api/v1/races/202608020311/final-odds"):
        for t in ("guest", "free"):
            assert client(t).get(path).status_code == 401, (path, t)
        for t in ("paid", "dev"):
            assert client(t).get(path).status_code != 401, (path, t)
    for t in ("guest", "free"):
        assert client(t).post("/api/v1/betting/optimize", json={}).status_code == 401, t
    for t in ("paid", "dev"):
        assert client(t).post("/api/v1/betting/optimize", json={}).status_code == 400, t
    st = {t: client(t).get("/api/v1/auth/status").get_json()["tier"] for t in TIERS}
    assert st == {"guest": "public", "free": "free", "paid": "member", "dev": "dev"}


def test_anti_scraping_keeps_guest_pages_open_and_the_rest_login_only(monkeypatch):
    """auth=enforce でも、ゲストが画面に使う API（出馬表・予測・予測成績・日付）は未ログインで開き、それ以外の /api/* は未ログインで 401 になる設定のまま。"""
    from src.utils import anti_scraping as asx

    monkeypatch.delenv("KEIBA_ANTI_SCRAPING", raising=False)
    eng = asx.AntiScraping(config_path=asx.CONFIG_PATH, env={})
    assert eng.mode("auth") == "enforce"
    for path in ("/api/race/202608020311", "/api/race/202608020311/predictions", "/api/race-list/20261024", "/api/race-calendar",
                 "/api/scrape-dates", "/api/prediction-results/news", "/api/prediction-results", "/api/auth/status", "/api/v1/auth/login",
                 "/api/health", "/api/maintenance/status"):
        assert eng._needs_auth(path) is False, path
    for path in ("/api/megu-index-dates", "/api/horse/2020100001/detail", "/api/v1/races", "/api/v1/data-analysis/query", "/api/race-quality/day"):
        assert eng._needs_auth(path) is True, path


# ── 実オッズ（市場オッズ）は開発者以外に出さない（actual_odds_public=false）──

_ACTUAL_ODDS_KEYS = {"win_odds", "final_odds", "popularity", "odds", "odds_sum", "odds_n", "pop_sum", "pop_n", "w_odds_sum", "w_odds_n", "w_pop_sum", "w_pop_n",
                     "avg_odds", "avg_popularity", "win_avg_odds", "win_avg_popularity", "odds_phase", "pair_odds", "single_odds", "ev_win"}


def test_prediction_results_paid_has_no_actual_odds_but_keeps_results(tier_env):
    _, c = _fastapi()
    for path in ("/api/prediction-results?scope=day&date=20261018", "/api/prediction-results?scope=week&date=20261018",
                 "/api/prediction-results/news?limit=10", "/api/prediction-results/mark-card"):
        paid, dev = _get(c, path, "paid").json(), _get(c, path, "dev").json()
        assert not (_ACTUAL_ODDS_KEYS & set(_walk_keys(paid))), (path, _ACTUAL_ODDS_KEYS & set(_walk_keys(paid)))
        assert _ACTUAL_ODDS_KEYS & set(_walk_keys(dev)) or path.endswith("mark-card") or "news" in path, path      # 開発者の応答は変えない
    full = _get(c, "/api/prediction-results?scope=day&date=20261018", "paid").json()
    assert full["kpi"]["invested"] > 0 and full["kpi"]["roi"] is not None            # 仮想購入の結果（金額・回収率）は有料に出る
    assert "bets" in full["races"][0] and not any("odds" in k for k in _walk_keys(full["races"]))
    news = _get(c, "/api/prediction-results/news?limit=10", "paid").json()["items"]
    assert any("回収率" in i["subtitle"] for i in news)
    assert not any("単勝" in (i["subtitle"] or "") for i in news)


def test_redact_prediction_results_member_does_not_mutate_and_strips_odds_phrases(tier_env):
    from src.api.tier import redact_prediction_results_member

    doc = {"items": [{"kind": "ana_hit", "subtitle": "新潟9R ★A（単勝19.7倍）が3着に", "subtitle_public": "新潟9R ★A（単勝19.7倍）が3着に", "roi": 1.2}],
           "highlights": [{"kind": "ana_hit", "value": 19.7, "text": "新潟9R ★A（単勝19.7倍）が3着に"}],
           "races": [{"top1": {"win_odds": 3.0, "popularity": 1, "final_odds": 2.9, "ev_win": 1.1, "horse_name": "A"},
                      "bets": {"total_bet": 100, "total_return": 300, "candidates": [{"bet_type": "tansho", "odds": 3.0, "bet_amount": 100}]}}]}
    before = json.dumps(doc, sort_keys=True)
    out = redact_prediction_results_member(doc)
    assert json.dumps(doc, sort_keys=True) == before
    assert "単勝" not in json.dumps(out, ensure_ascii=False) and out["items"][0]["roi"] == 1.2
    assert out["races"][0]["top1"] == {"horse_name": "A"} and out["races"][0]["bets"]["total_return"] == 300
    assert out["races"][0]["bets"]["candidates"] == [{"bet_type": "tansho", "bet_amount": 100}] and "value" not in out["highlights"][0]


def test_paid_predictions_get_predicted_odds_not_actual(tier_env, monkeypatch):
    """actual_odds_public=false の間、有料会員の単勝・複勝・想定人気・期待値は想定オッズ（最終オッズ予測のキャッシュ）から付く。"""
    appmod, c = _fastapi()

    async def fake(_rid):
        return JSONResponse(TOP)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake)
    import src.pipeline.inference.odds_source as OS

    from src.utils import paid_zone as PZ

    single = {"entries": [
        {"horse_number": 5, "win_odds": 12.0, "place_odds_min": 3.0, "place_odds_max": 4.0},
        {"horse_number": 6, "win_odds": 4.0, "place_odds_min": 1.5, "place_odds_max": 2.0}]}
    monkeypatch.setattr(OS, "predicted_single_odds", lambda _st, _rid: single)
    want = {e["horse_number"]: e["win_odds"] for e in PZ.calibrate_single(single)["entries"]}          # 有料ゾーンと同じ較正後の想定単勝
    paid = {e["horse_number"]: e for e in _get(c, "/api/race/202608020311/predictions", "paid").json()["predictions"]}
    assert paid[5]["win_odds"] == round(want[5], 1) and paid[6]["win_odds"] == round(want[6], 1) and paid[5]["odds_source"] == "predicted"
    assert paid[6]["popularity"] == 1 and paid[5]["popularity"] == 2                       # 想定人気（想定単勝の低い順）
    assert paid[5]["ev_win"] == round(ENTRY["win_prob"] * want[5], 3)                       # 想定オッズ × 勝率（実オッズ 8.5 ではない）
    assert ENTRY["win_odds"] not in [e["win_odds"] for e in paid.values()]
    for t in ("guest", "free"):
        assert not any("win_odds" in e for e in _get(c, "/api/race/202608020311/predictions", t).json()["predictions"]), t
    assert {e["horse_number"]: e["win_odds"] for e in _get(c, "/api/race/202608020311/predictions", "dev").json()["predictions"]} == {5: 8.5, 6: 8.5}


def test_actual_odds_never_in_member_responses_matrix(tier_env, monkeypatch):
    """4 区分 × 主要エンドポイント: 実オッズのキーは開発者にしか出ない。"""
    appmod, c = _fastapi()

    async def fake_pred(_rid):
        return JSONResponse(TOP)

    async def fake_detail(_rid):
        return JSONResponse(_DETAIL_FULL)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake_pred)
    monkeypatch.setattr(appmod, "_race_detail_impl", fake_detail)
    import src.pipeline.inference.odds_source as OS

    monkeypatch.setattr(OS, "predicted_single_odds", lambda _st, _rid: None)               # 想定オッズが無いレースでも実オッズへ落ちない
    actual = _ACTUAL_ODDS_KEYS | {"race_odds", "race_pair_odds", "top2_odds_est", "place_odds_min", "place_odds_max"}
    for path in ("/api/race/202608020311/predictions", "/api/race/202608020311", "/api/prediction-results?scope=day&date=20261018",
                 "/api/prediction-results/news", "/api/prediction-results/mark-card"):
        for t in ("guest", "free", "paid"):
            body = _get(c, path, t).json()
            assert not (actual & set(_walk_keys(body))), (path, t, actual & set(_walk_keys(body)))
    assert {"win_odds", "race_odds"} <= set(_walk_keys(_get(c, "/api/race/202608020311/predictions", "dev").json())) | set(_walk_keys(_get(c, "/api/race/202608020311", "dev").json()))


def test_fastapi_final_odds_is_member_only():
    assert member_only_path("/api/race/202608020311/final-odds") and member_only_path("/api/race/202608020311/final-odds/precompute")
