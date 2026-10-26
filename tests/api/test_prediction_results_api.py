import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import prediction_results_api as api
from src.pipeline.inference import prediction_results as pr
from tests.pipeline.test_prediction_results import _storage


def _build(tmp_path, monkeypatch, member: bool):
    monkeypatch.setenv("KEIBA_PREDICTION_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("KEIBA_ENV", "stg")                      # dev の都度計算を止める
    monkeypatch.setattr(api, "is_member", lambda _request: member)
    monkeypatch.setattr(api, "is_developer", lambda _request: member)      # member=True は開発者の視点（全項目）
    api._CACHE.clear()
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app), tmp_path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """開発者の視点（全項目が出る）。"""
    return _build(tmp_path, monkeypatch, True)


@pytest.fixture()
def guest_client(tmp_path, monkeypatch):
    """ゲスト・無料ログインの視点（オッズ・人気・仮想購入はサーバが落とす）。"""
    return _build(tmp_path, monkeypatch, False)


def _put(tmp_path, date, status_src="race_result", payoff=True, level=1):
    doc = pr.build_day("20261018", _storage(status_src, with_payoff=payoff))
    doc["date"] = date
    for r in doc["races"]:
        if r.get("included"):         # 画面の推奨馬券(target)の仮想購入。level 0 は見送り（購入なし）
            r["bets_t"] = {"status": "ok", "strategy": "target", "level": level, "total_bet": 1000 if level else 0, "total_return": 3000 if level else 0,
                           "by_type": {"sanrenpuku": {"invested": 1000, "returned": 3000, "n": 1, "hit": 1}} if level else {}, "estimated": False}
    pr.write_day(doc, tmp_path)


def test_empty_store_returns_no_days_and_no_news(client):
    c, _ = client
    assert c.get("/api/prediction-results/days").json() == {"days": [], "latest": None}
    assert c.get("/api/prediction-results/news").json() == {"items": []}


def test_day_scope_has_kpi_races_and_status(client):
    c, tmp = client
    _put(tmp, "20261018")
    d = c.get("/api/prediction-results?scope=day&date=20261018").json()
    assert d["status"] == "final" and d["label"] == "10/18(日)" and d["days"] == ["20261018"]
    assert d["kpi"]["honmei"] == {"n": 1, "win": 0.0, "top2": 1.0, "top3": 1.0}
    assert d["kpi"]["marks"]["star"]["rates"]["f1"] == 1.0 and d["summary"]["marks"]["pair"]["f3"] == 1
    assert d["races"][0]["race_id"] == "202610180101" and d["kpi"]["invested"] > 0


def test_week_scope_sums_days_and_provisional_wins_when_mixed(client):
    c, tmp = client
    _put(tmp, "20261017")
    _put(tmp, "20261018", "race_result_on_time", payoff=False)
    w = c.get("/api/prediction-results?scope=week&date=20261018").json()
    assert w["days"] == ["20261017", "20261018"] and w["status"] == "provisional"
    assert w["kpi"]["honmei"]["n"] == 2 and len(w["series"]) == 2 and "races" not in w


def test_news_orders_day_week_then_highlights_and_hides_roi_in_public_text(client):
    c, tmp = client
    _put(tmp, "20261018")
    n = c.get("/api/prediction-results/news?limit=5").json()
    kinds = [i["kind"] for i in n["items"]]
    assert kinds[:2] == ["day", "week"] and n["latest"] == "20261018" and n["items"][0]["badge"] == "確定"
    assert "回収率" in n["items"][0]["subtitle"] and "回収率" not in n["items"][0]["subtitle_public"]
    assert n["items"][0]["link"] == "/prediction-results?scope=day&date=20261018"


def test_news_never_includes_big_miss(client):
    c, tmp = client
    _put(tmp, "20261018")
    n = c.get("/api/prediction-results/news?limit=20").json()
    assert "big_miss" not in {i["kind"] for i in n["items"]}
    assert not any("大きく外れた" in i["title"] for i in n["items"])


def test_bad_params_are_rejected(client):
    c, _ = client
    assert c.get("/api/prediction-results?scope=year").status_code == 422
    assert c.get("/api/prediction-results?scope=day&date=2026-10-18").status_code == 422


def test_rated_block_has_standard_or_above_aggregate_for_members(client):
    c, tmp = client
    _put(tmp, "20261018")
    d = c.get("/api/prediction-results?scope=day&date=20261018").json()
    assert d["rated"]["label"] == "標準以上" and d["rated"]["min_level"] == 1
    assert set(d["rated"]["counts"]) == {"settled", "rated", "skipped", "unrated"} and "marks" in d["rated"]["kpi"]


def test_rated_block_is_not_shown_to_guests(guest_client):
    g, tmp = guest_client
    _put(tmp, "20261018")
    gd = g.get("/api/prediction-results?scope=day&date=20261018").json()
    assert "rated" not in gd and not any("rated" in k for pt in gd["series"] for k in pt)


def test_roi_never_includes_skipped_races(client):
    c, tmp = client
    _put(tmp, "20261018", level=0)                      # 見送りのレース（購入額はあるが推奨度は 0）
    n = c.get("/api/prediction-results/news?limit=10").json()["items"]
    assert all("回収率" not in i["subtitle"] and i["roi"] is None for i in n if i["kind"] in ("day", "week"))
    b = c.get("/api/prediction-results/bets-summary?date=20261018").json()
    assert b["races"] == {} and b["day"]["roi"] is None and b["day"]["races_bought"] == 0 and b["week"]["invested"] == 0
    d = c.get("/api/prediction-results?scope=day&date=20261018").json()
    assert d["rated"]["counts"]["rated"] == 0 and d["rated"]["kpi"]["roi"] is None and d["rated"]["kpi"]["invested"] == 0
    assert not any(h["kind"] == "big_return" for h in d["highlights"])


def test_bets_summary_counts_only_standard_or_above(client):
    c, tmp = client
    _put(tmp, "20261018", level=1)
    b = c.get("/api/prediction-results/bets-summary?date=20261018").json()
    assert len(b["races"]) == 1 and b["day"]["races_bought"] == 1 and b["day"]["roi"] is not None


def test_news_has_factual_hit_item_without_evaluative_wording(client):
    c, tmp = client
    _put(tmp, "20261018")
    items = c.get("/api/prediction-results/news?limit=20").json()["items"]
    assert {i["kind"] for i in items} <= {"day", "week", "hit"}
    assert not any(w in (i["title"] + i["subtitle"]) for i in items for w in ("大きな払戻", "穴馬", "注目:", "大きく外れた", "本命", "◎"))
    hit = next((i for i in items if i["kind"] == "hit"), None)
    assert hit and hit["title"].endswith("の的中レース") and "R" in hit["subtitle"] and "円" not in hit["subtitle"]


def test_hit_news_only_lists_races_with_recommended_bets(client):
    c, tmp = client
    _put(tmp, "20261018", level=0)                       # 見送り（買い目推奨なし）のレースは的中レースに出さない
    items = c.get("/api/prediction-results/news?limit=20").json()["items"]
    assert not any(i["kind"] == "hit" for i in items)


def test_news_for_guests_has_no_hit_and_no_bets(guest_client):
    g, tmp = guest_client
    _put(tmp, "20261018")
    items = g.get("/api/prediction-results/news?limit=20").json()["items"]
    assert items and not any(i["kind"] == "hit" for i in items)
    assert not any(w in i["subtitle"] for i in items for w in ("推奨馬券", "回収率", "円"))
