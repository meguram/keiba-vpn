"""予測が無いレースの応答（出馬表の馬 + 最新オッズ）。モックを返さず、画面が読む predictions[] の形で返すこと。"""

from __future__ import annotations

from src.api.entries_only_predictions import NO_PREDICTION_MESSAGE, entries_only_predictions


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


def test_lists_entries_with_real_odds_in_the_predictions_shape():
    st = FakeStorage({
        ("race_shutuba", "R1"): {"entries": [
            {"horse_number": 2, "horse_name": "ニ", "horse_id": "h2", "weight": 0, "weight_change": 0},
            {"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "bracket_number": 1, "sex_age": "牡3", "jockey_name": "丹内", "jockey_weight": 54.0,
             "trainer_name": "牧浦", "weight": 514, "weight_change": 0}]},
        ("race_odds", "R1"): {"entries": [{"horse_number": 1, "win_odds": 7.5, "place_odds_min": 2.4, "place_odds_max": 3.3, "popularity": 3}]},
    })
    d = entries_only_predictions("R1", st)
    assert d["status"] == "no_prediction" and d["has_prediction"] is False and d["message"] == NO_PREDICTION_MESSAGE
    assert [p["horse_number"] for p in d["predictions"]] == [1, 2]                 # 馬番順
    assert d["predictions"][0]["horse_name"] == "イチ" and d["predictions"][0]["win_odds"] == 7.5
    assert d["predictions"][1]["win_odds"] is None                               # オッズが無い馬は None（作り出さない）
    assert "win_prob" not in d["predictions"][0] and "_mock" not in d            # 予測値は一切入れない
    p1, p2 = d["predictions"]
    assert (p1["sex_age"], p1["jockey_name"], p1["trainer_name"], p1["jockey_weight"], p1["bracket_number"]) == ("牡3", "丹内", "牧浦", 54.0, 1)
    assert p1["weight"] == 514 and p1["weight_change"] == 0                      # 増減 0 は正当な値（None にしない）
    assert p2["weight"] is None and p2["weight_change"] is None                  # 馬体重が未発表（0）なら増減も None


def test_falls_back_to_results_and_reports_no_entries():
    st = FakeStorage({("race_result", "R2"): {"results": [{"horse_number": 3, "horse_name": "サン", "horse_id": "h3"}]}})
    assert entries_only_predictions("R2", st)["predictions"][0]["horse_name"] == "サン"
    empty = entries_only_predictions("R9", FakeStorage({}))
    assert empty["status"] == "no_entries" and empty["predictions"] == []


def test_storage_errors_do_not_break_the_response():
    class Boom:
        def load(self, *a):
            raise RuntimeError("gcs down")

    assert entries_only_predictions("R1", Boom())["predictions"] == []


def test_stg_endpoint_returns_entries_only_not_mock(monkeypatch):
    """stg で GCS に予測が無いレース: モックを返さない。KEIBA_STG_MOCK_PREDICTIONS=1 のときだけモック。"""
    from fastapi.testclient import TestClient

    from src.api import app as A

    st = FakeStorage({("race_shutuba", "R1"): {"entries": [{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1"}]}})
    monkeypatch.setattr(A, "_get_storage", lambda: st)
    monkeypatch.setattr(A, "_keiba_env", lambda: "stg")
    monkeypatch.delenv("KEIBA_STG_MOCK_PREDICTIONS", raising=False)
    c = TestClient(A.app)
    d = c.get("/api/race/R1/predictions").json()
    assert d["has_prediction"] is False and d["predictions"][0]["horse_name"] == "イチ" and "_mock" not in d
    monkeypatch.setenv("KEIBA_STG_MOCK_PREDICTIONS", "1")
    assert c.get("/api/race/R1/predictions").json().get("_mock") is True
