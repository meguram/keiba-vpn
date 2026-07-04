"""E2E: Flask /api/v1 レスポンス形式。"""

from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    from src.api.flask_app import create_app

    with patch("src.api.flask_app.init_engine"):
        app = create_app()
        app.config["TESTING"] = True
        yield app.test_client()


def test_predictions_spec_fields(client):
    sample = {
        "race_id": "202506010811",
        "model_version": "v1.2.0",
        "predicted_at": "2025-06-01T08:30:00+09:00",
        "pace_prediction": {"pace_category": "MIDDLE", "lap_times": [{"furlong_index": 1, "predicted_lap_sec": 12.3}]},
        "horses": [{
            "horse_id": "2019105678",
            "post_no": 3,
            "win_prob": 0.18,
            "expected_win_roi": 94.8,
            "expected_show_roi": 101.1,
            "is_value_bet": True,
        }],
    }
    with patch("src.api.flask_app.get_predictions_cached", return_value=sample):
        resp = client.get("/api/v1/races/202506010811/predictions")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pace_prediction"]["lap_times"][0]["furlong_index"] == 1
    assert body["horses"][0]["is_value_bet"] is True
