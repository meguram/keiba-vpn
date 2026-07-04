"""推論パイプラインのマッピングテスト。"""

from src.pipeline.inference.inference_pipeline import _map_stage1_to_spec


def test_map_stage1_to_spec():
    raw = {
        "race_id": "202506010811",
        "distance": 1600,
        "total_horses": 2,
        "predictions": [
            {"horse_id": "h1", "horse_number": 1, "pred_score": 0.8, "pred_rank": 1},
            {"horse_id": "h2", "horse_number": 2, "pred_score": 0.2, "pred_rank": 2},
        ],
    }
    payload = _map_stage1_to_spec(raw, model_version="v1.0.0")
    assert payload["race_id"] == "202506010811"
    assert len(payload["horses"]) == 2
    assert payload["pace_prediction"]["pace_category"] in ("HIGH", "MIDDLE", "SLOW")
    assert "expected_win_roi" in payload["horses"][0]
