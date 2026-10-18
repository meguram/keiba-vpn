from src.scraper import horse_result_fix as hf


def test_apply_overlay_replaces_only_matched_races_and_keeps_original():
    hr = {"horse_id": "h1", "race_history": [
        {"race_id": "r1", "passing_order": "66", "last_3f": 88.0, "weight": 500},
        {"race_id": "r0", "passing_order": "**", "last_3f": 0.0, "weight": 490}]}
    lk = {("h1", "r1"): {"passing_order": "6-6-6-6", "last_3f": 40.1}}
    out = hf.apply_overlay(hr, lk)
    assert out["race_history"][0]["passing_order"] == "6-6-6-6" and out["race_history"][0]["last_3f"] == 40.1
    assert out["race_history"][0]["weight"] == 500
    assert out["race_history"][1]["passing_order"] == "**"            # フラットに無い(2020 年より前)走はそのまま
    assert hr["race_history"][0]["passing_order"] == "66"              # 元は変えない


def test_empty_lookup_is_noop():
    hr = {"horse_id": "h1", "race_history": [{"race_id": "r1", "passing_order": "66", "last_3f": 88.0}]}
    assert hf.apply_overlay(hr, {}) is hr

