from src.scraper.export_tables import _extract_race_meta


def test_zero_distance_and_field_size_fall_back_to_shutuba_meta():
    data = {"race_id": "202106020708", "distance": 0, "surface": "", "field_size": 0, "grade": "OP", "date": "2021-03-20"}
    fallback = {"202106020708": {"distance": 3350, "surface": "障", "field_size": 14}}
    meta = _extract_race_meta(data, fallback)
    assert meta["distance"] == 3350 and meta["surface"] == "障" and meta["field_size"] == 14 and meta["grade"] == "OP"


def test_valid_values_are_not_overwritten():
    data = {"race_id": "r", "distance": 1600, "surface": "芝", "field_size": 16}
    meta = _extract_race_meta(data, {"r": {"distance": 9999, "surface": "ダート", "field_size": 1}})
    assert (meta["distance"], meta["surface"], meta["field_size"]) == (1600, "芝", 16)
