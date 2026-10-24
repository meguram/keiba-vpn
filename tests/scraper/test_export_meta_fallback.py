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


def test_float_distance_is_normalized_to_int_and_nan_falls_back():
    meta = _extract_race_meta({"race_id": "r", "distance": 1600.0, "field_size": 16.0}, None)
    assert meta["distance"] == 1600 and isinstance(meta["distance"], int) and isinstance(meta["field_size"], int)
    nan = _extract_race_meta({"race_id": "r", "distance": float("nan")}, {"r": {"distance": 2000}})
    assert nan["distance"] == 2000


def test_round_falls_back_to_race_id_suffix():
    assert _extract_race_meta({"race_id": "202606040904", "round": 0}, None)["round"] == 4
    assert _extract_race_meta({"race_id": "202606040911"}, None)["round"] == 11
