import pandas as pd

from src.scraper import export_tables as et


class _Storage:
    def __init__(self, races, shutuba=None):
        self.races = races
        self.shutuba = shutuba or {}

    def batch_list_blobs(self, category, year):
        return {rid: 1 for rid in self.races} if category == "race_result" else {}

    def load(self, category, key):
        if category == "race_shutuba":
            return self.shutuba.get(key)
        return self.races.get(key) if category == "race_result" else None


def _race(rid, date):
    return {"race_id": rid, "date": date, "race_name": "x", "venue": "東京", "distance": 1600, "surface": "芝", "field_size": 2,
            "entries": [{"horse_number": 1, "horse_id": "h1", "finish_position": 1}, {"horse_number": 2, "horse_id": "h2", "finish_position": 2}]}


def test_refresh_appends_new_races_but_default_never_does(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    old = _race("202605010101", "2026-10-04")
    new = _race("202605010201", "2026-10-11")
    st = _Storage({"202605010101": old})
    et.export_year("2026", categories=["race_result"], storage=st)
    flat = tmp_path / "2026" / "race_result_flat.parquet"
    assert set(pd.read_parquet(flat)["race_id"]) == {"202605010101"}

    st.races["202605010201"] = new
    et.export_year("2026", categories=["race_result"], storage=st)                 # 既定: 自分の flat が基準なので入らない
    assert set(pd.read_parquet(flat)["race_id"]) == {"202605010101"}

    et.export_year("2026", categories=["race_result"], storage=st, refresh=True)   # refresh: 追記される
    df = pd.read_parquet(flat)
    assert set(df["race_id"]) == {"202605010101", "202605010201"}
    assert df[df["race_id"] == "202605010201"]["date"].iloc[0] == "2026-10-11"


def test_refresh_fills_zero_distance_of_new_hurdle_race_from_shutuba(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    st = _Storage({"202605010101": _race("202605010101", "2026-10-04")})
    et.export_year("2026", categories=["race_result"], storage=st)
    hurdle = _race("202605010301", "2026-10-11")
    hurdle["distance"] = 0
    st.races["202605010301"] = hurdle
    st.shutuba["202605010301"] = {"race_id": "202605010301", "distance": 3350, "surface": "障", "field_size": 2}
    et.export_year("2026", categories=["race_result"], storage=st, refresh=True)
    df = pd.read_parquet(tmp_path / "2026" / "race_result_flat.parquet")
    assert set(df[df["race_id"] == "202605010301"]["distance"]) == {3350}
