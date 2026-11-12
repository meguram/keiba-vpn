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


class _ShutubaStorage:
    """出馬表だけを持つ取得元。段階収集で同じレースの版が入れ替わる。"""

    def __init__(self, data):
        self.data = data

    def batch_list_blobs(self, category, year):
        return {rid: 1 for rid in self.data} if category == "race_shutuba" else {}

    def load(self, category, key):
        return self.data.get(key) if category == "race_shutuba" else None


def _shutuba(rid, date, horses, drawn):
    return {"race_id": rid, "date": date, "race_name": "x", "venue": "東京", "distance": 1600, "surface": "芝",
            "race_class": "１勝クラス" if drawn else "", "start_time": "12:00" if drawn else "", "field_size": len(horses),
            "entries": [{"horse_number": (i + 1) if drawn else 0, "horse_id": h} for i, h in enumerate(horses)]}


def test_refresh_replaces_pre_draw_shutuba_with_final_and_keeps_race_table(tmp_path, monkeypatch):
    """登録段階（馬番が全頭 0）で入った出馬表は、最終版が出たら取り直してレースごと入れ替える（出走しない登録馬は残らない）。"""
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    today = datetime.now(timezone(timedelta(hours=9)))
    d_old = (today - timedelta(days=60)).strftime("%Y-%m-%d")      # 窓より古い
    d_new = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    old_final = _shutuba("202605010101", d_old, ["a1", "a2"], drawn=True)
    reg = _shutuba("202605010201", d_new, ["h1", "h2", "h3", "h4"], drawn=False)   # 登録 4 頭
    st = _ShutubaStorage({"202605010101": old_final, "202605010201": reg})
    et.export_year("2026", categories=["race_shutuba"], storage=st)
    flat = tmp_path / "2026" / "race_shutuba_flat.parquet"
    assert et._pre_draw_race_ids(flat) == {"202605010201"}
    assert et._existing_race_ids(flat) == {"202605010101"}

    st.data["202605010201"] = _shutuba("202605010201", d_new, ["h1", "h3"], drawn=True)   # 最終: 2 頭・馬番あり
    et.export_year("2026", categories=["race_shutuba"], storage=st, refresh=True)
    df = pd.read_parquet(flat)
    r = df[df["race_id"] == "202605010201"]
    assert sorted(r["horse_id"]) == ["h1", "h3"]                       # 出走しない登録馬の行は消える
    assert set(r["horse_number"]) == {1, 2}
    assert set(r["start_time"]) == {"12:00"}
    assert len(df[df["race_id"] == "202605010101"]) == 2               # ほかのレースはそのまま
    assert et._pre_draw_race_ids(flat) == set()
    race_tbl = pd.read_parquet(tmp_path / "2026" / "race_shutuba_race.parquet")
    assert set(race_tbl["race_id"]) == {"202605010101", "202605010201"}   # race 表は差分で上書きされない
    assert len(race_tbl) == 2


def test_pre_draw_ignores_races_older_than_lookback(tmp_path):
    p = tmp_path / "race_shutuba_flat.parquet"
    pd.DataFrame({"race_id": ["r1", "r1", "r2", "r3"], "horse_number": [0, 0, 0, 3],
                  "date": ["2026-10-01", "2026-10-01", "2026-11-10", "2026-11-10"]}).to_parquet(p)
    assert et._pre_draw_race_ids(p, today="2026-11-12") == {"r2"}


def test_refresh_keeps_pre_draw_rows_when_reload_fails(tmp_path, monkeypatch):
    """取り直せなかった（読み込み失敗の）レースの既存行は、ほかのレースを追記しても消さない。"""
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    d = (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=3)).strftime("%Y-%m-%d")
    st = _ShutubaStorage({"202605010201": _shutuba("202605010201", d, ["h1", "h2", "h3"], drawn=False),
                          "202605010202": _shutuba("202605010202", d, ["g1", "g2"], drawn=True)})
    et.export_year("2026", categories=["race_shutuba"], storage=st)
    st.data["202605010201"] = None                                                   # 読み込みに失敗する
    st.data["202605010203"] = _shutuba("202605010203", d, ["k1"], drawn=True)       # 新しいレース
    et.export_year("2026", categories=["race_shutuba"], storage=st, refresh=True)
    df = pd.read_parquet(tmp_path / "2026" / "race_shutuba_flat.parquet")
    assert len(df[df["race_id"] == "202605010201"]) == 3
    assert set(df["race_id"]) == {"202605010201", "202605010202", "202605010203"}
    race_tbl = pd.read_parquet(tmp_path / "2026" / "race_shutuba_race.parquet")
    assert set(race_tbl["race_id"]) == {"202605010201", "202605010202", "202605010203"}
