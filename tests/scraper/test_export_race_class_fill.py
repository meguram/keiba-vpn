"""結果の flat の race_class が、速報の橋渡しを経ると空のまま残った件（2026-10-24〜11-08）の再発防止。

確定結果・速報の文書には race_class が無く、flat の値は出馬表のメタの補いだけで入る。メタを race_result_flat から組むと、
橋渡しの行（空の race_class）がメタになり、確定結果に置き換わっても空だった。"""

import pandas as pd

from src.scraper import export_tables as et


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _rr(rid, no, src=None, race_class=None, date="2026-11-08"):
    r = {"race_id": rid, "horse_number": no, "horse_id": f"h{rid}{no}", "finish_position": no, "date": date,
         "race_class": race_class, "surface": "ダート", "distance": 1400}
    if src:
        r["result_source"] = src
    return r


def _sh(rid, race_class, no=1):
    return {"race_id": rid, "horse_number": no, "horse_id": f"h{rid}{no}", "race_class": race_class, "date": "2026-11-08"}


def test_bridged_empty_class_is_filled_from_local_shutuba_and_reaches_official_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_rr("A", 1, race_class="サラ系３歳 未勝利"), _rr("B", 1, "on_time"), _rr("B", 2, "on_time")])
    _write(y / "race_shutuba_flat.parquet", [_sh("B", "サラ系２歳 未勝利"), _sh("B", "サラ系２歳 未勝利", 2)])
    idx = et._build_meta_index_from_race_result_flat("2026")
    assert idx["B"]["race_class"] == "サラ系２歳 未勝利"
    assert idx["A"]["race_class"] == "サラ系３歳 未勝利"
    # 確定結果の文書には race_class が無い（db.netkeiba.com）。メタの補いで入る
    official = {"race_id": "B", "date": "2026-11-08", "entries": [{"horse_number": 1, "finish_position": 1}]}
    rows = et._data_to_flat_rows(official, "race_result", idx)
    assert rows and rows[0]["race_class"] == "サラ系２歳 未勝利"


def test_official_row_wins_over_bridged_row_for_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    # 橋渡しの行が先に並んでいても、確定結果の行のメタを採る
    _write(y / "race_result_flat.parquet", [_rr("A", 1, "on_time", race_class=""), _rr("A", 2, race_class="サラ系３歳以上 １勝クラス")])
    assert et._build_meta_index_from_race_result_flat("2026")["A"]["race_class"] == "サラ系３歳以上 １勝クラス"


def test_old_flat_without_result_source_and_missing_shutuba_flat(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [{"race_id": "A", "horse_number": 1, "race_class": "", "date": "2026-01-05"}])
    idx = et._build_meta_index_from_race_result_flat("2026")          # race_shutuba_flat が無くても落ちない
    assert idx["A"]["race_class"] == ""


def test_jump_race_stays_empty_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_rr("J", 1, race_class=None)])
    _write(y / "race_shutuba_flat.parquet", [_sh("J", None), _sh("X", "サラ系３歳 未勝利")])
    idx = et._build_meta_index_from_race_result_flat("2026")
    assert not idx["J"]["race_class"] and "X" not in idx              # 出馬表にだけあるレースはメタに足さない


def test_rebuild_keeps_shutuba_meta_for_races_not_yet_in_flat(tmp_path, monkeypatch):
    """race_result を 1 件でも追記した日に、確定結果が未公開のレースの（出馬表から補った）メタを捨てない。"""
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_rr("A", 1, race_class="サラ系３歳 未勝利")])
    prev = {"C": {"race_id": "C", "race_class": "サラ系２歳 新馬", "date": "2026-11-15"},
            "A": {"race_id": "A", "race_class": "x", "venue": "京都"}}
    idx = et._rebuild_meta_index("2026", prev)
    assert idx["C"]["race_class"] == "サラ系２歳 新馬"
    assert idx["A"]["race_class"] == "サラ系３歳 未勝利"              # flat の値を優先し、空の欄だけ前のメタで補う
    assert idx["A"]["venue"] == "京都"


def test_fill_race_class_counts_then_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet",
           [_rr("A", 1, race_class="サラ系３歳 未勝利"), _rr("B", 1), _rr("B", 2), _rr("J", 1)])
    _write(y / "race_shutuba_flat.parquet", [_sh("A", "違う値"), _sh("B", "サラ系２歳 未勝利"), _sh("J", "")])
    st = et.fill_race_class_from_local_shutuba("2026", ("race_result", "race_result_on_time"))
    assert st[0] == {"category": "race_result", "races": 3, "empty_races": 2, "fillable_races": 1,
                     "applied": False, "still_empty_races": 1}
    assert st[1]["skipped"]
    assert pd.read_parquet(y / "race_result_flat.parquet")["race_class"].isna().sum() == 3     # 数えるだけでは変えない
    et.fill_race_class_from_local_shutuba("2026", ("race_result",), apply=True)
    df = pd.read_parquet(y / "race_result_flat.parquet")
    got = df.groupby("race_id")["race_class"].first().to_dict()
    assert got["A"] == "サラ系３歳 未勝利" and got["B"] == "サラ系２歳 未勝利" and not got["J"]
    assert len(df) == 4


class _Storage:
    def __init__(self, docs):
        self.docs, self.loads = docs, []

    def load(self, cat, rid):
        self.loads.append((cat, rid))
        return self.docs.get(rid)


def test_add_shutuba_meta_refills_bridged_meta_with_empty_class(tmp_path, monkeypatch):
    """手元の出馬表に値が無い橋渡しのレースは、出馬表を名前で読んで空の欄だけ補う。確定結果の行のメタは読み直さない。"""
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_rr("A", 1, race_class="サラ系３歳 未勝利"), _rr("B", 1, "on_time"), _rr("C", 1)])
    idx = et._build_meta_index_from_race_result_flat("2026")
    need = [k for k in ("A", "B", "C", "D") if et._needs_shutuba_meta(idx, k)]
    assert need == ["B", "D"]                                         # C は確定結果の行（空でも読み直さない）
    st = _Storage({"B": {"race_class": "サラ系２歳 １勝クラス", "distance": 1800, "date": "x"}, "D": {"race_class": "z"}})
    assert et._add_shutuba_meta(st, need, idx) == 2
    assert idx["B"]["race_class"] == "サラ系２歳 １勝クラス"
    assert idx["B"]["distance"] == 1400 and idx["B"]["date"] == "2026-11-08"     # 値のある欄は変えない
    assert not et._needs_shutuba_meta(idx, "B")
    assert "_from_bridge" not in et._extract_race_meta({"race_id": "B"}, idx)


def test_rebuild_treats_zero_distance_as_missing(monkeypatch):
    prev = {"A": {"race_id": "A", "distance": 3000, "field_size": 14, "_from_bridge": True}}
    monkeypatch.setattr(et, "_build_meta_index_from_race_result_flat",
                        lambda year: {"A": {"race_id": "A", "distance": 0, "field_size": 12}})
    idx = et._rebuild_meta_index("2026", prev)
    assert idx["A"]["distance"] == 3000 and idx["A"]["field_size"] == 12 and "_from_bridge" not in idx["A"]


def test_fill_apply_keeps_other_column_types(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    y.mkdir(parents=True)
    tbl = pa.table({"race_id": ["B", "B"], "horse_number": pa.array([1, None], type=pa.int64()),
                    "weather": pa.nulls(2), "race_class": pa.nulls(2)})
    pq.write_table(tbl, y / "race_result_flat.parquet")
    _write(y / "race_shutuba_flat.parquet", [_sh("B", "サラ系２歳 未勝利")])
    et.fill_race_class_from_local_shutuba("2026", ("race_result",), apply=True)
    got = pq.read_table(y / "race_result_flat.parquet")
    assert got.schema.field("horse_number").type == pa.int64() and got.schema.field("weather").type == pa.null()
    assert got.column("race_class").to_pylist() == ["サラ系２歳 未勝利"] * 2


def test_incremental_append_without_meta_index_fills_class_from_local_shutuba(tmp_path, monkeypatch):
    """meta_index を渡さない増分の追記（recent-retry の後の _trigger_track_speed_for_date）でも、出馬表の race_class で補う（2026-11-21〜23）。"""
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    old, new = "202605050401", "202605050501"
    _write(y / "race_result_flat.parquet", [_rr(old, 1, race_class="サラ系３歳 未勝利", date="2026-11-15")])
    _write(y / "race_shutuba_flat.parquet", [_sh(new, "サラ系２歳 未勝利")])
    doc = {"race_id": new, "date": "2026-11-22", "surface": "ダート", "distance": 1300, "grade": "未勝利",
           "entries": [{"horse_number": 1, "finish_position": 1, "horse_id": "h1"}]}
    et.export_category_chunked("2026", "race_result", _Storage({new: doc}), keys=[new])
    f = pd.read_parquet(y / "race_result_flat.parquet")
    assert f.loc[f["race_id"].astype(str) == new, "race_class"].tolist() == ["サラ系２歳 未勝利"]
    assert f.loc[f["race_id"].astype(str) == old, "race_class"].tolist() == ["サラ系３歳 未勝利"]


def test_with_local_shutuba_meta_keeps_existing_values(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    _write(tmp_path / "2026" / "race_shutuba_flat.parquet", [_sh("A", "出馬表の値"), _sh("B", "出馬表の値")])
    idx = et._with_local_shutuba_meta("2026", {"A": {"race_class": "元の値"}, "B": {"race_class": ""}}, ["A", "B", "C"])
    assert idx["A"]["race_class"] == "元の値" and idx["B"]["race_class"] == "出馬表の値" and "C" not in idx
    assert et._with_local_shutuba_meta("2027", None, ["A"]) is None        # 出馬表の flat が無い年は何もしない


def test_with_local_shutuba_meta_fills_surface_and_jump_without_touching_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    _write(tmp_path / "2026" / "race_shutuba_flat.parquet",
           [{**_sh("A", "x"), "surface": "ダート"}, {**_sh("J", None), "surface": "障"}])
    src = {"A": {"surface": ""}, "J": {"surface": "芝"}}
    idx = et._with_local_shutuba_meta("2026", src, ["A", "J"])
    assert idx["A"]["surface"] == "ダート"                               # 空の面を補う
    assert idx["J"]["surface"] == "障"                                   # 出馬表が「障」なら誤った「芝」を上書き
    assert src == {"A": {"surface": ""}, "J": {"surface": "芝"}}          # 呼び出し元の dict は変えない
