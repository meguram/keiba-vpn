"""確定結果（race_result）の公開が遅れる間、速報（race_result_on_time）から race_result_flat に橋渡しする。"""

import pandas as pd

from src.scraper import export_tables as et


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(rid, no, fin, src=None):
    r = {"race_id": rid, "horse_number": no, "horse_id": f"h{no}", "finish_position": fin, "date": "2026-10-17"}
    if src:
        r["result_source"] = src
    return r


def test_bridge_adds_only_missing_races_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_row("A", 1, 1), _row("A", 2, 2)])
    _write(y / "race_result_on_time_flat.parquet", [_row("A", 1, 1), _row("B", 1, 1), _row("B", 2, 2)])
    assert et.bridge_on_time_results("2026")["bridged_races"] == 1
    df = pd.read_parquet(y / "race_result_flat.parquet")
    assert set(df["race_id"]) == {"A", "B"}
    assert (df[df["race_id"] == "B"]["result_source"] == "on_time").all()
    assert df[df["race_id"] == "A"]["result_source"].isna().all()
    assert et.bridge_on_time_results("2026")["bridged_races"] == 0           # 2 回目は何も足さない


def test_official_result_replaces_bridged_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2026"
    _write(y / "race_result_flat.parquet", [_row("B", 1, 1, "on_time"), _row("B", 2, 2, "on_time"), _row("B", 3, 3, "on_time")])
    _write(y / "race_result_on_time_flat.parquet", [_row("B", 1, 1), _row("B", 2, 2), _row("B", 3, 3)])
    assert et._existing_race_ids(y / "race_result_flat.parquet") == set()     # 橋渡しの行は確定結果として数えない
    et._append_parquet_rows(y / "race_result_flat.parquet", [_row("B", 1, 2), _row("B", 2, 1)])    # 確定結果（3 番の馬は取消）
    st = et.bridge_on_time_results("2026")
    df = pd.read_parquet(y / "race_result_flat.parquet")
    assert st["removed_stale_races"] == 1 and set(df["horse_number"]) == {1, 2}
    assert df["result_source"].isna().all() and dict(zip(df["horse_number"], df["finish_position"])) == {1: 2, 2: 1}


def test_append_keeps_all_horses_when_horse_number_is_zero(tmp_path):
    """枠順確定前の出馬表は horse_number が全頭 0。番号で重複除去すると 1 レース 1 頭に潰れていた。"""
    p = tmp_path / "race_shutuba_flat.parquet"
    rows = [{"race_id": "R1", "horse_number": 0, "horse_id": f"h{i}", "date": "2026-06-06"} for i in range(5)]
    et._append_parquet_rows(p, rows)
    et._append_parquet_rows(p, [{"race_id": "R2", "horse_number": 0, "horse_id": f"g{i}", "date": "2026-06-06"} for i in range(3)])
    df = pd.read_parquet(p)
    assert df.groupby("race_id").size().to_dict() == {"R1": 5, "R2": 3}


def test_append_replaces_same_horse_after_draw(tmp_path):
    p = tmp_path / "race_shutuba_flat.parquet"
    et._append_parquet_rows(p, [{"race_id": "R1", "horse_number": 0, "horse_id": f"h{i}", "date": "d"} for i in range(3)])
    et._append_parquet_rows(p, [{"race_id": "R1", "horse_number": i + 1, "horse_id": f"h{i}", "date": "d"} for i in range(3)])
    df = pd.read_parquet(p)
    assert len(df) == 3 and sorted(df["horse_number"]) == [1, 2, 3]


def test_append_without_horse_id_falls_back_to_number(tmp_path):
    p = tmp_path / "x_flat.parquet"
    et._append_parquet_rows(p, [{"race_id": "R1", "horse_number": 1}, {"race_id": "R1", "horse_number": 2}])
    et._append_parquet_rows(p, [{"race_id": "R1", "horse_number": 2}])
    assert len(pd.read_parquet(p)) == 2
