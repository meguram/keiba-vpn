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
