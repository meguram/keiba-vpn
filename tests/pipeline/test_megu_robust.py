"""めぐ指数の頑健化（範囲外の復元・[0,100] 丸め・信頼度フラグ）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.megu_index.compute import compute_for_dataframe
from src.pipeline.megu_index.robust import MEGU_VERSION, add_robust_columns


def _frame(rows):
    base = {"computation_status": "valid", "par_time_final": 70.0, "corrected_time_sec": 70.0,
            "par_sample_count": 100, "par_match_level": "L1_class"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_ok_row_keeps_value_and_is_high_confidence():
    out = add_robust_columns(_frame([{"corrected_time_sec": 69.0}]), finish_pos=pd.Series([1]))
    r = out.iloc[0]
    assert r["megu_index_raw"] == 60.0 and r["megu_index_robust"] == 60.0
    assert r["megu_index_status"] == "ok" and not r["low_confidence"] and r["megu_version"] == MEGU_VERSION


def test_out_of_range_row_is_restored_not_dropped():
    out = add_robust_columns(_frame([{"computation_status": "out_of_range", "corrected_time_sec": 71.5}]),
                             finish_pos=pd.Series([12]))
    r = out.iloc[0]
    assert r["megu_index_status"] == "restored_out_of_range"
    assert abs(r["megu_index_raw"] - 35.0) < 1e-9 and abs(r["megu_index_robust"] - 35.0) < 1e-9


def test_values_are_clipped_to_0_100_and_raw_is_kept():
    out = add_robust_columns(_frame([
        {"corrected_time_sec": 60.0},                                    # 150 → 100
        {"corrected_time_sec": 80.0, "computation_status": "out_of_range"},  # -50 → 0
        {"corrected_time_sec": 90.0},                                    # -150 → 0
    ]), finish_pos=pd.Series([1, 15, 16]))
    assert list(out["megu_index_raw"]) == [150.0, -50.0, -150.0]
    assert list(out["megu_index_robust"]) == [100.0, 0.0, 0.0]
    assert list(out["megu_index_status"]) == ["clipped", "restored_clipped", "clipped"]


def test_no_par_and_missing_finish_position_get_no_value():
    out = add_robust_columns(_frame([
        {"par_time_final": np.nan},
        {"corrected_time_sec": 69.0},
    ]), bad_par=pd.Series([True, False]), finish_pos=pd.Series([1, np.nan]))
    assert out["megu_index_raw"].isna().all()
    assert list(out["megu_index_status"]) == ["no_par", "invalid"]
    assert not out["low_confidence"].any()


def test_low_confidence_for_small_samples_and_pooled_fallback():
    out = add_robust_columns(_frame([
        {"par_sample_count": 12},
        {"par_match_level": "L3_pooled"},
        {"par_match_level": "L2_no_direction", "par_sample_count": 80},
    ]), finish_pos=pd.Series([1, 1, 1]))
    assert list(out["low_confidence"]) == [True, True, False]


def test_compute_for_dataframe_adds_columns_without_changing_existing_semantics():
    params = {"beta_pace": 0.79, "beta_track": 0.19, "beta_weight": 0.61, "beta_level": -0.93, "tsi_mean": 3.3}
    par = pd.DataFrame([{"distance": 1200, "course": "右", "surface": "芝", "track_condition": "良",
                         "class_bucket": "1勝", "par_time_sec": 70.5, "par_front_split_sec": 35.0, "sample_count": 200}])
    base = {"date": "2026-07-18", "venue": "函館", "venue_code": "01", "surface": "芝", "distance": 1200,
            "direction": "右", "track_condition": "良", "grade": "1勝", "race_class": "3歳以上1勝クラス",
            "race_name": "3歳以上1勝クラス", "jockey_weight": 55.0, "sex_age": "牡3", "lap_times": "35.0-34.5"}
    df = pd.DataFrame([
        {**base, "race_id": "202607181001", "horse_id": "a", "time_sec": 70.0, "finish_position": 2},
        {**base, "race_id": "202607181001", "horse_id": "b", "time_sec": 70.4, "finish_position": 1},
        {**base, "race_id": "202607181001", "horse_id": "c", "time_sec": 74.0, "finish_position": 9},   # 2着+2秒超
    ])
    out = compute_for_dataframe(df, params, par).set_index("horse_id")
    assert out.loc["c", "computation_status"] == "out_of_range" and pd.isna(out.loc["c", "megu_index"])
    assert out.loc["c", "megu_index_status"] == "restored_out_of_range"
    assert out.loc["c", "megu_index_raw"] < out.loc["a", "megu_index_raw"]
    assert out.loc["a", "megu_index"] == out.loc["a", "megu_index_raw"]
    assert (out["par_sample_count"] == 200).all() and not out["low_confidence"].any()


def test_recompute_keeps_rows_dropped_by_the_compute_filters():
    from src.pipeline.megu_index.compute import excluded_rows

    flat = pd.DataFrame([
        {"race_id": "r1", "horse_id": "a", "race_name": "3歳以上1勝クラス", "surface": "芝", "time_sec": 70.0, "distance": 1200},
        {"race_id": "r2", "horse_id": "b", "race_name": "障害4歳以上未勝利", "surface": "芝", "time_sec": 190.0, "distance": 2750},
        {"race_id": "r3", "horse_id": "c", "race_name": "新馬", "surface": "障", "time_sec": 190.0, "distance": 2750},
        {"race_id": "r4", "horse_id": "d", "race_name": "新馬", "surface": "ダート", "time_sec": float("nan"), "distance": 1200},
    ])
    computed = pd.DataFrame([{"race_id": "r1", "horse_id": "a"}])
    out = excluded_rows(flat, computed).set_index("horse_id")
    assert set(out.index) == {"b", "c", "d"}
    assert out.loc["b", "megu_index_status"] == "excluded_obstacle"
    assert out.loc["c", "megu_index_status"] == "excluded_surface"
    assert out.loc["d", "megu_index_status"] == "no_time"
    assert out["megu_index"].isna().all() and (out["computation_status"] == "excluded").all()


def test_parquet_merge_with_old_schema_keeps_old_rows_and_adds_new_columns(tmp_path, monkeypatch):
    from src.pipeline.megu_index import compute as C

    monkeypatch.setattr(C, "TABLES_DIR", tmp_path)
    d = tmp_path / "2025"
    d.mkdir()
    pd.DataFrame({"race_id": ["r1", "r2"], "horse_id": ["a", "b"], "megu_index": [99.0, 80.0],
                  "computation_status": ["valid", "valid"]}).to_parquet(d / "megu_index_flat.parquet")
    new = pd.DataFrame({"race_id": ["r2", "r3"], "horse_id": ["b", "c"], "megu_index": [48.0, 51.0],
                        "computation_status": ["valid", "valid"], "megu_index_raw": [48.0, 51.0],
                        "megu_index_robust": [48.0, 51.0], "megu_index_status": ["ok", "ok"],
                        "low_confidence": [False, True], "megu_version": ["v"] * 2})
    C._save_megu_parquet_backup(new, 2025)
    got = pd.read_parquet(d / "megu_index_flat.parquet").set_index("race_id")
    assert list(got.index) == ["r1", "r2", "r3"]
    assert got.loc["r2", "megu_index"] == 48.0                      # 重なった行は新しい値
    assert pd.isna(got.loc["r1", "megu_index_robust"]) and got.loc["r3", "megu_index_robust"] == 51.0


def test_frame_for_parquet_keeps_rows_without_par_and_excluded_rows():
    from src.pipeline.megu_index.compute import frame_for_parquet

    flat = pd.DataFrame([
        {"race_id": "r1", "horse_id": "a", "race_name": "新馬", "surface": "芝", "time_sec": 70.0, "distance": 1200},
        {"race_id": "r2", "horse_id": "b", "race_name": "障害未勝利", "surface": "芝", "time_sec": 190.0, "distance": 2750},
    ])
    res = pd.DataFrame([{"race_id": "r1", "horse_id": "a", "par_time_final": np.nan, "megu_index": np.nan,
                         "computation_status": "no_par", "megu_index_status": "no_par"}])
    out = frame_for_parquet(res, flat).set_index("horse_id")
    assert set(out.index) == {"a", "b"} and out.loc["a", "megu_index_status"] == "no_par"
    assert out.loc["b", "megu_index_status"] == "excluded_obstacle"
