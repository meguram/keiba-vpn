import pandas as pd

from src.pipeline.megu_index.par_time_resolve import attach_par_time_with_fallback


def _par():
    # 良 だけにセルがあり、重・不良 は L5 で 良 にフォールバックする
    return pd.DataFrame([
        {"distance": 1800, "surface": "ダート", "course": "右", "track_condition": "良", "class_bucket": "1勝",
         "par_time_sec": 112.0, "par_front_split_sec": 50.0, "sample_count": 100},
        {"distance": 1200, "surface": "ダート", "course": "右", "track_condition": "良", "class_bucket": "1勝",
         "par_time_sec": 70.0, "par_front_split_sec": 35.0, "sample_count": 100},
    ])


def test_l5_fallback_writes_back_to_the_right_rows_even_with_non_default_index():
    # 先頭に L1 で解決する行が並び、重馬場の行は後ろ・index は連番ではない（当日だけの計算と同じ状況）
    df = pd.DataFrame({
        "distance": [1200, 1200, 1800, 1200],
        "surface": ["ダート"] * 4,
        "direction": ["右"] * 4,
        "track_condition": ["良", "良", "重", "良"],
        "class_bucket": ["1勝"] * 4,
    }, index=[40, 41, 42, 43])
    out = attach_par_time_with_fallback(df, _par())
    assert out.loc[40, "par_time_final"] == 70.0 and out.loc[41, "par_time_final"] == 70.0 and out.loc[43, "par_time_final"] == 70.0
    assert out.loc[42, "par_time_final"] == 112.0 and out.loc[42, "par_match_level"].startswith("L5")


def test_l5_fallback_does_not_overwrite_unrelated_rows_with_default_index():
    df = pd.DataFrame({
        "distance": [1200, 1200, 1800],
        "surface": ["ダート"] * 3,
        "direction": ["右"] * 3,
        "track_condition": ["良", "良", "重"],
        "class_bucket": ["1勝"] * 3,
    })
    out = attach_par_time_with_fallback(df, _par())
    assert list(out["par_time_final"]) == [70.0, 70.0, 112.0]
