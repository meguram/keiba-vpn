import pandas as pd
import pytest

from src.pipeline.inference import expected_megu as em


def _runs(*vals, start="20250101", step=30, low=()):
    from datetime import date, timedelta

    d0 = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    return [((d0 + timedelta(days=step * i)).strftime("%Y%m%d"), v, i in low) for i, v in enumerate(vals)]


def test_recent_runs_weigh_more_and_only_before_race_day():
    runs = _runs(40.0, 50.0, 60.0)
    r = em.expected_from_runs(runs, "20250601")
    w = [1.0, em.DECAY, em.DECAY ** 2]
    exp = (60 * w[0] + 50 * w[1] + 40 * w[2]) / sum(w)
    assert r["value"] == round(exp, 1) and r["n_runs"] == 3 and r["confidence"] == "normal"
    assert r["latest_run_date"] == "2025-03-02"
    # 対象レース日以降の走は使わない（リーク防止）
    r2 = em.expected_from_runs(runs, "20250215")
    assert r2["n_runs"] == 2 and r2["latest_run_date"] == "2025-01-31"


def test_only_latest_five_runs_and_low_confidence_weighs_less():
    runs = _runs(10, 20, 30, 40, 50, 60, 70)
    r = em.expected_from_runs(runs, "20260101")
    assert r["n_runs"] == 5 and r["value"] > 50          # 古い 10・20 は入らない
    a = em.expected_from_runs(_runs(40.0, 60.0, 60.0), "20260101")
    b = em.expected_from_runs(_runs(40.0, 60.0, 60.0, low=(2,)), "20260101")
    assert b["value"] < a["value"] and b["low_conf_runs"] == 1     # 最新の走が低信頼だと重みが下がる


def test_few_runs_are_reference_and_zero_runs_is_none_and_gap_is_stale():
    assert em.expected_from_runs([], "20250601") is None
    assert em.expected_from_runs(_runs(float("nan")), "20250601") is None
    r = em.expected_from_runs(_runs(50.0, 52.0), "20250601")
    assert r["confidence"] == "reference" and r["range"] is not None
    one = em.expected_from_runs(_runs(50.0), "20250601")
    assert one["confidence"] == "reference" and one["range"] is None
    old = em.expected_from_runs(_runs(50.0, 51.0, 52.0, start="20230101"), "20250601")
    assert old["confidence"] == "stale"


def test_rank_and_gap_in_seconds():
    items = {1: {"value": 55.0}, 2: {"value": 50.0}, 3: None, 4: {"value": 57.5}}
    em.rank_and_gap(items)
    assert items[4]["rank"] == 1 and items[4]["gap_sec"] == 0.0
    assert items[1]["rank"] == 2 and items[1]["gap_sec"] == 0.25
    assert items[2]["rank"] == 3 and items[2]["gap_sec"] == 0.75 and items[3] is None


def _write_tables(base, rows, dates):
    d = base / "2025"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "megu_index_flat.parquet")
    pd.DataFrame(dates).to_parquet(d / "race_result_flat.parquet")


def test_attach_uses_past_runs_flags_unreflected_and_caches(tmp_path):
    em._cache.clear()
    _write_tables(
        tmp_path,
        [{"race_id": f"r{i}", "horse_id": "A", "megu_index_v3a_abs_robust": 40.0 + 5 * i, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}
         for i in range(3)]
        + [{"race_id": "rx", "horse_id": "A", "megu_index_v3a_abs_robust": 99.0, "megu_v3a_status": "steeplechase", "megu_v3a_low_confidence": False},
           {"race_id": "r0", "horse_id": "B", "megu_index_v3a_abs_robust": 50.0, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}],
        [{"race_id": "r0", "date": "2025-01-10"}, {"race_id": "r1", "date": "2025-02-10"}, {"race_id": "r2", "date": "2025-03-10"},
         {"race_id": "rx", "date": "2025-03-20"}],
    )
    resp = {"predictions": [{"horse_number": 1, "horse_id": "A"}, {"horse_number": 2, "horse_id": "B"}, {"horse_number": 3, "horse_id": "Z"}]}
    past = {"entries": [{"horse_id": "A", "past_races": [{"date": "2025/03/10"}, {"date": "2025/04/05"}]}]}
    out = em.attach_expected_megu(resp, "2025-05-01", base=tmp_path, past_doc=past)
    a, b, z = (p["expected_megu"] for p in out["predictions"])
    assert a["n_runs"] == 3 and 45 < a["value"] < 50 and a["rank"] == 2     # steeplechase の走(99)は除外
    assert a["latest_unreflected"] is True and a["latest_past_date"] == "2025-04-05"             # 馬柱に未反映の走がある
    assert b["n_runs"] == 1 and b["confidence"] == "reference" and "latest_unreflected" not in b
    assert b["rank"] == 1 and b["gap_sec"] == 0.0 and a["gap_sec"] > 0
    assert z is None
    assert out["expected_megu_meta"]["data_through"] == "2025-03-20" and out["expected_megu_meta"]["n_known"] == 2
    n = len(em._cache)
    em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path)
    assert len(em._cache) == n                     # 同じファイルは読み直さない


def test_legacy_columns_are_used_when_v3a_columns_are_missing(tmp_path):
    em._cache.clear()
    _write_tables(
        tmp_path,
        [{"race_id": "r0", "horse_id": "A", "megu_index": 48.0, "computation_status": "valid"}],
        [{"race_id": "r0", "date": "2025-01-10"}],
    )
    out = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path)
    assert out["predictions"][0]["expected_megu"]["value"] == 48.0


def test_no_predictions_is_returned_unchanged():
    resp = {"predictions": []}
    assert em.attach_expected_megu(resp, "2025-05-01") is resp and "expected_megu_meta" not in resp


# ── 条件の変化 ───────────────────────────────────────────────────────

def _run(d, v, surf="芝", dist=1600.0, venue="東京", level=2, w=55.0, age=3, low=False):
    return (d, v, low, surf, dist, venue, level, w, age)


def _today(**kw):
    t = {"surface": "芝", "distance": 1600.0, "venue": "東京", "level": 2, "weight": 55.0, "age": 4}
    t.update(kw)
    return t


def test_same_conditions_have_no_change_and_no_adjustment():
    runs = [_run("20250301", 50.0), _run("20250201", 52.0), _run("20250101", 48.0)]
    r = em.expected_from_runs(runs, "20250401", _today())
    assert r["change"] == "none" and r["reasons"] == [] and r["adjust"] == [] and r["value"] == r["base_value"]


def test_surface_distance_class_rest_and_venue_changes_are_detected_with_reasons():
    runs = [_run("20240101", 50.0), _run("20231201", 50.0)]
    t = _today(surface="ダ", distance=2200.0, venue="京都", level=5)
    r = em.expected_from_runs(runs, "20250101", t)
    assert r["change"] == "large"
    txt = " ".join(r["reasons"])
    assert "ダート初出走" in txt and "距離 +600m" in txt and "3クラス昇級" in txt and "休み明け" in txt and "京都は初めて" in txt


def test_model_adds_learned_adjustment_widens_range_and_is_capped():
    runs = [_run("20241201", 50.0), _run("20241101", 50.5), _run("20241001", 49.5)]
    model = {"coef": {"to_dirt": -4.0, "class_up": 1.0}, "center": {}, "intercept": 0.0, "cap": 8.0, "resid_sd": {"none": 3.0, "some": 4.0, "large": 6.0},
             "group_thresholds": {"some": 1.0, "large": 3.0}}
    plain = em.expected_from_runs(runs, "20250101", _today(), model)
    dirt = em.expected_from_runs(runs, "20250101", _today(surface="ダ"), model)
    assert dirt["value"] == round(dirt["base_value"] - 4.0, 1) and dirt["change"] == "large" and dirt["range"] >= 6.0
    assert dirt["adjust"][0]["key"] == "to_dirt" and dirt["adjust"][0]["delta"] == -4.0
    assert plain["adjust"] == [] and plain["range"] < dirt["range"]
    huge = em.expected_from_runs(runs, "20250101", _today(surface="ダ", level=9), {**model, "coef": {"to_dirt": -40.0}})
    assert huge["value"] == round(huge["base_value"] - 8.0, 1)


def test_same_surface_runs_weigh_more_when_the_model_says_so():
    runs = [_run("20241201", 40.0, surf="ダ"), _run("20241101", 60.0, surf="芝"), _run("20241001", 60.0, surf="芝")]
    model = {"weights": {"surface_mismatch": 0.2, "distance_scale": 1e9}}
    a = em.expected_from_runs(runs, "20250101", _today(), None)["base_value"]
    b = em.expected_from_runs(runs, "20250101", _today(), model)["base_value"]
    assert b > a                                    # 今回は芝。ダートの低い走を軽くする


def test_recent_lists_actual_runs_newest_first_with_low_flag():
    r = em.expected_from_runs(_runs(40.0, 50.0, 60.0, 70.0, 80.0, 90.0, low=(5,)), "20260101")
    assert [x["value"] for x in r["recent"]] == [90.0, 80.0, 70.0, 60.0, 50.0]      # 実測値そのまま（新しい順・最大5走）
    assert [x["low"] for x in r["recent"]] == [True, False, False, False, False]
    dates = [x["date"] for x in r["recent"]]
    assert dates == sorted(dates, reverse=True)


def test_recent_runs_carry_race_details_from_the_result_table(tmp_path):
    em._cache.clear()
    _write_tables(
        tmp_path,
        [{"race_id": "202501010101", "horse_id": "A", "megu_index_v3a_abs_robust": 55.0, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False},
         {"race_id": "202502010101", "horse_id": "A", "megu_index_v3a_abs_robust": 61.0, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}],
        [{"race_id": "202501010101", "horse_id": "A", "date": "2025-01-10", "race_name": "新馬", "round": 1, "finish_position": 3, "popularity": 2,
          "field_size": 14, "jockey_name": "騎手A", "last_3f": 34.56, "passing_order": "3-3", "weight": 460, "weight_change": 2, "track_condition": "良"},
         {"race_id": "202502010101", "horse_id": "A", "date": "2025-02-10", "race_name": "未勝利", "round": 5, "finish_position": 1, "popularity": float("nan")}],
    )
    out = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path)
    new, old = out["predictions"][0]["expected_megu"]["recent"]
    assert new["race_name"] == "未勝利" and new["finish"] == 1 and "pop" not in new          # 欠損は付けない
    assert old["race_name"] == "新馬" and old["finish"] == 3 and old["pop"] == 2 and old["l3f"] == 34.6
    assert old["passing"] == "3-3" and old["bw"] == 460 and old["bw_diff"] == 2 and old["going"] == "良" and old["field"] == 14


def test_attach_adds_class_benchmark_when_the_class_is_known(tmp_path, monkeypatch):
    from src.pipeline.inference import class_benchmark as cb

    em._cache.clear()
    _write_tables(tmp_path, [{"race_id": "202501010101", "horse_id": "A", "megu_index_v3a_abs_robust": 55.0, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}],
                  [{"race_id": "202501010101", "horse_id": "A", "date": "2025-01-10"}])
    monkeypatch.setattr(cb, "load_table", lambda path=None: {"period": "2023-2025", "cells": {"3歳|未勝利": {"n": 99, "median": 40.0, "p25": 35.0, "p75": 45.0}}})
    card = {"race_class": "サラ系３歳 未勝利", "grade": "未勝利", "race_name": "3歳未勝利", "surface": "芝", "distance": 1600}
    out = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path, card=card)
    b = out["class_benchmark"]
    assert b["label"] == "3歳未勝利" and b["line"] == 35.0 and b["median"] == 40.0 and b["n"] == 99 and b["period"] == "2023-2025"
    card2 = {"race_class": "", "grade": "", "race_name": "謎のレース", "surface": "芝", "distance": 1600}
    assert "class_benchmark" not in em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path, card=card2)


def test_recent_runs_get_age_and_class_label(tmp_path):
    em._cache.clear()
    _write_tables(
        tmp_path,
        [{"race_id": "202501010101", "horse_id": "A", "megu_index_v3a_abs_robust": 55.0, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}],
        [{"race_id": "202501010101", "horse_id": "A", "date": "2025-01-10", "race_name": "3歳以上1勝クラス", "race_class": "3歳以上1勝クラス", "grade": "", "finish_position": 2}],
    )
    out = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=tmp_path)
    assert out["predictions"][0]["expected_megu"]["recent"][0]["cls"] == "3歳以上1勝クラス"
