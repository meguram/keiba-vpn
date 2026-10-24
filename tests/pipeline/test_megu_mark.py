import pytest

from src.pipeline.inference import expected_megu as em
from src.pipeline.inference import megu_mark as mm

CFG = {"version": 1, "min_runners": 6, "require_line": True,
       "tiers": [{"key": "top", "label": "めぐ◎", "rank_max": 1, "rank_min": 1}, {"key": "good", "label": "めぐ○", "rank_max": 3, "rank_min": 2}]}


def _run(date, v, low=False):
    return (date, v, low, "芝", 1600.0, None, None, None, None, "r" + date)


def _entries(values):
    return [{"key": i, "horse_number": i + 1, "race_date": "20250601", "runs": [_run("20250301", v)] if v is not None else []} for i, v in enumerate(values)]


def test_top_and_good_marks_need_the_class_line_and_rank():
    out = mm.compute_marks(_entries([60, 55, 50, 45, 40, 35, None]), 48.0, CFG)
    assert out[0]["key"] == "top" and out[0]["label"] == "めぐ◎" and out[0]["rank"] == 1
    assert out[1]["key"] == "good" and out[2]["key"] == "good"
    assert out[3]["key"] is None and out[3]["rank"] == 4 and "未満" in out[3]["reason"]            # 4 位は印なし（突破ライン未満でもある）
    assert out[6]["key"] is None and "初出走" in out[6]["unavailable"]


def test_below_the_line_gets_no_mark_even_when_ranked_first():
    out = mm.compute_marks(_entries([40, 38, 36, 35, 34, 33]), 50.0, CFG)
    assert all(v["key"] is None for v in out.values()) and out[0]["margin"] == pytest.approx(-10.0, abs=0.1)


def test_no_line_or_small_field_gives_unavailable_reason():
    out = mm.compute_marks(_entries([60, 55, 50, 45, 40, 35]), None, CFG)
    assert all(v["key"] is None and "基準" in v["unavailable"] for v in out.values())
    small = mm.compute_marks(_entries([60, 55, 50]), 40.0, CFG)
    assert all(v["key"] is None and "6頭未満" in v["unavailable"] for v in small.values())


def test_ties_are_broken_by_horse_number_and_future_runs_are_not_used():
    es = _entries([60, 60, 55, 50, 45, 40])
    es[2]["runs"].append(_run("20250701", 99))          # 今走（6/1）より後の走は使わない
    out = mm.compute_marks(es, 40.0, CFG)
    assert out[0]["key"] == "top" and out[1]["key"] == "good" and out[2]["value"] == 55.0


def test_low_confidence_runs_weigh_less():
    a = mm.v3b_value([_run("20250301", 60), _run("20250201", 40)], "20250601")[0]
    b = mm.v3b_value([_run("20250301", 60, low=True), _run("20250201", 40)], "20250601")[0]
    assert b < a                                        # 最新の走が低信頼だと重みが下がる


def test_no_config_returns_empty():
    assert mm.compute_marks(_entries([60, 50]), 40.0, {}) == {}


def test_attach_expected_megu_adds_megu_mark(tmp_path, monkeypatch):
    import pandas as pd

    from src.pipeline.inference import class_benchmark as cb

    em._cache.clear()
    d = tmp_path / "2025"
    d.mkdir()
    horses = [(f"H{i}", 70.0 - 5 * i) for i in range(7)]
    pd.DataFrame([{"race_id": f"r{i}", "horse_id": h, "megu_index_v3a_abs_robust": v, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False}
                  for i, (h, v) in enumerate(horses)]).to_parquet(d / "megu_index_flat.parquet")
    pd.DataFrame([{"race_id": f"r{i}", "horse_id": h, "date": "2025-02-10"} for i, (h, _) in enumerate(horses)]).to_parquet(d / "race_result_flat.parquet")
    monkeypatch.setattr(cb, "load_table", lambda path=None: {"period": "2023-2025", "cells": {"3歳|未勝利": {"n": 99, "median": 55.0, "p25": 48.0, "p75": 60.0}}})
    card = {"race_class": "サラ系３歳 未勝利", "grade": "未勝利", "race_name": "3歳未勝利", "surface": "芝", "distance": 1600}
    resp = {"predictions": [{"horse_number": i + 1, "horse_id": h} for i, (h, _) in enumerate(horses)]}
    out = em.attach_expected_megu(resp, "2025-05-01", base=tmp_path, card=card)
    marks = [p.get("megu_mark") for p in out["predictions"]]
    assert marks[0]["key"] == "top" and marks[1]["key"] == "good" and marks[2]["key"] == "good" and marks[3]["key"] is None
    assert marks[6]["rank"] == 7 and marks[0]["line"] == 48.0
    out2 = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "H0"}]}, "2025-05-01", base=tmp_path, card=card)
    assert out2["predictions"][0]["megu_mark"]["unavailable"]            # 1 頭だけ（6 頭未満）は印なし
