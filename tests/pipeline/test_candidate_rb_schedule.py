"""影のランナーの材料の予約（candidate_rb_schedule）: 暦の読み、前夜の対象日とめぐ指数を待つ日、当日の朝の RB の作り直しの判断。外部接続なし。"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.pipeline.inference import candidate_rb_schedule as s

D = [date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 21), date(2026, 11, 22), date(2026, 11, 23), date(2026, 11, 28), date(2026, 11, 29)]


def test_race_days_reads_calendar_including_holiday(tmp_path):
    (tmp_path / "race_calendar_2026.json").write_text(json.dumps({"race_days": [
        {"date": "2026-11-23", "weekday": "祝日・月曜"}, {"date": "2026-11-22"}, {"date": "bad"}, {}]}), encoding="utf-8")
    assert s.race_days([2026, 2027], tmp_path) == [date(2026, 11, 22), date(2026, 11, 23)]


def test_eve_target_only_before_race_day():
    assert s.eve_target(date(2026, 11, 22), D) == date(2026, 11, 23)     # 祝日の月曜の前夜
    assert s.eve_target(date(2026, 11, 27), D) == date(2026, 11, 28)
    assert s.eve_target(date(2026, 11, 24), D) is None


def test_megu_wait_only_on_first_day_of_week():
    assert s.megu_wait_days(date(2026, 11, 21), D) == [date(2026, 11, 14), date(2026, 11, 15)]
    assert s.megu_wait_days(date(2026, 11, 28), D) == [date(2026, 11, 21), date(2026, 11, 22), date(2026, 11, 23)]   # 祝日の月曜まで
    assert s.megu_wait_days(date(2026, 11, 22), D) == []
    assert s.megu_wait_days(date(2026, 11, 14), D) == []                 # 暦に前の開催日が無い


def test_morning_waits_for_results_then_rebuilds_then_skips():
    t = date(2026, 11, 22)
    assert s.morning_action(t, D, "2026-11-15", None)[0] == "wait"              # 07:00 の書き出しの前（11/22 の朝に起きたこと）
    assert s.morning_action(t, D, "2026-11-15", "2026-11-21")[0] == "rebuild"
    assert s.morning_action(t, D, "2026-11-21", "2026-11-21")[0] == "skip"      # 作り直した後
    assert s.morning_action(date(2026, 11, 24), D, None, None)[0] == "skip"      # 開催日でない
    assert s.morning_action(date(2026, 11, 14), D, None, "2026-11-08")[0] == "skip"   # 暦に前の開催日が無い
    assert s.morning_action(date(2026, 11, 28), D, "2026-11-22", "2026-11-23")[0] == "rebuild"   # 前の開催日は祝日の 11/23


def test_results_complete_compares_with_card(tmp_path):
    y = tmp_path / "2026"
    y.mkdir()
    pd.DataFrame({"race_id": list("ABCD"), "date": ["2026/11/21"] + ["2026-11-21"] * 3}).to_parquet(y / "race_shutuba_flat.parquet")
    pd.DataFrame({"race_id": ["A"], "date": ["2026-11-21"]}).to_parquet(y / "race_result_flat.parquet")
    assert not s.results_complete(date(2026, 11, 21), tmp_path)                 # 橋渡しが一部だけ（欠け 3 件）
    pd.DataFrame({"race_id": ["A", "B", "C"], "date": ["2026-11-21"] * 3}).to_parquet(y / "race_result_flat.parquet")
    assert s.results_complete(date(2026, 11, 21), tmp_path)                     # 中止の 1 レースは欠けてよい
    assert not s.results_complete(date(2026, 11, 22), tmp_path)
    (y / "race_shutuba_flat.parquet").unlink()
    assert s.results_complete(date(2026, 11, 21), tmp_path)                     # 出馬表が無いときは 1 レースでも入った扱い


def test_read_cut(tmp_path):
    assert s.read_cut(tmp_path) is None
    (tmp_path / "params.json").write_text(json.dumps({"cut": "2026-11-22"}))
    assert s.read_cut(tmp_path) == "2026-11-22"
