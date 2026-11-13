"""推論で読む履歴の年が、起点年（FIRST_YEAR。本番のバンドルの学習の起点）以降にそろっていることを確かめる。
2026-11-07〜08、2016〜2019 年の表ができたことで推論が 2016 年からの履歴を読み、2020 年起点で学習した本番のモデルとずれた。"""

from __future__ import annotations

import pandas as pd

from src.pipeline.models import training_years as ty


def _tables(tmp_path, years, with_file=True):
    for y in years:
        d = tmp_path / str(y)
        d.mkdir()
        if with_file:
            pd.DataFrame({"race_id": [f"{y}01010101"], "horse_number": [1], "horse_id": [f"h{y}"], "date": [f"{y}-01-05"],
                          "finish_position": [1]}).to_parquet(d / "race_result_flat.parquet")
    (tmp_path / "jma_weather_hourly_flat.parquet").write_bytes(b"")      # 年でないものは無視する
    return tmp_path


def test_serving_years_starts_at_first_year(tmp_path, monkeypatch):
    base = _tables(tmp_path, range(2016, 2027))
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    assert ty.serving_years(base) == list(range(2020, 2027))          # 2016〜2019 は読まず、2026 年は読む
    monkeypatch.setattr(ty, "FIRST_YEAR", 2016)
    assert ty.serving_years(base) == list(range(2016, 2027))


def test_serving_years_need_file(tmp_path, monkeypatch):
    base = _tables(tmp_path, range(2020, 2026))
    (tmp_path / "2026").mkdir()                                         # 年のフォルダだけあって結果の表が無い
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    assert ty.serving_years(base) == list(range(2020, 2027))
    assert ty.serving_years(base, need="race_result_flat.parquet") == list(range(2020, 2026))


def test_layer_a_history_skips_years_before_first_year(tmp_path, monkeypatch):
    from src.pipeline.features import serving_layer_a as sla

    base = _tables(tmp_path, range(2016, 2027))
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    h = sla.load_history(base)
    assert sorted(pd.to_datetime(h["date"]).dt.year.unique().tolist()) == list(range(2020, 2027))


def test_v4_history_default_years(tmp_path, monkeypatch):
    from src.pipeline.features import serving_v4 as sv4

    base = _tables(tmp_path, range(2016, 2027), with_file=False)
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    assert sv4.V4History(base).years == list(range(2020, 2027))
    assert sv4.V4History(base, years=[2018]).years == [2018]           # 明示した年はそのまま


def test_ped_history_years_include_current_year(tmp_path, monkeypatch):
    from src.pipeline.inference import v3_serving as v3

    base = _tables(tmp_path, range(2016, 2027))
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    assert v3.ped_history_years(base) == list(range(2020, 2027))       # 2026 年（学習年の後）も数える。2016〜2019 は読まない


def test_serving_years_empty_is_clear_error(tmp_path, monkeypatch):
    import pytest

    _tables(tmp_path, range(2016, 2020))
    monkeypatch.setattr(ty, "FIRST_YEAR", 2020)
    with pytest.raises(FileNotFoundError, match="2020 年以降"):
        ty.serving_years(tmp_path)
