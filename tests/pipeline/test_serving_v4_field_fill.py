"""履歴の行（V4History._field_history）で、出馬表が空の馬場・天気などを結果表の値で埋める（2026-11 の出馬表は馬場が空のまま残る）。"""

from __future__ import annotations

import pandas as pd

import pytest

from src.pipeline.features import serving_v4 as sv4


@pytest.fixture(autouse=True)
def _fill_on(monkeypatch):
    """既定はオフ。このファイルのテストは、断りが無ければオン（本番に入れた後の形）で確かめる。"""
    monkeypatch.setenv(sv4.FIELD_FILL_ENV, "1")


def _tables(tmp_path, sb_going, rr_going, grade_sb="", year="2026", date="2026-11-22"):
    d = tmp_path / year
    d.mkdir(parents=True)
    base = {"race_id": [f"{year}05050501"] * 3, "horse_id": ["h1", "h2", "h3"], "horse_number": [1, 2, 3], "date": date,
            "venue_code": "05", "surface": "芝", "distance": 1600, "field_size": 3, "bracket_number": [1, 2, 3], "round": 1}
    pd.DataFrame({**base, "track_condition": sb_going, "weather": [None, "晴", ""], "start_time": ["", "10:00", "10:00"],
                  "grade": grade_sb, "jockey_id": ["j1", "", "j3"], "jockey_name": ["", "▲小林美", "丹内"]}).to_parquet(d / "race_shutuba_flat.parquet")
    pd.DataFrame({"race_id": base["race_id"], "horse_number": [1, 2, 3], "finish_position": [1, 2, 3], "track_condition": rr_going,
                  "weather": ["曇", "曇", None], "start_time": ["10:00"] * 3, "grade": "2勝", "jockey_id": ["j9", "j2", "j3"],
                  "jockey_name": ["丹内祐次", "小林美駒", "丹内祐次"]}).to_parquet(d / "race_result_flat.parquet")


def _hist(tmp_path, monkeypatch, train_path=None, years=(2026,)):
    h = object.__new__(sv4.V4History)
    h.train_path, h.tables, h.years = train_path, tmp_path, list(years)
    monkeypatch.setattr(h, "_style_table", lambda: pd.DataFrame(columns=["race_id", "horse_id", "a_style", "f3_r"]))
    return h._field_history()


def test_blank_shutuba_columns_are_filled_from_result_in_shutuba_notation(tmp_path, monkeypatch):
    _tables(tmp_path, ["", None, "良"], ["稍重", "不良", "良"])
    out = _hist(tmp_path, monkeypatch).sort_values("horse_number")
    assert len(out) == 3
    assert out["track_condition"].tolist() == ["稍", "不", "良"]          # 結果表の「稍重・不良」を出馬表の書き方に直す
    assert out["weather"].fillna("").tolist() == ["曇", "晴", ""]          # 空の行だけ埋める・結果も空なら空のまま
    assert out["start_time"].tolist() == ["10:00", "10:00", "10:00"]
    assert out["jockey_id"].tolist() == ["j1", "j2", "j3"]
    assert out["jockey_name"].tolist() == ["", "▲小林美", "丹内"]          # 書き方が違うので埋めない
    assert not [c for c in out.columns if c.endswith("__r")]


def test_grade_is_not_filled(tmp_path, monkeypatch):
    """grade は 2025 年の出馬表でも 2・3 勝が空で、学習表もその形。結果の値で埋めない。"""
    _tables(tmp_path, ["良"] * 3, ["良"] * 3, grade_sb="")
    out = _hist(tmp_path, monkeypatch)
    assert out["grade"].fillna("").eq("").all()


def test_train_table_rows_are_untouched(tmp_path, monkeypatch):
    """学習表（2025 年まで）の行はそのまま。2026 年は学習表の最終日より後の行だけ足す。"""
    tp = tmp_path / "train.parquet"
    pd.DataFrame({"date": ["2025-12-28"], "race_id": ["202506050901"], "horse_id": ["t1"], "horse_number": [1], "venue_code": ["06"],
                  "surface": ["芝"], "track_condition": [""], "weather": [""], "finish_position": [1]}).to_parquet(tp)
    _tables(tmp_path, ["", "", ""], ["稍重"] * 3)
    out = _hist(tmp_path, monkeypatch, train_path=tp, years=(2025, 2026))
    tr = out[out["race_id"] == "202506050901"]
    assert len(tr) == 1 and tr["track_condition"].iloc[0] == "" and tr["weather"].iloc[0] == ""
    assert out[out["race_id"] != "202506050901"]["track_condition"].eq("稍").all() and len(out) == 4


def test_fill_blank_from_result_keeps_present_values():
    m = pd.DataFrame({"track_condition": ["良", "", None, ""], "track_condition__r": ["重", "稍重", "不良", None]})
    out = sv4.fill_blank_from_result(m, ["track_condition"])
    assert out["track_condition"].fillna("").tolist() == ["良", "稍", "不", ""] and list(out.columns) == ["track_condition"]


def test_switch_off_keeps_blank_like_before(tmp_path, monkeypatch):
    """KEIBA_FIELD_FILL_FROM_RESULT=0 は直す前と同じ（影の候補の主の線は満期まで直す前の版）。"""
    monkeypatch.setenv(sv4.FIELD_FILL_ENV, "0")
    _tables(tmp_path, ["", None, "良"], ["稍重", "不良", "良"])
    h = object.__new__(sv4.V4History)
    h.train_path, h.tables, h.years = None, tmp_path, [2026]
    monkeypatch.setattr(h, "_style_table", lambda: pd.DataFrame(columns=["race_id", "horse_id", "a_style", "f3_r"]))
    out = h._field_history().sort_values("horse_number")
    assert out["track_condition"].fillna("").tolist() == ["", "", "良"]
    assert h.field_fill_info["rule"] == "skew_fill_none" and h.field_fill_info["2026"] == {}


def test_fill_info_counts_filled_rows(tmp_path, monkeypatch):
    _tables(tmp_path, ["", None, "良"], ["稍重", "不良", "良"])
    h = object.__new__(sv4.V4History)
    h.train_path, h.tables, h.years = None, tmp_path, [2026]
    monkeypatch.setattr(h, "_style_table", lambda: pd.DataFrame(columns=["race_id", "horse_id", "a_style", "f3_r"]))
    h._field_history()
    assert h.field_fill_info["rule"] == "skew_fill_v1" and h.field_fill_info["2026"]["track_condition"] == 2
    assert h.field_fill_info["2026"]["weather"] == 1 and len(h.field_fill_info["rule_sha"]) == 12


def test_candidate_runner_keeps_pre_fix_version():
    """影の候補（forward_stop_r2 の主の線、11/21〜）は満期まで直す前の版で動かす。"""
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[2] / "config/candidate_runtime.json").read_text(encoding="utf-8"))
    assert cfg["env"][sv4.FIELD_FILL_ENV] == "0"


def test_default_is_off(monkeypatch):
    """承認の前に本番の入力が変わらないよう、既定はオフ（ホストの .env で 1 にして入れる）。"""
    monkeypatch.delenv(sv4.FIELD_FILL_ENV, raising=False)
    assert not sv4.field_fill_enabled()
    assert sv4.field_fill_rule_sha() == sv4.field_fill_rule_sha(False) != sv4.field_fill_rule_sha(True)


def test_fill_info_counts_only_kept_rows(tmp_path, monkeypatch):
    """埋めた行の数は、履歴に残る行（着順あり・中央の場）だけで数える。出馬表にもとからある「稍重」はそのまま残る。"""
    d = tmp_path / "2026"
    d.mkdir(parents=True)
    base = {"race_id": ["202605050501", "202605050501", "202630050501"], "horse_id": ["h1", "h2", "h3"], "horse_number": [1, 2, 1],
            "date": "2026-11-22", "venue_code": ["05", "05", "30"], "surface": "芝", "distance": 1600, "field_size": 2, "bracket_number": [1, 2, 1], "round": 1}
    pd.DataFrame({**base, "track_condition": ["稍重", "", ""]}).to_parquet(d / "race_shutuba_flat.parquet")
    pd.DataFrame({"race_id": base["race_id"], "horse_number": [1, 2, 1], "finish_position": [1, 0, 1],
                  "track_condition": ["良", "重", "重"]}).to_parquet(d / "race_result_flat.parquet")
    h = object.__new__(sv4.V4History)
    h.train_path, h.tables, h.years = None, tmp_path, [2026]
    monkeypatch.setattr(h, "_style_table", lambda: pd.DataFrame(columns=["race_id", "horse_id", "a_style", "f3_r"]))
    out = h._field_history()
    assert out["track_condition"].tolist() == ["稍重"]                   # 取消（着順 0）と地方の場（30）は落ち、もとの値は残る
    assert h.field_fill_info["2026"]["track_condition"] == 0
