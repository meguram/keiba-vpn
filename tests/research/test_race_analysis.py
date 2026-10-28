"""めぐ分析（出馬表 + 分析モジュール）。騎手×コースは、開催日より前の結果だけで、ID をキーに数えること。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.race_analysis import runs as runs_mod
from src.research.race_analysis.base import ModuleOutput
from src.research.race_analysis.service import analyze_race


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


def _row(rid, date, venue, surface, dist, fin, jid):
    return {"race_id": rid, "date": date, "venue": venue, "surface": surface, "distance": dist, "finish_position": fin, "jockey_id": jid}


@pytest.fixture()
def flat_dir(tmp_path, monkeypatch):
    rows = [
        _row("A1", "2026-01-05", "東京", "芝", 1600, 1, "J1"),
        _row("A2", "2026-01-12", "東京", "芝", 1600, 2, "J1"),
        _row("A3", "2026-01-19", "東京", "芝", 1600, 5, "J1"),
        _row("A4", "2026-02-01", "東京", "芝", 2000, 3, "J1"),        # 同競馬場・同芝ダ（距離違い）
        _row("A5", "2026-02-08", "中山", "芝", 1600, 1, "J1"),        # 同芝ダ（別の競馬場）
        _row("A6", "2026-02-15", "東京", "ダート", 1600, 1, "J1"),     # 芝ダが違う → 数えない
        _row("A7", "2026-03-01", "東京", "芝", 1600, 1, "J2"),
        _row("FUT", "2026-10-31", "東京", "芝", 1600, 1, "J1"),        # 本レース当日の結果は入れない（リーク防止）
        _row("LATER", "2026-11-01", "東京", "芝", 1600, 1, "J1"),
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    runs_mod._cache.update(sig=None, df=None)
    return tmp_path


def _storage(entries, **card):
    base = {"race_name": "テスト特別", "date": "2026-10-31", "venue": "東京", "round": 11, "surface": "芝", "distance": 1600, "entries": entries}
    base.update(card)
    return FakeStorage({("race_shutuba", "202605040111"): base})


def test_counts_only_before_race_date_by_jockey_id(flat_dir):
    st = _storage([
        {"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_name": "甲", "jockey_id": "J1"},
        {"horse_number": 2, "horse_name": "ニ", "horse_id": "h2", "jockey_name": "乙", "jockey_id": "J2"},
        {"horse_number": 3, "horse_name": "サン", "horse_id": "h3", "jockey_name": "丙", "jockey_id": "J9"},
    ])
    d = analyze_race(st, "202605040111")
    assert d["status"] == "ok" and d["provisional_entries"] is False
    m = d["modules"][0]
    assert m["id"] == "jockey_course" and m["status"] == "ok" and m["data_until"] == "2026-03-01"
    a = {e["horse_number"]: e["analysis"]["jockey_course"] for e in d["entries"]}
    j1 = a[1]
    assert j1["course_rec"] == [1, 1, 0, 1] and j1["course_n"] == 3                  # A1 A2 A3（FUT・LATER は除外）
    assert j1["course_win"] == pytest.approx(1 / 3) and j1["course_top2"] == pytest.approx(2 / 3) and j1["course_top3"] == pytest.approx(2 / 3)
    assert j1["venue_surface_n"] == 4 and j1["venue_surface_rec"] == [1, 1, 1, 1]    # + A4
    assert j1["surface_n"] == 5 and j1["surface_rec"] == [2, 1, 1, 1]                # + A5（ダートの A6 は入らない）
    assert a[2]["course_n"] == 1 and a[2]["course_win"] == 1.0                       # 別の騎手は混ざらない
    assert a[3]["course_n"] == 0 and a[3]["course_win"] is None                       # 実績が無い騎手は率を作り出さない


def test_missing_jockey_id_gives_empty_cells_not_a_crash(flat_dir):
    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_name": None, "jockey_id": ""}])
    e = analyze_race(st, "202605040111")["entries"][0]["analysis"]["jockey_course"]
    assert e["course_rec"] is None and e["course_n"] == 0 and e["course_win"] is None


def test_provisional_entries_without_horse_numbers_keep_order(flat_dir):
    st = _storage([
        {"horse_number": 0, "horse_name": "ロク", "horse_id": "h6", "jockey_id": "J1"},
        {"horse_number": 0, "horse_name": "シチ", "horse_id": "h7", "jockey_id": "J2"},
    ])
    d = analyze_race(st, "202605040111")
    assert d["provisional_entries"] is True and [e["horse_name"] for e in d["entries"]] == ["ロク", "シチ"]
    assert all(e["horse_number"] is None for e in d["entries"])


def test_no_entries(flat_dir):
    d = analyze_race(FakeStorage({}), "202605040101")
    assert d["status"] == "no_entries" and d["entries"] == [] and d["modules"] == []


def test_a_failing_module_does_not_break_the_others(flat_dir):
    class Boom:
        id, label, description = "boom", "壊れる分析", ""

        def applicable(self, ctx):
            return True

        def compute(self, ctx) -> ModuleOutput:
            raise RuntimeError("x")

    from src.research.race_analysis.registry import MODULES

    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"}])
    d = analyze_race(st, "202605040111", modules=(Boom(), *MODULES))
    assert [(m["id"], m["status"]) for m in d["modules"]] == [("boom", "error"), ("jockey_course", "ok")]
    assert "boom" not in d["entries"][0]["analysis"] and "jockey_course" in d["entries"][0]["analysis"]


def test_module_not_applicable_when_course_unknown(flat_dir):
    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"}], surface=None, distance=None)
    d = analyze_race(st, "202605040111")
    assert d["modules"][0]["status"] == "not_applicable"


def test_analysis_endpoint_is_member_only():
    from src.api.tier import member_only_path

    assert member_only_path("/api/v1/races/202605040111/analysis")
    assert not member_only_path("/api/v1/races/202605040111/analysis/extra")
