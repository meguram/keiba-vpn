import pandas as pd

from src.scraper.parse_audit import audit_frame, problems


def _row(rid, **kw):
    base = {"race_id": rid, "surface": "芝", "distance": 1600, "weather": "晴", "track_condition": "良", "start_time": "10:00",
            "field_size": 16, "date": "2026-10-10", "venue": "東京", "direction": "右"}
    base.update(kw)
    return base


def test_detects_unparsed_course_and_ignores_hurdle_direction():
    df = pd.DataFrame([_row("a"), _row("b", surface="", distance=0, direction=""), _row("c", surface="障", direction="")])
    rep = audit_frame(df)
    assert rep["missing"]["surface"] == 1 and rep["missing"]["distance"] == 1
    assert rep["samples"]["surface"] == ["b"]
    assert "direction(障害以外)" in rep["missing"]      # b は障害ではない
    assert rep["missing"]["direction(障害以外)"] == 1   # c（障害）は数えない
    assert problems(rep)


def test_clean_frame_has_no_problems():
    assert problems(audit_frame(pd.DataFrame([_row("a"), _row("b", distance=2000)]))) == {}
