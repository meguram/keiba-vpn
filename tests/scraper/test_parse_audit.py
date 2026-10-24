import pandas as pd

from src.scraper.parse_audit import audit_frame, problems


def _row(rid, **kw):
    base = {"race_id": rid, "surface": "芝", "distance": 1600, "weather": "晴", "track_condition": "良", "start_time": "10:00",
            "field_size": 16, "date": "2026-10-10", "venue": "東京", "direction": "右"}
    base.update(kw)
    return base


def test_detects_unparsed_course():
    df = pd.DataFrame([_row("a"), _row("b", surface="", distance=0)])
    rep = audit_frame(df)
    assert rep["missing"]["surface"] == 1 and rep["missing"]["distance"] == 1
    assert rep["samples"]["surface"] == ["b"]
    assert problems(rep)


def test_clean_frame_has_no_problems():
    assert problems(audit_frame(pd.DataFrame([_row("a", race_class="1勝"), _row("b", race_class="2勝", distance=2000)]))) == {}


def test_hurdle_is_blank_class_and_direction_but_blank_class_with_direction_is_flagged():
    hurdle = _row("h", direction="", race_class="", distance=2860)
    ok = pd.DataFrame([_row("a", race_class="1勝"), hurdle])
    assert problems(audit_frame(ok)) == {}
    broken = pd.DataFrame([_row(f"m{i}", race_class="") for i in range(5)] + [hurdle])
    assert audit_frame(broken)["missing"]["race_class(方向あり)"] == 5
    assert "race_class(方向あり)" in problems(audit_frame(broken))
    assert problems(audit_frame(pd.DataFrame([_row("x", race_class=""), _row("y", race_class="1勝")]))) == {}   # 1〜2 レースは許容
