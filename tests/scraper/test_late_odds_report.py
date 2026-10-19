"""直前オッズの集計スクリプト（合成データのみ）。"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from src.scripts.data import late_odds_report as lr

JST = ZoneInfo("Asia/Tokyo")
POST = datetime(2026, 5, 2, 10, 5, tzinfo=JST)


def _doc(latest_phase, latest_odds, hist):
    d = {"phase": latest_phase, "snapshot_at": "2026-05-02T10:03:00+09:00",
         "entries": [{"horse_number": 1, "win_odds": latest_odds}], "snapshots": []}
    for ph, at, o in hist:
        d["snapshots"].append({"phase": ph, "snapshot_at": at, "entries": [{"horse_number": 1, "win_odds": o}]})
    return d


def test_empty_input_does_not_fail():
    a = lr.analyze([])
    assert a["races"] == 0 and a["missing_rate"]["t02"] is None
    assert "レース数 0" in lr._fmt(a)


def test_counts_missing_and_change_and_timing():
    full = _doc("t02", 2.5, [("t15", "2026-05-02T09:50:00+09:00", 3.0), ("t05", "2026-05-02T10:00:00+09:00", 2.8)])
    partial = _doc("t15", 3.0, [])
    a = lr.analyze([("A", POST, full), ("B", POST, partial), ("C", POST, None)])
    assert a["races"] == 3
    assert a["have"]["t02"] == 1 and a["have"]["t15"] == 2
    assert a["missing_rate"]["t02"] == round(1 - 1 / 3, 4)
    ch = a["log_odds_change"]["t15→t02"]
    assert ch["n"] == 1 and math.isclose(ch["median"], math.log(2.5 / 3.0))
    # t02 は 10:03（発走の 2 分前＝10:03）なのでずれ 0 秒、t05 は 10:00（5 分前）なのでずれ 0 秒
    assert a["timing_error_sec"]["t02"]["median"] == 0.0 and a["timing_error_sec"]["t05"]["median"] == 0.0


def test_phase_docs_prefers_latest_at_top_level():
    d = _doc("t02", 2.5, [("t02", "x", 9.9)])
    assert lr.win_odds(lr.phase_docs(d)["t02"]) == {1: 2.5}
