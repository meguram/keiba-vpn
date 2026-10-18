"""smartrc_prerace（出走前の値の不変スナップショット）。"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.scraper import smartrc_prerace as sp

JST = ZoneInfo("Asia/Tokyo")


class _Storage:
    def __init__(self):
        self.d = {}

    def load(self, cat, key, bypass_cache=False):
        return self.d.get((cat, key))

    def save(self, cat, key, doc):
        self.d[(cat, key)] = doc
        return True


DATA = {"rcode": "2026101804010101", "runners": [{"hcode": "2020100001", "blinker": "1", "old_pr": "A"}]}


class PhaseTest(unittest.TestCase):
    def test_phase_for(self):
        eve = datetime(2026, 10, 17, 18, 0, tzinfo=JST)
        self.assertEqual(sp.phase_for("20261018", "10:05", eve), "eve")
        self.assertEqual(sp.phase_for("2026-10-18", "10:05", datetime(2026, 10, 18, 9, 0, tzinfo=JST)), "t60")
        self.assertIsNone(sp.phase_for("20261018", "10:05", datetime(2026, 10, 18, 9, 45, tzinfo=JST)))     # 発走 30 分前を過ぎたら残さない
        self.assertIsNone(sp.phase_for("20261018", "10:05", datetime(2026, 10, 18, 17, 0, tzinfo=JST)))    # 出走後の取得
        self.assertIsNone(sp.phase_for("20261017", "10:05", datetime(2026, 10, 18, 7, 0, tzinfo=JST)))     # 翌日の取得
        self.assertIsNone(sp.phase_for("", "", eve))
        self.assertIsNone(sp.phase_for("20261018", "", datetime(2026, 10, 18, 9, 0, tzinfo=JST)))           # 当日で発走時刻不明

    def test_snapshot_is_immutable_and_post_race_is_skipped(self):
        st = _Storage()
        eve = datetime(2026, 10, 17, 18, 0, tzinfo=JST)
        self.assertEqual(sp.save_snapshot(st, "202604010101", DATA, "20261018", "10:05", eve), "eve")
        first = st.d[("smartrc_prerace", "202604010101")]
        self.assertEqual(first["phases"]["eve"]["runners"][0]["blinker"], "1")
        changed = {"rcode": DATA["rcode"], "runners": [{"hcode": "2020100001", "blinker": "0"}]}
        self.assertIsNone(sp.save_snapshot(st, "202604010101", changed, "20261018", "10:05", eve))          # 同じ phase は上書きしない
        self.assertEqual(st.d[("smartrc_prerace", "202604010101")]["phases"]["eve"]["runners"][0]["blinker"], "1")
        self.assertIsNone(sp.save_snapshot(st, "202604010102", DATA, "20261018", "10:05", datetime(2026, 10, 18, 17, 0, tzinfo=JST)))
        self.assertNotIn(("smartrc_prerace", "202604010102"), st.d)
        t60 = sp.save_snapshot(st, "202604010101", DATA, "20261018", "10:05", datetime(2026, 10, 18, 9, 0, tzinfo=JST))
        self.assertEqual(t60, "t60")
        self.assertEqual(set(st.d[("smartrc_prerace", "202604010101")]["phases"]), {"eve", "t60"})

    def test_disabled_by_default(self):
        import os

        os.environ.pop("KEIBA_EVE_SMARTRC_SNAPSHOT", None)
        self.assertFalse(sp.enabled())


if __name__ == "__main__":
    unittest.main()


class VerifyCompareTest(unittest.TestCase):
    def test_compare_runners(self):
        from src.scripts.data.verify_equipment_snapshot import compare_runners

        pre = [{"hcode": "a", "blinker": "1", "old_pr": "A"}, {"hcode": "b", "blinker": "0", "old_pr": "C"}]
        post = [{"hcode": "a", "blinker": "1", "old_pr": "B"}, {"hcode": "b", "blinker": "1", "old_pr": "C"}, {"hcode": "c", "blinker": "0"}]
        r = compare_runners(pre, post, ["blinker", "old_pr"])
        self.assertEqual(r["blinker"], [2, 1])
        self.assertEqual(r["old_pr"], [2, 1])
