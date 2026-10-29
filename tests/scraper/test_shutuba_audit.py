"""過去の出馬表の取りこぼし検査（shutuba_audit）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from src.scraper import entries_phase as EP
from src.scraper import shutuba_audit as SA


def _card(numbers: list[int]) -> dict:
    return {"date": "2026-05-24", "entries": [{"horse_id": str(i), "horse_number": n} for i, n in enumerate(numbers)]}


class _Storage:
    def __init__(self, lists: dict, cards: dict) -> None:
        self.lists, self.cards = lists, cards

    def list_keys(self, cat):
        return list(self.lists)

    def load(self, cat, key, bypass_cache=False):
        return (self.lists if cat == "race_lists" else self.cards).get(key)


class _Queue:
    def __init__(self) -> None:
        self.specs: list = []

    def bulk_add_jobs(self, specs):
        self.specs += specs
        return {"created": len(specs)}


class TestShutubaAudit(unittest.TestCase):
    def setUp(self) -> None:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = mock.patch.dict(os.environ, {"KEIBA_SHUTUBA_AUDIT_STATE": str(Path(d.name) / "st.json")})
        p.start()
        self.addCleanup(p.stop)
        races = [{"race_id": f"2026040108{r:02d}", "venue": "新潟", "round": r, "race_name": "x"} for r in (1, 2, 3, 4)]
        self.storage = _Storage(
            {"20260524": {"races": races}},
            {"202604010801": _card([0, 0, 0]), "202604010802": _card([1, 2, 3]), "202604010803": None, "202604010804": _card([])},
        )
        self.now = datetime(2026, 5, 27, 6, 40, tzinfo=EP.JST)

    def test_is_stale(self) -> None:
        self.assertTrue(SA.is_stale(None))
        self.assertTrue(SA.is_stale(_card([])))
        self.assertTrue(SA.is_stale(_card([0, 0])))
        self.assertFalse(SA.is_stale(_card([1, 2])))
        self.assertFalse(SA.is_stale(_card([1, 0])))  # 取消馬だけ 0 のレースは対象にしない

    def test_dry_run_lists_without_enqueue(self) -> None:
        q = _Queue()
        res = SA.run_audit(days=7, storage=self.storage, queue=q, now=self.now)
        self.assertEqual(sorted(res["race_ids"]), ["202604010801", "202604010803", "202604010804"])
        self.assertEqual(q.specs, [])

    def test_enqueue_and_attempt_cap(self) -> None:
        q = _Queue()
        for _ in range(SA.MAX_ATTEMPTS):
            res = SA.run_audit(days=7, enqueue=True, storage=self.storage, queue=q, now=self.now)
            self.assertEqual(res["to_enqueue"], 3)
        self.assertEqual(len(q.specs), 3 * SA.MAX_ATTEMPTS)
        res = SA.run_audit(days=7, enqueue=True, storage=self.storage, queue=q, now=self.now)
        self.assertEqual(res["to_enqueue"], 0)
        self.assertEqual(sorted(res["gave_up"]), ["202604010801", "202604010803", "202604010804"])
        self.assertTrue(all(s["tasks"] == ["race_shutuba"] and s["smart_skip"] is False for s in q.specs))

    def _direct(self, fetch, cap=SA.DIRECT_MAX_PER_RUN):
        return SA.run_audit(days=7, enqueue=True, storage=self.storage, queue=_Queue(), now=self.now,
                            direct=True, fetch=fetch, direct_cap=cap)

    def test_direct_fetches_in_place_without_queue(self) -> None:
        q, calls = _Queue(), []
        res = SA.run_audit(days=7, enqueue=True, storage=self.storage, queue=q, now=self.now,
                           direct=True, fetch=lambda rid, d: calls.append((rid, d)) or [])
        self.assertEqual(q.specs, [])
        self.assertEqual(sorted(c[0] for c in calls), ["202604010801", "202604010803", "202604010804"])
        self.assertEqual(res["direct"]["errors"], [])
        self.assertEqual(len(res["direct"]["fetched"]), 3)

    def test_direct_per_run_cap_defers_the_rest(self) -> None:
        calls = []
        res = self._direct(lambda rid, d: calls.append(rid) or [], cap=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(res["direct"]["deferred"], 1)

    def test_direct_attempt_cap_gives_up(self) -> None:
        for _ in range(SA.MAX_ATTEMPTS):
            self._direct(lambda rid, d: ["boom"])
        calls = []
        res = self._direct(lambda rid, d: calls.append(rid) or [])
        self.assertEqual(calls, [])
        self.assertEqual(res["to_enqueue"], 0)
        self.assertEqual(len(res["gave_up"]), 3)

    def test_direct_stops_on_access_restriction_and_reports_errors(self) -> None:
        calls = []

        def fetch(rid, d):
            calls.append(rid)
            return ["race_shutuba: access_restriction"]

        res = self._direct(fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["direct"]["fetched"], [])
        self.assertIn("access_restriction", res["direct"]["errors"][0])

    def test_only_past_dates_in_window(self) -> None:
        keys = ["20260510", "20260524", "20260527", "20260601"]
        out = SA.past_dates(keys, self.now.date(), 7)
        self.assertEqual(out, ["20260524"])


if __name__ == "__main__":
    unittest.main()
