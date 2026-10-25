"""出馬表を『常に最新』で出すための保存・キャッシュの挙動。"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.scraper.storage import HybridStorage
from src.utils import cache_epoch


def _card(numbers: list[int]) -> dict:
    return {"race_id": "202601010101", "date": "2026-01-04",
            "entries": [{"horse_id": str(i), "horse_name": f"h{i}", "horse_number": n} for i, n in enumerate(numbers)]}


class TestShutubaFreshness(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.base = Path(self._dir.name)
        with mock.patch.dict(os.environ, {"GCS_BUCKET": ""}):
            self.st = HybridStorage(base_dir=str(self.base), bucket_name="")

    def test_refresh_existing_local_cache_overwrites_old_file(self) -> None:
        p = self.st._local_cache_path("race_shutuba", "202601010101")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_card([0, 0])), encoding="utf-8")
        self.st._refresh_existing_local_cache("race_shutuba", "202601010101", _card([1, 2]))
        self.assertEqual([e["horse_number"] for e in json.loads(p.read_text())["entries"]], [1, 2])

    def test_refresh_does_not_create_missing_file(self) -> None:
        self.st._refresh_existing_local_cache("race_shutuba", "202601010101", _card([1, 2]))
        self.assertFalse(self.st._local_cache_path("race_shutuba", "202601010101").exists())

    def test_shutuba_disk_cache_expires_in_minutes_but_others_do_not(self) -> None:
        for cat in ("race_shutuba", "race_result"):
            p = self.st._local_cache_path(cat, "202601010101")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(_card([1])), encoding="utf-8")
            old = time.time() - 1800
            os.utime(p, (old, old))
        self.assertIsNone(self.st._read_local_cache("race_shutuba", "202601010101"))
        self.assertIsNotNone(self.st._read_local_cache("race_result", "202601010101"))

    def test_memory_ttl_is_short_only_for_shutuba(self) -> None:
        self.assertLessEqual(self.st._mem_ttl("race_shutuba"), 300.0)
        self.assertEqual(self.st._mem_ttl("race_result"), self.st._mem_cache_ttl)

    def test_final_card_is_not_overwritten_by_unnumbered_fetch(self) -> None:
        with mock.patch.object(self.st, "load", return_value=_card([1, 2, 3])), \
                mock.patch.object(self.st, "_save_inner", return_value=True) as inner:
            self.assertTrue(self.st.save("race_shutuba", "202601010101", _card([0, 0, 0])))
        inner.assert_not_called()

    def test_numbered_fetch_replaces_unnumbered_card(self) -> None:
        with mock.patch.object(self.st, "load", return_value=_card([0, 0, 0])), \
                mock.patch.object(self.st, "_save_inner", return_value=True) as inner:
            self.assertTrue(self.st.save("race_shutuba", "202601010101", _card([1, 2, 3])))
        inner.assert_called_once()

    def test_final_card_can_be_updated_by_a_final_fetch(self) -> None:
        with mock.patch.object(self.st, "load", return_value=_card([1, 2, 3])), \
                mock.patch.object(self.st, "_save_inner", return_value=True) as inner:
            self.st.save("race_shutuba", "202601010101", _card([1, 2]))
        inner.assert_called_once()


class TestTrailingEpochBump(unittest.TestCase):
    def test_throttled_save_is_flushed_after_the_interval(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cache_epoch.json"
            with mock.patch.object(cache_epoch, "_FILE", f), mock.patch.object(cache_epoch, "_BUMP_MIN_INTERVAL_SEC", 0.3), \
                    mock.patch.dict(cache_epoch._seen, {"bumped": 0.0}), mock.patch.dict(cache_epoch._pending, {"timer": None}):
                cache_epoch.bump_for_save("race_shutuba")
                self.assertEqual(json.loads(f.read_text())["reason"], "save:race_shutuba")
                cache_epoch.bump_for_save("race_shutuba")  # 間引かれる
                self.assertEqual(json.loads(f.read_text())["reason"], "save:race_shutuba")
                time.sleep(1.5)
                self.assertTrue(json.loads(f.read_text())["reason"].endswith(":trailing"))

    def test_flush_pending_bumps_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cache_epoch.json"
            with mock.patch.object(cache_epoch, "_FILE", f), mock.patch.object(cache_epoch, "_BUMP_MIN_INTERVAL_SEC", 60.0), \
                    mock.patch.dict(cache_epoch._seen, {"bumped": 0.0}), mock.patch.dict(cache_epoch._pending, {"timer": None}):
                cache_epoch.bump_for_save("race_shutuba")
                cache_epoch.bump_for_save("race_shutuba")
                cache_epoch.flush_pending()
                self.assertEqual(json.loads(f.read_text())["reason"], "save:flush")


if __name__ == "__main__":
    unittest.main()
