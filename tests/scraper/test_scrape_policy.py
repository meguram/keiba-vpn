"""scraper.scrape_policy の unittest。"""

from __future__ import annotations

import os
import unittest


class TestScrapePolicy(unittest.TestCase):
    def test_resolve_enqueue_overwrite_smart_skip(self):
        from scraper.scrape_policy import resolve_enqueue_overwrite_smart_skip

        ow, ss = resolve_enqueue_overwrite_smart_skip(
            {"job_kind": "horse", "target_id": "1", "tasks": ["horse_profile"], "overwrite": True},
        )
        self.assertTrue(ow)
        self.assertFalse(ss)

        ow2, ss2 = resolve_enqueue_overwrite_smart_skip(
            {"job_kind": "horse", "target_id": "1", "tasks": ["horse_profile"], "smart_skip": False},
        )
        self.assertFalse(ow2)
        self.assertFalse(ss2)

        prev = os.environ.get("SCRAPE_DEFAULT_OVERWRITE")
        try:
            os.environ["SCRAPE_DEFAULT_OVERWRITE"] = "1"
            ow3, ss3 = resolve_enqueue_overwrite_smart_skip(
                {"job_kind": "horse", "target_id": "1", "tasks": ["horse_profile"]},
            )
            self.assertTrue(ow3)
            self.assertFalse(ss3)
        finally:
            if prev is None:
                os.environ.pop("SCRAPE_DEFAULT_OVERWRITE", None)
            else:
                os.environ["SCRAPE_DEFAULT_OVERWRITE"] = prev

    def test_effective_smart_skip_respects_missing_overwrite_key(self):
        from scraper.scrape_policy import effective_smart_skip_for_queue_job

        prev = os.environ.get("SCRAPE_DEFAULT_OVERWRITE")
        try:
            os.environ["SCRAPE_DEFAULT_OVERWRITE"] = "1"
            self.assertFalse(
                effective_smart_skip_for_queue_job(
                    {"smart_skip": True},
                ),
            )
        finally:
            if prev is None:
                os.environ.pop("SCRAPE_DEFAULT_OVERWRITE", None)
            else:
                os.environ["SCRAPE_DEFAULT_OVERWRITE"] = prev


if __name__ == "__main__":
    unittest.main()
