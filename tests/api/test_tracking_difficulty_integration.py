"""追走難度 API / 位置取り反映のスモーク（サーバ localhost:8000 要）。"""

from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request

BASE = os.environ.get("KEIBA_TEST_API_BASE", "http://localhost:8000")
RACE_ID = os.environ.get("KEIBA_TEST_RACE_ID", "202605021012")


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


@unittest.skipUnless(
    os.environ.get("KEIBA_RUN_LIVE_API_TESTS", "1") == "1",
    "KEIBA_RUN_LIVE_API_TESTS=0 でスキップ",
)
class TestTrackingDifficultyLiveApi(unittest.TestCase):
    def test_page_loads(self):
        req = urllib.request.Request(f"{BASE}/tracking-difficulty")
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode()
        self.assertEqual(resp.status, 200)
        self.assertIn("追走難度", html)
        self.assertIn("position_flow", html)  # JS 内
        self.assertIn("buildPositionMapHtml", html)

    def test_api_position_flow_for_ui(self):
        try:
            data = _get(f"/api/race/{RACE_ID}/tracking-difficulty")
        except urllib.error.URLError as e:
            self.skipTest(f"server not reachable: {e}")

        self.assertGreater(len(data.get("entries") or []), 0)
        pf = data.get("position_flow") or []
        self.assertGreater(len(pf), 0, "position_flow が空 — UI に位置取りマップが出ない")
        row = pf[0]
        self.assertIn("positions", row)
        for stage in ("early", "mid", "late"):
            self.assertIn(stage, row["positions"])
            self.assertGreater(row["positions"][stage].get("position", 0), 0)
        self.assertIn(data.get("pace_prediction", {}).get("pace_type", ""), (
            "ハイ", "ミドル", "スロー",
        ))


if __name__ == "__main__":
    unittest.main()
