"""3連単オッズ（race_trifecta_odds）: type=8 応答のパース・保存・環境変数ゲート。"""

from __future__ import annotations

import itertools
import os
import unittest
from unittest import mock

from src.scraper import auto_scrape_queue as q
from src.scraper.parsers import OddsParser
from src.scraper.run import ScraperRunner


def _payload(n: int = 5, status: str = "result") -> dict:
    raw = {}
    for k, (a, b, c) in enumerate(itertools.permutations(range(1, n + 1), 3)):
        raw[f"{a:02d}{b:02d}{c:02d}"] = [f"{1000 + k * 7.5:,.1f}", "0.0", str(k + 1)]
    return {"status": status, "data": {"official_datetime": "2025-01-01 10:00:00", "odds": {"8": raw}}}


class _Store:
    def __init__(self):
        self.d = {}

    def exists(self, cat, key):
        return (cat, key) in self.d

    def load(self, cat, key, **kw):
        return self.d.get((cat, key))

    def save(self, cat, key, data, **kw):
        self.d[(cat, key)] = data


class _Resp:
    status_code = 200

    def __init__(self, j):
        self._j = j

    def json(self):
        return self._j


class _Session:
    def __init__(self, j):
        self.j, self.calls = j, 0

    def get(self, url, timeout=0):
        self.calls += 1
        return _Resp(self.j)


def _runner(j):
    r = ScraperRunner.__new__(ScraperRunner)
    r.storage = _Store()
    r.odds_parser = OddsParser()
    r.client = mock.Mock(_session=_Session(j))
    return r


class ParseTests(unittest.TestCase):
    def test_parse_full(self):
        d = OddsParser.parse_trifecta_payload(_payload(5), "R")
        self.assertEqual((d["n_head"], d["n_combos"], d["expected_combos"]), (5, 60, 60))
        self.assertEqual(d["trifecta"][0][:3], [1, 2, 3])
        self.assertEqual(d["trifecta"][0][3], 1000.0)
        self.assertEqual(d["trifecta"][-1][:3], [5, 4, 3])

    def test_comma_odds_and_pop(self):
        p = {"status": "result", "data": {"odds": {"8": {"010203": ["11,835.2", "0.0", "935"]}}}}
        d = OddsParser.parse_trifecta_payload(p, "R")
        self.assertEqual(d["trifecta"], [[1, 2, 3, 11835.2, 935]])

    def test_malformed_rows_skipped(self):
        p = _payload(4)
        p["data"]["odds"]["8"].update({"0102": ["5", "0", "1"], "0101ab": ["5", "0", "1"], "010103": ["5", "0", "1"], "000203": ["5", "0", "1"], "040302": ["x", "0", "1"]})
        d = OddsParser.parse_trifecta_payload(p, "R")
        self.assertEqual(d["n_combos"], 23)  # 040302 はオッズが読めず除外（期待 24 との差で欠けを検知）

    def test_unreadable_returns_none(self):
        self.assertIsNone(OddsParser.parse_trifecta_payload({"status": "yoso", "data": {}}, "R"))
        self.assertIsNone(OddsParser.parse_trifecta_payload({"status": "result", "data": {"odds": {"8": {}}}}, "R"))
        self.assertIsNone(OddsParser.parse_trifecta_payload({"status": "result", "data": {"odds": {}}}, "R"))
        self.assertIsNone(OddsParser.parse_trifecta_payload(None, "R"))

    def test_scratched_horse_shows_deficit(self):
        p = _payload(5)
        for k in [k for k in p["data"]["odds"]["8"] if "03" in (k[0:2], k[2:4], k[4:6])]:
            del p["data"]["odds"]["8"][k]
        d = OddsParser.parse_trifecta_payload(p, "R")
        self.assertEqual((d["n_head"], d["n_combos"], d["expected_combos"]), (4, 24, 24))


class ScrapeTests(unittest.TestCase):
    def test_save_and_skip_existing(self):
        r = _runner(_payload(4))
        self.assertIsNotNone(r.scrape_trifecta_odds("R1"))
        self.assertEqual(r.client._session.calls, 1)
        r.scrape_trifecta_odds("R1")
        self.assertEqual(r.client._session.calls, 1)

    def test_unreadable_not_saved(self):
        r = _runner({"status": "result", "data": {"odds": {}}})
        self.assertIsNone(r.scrape_trifecta_odds("R1"))
        self.assertEqual(r.storage.d, {})

    def test_phase_history_never_lost(self):
        r = _runner(_payload(4))
        r.scrape_trifecta_odds("R1", phase="t60")
        r.scrape_trifecta_odds("R1", phase="t15")
        r.scrape_trifecta_odds("R1", skip_existing=False)  # phase 無しの再取得は final として積む
        doc = r.storage.d[("race_trifecta_odds", "R1")]
        self.assertEqual(doc["phase"], "final")
        self.assertEqual({s["phase"] for s in doc["snapshots"]}, {"t60", "t15"})
        self.assertEqual(len(doc["snapshots"][0]["trifecta"]), 24)


class GateTests(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {"KEIBA_ODDS_TRIFECTA": ""}):
            for lst in (q.ODDS_T60_TASKS, q.T15_RACE_TASKS, q.EVENING_TASKS):
                self.assertFalse([t for t in q.enabled_tasks(lst) if "trifecta" in t])

    def test_on(self):
        with mock.patch.dict(os.environ, {"KEIBA_ODDS_TRIFECTA": "1"}):
            self.assertIn("race_trifecta_odds_t60", q.enabled_tasks(q.ODDS_T60_TASKS))
            self.assertIn("race_trifecta_odds_final", q.enabled_tasks(q.EVENING_TASKS))
            self.assertEqual(q.enabled_tasks(q.T15_RACE_TASKS), list(q.T15_RACE_TASKS))


if __name__ == "__main__":
    unittest.main()
