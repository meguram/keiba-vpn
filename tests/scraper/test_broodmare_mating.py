"""母馬ページ(own.netkeiba)の種付け情報: パース・産駒とのマッチング・保存の検証。"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.scraper import schemas
from src.scraper.broodmare_mating import (
    build_broodmare_mating_record,
    dam_id_from_ancestors,
    foal_birth_year,
    is_mating_lookup_eligible,
    match_foal_mating,
    parse_broodmare_mating,
    sire_from_ancestors,
)
from src.scraper.queue_tasks import validate_tasks_for_kind
from src.scraper.run import ScraperRunner

FIXTURE = Path(__file__).parent / "fixtures" / "own_netkeiba_broodmare_record_2017101417.html"
DAM = "2017101417"
WIN_BRIGHT = "2014100222"
FIERCE = "2015105075"


def _html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _ancestors(sire_id=WIN_BRIGHT, sire_name="ウインブライト", dam_id=DAM) -> list[dict]:
    return [
        {"generation": 1, "position": 0, "name": sire_name, "horse_id": sire_id, "sex": "牡"},
        {"generation": 1, "position": 1, "name": "ウインアルカンナ", "horse_id": dam_id, "sex": "牝"},
    ]


class ParseTest(unittest.TestCase):
    def test_parses_real_page(self):
        recs = parse_broodmare_mating(_html())
        self.assertEqual(
            [(r["mating_year"], r["mating_date"], r["sire_id"], r["sire_name"]) for r in recs],
            [
                (2025, "2025-05-17", WIN_BRIGHT, "ウインブライト"),
                (2024, "2024-05-06", WIN_BRIGHT, "ウインブライト"),
                (2023, "2023-04-15", FIERCE, "フィエールマン"),
            ],
        )

    def test_no_table_or_garbage_gives_empty(self):
        self.assertEqual(parse_broodmare_mating("<html><body>なし</body></html>"), [])
        self.assertEqual(parse_broodmare_mating(""), [])

    def test_invalid_date_is_dropped(self):
        html = _html().replace("5月17日", "2月31日")
        self.assertEqual([r["mating_year"] for r in parse_broodmare_mating(html)], [2024, 2023])

    def test_parsed_record_passes_schema(self):
        rec = build_broodmare_mating_record(DAM, parse_broodmare_mating(_html()))
        self.assertTrue(schemas.validate("broodmare_mating", rec)["passed"])


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.m = parse_broodmare_mating(_html())

    def test_mating_year_is_birth_year_minus_one(self):
        self.assertEqual(match_foal_mating(self.m, "2026100001", sire_id=WIN_BRIGHT)["mating_date"], "2025-05-17")
        self.assertEqual(match_foal_mating(self.m, "2025100001", sire_id=WIN_BRIGHT)["mating_date"], "2024-05-06")
        self.assertEqual(match_foal_mating(self.m, "2024100001", sire_id=FIERCE)["mating_date"], "2023-04-15")

    def test_birth_year_from_horse_id(self):
        self.assertEqual(foal_birth_year("2026100001"), 2026)
        self.assertIsNone(foal_birth_year("abc"))
        self.assertTrue(is_mating_lookup_eligible("2024100001"))
        self.assertFalse(is_mating_lookup_eligible("2023100001"))  # 2022年種付けは表の範囲外

    def test_out_of_range_and_no_record(self):
        self.assertEqual(match_foal_mating(self.m, "2019100001")["mating_match"], "out_of_range")
        r = match_foal_mating(self.m, "2030100001")
        self.assertEqual((r["mating_match"], r["mating_date"]), ("no_record", None))

    def test_sire_mismatch_is_not_adopted(self):
        r = match_foal_mating(self.m, "2026100001", sire_id="9999999999", sire_name="別の種牡馬")
        self.assertEqual((r["mating_match"], r["mating_date"]), ("sire_mismatch", None))

    def test_falls_back_to_sire_name_when_ids_missing(self):
        r = match_foal_mating(self.m, "2025100001", sire_name="ウインブライト")
        self.assertEqual(r["mating_date"], "2024-05-06")

    def test_unknown_sire_adopts_single_candidate(self):
        self.assertEqual(match_foal_mating(self.m, "2025100001")["mating_date"], "2024-05-06")

    def test_multiple_same_year_disambiguated_by_sire(self):
        twin = self.m + [{"mating_year": 2025, "mating_date": "2025-06-01", "sire_name": "X", "sire_id": "1"}]
        self.assertEqual(match_foal_mating(twin, "2026100001")["mating_match"], "ambiguous")
        self.assertEqual(match_foal_mating(twin, "2026100001", sire_id="1")["mating_date"], "2025-06-01")
        self.assertEqual(match_foal_mating(twin, "2026100001", sire_id=WIN_BRIGHT)["mating_date"], "2025-05-17")

    def test_ancestor_helpers(self):
        a = _ancestors()
        self.assertEqual(dam_id_from_ancestors(a), DAM)
        self.assertEqual(sire_from_ancestors(a), (WIN_BRIGHT, "ウインブライト"))
        self.assertEqual(dam_id_from_ancestors([]), "")


class _Store:
    def __init__(self):
        self.data: dict[tuple[str, str], dict] = {}
        self._base_dir = Path("/nonexistent")

    def load(self, cat, key):
        return self.data.get((cat, key))

    def save(self, cat, key, data):
        self.data[(cat, key)] = data
        return True


class _Client:
    def __init__(self, html=None, error=None):
        self.html, self.error, self.urls = html, error, []

    def fetch(self, url, **kw):
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.html


class _Archive:
    def __init__(self):
        self.saved = []

    def save(self, cat, key, html):
        self.saved.append((cat, key))


def _runner(client) -> ScraperRunner:
    r = ScraperRunner.__new__(ScraperRunner)
    r.storage, r.client, r.archive = _Store(), client, _Archive()
    return r


class RunnerTest(unittest.TestCase):
    def test_attach_sets_yyyy_mm_dd_and_caches_dam_page(self):
        client = _Client(_html())
        r = _runner(client)
        rec = {"horse_id": "2025100001", "ancestors": _ancestors()}
        r.attach_mating_date("2025100001", rec)
        self.assertEqual(rec["mating_date"], "2024-05-06")
        self.assertEqual((rec["mating_year"], rec["mating_match"], rec["mating_dam_id"]), (2024, "matched", DAM))
        self.assertTrue(schemas.validate("horse_pedigree_5gen", rec)["passed"])
        self.assertIn(("broodmare_mating", DAM), r.storage.data)
        self.assertEqual(client.urls, [f"https://own.netkeiba.com/db/db_broodmare_record.html?id={DAM}"])
        # 同じ母馬の別産駒（半きょうだい）は保存済みの母馬レコードを使い、再取得しない
        rec2 = {"horse_id": "2024100009", "ancestors": _ancestors(sire_id=FIERCE, sire_name="フィエールマン")}
        r.attach_mating_date("2024100009", rec2)
        self.assertEqual(rec2["mating_date"], "2023-04-15")
        self.assertEqual(len(client.urls), 1)

    def test_old_foal_makes_no_request(self):
        client = _Client(_html())
        r = _runner(client)
        rec = {"horse_id": "2019100001", "ancestors": _ancestors()}
        r.attach_mating_date("2019100001", rec)
        self.assertNotIn("mating_date", rec)
        self.assertEqual(client.urls, [])

    def test_fetch_failure_does_not_break_pedigree_record(self):
        r = _runner(_Client(error=RuntimeError("boom")))
        rec = {"horse_id": "2025100001", "ancestors": _ancestors()}
        r.attach_mating_date("2025100001", rec)
        self.assertNotIn("mating_match", rec)  # 次回の再試行が可能な状態のまま

    def test_block_suspect_error_is_propagated(self):
        r = _runner(_Client(error=RuntimeError("blocked")))
        with patch("src.scraper.scrape_access_pause.is_block_suspect_http_400", return_value=True):
            with self.assertRaises(RuntimeError):
                r.attach_mating_date("2025100001", {"ancestors": _ancestors()})

    def test_sire_mismatch_leaves_no_date(self):
        r = _runner(_Client(_html()))
        rec = {"ancestors": _ancestors(sire_id="1", sire_name="別の種牡馬")}
        r.attach_mating_date("2025100001", rec)
        self.assertEqual(rec["mating_match"], "sire_mismatch")
        self.assertNotIn("mating_date", rec)

    def test_stale_dam_record_without_year_is_refetched_but_fresh_one_is_not(self):
        client = _Client(_html())
        r = _runner(client)
        old = build_broodmare_mating_record(DAM, [])
        old["fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        r.storage.data[("broodmare_mating", DAM)] = old
        r.scrape_broodmare_mating(DAM, for_foal_id="2025100001")
        self.assertEqual(len(client.urls), 1)
        r.storage.data[("broodmare_mating", DAM)] = build_broodmare_mating_record(DAM, [])
        r.scrape_broodmare_mating(DAM, for_foal_id="2025100001")
        self.assertEqual(len(client.urls), 1)

    def test_resave_adds_mating_to_existing_pedigree(self):
        r = _runner(_Client(_html()))
        r.storage.data[("horse_pedigree_5gen", "2025100001")] = {"horse_id": "2025100001", "ancestors": _ancestors()}
        with patch("src.scraper.run._update_local_pedigree_10gen") as upd:
            out = r.scrape_horse_mating_date("2025100001")
        self.assertEqual(out["mating_date"], "2024-05-06")
        self.assertEqual(r.storage.data[("horse_pedigree_5gen", "2025100001")]["mating_date"], "2024-05-06")
        upd.assert_called_once()

    def test_queue_task_is_registered_for_horse_jobs(self):
        self.assertIsNone(validate_tasks_for_kind("horse", ["horse_mating_date"]))


if __name__ == "__main__":
    unittest.main()
