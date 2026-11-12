"""etl_stg_db FK 正規化テスト。"""

from __future__ import annotations

import unittest

from src.scripts.data import etl_stg_db as etl


class TestEtlStgDbNormalize(unittest.TestCase):
    def test_normalize_fk_id(self):
        self.assertIsNone(etl._normalize_fk_id(None))
        self.assertIsNone(etl._normalize_fk_id(""))
        self.assertIsNone(etl._normalize_fk_id("   "))
        self.assertEqual(etl._normalize_fk_id("01091"), "01091")
        self.assertEqual(etl._normalize_fk_id(" 01091 "), "01091")


class TestFilterRacesNeedingSync(unittest.TestCase):
    def test_skip_complete_with_results(self):
        race_ids = ["202607181101", "202607181102"]
        pg = {
            "202607181101": {"has_race": True, "entry_cnt": 12, "finisher_cnt": 12},
            "202607181102": {"has_race": True, "entry_cnt": 10, "finisher_cnt": 0},
        }
        shutuba = set(race_ids)
        results = {"202607181101", "202607181102"}
        to_sync, skipped = etl.filter_races_needing_sync(
            race_ids, pg, shutuba, results, skip_if_complete=True
        )
        self.assertEqual(to_sync, ["202607181102"])
        self.assertEqual(skipped["202607181101"], "pg_complete")

    def test_resync_when_undrawn_entries_remain_after_result(self):
        rid, rid2 = "202611081101", "202611081102"
        pg = {rid: {"has_race": True, "entry_cnt": 33, "finisher_cnt": 16, "undrawn_cnt": 17},
              rid2: {"has_race": True, "entry_cnt": 16, "finisher_cnt": 16, "undrawn_cnt": 0}}
        to_sync, skipped = etl.filter_races_needing_sync([rid, rid2], pg, {rid, rid2}, {rid, rid2}, skip_if_complete=True)
        self.assertEqual(to_sync, [rid])
        self.assertEqual(skipped[rid2], "pg_complete")
        # 結果が未公開の（これからの）レースは、登録段階でも読み直さない
        to_sync, skipped = etl.filter_races_needing_sync([rid], pg, {rid}, set(), skip_if_complete=True)
        self.assertEqual(to_sync, [])

    def test_card_is_drawn(self):
        self.assertFalse(etl._card_is_drawn({"entries": [{"horse_number": 0}, {"horse_number": None}]}))
        self.assertTrue(etl._card_is_drawn({"entries": [{"horse_number": 0}, {"horse_number": 3}]}))
        self.assertFalse(etl._card_is_drawn({}))

    def test_sync_missing_entries(self):
        rid = "202607191101"
        to_sync, skipped = etl.filter_races_needing_sync(
            [rid],
            {},
            {rid},
            set(),
            skip_if_complete=True,
        )
        self.assertEqual(to_sync, [rid])
        self.assertEqual(skipped, {})

    def test_skip_no_gcs_shutuba(self):
        rid = "202607191101"
        to_sync, skipped = etl.filter_races_needing_sync(
            [rid],
            {},
            set(),
            set(),
            skip_if_complete=True,
        )
        self.assertEqual(to_sync, [])
        self.assertEqual(skipped[rid], "no_gcs_shutuba")

    def test_force_sync_all_with_shutuba(self):
        rid = "202607191101"
        to_sync, skipped = etl.filter_races_needing_sync(
            [rid],
            {"202607191101": {"has_race": True, "entry_cnt": 12, "finisher_cnt": 12}},
            {rid},
            {rid},
            skip_if_complete=False,
        )
        self.assertEqual(to_sync, [rid])
        self.assertEqual(skipped, {})


class TestGetTargetDates(unittest.TestCase):
    def test_explicit_dates(self):
        out = etl.get_target_dates(None, None, dates=["20260719", "20260718"])
        self.assertEqual(out, ["20260718", "20260719"])


if __name__ == "__main__":
    unittest.main()
