"""requirement_row_catalog と trace ペイロードのスキーマ整合。"""

from __future__ import annotations

import unittest

from src.scraper import schemas
from src.scraper.requirement_row_catalog import (
    NETKEIBA_REQUIREMENT_ROWS,
    build_trace_payload,
    primary_id_for_scope,
    row_ids_netkeiba,
    trace_storage_key,
)


class TestRequirementRowCatalog(unittest.TestCase):
    def test_row_ids_unique(self) -> None:
        ids = [s.row_id for s in NETKEIBA_REQUIREMENT_ROWS()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_row_ids_frozen_set_matches(self) -> None:
        self.assertEqual(len(row_ids_netkeiba()), len(NETKEIBA_REQUIREMENT_ROWS()))

    def test_trace_key_shape(self) -> None:
        tk = trace_storage_key("race", "202309030811", "nk_speed_index")
        self.assertEqual(tk, "race_202309030811_nk_speed_index")

    def test_primary_id(self) -> None:
        self.assertEqual(
            primary_id_for_scope("race", "R1", "H1", "D1"),
            "R1",
        )

    def test_build_trace_validates_all_rows(self) -> None:
        r, h, d = "202309030811", "2019105219", "20230625"
        for spec in NETKEIBA_REQUIREMENT_ROWS():
            payload = build_trace_payload(spec, race_id=r, horse_id=h, date_fmt=d)
            vr = schemas.validate("requirement_row_trace", payload)
            self.assertTrue(
                vr.get("passed"),
                msg=f"{spec.row_id}: {vr}",
            )


if __name__ == "__main__":
    unittest.main()
