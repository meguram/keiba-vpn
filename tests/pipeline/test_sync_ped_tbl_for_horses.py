"""pipeline.sync_ped_tbl_for_horses の unittest。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline.horse_entity_layout import ped_parquet_path
from pipeline.sync_ped_tbl_for_horses import (
    normalize_horse_id_list,
    sync_ped_tbl_after_shutuba,
    sync_ped_tbl_for_horses,
)


class TestSyncPedTblForHorses(unittest.TestCase):
    def test_normalize_dedup(self) -> None:
        self.assertEqual(normalize_horse_id_list(["2019100001", "2019100001", "0", ""]), ["2019100001"])

    def test_writes_missing_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ped = tdp / "data" / "local" / "horse_pedigree_5gen" / "2019"
            ped.mkdir(parents=True)
            rec = {
                "ancestors": [
                    {"generation": 1, "position": 0, "name": "S", "horse_id": "1990110001"},
                ]
            }
            (ped / "2019100001.json").write_text(json.dumps(rec), encoding="utf-8")

            st = sync_ped_tbl_for_horses(
                ["2019100001", "2099999999"],
                base_dir=tdp,
                pedigree_json_dir=ped.parent,
                merge_gen5_sires=False,
                skip_if_parquet_exists=True,
            )
            self.assertEqual(st["written"], 1)
            self.assertEqual(st["missing_json"], 1)
            outp = ped_parquet_path("2019100001", tdp)
            self.assertTrue(outp.is_file())

    def test_skip_existing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ped = tdp / "data" / "local" / "horse_pedigree_5gen" / "2019"
            ped.mkdir(parents=True)
            rec = {"ancestors": [{"generation": 1, "position": 0, "horse_id": "1990110001"}]}
            (ped / "2019100001.json").write_text(json.dumps(rec), encoding="utf-8")
            outp = ped_parquet_path("2019100001", tdp)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_bytes(b"x" * 40)

            st = sync_ped_tbl_for_horses(
                ["2019100001"],
                base_dir=tdp,
                pedigree_json_dir=ped.parent,
                merge_gen5_sires=False,
            )
            self.assertEqual(st["skipped_existing_parquet"], 1)
            self.assertEqual(st["written"], 0)

    def test_sync_after_shutuba_off_without_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            os.environ.pop("KEIBA_SYNC_PED_TBL_ON_SHUTUBA", None)
            st = sync_ped_tbl_after_shutuba(
                {"entries": [{"horse_id": "2019100001"}]},
                base_dir=tdp,
            )
            self.assertIsNone(st)

    def test_sync_after_shutuba_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            os.environ["KEIBA_SYNC_PED_TBL_ON_SHUTUBA"] = "1"
            os.environ["KEIBA_PED_TBL_MERGE_GEN5"] = "0"
            try:
                ped = tdp / "data" / "local" / "horse_pedigree_5gen" / "2019"
                ped.mkdir(parents=True)
                rec = {"ancestors": [{"generation": 1, "position": 0, "horse_id": "1990110001"}]}
                (ped / "2019100001.json").write_text(json.dumps(rec), encoding="utf-8")

                st = sync_ped_tbl_after_shutuba(
                    {"entries": [{"horse_id": "2019100001"}]},
                    base_dir=tdp,
                )
                assert st is not None
                self.assertEqual(st["written"], 1)
            finally:
                os.environ.pop("KEIBA_SYNC_PED_TBL_ON_SHUTUBA", None)
                os.environ.pop("KEIBA_PED_TBL_MERGE_GEN5", None)


if __name__ == "__main__":
    unittest.main()
