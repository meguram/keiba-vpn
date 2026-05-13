"""pipeline.build_horse_entity_store の unittest。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.build_horse_entity_store import (
    RESULT_TBL_DROP_NESTED_COLS,
    build_horse_entity_store,
    _flatten_pedigree_json,
    _flatten_pedigree_json_with_gen5_merge,
)


class TestBuildHorseEntityStore(unittest.TestCase):
    def test_flatten_pedigree_json(self):
        rec = {
            "horse_id": "2019100001",
            "ancestors": [
                {"generation": 1, "position": 0, "name": "S", "horse_id": "1990110001"},
                {"generation": 1, "position": 1, "name": "D", "horse_id": None},
            ],
        }
        df = _flatten_pedigree_json(rec, "2019100001")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["path_fm"], "F")
        self.assertEqual(df.iloc[0]["source"], "primary")
        self.assertEqual(int(df.loc[0, "is_male_pedigree_slot"]), 1)
        self.assertEqual(int(df["pedigree_max_generation_observed"].iloc[0]), 1)
        self.assertIn("ancestor_occurrence_global", df.columns)
        self.assertEqual(int(df.loc[0, "ancestor_occurrence_global"]), 1)

    def test_build_minimal(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            y = "2024"
            td_tables = tdp / "data" / "local" / "tables" / y
            td_tables.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "race_id": ["202401010101", "202401010102"],
                        "horse_id": ["2019100001", "2019100001"],
                        "finish_position": [1, 2],
                    }
                ),
                td_tables / "race_result_flat.parquet",
            )
            pq.write_table(
                pa.table({"race_id": ["202401010101"], "horse_id": ["2019100002"]}),
                td_tables / "race_shutuba_flat.parquet",
            )
            ped_root = tdp / "data" / "local" / "horse_pedigree_5gen" / "2019"
            ped_root.mkdir(parents=True)
            rec = {
                "ancestors": [{"generation": 1, "position": 0, "name": "X", "horse_id": "1990100001"}],
            }
            (ped_root / "2019100001.json").write_text(json.dumps(rec), encoding="utf-8")

            man = build_horse_entity_store(base_dir=tdp, years=[y], overwrite=True)
            self.assertEqual(man["horse_id_universe"]["count"], 2)

            r1 = tdp / "data" / "features" / "horse" / "result_tbl" / "2019" / "2019100001.parquet"
            self.assertTrue(r1.is_file())
            df = pd.read_parquet(r1)
            self.assertEqual(len(df), 2)

            p1 = tdp / "data" / "features" / "horse" / "ped_tbl" / "2019" / "2019100001.parquet"
            self.assertTrue(p1.is_file())

    def test_result_tbl_drops_nested_race_meta(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            y = "2024"
            td_tables = tdp / "data" / "local" / "tables" / y
            td_tables.mkdir(parents=True)
            df_rr = pd.DataFrame(
                {
                    "race_id": ["202401010101"],
                    "horse_id": ["2019100001"],
                    "finish_position": [1],
                    "payoff": [{"tansho": []}],
                    "lap_times": [[12.3]],
                    "pace": [{"t1f": 12.3}],
                }
            )
            df_rr.to_parquet(td_tables / "race_result_flat.parquet", index=False)
            pq.write_table(
                pa.table({"race_id": ["202401010101"], "horse_id": ["2019100002"]}),
                td_tables / "race_shutuba_flat.parquet",
            )
            ped_root = tdp / "data" / "local" / "horse_pedigree_5gen" / "2019"
            ped_root.mkdir(parents=True)
            (ped_root / "2019100001.json").write_text(
                json.dumps({"ancestors": [{"generation": 1, "position": 0, "horse_id": "1990100001"}]}),
                encoding="utf-8",
            )
            (ped_root / "2019100002.json").write_text(
                json.dumps({"ancestors": [{"generation": 1, "position": 0, "horse_id": "1990100002"}]}),
                encoding="utf-8",
            )

            build_horse_entity_store(base_dir=tdp, years=[y], overwrite=True)
            out = pd.read_parquet(tdp / "data" / "features" / "horse" / "result_tbl" / "2019" / "2019100001.parquet")
            for c in RESULT_TBL_DROP_NESTED_COLS:
                self.assertNotIn(c, out.columns)
            self.assertIn("race_id", out.columns)
            self.assertIn("finish_position", out.columns)

    def test_merge_gen5_sires_ped_columns(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            y = "2024"
            td_tables = tdp / "data" / "local" / "tables" / y
            td_tables.mkdir(parents=True)
            pq.write_table(
                pa.table({"race_id": ["202401010101"], "horse_id": ["2019100001"], "finish_position": [1]}),
                td_tables / "race_result_flat.parquet",
            )
            ped_root = tdp / "data" / "local" / "horse_pedigree_5gen"
            (ped_root / "2019").mkdir(parents=True)
            (ped_root / "2001").mkdir(parents=True)
            rec_s = {
                "ancestors": [
                    {"generation": 5, "position": 0, "name": "Anchor", "horse_id": "2001100001"},
                ]
            }
            rec_h = {
                "ancestors": [
                    {"generation": 1, "position": 0, "name": "Deep", "horse_id": "1990100999"},
                ]
            }
            (ped_root / "2019" / "2019100001.json").write_text(json.dumps(rec_s), encoding="utf-8")
            (ped_root / "2001" / "2001100001.json").write_text(json.dumps(rec_h), encoding="utf-8")

            man = build_horse_entity_store(base_dir=tdp, years=[y], overwrite=True, merge_gen5_sires=True)
            self.assertTrue(man["ped_tbl"]["merge_gen5_sires"])

            pdf = pd.read_parquet(tdp / "data" / "features" / "horse" / "ped_tbl" / "2019" / "2019100001.parquet")
            self.assertIn("path_fm", pdf.columns)
            self.assertIn("source", pdf.columns)
            self.assertTrue((pdf["is_male_pedigree_slot"] == 1).all())
            self.assertEqual(len(pdf), 2)
            merged = pdf[pdf["source"] == "merged_gen5_sire"]
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged.iloc[0]["path_fm"], "FFFFFF")
            self.assertEqual(int(merged.iloc[0]["generation"]), 6)
            self.assertIn("ancestor_occurrence_global", pdf.columns)
            self.assertIn("ped_root_side", pdf.columns)

    def test_flatten_pedigree_json_with_gen5_merge_empty_ancestors(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ped_dir = tdp / "ped"
            ped_dir.mkdir()
            df = _flatten_pedigree_json_with_gen5_merge({"ancestors": []}, "2019100001", ped_dir)
            self.assertTrue(df.empty)
            self.assertIn("path_fm", df.columns)


if __name__ == "__main__":
    unittest.main()
