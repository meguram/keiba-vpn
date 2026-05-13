"""pipeline.jockey_trainer_stats のマージキー・スキーマ検証（unittest）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.jockey_trainer_stats import (
    JT_RACE_FEATURES_CONTEXT_COLS,
    JT_RACE_FEATURES_PRIMARY_KEYS,
    attach_jt_race_metadata,
    build_jockey_trainer_race_features,
    build_merge_spec,
    merge_jt_race_features_into_layer_a,
    validate_jt_race_features,
)


def _minimal_race_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["202001010101", "202001020101", "202001030101"],
            "horse_number": [1, 1, 1],
            "horse_id": ["2019100001", "2019100002", "2019100003"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "start_time": ["10:00", "10:00", "11:00"],
            "venue_code": ["05", "05", "05"],
            "surface": ["芝", "芝", "芝"],
            "distance": [1600, 1600, 1600],
            "grade": ["", "", ""],
            "race_class": ["", "", ""],
            "track_condition": ["良", "良", "良"],
            "finish_position": [2, 1, 3],
            "jockey_id": ["100", "100", "100"],
            "trainer_id": ["200", "200", "200"],
            "jockey_weight": [57.0, 57.0, 57.0],
            "weight": [480, 480, 480],
            "field_size": [16, 16, 16],
            "passing_order": ["4-4-2", "5-5-3", "6-6-4"],
        }
    )


class TestJockeyTrainerStats(unittest.TestCase):
    def test_jt_race_features_has_primary_and_context_columns(self):
        rr = _minimal_race_result()
        out = build_jockey_trainer_race_features(rr)
        self.assertEqual(validate_jt_race_features(out), [])
        for k in JT_RACE_FEATURES_PRIMARY_KEYS:
            self.assertIn(k, out.columns)
        for k in JT_RACE_FEATURES_CONTEXT_COLS:
            self.assertIn(k, out.columns)
        self.assertTrue(pd.isna(out["jk_prior_all_starts"].iloc[0]))
        self.assertEqual(out["jk_prior_all_starts"].iloc[2], 2.0)
        self.assertIn("jk_prior_all_avg_pass_first", out.columns)
        self.assertAlmostEqual(float(out["jk_prior_all_avg_pass_first"].iloc[2]), 4.5, places=5)

    def test_jt_race_features_unique_primary_keys(self):
        rr = _minimal_race_result()
        out = build_jockey_trainer_race_features(rr)
        dups = out.duplicated(subset=list(JT_RACE_FEATURES_PRIMARY_KEYS))
        self.assertFalse(dups.any())

    def test_merge_spec_documents_jt_columns(self):
        spec = build_merge_spec()
        self.assertEqual(
            spec["jt_race_features"]["primary_merge_keys"],
            list(JT_RACE_FEATURES_PRIMARY_KEYS),
        )
        self.assertIn("jt_result_date", spec["jt_race_features"]["context_columns"])
        self.assertIn("missing_prior_policy", spec)
        self.assertIn("passing_features", spec)

    def test_merge_into_layer_left_join(self):
        layer = pd.DataFrame(
            {
                "race_id": ["202001010101", "202001020101"],
                "horse_id": ["2019100001", "2019100002"],
                "foo": [1, 2],
            }
        )
        jt = build_jockey_trainer_race_features(_minimal_race_result())
        merged = merge_jt_race_features_into_layer_a(layer, jt_df=jt)
        self.assertEqual(len(merged), 2)
        self.assertIn("jk_prior_all_starts", merged.columns)
        self.assertIn("jt_result_date", merged.columns)
        self.assertTrue(pd.isna(merged["jk_prior_all_starts"].iloc[0]))
        self.assertEqual(merged["jk_prior_all_starts"].iloc[1], 1.0)

    def test_merge_rejects_duplicate_jt_keys(self):
        layer = pd.DataFrame({"race_id": ["202001010101"], "horse_id": ["2019100001"]})
        jt = build_jockey_trainer_race_features(_minimal_race_result())
        bad = pd.concat([jt.iloc[[0]], jt.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError) as ctx:
            merge_jt_race_features_into_layer_a(layer, jt_df=bad)
        self.assertIn("検証失敗", str(ctx.exception))

    def test_merge_rejects_missing_layer_key(self):
        layer = pd.DataFrame({"race_id": ["202001010101"]})
        jt = build_jockey_trainer_race_features(_minimal_race_result())
        with self.assertRaises(ValueError) as ctx:
            merge_jt_race_features_into_layer_a(layer, jt_df=jt)
        self.assertIn("horse_id", str(ctx.exception))

    def test_attach_metadata_one_to_one(self):
        feats = pd.DataFrame({"race_id": ["202001010101"], "horse_id": ["2019100001"], "jk_x": [1.0]})
        rr = _minimal_race_result()
        out = attach_jt_race_metadata(feats, rr)
        self.assertEqual(out["jt_row_jockey_id"].iloc[0], "100")

    def test_merge_spec_json_written_with_build(self):
        from pipeline import jockey_trainer_stats as jtmod

        rr = _minimal_race_result()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            with patch.object(jtmod, "load_race_result_for_stats", lambda **kw: rr):
                manifest = jtmod.write_jockey_trainer_stats(
                    years=None,
                    base_dir=td,
                    jra_only=False,
                    out_dir=tdp / "jt",
                )
            spec_p = Path(manifest["paths"]["merge_spec"])
            self.assertTrue(spec_p.is_file())
            spec = json.loads(spec_p.read_text(encoding="utf-8"))
            self.assertEqual(
                spec["jt_race_features"]["primary_merge_keys"],
                ["race_id", "horse_id"],
            )


if __name__ == "__main__":
    unittest.main()
