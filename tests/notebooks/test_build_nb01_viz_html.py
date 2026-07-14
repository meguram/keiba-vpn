"""build_nb01_viz_html のスモークテスト。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from notebooks.megu_index.build_nb01_viz_html import (
    build_overview_viz_payload,
    build_par_split_viz_payload,
    build_track_viz_payload,
    build_weight_viz_payload,
    write_dataset_overview_html,
    write_par_split_explorer_html,
    write_track_explorer_html,
    write_weight_explorer_html,
)


class TestBuildNb01VizHtml(unittest.TestCase):
    def test_weight_payload_and_html(self) -> None:
        coef = pd.DataFrame(
            {
                "surface": ["芝", "ダート"],
                "sex_group": ["牡", "牡"],
                "distance_num": [1600, 1800],
                "sec_per_kg_final": [0.16, 0.18],
                "weight_coef_source": ["cell_sex_within_race", "surface_sex_within_race"],
                "n_fit": [120, 80],
            }
        )
        fit = pd.DataFrame(
            {
                "surface": ["芝"] * 5,
                "sex_group": ["牡"] * 5,
                "distance_num": [1600] * 5,
                "weight_dev_dm": [0.0, 0.5, -0.5, 1.0, -1.0],
                "time_dm": [0.0, 0.08, -0.08, 0.16, -0.16],
            }
        )
        payload = build_weight_viz_payload(fit, coef)
        self.assertEqual(payload["meta"]["n_cells"], 2)
        self.assertIn("芝|牡|1600", payload["points"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "weight.html"
            write_weight_explorer_html(fit, coef, out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("斤量補正エクスプローラ", text)
            self.assertIn("plotly", text.lower())

    def test_track_and_par_split_html(self) -> None:
        day = pd.DataFrame(
            {
                "date_str": ["2024-01-01", "2024-01-02"],
                "venue": ["東京", "東京"],
                "surface": ["芝", "芝"],
                "track_dev_sec": [0.5, -0.3],
                "tsi_raw": [-0.5, 0.3],
                "n_races_track": [3, 5],
            }
        )
        track_payload = build_track_viz_payload(day)
        self.assertIn("東京|芝", track_payload["series"])

        par = pd.DataFrame(
            {
                "distance": [1600],
                "surface": ["芝"],
                "par_intercept": [35.0],
                "par_slope": [0.5],
                "t2nd_ref": [95.0],
                "n_fit": [200],
                "model": ["cell"],
            }
        )
        df2 = pd.DataFrame(
            {
                "distance": [1600, 1600],
                "surface": ["芝", "芝"],
                "race_t2nd_sec": [94.5, 95.5],
                "front_split_sec": [34.8, 35.2],
            }
        )
        par_payload = build_par_split_viz_payload(df2, par)
        self.assertEqual(par_payload["meta"]["n_cells"], 1)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_track_explorer_html(day, td_path / "track.html")
            write_par_split_explorer_html(df2, par, td_path / "par.html")
            self.assertGreater((td_path / "track.html").stat().st_size, 500)
            self.assertGreater((td_path / "par.html").stat().st_size, 500)

    def test_overview_payload(self) -> None:
        df = pd.DataFrame(
            {
                "surface": ["芝", "ダート", "芝"],
                "finish_time_sec": [95.0, 96.0, 94.5],
                "track_dev_sec": [0.1, 0.2, -0.1],
            }
        )
        payload = build_overview_viz_payload(df)
        self.assertIn("芝", payload["histograms"])
        self.assertIn("track_dev", payload["histograms"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "overview.html"
            write_dataset_overview_html(df, out)
            self.assertIn("megu_dataset", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
