"""build_pace_viz_html のスモークテスト。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_BUILD_SCRIPT = ROOT / "notebooks/megu_index/build_pace_viz_html.py"
_spec = importlib.util.spec_from_file_location("build_pace_viz_html", _BUILD_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
build_pace_viz_html = _mod.build_pace_viz_html
build_pace_viz_payload = _mod.build_pace_viz_payload
render_pace_viz_html = _mod.render_pace_viz_html


class TestBuildPaceVizHtml(unittest.TestCase):
    def test_payload_and_render(self):
        rows = []
        for i in range(30):
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "race_id": f"r{i:03d}",
                    "year": 2023,
                    "front_split_sec": 35.0 + (i % 5) * 0.1,
                    "par_front_split_sec": 35.2,
                    "adjusted_time_sec": 96.0 + (i % 7) * 0.15,
                }
            )
        df = pd.DataFrame(rows)
        coeff = pd.DataFrame(
            [
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "coeff_pace": 0.75,
                    "n_fit": 30,
                    "source": "cell_shrink",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = tmp_path / "data.parquet"
            cf = tmp_path / "coeff.parquet"
            out = tmp_path / "out.html"
            df.to_parquet(ds, index=False)
            coeff.to_parquet(cf, index=False)
            payload = build_pace_viz_payload(ds, cf, train_year_max=2024, horse_sample_per_cell=50)
            self.assertIn("東京|芝|1600", payload["races"])
            html = render_pace_viz_html(payload)
            self.assertIn("東京", html)
            self.assertIn("Plotly", html)
            build_pace_viz_html(ds, cf, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
