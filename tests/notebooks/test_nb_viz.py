"""nb_viz の iframe 生成テスト。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_NB_VIZ = ROOT / "notebooks/megu_index/nb_viz.py"
_spec = importlib.util.spec_from_file_location("nb_viz", _NB_VIZ)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
build_iframe_markup = _mod.build_iframe_markup
notebook_rel_path = _mod.notebook_rel_path
preview_url = _mod.preview_url


class TestNbViz(unittest.TestCase):
    def test_build_iframe_uses_http_preview(self) -> None:
        out = ROOT / "notebooks/megu_index/output/nb02/par_time_explorer.html"
        if not out.exists():
            self.skipTest("par_time_explorer.html missing")
        markup = build_iframe_markup(out, height=400)
        self.assertIn('src="http://127.0.0.1:', markup)
        self.assertNotIn("srcdoc=", markup)
        self.assertNotIn("isolated", markup)

    def test_notebook_rel_path(self) -> None:
        out = ROOT / "notebooks/megu_index/output/nb02/par_time_explorer.html"
        if out.exists():
            rel = notebook_rel_path(out)
            self.assertEqual(rel, "output/nb02/par_time_explorer.html")

    def test_preview_server_serves_plotly(self) -> None:
        plotly = ROOT / "notebooks/megu_index/static/plotly-2.35.2.min.js"
        if not plotly.exists():
            self.skipTest("plotly static file missing")
        import urllib.request

        url = preview_url(plotly)
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read(200)
        self.assertIn(b"Plotly", body)


if __name__ == "__main__":
    unittest.main()
