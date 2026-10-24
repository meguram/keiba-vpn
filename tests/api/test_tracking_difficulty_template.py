"""追走難度 HTML の静的チェック（fieldSize 未定義・UI 集約など）。"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "analysis" / "tracking_difficulty.html"


class TestTrackingDifficultyTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TEMPLATE.read_text(encoding="utf-8")

    def test_resolve_field_size_helper_exists(self):
        self.assertIn("function resolveFieldSize(data)", self.source)

    def test_master_horse_table_helpers_exist(self):
        self.assertIn("function buildHorseMasterTableHtml(data)", self.source)
        self.assertIn("function buildHorseMasterTableBody(data)", self.source)
        self.assertIn("function buildMasterHorseRows(data)", self.source)

    def test_render_results_uses_consolidated_table_not_cards(self):
        m = re.search(r"function renderResults\(data\)\s*\{", self.source)
        self.assertIsNotNone(m)
        chunk = self.source[m.start() : m.start() + 8000]
        self.assertIn("buildHorseMasterTableHtml(data)", chunk)
        self.assertNotIn("cards-grid", chunk)
        self.assertNotIn("summary-table", chunk)
        self.assertNotIn("gateEaseMount", chunk)
        self.assertNotIn("renderGateEasePanel()", chunk)

    def test_render_results_defines_field_size_before_master_table(self):
        m = re.search(r"function renderResults\(data\)\s*\{", self.source)
        self.assertIsNotNone(m)
        chunk = self.source[m.start() : m.start() + 8000]
        fs_idx = chunk.find("const fieldSize = resolveFieldSize")
        table_idx = chunk.find("buildHorseMasterTableHtml")
        self.assertGreater(fs_idx, 0)
        self.assertGreater(table_idx, fs_idx)

    def test_init_sortable_tables_targets_horse_table(self):
        m = re.search(r"function initSortableTables\(root\)\s*\{[^}]+\}", self.source, re.DOTALL)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn(".horse-table", body)
        self.assertNotIn(".summary-table", body)

    def test_position_map_early_ladder_compact(self):
        self.assertIn("const MAX_HORSES_PER_CLUSTER = 5", self.source)
        self.assertIn("cur.length < MAX_HORSES_PER_CLUSTER", self.source)
        self.assertIn("function buildPositionMapHtml", self.source)
        self.assertIn("early-pos-ladder", self.source)
        self.assertIn("pos-pill", self.source)
        self.assertNotIn("map-stages", self.source)
        self.assertNotIn("['early', 'mid', 'late']", self.source)

    def test_result_link_opens_new_tab(self):
        self.assertIn('target="_blank"', self.source)
        self.assertIn("rel=\"noopener noreferrer\"", self.source)

    def test_fast_init_loads_picker_before_upcoming(self):
        self.assertIn("function loadInitialPickerDates", self.source)
        self.assertIn("function loadUpcomingRacesInBackground", self.source)
        self.assertIn("await loadInitialPickerDates()", self.source)
        self.assertIn("local_only=1", self.source)
        self.assertNotIn("function loadAllDates", self.source)


if __name__ == "__main__":
    unittest.main()
