"""血統キー（父・母父の horse_id）を血統 JSON から引く処理のテスト。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.features import pedigree_keys as pk

ANC = [
    {"generation": 1, "position": 0, "name": "父", "horse_id": "S1"},
    {"generation": 1, "position": 1, "name": "母", "horse_id": "D1"},
    {"generation": 2, "position": 0, "name": "父父", "horse_id": "SS"},
    {"generation": 2, "position": 2, "name": "母父", "horse_id": "DS"},
]


class ParentKeysTest(unittest.TestCase):
    def test_positions(self):
        self.assertEqual(pk.parents_from_ancestors(ANC), ("S1", "DS"))
        self.assertEqual(pk.parents_from_ancestors([]), ("", ""))

    def test_table_prefers_first_dir_and_unknown_is_blank(self):
        with tempfile.TemporaryDirectory() as t:
            d5, d10 = Path(t) / "5", Path(t) / "10"
            (d5 / "h001").mkdir(parents=True)
            (d10 / "h002").mkdir(parents=True)
            (d5 / "h001" / "h001a.json").write_text(json.dumps({"ancestors": ANC}), encoding="utf-8")
            (d10 / "h002" / "h002b.json").write_text(json.dumps({"ancestors": [ANC[0]]}), encoding="utf-8")
            tab = pk.build_parent_table(["h001a", "h002b", "h003c"], dirs=[d5, d10]).set_index("horse_id")
            self.assertEqual(tab.loc["h001a"].tolist(), ["S1", "DS"])
            self.assertEqual(tab.loc["h002b"].tolist(), ["S1", ""])
            self.assertEqual(tab.loc["h003c"].tolist(), ["", ""])

    def test_attach(self):
        tab = pd.DataFrame({"horse_id": ["a"], "sire_id": ["S1"], "dam_sire_id": ["DS"]})
        out = pk.attach_parent_ids(pd.DataFrame({"horse_id": ["a", "z"], "sire": ["", ""], "dam_sire": ["", ""]}), tab)
        self.assertEqual(out["sire"].tolist(), ["S1", ""])
        self.assertEqual(out["dam_sire"].tolist(), ["DS", ""])


if __name__ == "__main__":
    unittest.main()
