"""pipeline.features.horse_entity_layout の unittest。"""

from __future__ import annotations

import unittest

from src.pipeline.features.horse_entity_layout import horse_shard4


class TestHorseShard4(unittest.TestCase):
    def test_numeric_id_prefix(self) -> None:
        self.assertEqual(horse_shard4("2019100001"), "2019")

    def test_hex_style_id_matches_pedigree_mirror(self) -> None:
        """GCS ミラーは key[:4]（例: 000a/000a000e46.json）。数字のみ抜き出しと異なる。"""
        self.assertEqual(horse_shard4("000a000e46"), "000a")

    def test_short_zero_padded(self) -> None:
        self.assertEqual(horse_shard4("12"), "0012")


if __name__ == "__main__":
    unittest.main()
