"""etl_stg_db FK 正規化テスト。"""

from __future__ import annotations

import unittest

from src.scripts.data import etl_stg_db as etl


class TestEtlStgDbNormalize(unittest.TestCase):
    def test_normalize_fk_id(self):
        self.assertIsNone(etl._normalize_fk_id(None))
        self.assertIsNone(etl._normalize_fk_id(""))
        self.assertIsNone(etl._normalize_fk_id("   "))
        self.assertEqual(etl._normalize_fk_id("01091"), "01091")
        self.assertEqual(etl._normalize_fk_id(" 01091 "), "01091")


if __name__ == "__main__":
    unittest.main()
