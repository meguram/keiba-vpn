"""update_serving_state profile: 誕生日の補完が重複列で落ちないこと（10/20〜 毎朝 MergeError で失敗していた）。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.scripts.data.update_serving_state import attach_birthday


class TestAttachBirthday(unittest.TestCase):
    def test_existing_birthday_is_kept_and_gaps_are_filled(self) -> None:
        new = pd.DataFrame({"horse_id": ["a", "b"], "birthday": ["2023-01-01", None], "owner": ["x", "y"]})
        attrs = pd.DataFrame({"birthday": ["2024-09-09", "2023-02-02"]}, index=pd.Index(["a", "b"]))
        out = attach_birthday(new, attrs)
        self.assertEqual(list(out.columns), ["horse_id", "birthday", "owner"])
        self.assertEqual(out["birthday"].tolist(), ["2023-01-01", "2023-02-02"])

    def test_missing_birthday_column_is_added(self) -> None:
        new = pd.DataFrame({"horse_id": ["a"], "owner": ["x"]})
        attrs = pd.DataFrame({"birthday": ["2023-02-02"]}, index=pd.Index(["a"]))
        self.assertEqual(attach_birthday(new, attrs)["birthday"].tolist(), ["2023-02-02"])

    def test_attrs_without_birthday_returns_unchanged(self) -> None:
        new = pd.DataFrame({"horse_id": ["a"]})
        self.assertTrue(attach_birthday(new, pd.DataFrame({"x": [1]}, index=pd.Index(["a"]))).equals(new))


if __name__ == "__main__":
    unittest.main()
