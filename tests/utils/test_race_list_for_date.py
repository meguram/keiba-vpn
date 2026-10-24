"""race_list_for_date ユニットテスト。

opening_date_kind / opening_date_display は data/page_reference/race_lists/ 配下の
実データを読む関数だが、このディレクトリは AGENTS.md 記載の通り運用バンドル
（別PCへは BUNDLE.md の手順でコピー）であり .gitignore 対象のため、
CI やクリーンチェックアウトには存在しない。テストは一時ディレクトリに
最小限の race_lists フィクスチャを書き込み、RACE_LIST_DIR を差し替えて検証する。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import race_list_for_date
from src.utils.race_list_for_date import decode_race_id, opening_date_display, opening_date_kind


class TestRaceListForDate(unittest.TestCase):
    def test_decode_race_id_not_calendar_date(self):
        d = decode_race_id("202007010201")
        self.assertEqual(d["year"], "2020")
        self.assertEqual(d["venue"], "中京")
        self.assertEqual(d["kaisai_round"], 1)
        self.assertEqual(d["kaisai_day"], 2)
        self.assertNotEqual(d["kaisai_day"], 7)  # 07 は場コード


class TestOpeningDateKindWithFixtures(unittest.TestCase):
    """opening_date_kind / opening_date_display 用のフィクスチャ付きテスト。

    実データ（data/page_reference/race_lists/）に依存せず、一時ディレクトリに
    最小限の race_lists JSON を書き込んで検証する。
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = mock.patch.object(
            race_list_for_date, "RACE_LIST_DIR", Path(self._tmpdir.name)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _write_race_list(self, date: str, races: list[dict], meta: dict | None = None) -> None:
        path = Path(self._tmpdir.name) / f"{date}.json"
        payload: dict = {"races": races}
        if meta is not None:
            payload["_meta"] = meta
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_opening_date_kind_missing(self):
        # フィクスチャ未作成 = ファイル無し
        self.assertEqual(opening_date_kind("20220110"), "missing")

    def test_opening_date_kind_no_meeting(self):
        # JRA レースが1件も無い日（非開催プレースホルダ）
        self._write_race_list("20220110", races=[])
        self.assertEqual(opening_date_kind("20220110"), "no_meeting")

    def test_opening_date_kind_meeting(self):
        # 過去日で JRA レース（場コード "05" = 東京）が1件でもあれば meeting
        self._write_race_list(
            "20200301",
            races=[{"race_id": "202005010511", "race_name": "3歳未勝利"}],
        )
        self.assertEqual(opening_date_kind("20200301"), "meeting")

    def test_opening_date_display_no_meeting(self):
        self._write_race_list("20220110", races=[])
        d = opening_date_display("20220110")
        self.assertEqual(d["kind"], "no_meeting")
        self.assertEqual(d["label"], "非開催（対象外）")
        self.assertFalse(d["quality_applicable"])
        self.assertFalse(d["monitor_data_applicable"])

    def test_opening_date_display_meeting(self):
        self._write_race_list(
            "20200301",
            races=[{"race_id": "202005010511", "race_name": "3歳未勝利"}],
        )
        d = opening_date_display("20200301")
        self.assertEqual(d["kind"], "meeting")
        self.assertEqual(d["label"], "開催日")
        self.assertTrue(d["quality_applicable"])
        self.assertTrue(d["monitor_data_applicable"])


if __name__ == "__main__":
    unittest.main()
