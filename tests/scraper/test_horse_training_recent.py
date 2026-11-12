"""出走馬の調教ページの取り直し（最新 1 ページを保存済みの記録に合わせる）のテスト。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.scraper import horse_training_recent as htr


def _e(date, race_date, tag=""):
    return {"date": date, "course": "栗坂", "time_raw": tag, "race_info": f"{race_date.replace('-', '/')}  東京11R 結果 ： 着"}


class MergeTest(unittest.TestCase):
    def test_replaces_same_race_date_and_keeps_older(self):
        existing = {"horse_id": "H", "total_items": 5, "pages_fetched": 3, "entries": [_e("2026-09-02", "2026-09-05", "old1"), _e("2026-08-01", "2026-08-08", "old2")]}
        fresh = {"horse_id": "H", "total_items": 6, "entries": [_e("2026-10-14", "2026-10-17", "new1"), _e("2026-09-02", "2026-09-05", "new2")]}
        m = htr.merge_recent(existing, fresh)
        tags = [e["time_raw"] for e in m["entries"]]
        self.assertEqual(tags, ["new1", "new2", "old2"])                   # 9/5 の行は新しいものに入れ替え、8/8 は残る
        self.assertEqual((m["total_items"], m["pages_fetched"]), (6, 3))

    def test_no_existing(self):
        m = htr.merge_recent(None, {"horse_id": "H", "total_items": 1, "entries": [_e("2026-10-14", "2026-10-17")]})
        self.assertEqual(len(m["entries"]), 1)
        self.assertEqual(m["pages_fetched"], 1)


    def test_keeps_extra_fields_of_existing(self):
        existing = {"horse_id": "H", "total_items": 24, "pages_fetched": 2, "truncated_before": "2015-01-01",
                    "_meta": {"x": 1}, "entries": [_e("2015-05-01", "2015-05-03")]}
        m = htr.merge_recent(existing, {"horse_id": "H", "total_items": 25, "entries": [_e("2026-10-14", "2026-10-17")]})
        self.assertEqual(m["truncated_before"], "2015-01-01")              # 打ち切りの印は追記で消さない
        self.assertNotIn("_meta", m)
        self.assertEqual((m["total_items"], len(m["entries"])), (25, 2))


class RefreshTest(unittest.TestCase):
    def _runner(self, entries, html="<table class='race_table_01'></table>"):
        r = mock.MagicMock()
        r.HORSE_TRAINING_URL = "https://x/{horse_id}"
        r.client.fetch.return_value = html
        r.training_parser.parse.return_value = {"entries": entries, "total_items": len(entries)}
        r.storage.load.return_value = None
        return r

    def test_saves_when_entries(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        out = htr.refresh_one(r, "H")
        self.assertEqual(out["status"], "saved")
        r._persist_horse_training.assert_called_once()

    def test_new_horse_with_one_page_starts_from_latest_page(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 1
        self.assertEqual(htr.refresh_one(r, "H")["status"], "saved")       # 新馬: 1 ページ目がそのまま土台
        r.scrape_horse_training.assert_not_called()
        self.assertEqual(len(r._persist_horse_training.call_args[0][1]["entries"]), 1)

    @mock.patch.object(htr, "_load_local", return_value=None)
    def test_new_horse_with_many_pages_fetches_all_pages(self, _):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 3
        r.scrape_horse_training.return_value = {"entries": [{}] * 30}
        out = htr.refresh_one(r, "H")
        r.scrape_horse_training.assert_called_once_with("H", force=True, persist=False)   # 記録が無く 3 ページ → 全ページで土台
        r._persist_horse_training.assert_called_once()                     # 保存は 1 回だけ
        self.assertEqual((out["status"], out["n_new"]), ("saved", 30))

    def test_gcs_miss_with_local_record_does_not_refetch_all_pages(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 3
        with mock.patch.object(htr, "_load_local", return_value={"entries": [_e("2020-01-01", "2020-01-05", "old")]}):   # GCS が読めず None でも、手元に記録がある
            self.assertEqual(htr.refresh_one(r, "H")["status"], "saved")
        r.scrape_horse_training.assert_not_called()
        self.assertEqual(len(r._persist_horse_training.call_args[0][1]["entries"]), 2)   # 手元の行を消さずに追記

    def test_load_local(self):
        import json as _json
        import tempfile
        from pathlib import Path

        from src.scraper.horse_training_local import horse_training_json_path

        with tempfile.TemporaryDirectory() as d:
            r = mock.MagicMock()
            r.storage._base_dir = d
            self.assertIsNone(htr._load_local(r, "2021104855"))
            f = horse_training_json_path(d, "2021104855")
            f.parent.mkdir(parents=True)
            f.write_text(_json.dumps({"entries": [{}]}), encoding="utf-8")
            self.assertEqual(htr._load_local(r, "2021104855"), {"entries": [{}]})

    def test_incomplete_record_is_rebuilt_with_all_pages(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 3
        r.storage.load.return_value = {"horse_id": "H", "incomplete": True, "entries": [_e("2026-08-01", "2026-08-08")]}
        r.scrape_horse_training.return_value = {"entries": [{}] * 30}
        self.assertEqual(htr.refresh_one(r, "H")["status"], "saved")
        r.scrape_horse_training.assert_called_once_with("H", force=True, persist=False)   # 前回の土台が途中まで → 全ページで作り直す

    def test_rebuild_keeps_rows_the_rebuild_could_not_fetch(self):
        r = self._runner([_e("2026-10-14", "2026-10-17", "n")])
        old = [_e("2026-10-14", "2026-10-17", "n"), _e("2020-01-01", "2020-01-05", "old")]
        r.storage.load.return_value = {"horse_id": "H", "incomplete": True, "rebuild_tries": 1, "entries": old}
        r.scrape_horse_training.return_value = {"horse_id": "H", "entries": [_e("2026-10-14", "2026-10-17", "n")], "incomplete": True}
        htr.refresh_one(r, "H")
        rec = r._persist_horse_training.call_args[0][1]
        self.assertEqual([e["time_raw"] for e in rec["entries"]], ["n", "old"])   # 取れなかった古い行は消さない
        self.assertEqual((rec["incomplete"], rec["rebuild_tries"]), (True, 2))

    def test_rebuild_gives_up_after_max_tries(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.storage.load.return_value = {"horse_id": "H", "incomplete": True, "rebuild_tries": htr.MAX_REBUILD_TRIES - 1, "entries": [_e("2020-01-01", "2020-01-05")]}
        r.scrape_horse_training.return_value = {"horse_id": "H", "entries": [_e("2026-10-14", "2026-10-17")], "incomplete": True}
        htr.refresh_one(r, "H")
        rec = r._persist_horse_training.call_args[0][1]
        self.assertNotIn("incomplete", rec)
        self.assertNotIn("rebuild_tries", rec)

    def test_complete_rebuild_clears_mark(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.storage.load.return_value = {"horse_id": "H", "incomplete": True, "rebuild_tries": 1, "truncated_before": "2015-01-01",
                                       "entries": [_e("2020-01-01", "2020-01-05")]}
        r.scrape_horse_training.return_value = {"horse_id": "H", "entries": [_e("2026-10-14", "2026-10-17")]}
        htr.refresh_one(r, "H")
        rec = r._persist_horse_training.call_args[0][1]
        for k in ("incomplete", "rebuild_tries", "truncated_before"):     # 全ページ取れたら印はすべて外す
            self.assertNotIn(k, rec)

    def test_union_replaces_rows_of_same_race_date(self):
        full = {"entries": [_e("2026-10-14", "2026-10-17", "new-notation")]}
        existing = {"entries": [_e("2026-10-14", "2026-10-17", "old-notation"), _e("2020-01-01", "2020-01-05", "old")]}
        self.assertEqual([e["time_raw"] for e in htr._union(full, existing)["entries"]], ["new-notation", "old"])

    @mock.patch.object(htr, "_load_local", return_value=None)
    def test_failed_rebuild_falls_back_to_latest_page_and_marks(self, _):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 3
        r.scrape_horse_training.side_effect = RuntimeError("HTTP 500")
        self.assertEqual(htr.refresh_one(r, "H")["status"], "saved")
        rec = r._persist_horse_training.call_args[0][1]
        self.assertEqual((len(rec["entries"]), rec["incomplete"], rec["rebuild_tries"]), (1, True, 1))   # 最新ページは保存し、次回に全ページ

    def test_incomplete_mark_survives_merge(self):
        m = htr.merge_recent({"horse_id": "H", "incomplete": True, "entries": []}, {"horse_id": "H", "entries": [_e("2026-10-14", "2026-10-17")]})
        self.assertTrue(m["incomplete"])

    def test_known_horse_with_many_pages_only_merges_latest_page(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        r.training_parser.parse.return_value["max_page"] = 3
        r.storage.load.return_value = {"horse_id": "H", "entries": [_e("2026-08-01", "2026-08-08")]}
        self.assertEqual(htr.refresh_one(r, "H")["status"], "saved")
        r.scrape_horse_training.assert_not_called()
        self.assertEqual(len(r._persist_horse_training.call_args[0][1]["entries"]), 2)

    def test_unreadable_page_is_not_saved(self):
        r = self._runner([])
        out = htr.refresh_one(r, "H")
        self.assertEqual(out["status"], "unreadable")                      # 表があるのに 0 件 → 保存しない
        r._persist_horse_training.assert_not_called()

    def test_empty_page_is_not_saved(self):
        r = self._runner([], html="<html></html>")
        self.assertEqual(htr.refresh_one(r, "H")["status"], "empty")
        r._persist_horse_training.assert_not_called()

    def test_one_failure_does_not_stop_others(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        calls = {"n": 0}

        def fetch(url, use_cache=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return "<table class='race_table_01'></table>"

        r.client.fetch.side_effect = fetch
        with mock.patch.object(htr.time, "sleep", lambda *_: None):
            st = htr.refresh_for_horses(r, ["A", "B"])
        self.assertEqual((st["error"], st["saved"]), (1, 1))


class EveSwitchTest(unittest.TestCase):
    def test_disabled_by_default(self):
        from src.scraper import auto_scrape_queue as aq

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEIBA_EVE_HORSE_TRAINING", None)
            self.assertEqual(aq._eve_horse_training(mock.MagicMock(), ["r1"]), {"enabled": False})


class ScrapeAllPagesIncompleteTest(unittest.TestCase):
    """scrape_horse_training: 2 ページ目以降が取れずに止めた記録に incomplete の印を付ける。"""

    def _run(self, fail_page=None, empty_page=None, max_page=3, max_pages=20):
        import types

        from src.scraper import run as R

        self_ = mock.MagicMock()
        self_.HORSE_TRAINING_URL = "https://x/{horse_id}"
        self_.client.is_logged_in = True

        def fetch(url, **_k):
            page = int(url.split("page=")[1]) if "page=" in url else 1
            if page == fail_page:
                raise RuntimeError("HTTP 500")
            return str(page)

        self_.client.fetch.side_effect = fetch
        self_.training_parser.parse.side_effect = lambda html, horse_id=None: {
            "total_items": 25, "max_page": max_page,
            "entries": [] if int(html) == empty_page else [{"date": f"2026-0{html}-01", "course": "栗坂", "time_raw": html}]}
        saved = {}
        self_._persist_horse_training.side_effect = lambda hid, res: saved.update(res=res)
        with mock.patch.object(R.time, "sleep", lambda *_: None):
            res = types.MethodType(R.ScraperRunner.scrape_horse_training, self_)("H", max_pages=max_pages, force=True)
        return res, saved["res"]

    def test_marks_incomplete_when_later_page_fails(self):
        res, saved = self._run(fail_page=3)
        self.assertTrue(saved.get("incomplete"))
        self.assertEqual(len(saved["entries"]), 2)                         # 取れた 2 ページ分は保存する

    def test_no_mark_when_last_page_is_empty_without_error(self):
        res, saved = self._run(empty_page=3)                               # 例外なしの 0 件は正常の終わり
        self.assertNotIn("incomplete", saved)

    def test_no_mark_when_stopped_by_max_pages(self):
        res, saved = self._run(max_page=5, max_pages=2)
        self.assertNotIn("incomplete", saved)
        self.assertEqual(len(saved["entries"]), 2)

    def test_no_mark_when_all_pages_fetched(self):
        res, saved = self._run(fail_page=None)
        self.assertNotIn("incomplete", saved)
        self.assertEqual(len(saved["entries"]), 3)


if __name__ == "__main__":
    unittest.main()
