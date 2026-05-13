"""scraper.horse_training_local と調教ローカルパスの unittest。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import requests

from pipeline.horse_entity_layout import horse_shard4
from scraper.horse_training_local import horse_training_local_root, write_horse_training_json
from scraper.run import _horse_training_http_means_empty


class TestHorseTrainingLocal(unittest.TestCase):
    def test_horse_training_http_means_empty(self):
        url = "https://db.netkeiba.com/horse/training.html?id=2009102606"
        r400 = requests.Response()
        r400.status_code = 400
        r400.url = url
        e400 = requests.HTTPError(response=r400)
        self.assertFalse(_horse_training_http_means_empty(e400, url))
        r404 = requests.Response()
        r404.status_code = 404
        r404.url = url
        e404 = requests.HTTPError(response=r404)
        self.assertTrue(_horse_training_http_means_empty(e404, url))
        self.assertFalse(
            _horse_training_http_means_empty(e400, "https://db.netkeiba.com/horse/"),
        )
        self.assertTrue(
            _horse_training_http_means_empty(
                RuntimeError(
                    "404 Client Error: Not Found for url: "
                    "https://db.netkeiba.com/horse/training.html?id=2009102606",
                ),
                url,
            ),
        )

    def test_access_pause_treats_http_400_as_block(self):
        from scraper.scrape_access_pause import is_access_or_transport_error

        r = requests.Response()
        r.status_code = 400
        e = requests.HTTPError(response=r)
        self.assertTrue(is_access_or_transport_error(e))

    def test_write_horse_training_json_shard_and_content(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            hid = "2023106216"
            payload = {
                "horse_id": hid,
                "total_items": 4,
                "pages_fetched": 1,
                "entries": [{"date": "2026-03-11", "course": "栗坂", "time_raw": "- 53.1"}],
                "_meta": {"scraped_at": 1.0},
            }
            outp = write_horse_training_json(base, hid, payload)
            self.assertEqual(outp, horse_training_local_root(base) / horse_shard4(hid) / f"{hid}.json")
            self.assertTrue(outp.is_file())
            loaded = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["horse_id"], hid)
            self.assertEqual(len(loaded["entries"]), 1)

    def test_local_mirror_exists_for_sample_horses(self):
        """
        リポジトリルートに対し、数頭分のスクレイプ後に JSON が置かれていること。
        （CI では未実行のため、ファイルが無ければスキップ）
        """
        repo = Path(__file__).resolve().parent
        root = horse_training_local_root(repo)
        samples = [
            ("2021105143", "2021"),
            ("2023106216", "2023"),
            ("2022103639", "2022"),
            ("2018105689", "2018"),
        ]
        missing = [hid for hid, shard in samples if not (root / shard / f"{hid}.json").is_file()]
        if missing:
            self.skipTest(f"ローカルミラー未作成（先に scraper.run horse-training を実行）: {missing}")
        for hid, shard in samples:
            p = root / shard / f"{hid}.json"
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data.get("horse_id"), hid)
            self.assertIn("entries", data)

    def test_scraper_runner_storage_root_is_repo_absolute(self):
        """キューが ``ScraperRunner()`` を使うときと同じく、cwd に依存しないリポジトリルートを向く。"""
        from scraper.run import REPO_ROOT, ScraperRunner

        runner = ScraperRunner()
        self.assertEqual(runner.storage._base_dir.resolve(), REPO_ROOT.resolve())

    def test_queue_execute_job_dispatches_horse_training(self):
        """``execute_job`` → ``scrape_horse_training``（キューと同じディスパッチ）。"""
        from scraper.queue_tasks import execute_job

        calls: list[tuple[str, bool]] = []

        class FakeRunner:
            def scrape_horse_training(self, horse_id: str, smart_skip: bool = True, **_kw):
                calls.append((horse_id, smart_skip))

        execute_job(
            FakeRunner(),
            {
                "job_kind": "horse",
                "target_id": "2023106216",
                "tasks": ["horse_training"],
                "smart_skip": True,
            },
        )
        self.assertEqual(calls, [("2023106216", True)])

    def test_execute_job_real_runner_writes_local_json(self):
        """
        キューと同じ ``ScraperRunner`` + ``execute_job`` で、取得成功時にローカル JSON が付くこと。
        ネット・.env の netkeiba 認証が必要。無い場合はスキップ。
        """
        import os

        from scraper.client import _load_env

        _load_env(None)
        if not os.environ.get("netkeiba_id"):
            self.skipTest("netkeiba_id なし（.env を確認）")

        from scraper.queue_tasks import execute_job
        from scraper.run import REPO_ROOT, ScraperRunner

        hid = "2023106216"
        root = horse_training_local_root(REPO_ROOT)
        out_path = root / horse_shard4(hid) / f"{hid}.json"

        runner = ScraperRunner()
        execute_job(
            runner,
            {
                "job_kind": "horse",
                "target_id": hid,
                "tasks": ["horse_training"],
                "smart_skip": True,
            },
        )

        self.assertTrue(out_path.is_file(), msg=f"期待パス: {out_path}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("horse_id"), hid)
        self.assertIsInstance(data.get("entries"), list)


if __name__ == "__main__":
    unittest.main()
