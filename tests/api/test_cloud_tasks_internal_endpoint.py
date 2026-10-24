"""
POST /api/internal/cloud-tasks/process-job の統合テスト。

Cloud Tasks（GCP側キュー基盤）がPUSHするワーカー用の内部エンドポイント
（docs/operations/deployment-vps-vs-gcp.md / docs/git_management/todo/scraping-queue.md）。
実スクレイピングは行わず、`src.scraper.queue_tasks.execute_job` と
`src.scraper.run.ScraperRunner` をモックしてレスポンスのみ検証する。
"""
from __future__ import annotations

import os
import unittest
import unittest.mock

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app, raise_server_exceptions=False)


class TestCloudTasksProcessJobEndpoint(unittest.TestCase):
    def setUp(self):
        os.environ.pop("KEIBA_CLOUD_TASKS_VERIFY_OIDC", None)

    def tearDown(self):
        os.environ.pop("KEIBA_CLOUD_TASKS_VERIFY_OIDC", None)

    def test_valid_job_payload_executes_job_and_returns_completed(self):
        payload = {
            "job_id": "ct_123_abcd",
            "job_kind": "race",
            "target_id": "202501010101",
            "tasks": ["race_shutuba"],
        }
        with unittest.mock.patch("src.scraper.run.ScraperRunner") as mock_runner_cls, \
             unittest.mock.patch("src.scraper.queue_tasks.execute_job") as mock_execute_job:
            mock_runner_cls.return_value = unittest.mock.MagicMock()
            r = client.post("/api/internal/cloud-tasks/process-job", json=payload)

        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["job_id"], "ct_123_abcd")
        self.assertEqual(mock_execute_job.call_count, 1)
        called_runner, called_job = mock_execute_job.call_args[0]
        self.assertIs(called_runner, mock_runner_cls.return_value)
        self.assertEqual(called_job, payload)

    def test_execute_job_exception_returns_500_for_cloud_tasks_retry(self):
        """失敗時は非2xxを返す（Cloud Tasksのpushリトライを機能させるため意図的に500）。"""
        payload = {
            "job_id": "ct_err",
            "job_kind": "race",
            "target_id": "202501010101",
            "tasks": ["race_shutuba"],
        }
        with unittest.mock.patch("src.scraper.run.ScraperRunner") as mock_runner_cls, \
             unittest.mock.patch(
                 "src.scraper.queue_tasks.execute_job",
                 side_effect=RuntimeError("boom"),
             ):
            mock_runner_cls.return_value = unittest.mock.MagicMock()
            r = client.post("/api/internal/cloud-tasks/process-job", json=payload)

        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertIn("boom", body["error"])
        self.assertEqual(body["job_id"], "ct_err")

    def test_invalid_json_body_returns_400(self):
        r = client.post(
            "/api/internal/cloud-tasks/process-job",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)

    def test_non_object_json_body_returns_400(self):
        r = client.post(
            "/api/internal/cloud-tasks/process-job",
            content=b"[1, 2, 3]",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)

    def test_null_json_body_returns_400_no_500(self):
        r = client.post(
            "/api/internal/cloud-tasks/process-job",
            content=b"null",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)

    def test_oidc_verification_disabled_by_default_no_auth_header_needed(self):
        """KEIBA_CLOUD_TASKS_VERIFY_OIDC 未設定時は Authorization ヘッダ無しでも 401 にならない。"""
        payload = {"job_kind": "race", "target_id": "1", "tasks": ["race_shutuba"]}
        with unittest.mock.patch("src.scraper.run.ScraperRunner"), \
             unittest.mock.patch("src.scraper.queue_tasks.execute_job"):
            r = client.post("/api/internal/cloud-tasks/process-job", json=payload)
        self.assertNotEqual(r.status_code, 401)

    def test_oidc_verification_enabled_rejects_missing_bearer_token(self):
        os.environ["KEIBA_CLOUD_TASKS_VERIFY_OIDC"] = "1"
        payload = {"job_kind": "race", "target_id": "1", "tasks": ["race_shutuba"]}
        r = client.post("/api/internal/cloud-tasks/process-job", json=payload)
        self.assertEqual(r.status_code, 401)

    def test_oidc_verification_enabled_accepts_bearer_token(self):
        os.environ["KEIBA_CLOUD_TASKS_VERIFY_OIDC"] = "1"
        payload = {"job_kind": "race", "target_id": "1", "tasks": ["race_shutuba"]}
        with unittest.mock.patch("src.scraper.run.ScraperRunner"), \
             unittest.mock.patch("src.scraper.queue_tasks.execute_job"):
            r = client.post(
                "/api/internal/cloud-tasks/process-job",
                json=payload,
                headers={"Authorization": "Bearer fake-oidc-token"},
            )
        self.assertNotEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()


class TestPredictRaceJobDispatch(unittest.TestCase):
    """job_kind=predict_race（開催日のT-45予測）がワーカーで予測ワークフローに振り分けられる。"""

    def test_success_returns_200_completed(self):
        with unittest.mock.patch(
            "src.pipeline.inference.race_day_workflow.handle_predict_job",
            return_value={"status": "success", "race_id": "202605030811", "persisted": True},
        ) as handler:
            resp = client.post(
                "/api/internal/cloud-tasks/process-job",
                json={"job_kind": "predict_race", "race_id": "202605030811"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "completed")
        handler.assert_called_once()

    def test_failure_returns_500_so_cloud_tasks_retries(self):
        with unittest.mock.patch(
            "src.pipeline.inference.race_day_workflow.handle_predict_job",
            return_value={"status": "error", "race_id": "202605030811", "error": "出馬表データがありません"},
        ):
            resp = client.post(
                "/api/internal/cloud-tasks/process-job",
                json={"job_kind": "predict_race", "race_id": "202605030811"},
            )
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json()["status"], "failed")

    def test_scraping_jobs_are_not_routed_to_prediction(self):
        with unittest.mock.patch("src.pipeline.inference.race_day_workflow.handle_predict_job") as handler, \
             unittest.mock.patch("src.scraper.run.ScraperRunner"), \
             unittest.mock.patch("src.scraper.queue_tasks.execute_job"):
            resp = client.post(
                "/api/internal/cloud-tasks/process-job",
                json={"job_kind": "race", "target_id": "202501010101", "tasks": ["race_shutuba"]},
            )
        self.assertEqual(resp.status_code, 200)
        handler.assert_not_called()
