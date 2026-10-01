"""
src.scraper.cloud_tasks_queue のユニットテスト。

GCP(Cloud Tasks)側キュー基盤への移行（docs/operations/deployment-vps-vs-gcp.md /
docs/git_management/todo/scraping-queue.md）の検証:
  - enqueue_via_cloud_tasks が CloudTasksClient.create_task を正しいパラメータで呼ぶこと
    （google.cloud.tasks_v2.CloudTasksClient はモック。実GCP接続はしない）
  - is_cloud_tasks_backend_enabled の KEIBA_QUEUE_BACKEND 分岐
  - ScrapeJobQueue.add_job が KEIBA_QUEUE_BACKEND 未設定時は従来通りローカルJSONキューに
    投入されること（regressionが無いこと）
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")


# ─── is_cloud_tasks_backend_enabled ──────────────────────────────


def test_backend_disabled_by_default(monkeypatch):
    from src.scraper.cloud_tasks_queue import is_cloud_tasks_backend_enabled

    monkeypatch.delenv("KEIBA_QUEUE_BACKEND", raising=False)
    assert is_cloud_tasks_backend_enabled() is False


def test_backend_enabled_when_cloud_tasks(monkeypatch):
    from src.scraper.cloud_tasks_queue import is_cloud_tasks_backend_enabled

    monkeypatch.setenv("KEIBA_QUEUE_BACKEND", "cloud_tasks")
    assert is_cloud_tasks_backend_enabled() is True


def test_backend_disabled_for_other_values(monkeypatch):
    from src.scraper.cloud_tasks_queue import is_cloud_tasks_backend_enabled

    monkeypatch.setenv("KEIBA_QUEUE_BACKEND", "local")
    assert is_cloud_tasks_backend_enabled() is False


# ─── enqueue_via_cloud_tasks ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_gcp_credentials(monkeypatch):
    # .env の実サービスアカウント情報を使わない（ADC相当のNoneで CloudTasksClient を呼ぶ）。
    monkeypatch.setattr(
        "src.config.gcp_credentials.build_gcp_credentials",
        lambda *a, **k: None,
    )
    monkeypatch.delenv("GCS_PROJECT_ID", raising=False)


def test_enqueue_via_cloud_tasks_calls_create_task_with_expected_params(monkeypatch):
    from google.cloud import tasks_v2

    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.queue_path.return_value = (
            "projects/test-project/locations/asia-northeast1/queues/keiba-scrape-queue"
        )
        mock_client.create_task.return_value.name = (
            "projects/test-project/locations/asia-northeast1/queues/keiba-scrape-queue/tasks/t1"
        )

        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
        monkeypatch.delenv("CLOUD_TASKS_QUEUE", raising=False)
        monkeypatch.delenv("CLOUD_TASKS_LOCATION", raising=False)
        monkeypatch.setenv(
            "CLOUD_RUN_JOBS_WORKER_URL",
            "https://worker.example.com/api/internal/cloud-tasks/process-job",
        )
        monkeypatch.delenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", raising=False)

        payload = {"job_kind": "race", "target_id": "202501010101", "tasks": ["race_shutuba"]}
        task_name = ctq.enqueue_via_cloud_tasks(payload)

        assert task_name == (
            "projects/test-project/locations/asia-northeast1/queues/keiba-scrape-queue/tasks/t1"
        )

        # queue_path は project/location/queue の既定値で呼ばれる
        mock_client.queue_path.assert_called_once_with(
            "test-project", "asia-northeast1", "keiba-scrape-queue"
        )

        # create_task の request を検証
        assert mock_client.create_task.call_count == 1
        _, kwargs = mock_client.create_task.call_args
        request = kwargs["request"]
        assert request["parent"] == (
            "projects/test-project/locations/asia-northeast1/queues/keiba-scrape-queue"
        )
        http_request = request["task"]["http_request"]
        assert http_request["url"] == (
            "https://worker.example.com/api/internal/cloud-tasks/process-job"
        )
        assert http_request["http_method"] == tasks_v2.HttpMethod.POST
        body = json.loads(http_request["body"].decode("utf-8"))
        assert body == payload
        assert "oidc_token" not in http_request


def test_enqueue_via_cloud_tasks_uses_override_args_and_oidc(monkeypatch):
    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.queue_path.return_value = (
            "projects/test-project/locations/us-central1/queues/custom-queue"
        )
        mock_client.create_task.return_value.name = (
            "projects/test-project/locations/us-central1/queues/custom-queue/tasks/t2"
        )

        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
        monkeypatch.setenv(
            "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", "sa@test-project.iam.gserviceaccount.com"
        )

        task_name = ctq.enqueue_via_cloud_tasks(
            {"job_kind": "horse", "target_id": "2020100001", "tasks": ["horse_profile"]},
            queue="custom-queue",
            location="us-central1",
            worker_url="https://worker2.example.com/push",
        )

        assert task_name == (
            "projects/test-project/locations/us-central1/queues/custom-queue/tasks/t2"
        )
        mock_client.queue_path.assert_called_once_with(
            "test-project", "us-central1", "custom-queue"
        )
        _, kwargs = mock_client.create_task.call_args
        http_request = kwargs["request"]["task"]["http_request"]
        assert http_request["url"] == "https://worker2.example.com/push"
        assert http_request["oidc_token"] == {
            "service_account_email": "sa@test-project.iam.gserviceaccount.com"
        }


def test_enqueue_via_cloud_tasks_requires_project_id(monkeypatch):
    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient"):
        monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
        monkeypatch.setenv("CLOUD_RUN_JOBS_WORKER_URL", "https://worker.example.com/push")
        with pytest.raises(ValueError):
            ctq.enqueue_via_cloud_tasks(
                {"job_kind": "race", "target_id": "1", "tasks": ["race_shutuba"]}
            )


def test_enqueue_via_cloud_tasks_requires_worker_url(monkeypatch):
    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient"):
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
        monkeypatch.delenv("CLOUD_RUN_JOBS_WORKER_URL", raising=False)
        with pytest.raises(ValueError):
            ctq.enqueue_via_cloud_tasks(
                {"job_kind": "race", "target_id": "1", "tasks": ["race_shutuba"]}
            )


# ─── ScrapeJobQueue.add_job: backend 分岐 ─────────────────────────


@pytest.fixture
def queue_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.scraper.job_queue as jq

    queue_file = tmp_path / "scrape_queue.json"
    monkeypatch.setattr(jq, "QUEUE_FILE", queue_file)
    return jq, queue_file


def test_add_job_default_backend_uses_local_queue_file(queue_paths, monkeypatch):
    """KEIBA_QUEUE_BACKEND 未設定時: 従来通りローカルJSONキューに投入される（regressionなし）。"""
    jq, queue_file = queue_paths
    monkeypatch.delenv("KEIBA_QUEUE_BACKEND", raising=False)

    q = jq.ScrapeJobQueue()
    result = q.add_job(
        {"job_kind": "race", "target_id": "202501010101", "tasks": ["race_shutuba"]}
    )

    assert result["action"] == "created"
    assert queue_file.exists()
    saved = json.loads(queue_file.read_text(encoding="utf-8"))
    jobs = saved["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == result["job_id"]
    assert jobs[0]["target_id"] == "202501010101"
    assert jobs[0]["job_kind"] == "race"


def test_add_job_cloud_tasks_backend_skips_local_queue_file(queue_paths, monkeypatch):
    """KEIBA_QUEUE_BACKEND=cloud_tasks のとき: ローカルJSONキューファイルには書き込まない。"""
    jq, queue_file = queue_paths
    monkeypatch.setenv("KEIBA_QUEUE_BACKEND", "cloud_tasks")

    with mock.patch(
        "src.scraper.cloud_tasks_queue.enqueue_via_cloud_tasks",
        return_value="projects/p/locations/l/queues/q/tasks/fake",
    ) as mock_enqueue:
        q = jq.ScrapeJobQueue()
        result = q.add_job(
            {"job_kind": "race", "target_id": "202501010101", "tasks": ["race_shutuba"]}
        )

    assert result["backend"] == "cloud_tasks"
    assert result["cloud_tasks_task_name"] == "projects/p/locations/l/queues/q/tasks/fake"
    assert mock_enqueue.call_count == 1
    sent_payload = mock_enqueue.call_args[0][0]
    assert sent_payload["job_kind"] == "race"
    assert sent_payload["target_id"] == "202501010101"
    assert sent_payload["tasks"] == ["race_shutuba"]
    assert sent_payload["job_id"] == result["job_id"]

    # ローカルキューファイルは作られない（Cloud Tasks経路のみ使われたこと）
    assert not queue_file.exists()


def test_enqueue_sets_schedule_time_and_fixed_task_name(monkeypatch):
    """予約配信(schedule_time)とタスク名固定(task_id)が create_task に渡される。"""
    from datetime import datetime, timezone

    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = "projects/p/locations/l/queues/q/tasks/predict-202605030811"
        mock_client.create_task.return_value.name = "projects/p/locations/l/queues/q/tasks/predict-202605030811"

        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
        monkeypatch.setenv("CLOUD_RUN_JOBS_WORKER_URL", "https://worker.example.com/x")
        monkeypatch.delenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", raising=False)

        when = datetime(2026, 10, 4, 0, 10, tzinfo=timezone.utc)  # = 09:10 JST
        ctq.enqueue_via_cloud_tasks(
            {"job_kind": "predict_race", "race_id": "202605030811"},
            schedule_time=when,
            task_id="predict-202605030811",
        )

        mock_client.task_path.assert_called_once_with("test-project", "asia-northeast1", "keiba-scrape-queue", "predict-202605030811")
        _, kwargs = mock_client.create_task.call_args
        task = kwargs["request"]["task"]
        assert task["name"].endswith("/tasks/predict-202605030811")
        assert task["schedule_time"].ToDatetime() == datetime(2026, 10, 4, 0, 10)


def test_enqueue_without_schedule_time_has_no_schedule_or_name(monkeypatch):
    from src.scraper import cloud_tasks_queue as ctq

    with mock.patch("google.cloud.tasks_v2.CloudTasksClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.create_task.return_value.name = "n"
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
        monkeypatch.setenv("CLOUD_RUN_JOBS_WORKER_URL", "https://worker.example.com/x")
        monkeypatch.delenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", raising=False)

        ctq.enqueue_via_cloud_tasks({"job_kind": "race"})
        task = mock_client.create_task.call_args.kwargs["request"]["task"]
        assert "schedule_time" not in task and "name" not in task
