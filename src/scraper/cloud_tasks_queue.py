"""
Cloud Tasks 経由のスクレイピングキュー投入（GCP側専用）。

VPS/ConoHaとGCPの役割分担（`docs/operations/deployment-vps-vs-gcp.md`）に基づき、
netkeibaスクレイピングのキュー投入経路を切り替えるための薄いアダプタ。

- 既定（``KEIBA_QUEUE_BACKEND`` 未設定）: ``src.scraper.job_queue.ScrapeJobQueue`` の
  ローカルJSONファイル＋ファイルロックへそのまま投入する（VPS側の既存動作を変えない）。
- ``KEIBA_QUEUE_BACKEND=cloud_tasks``: ``enqueue_via_cloud_tasks`` を使い、ジョブ仕様
  （``job_queue.add_job`` に渡す形式の dict）をCloud Tasksキューへ HTTP POST タスクとして
  積む。実行は Cloud Run Jobs/サービス側の push ワーカー
  （``POST /api/internal/cloud-tasks/process-job``、``src/api/app.py``）が
  ``src.scraper.queue_tasks.execute_job`` を呼んで1件処理する。

環境変数（``.env.example`` の「GCP側（スクレイピング・ML・スケジュール実行）専用設定」参照）:
    KEIBA_QUEUE_BACKEND          ``cloud_tasks`` のときのみ Cloud Tasks 経路を使う（既定local）
    GCP_PROJECT_ID                GCPプロジェクトID（必須）
    CLOUD_TASKS_QUEUE             Cloud Tasksキュー名（既定 ``keiba-scrape-queue``）
    CLOUD_TASKS_LOCATION          リージョン（既定 ``asia-northeast1``）
    CLOUD_RUN_JOBS_WORKER_URL     Cloud TasksがPUSHするワーカーURL（必須）
    CLOUD_TASKS_OIDC_SERVICE_ACCOUNT  設定時はタスクにOIDCトークンを付与（任意）

GCP接続自体は疎通している前提で実装する。``src.config.gcp_credentials.
ensure_google_application_credentials()`` を呼べば ``config/gcp-service-account.json``
からADC認証される（実ファイルは開発環境には無いため、本モジュールの単体テストは
``google.cloud.tasks_v2.CloudTasksClient`` をモックして検証する）。
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_TASKS_QUEUE = "keiba-scrape-queue"
DEFAULT_CLOUD_TASKS_LOCATION = "asia-northeast1"


def is_cloud_tasks_backend_enabled() -> bool:
    """``KEIBA_QUEUE_BACKEND=cloud_tasks`` のとき True（既定はローカルJSONキュー）。"""
    return os.environ.get("KEIBA_QUEUE_BACKEND", "").strip().lower() == "cloud_tasks"


def enqueue_via_cloud_tasks(
    job_payload: dict,
    *,
    queue: str | None = None,
    location: str | None = None,
    worker_url: str | None = None,
) -> str:
    """
    ``job_payload``（``ScrapeJobQueue.add_job`` と同形式のジョブ仕様 dict）を
    Cloud Tasksキューへ HTTP POST タスクとして作成する。

    ``queue`` / ``location`` / ``worker_url`` を渡すと、同名の環境変数
    （``CLOUD_TASKS_QUEUE`` / ``CLOUD_TASKS_LOCATION`` / ``CLOUD_RUN_JOBS_WORKER_URL``）
    より優先される。

    Returns:
        作成されたタスクのフルリソース名（``projects/.../locations/.../queues/.../tasks/...``）。
    """
    from google.cloud import tasks_v2

    from src.config.gcp_credentials import ensure_google_application_credentials

    ensure_google_application_credentials()

    project_id = os.environ.get("GCP_PROJECT_ID", "").strip()
    if not project_id:
        raise ValueError("GCP_PROJECT_ID が未設定です（Cloud Tasksキューには必須）")

    queue_name = (
        queue or os.environ.get("CLOUD_TASKS_QUEUE") or DEFAULT_CLOUD_TASKS_QUEUE
    ).strip()
    location_name = (
        location
        or os.environ.get("CLOUD_TASKS_LOCATION")
        or DEFAULT_CLOUD_TASKS_LOCATION
    ).strip()
    target_url = (
        worker_url or os.environ.get("CLOUD_RUN_JOBS_WORKER_URL") or ""
    ).strip()
    if not target_url:
        raise ValueError(
            "CLOUD_RUN_JOBS_WORKER_URL が未設定です（Cloud TasksのPush先ワーカーURLが必須）"
        )

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project_id, location_name, queue_name)

    body_bytes = json.dumps(job_payload, ensure_ascii=False, default=str).encode("utf-8")
    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": target_url,
        "headers": {"Content-Type": "application/json"},
        "body": body_bytes,
    }

    # Cloud Run等の認証付きPushワーカー向けOIDCトークン。
    # ワーカー側の検証は KEIBA_CLOUD_TASKS_VERIFY_OIDC=1 のときのみ有効化するスタブ
    # （POST /api/internal/cloud-tasks/process-job 内、src/api/app.py）。
    service_account_email = os.environ.get("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT", "").strip()
    if service_account_email:
        http_request["oidc_token"] = {"service_account_email": service_account_email}

    task = {"http_request": http_request}

    response = client.create_task(request={"parent": parent, "task": task})
    logger.info(
        "Cloud Tasksへ投入: %s (queue=%s, location=%s)",
        response.name,
        queue_name,
        location_name,
    )
    return response.name
