# feature/scraping-queue

**対象領域**: スクレイピング・キュー管理
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 43（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| POST | `/api/auto-scrape/run` | `api_auto_scrape_run` (L11991) |
| GET | `/api/auto-scrape/run-status` | `api_auto_scrape_run_status` (L12028) |
| GET | `/api/auto-scrape/status` | `api_auto_scrape_status` (L11842) |
| POST | `/api/check-scraped-status` | `check_scraped_status` (L6264) |
| POST | `/api/cushion/scrape` | `api_cushion_scrape` (L11598) |
| GET | `/api/cushion/scrape/status` | `api_cushion_scrape_status` (L11630) |
| POST | `/api/fetch-future-calendar` | `fetch_future_calendar` (L6299) |
| GET | `/api/fetch-future-calendar/status` | `get_calendar_fetch_status` (L6342) |
| GET | `/api/scrape-dates` | `get_scraped_dates` (L2486) |
| GET | `/api/scrape-jobs` | `get_scrape_jobs` (L4447) |
| POST | `/api/scrape-missing` | `scrape_missing` (L2296) |
| POST | `/api/scrape-queue/add` | `add_to_scrape_queue` (L4702) |
| POST | `/api/scrape-queue/add-batch` | `add_batch_to_scrape_queue` (L5559) |
| POST | `/api/scrape-queue/add-job` | `scrape_queue_add_generic_job` (L5354) |
| POST | `/api/scrape-queue/clear` | `clear_scrape_queue` (L5410) |
| POST | `/api/scrape-queue/dismiss-auto-cleared-notice` | `api_scrape_queue_dismiss_auto_cleared_notice` (L4887) |
| POST | `/api/scrape-queue/enqueue-incomplete-dates` | `scrape_queue_enqueue_incomplete_dates` (L5385) |
| POST | `/api/scrape-queue/enqueue-scrape-period` | `api_scrape_queue_enqueue_scrape_period` (L6071) |
| POST | `/api/scrape-queue/failed/remove` | `scrape_queue_failed_remove` (L5523) |
| POST | `/api/scrape-queue/failed/requeue` | `scrape_queue_failed_requeue` (L5488) |
| POST | `/api/scrape-queue/hourly-maintenance-run` | `api_scrape_queue_hourly_maintenance_run` (L5429) |
| POST | `/api/scrape-queue/kick` | `api_scrape_queue_kick` (L4933) |
| GET | `/api/scrape-queue/load-settings` | `api_scrape_queue_load_settings_get` (L5283) |
| POST | `/api/scrape-queue/load-settings` | `api_scrape_queue_load_settings_post` (L5297) |
| GET | `/api/scrape-queue/local-mirror-config` | `api_scrape_queue_local_mirror_config_get` (L5219) |
| POST | `/api/scrape-queue/local-mirror-config` | `api_scrape_queue_local_mirror_config_post` (L5247) |
| POST | `/api/scrape-queue/migrate-precheck` | `api_scrape_queue_migrate_precheck` (L5190) |
| GET | `/api/scrape-queue/progress` | `api_scrape_queue_progress` (L4794) |
| POST | `/api/scrape-queue/recover` | `api_scrape_queue_recover` (L4906) |
| POST | `/api/scrape-queue/resume` | `api_scrape_queue_resume` (L4849) |
| GET | `/api/scrape-queue/status` | `get_scrape_queue_status` (L4746) |
| POST | `/api/scrape-queue/stop-and-clear` | `api_scrape_queue_stop_and_clear` (L5467) |
| GET | `/api/scrape-queue/tasks` | `scrape_queue_task_catalog` (L5339) |
| POST | `/api/scrape-queue/verify-horse-coverage` | `api_scrape_queue_verify_horse_coverage` (L5144) |
| GET | `/api/scrape-queue/worker-logs` | `api_scrape_queue_worker_logs` (L4813) |
| POST | `/api/scrape-queue/worker-logs/clear` | `api_scrape_queue_worker_logs_clear` (L4834) |
| GET | `/api/scrape-status` | `get_scrape_status` (L1792) |
| GET | `/api/scrape-summary-all` | `get_scrape_summary_all` (L2822) |
| POST | `/api/scrape-trigger` | `trigger_scrape` (L4360) |
| GET | `/queue-status` | `queue_status_page` (L9087) |
| GET | `/scrape` | `scrape_management_page` (L9068) |
| GET | `/scrape-control` | `scrape_control_page` (L9081) |
| GET | `/scrape-upcoming` | `scrape_upcoming_page` (L9231) |
