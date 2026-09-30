# feature/admin-ops

**対象領域**: 管理者運用（cron・構造チェック・システム統計・ログ）
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 18（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/ai-sla` | `ai_sla_page` (L9039) |
| GET | `/api/admin/auto-scrape-status` | `get_auto_scrape_status` (L8034) |
| POST | `/api/admin/auto-scrape/{task}/trigger` | `trigger_auto_scrape` (L8068) |
| GET | `/api/admin/cron-jobs` | `get_cron_jobs` (L7563) |
| POST | `/api/admin/cron-jobs/daily-shutuba/trigger` | `trigger_daily_shutuba` (L7749) |
| POST | `/api/admin/cron-jobs/disk-cache-cleanup/trigger` | `trigger_disk_cache_cleanup` (L7658) |
| POST | `/api/admin/cron-jobs/logs-retention/trigger` | `trigger_logs_retention` (L7719) |
| POST | `/api/admin/cron-jobs/queue-maintain/trigger` | `trigger_queue_maintain` (L7689) |
| POST | `/api/admin/invalidate-race-list-caches` | `api_invalidate_race_list_caches` (L12050) |
| GET | `/api/admin/server-logs` | `api_admin_server_logs` (L9177) |
| GET | `/api/admin/system-stats` | `get_system_stats` (L7910) |
| POST | `/api/structure-check` | `trigger_structure_check` (L7487) |
| GET | `/api/structure-check/schedule` | `get_structure_check_schedule` (L7537) |
| GET | `/api/structure-fingerprints` | `get_structure_fingerprints` (L8088) |
| GET | `/api/structure-report` | `get_structure_report` (L7474) |
| GET | `/api/structure-status` | `get_structure_status` (L7457) |
| GET | `/cron-jobs` | `cron_jobs_page` (L9093) |
| GET | `/server-logs` | `server_logs_page` (L9217) |
