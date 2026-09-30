# feature/monitor-quality

**対象領域**: 監視・データ品質チェック
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 15（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/coverage-calendar` | `get_coverage_calendar` (L2986) |
| GET | `/api/date-calculated-matrix` | `get_date_calculated_matrix` (L2141) |
| GET | `/api/date-race-matrix` | `get_date_race_matrix` (L1985) |
| GET | `/api/date-raw-matrix` | `get_date_raw_matrix` (L2119) |
| GET | `/api/monitor/context` | `get_monitor_context` (L2096) |
| GET | `/api/monitor/missing-dates-summary` | `api_monitor_missing_dates_summary` (L2413) |
| GET | `/api/monitor/opening-date-info` | `get_opening_date_info` (L2197) |
| GET | `/api/quality-check/calendar` | `get_quality_check_calendar` (L2239) |
| POST | `/api/quality-check/enqueue` | `post_quality_check_enqueue` (L2166) |
| GET | `/api/quality-check/health` | `get_quality_check_health` (L2184) |
| GET | `/api/quality-check/jobs` | `get_quality_check_jobs` (L2177) |
| POST | `/api/quality-check/remediate` | `post_quality_check_remediate` (L2217) |
| GET | `/api/row-data-coverage` | `get_row_data_coverage` (L2400) |
| GET | `/data-viewer` | `data_viewer` (L4020) |
| GET | `/monitor` | `scrape_monitor_page` (L1374) |
