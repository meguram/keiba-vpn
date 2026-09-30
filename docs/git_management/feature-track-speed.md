# feature/track-speed

**対象領域**: トラックスピード指標
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 12（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| POST | `/api/track-speed/assign` | `track_speed_assign` (L12176) |
| GET | `/api/track-speed/by-category` | `api_track_speed_by_category` (L12283) |
| GET | `/api/track-speed/dates` | `api_track_speed_dates` (L12124) |
| GET | `/api/track-speed/day` | `api_track_speed_day` (L12144) |
| GET | `/api/track-speed/meta` | `api_track_speed_meta` (L12103) |
| GET | `/api/track-speed/race-horses` | `api_race_horses` (L12203) |
| POST | `/api/track-speed/rebuild-baselines` | `track_speed_rebuild_baselines` (L12168) |
| GET | `/api/track-speed/status` | `track_speed_status` (L12156) |
| GET | `/api/track-speed/validate-perf` | `api_validate_perf` (L12191) |
| GET | `/api/track-speed/venues` | `api_track_speed_venues` (L12133) |
| GET | `/track-speed` | `track_speed_page` (L12081) |
| GET | `/track-speed/dev` | `track_speed_dev_page` (L12090) |
