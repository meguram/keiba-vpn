# feature/cushion

**対象領域**: クッション値（トラックコンディション）
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 8（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| POST | `/api/cushion/admin/sync-gcs` | `api_cushion_admin_sync_gcs` (L11639) |
| POST | `/api/cushion/admin/sync-preprocessed` | `api_cushion_admin_sync_preprocessed` (L11668) |
| GET | `/api/cushion/data` | `api_cushion_data` (L11553) |
| POST | `/api/cushion/live` | `api_cushion_live` (L11723) |
| GET | `/api/cushion/live/check` | `api_cushion_live_check` (L11799) |
| GET | `/api/cushion/live/status` | `api_cushion_live_status` (L11755) |
| GET | `/api/cushion/schedule` | `api_cushion_schedule` (L11809) |
| GET | `/api/cushion/stats` | `api_cushion_stats` (L11575) |
