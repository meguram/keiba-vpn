# feature/core-platform

**対象領域**: コアプラットフォーム（ヘルス/認証/ダッシュボード）
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 10（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/` | `dashboard` (L962) |
| GET | `/api/auth/status` | `auth_status` (L951) |
| GET | `/api/data/{category}/{key}` | `get_raw_data` (L3166) |
| GET | `/api/gcs-stats` | `get_gcs_stats` (L980) |
| GET | `/api/health` | `health_check` (L880) |
| POST | `/api/html-archive/cleanup` | `html_archive_cleanup` (L896) |
| GET | `/api/inference/health` | `api_inference_health` (L7232) |
| GET | `/login` | `login_page` (L914) |
| POST | `/login` | `login_submit` (L928) |
| GET | `/logout` | `logout` (L946) |
