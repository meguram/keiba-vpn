# feature/race-quality

**対象領域**: レース質分析
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 5（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/race-quality/day` | `api_race_quality_day` (L7095) |
| GET | `/api/race-quality/entrants-aptitude` | `api_race_quality_entrants_aptitude` (L7157) |
| GET | `/api/race-quality/meta` | `api_race_quality_meta` (L7081) |
| GET | `/api/race-quality/race` | `api_race_quality_race` (L7121) |
| GET | `/race-quality` | `race_quality_page` (L9044) |
