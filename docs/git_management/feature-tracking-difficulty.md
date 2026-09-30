# feature/tracking-difficulty

**対象領域**: 追走難度
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 3（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/tracking-difficulty/status` | `api_tracking_difficulty_status` (L6983) |
| POST | `/api/tracking-difficulty/train` | `api_train_tracking_difficulty` (L7259) |
| GET | `/tracking-difficulty` | `tracking_difficulty_page` (L9030) |
