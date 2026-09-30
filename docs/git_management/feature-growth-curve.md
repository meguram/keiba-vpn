# feature/growth-curve

**対象領域**: 成長曲線
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 3（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/growth-curve/status` | `api_growth_curve_status` (L12662) |
| GET | `/api/growth-curve/{horse_id}` | `growth_curve_data` (L12674) |
| GET | `/growth-curve` | `growth_curve_page` (L12568) |
