# feature/betting

**対象領域**: 馭券戦略
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 3（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| POST | `/api/betting/optimize` | `api_betting_optimize` (L8878) |
| GET | `/api/betting/pair-odds/{race_id}` | `api_pair_odds` (L8968) |
| GET | `/betting` | `betting_page` (L8869) |
