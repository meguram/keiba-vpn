# feature/odds-final-odds

**対象領域**: オッズ・最終オッズ
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 5（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/odds/history/{race_id}` | `get_odds_history` (L8406) |
| GET | `/api/odds/predict/{race_id}` | `get_predicted_odds_api` (L8425) |
| POST | `/api/odds/snapshot/{race_id}` | `record_odds_snapshot` (L8380) |
| POST | `/api/odds/train` | `train_odds_model` (L8320) |
| GET | `/api/odds/train/status` | `get_odds_training_status` (L8359) |
