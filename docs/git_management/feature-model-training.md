# feature/model-training

**対象領域**: モデル学習・シミュレーション・バックフィル
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 13（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| POST | `/api/backfill/start` | `start_backfill` (L8589) |
| GET | `/api/backfill/status` | `get_backfill_status` (L8535) |
| GET | `/api/model/info` | `get_model_info` (L8273) |
| POST | `/api/race-lists-backfill/start` | `race_lists_backfill_start` (L8712) |
| GET | `/api/race-lists-backfill/status` | `race_lists_backfill_status` (L8657) |
| POST | `/api/race-lists-backfill/stop` | `race_lists_backfill_stop` (L8818) |
| GET | `/api/simulation/params` | `get_current_composite_params` (L8510) |
| POST | `/api/simulation/run` | `run_composite_simulation` (L8455) |
| GET | `/api/simulation/status` | `get_simulation_status` (L8498) |
| POST | `/api/train` | `trigger_training` (L8123) |
| POST | `/api/train/ensemble` | `trigger_ensemble_training` (L8187) |
| GET | `/api/train/ensemble/status` | `get_ensemble_training_status` (L8261) |
| GET | `/api/train/status` | `get_training_status` (L8170) |
