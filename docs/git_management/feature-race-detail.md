# feature/race-detail

**対象領域**: レース詳細・予測表示
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 15（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/megu-index-dates` | `get_megu_index_dates` (L2778) |
| GET | `/api/predictions` | `get_predictions` (L972) |
| GET | `/api/race-list/{date}` | `get_race_list_for_date` (L2618) |
| GET | `/api/race/{race_id}` | `get_race_detail` (L6511) |
| GET | `/api/race/{race_id}/bloodline-aptitude` | `get_bloodline_aptitude` (L6646) |
| GET | `/api/race/{race_id}/bundle` | `api_race_bundle` (L3618) |
| GET | `/api/race/{race_id}/final-odds` | `api_final_odds` (L7182) |
| POST | `/api/race/{race_id}/final-odds/precompute` | `api_precompute_final_odds` (L7206) |
| POST | `/api/race/{race_id}/predict` | `run_race_prediction` (L6787) |
| GET | `/api/race/{race_id}/predictions` | `get_race_predictions` (L6608) |
| GET | `/api/race/{race_id}/result-status` | `api_race_result_status` (L6972) |
| GET | `/api/race/{race_id}/tracking-difficulty` | `api_tracking_difficulty` (L7000) |
| POST | `/api/race/{race_id}/tracking-difficulty/precompute` | `api_precompute_tracking_difficulty` (L7046) |
| GET | `/api/upcoming-races` | `get_upcoming_races` (L4561) |
| GET | `/race/{race_id}` | `race_detail_page` (L6479) |
