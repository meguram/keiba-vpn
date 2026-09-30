# feature/horse-profile

**対象領域**: 馬プロフィール・馬名検索・関係者統計
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 6（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/horse-names/index-meta` | `horse_names_index_meta` (L12812) |
| GET | `/api/horse-names/search` | `search_horse_names` (L12823) |
| GET | `/api/horse/{horse_id}/detail` | `api_horse_detail` (L3226) |
| GET | `/api/horse/{horse_id}/race_performance_history` | `api_horse_race_performance_history` (L3401) |
| GET | `/api/horse/{horse_id}/recent_races` | `api_horse_recent_races` (L3306) |
| GET | `/api/person/{ptype}/{person_id}/stats` | `api_person_stats` (L3696) |
