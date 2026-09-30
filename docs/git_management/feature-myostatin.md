# feature/myostatin

**対象領域**: ミオスタチン遺伝子解析
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 4（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/myostatin` | `myostatin_data` (L12363) |
| POST | `/api/myostatin/predict` | `myostatin_predict` (L12377) |
| POST | `/api/myostatin/recalculate` | `myostatin_recalculate` (L12399) |
| GET | `/myostatin` | `myostatin_page` (L12355) |
