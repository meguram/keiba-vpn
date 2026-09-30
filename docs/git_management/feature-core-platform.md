# feature/core-platform

**対象領域**: コアプラットフォーム（ヘルス/認証/ダッシュボード）
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象ページURL数**: 3（`src/api/app.py`, FastAPI legacy, :8000。ブラウザで直接開く画面のURLのみを対象とし、画面から呼ばれる `/api/*` のJSON APIはここでは数えない）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のページURLに関する変更（画面・および画面を支える`/api/*`実装を含む）はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象ページURL

| Path | Handler |
|---|---|
| `/` | `dashboard`(L962) |
| `/login` | `login_page`(L914), `login_submit`(L928) |
| `/logout` | `logout`(L946) |
