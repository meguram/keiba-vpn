# feature/scraping-queue

**対象領域**: スクレイピング・キュー管理
**分岐元**: `main` @ `42ade1e`（2026-09-30、`hotfix/scraping-queue` から改名して再作成）
**対象ページURL数**: 4（`src/api/app.py`, FastAPI legacy, :8000。ブラウザで直接開く画面のURLのみを対象とし、画面から呼ばれる `/api/*` のJSON APIはここでは数えない）

> **2026-09-30 解決済み**: `hotfix/scraping-queue` として問題を確認済み。分類ミス（`/api/cushion/scrape`
> (+status) が誤って本領域に分類されていた点、`feature/cushion` へ移動済み）は解消済み。それ以外の
> 設計上の問題は無いことを確認した（`add`/`add-job`/`add-batch`系は用途の異なる別実装で重複ではない、
> `HybridStorage()`直接生成もなし）。未解決の問題が無いため `feature/scraping-queue` に改名した。

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のページURLに関する変更（画面・および画面を支える`/api/*`実装を含む）はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象ページURL

| Path | Handler |
|---|---|
| `/queue-status` | `queue_status_page`(L9087) |
| `/scrape` | `scrape_management_page`(L9068) |
| `/scrape-control` | `scrape_control_page`(L9081) |
| `/scrape-upcoming` | `scrape_upcoming_page`(L9231) |
