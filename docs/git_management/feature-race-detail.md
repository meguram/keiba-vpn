# feature/race-detail

**対象領域**: レース詳細・予測表示
**分岐元**: `main` @ `dfe590e`（2026-09-30、`hotfix/race-detail` から改名して再作成）
**対象ページURL数**: 1（`src/api/app.py`, FastAPI legacy, :8000。ブラウザで直接開く画面のURLのみを対象とし、画面から呼ばれる `/api/*` のJSON APIはここでは数えない）

> **2026-09-30 解決済み**: `hotfix/race-detail` として問題を確認・対応済み。
> (1) 分類ミス — `.../tracking-difficulty`・`.../final-odds` が誤って本領域に分類されていた点を修正
> （それぞれ `feature/tracking-difficulty`・`hotfix/odds-final-odds` へ移動済み）。
> (2) 設計上の懸念 — `/api/race/{race_id}/predictions`（GCS）とFlask v1版（PostgreSQL）のデータソース分裂は
> 既に詳細調査済み。3系統の書き込みパス（推論パイプライン・legacy手動トリガ・バッチCLI）がいずれも
> 自動実行されていないことを`crontab`等で確認しており、現状は実害が無いためコード統合は行わない方針を維持。
> 詳細は `docs/operations/service-endpoints.md`「レース予測の書き込みパスが3系統ある」参照。
> 未解決の問題が無いため `feature/race-detail` に改名した。

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のページURLに関する変更（画面・および画面を支える`/api/*`実装を含む）はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象ページURL

| Path | Handler |
|---|---|
| `/race/{race_id}` | `race_detail_page`(L6479) |
