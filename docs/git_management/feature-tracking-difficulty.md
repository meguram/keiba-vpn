# feature/tracking-difficulty

**対象領域**: 追走難度
**分岐元**: `main` @ `b97874f`（2026-09-30、`hotfix/tracking-difficulty` から改名して再作成）
**対象ページURL数**: 1（`src/api/app.py`, FastAPI legacy, :8000。ブラウザで直接開く画面のURLのみを対象とし、画面から呼ばれる `/api/*` のJSON APIはここでは数えない）

> **2026-09-30 解決済み**: `hotfix/tracking-difficulty` として問題を修正済み。
> (1) 分類ミス — `/api/race/{race_id}/tracking-difficulty`(+precompute) が誤って `race-detail` に
> 分類されていた点を修正。
> (2) 設計上の問題 — Flask v1 (`/api/v1/races/<race_id>/tracking-difficulty`) が legacy に対して
> パラメータ不足（`force_refresh`等）だった点を解消し完全パリティ化、`POST .../precompute` をv1にも新規追加、
> v1側の `HybridStorage()` 直接生成をシングルトン化。`make test`（439 passed）で検証済み。
> 未解決の問題が無くなったため `feature/tracking-difficulty` に改名した。

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のページURLに関する変更（画面・および画面を支える`/api/*`実装を含む）はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象ページURL

| Path | Handler |
|---|---|
| `/tracking-difficulty` | `tracking_difficulty_page`(L9030) |
