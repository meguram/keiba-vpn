# feature/bloodline-pedigree

**対象領域**: 血統・種牡馬クラスタ
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象ページURL数**: 7（`src/api/app.py`, FastAPI legacy, :8000。ブラウザで直接開く画面のURLのみを対象とし、画面から呼ばれる `/api/*` のJSON APIはここでは数えない）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のページURLに関する変更（画面・および画面を支える`/api/*`実装を含む）はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象ページURL

| Path | Handler |
|---|---|
| `/bloodline` | `bloodline_page`(L10428) |
| `/bloodline-cluster` | `bloodline_cluster_page`(L10660) |
| `/bloodline-vector` | `bloodline_vector_page`(L9237) |
| `/course-bloodline` | `course_bloodline_page_redirect`(L11346) |
| `/note-aptitude-race` | `note_aptitude_race_page`(L9274) |
| `/pedigree-map` | `pedigree_map_page`(L9265) |
| `/pedigree-race-stats` | `pedigree_race_stats_page`(L12964) |
