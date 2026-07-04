# keiba-vpn — Agent Activity Dashboard

_Last updated: 2026-07-04_

---

## Active Tasks

| Task ID | Title | Agent(s) | Status | Updated |
|---|---|---|---|---|
| — | — | — | — | — |

**Status values**: `IN_PROGRESS` | `NEEDS_REVIEW` | `DONE` | `BLOCKED`

---

## Pending Reviews

| Review ID | Title | Agent | Requested By | Status |
|---|---|---|---|---|
| REVIEW-001 | 要件定義書レビュー（認証ポリシー・コスト・法的リスク確認） | Orchestrator | Human (PO / 開発リード) | PENDING_REVIEW |

**確認項目**:
1. コスト試算の規模前提（小規模: 〜50 req/日など）が実態と合致するか
2. AI モデル精度基準（N-24: 複勝的中率 ≥ 35%）の目標値が妥当か
3. スクレイピング対象サイトの利用規約・法的リスク（人間判断必須）
4. `auth_required` フラグ未決定の 3 エンドポイントの決定（DEC-005 残課題）
5. Phase 1 の期間目安（4〜6 週）が想定人員規模と合致するか
6. Temporal Leakage 禁止制約（N-23）の具体的な feature リスト確認

---

## Recent Decisions

| Decision ID | Title | Agent | Date |
|---|---|---|---|
| DEC-001 | フロントエンド・バックエンド言語選定 | 複数エージェント | 2026-07-02 |
| DEC-002 | GCS 活用方針 | 複数エージェント | 2026-07-02 |
| DEC-003 | 追加決定事項 | 複数エージェント | 2026-07-02 |
| DEC-004 | Phase 1 最優先: スクレイパー・推論バッチを Web サーバから独立プロセスとして切り離す | 複数エージェント | 2026-07-02 |
| DEC-005 | 全エージェント合意済みの要件定義書 v1.0 確定（F-26機能要件・N-15非機能要件） | 複数エージェント | 2026-07-02 |
| DEC-006 | 運用コスト試算セクション追加（Phase 1〜3 月額費用を項目別に試算） | 複数エージェント | 2026-07-02 |
| DEC-007 | ドメイン別技術スタック・レイテンシ SLO・コスト試算を要件定義書に統合（TASK-022/023） | 複数エージェント | 2026-07-03 |
| DEC-008 | archive（既存コードベース）参照による仕様書リニューアル — GCS パス・Cron SLA・スクレイピング設定・UI を実装準拠に更新 | Orchestrator | 2026-07-04 |

---

## Open Issues / Blockers

| # | Description | Blocking Task | Owner |
|---|---|---|---|
| 1 | `GET /api/v1/races/today` の `auth_required` フラグ未決定 | Phase 2 実装開始 | Human (PO) |
| 2 | `/api/v1/predictions/{race_id}` の `auth_required` フラグ未決定 | Phase 2 実装開始 | Human (PO) |
| 3 | `POST /api/v1/filter/stats` の `auth_required` フラグ未決定 | Phase 2 実装開始 | Human (PO) |
| 4 | スクレイピング対象サイトの利用規約・法的リスク確認 | Phase 1 実装開始 | Human (PO / 法務) |

---

## Thinking Log Index

| Date | Agent | Task ID | Topic |
|---|---|---|---|
| 2026-07-02 | operations-engineer, backend-engineer, frontend-engineer, cost-optimizer | TASK-001 | GCS fetch + Python スタック検討 |
| 2026-07-02 | frontend-engineer | TASK-002 | JavaScript UX 検討 |
| 2026-07-02 | backend-engineer | TASK-003 | バックエンド設計 |
| 2026-07-02 | frontend-engineer, backend-engineer, data-engineer, operations-engineer, cost-optimizer | TASK-004 | keiba-vpn 総合検討 |
| 2026-07-02 | frontend-engineer, backend-engineer | TASK-005 | keiba-vpn フロント・バック統合 |
| 2026-07-02 | backend-engineer, cost-optimizer, operations-engineer | TASK-006 | keiba-vpn コスト・運用 |
| 2026-07-02 | brainstorm | TASK-007 | GCS PC ブレインストーム |
| 2026-07-02 | brainstorm | TASK-008 | ブレインストーム |
| 2026-07-02 | proposal-agent | TASK-009 | 提案作成 |
| 2026-07-02 | operations-engineer | TASK-010 | git push / サイト稼働 |
| 2026-07-02 | brainstorm | TASK-011 | ブレインストーム |
| 2026-07-02 | revision-agent | TASK-012 | リビジョン |
| 2026-07-02 | revision-agent | TASK-013 | GCS リビジョン |
| 2026-07-02 | brainstorm | TASK-014 | Phase 1 AI ブレインストーム |
| 2026-07-02 | brainstorm | TASK-015 | Decision / Requirements ブレインストーム |
| 2026-07-02 | revision-agent | TASK-016 | Flask → FastAPI リビジョン |
| 2026-07-02 | revision-agent | TASK-017 | quality-reviewer リビジョン |
| 2026-07-02 | revision-agent | TASK-018 | リビジョン |
| 2026-07-02 | brainstorm | TASK-019 | Decision / Requirements ブレインストーム |
| 2026-07-02 | brainstorm | TASK-020 | AI Web API ブレインストーム |
| 2026-07-02 | revision-agent | TASK-021 | リビジョン |
| 2026-07-03 | brainstorm | TASK-022 | ドメイン別技術スタック・レイテンシ・コストを要件定義書に追加 |
| 2026-07-03 | revision-agent | TASK-023 | DASHBOARD.md 補完 + thinking-log 追加 |

---

## Architecture Snapshot

- **Frontend**: TypeScript / Next.js 15 (App Router) — Tailwind CSS v3、Chart.js v4、D3.js v7
- **Backend**: Python / Flask — ConoHa VPS（RAM 2GB 制約）、Redis（予測キャッシュ + JWT）
- **Database**: PostgreSQL（VPS）+ Redis（キャッシュ・セッション）
- **Scraping**: Python、netkeiba.com（一次）+ SmartRC（二次）+ JRA公式（jra_cushion）— Cron SLA 0〜6 + バックフィル
- **AI Model**: LightGBM — keiba_lgbm / tracking_difficulty / final_odds / pace_predictor（MLflow 管理）
- **Object Storage**: GCS（`gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/`）— HybridStorage（L1メモリ→L2ディスク→L3 GCS）
- **Data Sources**: netkeiba.com（17 カテゴリ）、SmartRC（smartrc_race）、JRA公式（jra_cushion）

## Next Actions

1. REVIEW-001 の Human レビュー完了（プロダクトオーナー / 開発リード）
2. `auth_required` フラグ未決定エンドポイント 3 件の決定（DEC-005 残課題）
3. Phase 1 実装開始（レビュー完了後）
