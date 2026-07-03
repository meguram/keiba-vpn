# Thinking Log: TASK-022 要件定義書 — ドメイン別技術スタック・レイテンシ・コスト統合

- **Date**: 2026-07-03
- **Author**: Orchestrator（全サブエージェント統合）
- **Task**: TASK-022 — 要件定義書にドメイン別技術スタック・レイテンシ・コスト記載を追加
- **Consensus**: YES（Loop 5/5）

---

## 主要な判断記録

### 判断1: レイテンシ SLO を 3 軸に分離した理由

- **問題**: API レイテンシ・データ鮮度・バッチ処理時間を単一のSLOセクションに混在させると、障害時の原因切り分けが困難になる
- **根拠**: data-engineer（データ鮮度の独立管理）と ai-model-engineer（バッチ処理レイテンシの独立管理）の両エージェントから Loop 2/3 で指摘
- **決定**: ① API レイテンシ SLO（エンドポイント別 P50/P95/P99）、② データ鮮度 SLO（パイプライン完了時刻）、③ バッチ処理レイテンシ（ジョブ実行時間）の 3 軸に分離

### 判断2: Flask → FastAPI への変更

- **経緯**: Loop 1〜3 で backend-engineer が継続的に指摘
- **根拠**: VPS 2GB 制約下（DEC-004）での非同期処理優位性・型安全性・OpenAPI 自動生成
- **決定**: FastAPI を採用。「ML 資産との親和性」は Python であれば Flask/FastAPI に差がないため選定理由から除外

### 判断3: コスト試算と DEC-006 の整合方針

- **問題**: DEC-006（フェーズ別月額合計）と今回の試算（サービス別詳細）が重複する懸念
- **決定**: DEC-006 を Single Source of Truth（フェーズ別合計）として維持し、本要件定義書のコスト試算はドメイン別補足として位置づける
- **位置づけ**: DEC-006（フェーズ別合計）+ 要件定義書（サービス別詳細）で相互補完

### 判断4: Temporal Leakage 禁止（N-23）を非機能要件に格上げ

- **提案者**: ai-model-engineer（Loop 2）
- **根拠**: 予測精度の不正向上を防ぐ根本制約であり、実装フェーズ全体に影響する
- **決定**: AI モデル品質制約として独立項目化（N-23）。Phase 1 完了前に feature 設計レビューを必須とする

### 判断5: ドメイン別スタック表への ETL 行・AI/ML 行の追加

- **要望**: data-engineer（ETL 固有スタック行とデータ鮮度 SLO の独立化）、ai-model-engineer（AI/ML パイプライン行の明示化）
- **決定**: D-4（ETL/スクレイパー）と D-5（AI/ML 推論バッチ）を独立ドメインとして明示

---

## ブレインストーム合意プロセス（Loop 要約）

| Loop | 状態 | 主な変更 |
|------|------|---------|
| Loop 0 | 初期提案生成 | decisions-context-agent が DEC-001〜006 を確認。提案 3〜5 件を生成 |
| Loop 1 | 懸念あり | backend-engineer: Flask→FastAPI、ai-model-engineer: コスト試算の途中切断、data-engineer: オッズ高頻度取得フロー |
| Loop 2 | 部分承認 | backend-engineer: FastAPI 選定理由強化要求。ai-model-engineer: コスト試算完成を条件付き承認。data-engineer: ETL 行独立化要求 |
| Loop 3〜4 | 懸念あり → 修正 | 各エージェントが細部の整合を要求、proposal-agent が修正を反映 |
| Loop 5 | 全員承認 ✓ | backend-engineer・data-engineer・ai-model-engineer 全員が APPROVE |

---

## 未解決事項（Human 判断待ち）

- `GET /api/v1/races/today`、`/api/v1/predictions/{race_id}`、`POST /api/v1/filter/stats` の `auth_required` フラグ（DEC-005 残課題）
- スクレイピング対象サイトの利用規約・法的リスク確認（機械的判断不可）
- AI モデル精度基準 N-24（複勝的中率 ≥ 35%）の実績ベースライン確認

---

## 参照

- DEC-004: スクレイパー・推論バッチの独立プロセス化
- DEC-005: 要件定義書 v1.0（機能要件・非機能要件ベース）
- DEC-006: 運用コスト試算（フェーズ別月額）
- DEC-007: 本タスク成果（ドメイン別スタック統合版）
