# DEC-024: AI 精度目標 & スクレイピング対象範囲確定

**Status**: ACCEPTED
**Date**: 2026-07-06
**Author**: Orchestrator（REVIEW-001 確認結果）
**Related**: DEC-005, DEC-015, AREA-01, AREA-04, AREA-06, AREA-07

---

## Context

REVIEW-001（要件定義書レビュー）において、AI モデル精度基準とスクレイピング対象サイトについてプロダクトオーナーが最終確認を実施した。

## Decision

### 1. AI モデル精度目標（N-24）

| 指標 | 目標値 |
|------|--------|
| 複勝的中率 | **≥ 40%** |

- 旧暫定値（35%）を 40% に引き上げ確定
- Phase 1 では動作優先のため精度ゲートは設けない
- Phase 2 以降のモデル評価・デプロイ判断基準として 40% を適用する

### 2. スクレイピング対象サイト（確定）

| サイト | 対象 | 理由 |
|--------|------|------|
| netkeiba.com | **継続** | 法的リスクなし（元データをそのまま提供しない・AI 予測付与のみ） |
| JRA 公式（jra_cushion） | **継続** | 同上 |
| SmartRC（smartrc.jp） | **除外** | スクレイピング対象から削除 |

**法的リスク判断根拠**: ユーザーに提供するのは出馬表の一部情報抜粋 + 独自 AI 予測結果のみ。元データをそのままの形で提供して金銭を得る用途ではないため問題なし。

### 3. SmartRC 除外による影響

SmartRC から取得していた特徴量（`cr_value`・`first_furlong_time`・`estimated_popularity`）は廃止。

- `smartrc_race` テーブル・GCS パス・スクレイピングジョブをすべて削除
- SLA 1/3/6 の cron ジョブから `smartrc_race` を除去
- AREA-04 の scraper 設定から SmartRC を削除
- AREA-06 の GCS パス一覧から `smartrc_race` を削除
- AREA-07 の特徴量リストから SmartRC 特徴量を削除

## Consequences

- AREA-01: N-24 を 複勝的中率 ≥ 40% として明記、SmartRC データソース欄を削除
- AREA-04: SmartRC スクレイパー設定・SLA 記述を削除
- AREA-06: `smartrc_race` GCS パス・ETL パイプライン記述を削除
- AREA-07: SmartRC 由来特徴量（cr_value, first_furlong_time, estimated_popularity）を削除
- MASTER.md: SmartRC をデータソース一覧から削除

---
*Last updated: 2026-07-06*
