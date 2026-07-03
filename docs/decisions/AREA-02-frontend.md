# AREA-05 — フロントエンド要件（Next.js, ISR/CSR/SSG, UX設計, PWA, Lighthouse, パフォーマンス最適化）
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001

---

## 1. 概要

本仕様書は、競馬予測システム「keiba-vpn」のフロントエンド実装に関する要件を定義する。DEC-001 に記載されたフロントエンド関連要件（F-13〜F-15）を基礎として整理する。

> **注意**: DEC-001 はデータ基盤・モデリング・API 要件を主軸とした決定文書であり、フロントエンド固有の技術要件（Next.js のレンダリング戦略、PWA、Lighthouse スコア目標、詳細なパフォーマンス最適化方針など）は現時点で明示的に決定されていない。本仕様書は DEC-001 から抽出可能な範囲の要件を確定事項として記載し、未決定事項を明示する。

---

## 2. 機能要件（フロントエンド）

DEC-001 § 4 より抽出した、フロントエンドが担当する機能要件は以下のとおり。

| # | 要件 | 優先度 |
|---|---|---|
| F-13 | レース一覧・出馬表・AI予測を統合表示する UI を提供する | 中 |
| F-14 | 回収率100以上の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 |

### 2-1. レース一覧・出馬表・AI予測統合画面（F-13）

表示すべきデータ項目（DEC-001 § 8 APIレスポンス仕様より）:

- レース基本情報: `race_id`、開催日時、コース（距離・芝/ダート・方向）、馬場状態、天候、グレード
- 出走馬ごと:
  - 馬番 (`post_no`)、馬名 (`horse_id`)
  - 勝率 (`win_prob`)、連対率 (`place_prob`)、複勝率 (`show_prob`)
  - 予測単勝オッズ (`predicted_win_odds`)、予測複勝オッズ (`predicted_place_odds`)
  - 単回収率 (`expected_win_roi`)、複回収率 (`expected_show_roi`)
  - 予測ポジション (`predicted_position`)、脚質予測 (`predicted_running_style`: FRONT / STALKER / MID / CLOSER)
  - バリューベットフラグ (`is_value_bet`)

### 2-2. バリューベット候補ハイライト（F-14）

- 判定条件: `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100`
- 表示仕様: バリューベット対象馬を視覚的にハイライト（色・アイコン等で強調）
- API フィールド `is_value_bet: true` を参照して表示を切り替える

### 2-3. ラップ予測折れ線グラフ（F-15）

- 表示データ: `pace_prediction.lap_times` 配列（`furlong_index` × `predicted_lap_sec`）
- グラフ種別: 折れ線グラフ（X軸: ハロン番号、Y軸: ラップタイム（秒））
- 付帯情報: ペースカテゴリ（`pace_category`: HIGH / MIDDLE / SLOW）をグラフタイトルまたはラベルで表示

---

## 3. データ連携（API仕様）

フロントエンドが消費する REST API エンドポイント（DEC-001 § 8 より）:

```
GET /api/v1/races/{race_id}/predictions
```

レスポンス構造（抜粋）:

```json
{
  "race_id": "202506010811",
  "model_version": "v1.2.0",
  "predicted_at": "2025-06-01T08:30:00+09:00",
  "pace_prediction": {
    "pace_category": "MIDDLE",
    "lap_times": [
      { "furlong_index": 1, "predicted_lap_sec": 12.3 },
      { "furlong_index": 2, "predicted_lap_sec": 11.8 }
    ]
  },
  "horses": [
    {
      "horse_id": "2019105678",
      "post_no": 3,
      "win_prob": 0.1823,
      "place_prob": 0.3241,
      "show_prob": 0.4815,
      "predicted_win_odds": 5.2,
      "predicted_place_odds": 2.1,
      "expected_win_roi": 94.8,
      "expected_show_roi": 101.1,
      "predicted_position": 2,
      "predicted_running_style": "STALKER",
      "is_value_bet": true
    }
  ]
}
```

キャッシュ制約（DEC-001 § 5 N-12 より）:
- Redis TTL: 発走時刻まで有効 / 発走後60秒で自動失効
- キャッシュヒット時のレスポンスタイム目標: ≤ 200 ms（N-1）
- キャッシュミス時のレスポンスタイム目標: ≤ 2,000 ms（N-2）

---

## 4. 非機能要件（フロントエンド関連）

DEC-001 から抽出できるフロントエンドに影響する非機能要件:

| # | 要件 | 目標値 |
|---|---|---|
| N-1 | API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |

---

## 5. 未決定事項（後続 DEC による決定が必要）

以下の事項は DEC-001 に記載がなく、現時点で未決定である。後続の決定文書（DEC-002 以降）で確定させること。

| 項目 | 内容 |
|---|---|
| フレームワーク | Next.js の採用可否・バージョン（App Router / Pages Router）|
| レンダリング戦略 | 各画面への ISR / CSR / SSG / SSR の適用方針と revalidate 間隔 |
| PWA対応 | Service Worker・オフライン対応・インストール可否 |
| Lighthouseスコア目標 | Performance / Accessibility / Best Practices / SEO の各スコア目標値 |
| パフォーマンス最適化 | Core Web Vitals (LCP / FID / CLS) 目標値、画像最適化・コード分割方針 |
| グラフライブラリ | F-15 折れ線グラフの実装ライブラリ（Recharts / Chart.js / Victory 等）|
| デザインシステム | コンポーネントライブラリ・UIフレームワーク・カラーパレット |
| 認証・アクセス制御 | ユーザー認証の有無・ルートガード方針 |
| 国際化 | 日本語専用か多言語対応か |
| 実装フェーズ | Phase 3 に該当（DEC-001 § 6 参照）|

---

## 6. 実装フェーズ位置づけ

DEC-001 § 6 ロードマップより、フロントエンド実装（F-13〜F-15）は **Phase 3** に位置する。

```
Phase 0: ラップデータ可用性検証
  └─→ Phase 1: データ基盤構築
        └─→ Phase 2: 特徴量パイプライン + Stage 1 モデル
              └─→ Phase 3: Stage 2 + API + UI 構築  ← フロントエンド実装
```

Phase 3 の完了条件: 任意レースの全予測ターゲットが API 経由で取得でき、UI に表示されること。

---

## 7. 用語定義（フロントエンド関連）

| 用語 | 定義（DEC-001 § 9 より） |
|---|---|
| バリューベット | 回収率が100以上、すなわち期待値がプラスの馬券 |
| 単回収率 | `勝率 × 単勝オッズ × 100`。100超 = 期待値プラス |
| 複回収率 | `複勝率 × 複勝オッズ中値 × 100`。100超 = 期待値プラス |
| ペースカテゴリ | 前半3F/後半3Fの差分から分類: HIGH（前傾）・MIDDLE（平均）・SLOW（後傾）|
| 脚質スコア | −5（逃）〜 +5（追込）。UI 表示時は FRONT / STALKER / MID / CLOSER のラベルで表現 |