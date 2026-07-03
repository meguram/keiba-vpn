# AREA-03 — アプリケーション要件（機能要件・非機能要件・API仕様・ユーザーストーリー・データフロー）
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001-本要件定義書の最重要事項はas_of_race_id-によるスナップショット管理でテンポラルリークを.md

---

## 0. 本仕様書の最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、データ基盤・モデリング・評価・API設計の全工程の前提条件となる。

---

## 1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F単位ラップ予測** を実現する競馬予測システム。

対象プロジェクト: `keiba-vpn`

---

## 2. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 |
|---|---|---|---|
| T-1 | 勝率 (`win_prob`) | 多クラス分類（レース内1頭が1着） | `NUMERIC(5,4)` |
| T-2 | 連対率 (`place_prob`) | バイナリ分類（2着以内）× 頭数 | `NUMERIC(5,4)` |
| T-3 | 複勝率 (`show_prob`) | バイナリ分類（3着以内）× 頭数 | `NUMERIC(5,4)` |
| T-4 | 単勝オッズ予測 (`predicted_win_odds`) | 回帰 | `NUMERIC(7,1)` |
| T-5 | 複勝オッズ予測 (`predicted_place_odds`) | 回帰 | `NUMERIC(7,1)` |
| T-6 | 単回収率 (`win_roi`) | 計算値: `win_prob × predicted_win_odds × 100` | `NUMERIC(7,2)` |
| T-7 | 複回収率 (`show_roi`) | 計算値: `show_prob × predicted_place_odds × 100` | `NUMERIC(7,2)` |
| T-8 | ポジション予測 (`predicted_position`) | 順位回帰 / ランキング学習 | `SMALLINT` |
| T-9 | 脚質予測 (`predicted_running_style`) | 4値分類: `FRONT`/`STALKER`/`MID`/`CLOSER` | `VARCHAR(10)` |
| T-10 | ペースカテゴリ予測 (`pace_category`) | 3値分類: `HIGH`/`MIDDLE`/`SLOW` | `VARCHAR(10)` |
| T-11 | 1F単位ラップ予測 (`predicted_lap_sec[]`) | 時系列回帰（系列出力） | `NUMERIC(4,2)[]` |

> T-6・T-7（回収率）はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100以上 = 期待値プラスのバリューベット候補。

---

## 3. 機能要件

| # | 要件 | 優先度 |
|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 |
| F-4 | オッズを指定スケジュール（発走前5分毎〜1分毎）でスナップショット取得し Layer 5 に格納する | 高 |
| F-5 | スクレイプ実行ログ（`target_type`・`status`・`retry_count`）を `scrape_runs` テーブルで管理する | 高 |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 |
| F-11 | ラップ予測結果を系列形式（`furlong_index` 順）で API から提供する | 中 |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 |
| F-13 | レース一覧・出馬表・AI予測を統合表示する UI を提供する | 中 |
| F-14 | 回収率100以上の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能 | 中 |
| F-17 | ラップデータ可用性の事前検証（サンプル10レースで手動確認）を実施する（Phase 0 前提条件） | 高 |

---

## 4. 非機能要件

| # | 要件 | 目標値 |
|---|---|---|
| N-1 | 予測 API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | 予測 API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ラップ予測 MAE（1F単位） | ≤ 0.3 秒 |
| N-4 | 勝率予測 Log Loss（ベースラインオッズ逆数モデル比） | −5% 以上改善 |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | スクレイピング後の DB 反映遅延 | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前5分以内） | ≤ 1% |
| N-9 | モデル推論バッチ完了時刻 | 発走3時間前までに完了 |
| N-10 | 特徴量リーク（未来情報混入）ゼロ | テストデータ時系列分割で検証 |
| N-11 | DDL マイグレーション管理 | 全スキーマ変更を Alembic 等でバージョン管理 |
| N-12 | Redis キャッシュ TTL | 発走まで有効 / 発走後60秒で自動失効 |
| N-13 | テストカバレッジ（スクレイパー・特徴量パイプライン） | ≥ 80% |
| N-14 | 障害レース・海外レースは予測対象外として明示除外 | フラグ管理 |

---

## 5. API 仕様

### 5-1. エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` | `/api/v1/races/{race_id}/predictions` | レース全体の予測結果取得（T-1〜T-11） |
| `GET` | `/api/v1/races/{race_id}/laps` | ラップ予測系列取得（`furlong_index` 順） |

### 5-2. `GET /api/v1/races/{race_id}/predictions` レスポンス仕様

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

**フィールド定義**

| フィールド | 型 | 説明 |
|---|---|---|
| `race_id` | `STRING` | レースID（例: `202506010811`） |
| `model_version` | `STRING` | 推論に使用したモデルバージョン |
| `predicted_at` | `TIMESTAMPTZ` | 推論実行日時（ISO 8601） |
| `pace_prediction.pace_category` | `STRING` | `HIGH` / `MIDDLE` / `SLOW` |
| `pace_prediction.lap_times[].furlong_index` | `INT` | 1始まり（1F目=スタート直後） |
| `pace_prediction.lap_times[].predicted_lap_sec` | `FLOAT` | 予測ラップタイム（秒） |
| `horses[].win_prob` | `FLOAT(5,4)` | 勝率（T-1） |
| `horses[].place_prob` | `FLOAT(5,4)` | 連対率（T-2） |
| `horses[].show_prob` | `FLOAT(5,4)` | 複勝率（T-3） |
| `horses[].predicted_win_odds` | `FLOAT(7,1)` | 予測単勝オッズ（T-4） |
| `horses[].predicted_place_odds` | `FLOAT(7,1)` | 予測複勝オッズ（T-5） |
| `horses[].expected_win_roi` | `FLOAT(7,2)` | 単回収率（T-6） |
| `horses[].expected_show_roi` | `FLOAT(7,2)` | 複回収率（T-7） |
| `horses[].predicted_position` | `INT` | 予測着順（T-8） |
| `horses[].predicted_running_style` | `STRING` | 脚質予測（T-9）: `FRONT`/`STALKER`/`MID`/`CLOSER` |
| `horses[].is_value_bet` | `BOOLEAN` | `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100` の場合 `true` |

### 5-3. キャッシュ仕様

| 項目 | 仕様 |
|---|---|
| キャッシュストア | Redis |
| キャッシュキー（予測結果） | `prediction:{race_id}:{model_version}` |
| キャッシュキー（ラップ予測） | `lap:prediction:{race_id}:{model_version}` |
| TTL | 発走時刻まで有効 / 発走後60秒で自動失効 |

---

## 6. ユーザーストーリー

| ID | ストーリー | 受入条件 |
|---|---|---|
| US-1 | ユーザーとして、任意レースの全出走馬の勝率・連対率・複勝率を一覧で確認したい | `/api/v1/races/{race_id}/predictions` が全馬の T-1〜T-3 を返す |
| US-2 | ユーザーとして、期待値プラスの馬（バリューベット候補）を即座に識別したい | `is_value_bet: true` の馬が UI でハイライト表示される |
| US-3 | ユーザーとして、レースの展開（ペース・ラップ推移）を視覚的に把握したい | ペースカテゴリと1F毎ラップ予測が折れ線グラフで表示される（F-15） |
| US-4 | ユーザーとして、各馬の予測脚質とポジションを確認し展開を読みたい | 全馬の `predicted_running_style`・`predicted_position` が一覧表示される |
| US-5 | データエンジニアとして、スクレイピングの成否・リトライ状況を追跡したい | `scrape_runs` テーブルに `target_type`・`status`・`retry_count