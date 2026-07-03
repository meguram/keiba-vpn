# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照: AREA-01〜AREA-09

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの予測（勝率・連対率・複勝率・オッズ・回収率・ポジション・脚質・ペース・ラップ）を提供し、かつユーザーが予想の根拠を自ら探索・検証できるデータ分析機能を備えた競馬予測 Web アプリ。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB — 2GB メモリ以内での安定稼働を最優先 |

### 1-2. 最重要設計原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること** が、データ基盤・モデリング・評価・API 設計の全工程の絶対的前提条件である。

---

## 2. 技術スタック

| レイヤー | 採用技術 | 備考 |
|---|---|---|
| データベース | PostgreSQL | `BIGSERIAL` / `TIMESTAMPTZ` / `NUMERIC` / `ARRAY` 型使用 |
| キャッシュ | Redis | TTL 管理・キャッシュキー規約あり（§3-4 参照） |
| API サーバー | Flask | ベースパス `/api/v1`、JSON レスポンス |
| フロントエンド | 未確定（Next.js 採用可否・バージョン等は後続 DEC で決定） | Phase 3 実装 |
| ML フレームワーク | LightGBM（初期）→ LSTM（Phase 4 以降） | セルフホスト推論、外部 API 委託なし |
| スキーママイグレーション | Alembic | 全 DDL 変更をバージョン管理 |
| モデル管理 | MLflow（または同等ツール） | F-16 要件 |
| スクレイピング | Python（`SCRAPING_CONFIG` 準拠） | シングルワーカー・IP ブロック対策あり |
| ストレージ（GCS） | `gs://keiba-vpn-data/` | raw / processed / features / models / predictions / logs |
| 特徴量フォーマット | Parquet | GCS features/ 以下に保存 |

---

## 3. アーキテクチャ設計

### 3-1. システムコンポーネント構成

```
scraper        ── netkeiba.com スクレイパー（concurrent_workers: 1）
db (PostgreSQL) ── Layer 1〜5 データ格納
redis          ── 予測結果キャッシュ（TTL 管理）
api            ── REST API（Flask, /api/v1/...）
ml-worker      ── 特徴量生成・モデル学習・バッチ推論
mlflow         ── モデルバージョン管理
frontend       ── レース一覧・予測表示・データ分析 UI
```

### 3-2. データ層アーキテクチャ（5層構造）

| 層 | 名称 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot` | 追記のみ・削除不可 |

### 3-3. GCS パス設計（`gcs_paths.py` SSoT）

全モジュールは `gcs_paths.py` をインポートして使用する。直接パス文字列を記述することを禁止する。

```
gs://keiba-vpn-data/
├── raw/                          # Layer 1〜5 スクレイピング生データ（HTML）
│   ├── layer1/race_card/{YYYYMMDD}/{race_id}.html
│   ├── layer1/horses/{horse_id}.html
│   ├── layer1/jockeys/{jockey_id}.html
│   ├── layer1/trainers/{trainer_id}.html
│   ├── layer2/race_results/{YYYYMMDD}/{race_id}.html
│   ├── layer4/laps/{YYYYMMDD}/{race_id}.html
│   └── layer5/odds/{YYYYMMDD}/{race_id}/{snapshot_at_iso}.html
├── processed/                    # クリーニング・変換済み Parquet
│   ├── layer1/race_card/{YYYYMMDD}/{race_id}.parquet
│   ├── layer2/race_results/{YYYYMMDD}/{race_id}.parquet
│   ├── layer4/laps/{YYYYMMDD}/{race_id}.parquet
│   └── layer5/odds/{YYYYMMDD}/{race_id}.parquet
├── features/                     # Feature Store（Layer 3 スナップショット）
│   ├── horse_stats/{YYYYMMDD}/{as_of_race_id}/{horse_id}.parquet
│   ├── jockey_stats/{YYYYMMDD}/{as_of_race_id}/{jockey_id}.parquet
│   ├── trainer_stats/{YYYYMMDD}/{as_of_race_id}/{trainer_id}.parquet
│   └── race_combined/{YYYYMMDD}/{race_id}.parquet
├── models/{model_name}/{version}/model.pkl
├── models/{model_name}/{version}/metadata.json
├── predictions/{YYYYMMDD}/{race_id}/{model_version}.parquet
└── logs/scrape_runs/{YYYYMMDD}.jsonl
```

### 3-4. キャッシュ設計（4層構成）

| 層 | 種別 | 対象 | TTL | キャッシュキー |
|---|---|---|---|---|
| L1 | Flask インメモリ（`lru_cache`） | コース・マスター等静的情報 | プロセス再起動まで | 関数単位 |
| L2 | Redis — 予測結果 | T-1〜T-9 全予測 | 発走時刻まで（`EXPIREAT`）/ 発走後 60 秒で自動失効 | `prediction:{race_id}:{model_version}` |
| L3 | Redis — ラップ予測 | T-10〜T-11 ラップ系列 | 発走時刻まで / 発走後 60 秒で自動失効 | `lap:prediction:{race_id}:{model_version}` |
| L4 | Redis — オッズ直近値 | 推論特徴量用最新オッズ | 5 分（オッズ更新間隔に合わせる） | `odds:latest:{race_id}` |

### 3-5. プロセス分離

| プロセス | 役割 | 制約 |
|---|---|---|
| スクレイパー | netkeiba.com データ収集（Layer 1〜5 格納） | `concurrent_workers: 1`（IP ブロック防止） |
| スナップショット集計バッチ | Layer 3 スナップショット生成 | `as_of_race_id` 紐付け必須、results 収集完了後に起動 |
| オッズ収集スケジューラ | 発走当日 08:00〜発走時刻まで定期収集 | タイムウィンドウ制御が必要 |
| 推論バッチ | Stage 1 → Stage 2 順次推論・保存 | 発走 3 時間前までに完了（N-9） |
| API サーバー | REST API 提供・Redis キャッシュ参照 | — |
| Alembic マイグレーション | DDL バージョン管理 | API サーバー起動前に独立実行 |

### 3-6. デプロイ順序

```
1. Alembic DDL マイグレーション実行
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. API サーバー起動
```

---

## 4. 機能要件（確定版）

### 4-1. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 | 担当 Stage |
|---|---|---|---|---|
| T-1 | `win_prob`（勝率） | 多クラス分類（1着） | `NUMERIC(5,4)` | Stage 1 |
| T-2 | `place_prob`（連対率） | バイナリ分類（2着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-3 | `show_prob`（複勝率） | バイナリ分類（3着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-4 | `predicted_win_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-5 | `predicted_place_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-6 | `win_roi`（単回収率） | ポスト計算: `win_prob × predicted_win_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-7 | `show_roi`（複回収率） | ポスト計算: `show_prob × predicted_place_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-8 | `predicted_position` | 順位回帰 / ランキング学習 | `SMALLINT` | Stage 1 |
| T-9 | `predicted_running_style` | 4値分類（`FRONT`/`STALKER`/`MID`/`CLOSER`） | `VARCHAR(10)` | Stage 1 |
| T-10 | `pace_category` | 3値分類（`HIGH`/`MIDDLE`/`SLOW`） | `VARCHAR(10)` | Stage 2 |
| T-11 | `predicted_lap_sec[]` | 時系列回帰（1F 単位系列出力） | `NUMERIC(4,2)[]` | Stage 2 |

> T-6・T-7 は推論結果ではなくポスト計算値。100 以上 = 期待値プラスのバリューベット候補。

### 4-2. 予測機能要件

| # | 要件 | 優先度 |
|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 |
| F-4 | オッズを指定スケジュールでスナップショット取得し Layer 5 に格納する | 高 |
| F-5 | スクレイプ実行ログを `scrape_runs` テーブルで管理する | 高 |
| F-6 | 脚質スコア・クロス特徴量・相対特徴量を特徴量パイプラインで自動生成する | 高 |
| F-7 | Stage 1 モデル（T-1〜T-5, T-8, T-9）を学習・バッチ推論する | 高 |
| F-8 | Stage 2 モデル（T-10〜T-11）を Stage 1 出力を受けて学習・バッチ推論する | 高 |
| F-9 | 推論後に T-6・T-7 を計算し `prediction_results` に保存する | 高 |
| F-10 | 任意レースの全予測ターゲット（T-1〜T-9）を REST API で提供する | 高 |
| F-11 | ラップ予測結果を `furlong_index` 順の系列形式で API から提供する | 中 |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 |
| F-13 | レース一覧・出馬表・AI 予測を統合表示する UI を提供する | 中 |
| F-14 | `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100` の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測を折れ線グラフ（X軸: ハロン番号、Y軸: ラップタイム秒）で可視化する | 低 |
| F-16 | 学習済みモデルのバージョン管理（MLflow）と旧バージョンへのロールバック機能を提供する | 中 |
| F-17 | ラップデータ可用性をサンプル 10 レースで手動確認する（Phase 0 前提条件） | 高 |

### 4-3. データ分析機能要件

| # | 機能名 | 優先度 |
|---|---|---|
| AN-01 | 種牡馬成績多軸フィルタリング分析 | 高 |
| AN-02 | コース別・条件別統計ダッシュボード | 高 |
| AN-03 | 騎手・調教師成績分析 | 高 |
| AN-04 | マイ分析（条件保存・再実行） | 中 |

> **UI 注記要件**: 分析画面には「※ この統計はリアルタイム集計です。AIモデルが予測に使用した時点の特徴量とは異なる場合があります。」を表示する。

> **ML パイプラインとの分離**: 分析 UI は `as_of` 制約不要（事後統計）。AI 予測特徴量は `race_date < as_of_race_id` 制約必須（テンポラルリーク防止）。

### 4-4. REST API エンドポイント

| メソッド | エンドポイント | 説明 | 機能要件 |
|---|---|---|---|
| `GET` | `/api/v1/races` | レース一覧取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}/predictions` | 全予測（T-1〜T-9）取得 | F-10 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11）取得 | F-11 |

#### `GET /api/v1/races/{race_id}/predictions` レスポンス仕様

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

### 4-5. スクレイピング収集スケジュール

| ジョブ名 | トリガー | 間隔 |
|---|---|---|
| `race_card` | レース 3 日前 06:00 JST | 毎日 06:00 更新（発走まで） |
| `odds_snapshot` | 発走当日 08:00〜発走時刻 | 通常: 5分毎 / 発走 30 分前: 2 分毎 / 発走 5 分前: 1 分毎 |
| `race_results` | 発走予定時刻 + 35 分 | リトライ: 5 分間隔 × 最大 6 回 |
| `horse_history` | `race_results` 収集完了後 | 1 回（前走成績更新後に再取得） |

### 4-6. スクレイピング設定

```python
SCRAPING_CONFIG = {
    "request_interval_sec": 2.0,
    "jitter_sec": (0.5, 1.5),
    "concurrent_workers": 1,
    "session_rotate_interval": 50,
    "retry_on_429": True,
    "retry_backoff_base_sec": 30,
    "user_agent_rotate": True,
}
```

---

## 5. 非機能要件（確定版）

| ID | 要件 | 目標値 |
|---|---|---|
| N-1 | API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ラップタイム予測 MAE | ≤ 0.3 秒 |
| N-4 | 勝率 Log Loss 改善（vs. ベースライン） | ベースライン比 −5% 以上 |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | DB 反映遅延（スクレイピング完了から） | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% |
| N-9 | 推論バッチ完了目標 | 発走 3 時間前まで |
| N-10 | テンポラルリーク検知 | CI パイプラインで自動実行・違反時デプロイブロック |
| N-11 | スキーマ変更管理 | Alembic によるバージョン管理必須 |
| N-12 | Redis キャッシュ TTL | 発走時刻まで有効 / 発走後 60 秒で自動失効 |
| N-13 | テストカバレッジ | スクレイパー・特徴量パイプライン・モデル・API 各 ≥ 80% |
| N-14 | 障害・海外レース除外 | `races.is_excluded BOOLEAN DEFAULT FALSE` で予測対象外を管理 |

### 5-1. 監視・アラート SLO

| 監視対象 | SLO / アラート条件 |
|---|---|
| スクレイピング成功率 | 週次で 99% 以下になった場合に通知 |
| DB 反映遅延 | 10 分超過時アラート |
| オッズ欠損率 | 1% 超過時アラート |
| 推論バッチ未完了 | 発走 3 時間前時点で未完了の場合アラート |
| API