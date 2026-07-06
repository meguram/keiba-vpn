# keiba-vpn 競馬予測システム — マスター仕様書

> 最終更新: 2026-07-06 | 参照: AREA-01〜AREA-10

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測、ならびに逃げ馬ペース予測・1F 単位ラップ予測を実現する競馬予測 Web アプリ。加えて、ユーザーが予想の根拠を自ら探索・検証するためのデータ分析機能（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com（一次）、SmartRC（二次）、JRA 公式 |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ）/ ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB（prod 環境）— 2GB 以内での安定稼働を最優先 |

### 1-2. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、データ基盤・モデリング・評価・API 設計の全工程の前提条件となる。

---

## 2. 技術スタック

### 2-1. バックエンド

| コンポーネント | 技術 | 用途 |
|---|---|---|
| API サーバー | Flask | REST API 提供・Redis キャッシュ参照 |
| DB | PostgreSQL | Layer 1〜5 データ格納 |
| キャッシュ | Redis | 予測結果・ラップ予測・オッズスナップショットのキャッシュ |
| スキーマ管理 | Alembic | DDL バージョン管理・マイグレーション |
| モデル管理 | MLflow | モデルバージョン管理・アーティファクト保存 |
| ML フレームワーク | LightGBM（初期）、LSTM（Phase 4 以降） | 推論バッチ |
| ストレージ | Google Cloud Storage（GCS） | スクレイプ済み JSON の永続ストレージ |

### 2-2. フロントエンド

| コンポーネント | 技術 |
|---|---|
| フレームワーク | Next.js 15（App Router） |
| チャートライブラリ | Chart.js v4 |
| グラフ | D3.js v7 |
| スタイリング | Tailwind CSS v3 + CSS Variables |
| 国際化 | 日本語専用（i18n 不要） |

### 2-3. MLflow サーバー構成（prod）

| インスタンス | ポート | 用途 |
|---|---|---|
| keiba_lgbm | 5010 | 主要推論モデル |
| tracking_difficulty | 5001 | 位置追跡難易度 |
| final_odds | 5003 | 最終オッズ予測 |
| pace_predictor | 5004 | ペース予測 |

---

## 3. アーキテクチャ設計

### 3-1. 全体構成

```
netkeiba.com / SmartRC / JRA
    │ HTTP スクレイピング（SLA 0〜6 + バックフィル）
    ▼
[HybridStorage: L1 メモリ → L2 ディスク → L3 GCS]
    │ data_paths.py（SSoT）経由
    ▼
[ETL: Parse / Normalize]
    │ → PostgreSQL（Layer 1〜5）
    ▼
[Snapshot Batch]（as_of_race_id 付与）
    │ → horse/jockey/trainer_stats_snapshot
    ▼
[AI Trigger（T-15バンドル完了後）]
    │ → Stage 1 推論 → Stage 2 推論
    ▼
[prediction_results / prediction_lap_times]
    │ → Redis キャッシュ（TTL: 発走時刻まで）
    ▼
[Flask REST API] ← Next.js 15 フロントエンド
```

### 3-2. データ層（5層構造）

| 層 | 名称 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot`（snapshot_at 付き） | 追記のみ・削除不可 |

### 3-3. GCS パス設計（`data_paths.py` SSoT）

```
gs://${GCS_BUCKET}/
└── chuou/data/preprocessed/netkeiba/pc/
    ├── {category}/{year}/{race_id}.json      ← レース単位データ
    └── {category}/{prefix}/{horse_id}.json   ← 馬単位データ（prefix = horse_id[:4]）

gs://${GCS_BUCKET}/chuou/data/others/
└── {category}/{key}.json                     ← jra_cushion 等
```

ローカルのみ（GCS 非使用）:

```
data/page_reference/race_lists/{YYYYMMDD}.json
data/page_reference/race_day_schedule/{YYYYMMDD}.json
data/calculated_data/horse_index/{prefix}/{horse_id}.json
data/calculated_data/horse_names.json
```

### 3-4. 2ステージ推論モデル構成

```
Stage 1: 共有表現マルチタスクモデル（LightGBM）
  入力: Layer 1〜3 特徴量
  出力:
    ├── Head A: 勝率/連対率/複勝率（分類）         → T-1/T-2/T-3
    ├── Head B: ポジション予測（LambdaMART）        → T-8
    └── Head C: オッズ予測（回帰）                  → T-4/T-5

Stage 2: ラップ・ペース予測モデル
  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量
  出力:
    ├── ペースカテゴリ（HIGH/MIDDLE/SLOW）          → T-10
    └── 1F 毎ラップ予測値                           → T-11

ポスト計算:
  win_roi  = win_prob  × predicted_win_odds       × 100  → T-6
  show_roi = show_prob × predicted_place_odds_mid × 100  → T-7
```

### 3-5. キャッシュ設計（4層）

| 層 | 種別 | 対象 | TTL | キャッシュキー例 |
|---|---|---|---|---|
| L1 | Flask 内メモリ（`lru_cache`） | コース・マスターデータ等静的情報 | プロセス再起動まで | N/A |
| L2 | Redis — 予測結果 | T-1〜T-9 全予測 | 発走時刻まで / 発走後 60 秒で自動失効 | `prediction:{race_id}:{model_version}` |
| L3 | Redis — ラップ予測 | T-10〜T-11 | 同上 | `lap:prediction:{race_id}:{model_version}` |
| L4 | Redis — オッズスナップショット | 直近オッズ（推論特徴量用） | 5 分 | `odds:latest:{race_id}` |

追加 Redis キー:

| キー | TTL |
|---|---|
| `race:entries:{race_id}` | 3,600 秒（再スクレイピング完了時に明示削除） |
| `race:results:{race_id}` | 300 秒 |
| `track:speed:{date}:{venue}` | 86,400 秒 |

---

## 4. 機能要件（確定版）

### 4-1. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 | モデル担当 |
|---|---|---|---|---|
| T-1 | `win_prob`（勝率） | 多クラス分類（1着） | `NUMERIC(5,4)` | Stage 1 |
| T-2 | `place_prob`（連対率） | バイナリ分類（2着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-3 | `show_prob`（複勝率） | バイナリ分類（3着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-4 | `predicted_win_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-5 | `predicted_place_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-6 | `win_roi`（単回収率） | ポスト計算値 | `NUMERIC(7,2)` | ポスト処理 |
| T-7 | `show_roi`（複回収率） | ポスト計算値 | `NUMERIC(7,2)` | ポスト処理 |
| T-8 | `predicted_position` | 順位回帰 / ランキング学習 | `SMALLINT` | Stage 1 |
| T-9 | `predicted_running_style` | 4値分類（FRONT/STALKER/MID/CLOSER） | `VARCHAR(10)` | Stage 1 |
| T-10 | `pace_category` | 3値分類（HIGH/MIDDLE/SLOW） | `VARCHAR(10)` | Stage 2 |
| T-11 | `predicted_lap_sec[]` | 時系列回帰（1F 単位系列出力） | `NUMERIC(4,2)[]` | Stage 2 |

> T-6・T-7 は推論結果のポスト計算値。100 以上 = バリューベット候補。

### 4-2. REST API エンドポイント

ベースパス: `/api/v1`、レスポンス形式: `application/json`

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` | `/api/v1/races` | レース一覧取得（`?date=YYYYMMDD`） |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表取得 |
| `GET` | `/api/v1/races/{race_id}/entries` | 出馬表 |
| `GET` | `/api/v1/races/{race_id}/results` | 着順・ラップ・コーナー |
| `GET` | `/api/v1/races/{race_id}/predictions` | AI 予測（T-1〜T-9）取得 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11） |
| `GET` | `/api/v1/races/{race_id}/tracking-difficulty` | 位置追跡難易度 |
| `GET` | `/api/v1/horse/{id}/growth-curve` | 成長曲線 |
| `GET` | `/api/v1/track-speed/day?date=X&venue=Y` | TSI 指数 |
| `GET` | `/api/v1/race-quality/race?id=X` | NNLS 分析 |
| `GET` | `/api/v1/pedigree/race-note?race_id=X` | 血統適性マップデータ |
| `GET` | `/api/v1/bloodline-cluster/lookup?q=X` | 血統クラスター検索 |
| `GET` | `/api/v1/pedigree-race-stats/query` | 種牡馬成績クエリ |
| `POST` | `/api/v1/betting/optimize` | Kelly 最適化 |

**`GET /api/v1/races/{race_id}/predictions` レスポンス例**:

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
      "win_roi": 94.8,
      "show_roi": 101.1,
      "predicted_position": 2,
      "predicted_running_style": "STALKER",
      "is_value_bet": true
    }
  ]
}
```

> `is_value_bet`: `win_roi ≥ 100` または `show_roi ≥ 100` の場合に `true`。

### 4-3. 認証・認可

| 項目 | 方針 |
|---|---|
| 認証方式 | API キー（`Authorization: Bearer <token>`）または セッション Cookie（UI 向け） |
| 認証画面 | パスワードのみ（ユーザー名なし）、「30日間ログイン保持」チェックボックス付き |
| 認可スコープ | 予測 API・レース API は読み取り専用。管理系は内部ネットワーク限定 |
| 管理系エンドポイント | `/api/v1/admin/*` は 127.0.0.1 / VPN 内のみ |

### 4-4. ユーザーアクセス制御

| 機能 | ゲスト | ログイン済 |
|---|---|---|
| レース一覧・基本情報 | ○ | ○ |
| AI 予測（全頭） | 上位 3 頭のみ | ○ |
| 出走馬過去成績（10 走） | ✗ | ○ |
| データ分析全機能 | ○ | ○ |
| マイ分析（条件保存） | ✗ | ○ |
| 馬券最適化 | ✗ | ○ |
| 血統ツール全機能 | ○ | ○ |

### 4-5. データ収集スケジュール（Cron SLA）

システムタイムゾーン: UTC。cron 記述は UTC。

#### 常時・毎日

| SLA | cron（UTC） | JST | タスク名 | 内容 |
|---|---|---|---|---|
| Watchdog | `*/3 * * * *` | 常時 3 分ごと | `server_watchdog` | API + MLflow 死活監視・自動再起動 |
| SLA 0a | `0 22 * * *` | 07:00 | `daily-race-lists` | 全開催日 race_lists 取得・更新 |
| SLA 0b | `0 8 * * *` | 17:00 | `daily-race-lists` | 同上（夕方更新） |
| SLA 1 | `0 9 * * *` | 18:00 | `raceday-eve` | 翌開催日: race_shutuba + race_shutuba_past + race_oikiri + horse_training + smartrc_race → precompute（非開催日は即終了） |
| SLA 2 | `*/10 20-23 * * *` | 05:00-08:50 毎 10 分 | `jra-baba-morning` | jra_cushion ポーリング（開催日のみ実取得） |

#### 開催日リアルタイム

| SLA | cron（UTC） | JST | タスク名 | 内容 |
|---|---|---|---|---|
| SLA 3 | `30 22 * * *` | 07:30 | `raceday-runner` | 各 R T-15 分: race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + smartrc_race + JRA 馬場ライブ → AI 予測トリガ |
| SLA 4 | 同上 | — | `raceday-result-runner` | 各 R T+15 分: race_result_on_time 速報取得 |
| SLA 5 | `30 8 * * *` | 17:30 | `raceday-evening` | race_result + race_result_lap + race_index + race_pair_odds → 馬場速度指数計算 |

#### 週次・バックフィル

| SLA | cron（UTC） | JST | タスク名 | 内容 |
|---|---|---|---|---|
| SLA 6 | `30 8 * * 5` | 17:30 金曜 | `weekly-update` | 先週分 horse_result・指数・偏差値・馬情報更新 |
| — | `0 9 * * 5` | 18:00 金曜 | `horse-name-index` | 馬名リスト + 成長曲線 → calculated_data 更新 |
| バックフィル | `0 15〜0 * * *` | 00:00〜09:00 | 年度別 fast/full | 2020〜2026 年度別（夜間バッチ） |

### 4-6. スクレイピング設定

```
netkeiba.com:
  リクエスト間隔: 2.2〜4.0 秒（ランダム + ガウスジッター）
  バースト制限: 14 req ごとに 6〜12 秒クールダウン
  セッションクールダウン: 60 req ごとに 22〜40 秒
  セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  グローバル最大同時スロット: 4
  UA ローテーション: Chrome/Firefox/Edge × Windows/Mac/Linux 8 種
  429/503 バックオフ: 初期 5s・係数 2.5・最大 3 リトライ
  日次上限: 5,000 req / セッション上限: 500 req

SmartRC:
  リクエスト間隔: 2.0〜5.0 秒
  セッション上限: 200 req / 日次上限: 1,000 req
  クールダウン: 60 秒 / 最大 3 リトライ・係数 2.0
  robots.txt 準拠、ブロック検知時に即停止
```

### 4-7. フロントエンド ページカタログ

```
/                           ← ダッシュボード（ISR revalidate 60s）
/login                      ← ログイン
/races?date=YYYYMMDD        ← レース一覧（ISR revalidate 120s）
/race/{race_id}             ← レース詳細（4タブ: 出馬表/結果/AI予測/出走馬詳細）
/tracking-difficulty        ← 位置追跡難易度分析
/growth-curve               ← 成長曲線（CSR）
/track-speed                ← トラックスピード指数（CSR）
/race-quality               ← レース品質 NNLS 分析
/ai-sla                     ← AI パイプライン SLA ドキュメント
/bloodline                  ← 血統 × 距離/コース研究
/bloodline-cluster          ← 血統クラスター検索
/bloodline-vector           ← 血統ベクトル空間（SSG）
/pedigree-map               ← 血統マップ（D3.js）
/pedigree-race-stats        ← 種牡馬成績クエリ
/note-aptitude-race         ← 血統適性マップ（SVG）
/myostatin                  ← Myostatin 遺伝子ダッシュボード
/betting                    ← 馬券最適化（Kelly 基準、ログイン必須）
```

---

## 5. 非機能要件（確定版）

### 5-1. パフォーマンス SLO

| ID | 指標 | 目標値 |
|---|---|---|
| N-1 | API レスポンス（キャッシュヒット時） | ≤ 200 ms |
| N-2 | API レスポンス（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ラップタイム MAE | ≤ 0.3 秒 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | DB 反映遅延 | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% |
| N-9 | 推論バッチ完了期限 | 発走 3 時間前まで |
| LCP | Largest Contentful Paint | ≤ 2.5s（3G） |
| CLS | Cumulative Layout Shift | ≤ 0.1 |

### 5-2. モデル精度ゲート（CI ブロッキング）

| ID | ターゲット | 指標 | 合格閾値 |
|---|---|---|---|
| N-4 | 勝率（T-1） | Log Loss（ベースライン比） | −5% 以上改善 |
| N-5 | ポジション予測（T-8） | Spearman ρ | ≥ 0.55 |
| N-3 | ラップタイム（T-11） | MAE（秒） | ≤ 0.3 秒 |

### 5-3. テスト・品質

| ID | 指標 | 目標値 |
|---|---|---|
| N-10 | テンポラルリーク検知テスト | CI で必須実行・ブロッキング |
| N-11 | スキーマ変更管理 | Alembic 全変更バージョン管理 |
| N-12 | Redis キャッシュ | 発走時刻まで有効 / 発走後 60 秒で自動失効 |
| N-13 | Unit テストカバレッジ | ≥ 80%（スクレイパー・特徴量パイプライン・モデル・API・バッチ） |
| N-14 | 障害レース除外 | `races.is_excluded = TRUE` フラグで予測対象外を明示管理 |

### 5-4. フロントエンド品質

| 指標 | 目標値 |
|---|---|
| Lighthouse Performance | ≥ 85 |
| Lighthouse Accessibility | ≥ 90 |
| モバイル対応 | 必須（ナビ折りたたみ実装） |

### 5-5. インフラ・メモリバジェット（prod 専用: ConoHa VPS 2GB）

| プロセス | 通常使用量 | 上限（kill 対象） |
|---|---|---|
| Flask API | 80-120 MB | 300 MB |
| PostgreSQL | 100-200 MB | 400 MB |
| Redis | 30-80 MB | 150 MB |
| MLflow サーバー × 4 | 各 80-150 MB | 各 250 MB |
| スクレイパー | 60-100 MB | 200 MB |
| **合計上限目安** | 550-850 MB | **1,800 MB**（残 200MB はカーネル・OS 予備） |

**prod 専用メモリ節約設定**:

```
PostgreSQL: shared_buffers=64MB, work_mem=4MB, max_connections=20
Redis: maxmemory 128mb, maxmemory-policy allkeys-lru
Flask/gunicorn: workers=2, worker_class=sync, timeout=120
```

> dev/stg 環境には上記制限を適用しない。搭載 RAM に応じた独自設定を採用すること。

### 5-6. デプロイ順序

```
1. Alembic DDL マイグレーション実行
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. Flask API サーバー起動
```

### 5-7. ロールバック方針

| 対象 | 方針 |
|---|---|
| モデルロールバック | `prediction_results.model_version` を旧バージョンに切り替え（MLflow 管理） |
| スキーマロールバック | Alembic downgrade を使用 |
| データロールバック | Layer 2〜5 は追記型・不変のため不可（設計原則） |

---

## 6. AI / ML パイプライン

### 6-1. 特徴量定義

#### 基本特徴量（Layer 1〜2 由来）

| 特徴量名 | 説明 | 型 |
|---|---|---|
| `distance` | レース距離（m） | INT |
| `surface` | 芝/ダート/障害 | CATEGORY |
| `direction` | 左/右/直線 | CATEGORY |
| `going` | 馬場状態（良/稍重/重/不良） | CATEGORY |
| `weather` | 天候 | CATEGORY |
| `grade` | レースクラス（G1〜未勝利） | CATEGORY |
| `horse_num` | 出走頭数 | INT |
| `frame_no` | 枠番 | INT |
| `post_no` | 馬番 | INT |
| `weight_carried` | 斤量（kg） | FLOAT |
| `horse_weight` | 馬体重（kg） | INT |
| `horse_weight_diff` | 馬体重増減 | INT |
| `days_since_last` | 前走からの間隔（日） | INT |
| `horse_age` | 馬齢 | INT |
| `sex` | 性別（牡/牝/セン） | CATEGORY |

#### 集計特徴量（Layer 3 スナップショット由来）

`as_of_race_id = 予測対象レース ID` のスナップショットのみ使用すること（テンポラルリーク防止必須）。

| 特徴量名 | 説明 |
|---|---|
| `win_rate_all` / `place_rate_all` / `show_rate_all` | 生涯勝率・連対率・複勝率 |
| `win_rate_distance` / `win_rate_course` / `win_rate_going` | 条件別勝率 |
| `avg_last_3f` / `speed_index_avg` / `speed_index_max` | タイム・スピード指数 |
| `running_style_score` | 脚質スコア（−5=逃〜+5=追込） |
| `jockey.win_rate_all` | 騎手勝率 |
| `trainer.win_rate_all` | 調教師勝率 |

#### クロス・相対特徴量（前処理で自動生成）

```python
df["style_x_straight"] = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"] = df["running_style_score"] * df["distance_category_encoded"]
df["front_runner_count"] = df.groupby("race_id")["running_style_score"] \
                             .transform(lambda x: (x < -2).sum())
df["rel_speed_index"]     = df["speed_index_avg"] / \
                              df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - \
                              df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"]       = df.groupby("race_id")["odds_value"].rank(ascending=True)
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]) \
    .apply(lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 6-2. アルゴリズム選定

| ターゲット | アルゴリズム | 理由 |
|---|---|---|
| 勝率（T-1） | LightGBM softmax | 表形式データ・欠損耐性 |
| 連対率・複勝率（T-2/3） | LightGBM binary | 同上 |
| ポジション予測（T-8） | LambdaMART（LightGBM ranker） | 相対順位直接最適化 |
| オッズ予測（T-4/5） | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ（T-10） | LightGBM multiclass | 3 クラス・解釈性重視 |
| 1F 毎ラップ予測（T-11） | LightGBM per-furlong → LSTM（Phase 4 以降） | MAE ≤ 0.3 秒未達時に移行 |
| 解釈性 | TreeSHAP（`shap.TreeExplainer`） | 全 LightGBM モデルに適用 |

### 6-3. 学習パイプライン

- **データ分割**: 時系列順に train / validation / test を分割。**ランダムシャッフル禁止**。
- **学習データ条件**: 常に過去レースで学習、未来レースで評価。
- **モデル登録**: 学習完了後 MLflow へ登録・バージョニング。SHAP 値もアーティファクトとして保存。

### 6-4. 回収率ポスト計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float,
    win_odds: float,
    show_prob: float,
    place_odds_mid: float,
) -> dict:
    win_roi  = win_prob  * win_odds       * 100  # T-6
    show_roi = show_prob * place_odds_mid * 100  # T-7
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}
```

### 6-5. テンポラルリーク防止チェックリスト

| チェック項目 | 担保方法 |
|---|---|
| 訓練データの時系列分割 | 過去レース学習・未来レース評価のみ（ランダムシャッフル禁止） |
| スナップショット参照 | 推論時も同 race_id のスナップショットのみ使用 |
| オッズ特徴量 | 発走 N 分前の最終スナップショット固定使用 |
| CI テスト | テンポラルリーク検知テストを CI 必須ゲートとして自動実行 |

### 6-6. フェーズ別リリース計画

| Phase | 主要対象 | 完了条件 |
|---|---|---|
| Phase 0 | `scrape_runs` テーブル・基本スキーマ | ラップデータ可用性確認完了（10 レースサンプル） |
| Phase 1 | Layer 1〜5 全テーブル DDL・スクレイパー群・集計バッチ | 過去 2 年分データ格納済み |
| Phase 2 | 特徴量パイプライン・Stage 1 モデル・回収率計算ロジック | 勝率 Log Loss ベースライン比 −5% 改善 |
| Phase 3 | Stage 2 モデル・REST API・Redis キャッシュ・UI | T-1〜T-11 全ターゲットが API 経由で取得可能 |
| Phase 4 | LSTM ラップモデル・自動再学習スケジューラ・モデル品質モニタリング | 継続運用 |

---

## 7. 運用コスト

### 7-1. 確定済みインフラコンポーネント

| コンポーネント | 用途 | 備考 |
|---|---|---|
| ConoHa VPS（2GB RAM） | prod サーバー全般 | 月額費用未定義 |
| PostgreSQL | Layer 1〜5 データ格納 | prod: shared_buffers=64MB |
| Redis | 予測結果キャッシュ | prod: maxmemory=128mb |
| LightGBM（セルフホスト） | Stage 1/2 モデル推論 | 外部 API 委託なし |
| LSTM（Phase 4 以降） | ラップ系列予測 | 移行コスト未評価 |
| Google Cloud Storage | スクレイプ済み JSON 永続化 | 転送費未試算 |
| MLflow（セルフホスト） | モデルバージョン管理 | prod: 4 インスタンス |

### 7-2. 未定義事項（別途 DEC 要作成）

| 項目 | 状態 |
|---|---|
| 月額費用内訳（サーバー・DB・Redis・GCS 転送費等） | **未定義** |
| スケールアップ判断基準（CPU・メモリ・レイテンシ閾値） | **未定義** |
| AI 推論外部化コスト比較（LightGBM セルフホスト vs. SageMaker / Vertex AI 等） | **未定義** |
| コスト削減方針（スポットインスタンス・コールドストレージ移行等） | **未定義** |

---

## 8. 未解決事項・Human 判断待ち

### 8-1. 運用インフラ（AREA-04 §9）

| # | 項目 |
|---|---|
| OP-1 | VPS メモリバジェット詳細（各プロセスへの割り当て上限の正式決定） |
| OP-2 | Circuit Breaker ライブラリ選定（`pybreaker` / `tenacity` 等）・閾値定義（連続失敗 N 回でオープン遷移） |
| OP-3 | 監視基盤ツール選定（Prometheus + Grafana / Sentry / Datadog / 自作スクリプト） |
| OP-4 | アラート通知チャネル（Slack / メール / LINE）と宛先 |
| OP-5 | デプロイ自動化手段（GitHub Actions / Ansible 等） |
| OP-6 | MLflow 以外のモデルレジストリ候補の評価 |

### 8-2. データ管理（AREA-06 §7）

| # | 項目 |
|---|---|
| DM-1 | GCS バケット命名規則（本番・ステージング分離） |
| DM-2 | ディスクキャッシュ容量上限の明示（現状はアクセス頻度によるヒューリスティック削除のみ） |
| DM-3 | GCS 書き込み失敗時のリトライ・アラート設計（HybridStorage 障害挙動） |
| DM-4 | Feature Store の GCS バックアップ設計（DB スナップショット補完） |

### 8-3. 開発環境（AREA-09 §6）

| # | 項目 |
|---|---|
| DEV-1 | dev / stg / prod の環境分離方針（ローカル PC・GPU サーバー・VPS 割り当て） |
| DEV-2 | docker-compose ファイル設計（サービス定義・ネットワーク・ボリューム） |
| DEV-3 | CI/CD パイプライン構成・デプロイフロー |
| DEV-4 | 環境変数管理方法（`.env` / シークレット管理） |
| DEV-5 | GPU 環境要件（CUDA バージョン・GPU メモリ）—LSTM 移行時に必要 |

### 8-4. モニタリング（AREA-10 §6）

| # | 項目 |
|---|---|
| MON-1 | 監視基盤ツール選定（OP-3 と共通） |
| MON-2 | アラート通知チャネルと宛先（OP-4 と共通） |
| MON-3 | PostgreSQL `shared_buffers` 最適値（実負荷計測後に調整） |
| MON-4 | `scrape_runs` テーブルの成長監視・VACUUM スケジュール設計 |

### 8-5. スキーマ確認事項（AREA-01 §3-3）

| # | 項目 |
|---|---|
| SCH-1 | `entries.post_position` が馬番・枠番どちらかの確認（Phase 0-S で要検証）。馬番なら `frame_no` カラム追加が必須。 |

---

## 9. 参照 AREA 一覧

| AREA | タイトル | Status | 最終更新 |
|---|---|---|---|
| AREA-01 | アプリケーション要件 | FINAL | 2026-07-06 |
| AREA-02 | フロントエンド要件 | REVISED | 2026-07-04 |
| AREA-03 | バックエンド要件（Flask API / DB スキーマ / 認証・認可 / 4 層キャッシュ / レート制限） | FINAL | 2026-07-04 |
| AREA-04 | 運用最適化要件（Cron SLA / プロセス分離 / Circuit Breaker / 監視・アラート / デプロイ / ロールバック） | FINAL | 2026-07-04 |
| AREA-05 | コスト計算要件 | FINAL | 2026-07-04 |
| AREA-06 | データ管理要件（GCS パス設計 SSoT / ETL パイプライン / Feature Store / Redis TTL 設計） | FINAL | 2026-07-04 |
| AREA-07 | モデリング管理要件（LightGBM バッチ推論 / 学習パイプライン / SHAP / ModelRegistry / バージョニング / CI ゲート） | FINAL | 2026-07-04 |
| AREA-08 | テスト要件（Unit / Integration / E2E / ML テスト / CI ゲート / カバレッジ目標 / テストデータ管理） | FINAL | 2026-07-04 |
| AREA-09 | 開発環境要件 | FINAL | 2026-07-04 |
| AREA-10 | インフラ・サーバーモニタリング要件 | FINAL | 2026-07-04 |

> **矛盾解決メモ**: AREA-03 の `prediction_results` テーブルで `expected_win_roi` / `expected_show_roi` と記載されているカラム名は、AREA-01（2026-07-06、より新しい）が `win_roi` / `show_roi` に統一していることを優先し、本仕様書では `win_roi` / `show_roi` を正式名称とする。