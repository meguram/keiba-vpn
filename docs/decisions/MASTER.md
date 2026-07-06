# keiba-vpn 競馬予測システム — マスター仕様書
> 最終更新: 2026-07-06 | 参照: AREA-01〜AREA-10

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの**勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに**逃げ馬ペース予測・1F単位ラップ予測**を実現する競馬予測 Web アプリ。加えて、ユーザーが予想の根拠を自ら探索・検証するための**データ分析機能**（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com（一次）、SmartRC smartrc.jp（二次）、JRA公式（クッション値） |
| ユーザー種別 | ゲスト（上位3頭閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB RAM — 2GB 以内での安定稼働を最優先 |

### 1-2. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、データ基盤・モデリング・評価・API 設計の全工程の前提条件となる。

### 1-3. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 | 担当 Stage |
|---|---|---|---|---|
| T-1 | `win_prob`（勝率） | 多クラス分類（1着） | `NUMERIC(5,4)` | Stage 1 |
| T-2 | `place_prob`（連対率） | バイナリ分類（2着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-3 | `show_prob`（複勝率） | バイナリ分類（3着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-4 | `predicted_win_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-5 | `predicted_place_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-6 | `win_roi`（単回収率） | ポスト計算値: `win_prob × predicted_win_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-7 | `show_roi`（複回収率） | ポスト計算値: `show_prob × predicted_place_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-8 | `predicted_position` | 順位回帰 / ランキング学習 | `SMALLINT` | Stage 1 |
| T-9 | `predicted_running_style` | 4値分類: `FRONT`/`STALKER`/`MID`/`CLOSER` | `VARCHAR(10)` | Stage 1 |
| T-10 | `pace_category` | 3値分類: `HIGH`/`MIDDLE`/`SLOW` | `VARCHAR(10)` | Stage 2 |
| T-11 | `predicted_lap_sec[]` | 時系列回帰（1F単位系列出力） | `NUMERIC(4,2)[]` | Stage 2 |

> T-6・T-7 は推論後のポスト計算値。100以上 = 期待値プラスのバリューベット候補。

---

## 2. 技術スタック

### 2-1. フロントエンド

| 項目 | 採用技術 |
|---|---|
| フレームワーク | Next.js 15（App Router） |
| チャートライブラリ | Chart.js v4 |
| グラフ | D3.js v7 |
| スタイリング | Tailwind CSS v3 + CSS Variables |
| 言語 | 日本語専用（i18n 不要） |

### 2-2. バックエンド

| 項目 | 採用技術 |
|---|---|
| API フレームワーク | Flask |
| WSGI サーバー | Gunicorn（workers=2、worker_class=sync、timeout=120） |
| DB | PostgreSQL |
| キャッシュ | Redis（maxmemory 128MB、allkeys-lru） |
| スキーママイグレーション | Alembic |
| モデル管理 | MLflow |

### 2-3. ML / AI

| 項目 | 採用技術 |
|---|---|
| Stage 1 モデル | LightGBM（softmax / binary / regression / LambdaMART） |
| Stage 2 モデル | LightGBM per-furlong（初期）→ LSTM（Phase 4 以降） |
| 解釈性 | TreeSHAP（`shap.TreeExplainer`） |

### 2-4. インフラ

| 項目 | 採用技術 |
|---|---|
| VPS | ConoHa VPS 2GB RAM（prod） |
| オブジェクトストレージ | GCS（Google Cloud Storage） |
| プロセス管理 | systemd |
| タスクスケジューラ | cron（UTC 記述） |

---

## 3. アーキテクチャ設計

### 3-1. システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│ keiba-vpn システム構成                                            │
│                                                                   │
│  [Next.js 15 Frontend]                                           │
│       │ /api/v1/* REST                                           │
│  [Flask API + Gunicorn]                                          │
│       │  ├── Redis（4層キャッシュ）                               │
│       │  └── PostgreSQL（Layer 1〜5）                             │
│  [ML Worker]  ── Stage 1 → Stage 2 推論バッチ                   │
│  [MLflow ×4]  ── keiba_lgbm(5010) / tracking_difficulty(5001)  │
│                   final_odds(5003) / pace_predictor(5004)        │
│  [Scraper]    ── netkeiba.com / SmartRC / JRA公式                │
│       └── GCS（JSON 永続ストレージ）                              │
│  [Watchdog]   ── 3分ごと死活監視・自動再起動                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3-2. データ層アーキテクチャ（5層構造）

| 層 | 名称 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot`（snapshot_at 付き） | 追記のみ・削除不可 |

### 3-3. GCS ストレージ構成（`data_paths.py` SSoT）

```
gs://${GCS_BUCKET}/
└── chuou/data/preprocessed/netkeiba/pc/
    ├── {category}/{year}/{race_id}.json      # レース単位データ
    └── {category}/{prefix}/{horse_id}.json   # 馬単位データ（prefix = horse_id[:4]）
chuou/data/others/{category}/{key}.json       # jra_cushion 等
```

ローカルのみ:
```
data/page_reference/race_lists/{YYYYMMDD}.json
data/page_reference/race_day_schedule/{YYYYMMDD}.json
data/calculated_data/horse_index/{prefix}/{horse_id}.json
data/calculated_data/horse_names.json
```

### 3-4. 4層キャッシュ構成（Redis）

| 層 | 対象 | TTL | キャッシュキー例 |
|---|---|---|---|
| L1 | Flask 内メモリ LRU（静的マスター） | プロセス再起動まで | N/A |
| L2 | 予測結果（T-1〜T-9） | 発走時刻まで / 発走後60秒失効 | `prediction:{race_id}:{model_version}` |
| L3 | ラップ予測（T-10〜T-11） | 発走時刻まで / 発走後60秒失効 | `lap:prediction:{race_id}:{model_version}` |
| L4 | 直近オッズ | 5分 | `odds:latest:{race_id}` |

追加キー:
```
race:entries:{race_id}     TTL: 3,600秒
race:results:{race_id}     TTL: 300秒
track:speed:{date}:{venue} TTL: 86,400秒
predictions:{race_id}:full TTL: 発走時刻まで（BFF エンドポイント用）
```

### 3-5. VPS メモリバジェット（prod 専用）

| プロセス | systemd サービス | MemoryMax | CPUQuota |
|---|---|---|---|
| 推論バッチ | `keiba-infer.service` | 512 MB | 60% |
| Web サーバー（Gunicorn） | `keiba-web.service` | 384 MB | — |
| Redis | `redis.conf maxmemory` | 128 MB | — |
| PostgreSQL | — | 400 MB | — |
| MLflow ×4 | — | 各 250 MB | — |
| **全プロセス合計上限** | — | **≤ 1,800 MB** | — |

> 残余 200 MB はカーネル・OS 予備。LightGBM / LSTM ロード時の実測メモリは 512 MB 制約内に収めること（要対応 OP-1）。

### 3-6. 2ステージ推論アーキテクチャ

```
Stage 1（LightGBM 共有表現マルチタスク）
  入力: Layer 1〜3 特徴量
  ├── Head A: 勝率/連対率/複勝率（分類）
  ├── Head B: ポジション予測（LambdaMART）
  └── Head C: オッズ予測（回帰）
        │ predicted_position を受け渡し
Stage 2（LightGBM → LSTM 移行予定）
  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量
  ├── ペースカテゴリ（HIGH/MIDDLE/SLOW）
  └── 1F毎ラップ予測値
```

---

## 4. 機能要件（確定版）

### 4-1. データ収集（スクレイピング）

#### データソース別収集カテゴリ

**netkeiba.com（一次）**

| カテゴリ | 内容 | GCS パス |
|---|---|---|
| `race_shutuba` | 出馬表（枠順・馬番・騎手・馬体重・オッズ） | `netkeiba/pc/race_shutuba/{year}/{race_id}.json` |
| `race_shutuba_past` | 馬柱（過去成績・調教） | `netkeiba/pc/race_shutuba_past/{year}/{race_id}.json` |
| `race_oikiri` | 追い切りデータ | `netkeiba/pc/race_oikiri/{year}/{race_id}.json` |
| `race_result` | 確定結果（着順・タイム・馬体重・コーナー・払戻） | `netkeiba/pc/race_result/{year}/{race_id}.json` |
| `race_result_on_time` | 速報結果（発走後T+15分） | `netkeiba/pc/race_result_on_time/{year}/{race_id}.json` |
| `race_result_lap` | ラップ詳細・ペース・コーナー通過順 | `netkeiba/pc/race_result_lap/{year}/{race_id}.json` |
| `race_index` | 速度指数 | `netkeiba/pc/race_index/{year}/{race_id}.json` |
| `race_odds` | 単複オッズ | `netkeiba/pc/race_odds/{year}/{race_id}.json` |
| `race_pair_odds` | 連複オッズ（馬連・ワイド・馬単） | `netkeiba/pc/race_pair_odds/{year}/{race_id}.json` |
| `race_paddock` | パドック評価 | `netkeiba/pc/race_paddock/{year}/{race_id}.json` |
| `race_barometer` | バロメーター偏差値 | `netkeiba/pc/race_barometer/{year}/{race_id}.json` |
| `race_detail` | レース総合詳細 | `netkeiba/pc/race_detail/{year}/{race_id}.json` |
| `race_trainer_comment` | 調教師コメント | `netkeiba/pc/race_trainer_comment/{year}/{race_id}.json` |
| `race_performance` | パイプライン生成パフォーマンス指数 | `netkeiba/pc/race_performance/{year}/{race_id}.json` |
| `horse_result` | 馬全成績・プロフィール・重賞実績 | `netkeiba/pc/horse_result/{prefix}/{horse_id}.json` |
| `horse_pedigree_5gen` | 5世代血統 | `netkeiba/pc/horse_pedigree_5gen/{prefix}/{horse_id}.json` |
| `horse_training` | 馬調教履歴 | `netkeiba/pc/horse_training/{prefix}/{horse_id}.json` |

**SmartRC（二次）**

| カテゴリ | 内容 | GCS パス |
|---|---|---|
| `smartrc_race` | cr_value・first_furlong_time・estimated_popularity | `netkeiba/pc/smartrc_race/{year}/{race_id}.json` |

**JRA公式**

| カテゴリ | 内容 | GCS パス |
|---|---|---|
| `jra_cushion` | クッション値・含水率 | `others/jra_cushion/{year}.json` |

#### スクレイピング制御設定

```
netkeiba.com:
  リクエスト間隔: 2.2〜4.0 秒（ランダム + ガウスジッター）
  バースト制限: 14 req ごとに 6〜12 秒クールダウン
  セッションクールダウン: 60 req ごとに 22〜40 秒
  セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  グローバル最大同時スロット: 4
  UA ローテーション: Chrome/Firefox/Edge × Windows/Mac/Linux 8種
  429/503 バックオフ: 初期5s・係数2.5・最大3リトライ
  日次上限: 5,000 req / セッション上限: 500 req

SmartRC:
  リクエスト間隔: 2.0〜5.0 秒 / 日次上限: 1,000 req
  robots.txt 準拠、ブロック検知時に即停止
```

### 4-2. Cron スケジュール（全 SLA 確定版）

システムタイムゾーン: UTC。実行スクリプトは `TZ=Asia/Tokyo` 付与。

#### 常時監視

| cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|
| `*/3 * * * *` | 常時3分ごと | `server_watchdog` | API + MLflow 死活監視・自動再起動（`@reboot` にも登録） |
| 毎日 JST 06:00 | 06:00 | `structure-scheduler` | ページ構造変更検知・`versions.json` 更新 |
| `30 19 * * *` | 04:30 | `rotate_logs` | ログローテーション |
| 86,400秒ごと | — | `disk_cache_cleanup` | `data/cache/` 古ファイル削除 |
| 3,600秒ごと | — | `queue_hourly_maintain` | 失敗ジョブ待機復帰・完了レコード削除 |

#### 定期取得（毎日）

| cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|
| `0 22 * * *` | 07:00 | `daily-race-lists` | 今日〜末尾の全開催日 race_lists 取得・更新 |
| `0 8 * * *` | 17:00 | `daily-race-lists` | 同上（夕方更新） |
| `30 20 * * *` | 05:30 | `update_jt_stats` | 騎手・調教師統計再生成 |
| `0 9 * * *` | 18:00 | `raceday-eve` | 翌日開催時のみ: shutuba + shutuba_past + oikiri + horse_training + smartrc_race → 追走難度・最終オッズ precompute |
| `*/10 20-23 * * *` | 05:00-08:50 毎10分 | `jra-baba-morning` | jra_cushion ポーリング（開催日のみ実取得） |

#### 開催日リアルタイム

| cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|
| `30 22 * * *` | 07:30 | `raceday-runner` | 各R T-15分: race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + smartrc_race + JRA馬場ライブ → AI 予測トリガ |
| `30 22 * * *` | 07:30 | `raceday-result-runner` | 各R T+15分: race_result_on_time 速報取得 |
| `30 8 * * *` | 17:30 | `raceday-evening` | race_result + race_pair_odds + race_index → 馬場速度指数計算トリガ |

#### 週次

| cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|
| `30 8 * * 5` | 17:30 金曜 | `weekly-update` | race_result・race_pair_odds・race_index・smartrc・horse_result 一括更新 |
| `0 9 * * 5` | 18:00 金曜 | `horse-name-index` | 馬名インデックス + 成長曲線 calculated_data 一括更新 |

#### バックフィル（夜間）

| cron（UTC） | JST | 対象年 | フェーズ | 最大件数 |
|---|---|---|---|---|
| `0 15 * * *` | 00:00 | 2026 | fast（race_result + race_shutuba） | 7日分 |
| `0 16 * * *` | 01:00 | 2025 | fast | 5日分 |
| `0 17 * * *` | 02:00 | 2024 | fast | 5日分 |
| `0 18 * * *` | 03:00 | 2023 | fast | 5日分 |
| `0 19 * * *` | 04:00 | 2022 | fast | 5日分 |
| `0 21 * * *` | 06:00 | 全年 | horse（horse_result 一括） | 一括 |
| `30 22 * * *` | 07:30 | 2026 | full（race_index・race_odds・race_barometer・race_oikiri 等） | 5日分 |
| `0 23 * * *` | 08:00 | 2025 | full | 3日分 |
| `0 0 * * *` | 09:00 | 2024 | full | 3日分 |
| `0 17 * * 1,4` | 02:00 月木 | 2021 | fast（週2回） | 5日分 |
| `0 18 * * 2,5` | 03:00 火金 | 2020 | fast（週2回） | 5日分 |

### 4-3. REST API エンドポイント

ベースパス: `/api/v1`、レスポンス形式: `application/json`、タイムゾーン: UTC（表示用 JST）

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` | `/api/v1/races/{race_id}/predictions` | 全予測ターゲット（T-1〜T-9）取得 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11）取得 |
| `GET` | `/api/v1/races/{race_id}/full` | BFF 集約エンドポイント（p95 ≤ 300ms） |
| `GET` | `/api/v1/races/{race_id}/entries` | 出馬表取得 |
| `GET` | `/api/v1/races/{race_id}/results` | 着順・ラップ・コーナー取得 |
| `GET` | `/api/v1/races` | レース一覧取得（`?date=YYYYMMDD`） |
| `GET` | `/api/v1/races/{race_id}/tracking-difficulty` | 位置追跡難易度 |
| `GET` | `/api/v1/horse/{id}/growth-curve` | 成長曲線 |
| `GET` | `/api/v1/track-speed/day` | TSI 指数（`?date=X&venue=Y`） |
| `GET` | `/api/v1/race-quality/race` | NNLS 分析（`?id=X`） |
| `GET` | `/api/v1/pedigree/race-note` | 適性マップデータ（`?race_id=X`） |
| `GET` | `/api/v1/bloodline-cluster/lookup` | クラスター検索（`?q=X`） |
| `GET` | `/api/v1/pedigree-race-stats/query` | 種牡馬成績クエリ |
| `POST` | `/api/v1/betting/optimize` | Kelly 最適化 |

#### `/api/v1/races/{race_id}/predictions` レスポンス仕様

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

> `is_value_bet`: `win_roi >= 100` または `show_roi >= 100` の場合に `true`。
> ※ AREA-01（2026-07-06 更新）と AREA-03（2026-07-04 更新）でフィールド名に差異（`expected_win_roi` vs `win_roi`）があるため、最新更新日 AREA-01 の `win_roi` / `show_roi` を採用。

### 4-4. 認証・認可

| 項目 | 方針 |
|---|---|
| 認証方式 | パスワードのみ（ユーザー名なし）、30日間ログイン保持 |
| API 認証 | `Authorization: Bearer <token>` ヘッダーまたはセッション Cookie |
| 管理系エンドポイント（`/api/v1/admin/*`） | 内部 IP（127.0.0.1 / VPN 内）のみ許可 |
| DDL 操作 | Alembic マイグレーション権限はサービスアカウントに限定 |
| Row Level Security | `saved_analyses` テーブルに RLS 適用（`user_id = current_setting('app.current_user_id')::UUID`） |

### 4-5. フロントエンド機能

#### ユーザー種別アクセス制御

| 機能 | ゲスト | ログイン済 |
|---|---|---|
| レース一覧・基本情報 | ○ | ○ |
| AI予測（全頭） | 上位3頭のみ | ○ |
| 出走馬過去成績（10走） | ✗ | ○ |
| データ分析全機能 | ○ | ○ |
| マイ分析（条件保存） | ✗ | ○ |
| 馬券最適化 | ✗ | ○ |
| 血統ツール全機能 | ○ | ○ |

#### ページ構成

```
/                           ← ダッシュボード（ISR、revalidate 60s）
/login                      ← ログイン
/races?date=YYYYMMDD        ← レース一覧（ISR、revalidate 120s）
/race/{race_id}             ← レース詳細（4タブ: 出馬表/結果/AI予測/出走馬詳細）
/tracking-difficulty        ← 位置追跡難易度分析
/growth-curve               ← 成長曲線
/track-speed                ← トラックスピード指数（TSI）
/race-quality               ← レース品質 NNLS 分析
/ai-sla                     ← AI パイプライン SLA
/bloodline                  ← 血統 × 距離/コース研究
/bloodline-cluster          ← 血統クラスター検索
/bloodline-vector           ← 血統ベクトル空間（Canvas）
/pedigree-map               ← 血統マップ（D3.js）
/pedigree-race-stats        ← 種牡馬成績クエリ
/note-aptitude-race         ← 血統適性マップ（SVG）
/myostatin                  ← Myostatin 遺伝子ダッシュボード
/betting                    ← 馬券最適化（Kelly 基準）
```

---

## 5. 非機能要件（確定版）

### 5-1. パフォーマンス SLO

| SLO ID | 指標 | 目標値 |
|---|---|---|
| N-1 | API レスポンス（キャッシュヒット時） | ≤ 200 ms |
| N-2 | API レスポンス（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ラップタイム予測 MAE | ≤ 0.3 秒 |
| N-4 | 勝率モデル Log Loss 改善（ベースライン比） | ≥ −5% |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | DB 反映遅延（スクレイピング完了から） | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前5分以内） | ≤ 1% |
| N-9 | 推論バッチ完了タイミング | 発走3時間前まで |
| N-13 | Unit テストカバレッジ | ≥ 80%（スクレイパー・特徴量パイプライン・モデル・API） |
| N-14 | 障害・海外レース除外 | `races.is_excluded = TRUE` で予測対象外管理 |
| — | BFF エンドポイント p95 | ≤ 300 ms |
| — | Redis キャッシュヒット率（`predictions:{race_id}:full`） | ≥ 80% |
| — | LCP（Largest Contentful Paint） | ≤ 2.5s（3G） |
| — | CLS（Cumulative Layout Shift） | ≤ 0.1 |

### 5-2. テスト要件

#### テンポラルリーク防止（最重要 CI ゲート）

- CI パイプラインでテンポラルリーク検知テストを自動実行・ブロッキング必須
- `as_of_race_id` 紐付けの単体テストを必須化
- 時系列分割: `train_test_split(shuffle=True)` の使用禁止（静的チェック）

#### CI ゲート（ブロッキング）

| ゲート | 条件 |
|---|---|
| テンポラルリーク検知失敗 | ブロック |
| Unit テストカバレッジ < 80% | ブロック |
| 勝率 Log Loss 改善 < −5% | ブロック |
| Spearman ρ < 0.55 | ブロック |
| ラップ MAE > 0.3 秒 | ブロック |
| API レスポンス（キャッシュヒット）> 200 ms | ブロック |
| API レスポンス（キャッシュミス）> 2,000 ms | ブロック |

### 5-3. 信頼性・障害対応

#### リトライ・バックオフ

```
結果スクレイパー:
  トリガー: 発走予定時刻 + 35分
  リトライ: 5分間隔 × 最大6回（合計最大30分）
```

#### スキーママイグレーション

- Alembic でバージョン管理。デプロイ時は API サーバー起動前に独立実行。

#### デプロイ順序

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. 推論バッチプロセス起動
4. Flask API サーバー起動（Gunicorn）
5. Next.js フロントエンドビルド・デプロイ
```

---

## 6. AI / ML パイプライン

### 6-1. 特徴量定義

#### 基本特徴量（Layer 1〜2 由来）

| 特徴量名 | 説明 | 型 |
|---|---|---|
| `distance` | レース距離 (m) | INT |
| `surface` | 芝/ダート/障害 | CATEGORY |
| `direction` | 左/右/直線 | CATEGORY |
| `going` | 馬場状態（良/稍重/重/不良） | CATEGORY |
| `weather` | 天候 | CATEGORY |
| `grade` | レースクラス（G1〜未勝利） | CATEGORY |
| `horse_num` | 出走頭数 | INT |
| `frame_no` | 枠番 | INT |
| `post_no` | 馬番 | INT |
| `weight_carried` | 斤量 (kg) | FLOAT |
| `horse_weight` | 馬体重 (kg) | INT |
| `horse_weight_diff` | 馬体重増減 | INT |
| `days_since_last` | 前走からの間隔（日） | INT |
| `horse_age` | 馬齢 | INT |
| `sex` | 性別（牡/牝/セン） | CATEGORY |

#### 集計特徴量（Layer 3 スナップショット由来）

| 特徴量名 | 説明 |
|---|---|
| `win_rate_all` / `place_rate_all` / `show_rate_all` | 生涯勝率・連対率・複勝率 |
| `win_rate_distance` / `win_rate_course` / `win_rate_going` | 条件別勝率 |
| `avg_last_3f` / `speed_index_avg` / `speed_index_max` | タイム・スピード指数 |
| `running_style_score` | 脚質スコア（−5=逃 〜 +5=追込） |
| `jockey.win_rate_all` | 騎手勝率 |
| `trainer.win_rate_all` | 調教師勝率 |

**必須制約**: Layer 3 集計値は必ず `as_of_race_id = 予測対象レース ID` のスナップショットを参照。

#### クロス・相対特徴量（前処理自動生成）

```python
# 脚質 × コース形状クロス
df["style_x_straight"] = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"] = df["running_style_score"] * df["distance_category_encoded"]

# 逃げ・先行馬数（ペース予測用）
df["front_runner_count"] = df.groupby("race_id")["running_style_score"] \
                             .transform(lambda x: (x < -2).sum())

# 同レース内相対化
df["rel_speed_index"]     = df["speed_index_avg"] / \
                              df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - \
                              df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"]       = df.groupby("race_id")["odds_value"].rank(ascending=True)

# ペース事前シナリオ
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]) \
    .apply(lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 6-2. 学習パイプライン

- **時系列分割**: 常に過去レースで学習・未来レースで評価。ランダムシャッフル禁止。
- **Stage 1**: 特徴量エンジニアリング → 時系列分割 → LightGBM binary（連対・複勝）→ softmax（勝率）→ LambdaMART（ポジション）→ regression（オッズ）→ ModelRegistry 登録
- **Stage 2**: Stage 1 ポジション予測値 + Layer 4 結合 → LightGBM multiclass（ペース）→ per-furlong（ラップ）→ ModelRegistry 登録
- **推論バッチ**: 発走3時間前までに完了。オッズ特徴量は「発走N分前の最終スナップショット」を固定使用。

### 6-3. 回収率ポスト計算

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

### 6-4. ModelRegistry・バージョニング

- MLflow（または同等ツール）でモデル管理（F-16）
- SHAP 値（TreeSHAP）をモデルバージョンに紐付けてアーティファクト保存
- Phase 4 以降: 特徴量重要度定期モニタリング・データドリフト検知

### 6-5. 評価指標

| ターゲット | 主指標 | 補助指標 | CI 合格閾値 |
|---|---|---|---|
| 勝率（T-1） | Log Loss | Calibration Error, Top-1 Accuracy | ベースライン比 −5% 以上改善 |
| 連対率・複勝率（T-2/3） | Binary Log Loss | AUC-ROC, Calibration | — |
| ポジション（T-8） | Spearman ρ | MAE | ≥ 0.55 |
| オッズ予測（T-4/5） | MAE（オッズ単位） | RMSE | — |
| ラップタイム（T-11） | MAE（秒） | RMSE per furlong | ≤ 0.3 秒 |
| ペースカテゴリ（T-10） | Accuracy | Macro F1 | — |
| 回収率バックテスト | 通算 ROI | Sharpe Ratio | ROI プラス |

---

## 7. 運用コスト

### 7-1. 確定済みコンポーネント（コスト試算の前提）

| コンポーネント | 用途 | 備考 |
|---|---|---|
| ConoHa VPS 2GB | メインサーバー（prod） | 月額固定費（金額は要確認） |
| GCS | JSON 永続ストレージ | race_odds_snapshot が最大成長源（月 100-200 MB 見込み） |
| PostgreSQL（オンプレ） | Layer 1〜5 データ格納 | VPS 内稼働 |
| Redis（オンプレ） | 予測結果キャッシュ | VPS 内稼働、maxmemory 128MB |
| LightGBM（セルフホスト） | Stage 1/2 推論 | 外部 API 委託なし |
| MLflow ×4（オンプレ） | モデルバージョン管理 | VPS 内稼働（各 port 5001/5003/5004/5010） |

### 7-2. 未定義事項（別途 DEC で確定が必要）

| 項目 | 状態 |
|---|---|
| 月額費用内訳（サーバー・GCS 転送・その他） | **未定義** |
| スケールアップ判断基準（CPU・メモリ・レイテンシ閾値） | **未定義** |
| AI 推論外部化コスト比較（LightGBM セルフホスト vs 外部 API） | **未定義** |
| コスト削減方針（スポットインスタンス・コールドストレージ移行等） | **未定義** |

---

## 8. 未解決事項・Human 判断待ち

| # | 項目 | 優先度 | 関連 AREA |
|---|---|---|---|
| **OP-1** | 各 MLflow モデルロード時の実測メモリ使用量計測・512MB 制約との適合確認 | 高 | AREA-04, AREA-10 |
| **OP-3** | 監視基盤ツール選定（Prometheus + Grafana / Sentry / Datadog / 自作スクリプト） | 中 | AREA-04, AREA-10 |
| **OP-4** | アラート通知チャネル（Slack / メール / LINE）と宛先の確定 | 中 | AREA-04, AREA-10 |
| **DM-1** | GCS バケット命名規則（本番・ステージング分離） | 中 | AREA-06 |
| **DM-2** | ディスクキャッシュ容量上限の明示 | 低 | AREA-06 |
| **DM-3** | GCS 書き込み失敗時のリトライ・アラート設計 | 中 | AREA-06 |
| **DM-4** | Feature Store の GCS バックアップ設計 | 低 | AREA-06 |
| **MON-3** | PostgreSQL `shared_buffers` 最適値（実負荷計測後） | 中 | AREA-10 |
| **MON-4** | `scrape_runs` テーブルの成長監視・VACUUM スケジュール設計 | 低 | AREA-10 |
| **CB-1** | Circuit Breaker ライブラリ選定（`pybreaker` / `tenacity`）・閾値定義・適用対象確定 | 中 | AREA-04 |
| **ENV-1** | dev / stg / prod 環境分離方針・docker-compose 設計 | 高 | AREA-09 |
| **ENV-2** | CI/CD パイプライン・デプロイフロー確定 | 高 | AREA-09 |
| **ENV-3** | 環境変数管理方法（`.env` / シークレット管理） | 高 | AREA-09 |
| **COST-1** | 月額費用内訳・スケールアップ判断基準・コスト削減方針を定める新規 DEC 作成 | 中 | AREA-05 |
| **SCHEMA-1** | `entries.post_position` が馬番か枠番かを確認し、必要に応じて `frame_no` カラム追加（Phase 0-S） | 高 | AREA-01 |

---

## 9. 参照 AREA 一覧

| AREA | タイトル | Status | 最終更新 |
|---|---|---|---|
| AREA-01 | アプリケーション要件 | FINAL | 2026-07-06 |
| AREA-02 | フロントエンド要件 | REVISED | 2026-07-04 |
| AREA-03 | バックエンド要件 | FINAL | 2026-07-04 |
| AREA-04 | 運用最適化要件 | FINAL | 2026-07-06 |
| AREA-05 | コスト計算要件 | FINAL | 2026-07-04 |
| AREA-06 | データ管理要件 | FINAL | 2026-07-04 |
| AREA-07 | モデリング管理要件 | FINAL | 2026-07-04 |
| AREA-08 | テスト要件 | FINAL | 2026-07-04 |
| AREA-09 | 開発環境要件 | FINAL | 2026-07-04 |
| AREA-10 | インフラ・サーバーモニタリング要件 | FINAL | 2026-07-04 |