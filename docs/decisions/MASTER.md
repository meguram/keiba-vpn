# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-04 | 参照: AREA-01〜AREA-09

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com および smartrc.jp から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F 単位ラップ予測** を実現する競馬予測 Web アプリ。加えてユーザーが予想の根拠を自ら探索・検証するための **データ分析機能**（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析・血統分析・成長曲線・馬場速度指数・Myostatin 遺伝子・馬券最適化）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| 一次データソース | netkeiba.com |
| 二次データソース | smartrc.jp（SmartRC 独自指標）、JRA 公式（クッション値・含水率） |
| ユーザー種別 | ゲスト（予測上位3頭閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存・馬券最適化） |
| 主要制約 | ConoHa VPS 2GB（**prod 環境のみ**）— prod での 2GB 以内安定稼働を最優先 |

### 1-2. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること** が、データ基盤・モデリング・評価・API 設計の全工程の前提条件である。

### 1-3. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 | Stage |
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
| T-11 | `predicted_lap_sec[]` | 時系列回帰（1F 単位系列出力） | `NUMERIC(4,2)[]` | Stage 2 |

> T-6・T-7 はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100 以上 = 期待値プラスのバリューベット候補。

---

## 2. 技術スタック

| レイヤー | 技術 | 備考 |
|---|---|---|
| データベース | PostgreSQL | `BIGSERIAL`・`TIMESTAMPTZ`・`NUMERIC`・`ARRAY` 型使用 |
| キャッシュ | Redis | TTL 管理付き、発走後 60 秒で自動失効 |
| スキーママイグレーション | Alembic | 全 DDL 変更をバージョン管理 |
| API フレームワーク | Flask | ベースパス `/api/v1`、レスポンス `application/json` |
| ML フレームワーク（初期） | LightGBM | Stage 1/2 共通、表形式データ向け |
| ML フレームワーク（拡張） | LSTM | Phase 4 以降、ラップ系列予測に適用 |
| モデル管理 | MLflow（または同等） | バージョニング・ロールバック対応 |
| 特徴量重要度 | TreeSHAP（`shap.TreeExplainer`） | 全 LightGBM モデルに適用 |
| テスト | pytest / pytest-cov | カバレッジ目標 ≥ 80% |
| GCS ストレージ | `gs://${GCS_BUCKET}/` | バケット名は環境変数 `GCS_BUCKET` で指定 |
| フロントエンド | Next.js 15（App Router） | Chart.js v4・D3.js v7・Tailwind CSS v3 |
| CI/CD | 未確定 | GitHub Actions 等、後続 DEC で確定 |

---

## 3. アーキテクチャ設計

### 3-1. システムコンポーネント構成

```
scraper        ── netkeiba.com / smartrc.jp スクレイパー（concurrent 制限付き）
gcs            ── スクレイピング生データ・計算済みデータ（プライマリストア）
db (PostgreSQL) ── Layer 1〜5 構造化データ格納
redis          ── 予測結果・ラップ予測・オッズキャッシュ
api (Flask)    ── REST API（/api/v1/...）
ml-worker      ── 特徴量生成・モデル学習・推論バッチ
mlflow         ── モデルバージョン管理
frontend (Next.js) ── UI（14 画面）
```

### 3-2. データ層アーキテクチャ（5層構造）

| 層 | 名称 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot` | 追記のみ・削除不可 |

**特徴量リーク防止の原則**: Layer 3 の集計値は必ず `as_of_race_id`（予測対象レース）に紐付けて保存し、そのレース以後の情報を含めない。

### 3-3. 2ステージ ML アーキテクチャ

```
Stage 1: 共有表現マルチタスクモデル
  入力: Layer 1〜3 特徴量（馬×レース単位）
  [Shared Encoder (LightGBM)]
    ├── Head A: 勝率/連対率/複勝率（分類）       → T-1/T-2/T-3
    ├── Head B: ポジション予測（LambdaMART）     → T-8
    └── Head C: オッズ予測（回帰）               → T-4/T-5
                    │ T-8 ポジション予測値を受け渡し
Stage 2: ラップ・ペース予測モデル
  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量
  出力: ペースカテゴリ（T-10）+ 1F 毎ラップ予測（T-11）
```

### 3-4. GCS バケット構成（実装準拠）

バケット名は環境変数 `GCS_BUCKET` で指定。

```
gs://${GCS_BUCKET}/
├── chuou/data/preprocessed/netkeiba/pc/
│   ├── race_shutuba/{year}/{race_id}.json       # 出馬表
│   ├── race_result/{year}/{race_id}.json        # レース確定結果
│   ├── race_result_on_time/{year}/{race_id}.json # 速報結果
│   ├── race_index/{year}/{race_id}.json         # 速度指数
│   ├── race_odds/{year}/{race_id}.json          # 単複オッズ
│   ├── race_pair_odds/{year}/{race_id}.json     # 連複オッズ
│   ├── race_paddock/{year}/{race_id}.json       # パドック評価
│   ├── race_barometer/{year}/{race_id}.json     # バロメーター偏差値
│   ├── race_oikiri/{year}/{race_id}.json        # 追い切り
│   ├── race_detail/{year}/{race_id}.json        # レース総合詳細
│   ├── race_result_lap/{year}/{race_id}.json    # ラップ詳細
│   ├── race_trainer_comment/{year}/{race_id}.json # 調教師コメント
│   ├── race_performance/{year}/{race_id}.json   # パフォーマンス指数
│   ├── race_shutuba_past/{year}/{race_id}.json  # 馬柱（過去走付き）
│   ├── smartrc_race/{year}/{race_id}.json       # SmartRC指標
│   ├── race_predictions/{year}/{race_id}.json   # 予測キャッシュ
│   ├── tracking_difficulty/{year}/{race_id}.json # 追走難度
│   ├── final_odds_prediction/{year}/{race_id}.json # 最終オッズ予測
│   ├── horse_result/{prefix}/{horse_id}.json    # 馬全成績（prefix=先頭4桁）
│   ├── horse_pedigree_5gen/{prefix}/{horse_id}.json # 5世代血統
│   └── horse_training/{prefix}/{horse_id}.json # 馬調教履歴
└── chuou/data/others/
    ├── jra_cushion/{year}.json                  # クッション値・含水率
    └── horse_name/{key}.json                    # 馬名インデックス
```

**ローカルのみ（GCS 非使用）**:
- `data/page_reference/race_lists/{YYYYMMDD}.json` — 開催日別レース一覧
- `data/page_reference/race_day_schedule/{YYYYMMDD}.json` — 発走時刻スナップショット

**計算済みデータ（`data/calculated_data/`）**:
- `predictions/predictions.json` — 予測ダッシュボード
- `tracking_difficulty/` — 追走難度事前計算
- `growth_curve/` — 成長曲線（週次更新）
- `track_speed/` — 馬場速度指数（年別 Parquet）
- `bloodline_vector/v2_l2/` — 血統ベクトル（UMAP 埋め込み）
- `note_aptitude_race/` — 適性ノートアーティファクト
- `pedigree_map/` — 血統類似度マップ
- `knowledge/` — 馬名インデックス・コースプロファイル

### 3-5. ストレージ階層（アクセス優先順）

| 層 | 種別 | パス | TTL |
|---|---|---|---|
| L1 | プロセスメモリ LRU | プロセス内 dict | 3600 秒、最大 8,000 エントリ |
| L2 | ディスクキャッシュ | `data/cache/{category}/...` | 当年: 12h / 過去年: 2日（週2回未満アクセスは書込しない） |
| L3 | GCS（唯一の source of truth） | `gs://${GCS_BUCKET}/chuou/...` | 永続 |

### 3-6. 4 層 Redis キャッシュ構成

| 層 | 対象 | TTL | キャッシュキー |
|---|---|---|---|
| L1 | Flask 内 `lru_cache`（静的マスター） | プロセス再起動まで | N/A |
| L2 | 予測結果（T-1〜T-9） | 発走時刻まで / 発走後 60 秒失効 | `prediction:{race_id}:{model_version}` |
| L3 | ラップ予測（T-10〜T-11） | 発走時刻まで / 発走後 60 秒失効 | `lap:prediction:{race_id}:{model_version}` |
| L4 | オッズスナップショット | 5 分 | `odds:latest:{race_id}` |

### 3-7. Cron スケジュール・SLA（実装準拠）

システムタイムゾーン UTC。crontime は UTC 記述。

| SLA | cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|---|
| Watchdog | `*/3 * * * *` | 常時 3分ごと | `server_watchdog` | API + MLflow プロセス死活監視・自動再起動 |
| SLA 0 | `0 22 * * *` | 07:00 | `daily-race-lists` | 今日〜カレンダー末尾の全開催日 race_lists 取得・更新 |
| SLA 0 | `0 8 * * *` | 17:00 | `daily-race-lists` | 同上（夕方更新） |
| SLA 1 | `0 9 * * *` | 18:00 | `raceday-eve` | 翌開催日 race_shutuba + 馬柱 + 追い切り + SmartRC → 追走難度 precompute |
| SLA 2 | `*/10 20-23 * * *` | 05:00-08:50 毎10分 | `jra-baba-morning` | JRA 馬場情報・クッション値ポーリング（開催日のみ取得） |
| SLA 3 | `30 22 * * *` | 07:30 | `raceday-runner` | 各R発走 **T-15分** に出馬表+オッズ+SmartRC+馬場ライブ → AI 予測トリガ |
| SLA 4 | `30 22 * * *` | 07:30 | `raceday-result-runner` | 各R発走 **T+15分** に速報結果（`race_result_on_time`）取得 |
| SLA 5 | `30 8 * * *` | 17:30 | `raceday-evening` | 確定結果＋確定オッズ＋ペアオッズ → 馬場速度指数計算トリガ |
| SLA 6 | `30 8 * * 5` | 17:30 金曜 | `weekly-update` | 先週分の結果・指数・偏差値・馬情報更新 |
| JT統計 | `30 20 * * *` | 05:30 | `update_jockey_trainer_stats` | 騎手・調教師統計再生成 |
| 構造監視 | 毎日 JST 06:00 | 06:00 | `structure-scheduler` | ページ構造変更検知（`versions.json` 更新） |

**バックフィル（夜間）**:

| cron（UTC） | JST | 年度 | フェーズ | 最大件数 |
|---|---|---|---|---|
| `0 15 * * *` | 00:00 | 2026 | fast（結果+出馬表） | 7日分 |
| `0 16 * * *` | 01:00 | 2025 | fast | 5日分 |
| `0 17 * * *` | 02:00 | 2024 | fast | 5日分 |
| `0 18 * * *` | 03:00 | 2023 | fast | 5日分 |
| `0 19 * * *` | 04:00 | 2022 | fast | 5日分 |
| `0 21 * * *` | 06:00 | 全年 | horse（馬情報） | 一括 |
| `30 22 * * *` | 07:30 | 2026 | full（補助データ含む） | 5日分 |
| `0 23 * * *` | 08:00 | 2025 | full | 3日分 |
| `0 0 * * *` | 09:00 | 2024 | full | 3日分 |
| `0 17 * * 1,4` | 02:00 月木 | 2021 | fast | 5日分 |
| `0 18 * * 2,5` | 03:00 火金 | 2020 | fast | 5日分 |

### 3-8. プロセス分離

| プロセス種別 | 役割 | 備考 |
|---|---|---|
| スクレイパー | netkeiba.com / smartrc.jp データ収集 | netkeiba: 2.2〜4.0s間隔・バースト制限・UA ローテーション |
| スナップショット集計バッチ | `*_stats_snapshot` 生成 | results 収集完了後に起動 |
| オッズ収集スケジューラ | 発走当日収集（T-15 バンドルで取得） | タイムウィンドウ制御 |
| 推論バッチプロセス | Stage 1 → Stage 2 順次推論・結果書込 | 発走 3 時間前までに完了（N-9） |
| API サーバー（Flask） | REST API 提供・Redis キャッシュ参照 | キャッシュヒット ≤ 200 ms |
| DDL マイグレーション | Alembic スキーマバージョン管理 | デプロイ時に独立実行 |

### 3-9. デプロイ順序

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. API サーバー起動
```

---

## 4. 機能要件（確定版）

### 4-1. 予測機能

| # | 要件 | 優先度 |
|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 |
| F-4 | オッズを発走 T-15 バンドルでスナップショット取得し Layer 5 に格納する | 高 |
| F-5 | スクレイプ実行ログを `scrape_runs` テーブルで管理する | 高 |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F 毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 |
| F-11 | ラップ予測結果を系列形式（`furlong_index` 順）で API から提供する | 中 |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 |
| F-13 | レース一覧・出馬表・AI 予測を統合表示する UI を提供する | 中 |
| F-14 | 回収率 100 以上の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能 | 中 |
| F-17 | ラップデータ可用性の事前検証（サンプル 10 レースで手動確認）— Phase 0 前提条件 | 高 |

### 4-2. データ分析・専門機能

| # | 機能名 | 対象ページ | 優先度 |
|---|---|---|---|
| AN-01 | 種牡馬成績多軸フィルタリング分析 | `/pedigree-race-stats` | 高 |
| AN-02 | コース別・条件別統計ダッシュボード | `/race/{id}` レース品質タブ | 高 |
| AN-03 | 騎手/調教師成績分析 | `/race/{id}` 出走馬詳細タブ | 高 |
| AN-04 | マイ分析（フィルター条件保存・再実行、ログインユーザー限定） | 専用 UI | 中 |
| AN-05 | 位置追跡難易度分析（ease スコア・ペース予想・序盤ラダー） | `/tracking-difficulty` | 高 |
| AN-06 | 成長曲線（馬体重×タイム指数・レース間隔×タイム指数） | `/growth-curve` | 高 |
| AN-07 | トラックスピード指数（TSI: Z スコア正規化馬場速度） | `/track-speed` | 高 |
| AN-08 | 血統クラスター検索（L2 クラスタ分類・適性プロファイル） | `/bloodline-cluster` | 中 |
| AN-09 | 血統ベクトル空間（Canvas 2D マップ、PCA/UMAP/t-SNE） | `/bloodline-vector` | 中 |
| AN-10 | 血統マップ（D3.js サイアー系図フォースグラフ） | `/pedigree-map` | 中 |
| AN-11 | 血統適性マップ（SVG パン/ズームマップ） | `/note-aptitude-race` | 中 |
| AN-12 | Myostatin 遺伝子ダッシュボード（MSTN 型別距離適性） | `/myostatin` | 低 |
| AN-13 | 馬券最適化（Kelly 基準によるポートフォリオ生成、ログイン必須） | `/betting` | 中 |

> **UI 注記要件**: 分析画面に「※ この統計はリアルタイム集計です。AI モデルが予測に使用した時点の特徴量とは異なる場合があります。」を表示すること。

### 4-3. REST API エンドポイント

| メソッド | エンドポイント | 説明 | 機能要件 |
|---|---|---|---|
| `GET` | `/api/v1/races` | レース一覧取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}/entries` | 出走馬一覧 | F-13 |
| `GET` | `/api/v1/races/{race_id}/results` | 着順・ラップ・コーナー | — |
| `GET` | `/api/v1/races/{race_id}/predictions` | 全予測ターゲット（T-1〜T-9） | F-10 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11） | F-11 |
| `GET` | `/api/v1/races/{race_id}/tracking-difficulty` | 位置追跡難易度 | AN-05 |
| `GET` | `/api/v1/races/{race_id}/final-odds` | 最終オッズ予測 | — |
| `GET` | `/api/v1/horse/{horse_id}/growth-curve` | 成長曲線 | AN-06 |
| `GET` | `/api/v1/track-speed/day` | 日別馬場速度指数 | AN-07 |
| `GET` | `/api/v1/pedigree-race-stats/query` | 種牡馬成績クエリ | AN-01 |
| `GET` | `/api/v1/bloodline-cluster/horse-aptitude` | 血統クラスタ適性 | AN-08 |
| `GET` | `/api/v1/pedigree/race-note-3d-v2` | 血統適性マップデータ | AN-11 |
| `POST` | `/api/v1/betting/optimize` | Kelly 馬券最適化 | AN-13 |

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

- `is_value_bet`: `expected_win_roi >= 100` または `expected_show_roi >= 100` の場合に `true`

### 4-4. スクレイピング収集カテゴリ（実装準拠）

#### netkeiba.com レース系

| カテゴリ | 取得タイミング | GCS パス |
|---|---|---|
| `race_shutuba` | レース前日 18:00（SLA 1） | `netkeiba/pc/race_shutuba/{year}/{race_id}.json` |
| `race_shutuba_past` | レース前日 18:00（SLA 1） | `netkeiba/pc/race_shutuba_past/{year}/{race_id}.json` |
| `race_result` | 発走後確定（SLA 5） | `netkeiba/pc/race_result/{year}/{race_id}.json` |
| `race_result_on_time` | 発走 T+15分（SLA 4） | `netkeiba/pc/race_result_on_time/{year}/{race_id}.json` |
| `race_result_lap` | 発走後確定（SLA 5） | `netkeiba/pc/race_result_lap/{year}/{race_id}.json` |
| `race_index` | 発走後確定（SLA 5） | `netkeiba/pc/race_index/{year}/{race_id}.json` |
| `race_odds` | T-15バンドル（SLA 3） | `netkeiba/pc/race_odds/{year}/{race_id}.json` |
| `race_pair_odds` | 発走後確定（SLA 5） | `netkeiba/pc/race_pair_odds/{year}/{race_id}.json` |
| `race_paddock` | T-15バンドル（SLA 3） | `netkeiba/pc/race_paddock/{year}/{race_id}.json` |
| `race_barometer` | T-15バンドル（SLA 3） | `netkeiba/pc/race_barometer/{year}/{race_id}.json` |
| `race_oikiri` | レース前日 18:00（SLA 1） | `netkeiba/pc/race_oikiri/{year}/{race_id}.json` |
| `race_detail` | T-15バンドル（SLA 3） | `netkeiba/pc/race_detail/{year}/{race_id}.json` |
| `race_trainer_comment` | T-15バンドル（SLA 3） | `netkeiba/pc/race_trainer_comment/{year}/{race_id}.json` |

#### netkeiba.com 馬系

| カテゴリ | 取得タイミング | GCS パス |
|---|---|---|
| `horse_result` | 週次（SLA 6）/ バックフィル | `netkeiba/pc/horse_result/{prefix}/{horse_id}.json` |
| `horse_pedigree_5gen` | バックフィル horse フェーズ | `netkeiba/pc/horse_pedigree_5gen/{prefix}/{horse_id}.json` |
| `horse_training` | レース前日 18:00（SLA 1） | `netkeiba/pc/horse_training/{prefix}/{horse_id}.json` |

#### SmartRC + JRA

| カテゴリ | 取得タイミング | GCS パス |
|---|---|---|
| `smartrc_race` | T-15バンドル（SLA 3）+ 前日（SLA 1） | `netkeiba/pc/smartrc_race/{year}/{race_id}.json` |
| `jra_cushion` | 毎朝 JST 05:00-08:50 10分ごと（SLA 2） | `others/jra_cushion/{year}.json` |

#### ローカルのみ（GCS 非使用）

| カテゴリ | パス |
|---|---|
| `race_lists` | `data/page_reference/race_lists/{YYYYMMDD}.json` |
| `race_day_schedule` | `data/page_reference/race_day_schedule/{YYYYMMDD}.json` |

### 4-5. スクレイピング設定（実装値）

```
netkeiba.com:
  - インターバル: 2.2〜4.0 秒（ランダム + ガウスジッター）
  - バースト制限: 14 req ごとに 6〜12 秒クールダウン
  - セッションクールダウン: 60 req ごとに 22〜40 秒
  - セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  - グローバル最大同時スロット: 4
  - UA ローテーション: Chrome/Firefox/Edge × Windows/Mac/Linux 8種
  - 429/503 バックオフ: 初期 5s・係数 2.5・最大 3 リトライ

SmartRC:
  - インターバル: 2.0〜5.0 秒
  - セッション上限: 200 req、日次上限: 1000 req
  - クールダウン: 60 秒
  - リトライ: 最大 3回、係数 2.0
```

### 4-6. フェーズ別ロードマップ

| Phase | 主要対象 | 完了条件 |
|---|---|---|
| Phase 0 | `scrape_runs` テーブル・基本スキーマ | ラップデータ可用性確認完了（F-17） |
| Phase 1 | Layer 1〜5 全テーブル DDL・スクレイパー群・集計バッチ | 過去 2 年分データ格納済み |
| Phase 2 | 特徴量パイプライン・Stage 1 モデル・回収率計算ロジック | 勝率 Log Loss ベースライン比 −5% 改善 |
| Phase 3 | Stage 2 モデル・REST API・Redis キャッシュ・UI（全 14 画面） | 全予測ターゲット T-1〜T-11 が API 経由で取得可能 |
| Phase 4 | LSTM ラップモデル・自動再学習スケジューラ | 継続運用 |

---

## 5. 非機能要件（確定版）

| ID | 区分 | 要件 | 目標値 |
|---|---|---|---|
| N-1 | パフォーマンス | API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | パフォーマンス | API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ML 精度 | ラップタイム予測 MAE | ≤ 0.3 秒 |
| N-4 | ML 精度 | 勝率 Log Loss（ベースライン比改善） | −5% 以上 |
| N-5 | ML 精度 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | 可用性 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | 鮮度 | DB 反映遅延（スクレイピング完了から） | ≤ 10 分 |
| N-8 | 鮮度 | オッズスナップショット欠損率（発走前 T-15 以内） | ≤ 1% |
| N-9 | タイミング | 推論バッチ完了タイミング | 発走 3 時間前まで |
| N-10 | 品質 | テンポラルリーク検知テストを CI に必須実装 | CI 合格必須 |
| N-11 | 管理 | 全スキーマ変更を Alembic でバージョン管理 | マイグレーションファイル必須 |
| N-12 | キャッシュ | 予測結果 Redis TTL（発走後自動失効） | 発走後 60 秒 |
| N-13 | テスト | 単体テストカバレッジ（スクレイパー・特徴量 PL・モデル・API） | ≥ 80% |
| N-14 | データ品質 | 障害レース・海外レースを予測対象外管理 | `is_excluded` フラグで明示 |

### 5-1. 監視・アラート

| 監視項目 | SLO / 閾値 | アラート条件 |
|---|---|---|
| スクレイピング成功率 | ≥ 99% / 月 | 週次で閾値以下 |
| DB 反映遅延 | ≤ 10 分 | 超過時 |
| オッズスナップショット欠損率 | ≤ 1% | 超過時 |
| API レスポンス（キャッシュヒット） | ≤ 200 ms | 超過時 |
| API レスポンス（キャッシュミス） | ≤ 2,000 ms | 超過時 |
| 推論バッチ完了 | 発走 3 時間前 | 未完了時 |
| Watchdog（プロセス死活） | 3 分以内に再起動 | 再起動失敗時 |
| ページ構造変更（structure-monitor） | 毎日 JST 06:00 | 構造変更検知時 |

### 5-2. 認証・認可方針

| 項目 | 方針 |
|---|---|
| 認証方式 | パスワード認証（ユーザー名なし）、「30日間ログイン保持」オプション付き |
| 認可スコープ | 予測 API・レース API は読み取り専用。スクレイプ実行・モデル管理は内部ネットワーク限定 |
| 管理系エンドポイント | `/api/v1/admin/*` は内部 IP（127.0.0.1 / VPN 内）のみ許可 |
| DEV モード | `is_developer()` チェックで馬券最適化・開発者ツールを分岐表示 |

### 5-3. テスト要件

#### CI ゲート（全項目ブロッキング）

| ゲート | ブロッキング条件 |
|---|---|
| テンポラルリーク検知 | `as_of_race_id` リーク検知テスト失敗 |
| Unit テストカバレッジ | < 80% |
| 勝率 Log Loss 改善 | < −5%（ベースライン比） |
| Spearman ρ | < 0.55 |
| ラップ MAE | > 0.3 秒 |
| API レスポンス（キャッシュヒット） | > 200 ms |
| API レスポンス（キャッシュミス） | > 2,000 ms |
| 時系列分割ランダムシャッフル検出 | `train_test_split(shuffle=True)` 使用を検知 |

---

## 6. AI / ML パイプライン

### 6-1. アルゴリズム選定

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率 (T-1) | LightGBM softmax | 表形式データに最強、欠損耐性が高い |
| 連対率・複勝率 (T-2/3) | LightGBM binary | 同上 |
| ポジション予測 (T-8) | LambdaMART（LightGBM ranker） | 相対順位を直接最適化 |
| オッズ予測 (T-4/5) | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ (T-10) | LightGBM multiclass | 3 クラス・解釈性を重視 |
| 1F 毎ラップ予測 (T-11) | LightGBM per-furlong（初期）→ LSTM（Phase 4） | 解釈しやすい単独回帰から開始 |
| 追走難度（ease スコア） | LightGBM（専用モデル） | 独立モデルとして管理 |
| 最終オッズ予測 | LightGBM regression（専用モデル） | T-4/5 とは別途管理 |

### 6-2. 特徴量カテゴリ（実装準拠）

| カテゴリ | 主要特徴量 |
|---|---|
| 馬基本属性 | `sex`, `age`, `bracket_number`, `jockey_weight`, `weight`, `weight_change` |
| レース条件 | `venue`, `surface`, `distance`, `direction`, `weather`, `track_condition`, `field_size` |
| タイム指数 | `speed_max`, `speed_avg`, `speed_recent_1/2/3` |
| 過去5走成績 | `prev{i}_finish`, `prev{i}_last_3f`, `prev{i}_distance`, `prev{i}_time_sec`, `prev{i}_pass_first/last`（i=1〜5） |
| 直近5走統計 | `avg_finish_5`, `win_count_5`, `top3_count_5`, `avg_last_3f_5`, `position_trend`, `last_3f_trend` |
| 調教 | `training_count`, `oikiri_evaluation`, `oikiri_lap_best`, `training_impression_score` |
| 通算戦績 | `career_win_rate`, `career_top3_rate`, `same_surface_runs/win_rate`, `same_dist_win_rate` |
| 血統 | `sire`, `dam_sire` |
| Myostatin遺伝子 | `mstn_cc/ct/tt_prob`, `mstn_speed_index`, `mstn_distance_affinity` |
| NOTE適性 | `note_apt_dist_fit`, `note_apt_l2` |
| パドック | `paddock_rank`, `paddock_score` |
| SmartRC | `cr_value`, `first_furlong_time`, `smartrc_est_pop`, `smartrc_cr`, `smartrc_ten1f` |
| 市場乖離派生 | `avg_prev_pop_rank_diff`, `popularity_trend_slope`, `upset_count` |

**完全除外（オッズリーケージ防止）**: `odds`, `win_odds`, `place_odds`, `popularity`, `popularity_rank` およびすべての `prev{i}_odds` 生値

### 6-3. 回収率ポスト計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float, win_odds: float,
    show_prob: float, place_odds_mid: float
) -> dict:
    win_roi  = win_prob  * win_odds       * 100   # T-6
    show_roi = show_prob * place_odds_mid * 100   # T-7
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}
```

### 6-4. ModelRegistry・バージョニング

- **MLflow**（または同等ツール）でモデル管理（F-16）
- 管理対象モデル: `keiba_lgbm`（主モデル）・`tracking_difficulty`・`final_odds`・`pace_predictor`・`finish_order`（planned）
- `prediction_results` テーブルの `model_version` カラムで推論結果を追跡
- SHAP 値（TreeSHAP）をモデルバージョンに紐付けてアーティファクトとして保存

### 6-5. MLパイプラインと分析バッチの時点整合性分離

| 用途 | `as_of` 制約 | 集計範囲 |
|---|---|---|
| AI 予測モデル特徴量 | **必須**（`race_date < as_of_race_id`） | 予測時点以前のみ |
| ユーザー向け分析 UI | **不要** | 全期間またはUI選択 |

---

## 7. 運用コスト

### 7-1. 使用コンポーネント（確定済み）

| コンポーネント | 用途 | 推論方式 |
|---|---|---|
| PostgreSQL | Layer 1〜5 データ格納 | オンプレミス |
| Redis | 予測結果キャッシュ（TTL 発走後 60 秒失効） | オンプレミス |
| LightGBM | Stage 1/2 モデル推論 | セルフホスト |
| LSTM（Phase 4） | ラップ系列予測 | セルフホスト |
| Flask API | 予測結果配信（≤ 200 ms / ≤ 2,000 ms） | オンプレミス |
| MLflow | モデルバージョン管理 | セルフホスト |
| スクレイピングワーカー | netkeiba.com + SmartRC 定期取得 | オンプレミス |
| GCS | スクレイピング生データ・計算済みデータ | Google Cloud Storage |

> AI 推論は外部 API（OpenAI 等）に委託しない。すべて LightGBM / LSTM によるセルフホスト推論として設計する。

### 7-2. 未確定コスト項目（Human 判断待ち）

| 項目 | 状態 |
|---|---|
| 月額費用内訳（サーバー・DB・Redis・CDN・GCS 等） | **未定義** |
| VPS スペック・各プロセスへのメモリ割り当て上限 | **未定義** |
| スケールアップ判断基準（CPU・メモリ・レイテンシ閾値） | **未定義** |
| GCS バケット名・ストレージクラス・リージョン | **未定義** |

---

## 8. 未解決事項・Human 判断待ち

| ID | カテゴリ | 項目 | 補足 |
|---|---|---|---|
| H-01 | インフラ | VPS メモリバジェット（各プロセスへの割り当て上限） | prod: ConoHa VPS 2GB 制約内での割り当てを確定する必要がある。dev/stg は異なるサーバーのため制約なし（AREA-10 §0 参照） |
| H-02 | インフラ | Circuit Breaker ライブラリ選定・閾値定義 | 候補: `pybreaker`・`tenacity`。連続失敗 N 回でオープン遷移 |
| H-03 | 監視 | 監視基盤ツール選定 | 候補: Prometheus / Grafana / Sentry 等 |
| H-04 | 監視 | アラート通知チャネル | 候補: Slack / PagerDuty 等 |
| H-05 | CI/CD | デプロイ自動化手段 | 候補: GitHub Actions / Ansible 等 |
| H-06 | ML | MLflow 以外のモデルレジストリ候補の評価 | 「MLflow 等」と記載のみ |
| H-07 | コスト | 月額費用内訳・スケールアップ判断基準 | 新規 DEC で確定が必要 |
| H-08 | インフラ | GCS バケット名・ストレージクラス・リージョン | 環境変数 `GCS_BUCKET` で注入するが命名規則を確定する必要がある |
| H-09 | 開発環境 | GPU 環境要件（CUDA バージョン、GPU メモリ）— Phase 4 LSTM 対応前に確定 | — |

---

## 9. 参照 AREA 一覧

| AREA ID | タイトル | 最終更新 | ステータス |
|---|---|---|---|
| AREA-01 | アプリケーション要件（予測ターゲット・スクレイピング・データ要件・機能要件・スキーマ定義・分析機能） | 2026-07-04 | FINAL |
| AREA-02 | フロントエンド要件（Next.js 15 App Router, デザインシステム, ページカタログ 14画面, 血統・成長曲線・馬券最適化） | 2026-07-04 | FINAL |
| AREA-03 | バックエンド要件（Flask API, DB スキーマ, 認証・認可, 4 層キャッシュ設計, レート制限） | 2026-07-03 | FINAL |
| AREA-04 | 運用最適化要件（Cron SLA, プロセス分離, Circuit Breaker, 監視・アラート, デプロイ, ロールバック） | 2026-07-04 | FINAL |
| AREA-05 | コスト計算要件（月額費用・スケールアップ判断・AI 推論外部化比較・コスト削減方針） | 2026-07-03 | FINAL |
| AREA-06 | データ管理要件（GCS パス設計 SSoT, ETL パイプライン, Feature Store, Redis TTL 設計, ストレージ階層） | 2026-07-04 | FINAL |
| AREA-07 | モデリング管理要件（LightGBM バッチ推論, 学習パイプライン, SHAP, ModelRegistry, バージョニング, CI ゲート） | 2026-07-03 | FINAL |
| AREA-08 | テスト要件（Unit/Integration/E2E/ML テスト, CI ゲート, カバレッジ目標, テストデータ管理） | 2026-07-03 | FINAL |
| AREA-09 | 開発環境要件（実行環境前提条件・コンポーネント構成・未定義事項） | 2026-07-03 | FINAL |
