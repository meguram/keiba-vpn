# keiba-vpn 競馬予測システム — マスター仕様書
> 最終更新: 2026-07-06 | 参照: AREA-01〜AREA-10

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測、ならびに逃げ馬ペース予測・1F 単位ラップ予測を実現する競馬予測 Web アプリ。加えて、ユーザーが予想の根拠を自ら探索・検証するためのデータ分析機能（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com（一次）、SmartRC / smartrc.jp（二次）、JRA 公式 |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB — prod 環境 2GB 以内での安定稼働を最優先 |

### 1-2. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、データ基盤・モデリング・評価・API 設計の全工程の前提条件である。

---

## 2. 技術スタック

### 2-1. バックエンド

| カテゴリ | 技術 | バージョン / 備考 |
|---|---|---|
| API フレームワーク | Flask + Gunicorn | prod: workers=2, worker_class=sync, timeout=120 |
| データベース | PostgreSQL | shared_buffers=64MB（prod）、max_connections=20（prod） |
| キャッシュ | Redis | maxmemory=128mb、maxmemory-policy=allkeys-lru（prod） |
| スキーマ管理 | Alembic | 全 DDL 変更をマイグレーションファイルで管理 |
| ML フレームワーク | LightGBM（Stage 1/2）、LSTM（Stage 3 / Phase 4） | |
| モデル管理 | MLflow | keiba_lgbm(5010) / tracking_difficulty(5001) / final_odds(5003) / pace_predictor(5004) |
| ストレージ | GCS（永続 JSON）+ ローカルディスクキャッシュ | `GCS_BUCKET` 環境変数で注入（本番値: `magu-keiba-horse-racing-ai`） |

### 2-2. フロントエンド

| カテゴリ | 技術 | バージョン |
|---|---|---|
| フレームワーク | Next.js（App Router） | v15 |
| チャートライブラリ | Chart.js | v4 |
| グラフ（系統マップ） | D3.js | v7 |
| スタイリング | Tailwind CSS + CSS Variables | v3 |
| 国際化 | 日本語専用（i18n 不要） | — |

### 2-3. インフラ

| カテゴリ | 技術 |
|---|---|
| 本番サーバー | ConoHa VPS 2GB |
| プロセス管理 | systemd（MemoryMax / CPUQuota 制限付き） |
| スクレイピング | Python + requests（UA ローテーション・ガウスジッター付き） |
| CI/CD | 未確定（AREA-09 未決定事項） |
| 監視基盤 | 未確定（OP-3: Prometheus+Grafana / Sentry 等から選定） |

---

## 3. アーキテクチャ設計

### 3-1. データ層（5層構造）

| 層 | 役割 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot`, `prediction_results`, `course_stats_cache`, `saved_analyses` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary`, `prediction_lap_times` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot` | 追記のみ・削除不可 |

### 3-2. GCS ストレージ構造

パス定義の SSoT は `data_paths.py`。バケット名は環境変数 `GCS_BUCKET` で注入（ハードコード禁止）。  
**本番バケット**: `magu-keiba-horse-racing-ai` / **ルートプレフィックス**: `chuou/`

```
gs://magu-keiba-horse-racing-ai/          ← GCS_BUCKET=magu-keiba-horse-racing-ai
└── chuou/data/preprocessed/netkeiba/pc/
    ├── {category}/{year}/{race_id}.json       ← レース単位データ
    └── {category}/{prefix}/{horse_id}.json    ← 馬単位データ（prefix = horse_id[:4]）
chuou/data/others/{category}/{key}.json        ← jra_cushion 等
```

ローカルのみ（GCS 非使用）:

```
data/page_reference/race_lists/{YYYYMMDD}.json
data/page_reference/race_day_schedule/{YYYYMMDD}.json
data/calculated_data/horse_index/{prefix}/{horse_id}.json
data/calculated_data/horse_names.json
```

### 3-3. ストレージ階層（HybridStorage）

| 層 | 種別 | TTL | 用途 |
|---|---|---|---|
| L1 | メモリ LRU キャッシュ | 3,600 秒 | 高頻度アクセスデータ |
| L2 | ディスクキャッシュ（`data/cache/`） | レース系 12時間 / 馬系 2日 | GCS フォールバック前段 |
| L3 | GCS（`gs://magu-keiba-horse-racing-ai/`） | 永続 | 単一永続 JSON ストア |

読取フロー: L1 ヒット → 即返却 → L1 ミス → L2 → L3 フォールバック → L2/L1 ウォームアップ

### 3-4. AI 推論パイプライン（3ステージ）

```
Stage 1: LightGBM（T-1勝率 / T-2連対率 / T-3複勝率 / T-6脚質分類）
       ↓ T-6 出力を特徴量として受け渡し
Stage 2: LightGBM（T-8ペースカテゴリ → T-4上り3F / T-5位置取り → T-9走破タイム）
       ↓（Phase 4 フラグ ON のみ）
Stage 3: LSTM / Transformer（T-7 ラップ系列、現時点フリーズ）
```

### 3-5. VPS メモリバジェット（prod 専用）

| プロセス | systemd サービス | MemoryMax | CPUQuota |
|---|---|---|---|
| 推論バッチ | `keiba-infer.service` | 512MB | 60% |
| Web サーバー（Gunicorn） | `keiba-web.service` | 384MB | — |
| Redis | `redis.conf` maxmemory | 128MB | — |
| 全プロセス合計 | — | ≤ 1,500MB | — |

> 残余 ~500MB は PostgreSQL・OS・その他プロセス向け。

---

## 4. 機能要件（確定版）

### 4-1. 予測ターゲット定義

| ID | ターゲット名 | 問題設定 | 出力型 | Stage |
|---|---|---|---|---|
| T-1 | 勝率（`win_prob`） | 多クラス分類（1着） | `NUMERIC(5,4)` | Stage 1 |
| T-2 | 連対率（`place_prob`） | バイナリ分類（2着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-3 | 複勝率（`show_prob`） | バイナリ分類（3着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-4 | 上り3Fタイム予測 | 回帰（秒） | `FLOAT` | Stage 2 |
| T-5 | 数値位置取り予測 | 順序回帰（1〜頭数） | `INT` | Stage 2 |
| T-6 | 脚質分類予測（`FRONT`/`STALKER`/`MID`/`CLOSER`） | 4クラス分類 | `ENUM` | Stage 1 |
| T-7 | ラップ系列予測（`predicted_lap_sec[]`） | 1F単位回帰（系列出力） | `FLOAT[]` | Stage 3（Phase 4） |
| T-8 | ペースカテゴリ予測（`HIGH`/`MIDDLE`/`SLOW`） | 3クラス分類 | `ENUM` | Stage 2 |
| T-9 | 想定走破タイム予測（派生） | T-4 + L3F回帰モデルの合成 | `FLOAT` | Stage 2（派生） |
| T-10（旧） | `win_roi` | ポスト計算: `win_prob × predicted_win_odds × 100` | `NUMERIC(7,2)` | — |
| T-11（旧） | `show_roi` | ポスト計算: `show_prob × predicted_place_odds × 100` | `NUMERIC(7,2)` | — |

> `win_roi` / `show_roi` はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100 以上 = バリューベット候補。

> **矛盾解決**: AREA-01 の T-4〜T-11 定義（旧: predicted_win_odds 等）と AREA-07 の T-4〜T-9 定義（新: 上り3F・位置取り等）が競合。AREA-07 の最終更新日（2026-07-06）が最新のため AREA-07 の T-4〜T-9 を採用。旧 `predicted_win_odds`・`predicted_place_odds` はレスポンスフィールドとして残存するが予測ターゲット番号からは除外。

### 4-2. データソース

#### netkeiba.com（一次）

| スクレイプカテゴリ | 内容 | 取得タイミング（SLA） | GCS 格納パス |
|---|---|---|---|
| `race_shutuba` | 出馬表（枠順・馬番・騎手・馬体重・オッズ） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_shutuba/{year}/{race_id}.json` |
| `race_shutuba_past` | 馬柱（各馬の過去成績） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_shutuba_past/{year}/{race_id}.json` |
| `race_oikiri` | 追い切りデータ | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_oikiri/{year}/{race_id}.json` |
| `race_result` | 確定結果 | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_result/{year}/{race_id}.json` |
| `race_result_on_time` | 速報結果（T+15分） | T+15分（SLA 4） | `netkeiba/pc/race_result_on_time/{year}/{race_id}.json` |
| `race_result_lap` | ラップ詳細・ペース | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_result_lap/{year}/{race_id}.json` |
| `race_index` | 速度指数 | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_index/{year}/{race_id}.json` |
| `race_odds` | 単複オッズ | T-15バンドル（SLA 3） | `netkeiba/pc/race_odds/{year}/{race_id}.json` |
| `race_pair_odds` | 連複オッズ（馬連・ワイド・馬単） | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_pair_odds/{year}/{race_id}.json` |
| `race_paddock` | パドック評価 | T-15バンドル（SLA 3） | `netkeiba/pc/race_paddock/{year}/{race_id}.json` |
| `race_barometer` | バロメーター偏差値 | T-15バンドル（SLA 3） | `netkeiba/pc/race_barometer/{year}/{race_id}.json` |
| `race_detail` | レース総合詳細 | T-15バンドル（SLA 3） | `netkeiba/pc/race_detail/{year}/{race_id}.json` |
| `race_trainer_comment` | 調教師コメント | T-15バンドル（SLA 3） | `netkeiba/pc/race_trainer_comment/{year}/{race_id}.json` |
| `race_performance` | パイプライン生成パフォーマンス指数 | 計算後 | `netkeiba/pc/race_performance/{year}/{race_id}.json` |
| `horse_result` | 馬の全成績・プロフィール | 週次（SLA 6）/ バックフィル | `netkeiba/pc/horse_result/{prefix}/{horse_id}.json` |
| `horse_pedigree_5gen` | 5世代血統 | バックフィル horse フェーズ | `netkeiba/pc/horse_pedigree_5gen/{prefix}/{horse_id}.json` |
| `horse_training` | 馬調教履歴 | 前日 JST 18:00（SLA 1） | `netkeiba/pc/horse_training/{prefix}/{horse_id}.json` |

#### SmartRC（二次）

| スクレイプカテゴリ | 内容 | 取得タイミング（SLA） | GCS 格納パス |
|---|---|---|---|
| `smartrc_race` | cr_value・first_furlong_time・estimated_popularity | T-15バンドル（SLA 3）+ 前日（SLA 1） | `netkeiba/pc/smartrc_race/{year}/{race_id}.json` |

#### JRA 公式

| スクレイプカテゴリ | 内容 | 取得タイミング | GCS 格納パス |
|---|---|---|---|
| `jra_cushion` | クッション値・含水率 | JST 05:00-08:50 毎10分（SLA 2） | `others/jra_cushion/{year}.json` |

### 4-3. 主要 DB スキーマ

#### `sires`（種牡馬マスター）

```sql
CREATE TABLE sires (
    sire_id    VARCHAR(20)   PRIMARY KEY,
    sire_name  VARCHAR(100)  NOT NULL,
    sire_line  VARCHAR(50),
    created_at TIMESTAMPTZ   DEFAULT NOW()
);
ALTER TABLE horses ADD COLUMN sire_id VARCHAR(20) REFERENCES sires(sire_id);
CREATE INDEX idx_horses_sire_id ON horses (sire_id);
```

#### `horse_stats_snapshot`（Layer 3）

```sql
CREATE TABLE horse_stats_snapshot (
    snapshot_id          BIGSERIAL      PRIMARY KEY,
    horse_id             VARCHAR(20)    NOT NULL,
    as_of_race_id        VARCHAR(20)    NOT NULL,
    as_of_date           DATE           NOT NULL,
    win_rate_all         NUMERIC(5,4),
    win_rate_turf        NUMERIC(5,4),
    win_rate_dirt        NUMERIC(5,4),
    place_rate_all       NUMERIC(5,4),
    show_rate_all        NUMERIC(5,4),
    win_rate_distance    NUMERIC(5,4),
    win_rate_course      NUMERIC(5,4),
    win_rate_going       NUMERIC(5,4),
    avg_last_3f          NUMERIC(5,2),
    speed_index_avg      NUMERIC(6,2),
    speed_index_max      NUMERIC(6,2),
    running_style_score  NUMERIC(5,2),
    sample_count         SMALLINT,
    created_at           TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (horse_id, as_of_race_id)
);
```

#### `race_odds_snapshot`（Layer 5）

```sql
CREATE TABLE race_odds_snapshot (
    snapshot_id      BIGSERIAL     PRIMARY KEY,
    race_id          VARCHAR(20)   NOT NULL,
    horse_id         VARCHAR(20)   NOT NULL,
    snapshot_type    VARCHAR(20)   NOT NULL
                     CHECK (snapshot_type IN ('WIN','PLACE','EXACTA','QUINELLA','WIDE')),
    odds_value       NUMERIC(7,1)  NOT NULL,
    odds_place_low   NUMERIC(7,1),
    odds_place_high  NUMERIC(7,1),
    snapshot_at      TIMESTAMPTZ   NOT NULL,
    CONSTRAINT uq_odds_snapshot
        UNIQUE (race_id, horse_id, snapshot_type, snapshot_at)
);
CREATE INDEX idx_odds_race_horse_time
    ON race_odds_snapshot (race_id, horse_id, snapshot_at DESC);
```

#### `prediction_results`（Layer 3）

```sql
CREATE TABLE prediction_results (
    prediction_id           BIGSERIAL     PRIMARY KEY,
    race_id                 VARCHAR(20)   NOT NULL,
    horse_id                VARCHAR(20)   NOT NULL,
    model_version           VARCHAR(50)   NOT NULL,
    predicted_at            TIMESTAMPTZ   DEFAULT NOW(),
    win_prob                NUMERIC(5,4),
    place_prob              NUMERIC(5,4),
    show_prob               NUMERIC(5,4),
    predicted_win_odds      NUMERIC(7,1),
    predicted_place_odds    NUMERIC(7,1),
    win_roi                 NUMERIC(7,2),
    show_roi                NUMERIC(7,2),
    predicted_position      SMALLINT,
    predicted_running_style VARCHAR(10),
    UNIQUE (race_id, horse_id, model_version)
);
```

> **矛盾解決**: AREA-03 の `expected_win_roi` / `expected_show_roi` と AREA-01 の `win_roi` / `show_roi` が競合。AREA-01 の最終更新日（2026-07-06）が最新のため `win_roi` / `show_roi` に統一。

#### `scrape_runs`（実行管理）

```sql
CREATE TABLE scrape_runs (
    run_id        BIGSERIAL    PRIMARY KEY,
    target_type   VARCHAR(50)  NOT NULL,
    target_id     VARCHAR(50),
    status        VARCHAR(20)  NOT NULL
                  CHECK (status IN ('SUCCESS','FAILED','RETRY')),
    retry_count   SMALLINT     DEFAULT 0,
    started_at    TIMESTAMPTZ  NOT NULL,
    finished_at   TIMESTAMPTZ,
    error_message TEXT,
    gcs_path      TEXT
);
```

#### 分析機能テーブル

```sql
CREATE TABLE course_stats_cache (
    id              SERIAL PRIMARY KEY,
    track           VARCHAR(20) NOT NULL,
    distance        INTEGER     NOT NULL,
    surface         VARCHAR(10) NOT NULL,
    track_condition VARCHAR(10) NOT NULL,
    stat_type       VARCHAR(30) NOT NULL,
    stat_key        VARCHAR(50) NOT NULL,
    n_runs          INTEGER,
    win_rate        NUMERIC(5,4),
    place_rate      NUMERIC(5,4),
    roi_win         NUMERIC(7,4),
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (track, distance, surface, track_condition, stat_type, stat_key)
);

CREATE TABLE saved_analyses (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
    name              VARCHAR(100) NOT NULL,
    analysis_type     VARCHAR(20) NOT NULL
                        CHECK (analysis_type IN ('sire','course','jockey','trainer')),
    filter_conditions JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    last_run_at       TIMESTAMPTZ
);
ALTER TABLE saved_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY saved_analyses_user_isolation
  ON saved_analyses FOR ALL
  USING (user_id = current_setting('app.current_user_id')::UUID);
```

### 4-4. REST API エンドポイント

ベースパス: `/api/v1`、レスポンス形式: `application/json`

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` | `/api/v1/races` | レース一覧（`?date=YYYYMMDD`） |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表 |
| `GET` | `/api/v1/races/{race_id}/entries` | 出馬表 |
| `GET` | `/api/v1/races/{race_id}/results` | 着順・ラップ・コーナー |
| `GET` | `/api/v1/races/{race_id}/predictions` | AI 予測（T-1〜T-9）|
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-7/T-8） |
| `GET` | `/api/v1/races/{race_id}/tracking-difficulty` | 位置追跡難易度 |
| `GET` | `/api/v1/horse/{id}/growth-curve` | 成長曲線 |
| `GET` | `/api/v1/track-speed/day` | TSI 指数（`?date=X&venue=Y`） |
| `GET` | `/api/v1/race-quality/race` | NNLS 分析（`?id=X`） |
| `GET` | `/api/v1/pedigree/race-note` | 血統適性マップデータ（`?race_id=X`） |
| `GET` | `/api/v1/bloodline-cluster/lookup` | クラスター検索（`?q=X`） |
| `GET` | `/api/v1/pedigree-race-stats/query` | 種牡馬成績クエリ |
| `POST` | `/api/v1/betting/optimize` | Kelly 基準 馬券最適化 |

`GET /api/v1/races/{race_id}/predictions` レスポンス例:

```json
{
  "race_id": "202506010811",
  "model_version": "v1.2.0",
  "predicted_at": "2025-06-01T08:30:00+09:00",
  "pace_prediction": {
    "pace_category": "MIDDLE",
    "lap_times": [
      { "furlong_index": 1, "predicted_lap_sec": 12.3 }
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

`is_value_bet`: `win_roi >= 100` または `show_roi >= 100` の場合に `true`。

### 4-5. フロントエンド ページカタログ

| URL | ページ名 | 説明 |
|---|---|---|
| `/` | ダッシュボード | 中央ハブ。最新開催日・レース一覧へのクイックアクセス |
| `/login` | ログイン | パスワード認証（「30日間ログイン保持」付き） |
| `/races?date=YYYYMMDD` | レース一覧 | 指定日の全レースカード |
| `/race/{race_id}` | レース詳細 | 4タブ（出馬表・結果・AI予測・出走馬詳細） |
| `/tracking-difficulty` | 位置追跡難易度 | ease スコア予測・可視化 |
| `/growth-curve` | 成長曲線 | キャリアアーク（馬体重×タイム指数）可視化 |
| `/track-speed` | トラックスピード指数 | TSI（Z スコア正規化、50=標準）表示 |
| `/race-quality` | レース品質分析 | NNLS 8アーキタイプ分類 |
| `/ai-sla` | AI SLA | パイプライン SLA ドキュメント |
| `/bloodline` | 血統分析 | 14分析タイプ（距離/コース研究） |
| `/bloodline-cluster` | 血統クラスター検索 | 馬・種牡馬の血統クラスター分類 |
| `/bloodline-vector` | 血統ベクトル空間 | Canvas 描画 2D ベクトルマップ（PCA/UMAP/t-SNE） |
| `/pedigree-map` | 血統マップ | D3.js ツリー/フォースグラフ |
| `/pedigree-race-stats` | 種牡馬成績クエリ | 多軸フィルタによる血統別成績統計 |
| `/note-aptitude-race` | 血統適性マップ | SVG パン/ズームマップ |
| `/myostatin` | Myostatin 遺伝子 | MSTN 遺伝子型（CC/CT/TT）距離適性追跡 |
| `/betting` | 馬券最適化 | Kelly 基準（Quarter Kelly デフォルト）、ログイン必須 |

### 4-6. ユーザー種別とアクセス制御

| 機能 | ゲスト | ログイン済 |
|---|---|---|
| レース一覧・基本情報 | ○ | ○ |
| AI 予測（全頭） | 上位3頭のみ | ○ |
| 出走馬過去成績（10走） | ✗ | ○ |
| データ分析全機能 | ○ | ○ |
| マイ分析（条件保存） | ✗ | ○ |
| 馬券最適化 | ✗ | ○ |
| 血統ツール全機能 | ○ | ○ |

### 4-7. デザインシステム（主要要素）

**カラーパレット（ダーク Navy テーマ）**:

```css
--bg: #0a0e17; --surface: #131926; --surface2: #1a2235; --border: #243049;
--text: #c8d6e5; --text-dim: #6b7d95;
--accent: #3b82f6; --ok: #22c55e; --warn: #f59e0b; --err: #ef4444;
--purple: #a78bfa; --cyan: #06b6d4; --teal: #2dd4bf;
```

**フォント**: Inter（本文）/ Noto Sans JP（日本語）/ Shippori Mincho（血統見出し）/ JetBrains Mono（数値・コード）

**枠番カラー**: 1=白 / 2=黒 / 3=赤(`#ef4444`) / 4=青(`#3b82f6`) / 5=黄(`#f59e0b`) / 6=緑(`#22c55e`) / 7=オレンジ(`#f97316`) / 8=ピンク(`#ec4899`)

---

## 5. 非機能要件（確定版）

### 5-1. パフォーマンス要件

| ID | 要件 | 目標値 |
|---|---|---|
| N-1 | API レスポンス（キャッシュヒット） | ≤ 200 ms |
| N-2 | API レスポンス（キャッシュミス） | ≤ 2,000 ms |
| N-3 | ラップ予測 MAE | ≤ 0.3 秒 |
| N-4 | 勝率モデル Log Loss | ベースライン比 −5% 以上改善 |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | DB 反映遅延 | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前5分以内） | ≤ 1% |
| N-9 | 推論バッチ完了タイミング | 発走 3 時間前まで |
| — | LCP（Largest Contentful Paint） | ≤ 2.5s（3G） |
| — | CLS | ≤ 0.1 |
| — | BFF エンドポイント `GET /api/v1/races/{race_id}/full` p95 | ≤ 300 ms |
| — | Redis キャッシュヒット率（`predictions:{race_id}:full`） | ≥ 80% |
| — | Gunicorn OOM Kill 発生率 | 0件/月 |

### 5-2. 可用性・品質要件

| ID | 要件 | 目標値 |
|---|---|---|
| N-10 | CI テスト（テンポラルリーク検知） | 自動実行、CI ゲートブロッキング |
| N-11 | スキーマ管理 | Alembic マイグレーションによるバージョン管理 |
| N-12 | Redis キャッシュ | 発走まで有効 / 発走後 60 秒で自動失効 |
| N-13 | テストカバレッジ | スクレイパー・特徴量パイプライン・モデル・API 各 ≥ 80% |
| N-14 | 障害・海外レース除外 | `races.is_excluded = TRUE` フラグで推論対象外管理 |
| — | Lighthouse Performance | ≥ 85 |
| — | Lighthouse Accessibility | ≥ 90 |

### 5-3. Redis TTL 設計

| キャッシュキー | TTL | 無効化タイミング |
|---|---|---|
| `prediction:{race_id}:{model_version}` | 発走時刻まで（`EXPIREAT` 設定） + 発走後 60 秒 | 発走時刻 + 60s |
| `lap:prediction:{race_id}:{model_version}` | 同上 | 同上 |
| `race:entries:{race_id}` | 3,600 秒 | 再スクレイピング完了時に `DEL` |
| `race:results:{race_id}` | 300 秒 | — |
| `track:speed:{date}:{venue}` | 86,400 秒 | — |
| `odds:latest:{race_id}` | 300 秒（オッズ更新間隔） | — |

---

## 6. AI / ML パイプライン

### 6-1. 予測ターゲット詳細（推論順序・アルゴリズム）

| ターゲット | アルゴリズム | 主評価指標 | 合格閾値 |
|---|---|---|---|
| T-1 勝率 | LightGBM softmax | Log Loss（ベースライン比） | −5% 以上改善 |
| T-2 連対率 | LightGBM binary | Binary Log Loss、AUC-ROC | — |
| T-3 複勝率 | LightGBM binary | 同上 | — |
| T-4 上り3Fタイム | LightGBM Regressor | MAE（秒）、RMSE | RMSE < 0.3秒（T-9投入ゲート） |
| T-5 位置取り予測 | LightGBM lambdarank | Spearman ρ、MAE | Spearman ρ ≥ 0.55 |
| T-6 脚質分類 | LightGBM Classifier | Accuracy、F1-macro | 自動ラベル一致率 > 80% |
| T-7 ラップ系列 | LightGBM per-furlong → LSTM（Phase 4） | RMSE per furlong | MAE ≤ 0.3秒 |
| T-8 ペースカテゴリ | LightGBM Classifier | F1-macro | — |
| T-9 想定走破タイム（派生） | T-4 + L3F回帰モデル合成 | RMSE | T-4 RMSE < 0.3秒 通過後に本番投入 |

推論実行順序:
```
Stage 1: T-6（脚質分類）
Stage 2: T-8（ペース）→ T-4/T-5（同時）→ T-9（T-4完了後）
Stage 3: T-7（Phase 4 フラグ ON のみ）
```

### 6-2. 主要特徴量

**基本特徴量（Layer 1〜2）**: distance, surface, direction, going, weather, grade, horse_num, frame_no, post_no, weight_carried, horse_weight, horse_weight_diff, days_since_last, horse_age, sex

**集計特徴量（Layer 3 スナップショット）**: win_rate_all / place_rate_all / show_rate_all、win_rate_distance / win_rate_course / win_rate_going、avg_last_3f / speed_index_avg / speed_index_max、running_style_score（−5=逃〜+5=追込）、jockey.win_rate_all、trainer.win_rate_all

**クロス・相対特徴量（前処理自動生成）**:
- `style_x_straight = running_style_score × final_straight_length`
- `front_runner_count`（同レース内逃げ先行馬数）
- `rel_speed_index`（同レース内相対スピード指数）
- `rel_odds_rank`（同レース内オッズ順位）
- `pace_scenario_prior`（逃げ馬比率 > 0.3 = HIGH、< 0.1 = SLOW、それ以外 = MIDDLE）

**T-9 用特徴量（L3F地点回帰モデル）**: 過去N走の `l3f_split_time_sec` 平均、距離、馬場状態、T-8 ペースカテゴリ予測値、騎手別 l3f オフセット

```sql
-- 教師データ生成クエリ
SELECT horse_id, race_id,
       finish_time_sec - last3f_sec AS l3f_split_time_sec
FROM race_results
WHERE finish_time_sec IS NOT NULL AND last3f_sec IS NOT NULL;
```

### 6-3. 学習パイプライン原則

- 時系列順に train / validation / test を分割（ランダムシャッフル禁止）
- 常に過去レースで学習・未来レースで評価
- Layer 3 集計値は `as_of_race_id = 予測対象レース ID` のスナップショットのみ参照
- Feature Store API: `get_snapshot(race_id, as_of=race_id)` の `window_end=as_of_race_id` 必須引数

### 6-4. CI ゲート（ブロッキング条件）

| ゲート | 合格条件 |
|---|---|
| テンポラルリーク検知テスト | PASS 必須（ブロッキング） |
| Unit テストカバレッジ | 各モジュール ≥ 80% |
| 勝率 Log Loss | ベースライン比 −5% 以上改善 |
| Spearman ρ | ≥ 0.55 |
| ラップ MAE | ≤ 0.3 秒 |
| API レスポンス（キャッシュヒット） | ≤ 200 ms |
| API レスポンス（キャッシュミス） | ≤ 2,000 ms |
| 時系列分割ランダムシャッフル | `train_test_split(shuffle=True)` 検出でブロック |

---

## 7. 運用コスト

### 7-1. 確定済みインフラコンポーネント（コスト試算の前提）

| コンポーネント | 用途 | 備考 |
|---|---|---|
| ConoHa VPS 2GB | prod サーバー全般 | 月額費用は未確定（AREA-05 未決定事項） |
| PostgreSQL | Layer 1〜5 データ格納 | VPS 上でセルフホスト |
| Redis | 予測結果キャッシュ | VPS 上でセルフホスト、maxmemory=128mb |
| GCS | 永続 JSON ストア | バケット: `magu-keiba-horse-racing-ai`（`GCS_BUCKET` 環境変数で注入） |
| LightGBM | 推論（セルフホスト） | 外部 API 委託なし |
| MLflow（×4プロセス） | モデルバージョン管理 | ports: 5001/5003/5004/5010 |

### 7-2. 未確定事項（別途 DEC 作成が必要）

| 項目 | 状態 |
|---|---|
| 月額費用内訳（サーバー・DB・Redis・GCS 等） | **未定義** |
| スケールアップ判断基準（CPU・メモリ・レイテンシ閾値） | **未定義** |
| AI 推論外部化コスト比較（LightGBM セルフホスト vs. 外部 API） | **未定義** |
| コスト削減方針（スポットインスタンス・コールドストレージ移行等） | **未定義** |

---

## 8. 未解決事項・Human 判断待ち

### 8-1. 運用・インフラ系

| # | 項目 | 関連 AREA |
|---|---|---|
| OP-1 | 各モデルロード時の実測メモリ使用量計測（512MB 制約適合確認） | AREA-04/10 |
| OP-3 | 監視基盤ツール選定（Prometheus+Grafana / Sentry / Datadog / 自作） | AREA-04/10 |
| OP-4 | アラート通知チャネル（Slack / メール / LINE）と宛先 | AREA-04/10 |
| MON-3 | PostgreSQL `shared_buffers` 最適値（実負荷計測後に調整） | AREA-10 |
| MON-4 | `scrape_runs` テーブルの成長監視・VACUUM スケジュール設計 | AREA-10 |

### 8-2. Circuit Breaker

Circuit Breaker ライブラリの選定（`pybreaker` / `tenacity` 等）、閾値定義（連続失敗 N 回でオープン状態遷移・クールダウン時間）、適用対象（netkeiba.com HTTP クライアント・Redis・PostgreSQL 接続）が未確定。

### 8-3. データ管理系

| # | 項目 | 関連 AREA |
|---|---|---|
| DM-1 | GCS バケット命名規則（本番・ステージング分離） | AREA-06 |
| DM-2 | ディスクキャッシュ容量上限の明示 | AREA-06 |
| DM-3 | GCS 書き込み失敗時のリトライ・アラート設計 | AREA-06 |
| DM-4 | Feature Store の GCS バックアップ設計 | AREA-06 |

### 8-4. 開発環境系

| # | 項目 | 関連 AREA |
|---|---|---|
| — | dev / stg / prod の環境分離方針（ローカル PC・GPU サーバー・VPS 割り当て） | AREA-09 |
| — | docker-compose ファイルの具体的設計 | AREA-09 |
| — | CI/CD パイプラインおよびデプロイフロー | AREA-09 |
| — | 環境変数管理方法（`.env` / シークレット管理） | AREA-09 |
| — | GPU 環境要件（CUDA バージョン・GPU メモリ等） | AREA-09 |

### 8-5. フロントエンド系

| # | 項目 | 関連 AREA |
|---|---|---|
| — | `entries.post_position` が馬番・枠番どちらかの確認（Phase 0-S 検証後に `frame_no` カラム追加要否決定） | AREA-01 |

### 8-6. ML パイプライン系

| # | 項目 | 関連 AREA |
|---|---|---|
| — | Phase 4 `PHASE4_LAP_PREDICTION` フィーチャーフラグの有効化タイミング | AREA-07 |
| — | T-9（想定走破タイム）本番投入の T-4 RMSE ゲート通過確認 | AREA-07 |

---

## 9. 参照 AREA 一覧

| AREA | ファイル名 | 内容 | Status | Last Updated |
|---|---|---|---|---|
| AREA-01 | `AREA-01-app-requirements.md` | アプリケーション要件（スクレイピング・DB スキーマ・予測ターゲット・ロードマップ） | FINAL | 2026-07-06 |
| AREA-02 | `AREA-02-frontend.md` | フロントエンド要件（デザインシステム・ページカタログ・API 消費仕様） | REVISED | 2026-07-04 |
| AREA-03 | `AREA-03-backend.md` | バックエンド要件（Flask API・DB スキーマ・認証・キャッシュ・レート制限） | FINAL | 2026-07-04 |
| AREA-04 | `AREA-04-ops.md` | 運用最適化要件（Cron SLA・プロセス分離・Circuit Breaker・監視・デプロイ） | FINAL | 2026-07-06 |
| AREA-05 | `AREA-05-cost.md` | コスト計算要件（月額費用・スケール判断基準・コスト削減方針） | FINAL | 2026-07-04 |
| AREA-06 | `AREA-06-data.md` | データ管理要件（GCS パス設計 SSoT・ETL パイプライン・Feature Store・Redis TTL） | FINAL | 2026-07-04 |
| AREA-07 | `AREA-07-modeling.md` | モデリング管理要件（LightGBM バッチ推論・学習パイプライン・SHAP・ModelRegistry） | FINAL | 2026-07-06 |
| AREA-08 | `AREA-08-testing.md` | テスト要件（Unit/Integration/E2E/ML テスト・CI ゲート・カバレッジ・テストデータ） | FINAL | 2026-07-04 |
| AREA-09 | `AREA-09-dev-environment.md` | 開発環境要件（環境分離・docker-compose・デプロイフロー） | FINAL | 2026-07-04 |
| AREA-10 | `AREA-10-infra-monitoring.md` | インフラ・サーバーモニタリング要件（リソース監視・アラート・ログ・SLO） | FINAL | 2026-07-04 |