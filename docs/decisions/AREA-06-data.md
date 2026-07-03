# AREA-03 — データ管理要件（GCS パス設計 gcs_paths.py SSoT, ETL パイプライン, Feature Store, Redis TTL 設計）
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001, DEC-010

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるデータ管理要件を定義する。対象範囲は以下の4領域とする。

1. **GCS パス設計（`gcs_paths.py` SSoT）**
2. **ETL パイプライン**
3. **Feature Store（特徴量スナップショット管理）**
4. **Redis TTL 設計**

全設計の大前提：**`as_of_race_id` によるスナップショット管理でテンポラルリーク（未来情報混入）を構造的に排除すること**。これはデータ基盤・モデリング・評価の全工程に適用される不変制約である。

---

## 2. データ層アーキテクチャ（5層構造）

ETL・Feature Store・GCS パス設計の全領域は以下の5層スキーマを基準とする。

| 層 | 名称 | 内容 | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | races, entries, horses, jockeys, trainers, courses | 追記・参照更新 |
| Layer 2 | 確定結果 | race_results（着順・タイム・馬体重・コーナー通過順） | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | horse_stats_snapshot, jockey_stats_snapshot, trainer_stats_snapshot | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | race_lap_times, race_corner_positions, race_pace_summary | 追記のみ |
| Layer 5 | オッズ時系列 | race_odds_snapshot（snapshot_at 付き） | 追記のみ・削除不可 |

---

## 3. GCS パス設計（`gcs_paths.py` SSoT）

### 3-1. 設計方針

- `gcs_paths.py` をパス定義の **Single Source of Truth（SSoT）** とし、全モジュールからインポートして使用する
- パスはデータ層・エンティティ種別・日付・レースIDの階層で構成する
- 生データ・変換済みデータ・モデル成果物・特徴量を明確に分離したプレフィックスで管理する

### 3-2. バケット構成

```
gs://keiba-vpn-data/
├── raw/                          # スクレイピング生データ（Layer 1〜5 原形）
├── processed/                    # クリーニング・変換済みデータ
├── features/                     # Feature Store（Layer 3 スナップショット）
├── models/                       # 学習済みモデル成果物
├── predictions/                  # 推論結果
└── logs/                         # スクレイピング実行ログ
```

### 3-3. `gcs_paths.py` 定義

```python
# gcs_paths.py  ── GCS パス定義 SSoT
# 全モジュールはこのファイルから定数をインポートすること

BUCKET = "keiba-vpn-data"

# ── Layer 1: 静的マスター ──────────────────────────────────────
def raw_race_card(race_id: str) -> str:
    """出馬表 HTML 生データ"""
    date_prefix = race_id[:8]  # YYYYMMDD
    return f"raw/layer1/race_card/{date_prefix}/{race_id}.html"

def raw_horse(horse_id: str) -> str:
    """馬プロフィール HTML 生データ"""
    return f"raw/layer1/horses/{horse_id}.html"

def raw_jockey(jockey_id: str) -> str:
    """騎手プロフィール HTML 生データ"""
    return f"raw/layer1/jockeys/{jockey_id}.html"

def raw_trainer(trainer_id: str) -> str:
    """調教師プロフィール HTML 生データ"""
    return f"raw/layer1/trainers/{trainer_id}.html"

# ── Layer 2: 確定結果 ─────────────────────────────────────────
def raw_race_result(race_id: str) -> str:
    """レース結果 HTML 生データ"""
    date_prefix = race_id[:8]
    return f"raw/layer2/race_results/{date_prefix}/{race_id}.html"

# ── Layer 4: ラップ・コーナー ──────────────────────────────────
def raw_race_lap(race_id: str) -> str:
    """ラップタイム・コーナー通過 HTML 生データ"""
    date_prefix = race_id[:8]
    return f"raw/layer4/laps/{date_prefix}/{race_id}.html"

# ── Layer 5: オッズスナップショット ────────────────────────────
def raw_odds_snapshot(race_id: str, snapshot_at_iso: str) -> str:
    """
    オッズスナップショット HTML 生データ
    snapshot_at_iso: ISO8601 形式 (例: "20250601T083000")
    """
    date_prefix = race_id[:8]
    return f"raw/layer5/odds/{date_prefix}/{race_id}/{snapshot_at_iso}.html"

# ── processed: 変換済みデータ（Parquet） ───────────────────────
def processed_race_card(race_id: str) -> str:
    date_prefix = race_id[:8]
    return f"processed/layer1/race_card/{date_prefix}/{race_id}.parquet"

def processed_race_result(race_id: str) -> str:
    date_prefix = race_id[:8]
    return f"processed/layer2/race_results/{date_prefix}/{race_id}.parquet"

def processed_race_lap(race_id: str) -> str:
    date_prefix = race_id[:8]
    return f"processed/layer4/laps/{date_prefix}/{race_id}.parquet"

def processed_odds_snapshot(race_id: str) -> str:
    """当該レースの全オッズスナップショットを結合した Parquet"""
    date_prefix = race_id[:8]
    return f"processed/layer5/odds/{date_prefix}/{race_id}.parquet"

# ── features: Feature Store（Layer 3 スナップショット） ─────────
def feature_horse_snapshot(horse_id: str, as_of_race_id: str) -> str:
    """
    馬スナップショット特徴量 Parquet
    as_of_race_id でテンポラルリークを防止
    """
    date_prefix = as_of_race_id[:8]
    return f"features/horse_stats/{date_prefix}/{as_of_race_id}/{horse_id}.parquet"

def feature_jockey_snapshot(jockey_id: str, as_of_race_id: str) -> str:
    date_prefix = as_of_race_id[:8]
    return f"features/jockey_stats/{date_prefix}/{as_of_race_id}/{jockey_id}.parquet"

def feature_trainer_snapshot(trainer_id: str, as_of_race_id: str) -> str:
    date_prefix = as_of_race_id[:8]
    return f"features/trainer_stats/{date_prefix}/{as_of_race_id}/{trainer_id}.parquet"

def feature_race_combined(race_id: str) -> str:
    """レース単位で全エンティティの特徴量を結合した学習用 Parquet"""
    date_prefix = race_id[:8]
    return f"features/race_combined/{date_prefix}/{race_id}.parquet"

# ── models: 学習済みモデル成果物 ────────────────────────────────
def model_artifact(model_name: str, version: str) -> str:
    """
    model_name: "stage1_win_prob" / "stage1_position" / "stage2_lap" 等
    version: "v1.2.0" 形式
    """
    return f"models/{model_name}/{version}/model.pkl"

def model_metadata(model_name: str, version: str) -> str:
    return f"models/{model_name}/{version}/metadata.json"

# ── predictions: 推論結果 ───────────────────────────────────────
def prediction_result(race_id: str, model_version: str) -> str:
    date_prefix = race_id[:8]
    return f"predictions/{date_prefix}/{race_id}/{model_version}.parquet"

# ── logs: スクレイピング実行ログ ────────────────────────────────
def scrape_log(date_str: str) -> str:
    """date_str: "YYYYMMDD" 形式"""
    return f"logs/scrape_runs/{date_str}.jsonl"
```

---

## 4. ETL パイプライン

### 4-1. パイプライン全体像

```
netkeiba.com
    │ HTTP スクレイピング
    ▼
[Extract]  生HTML → GCS raw/ (gcs_paths.py 経由)
    │
    ▼
[Transform] HTML解析 → Parquet → GCS processed/
    │
    ▼
[Load]     Parquet → PostgreSQL (Layer 1〜5 テーブル)
               │
               ▼
           [Snapshot Batch]
           Layer 3 スナップショット生成
           (as_of_race_id 付与)
               │
               ▼
           GCS features/ & PostgreSQL horse_stats_snapshot
```

### 4-2. スクレイピング収集スケジュール

| ジョブ名 | トリガー | 間隔 | 優先ウィンドウ |
|---|---|---|---|
| `race_card` | レース3日前 06:00 JST | 毎日 06:00（発走まで） | — |
| `odds_snapshot` | 発走当日 08:00〜発走時刻 | 5分毎 | 発走30分前: 2分毎 / 発走5分前: 1分毎 |
| `race_results` | 発走予定時刻 + 35分 | リトライ: 5分間隔 × 最大6回 | — |
| `horse_history` | results 収集完了後 | 結果確定後1回 | 前走成績更新後に再取得 |

### 4-3. スクレイピング設定

```python
SCRAPING_CONFIG = {
    "request_interval_sec": 2.0,
    "jitter_sec": (0.5, 1.5),        # ランダム遅延でBot検出回避
    "concurrent_workers": 1,          # シングルIP環境では並列1推奨
    "session_rotate_interval": 50,    # 50リクエスト毎セッション更新
    "retry_on_429": True,
    "retry_backoff_base_sec": 30,
    "user_agent_rotate": True,
}
```

### 4-4. Transform ルール

| 対象データ | 変換内容 | 出力形式 |
|---|---|---|
| 出馬表 HTML | レース基本情報・出走馬リスト抽出 | Parquet → Layer 1 |
| レース結果 HTML | 着順・タイム・馬体重・コーナー通過順抽出 | Parquet → Layer 2 |
| ラップ HTML | 1F毎ラップタイム・ペース区分算出 | Parquet → Layer 4 |
| オッズ HTML | 単勝・複勝・snapshot_at 付きレコード生成 | Parquet → Layer 5 |
| 馬過去成績 HTML | 勝率・連対率・複勝率・脚質スコア集計 | Parquet → Layer 3（スナップショット） |

### 4-5. 実行管理（`scrape_runs` テーブル）

```sql
CREATE TABLE scrape_runs (
    run_id         BIGSERIAL    PRIMARY KEY,
    target_type    VARCHAR(30)  NOT NULL,   -- 'race_card' / 'race_result' / 'odds' / 'horse_history'
    target_id      VARCHAR(20)  NOT NULL,   -- race_id または horse_id
    status         VARCHAR(10)  NOT NULL    -- 'SUCCESS' / 'FAILED' / 'RETRY'
                   CHECK (status IN ('SUCCESS','FAILED','RETRY')),
    retry_count    SMALLINT     DEFAULT 0,
    started_at     TIMESTAMPTZ  NOT NULL,
    finished_at    TIMESTAMPTZ,
    error_message  TEXT,
    gcs_path       TEXT                     -- 生データ保存先 GCS パス
);
```

- スクレイピング成功率の目標値: **≥ 99% / 月**
- DB 反映遅延の目標値: **≤ 10 分**（スクレイピング完了から）

---

## 5. Feature Store（特徴量スナップショット管理）

### 5-1. 設計原則

- Layer 3 の集計値は必ず **`as_of_race_id`（予測対象レース）に紐付けて保存** し、そのレース以後の情報を含めない（テンポラルリーク防止の根幹）
- スナップショットは **追記専用・不変（Immutable）** とし、更新・削除を禁止する
- DB には `UNIQUE(entity_id, as_of_race_id)` 制約を設け、二重登録を防止する

### 5-2. Feature Store テーブルスキーマ

#### `horse_stats_snapshot`

```sql
CREATE TABLE horse_stats_snapshot (
    snapshot_id          BIGSERIAL      PRIMARY KEY,
    horse_id             VARCHAR(20)    NOT NULL,
    as_of_race_id        VARCHAR(20)    NOT NULL,   -- 予測対象レース直前時点
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

---

## 6. スクレイピング耐障害性・Circuit Breaker 設計（DEC-010 確定）

### 6-1. Circuit Breaker（pybreaker）

```python
# scraper/circuit_breaker.py
import pybreaker

# 5 回連続失敗 → OPEN、60 秒後 HALF-OPEN 自動遷移
scrape_circuit = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)

@scrape_circuit
def fetch_race_card(race_id: str) -> bytes:
    """Circuit Breaker 保護下での出馬表 HTML 取得"""
    ...
```

**OPEN 時の挙動**:
1. `gcs_paths.py` で定義された最終成功データの GCS パスからフォールバック表示
2. フロントエンドに「データ更新中」バナーを表示（DEC-010 F-8）
3. Slack Webhook で 5 分以内にアラート送信（DEC-010 F-9）

### 6-2. 指数バックオフリトライ（tenacity）

```python
# scraper/scraper.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=4, min=4, max=60))
def scrape_with_retry(url: str) -> bytes:
    """最大 3 回リトライ（4s → 8s → 60s）"""
    ...
```

### 6-3. scrape_runs テーブルへの Circuit Breaker 状態記録（DEC-010 F-22）

```sql
-- Circuit Breaker 対応のため status に 'circuit_open' を追加
ALTER TABLE scrape_runs
    ADD COLUMN circuit_state VARCHAR(15)
        CHECK (circuit_state IN ('CLOSED','OPEN','HALF_OPEN'));
```

- Circuit Breaker 状態変化（CLOSED→OPEN 等）も `scrape_runs` テーブルに記録する
- `/admin/circuit-status` エンドポイントでサーキット状態・失敗回数・最終失敗時刻を返す（DEC-010 F-21）

### 6-4. データ鮮度チェック（DEC-010 F-23）

API レスポンスに `data_freshness` フィールドを含める:

| `FreshnessStatus` | 意味 |
|---|---|
| `FRESH` | 最新データ正常取得 |
| `STALE` | 前バージョンデータ使用中（スクレイプ遅延） |
| `STALE_CIRCUIT_OPEN` | Circuit OPEN 中・GCS フォールバック表示 |
| `UNKNOWN` | 鮮度情報取得不可 |