# AREA-03 — バックエンド要件（Flask API, DB スキーマ, 認証・認可, 4 層キャッシュ設計, レート制限）
**Status**: FINAL | **Last Updated**: 2026-07-04 | **Consolidates**: DEC-001（統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトのバックエンド（Flask API、DB スキーマ、認証・認可、キャッシュ設計、レート制限）に関する確定要件を定める。DEC-001 が定めるデータ基盤・スキーマ・非機能要件の記述を単一ソースとして統合する。

> **前提**: `as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除することが全工程の前提条件である（DEC-001）。

---

## 2. DB スキーマ

### 2-1. 層別構成（5 層）

| 層 | 役割 | 主テーブル |
|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses` |
| Layer 2 | 確定結果（追記のみ） | `race_results` |
| Layer 3 | 集計スナップショット（追記型・不変） | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` |
| Layer 4 | ラップ・ペース・通過順位 | `race_lap_times`, `race_corner_positions`, `race_pace_summary` |
| Layer 5 | オッズ時系列（追記型・削除不可） | `race_odds_snapshot` |

### 2-2. Layer 3: `horse_stats_snapshot`

```sql
CREATE TABLE horse_stats_snapshot (
    snapshot_id          BIGSERIAL      PRIMARY KEY,
    horse_id             VARCHAR(20)    NOT NULL,
    as_of_race_id        VARCHAR(20)    NOT NULL,   -- 予測対象レース直前時点
    as_of_date           DATE           NOT NULL,
    win_rate_all         NUMERIC(5,4),
    win_rate_turf        NUMERIC(5,4),
    win_rate_dirt        NUMERIC(5,4),
    place_rate_all       NUMERIC(5,4),              -- 連対率（2着以内）
    show_rate_all        NUMERIC(5,4),              -- 複勝率（3着以内）
    win_rate_distance    NUMERIC(5,4),              -- ±200m同距離帯
    win_rate_course      NUMERIC(5,4),              -- 同コース(場・距離・芝砂)
    win_rate_going       NUMERIC(5,4),              -- 同馬場状態
    avg_last_3f          NUMERIC(5,2),              -- 直近5走平均上がり3F
    speed_index_avg      NUMERIC(6,2),
    speed_index_max      NUMERIC(6,2),
    running_style_score  NUMERIC(5,2),              -- -5(逃)〜+5(追込)
    sample_count         SMALLINT,
    created_at           TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (horse_id, as_of_race_id)
);
```

### 2-3. Layer 4: ラップ・ペース・コーナー

```sql
CREATE TABLE race_lap_times (
    race_id          VARCHAR(20)   NOT NULL,
    furlong_index    SMALLINT      NOT NULL,   -- 1始まり（1F目=スタート直後）
    lap_time_sec     NUMERIC(4,2)  NOT NULL,
    cumulative_sec   NUMERIC(6,2),
    PRIMARY KEY (race_id, furlong_index)
);

CREATE TABLE race_corner_positions (
    race_id    VARCHAR(20)   NOT NULL,
    horse_id   VARCHAR(20)   NOT NULL,
    corner_1   SMALLINT,                      -- NULL = コース形状上存在しない
    corner_2   SMALLINT,
    corner_3   SMALLINT,
    corner_4   SMALLINT,
    PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE race_pace_summary (
    race_id            VARCHAR(20)   NOT NULL PRIMARY KEY,
    first_3f_sec       NUMERIC(5,2),
    last_3f_sec        NUMERIC(5,2),
    pace_category      VARCHAR(10)
                       CHECK (pace_category IN ('HIGH','MIDDLE','SLOW')),
    front_runner_count SMALLINT,
    created_at         TIMESTAMPTZ   DEFAULT NOW()
);
```

### 2-4. Layer 5: `race_odds_snapshot`

```sql
CREATE TABLE race_odds_snapshot (
    snapshot_id      BIGSERIAL     PRIMARY KEY,
    race_id          VARCHAR(20)   NOT NULL,
    horse_id         VARCHAR(20)   NOT NULL,
    snapshot_type    VARCHAR(20)   NOT NULL
                     CHECK (snapshot_type IN (
                         'WIN',        -- 単勝
                         'PLACE',      -- 複勝
                         'EXACTA',     -- 馬単
                         'QUINELLA',   -- 馬連
                         'WIDE'        -- ワイド
                     )),
    odds_value       NUMERIC(7,1)  NOT NULL,
    odds_place_low   NUMERIC(7,1),              -- 複勝下限（PLACEのみ）
    odds_place_high  NUMERIC(7,1),              -- 複勝上限（PLACEのみ）
    snapshot_at      TIMESTAMPTZ   NOT NULL,
    CONSTRAINT uq_odds_snapshot
        UNIQUE (race_id, horse_id, snapshot_type, snapshot_at)
);

CREATE INDEX idx_odds_race_horse_time
    ON race_odds_snapshot (race_id, horse_id, snapshot_at DESC);
```

### 2-5. 推論結果テーブル

```sql
CREATE TABLE prediction_results (
    prediction_id            BIGSERIAL     PRIMARY KEY,
    race_id                  VARCHAR(20)   NOT NULL,
    horse_id                 VARCHAR(20)   NOT NULL,
    model_version            VARCHAR(50)   NOT NULL,
    predicted_at             TIMESTAMPTZ   DEFAULT NOW(),
    win_prob                 NUMERIC(5,4),
    place_prob               NUMERIC(5,4),
    show_prob                NUMERIC(5,4),
    predicted_win_odds       NUMERIC(7,1),
    predicted_place_odds     NUMERIC(7,1),
    expected_win_roi         NUMERIC(7,2),
    expected_show_roi        NUMERIC(7,2),
    predicted_position       SMALLINT,
    predicted_running_style  VARCHAR(10),
    UNIQUE (race_id, horse_id, model_version)
);

CREATE TABLE prediction_lap_times (
    race_id              VARCHAR(20)   NOT NULL,
    model_version        VARCHAR(50)   NOT NULL,
    furlong_index        SMALLINT      NOT NULL,
    predicted_lap_sec    NUMERIC(4,2),
    predicted_pace_cat   VARCHAR(10)
                         CHECK (predicted_pace_cat IN ('HIGH','MIDDLE','SLOW')),
    PRIMARY KEY (race_id, model_version, furlong_index)
);
```

### 2-6. スクレイプ実行管理テーブル

```sql
-- F-5: scrape_runs テーブルでスクレイプ実行ログを管理する
CREATE TABLE scrape_runs (
    run_id        BIGSERIAL     PRIMARY KEY,
    target_type   VARCHAR(50)   NOT NULL,   -- 'race_card' | 'odds' | 'results' | 'horse_history' 等
    target_id     VARCHAR(50),              -- race_id / horse_id 等
    status        VARCHAR(20)   NOT NULL,   -- 'SUCCESS' | 'FAILURE' | 'RETRY'
    retry_count   SMALLINT      DEFAULT 0,
    started_at    TIMESTAMPTZ   NOT NULL,
    finished_at   TIMESTAMPTZ,
    error_message TEXT
);
```

### 2-7. スキーマ管理

- 全スキーマ変更は **Alembic** によるマイグレーションでバージョン管理する（N-11）。
- 障害レース・海外レースは `races` テーブルの `is_excluded` フラグ（`BOOLEAN DEFAULT FALSE`）で予測対象外を明示管理する（N-14）。

---

## 3. REST API 仕様

### 3-1. 基本仕様

| 項目 | 値 |
|---|---|
| フレームワーク | Flask |
| ベースパス | `/api/v1` |
| レスポンス形式 | `application/json` |
| タイムゾーン | UTC（`TIMESTAMPTZ`）、表示用は JST |

### 3-2. エンドポイント一覧

| メソッド | エンドポイント | 説明 | 機能要件 |
|---|---|---|---|
| `GET` | `/api/v1/races/{race_id}/predictions` | 全予測ターゲット（T-1〜T-9）取得 | F-10 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11）取得 | F-11 |
| `GET` | `/api/v1/races` | レース一覧取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表取得 | F-13 |

### 3-3. `GET /api/v1/races/{race_id}/predictions` レスポンス仕様

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

- `is_value_bet`: `expected_win_roi >= 100` または `expected_show_roi >= 100` の場合に `true`（F-14）。
- ラップ予測（T-10〜T-11）はレスポンス内の `pace_prediction` オブジェクトおよび `/predictions/laps` エンドポイントで `furlong_index` 昇順で提供する（F-11）。

### 3-4. パフォーマンス要件

| 条件 | 目標レスポンスタイム |
|---|---|
| キャッシュヒット時 | ≤ 200 ms（N-1） |
| キャッシュミス時 | ≤ 2,000 ms（N-2） |

---

## 4. 認証・認可

DEC-001 の現時点の記述には認証・認可の具体的スキームが明示されていない。以下は DEC-001 が定める機能・非機能要件から導出した最低限の実装方針とする。追加の認証要件が別 DEC で定義された場合は、その DEC を優先する。

| 項目 | 方針 |
|---|---|
| 認証方式 | API キー（`Authorization: Bearer <token>` ヘッダー）または セッション Cookie（UI 向け） |
| 認可スコープ | 予測 API・レース API は読み取り専用。スクレイプ実行・モデル管理は内部ネットワーク限定 |
| 管理系エンドポイント | `/api/v1/admin/*` は内部 IP（127.0.0.1 / VPN 内）のみアクセス許可 |
| DDL 操作 | Alembic マイグレーション実行権限はサービスアカウントに限定 |

---

## 5. キャッシュ設計

DEC-001 では Redis を用いた予測結果のキャッシュが明示されている（F-12、N-12）。以下の 4 層構成で実装する。

### 5-1. 4 層キャッシュ構成

| 層 | 種別 | 対象 | TTL | キャッシュキー例 |
|---|---|---|---|---|
| L1 | Flask アプリ内メモリ（`lru_cache`） | コース・マスターデータ等の静的情報 | プロセス再起動まで | N/A（関数単位） |
| L2 | Redis — 予測結果 | 全ターゲット予測（T-1〜T-9） | 発走時刻まで / 発走後 60 秒で自動失効 | `prediction:{race_id}:{model_version}` |
| L3 | Redis — ラップ予測 | ラップ予測系列（T-10〜T-11） | 発走時刻まで / 発走後 60 秒で自動失効 | `lap:prediction:{race_id}:{model_version}` |
| L4 | Redis — オッズスナップショット | 直近オッズ（推論特徴量用） | 5 分（オッズ更新間隔に合わせる） | `odds:latest:{race_id}` |

### 5-2. TTL 詳細ルール

- **発走前**: 予測結果キャッシュは発走予定時刻まで有効とし、発走予定時刻を `EXPIREAT` で設定する。
- **発走後**: 発走後 60 秒で自動失効させ、確定