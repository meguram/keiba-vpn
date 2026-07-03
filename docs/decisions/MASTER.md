# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照DEC: DEC-001

---

## 1. プロジェクト概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの**勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測**、ならびに**逃げ馬ペース予測・1F単位ラップ予測**を実現する競馬予測システム。

**最重要設計原則**: `as_of_race_id` によるスナップショット管理でテンポラルリーク（未来情報の混入）を構造的に排除する。これはデータ基盤・モデリング・評価の全工程の前提条件である。

| 項目 | 内容 |
|---|---|
| プロジェクト名 | keiba-vpn |
| データソース | netkeiba.com（単一ソース） |
| 予測ターゲット数 | 11種（T-1〜T-11） |
| 対象レース | 平地競走のみ（障害・海外レースは除外） |

---

## 2. 技術スタック

| レイヤー | 技術 | 備考 |
|---|---|---|
| データ収集 | Python スクレイパー（netkeiba.com専用） | リクエスト間隔2秒＋ジッター、並列数1 |
| データベース | PostgreSQL（Layer 1〜5） | Alembic によるDDLマイグレーション管理 |
| キャッシュ | Redis | TTL: 発走まで有効 / 発走後60秒で自動失効 |
| ML フレームワーク | LightGBM（主力）、LSTM（Phase 4 拡張） | Stage 1: LightGBM、Stage 2: LightGBM→LSTM |
| モデル管理 | MLflow（バージョン管理・ロールバック） | — |
| API | REST API（`/api/v1/`） | — |
| フロントエンド | UI（レース一覧・出馬表・AI予測統合表示） | — |
| スキーマ管理 | Alembic | 全DDL変更をバージョン管理 |

### スクレイピング設定

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

## 3. アーキテクチャ設計

### 3-1. データ層アーキテクチャ（5層構造）

```
Layer 1 — レース基本情報（静的マスター）
  └─ races, entries, horses, jockeys, trainers, courses

Layer 2 — 個別出走成績（確定結果・追記のみ）
  └─ race_results（着順・タイム・馬体重・コーナー通過順）

Layer 3 — 集計特徴量スナップショット（追記型・不変）
  └─ horse_stats_snapshot, jockey_stats_snapshot, trainer_stats_snapshot
     ※ UNIQUE(entity_id, as_of_race_id) で時点を固定

Layer 4 — ラップ・ペース・通過順位（確定後追記）
  └─ race_lap_times, race_corner_positions, race_pace_summary

Layer 5 — オッズスナップショット（時系列追記型）
  └─ race_odds_snapshot（snapshot_at 付き、削除不可）
```

**特徴量リーク防止の原則**: Layer 3 の集計値は必ず `as_of_race_id`（予測対象レース）に紐付けて保存し、そのレース以後の情報は含めない。

### 3-2. MLパイプライン概要

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: 共有表現 マルチタスクモデル                   │
│  入力: Layer 1〜3 特徴量（馬×レース単位）              │
│  Shared Encoder (LightGBM)                          │
│       ├── [Head A] 勝率/連対率/複勝率（分類）          │
│       ├── [Head B] ポジション予測（LambdaMART）        │
│       └── [Head C] オッズ予測（回帰）                  │
└──────────────────── ↓ ポジション予測値を受け渡し ─────┘
┌─────────────────────────────────────────────────────┐
│ Stage 2: ラップ・ペース予測モデル                      │
│  入力: Layer 4 + Stage 1 ポジション予測 + コース特徴量 │
│       ├── ペースカテゴリ (HIGH/MIDDLE/SLOW)           │
│       └── 1F毎ラップ予測値 (furlong_index別)          │
└─────────────────────────────────────────────────────┘
```

### 3-3. 主要テーブルスキーマ

#### Layer 3: `horse_stats_snapshot`

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

#### Layer 4: ラップ・ペース・コーナー

```sql
CREATE TABLE race_lap_times (
    race_id          VARCHAR(20)   NOT NULL,
    furlong_index    SMALLINT      NOT NULL,
    lap_time_sec     NUMERIC(4,2)  NOT NULL,
    cumulative_sec   NUMERIC(6,2),
    PRIMARY KEY (race_id, furlong_index)
);

CREATE TABLE race_corner_positions (
    race_id    VARCHAR(20)   NOT NULL,
    horse_id   VARCHAR(20)   NOT NULL,
    corner_1   SMALLINT,
    corner_2   SMALLINT,
    corner_3   SMALLINT,
    corner_4   SMALLINT,
    PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE race_pace_summary (
    race_id            VARCHAR(20)   NOT NULL PRIMARY KEY,
    first_3f_sec       NUMERIC(5,2),
    last_3f_sec        NUMERIC(5,2),
    pace_category      VARCHAR(10)   CHECK (pace_category IN ('HIGH','MIDDLE','SLOW')),
    front_runner_count SMALLINT,
    created_at         TIMESTAMPTZ   DEFAULT NOW()
);
```

#### Layer 5: `race_odds_snapshot`

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

#### 推論結果保存テーブル

```sql
CREATE TABLE prediction_results (
    prediction_id          BIGSERIAL     PRIMARY KEY,
    race_id                VARCHAR(20)   NOT NULL,
    horse_id               VARCHAR(20)   NOT NULL,
    model_version          VARCHAR(50)   NOT NULL,
    predicted_at           TIMESTAMPTZ   DEFAULT NOW(),
    win_prob               NUMERIC(5,4),
    place_prob             NUMERIC(5,4),
    show_prob              NUMERIC(5,4),
    predicted_win_odds     NUMERIC(7,1),
    predicted_place_odds   NUMERIC(7,1),
    expected_win_roi       NUMERIC(7,2),
    expected_show_roi      NUMERIC(7,2),
    predicted_position     SMALLINT,
    predicted_running_style VARCHAR(10),
    UNIQUE (race_id, horse_id, model_version)
);

CREATE TABLE prediction_lap_times (
    race_id              VARCHAR(20)   NOT NULL,
    model_version        VARCHAR(50)   NOT NULL,
    furlong_index        SMALLINT      NOT NULL,
    predicted_lap_sec    NUMERIC(4,2),
    predicted_pace_cat   VARCHAR(10)   CHECK (predicted_pace_cat IN ('HIGH','MIDDLE','SLOW')),
    PRIMARY KEY (race_id, model_version, furlong_index)
);
```

### 3-4. スクレイピング収集スケジュール

| ジョブ | トリガー | 頻度 |
|---|---|---|
| race_card（出馬表） | レース3日前 06:00 JST | 毎日 06:00 更新（発走まで） |
| odds_snapshot | 発走当日 08:00〜発走時刻 | 5分毎（発走30分前: 2分毎、発走5分前: 1分毎） |
| results（結果・ラップ） | 発走予定時刻 + 35分 | リトライ: 5分間隔 × 最大6回 |
| horse_history | results 収集完了後 | results 完了トリガー |

---

## 4. 機能要件（確定版）

| # | 要件 | 優先度 |
|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 |
| F-4 | オッズを指定スケジュール（発走前5分毎〜1分毎）でスナップショット取得し Layer 5 に格納する | 高 |
| F-5 | スクレイプ実行ログ（target_type・status・retry_count）を `scrape_runs` テーブルで管理する | 高 |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 |
| F-11 | ラップ予測結果を系列形式（furlong_index 順）で API から提供する | 中 |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 |
| F-13 | レース一覧・出馬表・AI予測を統合表示する UI を提供する | 中 |
| F-14 | 回収率100以上の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能を提供する | 中 |
| F-17 | ラップデータ可用性の事前検証（サンプル10レースで手動確認）を実施する【Phase 0 前提条件】 | 高 |

---

## 5. 非機能要件（確定版）

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
| N-10 | 特徴量リーク（未来情報混入）ゼロ | テストデータ時系列分割で検証、CI で自動実行 |
| N-11 | DDL マイグレーション管理（Alembic） | 全スキーマ変更をバージョン管理 |
| N-12 | Redis キャッシュ TTL | 発走まで有効 / 発走後60秒で自動失効 |
| N-13 | テストカバレッジ（スクレイパー・特徴量パイプライン） | ≥ 80% |
| N-14 | 障害レース・海外レースは予測対象外として明示除外 | フラグ管理 |

---

## 6. AI / ML パイプライン

### 6-1. 予測ターゲット定義（T-1〜T-11）

| ID | ターゲット | 問題設定 | 出力型 |
|---|---|---|---|
| T-1 | 勝率 `win_prob` | 多クラス分類（レース内1頭が1着） | NUMERIC(5,4) |
| T-2 | 連対率 `place_prob` | バイナリ分類（2着以内）× 頭数 | NUMERIC(5,4) |
| T-3 | 複勝率 `show_prob` | バイナリ分類（3着以内）× 頭数 | NUMERIC(5,4) |
| T-4 | 単勝オッズ予測 `predicted_win_odds` | 回帰 | NUMERIC(7,1) |
| T-5 | 複勝オッズ予測 `predicted_place_odds` | 回帰 | NUMERIC(7,1) |
| T-6 | 単回収率 `win_roi` | 計算値: `win_prob × predicted_win_odds × 100` | NUMERIC(7,2) |
| T-7 | 複回収率 `show_roi` | 計算値: `show_prob × predicted_place_odds × 100` | NUMERIC(7,2) |
| T-8 | ポジション予測 `predicted_position` | 順位回帰 / ランキング学習 | SMALLINT |
| T-9 | 脚質予測 `predicted_running_style` | 4値分類: FRONT/STALKER/MID/CLOSER | VARCHAR(10) |
| T-10 | ペースカテゴリ予測 `pace_category` | 3値分類: HIGH/MIDDLE/SLOW | VARCHAR(10) |
| T-11 | 1F単位ラップ予測 `predicted_lap_sec[]` | 時系列回帰（系列出力） | NUMERIC(4,2)[] |

> T-6・T-7（回収率）はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100以上 = バリューベット候補。

### 6-2. モデル選定

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率・連対率・複勝率（T-1〜T-3） | LightGBM (binary/softmax) | 表形式データに最強、欠損耐性が高い |
| ポジション予測（T-8） | LambdaMART (LightGBM ranker) | 相対順位を直接最適化できる |
| オッズ予測（T-4〜T-5） | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ（T-10） | LightGBM multiclass | 3クラス・解釈性を重視 |
| 1F毎ラップ予測（T-11） | LightGBM per-furlong（初期）→ LSTM（Phase 4） | まず解釈しやすい単独回帰から開始し、MAE ≤ 0.3秒 未達の場合 LSTM へ移行 |

### 6-3. 主要特徴量

#### 基本特徴量（Layer 1〜2 由来）

`distance`・`surface`・`direction`・`going`・`weather`・`grade`・`horse_num`・`frame_no`・`post_no`・`weight_carried`・`horse_weight`・`horse_weight_diff`・`days_since_last`・`horse_age`・`sex`

#### 集計特徴量（Layer 3 由来）

`win_rate_all / place_rate_all / show_rate_all`・`win_rate_distance / win_rate_course / win_rate_going`・`avg_last_3f / speed_index_avg / speed_index_max`・`running_style_score`・`jockey.win_rate_all`・`trainer.win_rate_all`

#### クロス・相対特徴量（前処理で生成）

```python
df["style_x_straight"]  = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"]  = df["running_style_score"] * df["distance_category_encoded"]
df["front_runner_count"] = df.groupby("race_id")["running_style_score"] \
                             .transform(lambda x: (x < -2).sum())
df["rel_speed_index"] = df["speed_index_avg"] / \
    df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - \
    df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"] = df.groupby("race_id")["odds_value"].rank(ascending=True)
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]) \
    .apply(lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 6-4. 回収率計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float,
    win_odds: float,
    show_prob: float,
    place_odds_mid: float,
) -> dict:
    win_roi  = win_prob  * win_odds       * 100   # T-6
    show_roi = show_prob * place_odds_mid * 100   # T-7
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}