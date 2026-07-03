# AREA-01 — アプリケーション要件
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001 (統合済み・削除), TASK-046 (データ分析機能要件定義書 統合済み)

---

## 0. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、
データ基盤・モデリング・評価・API 設計の全工程の前提条件となる。

---

## 1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F 単位ラップ予測** を実現する競馬予測 Web アプリ。加えて、**ユーザーが予想の根拠を自ら探索・検証するためのデータ分析機能**（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB — 2GB 以内での安定稼働を最優先 |

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
| T-11 | 1F 単位ラップ予測 (`predicted_lap_sec[]`) | 時系列回帰（系列出力） | `NUMERIC(4,2)[]` |

> T-6・T-7（回収率）はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100 以上 = 期待値プラスのバリューベット候補。

---

## 3. データ要件

### 3-1. データソース（netkeiba.com）

| データ種別 | URL パターン | 更新タイミング |
|---|---|---|
| レース基本情報・出馬表 | `/race/shutuba/{race_id}/` | レース3日前〜 |
| レース結果・ラップ・コーナー通過 | `/race/{race_id}/` | 発走後 約30分 |
| 馬の過去成績 | `/horse/{horse_id}/` | 結果確定後30分 |
| 騎手成績 | `/jockey/{jockey_id}/` | 結果確定後30分 |
| 調教師成績 | `/trainer/{trainer_id}/` | 結果確定後30分 |
| 単勝・複勝オッズ | `/odds/{race_id}/` | 発走当日〜数分毎 |

### 3-2. データ層アーキテクチャ（5層構造）

```
Layer 1 — レース基本情報（静的マスター）
  └─ races, entries, horses, jockeys, trainers, courses, sires

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

### 3-3. テーブルスキーマ定義

#### Layer 1 追加: `sires`（種牡馬マスター）

```sql
CREATE TABLE sires (
    sire_id    VARCHAR(20)   PRIMARY KEY,
    sire_name  VARCHAR(100)  NOT NULL,
    sire_line  VARCHAR(50),
    created_at TIMESTAMPTZ   DEFAULT NOW()
);

-- horses テーブルへの FK 追加
ALTER TABLE horses ADD COLUMN sire_id VARCHAR(20) REFERENCES sires(sire_id);
CREATE INDEX idx_horses_sire_id ON horses (sire_id);
```

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
    pace_category      VARCHAR(10)
                       CHECK (pace_category IN ('HIGH','MIDDLE','SLOW')),
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
    predicted_pace_cat   VARCHAR(10)
                         CHECK (predicted_pace_cat IN ('HIGH','MIDDLE','SLOW')),
    PRIMARY KEY (race_id, model_version, furlong_index)
);
```

#### 分析機能追加テーブル

```sql
-- コース統計事前集計キャッシュ
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

-- マイ分析（条件保存）
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

CREATE INDEX idx_saved_analyses_params
  ON saved_analyses USING gin(filter_conditions);
```

#### 分析機能必須インデックス

```sql
CREATE INDEX CONCURRENTLY idx_races_filter_axes
  ON races (course, surface, track_condition, class, distance, race_date);

CREATE INDEX CONCURRENTLY idx_results_race_finish
  ON results (race_id, finish_pos);

CREATE INDEX CONCURRENTLY idx_entries_horse_race
  ON entries (horse_id, race_id);
```

### 3-4. スクレイピング収集スケジュール

```yaml
race_card:
  trigger: "レース3日前 06:00 JST"
  refresh: "毎日 06:00（発走まで）"

odds_snapshot:
  trigger: "発走当日 08:00〜発走時刻"
  interval: "5分毎"
  priority_windows:
    - "発走30分前: 2分毎"
    - "発走5分前: 1分毎"

results:
  trigger: "発走予定時刻 + 35分"
  retry: "5分間隔 × 最大6回"

horse_history:
  trigger: "results 収集完了後"
  note: "前走成績更新後に再取得（前走情報が変化するため）"
```

### 3-5. スクレイピング設定（netkeiba.com 向け）

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

## 4. モデリング要件概要

> 詳細 → **[AREA-07-modeling.md](AREA-07-modeling.md)**

### 4-1. 2ステージ構成

```
Stage 1: 共有表現 マルチタスクモデル
  入力: Layer 1〜3 特徴量（馬×レース単位）
  出力: Head A（勝率/連対率/複勝率）/ Head B（ポジション）/ Head C（オッズ）

Stage 2: ラップ・ペース予測モデル
  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量
  出力: ペースカテゴリ (HIGH/MIDDLE/SLOW) + 1F毎ラップ予測値
```

### 4-2. モデル選定

| ターゲット | アルゴリズム |
|---|---|
| 勝率・連対率・複勝率 | LightGBM (binary/softmax) |
| ポジション予測 | LambdaMART (LightGBM ranker) |
| オッズ予測 | LightGBM regression |
| ペースカテゴリ | LightGBM multiclass |
| 1F 毎ラップ予測 | LightGBM per-furlong（初期）→ LSTM（拡張） |

### 4-3. 回収率計算ロジック

```python
def calculate_recovery_rate(win_prob, win_odds, show_prob, place_odds_mid):
    win_roi  = win_prob  * win_odds        * 100   # T-6
    show_roi = show_prob * place_odds_mid  * 100   # T-7
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}
```

### 4-4. テンポラルリーク防止ルール

1. 訓練データの時系列分割：常に過去レースで学習 → 未来レースで評価（ランダムシャッフル禁止）
2. スナップショットの `as_of_race_id` 参照：推論時も同レース ID のスナップショットのみ使用
3. オッズ特徴量：推論時は「発走 N 分前の最終スナップショット」を固定使用
4. 馬体重・馬場状態：レース当日の実測値を使用（出馬表確定後）

### 4-5. MLパイプラインと分析バッチの時点整合性分離

| 用途 | as_of 制約 | 集計範囲 | 理由 |
|---|---|---|---|
| AI予測モデル特徴量 | **必須**（`race_date < as_of_race_id`） | 予測時点以前のみ | テンポラルリーク防止 |
| ユーザー向け分析UI | **不要** | 全期間またはUI選択 | 事後統計であり未来情報混入の問題なし |

> **UIへの注記要件**: 分析画面に「※ この統計はリアルタイム集計です。AIモデルが予測に使用した時点の特徴量とは異なる場合があります。」を表示すること。

---

## 5. 機能要件

### 5-1. 予測機能

| # | 要件 | 優先度 | 担当 |
|---|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 | data-engineer |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 | data-engineer |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 | data-engineer |
| F-4 | オッズを指定スケジュール（発走前5分毎〜1分毎）でスナップショット取得し Layer 5 に格納する | 高 | data-engineer |
| F-5 | スクレイプ実行ログを `scrape_runs` テーブルで管理する | 高 | backend-engineer |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 | data-engineer / ai-model-engineer |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 | ai-model-engineer |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F 毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 | ai-model-engineer |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 | ai-model-engineer / backend-engineer |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 | backend-engineer |
| F-11 | ラップ予測結果を系列形式（`furlong_index` 順）で API から提供する | 中 | backend-engineer |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 | backend-engineer |
| F-13 | レース一覧・出馬表・AI 予測を統合表示する UI を提供する | 中 | frontend-engineer |
| F-14 | 回収率100以上の馬をバリューベット候補としてハイライト表示する | 中 | frontend-engineer |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 | frontend-engineer |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能 | 中 | ai-model-engineer / operations-engineer |
| F-17 | ラップデータ可用性の事前検証（サンプル10レースで手動確認） — Phase 0 前提条件 | 高 | data-engineer |

### 5-2. データ分析機能

| # | 機能名 | 優先度 | 担当 |
|---|---|---|---|
| AN-01 | 種牡馬成績多軸フィルタリング分析 | 高 | data-engineer / backend-engineer / frontend-engineer |
| AN-02 | コース別・条件別統計