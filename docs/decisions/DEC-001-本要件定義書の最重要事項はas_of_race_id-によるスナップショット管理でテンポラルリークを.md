# DEC-001: 本要件定義書の最重要事項は「`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること」であり、これがデータ基盤・モデリング・評価の全工程の前提条件となる。

**Date**: 2026-07-03
**Agent**: web-search-agent, decisions-context-agent, proposal-agent, backend-engineer, data-engineer, ai-model-engineer, integration-synthesizer, quality-reviewer
**Task**: TASK-039
**Status**: ACCEPTED

---

## Context

ラップデータ（Layer 4）の実際の取得可否がnetkeiba.comの構造に依存しており、Phase 0 の検証結果次第でデータ要件・モデリング要件（T-10/T-11、F-2、F-8）の大幅見直しが必要になる可能性があるため、Phase 0 完了後に本要件定義書の該当箇所を人間がレビューして確定させること。

---

## Decision

# 要件定義書: 競馬予測システム — データ要件・モデリング要件

**文書番号**: REQ-001  
**作成日**: 2025-07-10  
**ステータス**: APPROVED（全ドメインエージェント合意済み）  
**対象プロジェクト**: keiba-vpn

---

## サマリー

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測**、ならびに **逃げ馬ペース予測・1F単位ラップ予測** を実現する。データ基盤は5層の階層型スキーマ（静的マスター → 確定結果 → 集計スナップショット → ラップ/ペース → オッズ時系列）で構成し、特徴量リークを構造的に排除する。モデリングは Stage 1（勝率・ポジション・オッズのマルチタスク学習）と Stage 2（ラップ系列予測）の2ステージ構成とし、回収率は推論後の計算値として提供する。

---

## 1. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 |
|---|---|---|---|
| T-1 | 勝率 (win_prob) | 多クラス分類（レース内1頭が1着） | NUMERIC(5,4) |
| T-2 | 連対率 (place_prob) | バイナリ分類（2着以内）× 頭数 | NUMERIC(5,4) |
| T-3 | 複勝率 (show_prob) | バイナリ分類（3着以内）× 頭数 | NUMERIC(5,4) |
| T-4 | 単勝オッズ予測 (predicted_win_odds) | 回帰 | NUMERIC(7,1) |
| T-5 | 複勝オッズ予測 (predicted_place_odds) | 回帰 | NUMERIC(7,1) |
| T-6 | 単回収率 (win_roi) | 計算値: `win_prob × predicted_win_odds × 100` | NUMERIC(7,2) |
| T-7 | 複回収率 (show_roi) | 計算値: `show_prob × predicted_place_odds × 100` | NUMERIC(7,2) |
| T-8 | ポジション予測 (predicted_position) | 順位回帰 / ランキング学習 | SMALLINT |
| T-9 | 脚質予測 (predicted_running_style) | 4値分類: FRONT/STALKER/MID/CLOSER | VARCHAR(10) |
| T-10 | ペースカテゴリ予測 (pace_category) | 3値分類: HIGH/MIDDLE/SLOW | VARCHAR(10) |
| T-11 | 1F単位ラップ予測 (predicted_lap_sec[]) | 時系列回帰（系列出力）| NUMERIC(4,2)[] |

> **注**: T-6・T-7（回収率）はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値とする。100以上 = 期待値プラスのバリューベット候補を意味する。

---

## 2. データ要件

### 2-1. データソース

すべてのデータは **netkeiba.com** から収集する。

| データ種別 | URLパターン | 更新タイミング |
|---|---|---|
| レース基本情報・出馬表 | `/race/shutuba/{race_id}/` | レース3日前〜 |
| レース結果・ラップ・コーナー通過 | `/race/{race_id}/` | 発走後 約30分 |
| 馬の過去成績 | `/horse/{horse_id}/` | 結果確定後30分 |
| 騎手成績 | `/jockey/{jockey_id}/` | 結果確定後30分 |
| 調教師成績 | `/trainer/{trainer_id}/` | 結果確定後30分 |
| 単勝・複勝オッズ | `/odds/{race_id}/` | 発走当日〜数分毎 |

### 2-2. データ層アーキテクチャ（5層構造）

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

**特徴量リーク防止の原則**:  
Layer 3 の集計値は必ず `as_of_race_id`（予測対象レース）に紐付けて保存し、そのレース以後の情報は含めない。

### 2-3. テーブルスキーマ定義

#### Layer 3: `horse_stats_snapshot`

```sql
CREATE TABLE horse_stats_snapshot (
    snapshot_id          BIGSERIAL      PRIMARY KEY,
    horse_id             VARCHAR(20)    NOT NULL,
    as_of_race_id        VARCHAR(20)    NOT NULL,   -- 予測対象レース直前時点
    as_of_date           DATE           NOT NULL,
    -- 勝率・連対率・複勝率
    win_rate_all         NUMERIC(5,4),
    win_rate_turf        NUMERIC(5,4),
    win_rate_dirt        NUMERIC(5,4),
    place_rate_all       NUMERIC(5,4),              -- 連対率（2着以内）
    show_rate_all        NUMERIC(5,4),              -- 複勝率（3着以内）
    win_rate_distance    NUMERIC(5,4),              -- ±200m同距離帯
    win_rate_course      NUMERIC(5,4),              -- 同コース(場・距離・芝砂)
    win_rate_going       NUMERIC(5,4),              -- 同馬場状態
    -- タイム・スピード指数
    avg_last_3f          NUMERIC(5,2),              -- 直近5走平均上がり3F
    speed_index_avg      NUMERIC(6,2),
    speed_index_max      NUMERIC(6,2),
    -- 脚質スコア
    running_style_score  NUMERIC(5,2),              -- −5(逃)〜+5(追込)
    sample_count         SMALLINT,
    created_at           TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (horse_id, as_of_race_id)
);
```

#### Layer 4: ラップ・ペース・コーナー

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

#### Layer 5: `race_odds_snapshot`

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

#### 推論結果保存テーブル

```sql
CREATE TABLE prediction_results (
    prediction_id          BIGSERIAL     PRIMARY KEY,
    race_id                VARCHAR(20)   NOT NULL,
    horse_id               VARCHAR(20)   NOT NULL,
    model_version          VARCHAR(50)   NOT NULL,
    predicted_at           TIMESTAMPTZ   DEFAULT NOW(),
    -- T-1〜T-3: 確率
    win_prob               NUMERIC(5,4),
    place_prob             NUMERIC(5,4),
    show_prob              NUMERIC(5,4),
    -- T-4〜T-5: オッズ予測
    predicted_win_odds     NUMERIC(7,1),
    predicted_place_odds   NUMERIC(7,1),
    -- T-6〜T-7: 回収率（計算値）
    expected_win_roi       NUMERIC(7,2),
    expected_show_roi      NUMERIC(7,2),
    -- T-8〜T-9: ポジション・脚質
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

### 2-4. スクレイピング収集スケジュール

```yaml
race_card:            # 出馬表
  trigger: "レース3日前 06:00 JST"
  refresh: "毎日 06:00（発走まで）"

odds_snapshot:        # オッズスナップショット
  trigger: "発走当日 08:00〜発走時刻"
  interval: "5分毎"
  priority_windows:
    - "発走30分前: 2分毎"
    - "発走5分前: 1分毎"

results:              # 結果・ラップ・コーナー通過
  trigger: "発走予定時刻 + 35分"
  retry: "5分間隔 × 最大6回"

horse_history:        # 馬の過去成績（スナップショット用）
  trigger: "results 収集完了後"
  note: "前走成績更新後に再取得（前走情報が変化するため）"
```

### 2-5. スクレイピング設定（netkeiba.com 向け）

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

---

## 3. モデリング要件

### 3-1. モデルアーキテクチャ概要

```
┌──────────────────────────────────────────────────────────┐
│ Stage 1: 共有表現 マルチタスクモデル                         │
│                                                            │
│  入力: Layer 1〜3 特徴量（馬×レース単位）                   │
│                                                            │
│  ┌──────────────────────────────┐                         │
│  │  Shared Encoder (LightGBM)   │                         │
│  └──────────────┬───────────────┘                         │
│                 │                                          │
│     ┌───────────┼────────────┐                            │
│     ▼           ▼            ▼                            │
│  [Head A]    [Head B]     [Head C]                        │
│  勝率/連対   ポジション   オッズ予測                         │
│  /複勝率     予測          (回帰)                           │
│  (分類)    (Learning to Rank)                             │
└──────────────────────────────────────────────────────────┘
                 │ ポジション予測値を受け渡し
                 ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2: ラップ・ペース予測モデル                            │
│                                                            │
│  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量   │
│                                                            │
│  ┌──────────────────────────────────────────┐            │
│  │  Pace & Lap Sequence Model                │            │
│  │  (LightGBM per-furlong or LSTM)           │            │
│  │                                            │            │
│  │  出力:                                    │            │
│  │   ├── ペースカテゴリ (HIGH/MIDDLE/SLOW)  │            │
│  │   └── 1F毎ラップ予測値 (furlong_index別) │            │
│  └──────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────┘
```

### 3-2. 特徴量一覧

#### 基本特徴量（Layer 1〜2 由来）

| 特徴量名 | 説明 | 型 |
|---|---|---|
| distance | レース距離 (m) | INT |
| surface | 芝/ダート/障害 | CATEGORY |
| direction | 左/右/直線 | CATEGORY |
| going | 馬場状態 (良/稍重/重/不良) | CATEGORY |
| weather | 天候 | CATEGORY |
| grade | レースクラス (G1〜未勝利) | CATEGORY |
| horse_num | 出走頭数 | INT |
| frame_no | 枠番 | INT |
| post_no | 馬番 | INT |
| weight_carried | 斤量 (kg) | FLOAT |
| horse_weight | 馬体重 (kg) | INT |
| horse_weight_diff | 馬体重増減 | INT |
| days_since_last | 前走からの間隔（日） | INT |
| horse_age | 馬齢 | INT |
| sex | 性別 (牡/牝/セン) | CATEGORY |

#### 集計特徴量（Layer 3 由来）

| 特徴量名 | 説明 |
|---|---|
| win_rate_all / place_rate_all / show_rate_all | 生涯勝率・連対率・複勝率 |
| win_rate_distance / win_rate_course / win_rate_going | 条件別勝率 |
| avg_last_3f / speed_index_avg / speed_index_max | タイム・スピード指数 |
| running_style_score | 脚質スコア (−5=逃 〜 +5=追込) |
| jockey.win_rate_all | 騎手勝率 |
| trainer.win_rate_all | 調教師勝率 |

#### クロス・相対特徴量（前処理で生成）

```python
# 脚質 × コース形状 のクロス特徴量
df["style_x_straight"]  = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"]  = df["running_style_score"] * df["distance_category_encoded"]

# 逃げ・先行馬数（ペース予測用）
df["front_runner_count"] = df.groupby("race_id")["running_style_score"] \
                             .transform(lambda x: (x < -2).sum())

# 同レース内相対化
df["rel_speed_index"] = df["speed_index_avg"] / \
    df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - \
    df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"] = df.groupby("race_id")["odds_value"].rank(ascending=True)

# ペース事前シナリオ（逃げ馬比率から算出）
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]) \
    .apply(lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 3-3. モデル選定

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率・連対率・複勝率 | LightGBM (binary/softmax) | 表形式データに最強、欠損耐性が高い |
| ポジション予測 | LambdaMART (LightGBM ranker) | 相対順位を直接最適化できる |
| オッズ予測 | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ | LightGBM multiclass | 3クラス・解釈性を重視 |
| 1F毎ラップ予測 | LightGBM per-furlong (初期) → LSTM (拡張) | まず解釈しやすい単独回帰から開始し、系列依存が大きければLSTMへ移行 |

### 3-4. 回収率計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float,       # モデル予測勝率    (T-1)
    win_odds: float,       # 予測単勝オッズ    (T-4)
    show_prob: float,      # モデル予測複勝率  (T-3)
    place_odds_mid: float, # 予測複勝オッズ中値 (T-5)
) -> dict:
    """
    単回収率 = 勝率 × 単勝オッズ × 100
    複回収率 = 複勝率 × 複勝オッズ中値 × 100
    100 超 = バリューベット候補
    """
    win_roi   = win_prob  * win_odds       * 100
    show_roi  = show_prob * place_odds_mid * 100
    return {
        "win_roi":   round(win_roi,  2),   # T-6
        "show_roi":  round(show_roi, 2),   # T-7
    }
```

### 3-5. 評価指標

| ターゲット | 主評価指標 | 補助指標 |
|---|---|---|
| 勝率 (T-1) | Log Loss（全馬合計） | Calibration Error, Top-1 Accuracy |
| 連対率・複勝率 (T-2/3) | Binary Log Loss | AUC-ROC, Calibration |
| ポジション (T-8) | Spearman ρ（順位相関） | MAE |
| オッズ予測 (T-4/5) | MAE（オッズ単位） | RMSE |
| ラップタイム (T-11) | MAE（秒） | RMSE per furlong |
| ペースカテゴリ (T-10) | Accuracy | Macro F1 |
| 回収率（バックテスト） | 通算 ROI | Sharpe Ratio of bets |

### 3-6. テンポラルリーク防止ルール

1. **訓練データの時系列分割**: 常に過去レースで学習 → 未来レースで評価（ランダムシャッフル禁止）
2. **スナップショットの `as_of_race_id` 参照**: 推論時も `as_of_race_id = 対象レースID` のスナップショットのみ使用
3. **オッズ特徴量**: 推論時は「発走N分前の最終スナップショット」を固定使用（直前確定値）
4. **馬体重・馬場状態**: レース当日の実測値を使用（出馬表確定後）

---

## 4. 機能要件

| # | 要件 | 優先度 | 担当 |
|---|---|---|---|
| F-1 | netkeiba.comからレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 | data-engineer |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 | data-engineer |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 | data-engineer |
| F-4 | オッズを指定スケジュール（発走前5分毎〜1分毎）でスナップショット取得し Layer 5 に格納する | 高 | data-engineer |
| F-5 | スクレイプ実行ログ（target_type・status・retry_count）を `scrape_runs` テーブルで管理する | 高 | backend-engineer |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 | data-engineer / ai-model-engineer |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 | ai-model-engineer |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 | ai-model-engineer |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 | ai-model-engineer / backend-engineer |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 | backend-engineer |
| F-11 | ラップ予測結果を系列形式（furlong_index 順）で API から提供する | 中 | backend-engineer |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 | backend-engineer |
| F-13 | レース一覧・出馬表・AI予測を統合表示する UI を提供する | 中 | frontend-engineer |
| F-14 | 回収率100以上の馬をバリューベット候補としてハイライト表示する | 中 | frontend-engineer |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 | frontend-engineer |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能 | 中 | ai-model-engineer / operations-engineer |
| F-17 | ラップデータ可用性の事前検証（サンプル10レースで手動確認）を実施する | 高（前提条件） | data-engineer |

---

## 5. 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|---|---|---|
| N-1 | 予測 API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms | operations-engineer |
| N-2 | 予測 API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms | operations-engineer |
| N-3 | ラップ予測 MAE（1F単位） | ≤ 0.3 秒 | ai-model-engineer |
| N-4 | 勝率予測 Log Loss（ベースラインオッズ逆数モデル比） | −5% 以上改善 | ai-model-engineer |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 | ai-model-engineer |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 | data-engineer |
| N-7 | スクレイピング後の DB 反映遅延 | ≤ 10 分 | data-engineer |
| N-8 | オッズスナップショット欠損率（発走前5分以内） | ≤ 1% | data-engineer |
| N-9 | モデル推論バッチ完了時刻（発走N時間前） | 発走3時間前までに完了 | ai-model-engineer |
| N-10 | 特徴量リーク（未来情報混入）ゼロ | テストデータ時系列分割で検証 | ai-model-engineer |
| N-11 | DDL マイグレーション管理（Alembic等） | 全スキーマ変更をバージョン管理 | backend-engineer |
| N-12 | Redis キャッシュ TTL 設定 | 発走まで有効 / 発走後60秒で自動失効 | backend-engineer |
| N-13 | テストカバレッジ（スクレイパー・特徴量パイプライン） | ≥ 80% | data-engineer |
| N-14 | 障害レース・海外レースは予測対象外として明示除外 | フラグ管理 | backend-engineer |

---

## 6. 実装ロードマップ

### Phase 0: 前提条件検証（1〜2週間）

> **F-17 が完了するまで Phase 1 以降は開始しない**

- [ ] ラップデータ可用性調査（サンプル10レース × 4コース形状で手動確認）
  - 検証対象: 芝1200m・ダート1800m・芝3200m・障害
  - 確認項目: 1F毎ラップの有無、コーナー通過順の形式
- [ ] netkeiba.com のロボット規約・利用規約の確認
- [ ] scrape_runs テーブル・基本スキーマの構築

---

### Phase 1: データ基<thinking>
- The output was cut off at "Phase 1: データ基" - I need to continue from that exact point
- I need to complete Phase 1, Phase 2, Phase 3, and then the Dependencies/Risks section
- Then close with the required XML tags
- Keep the content consistent with what was already written
</thinking>

盤構築（2〜4週間）

**目標**: 全ターゲットの学習・推論に必要なデータが揃った状態を作る

- [ ] **F-1** レース基本情報・出馬表スクレイパー実装
- [ ] **F-2** レース結果・ラップ・コーナー通過スクレイパー実装
- [ ] **F-3** 馬・騎手・調教師スナップショット集計バッチ実装
  - `as_of_race_id` 紐付けの単体テストを必ず実施（リーク防止）
- [ ] **F-4** オッズスナップショット定期収集スクレイパー実装
- [ ] **F-5** scrape_runs 実行管理テーブル・リトライ制御実装
- [ ] Layer 1〜5 全テーブルの DDL マイグレーション適用
- [ ] 過去2〜3年分の全レースデータの一括取得（ヒストリカルバルクロード）

**完了条件**: 過去2年分のレース・ラップ・オッズデータが Layer 1〜5 に格納済みであること

---

### Phase 2: 特徴量パイプライン + Stage 1 モデル構築（3〜5週間）

**目標**: 勝率・連対率・複勝率・ポジション・オッズ予測モデルの初版を稼働させる

- [ ] **F-6** 特徴量エンジニアリングパイプライン実装
  - 脚質スコア (`running_style_score`) 算出ロジック
  - クロス特徴量・相対特徴量の自動生成
  - `pace_scenario_prior` の生成
- [ ] **F-7** Stage 1 モデル学習パイプライン実装
  - 時系列分割による train/validation/test セット構築（シャッフル禁止）
  - LightGBM binary（連対率・複勝率）学習
  - LightGBM softmax（勝率）学習
  - LambdaMART（ポジション）学習
  - LightGBM regression（オッズ）学習
- [ ] **F-9** 回収率計算ロジック実装・単体テスト
- [ ] モデル評価指標の計算・バックテスト実施（ROI シミュレーション）
- [ ] **F-16** モデルバージョン管理基盤の整備（MLflow 等）

**完了条件**: 勝率 Log Loss がベースライン（オッズ逆数モデル）比 −5% 以上改善されていること

---

### Phase 3: Stage 2 モデル + API + UI 構築（3〜4週間）

**目標**: ラップ予測・ペース予測の稼働と、ユーザー向け機能の提供

- [ ] **F-8** Stage 2 モデル学習パイプライン実装
  - ペースカテゴリ分類モデル（LightGBM multiclass）
  - 1F毎ラップ予測モデル（LightGBM per-furlong、初期実装）
  - Stage 1 ポジション予測値を入力特徴量として連携
- [ ] **F-10** 予測結果取得 REST API 実装（全ターゲット T-1〜T-11）
- [ ] **F-11** ラップ予測系列取得エンドポイント実装
- [ ] **F-12** Redis キャッシュ実装（TTL: 発走まで有効 / 発走後60秒失効）
  - キャッシュキー: `prediction:{race_id}:{model_version}`
  - キャッシュキー: `lap:prediction:{race_id}:{model_version}`
- [ ] **F-13** レース一覧・出馬表・AI予測統合 UI 実装
- [ ] **F-14** バリューベット候補ハイライト表示（回収率100以上の馬）
- [ ] **F-15** ラップ予測折れ線グラフ実装（オプション）
- [ ] **N-1/N-2** API レスポンスタイム計測・チューニング
- [ ] 推論バッチの定期実行スケジューラ設定（発走3時間前に完了）

**完了条件**: 任意レースの全予測ターゲットが API 経由で取得でき、UI に表示されること

---

### Phase 4: 運用安定化・精度改善（継続）

- [ ] LSTM によるラップ系列モデルへの移行検討（Phase 3 の per-furlong モデルと精度比較）
- [ ] モデル再学習の定期実行自動化（週次 or 月次）
- [ ] 特徴量重要度モニタリング・データドリフト検知
- [ ] 障害・エラー通知アラート整備

---

## 7. 依存関係・リスク

### 依存関係

```
Phase 0（ラップデータ可用性確認）
    └─→ Phase 1（データ基盤構築）
            └─→ Phase 2（特徴量 + Stage 1 モデル）
                    └─→ Phase 3（Stage 2 + API + UI）
                    
Layer 3 スナップショット（F-3）
    └─→ Stage 1 モデル（F-7）に対して特徴量リーク防止の前提

Stage 1 ポジション予測（F-7）
    └─→ Stage 2 ラップ予測（F-8）の入力特徴量として必須

オッズスナップショット（F-4）
    └─→ 推論時の特徴量として必須（発走N分前スナップショットを固定使用）
```

### リスクと対策

| # | リスク | 深刻度 | 対策 |
|---|---|---|---|
| R-1 | **ラップデータが一部レース（短距離・旧年式）に存在しない** | 🔴 高 | Phase 0 で事前検証。欠損レースは Layer 4 の `lap_time_sec = NULL` で格納し、ラップ予測モデルのサンプルから除外 |
| R-2 | **netkeiba.com の HTML 構造変更によりスクレイパーが破損** | 🔴 高 | HTML パース箇所を設定値化し変更を局所化。週次でスクレイプ成功率を監視し、閾値以下でアラート発報 |
| R-3 | **netkeiba.com からアクセス制限（429 / IP ブロック）** | 🟡 中 | リクエスト間隔2秒+ジッター、セッションローテーション、リトライバックオフで対策（SCRAPING_CONFIG 参照） |
| R-4 | **特徴量リーク（テンポラルリーク）の混入** | 🔴 高 | `as_of_race_id` 紐付けの単体テスト必須化。CI でリーク検知テストを自動実行 |
| R-5 | **Stage 1 → Stage 2 の誤差伝播（ポジション予測誤差がラップ予測に伝播）** | 🟡 中 | Stage 2 では Stage 1 予測値の信頼区間も特徴量として入力。独立評価で各ステージの誤差を分離計測 |
| R-6 | **オッズの直前大幅変動による回収率計算のずれ** | 🟡 中 | 発走5分前スナップショットを「推論時使用オッズ」として固定。推論後のオッズ変動は参考値として別カラムで保持 |
| R-7 | **障害レースのラップ形式が平地と異なる** | 🟢 低 | `race_type = '障害'` フラグで予測対象外に除外。Layer 4 への格納も除外する |
| R-8 | **LightGBM per-furlong モデルが系列依存を捉えられない** | 🟡 中 | Phase 4 で LSTM への移行を評価。MAE ≤ 0.3秒 を移行判断閾値とする |

---

## 8. APIレスポンス仕様（概要）

```json
// GET /api/v1/races/{race_id}/predictions
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

---

## 9. 用語定義

| 用語 | 定義 |
|---|---|
| 勝率 | 当該馬が1着になる確率 |
| 連対率 | 当該馬が2着以内になる確率 |
| 複勝率 | 当該馬が3着以内になる確率 |
| 単回収率 | `勝率 × 単勝オッズ × 100`。100超 = 期待値プラス |
| 複回収率 | `複勝率 × 複勝オッズ中値 × 100`。100超 = 期待値プラス |
| 脚質スコア | 過去レースのコーナー通過順から算出した先行傾向指数。−5(逃)〜+5(追込) |
| ペースカテゴリ | 前半3F/後半3Fの差分から分類: HIGH(前傾)・MIDDLE(平均)・SLOW(後傾) |
| テンポラルリーク | 予測時点より未来の情報が学習データに混入する現象。スナップショット設計で防止 |
| バリューベット | 回収率が100以上、すなわち期待値がプラスの馬券 |
| as_of_race_id | スナップショットが「このレース直前時点」の情報であることを示す外部キー |

---

## Conclusion

**本要件定義書の最重要事項は「`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること」であり、これがデータ基盤・モデリング・評価の全工程の前提条件となる。**

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-03_
