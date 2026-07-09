# AREA-01 — アプリケーション要件
**Status**: FINAL | **Last Updated**: 2026-07-09 | **Consolidates**: DEC-001 (統合済み・削除), TASK-046 (データ分析機能要件定義書 統合済み), TASK-048 (Phase 0-S 追加・実装ロードマップ更新 統合済み)

---

## 0. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、
データ基盤・モデリング・評価・API 設計の全工程の前提条件となる。

---

## 1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F 単位ラップ予測** を実現する競馬予測 Web アプリ。加えて、**ユーザーが予想の根拠を自ら探索・検証するためのデータ分析機能**（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析・**めぐ指数**）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB — 2GB 以内での安定稼働を最優先 |
| データベース | PostgreSQL **15** |

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

### 3-1. データソース（実装準拠）

#### netkeiba.com（一次データソース）

| スクレイプカテゴリ | 内容 | 取得タイミング（SLA） | GCS 格納パス |
|---|---|---|---|
| `race_shutuba` | 出馬表（枠順・馬番・騎手・馬体重・オッズ） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_shutuba/{year}/{race_id}.json` |
| `race_shutuba_past` | 馬柱（各馬の過去成績テーブル・調教） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_shutuba_past/{year}/{race_id}.json` |
| `race_oikiri` | 追い切りデータ（日時・コース・タイム・印象） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/race_oikiri/{year}/{race_id}.json` |
| `race_result` | 確定結果（着順・タイム・馬体重・コーナー・払戻） | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_result/{year}/{race_id}.json` |
| `race_result_on_time` | 速報結果（発走後 T+15分） | T+15分（SLA 4） | `netkeiba/pc/race_result_on_time/{year}/{race_id}.json` |
| `race_result_lap` | ラップ詳細・ペース・コーナー通過順 | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_result_lap/{year}/{race_id}.json` |
| `race_index` | 速度指数（speed_max/avg/distance/course/recent） | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_index/{year}/{race_id}.json` |
| `race_odds` | 単複オッズ（win_odds・place_odds_min/max・人気） | T-15バンドル（SLA 3） | `netkeiba/pc/race_odds/{year}/{race_id}.json` |
| `race_pair_odds` | 連複オッズ（馬連・ワイド・馬単） | 当日 JST 17:30（SLA 5） | `netkeiba/pc/race_pair_odds/{year}/{race_id}.json` |
| `race_paddock` | パドック評価（rank・コメント） | T-15バンドル（SLA 3） | `netkeiba/pc/race_paddock/{year}/{race_id}.json` |
| `race_barometer` | バロメーター偏差値（total/start/chase/closing） | T-15バンドル（SLA 3） | `netkeiba/pc/race_barometer/{year}/{race_id}.json` |
| `race_detail` | レース総合詳細（shutuba+index+past 統合） | T-15バンドル（SLA 3） | `netkeiba/pc/race_detail/{year}/{race_id}.json` |
| `race_trainer_comment` | 調教師コメント | T-15バンドル（SLA 3） | `netkeiba/pc/race_trainer_comment/{year}/{race_id}.json` |
| `race_performance` | パイプライン生成パフォーマンス指数 | 計算後 | `netkeiba/pc/race_performance/{year}/{race_id}.json` |
| `horse_result` | 馬の全成績・プロフィール・重賞実績 | 週次（SLA 6）/ バックフィル | `netkeiba/pc/horse_result/{prefix}/{horse_id}.json` |
| `horse_pedigree_5gen` | 5世代血統（ancestors・sire/dam/dam_sire） | バックフィル horse フェーズ | `netkeiba/pc/horse_pedigree_5gen/{prefix}/{horse_id}.json` |
| `horse_training` | 馬調教履歴（日時・コース・条件・ライダー・時計） | 前日 JST 18:00（SLA 1） | `netkeiba/pc/horse_training/{prefix}/{horse_id}.json` |

> `{prefix}` = horse_id の先頭4桁

#### JRA 公式

| スクレイプカテゴリ | 内容 | 取得タイミング（SLA） | GCS 格納パス |
|---|---|---|---|
| `jra_cushion` | クッション値・含水率（JRA PDF ライブ集約） | JST 05:00-08:50 毎10分（SLA 2） | `others/jra_cushion/{year}.json` |

#### ローカルのみ（GCS 非使用）

| カテゴリ | 内容 | ローカルパス |
|---|---|---|
| `race_lists` | 開催日別レース一覧 | `data/page_reference/race_lists/{YYYYMMDD}.json` |
| `race_day_schedule` | 発走時刻スナップショット | `data/page_reference/race_day_schedule/{YYYYMMDD}.json` |

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

#### Layer 1 追加: `entries` テーブル拡張（枠番対応）

```sql
-- Phase 0-S で entries.post_position が馬番・枠番どちらかを確認後、必要に応じて追加
ALTER TABLE entries ADD COLUMN frame_no SMALLINT;  -- 枠番 (1-8)
-- ⚠️ JRA では「枠番（1-8）」≠「馬番（1-18）」。post_position が馬番なら frame_no は別カラムとして必須
CREATE INDEX idx_entries_frame_no ON entries (race_id, frame_no);
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
    win_roi                 NUMERIC(7,2),  -- 旧: expected_win_roi → archive §F に合わせ win_roi に統一
    show_roi                NUMERIC(7,2),  -- 旧: expected_show_roi → archive §F に合わせ show_roi に統一
    predicted_position      SMALLINT,
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

### 3-4. スクレイピング収集スケジュール（Cron SLA）

システムタイムゾーン UTC。crontime は UTC 記述。

| SLA | cron（UTC） | JST | タスク名 | 取得内容 |
|---|---|---|---|---|
| SLA 0 | `0 22 * * *` | 07:00 毎日 | `daily-race-lists` | 今日〜カレンダー末尾の全開催日 race_lists 取得・更新 |
| SLA 0 | `0 8 * * *` | 17:00 毎日 | `daily-race-lists` | 同上（夕方更新） |
| SLA 1 | `0 9 * * *` | 18:00 毎日 | `raceday-eve` | 翌開催日: race_shutuba + race_shutuba_past + race_oikiri + horse_training → 追走難度・最終オッズ precompute |
| SLA 2 | `*/10 20-23 * * *` | 05:00-08:50 毎10分 | `jra-baba-morning` | jra_cushion ポーリング（開催日のみ実取得） |
| SLA 3 | `30 22 * * *` | 07:30 開催日 | `raceday-runner` | 各レース T-15分: race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + JRA馬場ライブ → AI 予測トリガ |
| SLA 4 | `30 22 * * *` | 07:30 開催日 | `raceday-result-runner` | 各レース T+15分: race_result_on_time 速報取得 |
| SLA 5 | `30 8 * * *` | 17:30 毎日 | `raceday-evening` | race_result + race_result_lap + race_index + race_pair_odds → 馬場速度指数計算トリガ + **めぐ指数計算トリガ** |
| SLA 6 | `30 8 * * 5` | 17:30 金曜 | `weekly-update` | horse_result（先週分）・指数・偏差値・馬情報更新 |

**バックフィル（夜間バッチ）**:

| cron（UTC） | JST | 対象年 | フェーズ | 最大件数 |
|---|---|---|---|---|
| `0 15 * * *` | 00:00 毎日 | 過去全年 | horse_result / horse_pedigree_5gen バックフィル | 200件/日 |

---

## 4. めぐ指数（レースパフォーマンス評価指数）

本アプリ独自の競走馬パフォーマンス評価指数。詳細仕様は **[AREA-11-megu-index.md](AREA-11-megu-index.md)** を参照。

| 項目 | 内容 |
|---|---|
| 定義 | 1点 = 0.1秒差。ペース・馬場・斤量・レースレベルを統合回帰で補正した絶対指数 |
| 粒度 | 馬 × レース |
| 基準値 | 100 = 同条件（距離×コース×芝/ダート×馬場カテゴリ）の補正後平均 |
| 算出タイミング | SLA 5（確定結果取得後）に自動トリガー |
| GCS パス | `netkeiba/pc/race_performance/{year}/{race_id}.json`（既存パス活用） |
| DB テーブル | `megu_index`, `megu_par_time`, `megu_regression_params` |
| 派生機能 | 今週のめぐ指数（A: 最大値 / B: 加重平均〈デフォルト〉 / C: 条件絞り最大値） |

---

## 5. 非機能要件

| ID | カテゴリ | 要件 | 指標・基準 | 備考 |
|---|---|---|---|---|
| N-01 | リソース | メモリ使用量 | ConoHa VPS 2GB 以内で安定稼働 | スパイク含む |
| N-24 | AI 精度 | AI モデル 複勝的中率 | ≥ 40%（Phase 2 以降のデプロイ判断基準） | Phase 1 は動作確認優先のため精度ゲートなし |

> N-24 は DEC-024 で決定。Phase 2 以降の本番デプロイ可否は複勝的中率 40% を最低基準とし、それ未満のモデルはステージング留め。

---

## 6. 提供サービス一覧（ルートページ起点のエンドポイント）

ルートページ（`/`）のナビゲーションおよびトップダッシュボードカードから到達できる全エンドポイントを機能カテゴリ別に整理する。

> **表示制御の凡例**
> - 🌐 全ユーザー向け（ログイン不要を含む）
> - 🔐 管理者のみ（`is_admin === true` のときのみ表示）
> - 🛠️ 開発者のみ（`is_dev` フラグが true のときのみナビに表示）

---

### 6-1. AI 予測

| URL | ページ名 | ユーザーへの提供価値 | 表示 |
|---|---|---|---|
| `/weekly-predictions` | 今週のAI予測 | 今週開催レースのAI推奨印（◎○▲△☆）を一覧表示。出馬表・オッズ・個別レース詳細へ遷移できる。今週注目すべき馬をすぐ確認するメインの入口。 | 🌐 |
| `/race/{race_id}` | レース詳細 | race_id を指定して、そのレースの出馬表・AI予測・オッズ・レース結果を詳細表示。勝率・連対率・複勝率・回収率・ポジション予測を全頭分一覧できる。 | 🌐 |
| `/races` | 過去の予測結果 | 過去レースの出馬表・確定着順とAI予測の照合結果を閲覧。モデルの精度感覚をユーザーが自ら確認できる。 | 🌐 |
| `/tracking-difficulty` | 追走難度分析 | ゲート位置・隣枠の馬の脚質・フィールド全体の先行意識から、各馬の序盤ポジション取得容易度（追走難度スコア）をMLモデルで推定する。ペース予想・位置取り分布も同時提供。 | 🌐 |
| `/race-quality` | レース質分析 | レースのペース傾向・展開パターンを統計的に分析する。どのような質のレースになりやすいかを事前に把握できる。 | 🌐 |
| `/pedigree-race-stats` | 今週の血統傾向分析 | 予測されたレース質と出走馬の血統を照合し、各馬の血統適性スコアを点数化する。対象レースでどれだけ血統的に優位かを可視化。 | 🌐 |

---

### 6-2. 血統分析

| URL | ページ名 | ユーザーへの提供価値 | 表示 |
|---|---|---|---|
| `/bloodline` | 血統 × 距離・コース研究 | 種牡馬・母父の血統が距離適性・コース特性（坂/直線/枠順/馬場/芝種）にどう作用するかを事前計算済みアーティファクトで閲覧できる。特定血統の「得意条件・苦手条件」を定量的に把握できる。 | 🌐 |
| `/bloodline-vector` | 血統マップ | 種牡馬を血統ベクトルで数値化し、2D マップ上に配置して類似性を可視化する。近い位置の種牡馬は傾向が似ており、代替産駒の探索などに使える。 | 🌐 |
| `/note-aptitude-race` | 血統適性マップ | 特定レースの馬ごとの血統適性を SVG パン/ズームマップで可視化する。L2クラスター・コース適合スコアを提示。 | 🌐 |
| `/stallion-notes` | 種牡馬メモ | 種牡馬・牝系の血統ドメイン知識ベース。特徴・コース適性・配合傾向を整理。一般ユーザーは種牡馬・牝系カテゴリのみ閲覧可能。 | 🌐（種牡馬・牝系のみ） / 🛠️（全カテゴリ＋編集） |
| `/myostatin` | MSTN遺伝子適性 | マイオスタチン遺伝子型（C:C / C:T / T:T）の予測と距離適性の関係を示す。短距離向き・長距離向きの遺伝的傾向を参照できる。 | 🛠️ |

> **廃止済み（2026-07-08）**: `/pedigree-map`（血統構造マップ、D3.js サイアー系図）および `/bloodline-cluster`（メタクラスタ判定）は UI として不要と判断し削除。対応ページディレクトリも削除済み。

---

### 6-3. データ分析

| URL | ページ名 | ユーザーへの提供価値 | 表示 |
|---|---|---|---|
| `/data-analysis` | 詳細データ分析 | 着順・タイム・AI予測・ROI など複数指標を自由に組み合わせて集計・可視化する Target Frontier スタイルの分析ツール。分布・散布・ランキング・時系列の4モード対応。PostgreSQL からリアルタイム集計。 | 🌐 |
| `/growth-curve` | 成長曲線 | 馬ごとの年齢別パフォーマンス推移を成長曲線として可視化する。早熟・晩成の傾向を把握でき、今後の成長余地を評価する材料になる。 | 🌐 |
| `/track-speed` | 馬場速度 | 開催日・競馬場・馬場種別ごとの馬場速度指数（TSI）を集計して表示する。当日の馬場バイアスや「速い馬場 / 時計のかかる馬場」の判断に使える。 | 🌐 |

---

### 6-4. 馬券最適化

| URL | ページ名 | ユーザーへの提供価値 | 表示 |
|---|---|---|---|
| `/betting` | 馬券最適化 | AI予測ランキングとオッズを組み合わせ、期待値最大の券種・買い目をシミュレートする。ケリー基準によるポジションサイジング・ROI可視化・バックテストとの整合確認ができる。 | 🔐 |

---

### 6-5. 開発者ツール

通常ユーザーには非表示。`is_dev` フラグが true の場合のみナビに表示される。

| URL | ページ名 | 機能概要 |
|---|---|---|
| `/monitor` | スクレイピングモニター | スクレイピングジョブの進捗・エラー・データ到達状況をリアルタイム監視する。`?date=YYYYMMDD` で日付指定可能。 |
| `/data-viewer` | データビューア | DB に格納されたレース・馬・オッズデータを直接確認する。予期しないデータ異常の調査に使う。 |
| `/scrape-control` | スクレイピングコントロール | スクレイピングジョブの手動実行・再実行を制御する。SLA 遅延時のリカバリ操作に使う。 |
| `/queue-status` | スクレイピングキュー | スクレイピングキューの現在状態（待機中・実行中・完了・失敗）を一覧確認する。 |
| `/scrape-upcoming` | 未来レース取得 | 今後の開催予定レース情報をスクレイピングして登録する。 |
| `/server-logs` | サーバーログ | FastAPIサーバーのログをブラウザ上で閲覧する。 |
| `/cron-jobs` | 定期実行ジョブ | cronジョブの一覧・実行状況・最終実行時刻を管理する。SLA監視の起点。 |