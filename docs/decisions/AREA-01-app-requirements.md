# AREA-01 — アプリケーション要件
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001 (統合済み・削除)

---

## 0. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること**が、
データ基盤・モデリング・評価・API 設計の全工程の前提条件となる。

---

## 1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F 単位ラップ予測** を実現する競馬予測 Web アプリ。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧） |
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

### 3-3. テーブルスキーマ定義

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

---

## 5. 機能要件

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

---

## 6. 非機能要件

| # | 要件 | 目標値 |
|---|---|---|
| N-1 | 予測 API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | 予測 API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ラップ予測 MAE（1F 単位） | ≤ 0.3 秒 |
| N-4 | 勝率予測 Log Loss（ベースライン比） | −5% 以上改善 |
| N-5 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | スクレイピング後の DB 反映遅延 | ≤ 10 分 |
| N-8 | オッズスナップショット欠損率（発走前5分以内） | ≤ 1% |
| N-9 | モデル推論バッチ完了時刻 | 発走3時間前までに完了 |
| N-10 | 特徴量リーク（未来情報混入）ゼロ | テストデータ時系列分割で検証 |
| N-11 | DDL マイグレーション管理 | Alembic 等で全変更をバージョン管理 |
| N-12 | Redis キャッシュ TTL | 発走まで有効 / 発走後60秒で自動失効 |
| N-13 | テストカバレッジ（スクレイパー・特徴量パイプライン） | ≥ 80% |
| N-14 | 障害レース・海外レースは予測対象外として明示除外 | フラグ管理 |

---

## 7. API 仕様

### エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `GET` | `/api/v1/races/{race_id}/predictions` | レース全体の予測結果取得（T-1〜T-11） |
| `GET` | `/api/v1/races/{race_id}/laps` | ラップ予測系列取得（`furlong_index` 順） |

### `GET /api/v1/races/{race_id}/predictions` レスポンス

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

| フィールド | 型 | 説明 |
|---|---|---|
| `horses[].is_value_bet` | `BOOLEAN` | `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100` で `true` |
| `pace_prediction.lap_times[].furlong_index` | `INT` | 1始まり（1F 目 = スタート直後） |

### キャッシュ仕様

| キャッシュキー | TTL |
|---|---|
| `prediction:{race_id}:{model_version}` | 発走まで有効 / 発走後60秒失効 |
| `lap:prediction:{race_id}:{model_version}` | 同上 |

---

## 8. ユーザーストーリー

| ID | ストーリー | 受入条件 |
|---|---|---|
| US-1 | 任意レースの全出走馬の勝率・連対率・複勝率を一覧で確認したい | T-1〜T-3 が全馬分返る |
| US-2 | 期待値プラスの馬（バリューベット候補）を即座に識別したい | `is_value_bet: true` の馬が UI でハイライト表示 |
| US-3 | レースの展開（ペース・ラップ推移）を視覚的に把握したい | ペースカテゴリと1F毎ラップ予測が折れ線グラフで表示（F-15） |
| US-4 | 各馬の予測脚質とポジションを確認し展開を読みたい | `predicted_running_style`・`predicted_position` が全馬分表示 |
| US-5 | スクレイピングの成否・リトライ状況を追跡したい | `scrape_runs` テーブルに `target_type`・`status`・`retry_count` が記録される |

---

## 9. 実装ロードマップ

### Phase 0: 前提条件検証（1〜2週間）

> **F-17 が完了するまで Phase 1 以降は開始しない**

- [ ] ラップデータ可用性調査（サンプル10レース × 4コース形状で手動確認）
- [ ] netkeiba.com のロボット規約・利用規約の確認（**H-01: 着手前に必須**）
- [ ] `scrape_runs` テーブル・基本スキーマの構築

### Phase 1: データ基盤構築（2〜4週間）

- [ ] F-1〜F-5 スクレイパー・実行管理テーブル実装
- [ ] Layer 1〜5 全テーブルの DDL マイグレーション適用
- [ ] 過去2〜3年分の全レースデータの一括取得

**完了条件**: 過去2年分のレース・ラップ・オッズデータが Layer 1〜5 に格納済みであること

### Phase 2: 特徴量パイプライン + Stage 1 モデル（3〜5週間）

- [ ] F-6 特徴量エンジニアリングパイプライン実装
- [ ] F-7 Stage 1 モデル学習パイプライン実装
- [ ] F-9 回収率計算ロジック実装・単体テスト
- [ ] F-16 モデルバージョン管理基盤整備

**完了条件**: 勝率 Log Loss がベースライン比 −5% 以上改善

### Phase 3: Stage 2 モデル + API + UI（3〜4週間）

- [ ] F-8 Stage 2 モデル学習パイプライン実装
- [ ] F-10〜F-15 API・Redis キャッシュ・UI 実装
- [ ] N-1/N-2 API レスポンスタイム計測・チューニング

**完了条件**: 任意レースの全予測ターゲットが API 経由で取得でき UI に表示される

### Phase 4: 運用安定化・精度改善（継続）

- [ ] LSTM によるラップ系列モデルへの移行検討
- [ ] モデル再学習の定期実行自動化（週次 or 月次）
- [ ] 特徴量重要度モニタリング・データドリフト検知

---

## 10. 依存関係・リスク

### 依存関係

```
Phase 0（ラップデータ可用性確認）
    └─→ Phase 1（データ基盤構築）
            └─→ Phase 2（特徴量 + Stage 1 モデル）
                    └─→ Phase 3（Stage 2 + API + UI）

Layer 3 スナップショット（F-3）→ Stage 1 モデル（F-7）前提
Stage 1 ポジション予測（F-7）  → Stage 2 ラップ予測（F-8）入力として必須
オッズスナップショット（F-4）  → 推論時特徴量として必須
```

### リスクと対策

| # | リスク | 深刻度 | 対策 |
|---|---|---|---|
| R-1 | ラップデータが一部レースに存在しない | 🔴 高 | Phase 0 で事前検証。欠損は `lap_time_sec = NULL` で格納し予測除外 |
| R-2 | netkeiba.com の HTML 構造変更によりスクレイパー破損 | 🔴 高 | HTML パース箇所を設定値化。週次で成功率を監視し閾値以下でアラート |
| R-3 | netkeiba.com からアクセス制限（429 / IP ブロック） | 🟡 中 | リクエスト間隔2秒＋ジッター、セッションローテーション、リトライバックオフ |
| R-4 | 特徴量リーク（テンポラルリーク）の混入 | 🔴 高 | `as_of_race_id` 紐付けの単体テスト必須化。CI でリーク検知テストを自動実行 |
| R-5 | Stage 1 → Stage 2 の誤差伝播 | 🟡 中 | Stage 2 では Stage 1 予測値の信頼区間も特徴量として入力。独立評価で誤差を分離計測 |
| R-6 | オッズの直前大幅変動による回収率計算のずれ | 🟡 中 | 発走5分前スナップショットを「推論時使用オッズ」として固定 |
| R-7 | 障害レースのラップ形式が平地と異なる | 🟢 低 | `race_type = '障害'` フラグで予測対象外に除外 |
| R-8 | LightGBM per-furlong が系列依存を捉えられない | 🟡 中 | Phase 4 で LSTM への移行を評価。MAE ≤ 0.3秒 を移行判断閾値とする |

---

## 11. 未解決事項（Human Review 必須）

| ID | 内容 | 優先度 |
|---|---|---|
| H-01 | **JRA スクレイピング ToS / robots.txt 法的確認** — ETL 開始前に必須 | 緊急 |
| H-02 | ETL 遅延フォールバックポリシー（503 全件 vs 前日データ配信） | 高 |

---

## 12. 用語定義

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
| `as_of_race_id` | スナップショットが「このレース直前時点」の情報であることを示す外部キー |
