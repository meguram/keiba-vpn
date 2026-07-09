# AREA-11 — めぐ指数（レースパフォーマンス評価指数）
**Status**: DRAFT | **Last Updated**: 2026-07-09

---

## 0. 最重要原則

**1点 = 0.1秒差**。ペース・馬場・斤量・レースレベルの補正はすべて「秒単位」に統一し、この等価性を構造的に保つ。

---

## 1. 定義・目的

### 1-1. めぐ指数とは

競走馬の各レースにおける走破パフォーマンスを、ペース・馬場・斤量・レースレベルの影響を統合回帰モデルで除去した上で**絶対スケール（1点=0.1秒）**で表現する独自指数。

| 項目 | 内容 |
|---|---|
| 粒度 | 馬 × レース（1走ごとに1スコア） |
| 基準値 | 100 = 同条件（距離×コース×芝/ダート×馬場カテゴリ）での補正後平均走破タイム |
| 単位 | 1点差 = 0.1秒差（直接タイム差に変換可能） |
| 障害レース | 除外（平地レースのみ対象） |
| データ不足 | `NULL`（`—`表示）。新馬・走破タイム未記録レースも同様 |

### 1-2. 他指数との差別化

netkeiba 等が提供するタイム指数は外部サービスの独自計算であり資産化できない。めぐ指数は本アプリ独自の理論・計算ロジックに基づき、将来の AI 特徴量としての利用も視野に入れた設計とする。

---

## 2. コアフォーミュラ

```
めぐ指数 = 100 + (par_time - adjusted_time) × 10

adjusted_time = raw_time
              - Δpace        # ペース補正（秒）
              - Δtrack       # 馬場補正（秒）
              - Δweight      # 斤量補正（秒）
              - Δlevel       # レースレベル補正（秒）
```

- `raw_time`: 実際の走破タイム（秒）
- `par_time`: 同条件での基準タイム（秒）。統合回帰の固定効果から算出
- 各 Δ はすべて**秒単位**で、正の値 = 馬が不利を受けていた分の補正（調整後タイムが短くなる = 指数が上がる）

**符号の直感**:

| 状況 | 補正値 | 効果 |
|---|---|---|
| スローペース（raw_time が遅くなりやすい） | Δpace > 0 | 指数アップ |
| 馬場が遅い日（同条件の平均より時計がかかる） | Δtrack > 0 | 指数アップ |
| 重い斤量を背負った | Δweight > 0 | 指数アップ |
| 強い相手と戦った（上位5着の賞金履歴平均が高い） | Δlevel > 0 | 指数アップ |

---

## 3. 統合回帰モデル

すべての補正係数を**単一の OLS 回帰**で同時推定する。各係数が秒単位で算出されるため、1点=0.1秒の等価性が構造的に保たれる。

### 3-1. 回帰式

```
raw_time = β₀
         + β₁ × front_split_dev          # ペース補正係数
         + β₂ × TSI_offset               # 馬場補正係数
         + β₃ × weight_dev × dist_scale  # 斤量補正係数
         + β₄ × log(FQ / par_FQ)         # レースレベル補正係数
         + Σ γᵢ × fixed_effect_i         # 距離・コース・芝ダート・馬場カテゴリの固定効果
         + ε
```

| 変数 | 定義 |
|---|---|
| `front_split_dev` | 実前半スプリット - 同条件の基準スプリット（秒）。正 = スロー |
| `TSI_offset` | TrackSpeedIndex の時計偏差（秒）。正 = 速い馬場 |
| `weight_dev` | 実斤量 - 基準斤量（kg）。牡・セン基準55kg、牝馬は -2kg オフセット |
| `dist_scale` | 距離 / 2000（無次元）。2000m 基準のスケール係数 |
| `FQ` | フィールド質 = 上位5着以内の馬の獲得賞金履歴平均（円） |
| `par_FQ` | 全レースの FQ の幾何平均（ log スケールの基準値） |
| `fixed_effect_i` | 距離帯 × コース × 芝/ダート × 馬場カテゴリ のダミー変数 |

### 3-2. 補正値の算出

回帰推定後、各補正値を以下のように計算する：

```python
Δpace   = β₁ × front_split_dev
Δtrack  = -β₂ × TSI_offset           # TSI が正 = 速い馬場 = 時計有利 → 負補正
Δweight = β₃ × weight_dev × dist_scale
Δlevel  = β₄ × log(FQ / par_FQ)
```

> `Δtrack` の符号に注意: TSI が正（速い馬場）なら時計上有利を除去するため負補正。
> 係数 β₁〜β₄ は NB-02（統合回帰 notebook）で推定する。

### 3-3. 基準タイム（par_time）

回帰の固定効果から算出。「距離×コース×芝/ダート×馬場カテゴリ」の組み合わせごとの切片値が par_time に相当する。

- クラス（G1/G3/条件戦等）はグルーピングしない。クラス差はフィールド質補正（Δlevel）が吸収する
- サンプル数が少ないセル（< 30件）は上位の距離帯にプーリング（固定効果の縮小推定）

---

## 4. 前半スプリット計測点の選択ルール

### 4-1. 基本ルール

**計測点 = レース距離の50%以下で最近傍の利用可能なスプリット点**

主要距離での対応例（200m刻みのラップデータ）:

| レース距離 | 50%地点 | 計測点 |
|---|---|---|
| 1200m | 600m | 600m |
| 1400m | 700m | 600m |
| 1600m | 800m | 800m |
| 1800m | 900m | 800m |
| 2000m | 1000m | 1000m |
| 2200m | 1100m | 1000m |
| 2400m | 1200m | 1200m |
| 3200m | 1600m | 1600m |
| 3600m | 1800m | 1800m |

### 4-2. フォールバック（100m・300m刻みのラップデータ）

| レース距離 | 計測点 |
|---|---|
| ≤ 900m | 500m |
| > 900m | 距離の50%以下で最近傍の100m刻み点 |

### 4-3. データ欠損時

対象スプリットが `race_result_lap` に存在しない場合は `Δpace = 0`（ペース補正なし）として他の補正のみ適用する。

---

## 5. フィールド質（FQ）の定義

```
FQ = Σ(上位5着以内の馬_i の獲得賞金総額) / 5
```

- **獲得賞金総額**: `horse_result` の生涯獲得賞金（当該レース以前の累計。テンポラルリーク防止）
- 5着以内の出走馬が5頭に満たない場合は実際の頭数で平均
- 全レースの `log(FQ)` の平均が `log(par_FQ)` となる基準点

---

## 6. 今週のめぐ指数（集計仕様）

任意の出馬レースに対して、各出走馬の直近フォームを1スコアで表現する集計指数。

### 6-1. 集計パターン（3パターン全表示）

| パターン | 集計方法 | 特性 |
|---|---|---|
| **A** | 直近5走の**最大値** | 潜在能力の上限 |
| **B**（デフォルト） | 直近5走の**加重平均**（最新走ウェイト高） | 直近の調子を反映 |
| **C** | 同距離帯・同馬場種別での直近3走の最大値 | 条件特化・外れ値耐性あり |

### 6-2. 加重平均（パターンB）のウェイト

```
直近1走: ×0.35
直近2走: ×0.25
直近3走: ×0.20
直近4走: ×0.12
直近5走: ×0.08
```

> ウェイトは NB-05（バリデーション）で調整可能。

### 6-3. 距離帯の定義（パターンC）

| 距離帯 | 範囲 |
|---|---|
| sprint | < 1500m |
| mile | 1500〜1799m |
| middle | 1800〜2399m |
| long | ≥ 2400m |

### 6-4. 表示イメージ

| 馬番 | 馬名 | めぐ指数(B) | めぐ指数(A) | めぐ指数(C) | 近5走推移 |
|---|---|---|---|---|---|
| 1 | ○○○ | **105.2** | 108 | 106 | 98, 103, 105, 108, 101 |
| 2 | △△△ | **98.4** | 101 | — | 96, 98, 101, — , — |

---

## 7. DB スキーマ

```sql
-- 馬×レース単位のめぐ指数
CREATE TABLE megu_index (
    id                    BIGSERIAL     PRIMARY KEY,
    race_id               VARCHAR(20)   NOT NULL,
    horse_id              VARCHAR(20)   NOT NULL,
    finish_time_sec       NUMERIC(6,2)  NOT NULL,      -- 生走破タイム（秒）
    par_time_sec          NUMERIC(6,2),                 -- 基準タイム（固定効果から）
    delta_pace_sec        NUMERIC(5,3)  NOT NULL DEFAULT 0,
    delta_track_sec       NUMERIC(5,3)  NOT NULL DEFAULT 0,
    delta_weight_sec      NUMERIC(5,3)  NOT NULL DEFAULT 0,
    delta_level_sec       NUMERIC(5,3)  NOT NULL DEFAULT 0,
    adjusted_time_sec     NUMERIC(6,2)  NOT NULL,      -- 補正済み走破タイム
    megu_index            NUMERIC(6,1)  NOT NULL,      -- めぐ指数（小数第1位）
    field_quality         NUMERIC(12,0),               -- FQ（円）
    front_split_sec       NUMERIC(5,2),                -- 実際のスプリットタイム
    split_point_m         SMALLINT,                    -- 計測点（m）
    model_version         VARCHAR(20)   NOT NULL DEFAULT 'v1',
    computed_at           TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (race_id, horse_id, model_version)
);

CREATE INDEX idx_megu_index_horse   ON megu_index (horse_id, computed_at DESC);
CREATE INDEX idx_megu_index_race    ON megu_index (race_id);

-- 統合回帰の推定係数
CREATE TABLE megu_regression_params (
    id            SERIAL       PRIMARY KEY,
    param_name    VARCHAR(50)  NOT NULL,    -- 'beta_pace', 'beta_track', 'beta_weight', 'beta_level'
    param_value   NUMERIC(10,6) NOT NULL,
    std_error     NUMERIC(10,6),
    sample_count  INTEGER,
    model_version VARCHAR(20)  NOT NULL DEFAULT 'v1',
    fitted_at     TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (param_name, model_version)
);

-- 基準タイム（固定効果）マスター
CREATE TABLE megu_par_time (
    id              SERIAL        PRIMARY KEY,
    distance        INTEGER       NOT NULL,
    course          VARCHAR(20)   NOT NULL,
    surface         VARCHAR(10)   NOT NULL,    -- '芝' / 'ダート'
    track_condition VARCHAR(10)   NOT NULL,    -- '良' / '稍重' / '重' / '不良'
    par_time_sec    NUMERIC(6,2)  NOT NULL,
    par_front_split_sec NUMERIC(5,2),          -- 基準前半スプリット
    par_fq          NUMERIC(12,0),             -- par_FQ（幾何平均）
    sample_count    INTEGER       NOT NULL,
    model_version   VARCHAR(20)   NOT NULL DEFAULT 'v1',
    computed_at     TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (distance, course, surface, track_condition, model_version)
);
```

---

## 8. GCS・データフロー

```
race_result（SLA 5: 確定結果）
  │
  ├─ horse_result（獲得賞金履歴 → FQ 算出）
  ├─ race_result_lap（前半スプリット → Δpace）
  ├─ TrackSpeedIndex（TSI_offset → Δtrack）
  └─ race_shutuba（実斤量 → Δweight）
        │
        ▼
  aggregate_megu_index.py
  （megu_regression_params + megu_par_time を参照）
        │
        ▼
  megu_index テーブル（DB）
  race_performance GCS（既存パス活用）
  └─ netkeiba/pc/race_performance/{year}/{race_id}.json
```

**実行タイミング**: SLA 5（JST 17:30 確定結果取得後）に自動トリガー

---

## 9. ノートブックタスク一覧

| # | タイトル | 目的 | 入力データ | 出力 |
|---|---|---|---|---|
| NB-01 | データ探索・前処理 | 走破タイム・スプリット・斤量・FQの分布確認、外れ値処理方針の決定 | race_result, horse_result, race_result_lap, race_shutuba | クリーニング済みデータセット、外れ値除去基準 |
| NB-02 | 統合回帰モデル推定 | β₁〜β₄ および固定効果（par_time）の OLS 推定 | NB-01 出力 | megu_regression_params / megu_par_time の初期値、係数の有意性・符号確認 |
| NB-03 | 斤量補正の検証 | β₃（斤量×距離スケール）の推定値を理論値0.2秒/kgと比較検証 | NB-02 出力 | β₃ 確定値・距離別感度分析 |
| NB-04 | TSI整合性検証 | TSI_offset が馬場の速さを正確に捉えているか確認（クッション値・含水率との相関） | race_result, jra_cushion | TSI 計算式の改善案（必要なら） |
| NB-05 | 集計関数バリデーション | 今週のめぐ指数（A/B/C）の着順予測有効性比較、加重平均ウェイト調整 | megu_index（全期間） | 推奨集計パターン・ウェイト確定値 |
| NB-06 | 有効性検証 | めぐ指数と実際の着順・回収率の相関。指数の予測有効性レポート | megu_index × race_result | 有効性サマリーレポート |

**実行順序**: NB-01 → NB-02 → NB-03（並列可） & NB-04（並列可） → NB-05 → NB-06

---

## 10. 確定済み仕様一覧

| 項目 | 決定内容 |
|---|---|
| クラスグルーピング | なし（距離×コース×馬場4軸のみ。クラス差は Δlevel が吸収） |
| 牝馬基準斤量 | 55kg 基準、牝馬は -2kg オフセット |
| 今週のめぐ指数 | デフォルト B（加重平均）、A/B/C 全パターン表示 |
| 障害レース | 除外 |
| データ不足 | NULL（`—` 表示） |
| スプリット欠損 | Δpace = 0（他補正は適用） |
| スプリット計測点 | 距離の50%以下で最近傍点。100m/300m刻みデータは専用フォールバック |
| レースレベル指標 | 上位5着以内の馬の獲得賞金履歴平均（log 変換） |
| 補正係数の推定 | 統合 OLS 回帰（NB-02）で一括推定 |

---

## 11. 未決定事項（NB 実施後に確定）

| # | 項目 | 確定手段 |
|---|---|---|
| U-1 | 加重平均ウェイト（B パターン）の最終値 | NB-05 |
| U-2 | スプリットの基準点（par_front_split）の算出方法 | NB-02 |
| U-3 | FQ 外れ値処理（新馬・輸入馬で FQ が異常に低い場合） | NB-01 |
| U-4 | 回帰の推定方式（OLS / Ridge / 固定効果モデル）の最終選択 | NB-02 |
| U-5 | モデル再推定の頻度（月次 / 四半期 / 年次） | NB-06 後に決定 |
