# AREA-07 — モデリング管理要件（LightGBM バッチ推論, 学習パイプライン, SHAP, ModelRegistry, バージョニング, CI ゲート）
**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）, TASK-052（依存関係・リスク D-1〜D-8 対策統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるモデリング管理全般—LightGBM バッチ推論・学習パイプライン・SHAP 解釈・ModelRegistry・バージョニング・CI ゲート—を定義する。テンポラルリーク排除を大前提とし、全工程において `as_of_race_id` によるスナップショット管理を厳守する。

本仕様書は依存関係・リスク D-1〜D-8 の対策をすべて各セクションに織り込み済みであり、**矛盾点ゼロ・リスク対策実装済み**の仕様書として管理する。

---

## 2. 予測ターゲット定義

<!-- TASK-052: T-4〜T-9 の定義を全面更新。旧 T-4(predicted_win_odds)〜T-11(predicted_lap_sec[]) を新定義に置き換え。新決定を優先。 -->

| ID | ターゲット名 | 問題設定 | 出力型 | Stage | データ収集元 | 収集状況 | DB格納Layer | 特徴量としての利用可否 | 推奨モデル | 主評価指標 |
|---|---|---|---|---|---|---|---|---|---|---|
| T-1 | win_prob（勝率） | 多クラス分類（1着） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM softmax | — |
| T-2 | place_prob（連対率） | バイナリ分類（2着以内） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM binary | — |
| T-3 | show_prob（複勝率） | バイナリ分類（3着以内） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM binary | — |
| T-4 | 上り3Fタイム予測 | 回帰（秒） | `FLOAT` | Stage 2 | `race_results.last3f_sec` | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可**（当日走は禁止） | LightGBM Regressor | MAE（秒）、RMSE |
| T-5 | 数値位置取り予測 | 順序回帰（1〜頭数） | `INT` | Stage 2 | `race_results.finish_pos` | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可**（当日走は禁止） | LightGBM（lambdarank損失） | Spearman相関、MAE |
| T-6 | 脚質分類予測 | 4クラス分類（逃/先/差/追） | `ENUM` | Stage 1 | `horse_profiles.running_style` + `corner_pos`（自動補完） | ⚠️ 一部欠損（F-7で補完） | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM Classifier | Accuracy、F1-macro |
| T-7 | ラップ系列予測 | 1F単位回帰（秒×フロン数） | `FLOAT[]` | Stage 3 (Phase 4) | `race_laps.lap_sec` | ✅ 収集済み | Layer 4: `race_lap_predictions`（新設） | ✅ **過去走のみ可** | LSTM / Transformer（Phase 4確定後） | RMSE per furlong |
| T-8 | ペースカテゴリ予測 | 3クラス分類（H/M/S） | `ENUM` | Stage 2 | `race_laps.lap_sec`（T-7と同一ソース） | ✅ 収集済み | Layer 3: `prediction_results` | ❌ **当日予測入力不可**（循環依存回避） | LightGBM Classifier | F1-macro |
| T-9 | 想定走破タイム予測 | 派生回帰（秒） | `FLOAT` | Stage 2（派生） | T-4 + L3F地点回帰モデルの合成 | ✅ 算出可能（`finish_time_sec - last3f_sec`） | Layer 3: `prediction_results` | ✅ **過去走のみ可** | 派生モデル（T-4依存） | RMSE、馬券上位3頭順序安定性 |

> T-9 はモデルを新たに学習させるのではなく、**T-4 と L3F回帰モデルの出力を合成した派生ターゲット**として扱う。

> **重要な前提補足（データ取得可能性）**:
> 上り3Fタイム（`last3f_sec`）およびラップタイムはレース結果テーブルから取得可能。各馬の **L3F地点タイム（ゴールの3F手前地点までの走破タイム）は `走破タイム - 上り3Fタイム` で算出可能**。これにより想定走破タイムを以下の式で導出できる:
> ```
> 想定走破タイム ≈ 予測L3F地点タイム + 予測上り3Fタイム
>               = (走破タイム - 上り3Fタイム の過去回帰予測) + T-4予測値
> ```

> **推論実行順序（Stage依存チェーン）**:
> ```
> Stage 1: T-6（脚質分類）
>       ↓ T-6出力を特徴量として使用
> Stage 2: T-8（ペースカテゴリ） → T-4（上り3F） → T-5（位置取り） → T-9（走破タイム）
>       ※ Stage 2内の実行順序: T-8完了 → T-4/T-5 同時実行可 → T-9（T-4完了後）
> Stage 3: T-7（ラップ系列、Phase 4）
> ```

> **教師データ生成クエリ（L3F地点タイム）**:
> ```sql
> SELECT
>   horse_id,
>   race_id,
>   finish_time_sec - last3f_sec AS l3f_split_time_sec
> FROM race_results
> WHERE finish_time_sec IS NOT NULL AND last3f_sec IS NOT NULL;
> ```

---

## 3. モデルアーキテクチャ

### 3-1. 2ステージ構成

<!-- TASK-052: Stage 2 の出力・Stage 3 の位置付けを更新。アーキテクチャ図は構成を維持しつつターゲット番号を新定義に合わせて修正。 -->

```
┌───────────────────────────────────────────────────────────┐
│ Stage 1: 共有表現マルチタスクモデル                          │
│                                                             │
│  入力: Layer 1〜3 特徴量（馬×レース単位）                    │
│                                                             │
│  ┌─────────────────────────────┐                           │
│  │  Shared Encoder (LightGBM)  │                           │
│  └─────────────┬───────────────┘                           │
│                │                                            │
│     ┌──────────┼───────────┐                               │
│     ▼          ▼           ▼                               │
│  [Head A]   [Head B]    [Head C]                           │
│  勝率/連対  脚質分類      win_roi/                           │
│  /複勝率    (T-6)        show_roi                           │
│  (T-1/2/3)              (ポスト)                            │
│  (分類)                                                     │
└───────────────────────────────────────────────────────────┘
                │ T-6出力を特徴量として受け渡し
                ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 2: ラップ・ペース・タイム予測モデル                     │
│                                                             │
│  入力: Layer 3/4 + Stage 1 T-6 出力 + コース形状特徴量       │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  実行順序:                                           │   │
│  │   1. T-8 ペースカテゴリ (HIGH/MIDDLE/SLOW)          │   │
│  │   2. T-4 上り3Fタイム回帰 ‖ T-5 位置取り順序回帰    │   │
│  │   3. T-9 想定走破タイム（T-4依存・派生モデル）       │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
                │（Phase 4 フラグON のとき）
                ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 3: ラップ系列予測モデル（Phase 4・現時点フリーズ）      │
│                                                             │
│  入力: Layer 4 + Stage 2 出力 + コース形状特徴量             │
│                                                             │
│  ┌────────────────────────────────────────┐               │
│  │  T-7: Lap Sequence Model               │               │
│  │  (LightGBM per-furlong → LSTM へ移行)  │               │
│  │                                          │               │
│  │  出力:                                  │               │
│  │   └── 1F毎ラップ予測値 (furlong別)     │               │
│  └────────────────────────────────────────┘               │
└───────────────────────────────────────────────────────────┘
```

### 3-2. アルゴリズム選定

<!-- TASK-052: ターゲット定義変更に合わせてアルゴリズム選定表を更新。旧 T-8〜T-11 を削除し新 T-4〜T-9 に置き換え。 -->

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率（T-1） | LightGBM softmax | 表形式データに最強、欠損耐性が高い |
| 連対率・複勝率（T-2/3） | LightGBM binary | 同上 |
| 上り3Fタイム（T-4） | LightGBM Regressor | 連続値回帰、解釈性を重視 |
| 位置取り予測（T-5） | LambdaMART（LightGBM lambdarank） | 相対順位を直接最適化できる（D-5対策） |
| 脚質分類（T-6） | LightGBM Classifier | 4クラス・解釈性を重視 |
| ラップ系列（T-7） | LightGBM per-furlong（初期）→ LSTM（Phase 4） | 解釈しやすい単独回帰から開始し、系列依存が大きければ移行 |
| ペースカテゴリ（T-8） | LightGBM Classifier | 3クラス・解釈性を重視 |
| 想定走破タイム（T-9） | 派生モデル（T-4 + L3F回帰の合成） | T-4 精度に依存、RMSE < 0.3秒ゲート通過後のみ本番投入 |

---

## 4. 特徴量定義

### 4-1. 基本特徴量（Layer 1〜2 由来）

| 特徴量名 | 説明 | 型 |
|---|---|---|
| distance | レース距離 (m) | INT |
| surface | 芝/ダート/障害 | CATEGORY |
| direction | 左/右/直線 | CATEGORY |
| going | 馬場状態（良/稍重/重/不良） | CATEGORY |
| weather | 天候 | CATEGORY |
| grade | レースクラス（G1〜未勝利） | CATEGORY |
| horse_num | 出走頭数 | INT |
| frame_no | 枠番 | INT |
| post_no | 馬番 | INT |
| weight_carried | 斤量 (kg) | FLOAT |
| horse_weight | 馬体重 (kg) | INT |
| horse_weight_diff | 馬体重増減 | INT |
| days_since_last | 前走からの間隔（日） | INT |
| horse_age | 馬齢 | INT |
| sex | 性別（牡/牝/セン） | CATEGORY |

### 4-2. 集計特徴量（Layer 3 スナップショット由来）

| 特徴量名 | 説明 |
|---|---|
| win_rate_all / place_rate_all / show_rate_all | 生涯勝率・連対率・複勝率 |
| win_rate_distance / win_rate_course / win_rate_going | 条件別勝率 |
| avg_last_3f / speed_index_avg / speed_index_max | タイム・スピード指数 |
| running_style_score | 脚質スコア（−5=逃 〜 +5=追込） |
| jockey.win_rate_all | 騎手勝率 |
| trainer.win_rate_all | 調教師勝率 |

> **必須制約**: Layer 3 集計値は必ず `as_of_race_id = 予測対象レース ID` のスナップショットを参照すること。そのレース以後の情報を含めることを禁止する。Feature Store API の `get_snapshot(race_id, as_of=race_id)` 呼び出し時に `window_end=as_of_race_id` を必須引数として渡すこと（D-4対策）。

### 4-3. クロス・相対特徴量（前処理で自動生成）

```python
# 脚質 × コース形状のクロス特徴量
df["style_x_straight"] = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"] = df["running_style_score"] * df["distance_category_encoded"]

# 逃げ・先行馬数（ペース予測用）
df["front_runner_count"] = df.groupby("race_id")["running_style_score"] \
                             .transform(lambda x: (x < -2).sum())

# 同レース内相対化
df["rel_speed_index"]     = df["speed_index_avg"] / \
                              df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - \
                              df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"]       = df.groupby("race_id")["odds_value"].rank(ascending=True)

# ペース事前シナリオ（逃げ馬比率から算出）
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]) \
    .apply(lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 4-4. T-9 想定走破タイム用特徴量（L3F地点回帰モデル）

<!-- TASK-052: T-9 用特徴量設計を追加（新規セクション）。 -->

| 特徴量 | 内容 | 時点制約（AREA-06） |
|---|---|---|
| 過去N走の `l3f_split_time_sec` 平均 | 馬の前半〜中盤ペース傾向 | `as_of_race_id` より前のみ ✅ |
| 距離 | レース距離（m） | 当日レース情報（リークなし） ✅ |
| 馬場状態 | 良/稍重/重/不良 | 当日レース情報（リークなし） ✅ |
| T-8 ペースカテゴリ予測値 | ハイペース時は前半が速く後半が遅い | T-8 出力（Stage 2、T-4より先行実行済み） ✅ |
| 騎手別 l3f オフセット | 騎手の前半ペース傾向 | `as_of_race_id` より前のみ ✅ |

---

## 5. 学習パイプライン

### 5-1. データ分割ルール

- 時系列順に train / validation / test を分割する。**ランダムシャッフルは禁止**。
- 常に過去レースで学習し、未来レースで評価する。
- 推論時も `as_of_race_id = 対象レース ID` のスナップショットのみ使用する。

### 5-2. Stage 1 学習手順

1. 特徴量エンジニアリングパイプライン実行（脚質スコア・クロス特徴量・相対特徴量の自動生成）
2. 時系列分割により train/validation/test セット構築
3. LightGBM binary で連対率・複勝率モデル（T-2/T-3）を学習
4. LightGBM softmax で勝率モデル（T-1）を学習
5. **T-6 脚質分類モデルを学習（前提: 自動ラベル一致率 > 80% の検証通過後・D-3対策）**
6. 各モデルの評価指標を計算（後述）
7. モデルを ModelRegistry へ登録・バージョニング

### 5-3. Stage 2 学習手順

<!-- TASK-052: 旧 Stage 2 手順を T-4〜T-9 定義に合わせて全面更新。 -->

1. Stage 1 の T-6（脚質分類）出力を入力特徴量として受け渡し
2. Layer 3/4 + コース形状特徴量を結合
3. **LightGBM Classifier でペースカテゴリモデル（T-8）を学習**（Phase Gate A 通過後）
4. **LightGBM Regressor で上り3Fタイムモデル（T-4）を学習**（Feature Store テンポラルリークテスト PASS 後）
5. **LightGBM lambdarank で位置取り順序回帰モデル（T-5）を学習**（D-5対策）
6. **T-4 の RMSE < 0.3秒 達成後に `model_t4.is_production_ready = True` フラグを設定し、L3F地点回帰モデルと合成して T-9（想定走破タイム）を実装**（D-8対策）
7. 評価指標を計算・記録
8. モデルを ModelRegistry へ登録・バージョニング

### 5-4. Stage 3 学習手順（Phase 4・現時点フリーズ）

<!-- TASK-052: 旧 Stage 2 のラップ予測を Stage 3 / Phase 4 に分離。`PHASE4_LAP_PREDICTION` フラグによる制御を追加（D-6対策）。 -->

> **【D-6対策】** `PHASE4_LAP_PREDICTION` フィーチャーフラグが有効化されるまで、T-7 の実装・`race_lap_predictions` テーブルへの書き込みは一切禁止する。

1. Stage 2 の出力を入力特徴量として受け渡し
2. Layer 4（ラップ・ペース・コーナー通過）の特徴量を結合
3. LightGBM per-furlong で 1F 毎ラップ予測モデル（T-7）を学習（初期実装）
4. RMSE per furlong の閾値を下回れない場合は LSTM