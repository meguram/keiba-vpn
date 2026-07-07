# AREA-07 — モデリング管理要件（LightGBM バッチ推論, 学習パイプライン, SHAP, ModelRegistry, バージョニング, CI ゲート）
**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）, TASK-052（依存関係・リスク D-1〜D-8 対策統合済み）, DEC-022（予測タスク二値分類統一）, DEC-009（モデル保持ポリシー）, DEC-015（特徴量スキーマ正規定義整合）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるモデリング管理全般—LightGBM バッチ推論・学習パイプライン・SHAP 解釈・ModelRegistry・バージョニング・CI ゲート—を定義する。テンポラルリーク排除を大前提とし、全工程において `as_of_race_id` によるスナップショット管理を厳守する。

本仕様書は依存関係・リスク D-1〜D-8 の対策をすべて各セクションに織り込み済みであり、**矛盾点ゼロ・リスク対策実装済み**の仕様書として管理する。

---

## 2. 予測ターゲット定義

<!-- TASK-052: T-4〜T-9 の定義を全面更新。旧 T-4(predicted_win_odds)〜T-11(predicted_lap_sec[]) を新定義に置き換え。新決定を優先。 -->

| ID | ターゲット名 | 問題設定 | 出力型 | Stage | データ収集元 | 収集状況 | DB格納Layer | 特徴量としての利用可否 | 推奨モデル | 主評価指標 |
|---|---|---|---|---|---|---|---|---|---|---|
| T-1 | win_probability（勝率） | 二値分類（1着=1, それ以外=0） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM binary (objective='binary') | AUC、Log Loss |
| T-2 | place_prob（連対率） | 二値分類（2着以内=1, それ以外=0） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM binary (objective='binary') | AUC、Log Loss |
| T-3 | show_prob（複勝率） | 二値分類（3着以内=1, それ以外=0） | NUMERIC(5,4) | Stage 1 | — | ✅ 収集済み | Layer 3: `prediction_results` | ✅ **過去走のみ可** | LightGBM binary (objective='binary') | AUC、Log Loss |
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

### 2-1. T-1 出力ポストプロセス（DEC-022）

<!-- DEC-022: T-1 を二値分類に統一。出力は softmax 正規化 → 勝率 p_i。連対率・複勝率は Harville 式で導出。 -->

**モデル定義**:

```python
# LightGBM binary: ラベル 1着=1, それ以外=0
params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 127,
}
# tansho_label: BOOLEAN（AREA-06 § 5-1 と整合）
df["tansho_label"] = (df["finish_pos"] == 1).astype(int)  # BOOLEAN → 0/1
```

**出力ポストプロセス**:

```python
import numpy as np

def postprocess_win_probs(raw_scores: list[float]) -> list[float]:
    """
    各馬の binary スコアを softmax 正規化して勝率 p_i を算出する。
    raw_scores: LightGBM binary predict_proba の出力（レース内全馬）
    """
    exp_scores = np.exp(raw_scores - np.max(raw_scores))  # 数値安定化
    return (exp_scores / exp_scores.sum()).tolist()

def harville_place_prob(p: list[float]) -> list[float]:
    """Harville 式による連対率（2着以内確率）"""
    n = len(p)
    place = []
    for i in range(n):
        # P(i が2着以内) = P(i が1着) + Σ_{j≠i} P(j が1着) × P(i が2着 | j が1着)
        prob = p[i]
        for j in range(n):
            if j != i:
                prob += p[j] * (p[i] / (1 - p[j] + 1e-9))
        place.append(min(prob, 1.0))
    return place

def harville_show_prob(p: list[float]) -> list[float]:
    """Harville 式による複勝率（3着以内確率）"""
    n = len(p)
    show = []
    for i in range(n):
        prob = p[i]
        for j in range(n):
            if j == i:
                continue
            pj_given_i = p[j] / (1 - p[i] + 1e-9)
            for k in range(n):
                if k == i or k == j:
                    continue
                prob += p[j] * pj_given_i * (p[i] / (1 - p[i] - p[j] + 1e-9))
        show.append(min(prob, 1.0))
    return show
```

**DB格納フィールド（AREA-02 整合）**:

| フィールド | 旧名称 | 新名称 | 型 |
|---|---|---|---|
| 勝率 | `prediction_score` | `win_probability` | `FLOAT (0.0〜1.0)` |
| 連対率 | — | `place_probability` | `FLOAT (0.0〜1.0)` |
| 複勝率 | — | `show_probability` | `FLOAT (0.0〜1.0)` |

> `tansho_label` は `BOOLEAN` 型（AREA-06 Layer 2 `race_results.tansho_label`）と整合。学習時は `int(0/1)` に変換して使用する。

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
| 勝率（T-1） | LightGBM binary (objective='binary') + softmax 正規化 | 二値分類で学習後、レース内確率を softmax 正規化して勝率 p_i を算出。Harville 式で連対率・複勝率を導出 |
| 連対率・複勝率（T-2/3） | LightGBM binary (objective='binary') | T-1 の Harville 出力と組み合わせて最終確率を補正 |
| 上り3Fタイム（T-4） | LightGBM Regressor | 連続値回帰、解釈性を重視 |
| 位置取り予測（T-5） | LambdaMART（LightGBM lambdarank） | 相対順位を直接最適化できる（D-5対策） |
| 脚質分類（T-6） | LightGBM Classifier | 4クラス・解釈性を重視 |
| ラップ系列（T-7） | LightGBM per-furlong（初期）→ LSTM（Phase 4） | 解釈しやすい単独回帰から開始し、系列依存が大きければ移行 |
| ペースカテゴリ（T-8） | LightGBM Classifier | 3クラス・解釈性を重視 |
| 想定走破タイム（T-9） | 派生モデル（T-4 + L3F回帰の合成） | T-4 精度に依存、RMSE < 0.3秒ゲート通過後のみ本番投入 |

---

## 4. 特徴量定義

<!-- DEC-015: カラム名の正規定義は AREA-06 § 5-1 を SSoT とする。本セクションの特徴量名は AREA-06 § 5-1 と一致させること。追加・変更は先に AREA-06 § 5-1 を更新し、本セクションへ伝播させる。 -->

> **カラム名 SSoT**: AREA-06 § 5-1「主要特徴量カラム一覧」を参照。本セクションの特徴量名（`horse_past_results`, `jockey_stats`, `course_affinity`, `odds_win`, `odds_place`, `odds_change_rate`, `track_condition` 等）は AREA-06 と一致させること。

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

### 5-0. 学習データセット概要

| 項目 | 内容 |
|---|---|
| **データソース** | netkeiba.com スクレイピング（中央競馬: JRA 全レース） |
| **収集開始日** | **2020年1月1日**（バックフィル済み） |
| **収集終了** | 直近レース（週次・日次 SLA で継続取得中） |
| **収集対象** | 出走表・レース結果・ラップタイム・オッズ・馬情報・血統（AREA-06 §4-2 参照） |
| **格納先** | PostgreSQL（Layer 1〜4）+ GCS（生 JSON バックアップ） |

> **ノートブック・コードで参照する場合の注意**: Train 分割は 2020-01-01 以降のデータのみ存在する。
> 2019 年以前のデータは収集対象外であり、Feature Store / ModelRegistry に存在しない。

### 5-1. データ分割ルール

- 時系列順に train / validation / test を分割する。**ランダムシャッフルは禁止**。
- 常に過去レースで学習し、未来レースで評価する。
- 推論時も `as_of_race_id = 対象レース ID` のスナップショットのみ使用する。

### 5-2. Stage 1 学習手順

1. 特徴量エンジニアリングパイプライン実行（脚質スコア・クロス特徴量・相対特徴量の自動生成）
2. 時系列分割により train/validation/test セット構築
3. LightGBM binary (objective='binary') で連対率モデル（T-2）・複勝率モデル（T-3）を学習
4. LightGBM binary (objective='binary') で勝率モデル（T-1）を学習。出力を softmax 正規化し `win_probability` を算出後、Harville 式で `place_probability` / `show_probability` を導出（§ 2-1 参照）
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
4. RMSE per furlong の閾値を下回れない場合は LSTM へ移行（Phase 4 判断）

---

## 6. モデル保持ポリシー（DEC-009）

<!-- DEC-009: AREA-06 § 6 と同一ポリシー。GCS モデルストアの保持ルールを統一定義。 -->

### 6-1. 複合保持ポリシー

`gs://${GCS_BUCKET}/chuou/models/v{model_version}/` に格納されるすべてのモデルアーティファクトに以下を適用する：

| ルール | 内容 |
|---|---|
| **最新バージョン保持** | 最新3バージョンを常時保持（削除不可） |
| **経過日数による削除** | 作成日から 365 日経過したバージョンを削除 |
| **優先順位** | 「最新3バージョン保持」が「365日削除」より優先（最新3件は365日超過でも保持） |

> AREA-06 § 6 と同一ポリシー。GCS ライフサイクル設定の実装詳細は AREA-06 § 6-2 を参照。

### 6-2. ModelRegistry バージョニング規則

- モデルバージョンは `v{model_version}` 形式（例: `v1`, `v2`, `v3`）
- 各バージョンに `_manifest_{YYYYMMDD}.json` を付与し、`created_at`・`feature_version`・`eval_metrics` を記録する
- Stage 1 モデル（T-1/T-2/T-3/T-6）と Stage 2 モデル（T-4/T-5/T-8/T-9）は同一 `model_version` タグで管理し、セットで保持・削除する
- CI ゲートは本番デプロイ前に以下を検証する：
  - T-1: AUC ≥ 0.65、Log Loss ≤ 0.65
  - T-4: RMSE < 0.3 秒（T-9 の本番投入条件）
  - T-6: ラベル一致率 > 80%（D-3対策）

---

## 10. 馬券選択モデリング基盤（Stage 4: Betting Strategy）

<!-- AREA-07 T-10 / AREA-01 T-12 として追加。
     フィーチャーフラグ PHASE5_BETTING_STRATEGY + PHASE5_SCOPE による段階投入。
     Round 2 改訂: integration-synthesizer 統合指示 P1〜P3 全件反映済み。
     Round 3 最終: ai-model-engineer 懸念1〜3・data-engineer P1・backend-engineer 軽微指摘・
                   cost-optimizer C-1〜C-2 を全件反映し確定。 -->

> **【段階投入原則】** `PHASE5_BETTING_STRATEGY=true` が設定されるまで、
> T-10 の実装・`bet_recommendations` テーブルへの書き込みは一切禁止する。
> Stage 1（T-1/T-2/T-3）の AUC ≥ 0.65 CI ゲート通過が前提条件。
> 本セクションは AREA-07 §6 の CI ゲート体系を継承する。

---

### 10-1. 概要・設計思想

#### 10-1-1. 設計目的

Stage 1〜3 の予測出力とオッズを統合し、**期待値最大化 + 的中率バランス** を実現する馬券購入推薦を自動生成する。単一の巨大実装としてではなく、**3段階のフェーズ分割**によってリスクを段階的に管理しながら本番投入する。

#### 10-1-2. フェーズ分割原則（Phase 5a / 5b / 5c）

| フェーズ | 追加 Step | 有効化条件 | 馬券種スコープ | 学習インフラ |
|---|---|---|---|---|
| **Phase 5a: EV-only MVP** | Step A（EV計算）+ Step D（Kellyフィルタ） | `PHASE5_SCOPE=EV_ONLY` + Stage 1 AUC ≥ 0.65 | **単勝・複勝・馬連・ワイド・馬単**（5種のみ） | 既存 VPS 2GB CPU — 追加コスト不要 |
| **Phase 5b: DL 追加** | Step B（Betting Transformer） | `PHASE5_SCOPE=DL` + Phase 5a Val ROI > −5% + EV MAE < 0.1 検証済み | 5種 + 3連複・3連単追加可（race_pair_odds SLA 3 収集確認後） | GPU インスタンス必須（§10-9 参照） |
| **Phase 5c: RL 追加** | Step C（PPO + MCTS） | `PHASE5_SCOPE=RL` + Phase 5b Val Sharpe > 0.0 | 同上 | GPU + CPU 併用 |

> **3連複・3連単の扱い**: Phase 5a では EV 計算に必要な `race_pair_odds` のリアルタイム収集（AREA-06 SLA 3 追加）が未確認のため、スコープ外とする。Phase 5a 稼働後に AREA-06 で収集追加が承認された時点で Phase 5b 以降で追加する。

#### 10-1-3. 全体アーキテクチャフロー

```
Stage 1〜3 出力
 ├── win_prob[N], place_prob[N], show_prob[N]     (N=出走頭数)
 └── odds_win[N], odds_place[N]  ← Layer 5 T-15 スナップショット固定
                                    (snapshot_at < race_start_time 必須)

         │ SLA 3 トリガ後、非同期ジョブとしてキュー投入
         ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 4: Betting Strategy Layer                                │
│ 【推論実行方式: 非同期バッチ推論】                               │
│ 発走 10 分前（SLA 3 + 5 分以内）に書込完了を目標とする          │
│                                                                │
│  Step A: EV 計算エンジン  ← Phase 5a〜5c 全フェーズ共通         │
│       │                                                        │
│  Step B: Betting Transformer（DL）← Phase 5b 以降             │
│       │                                                        │
│  Step C: PPO + MCTS ← Phase 5c 以降                           │
│       │                                                        │
│  Step D: Kelly + バランスフィルタ ← Phase 5a〜5c 全フェーズ     │
│       │                                                        │
│  bet_recommendations テーブル書込                               │
│  （stage カラムでどのフェーズ生成かをトレース）                  │
└────────────────────────────────────────────────────────────────┘

         │
         ▼
GET /api/v1/races/{race_id}/bet-recommendations
（推論完了前アクセス時は空配列を返却）
```

#### 10-1-4. 推論実行 SLA

| フェーズ | 推論時間目標 | 環境 |
|---|---|---|
| Phase 5a（EV のみ） | ≤ 5 秒/レース | VPS 2GB CPU |
| Phase 5b（DL 追加） | ≤ 30 秒/レース | 同上 |
| Phase 5c（MCTS 追加） | ≤ 120 秒/レース | 同上 |

**タイムアウト時フォールバック**: Phase 5c がタイムアウトした場合、Phase 5b（DL スコアのみ）の結果を採用して `bet_recommendations` に書込む（`stage='DL'` で記録）。

---

### 10-2. 期待値計算エンジン（T-10: Bet EV Model）

#### 10-2-1. T-10 予測ターゲット正式定義

| 属性 | 定義 |
|---|---|
| **ID** | T-10（AREA-07 固有 / AREA-01 では T-12） |
| **名称** | `bet_recommendation_score` |
| **出力 1** | 馬券種×馬（組み合わせ）ごとの期待値スコア（EV）: `NUMERIC(7,4)` |
| **出力 2** | 購入推薦バイナリ（EV・的中率バランス・Kelly 条件を満たす場合 TRUE）: `BOOLEAN` |
| **出力 3** | Kelly Criterion ベット比率（資金に対する推奨投資割合）: `NUMERIC(5,4)` |
| **Stage** | Stage 4（`PHASE5_BETTING_STRATEGY=true` かつ AUC ≥ 0.65 通過後のみ） |
| **依存** | T-1 / T-2 / T-3（Stage 1 出力必須）、Layer 5 `odds_snapshot`（T-15 バンドル確定後） |
| **DB 格納** | `bet_recommendations`（新設テーブル — §10-8 参照） |
| **自己参照** | 推論入力への再帰利用禁止 |

#### 10-2-2. 基礎確率（修正 Harville 式）

T-1 の出力 `win_prob[i]` は「1着確率 P(1st)」、T-2 の出力 `place_prob[i]` は「2着以内の確率 P(top 2)」として定義されているため、Harville 式の分子には `place_prob[j] - win_prob[j]`（= P(exactly 2nd) の近似）を使用する。

```python
# 1着確率: T-1 出力をそのまま使用（softmax 正規化済み、sum ≈ 1.0）
p1 = win_prob  # shape: (N,)

# P(exactly 2nd) の近似: place_prob[j] には win_prob[j] 分が包含されるため除去
# NOTE: この近似は T-1/T-2 が独立ヘッドで出力されている前提に依存する。
# 厳密な Harville 式（win_prob[j] / sum_k≠i win_prob[k]）は独立同一分布仮定の近似であり、
# 本モデルの独立出力設計では上記のほうが概念的に整合的。
p_exactly_2nd = {j: max(place_prob[j] - win_prob[j], 1e-9) for j in range(N)}

# 2着確率（馬 i が 1着の条件下で馬 j が 2着）
def p2_given_1(i: int, j: int) -> float:
    if i == j:
        return 0.0
    denom = max(1.0 - win_prob[i], 1e-9)
    return p_exactly_2nd[j] / denom

# P(exactly 3rd) の近似: show_prob[k] = P(top 3) から P(top 2) を除去
p_exactly_3rd = {k: max(show_prob[k] - place_prob[k], 1e-9) for k in range(N)}

# 3着確率（馬 i,j が 1,2着の条件下で馬 k が 3着）
def p3_given_12(i: int, j: int, k: int) -> float:
    if k in (i, j):
        return 0.0
    # 【修正】1.0 - win_prob[i] - p_exactly_2nd[j] は負値になりうるため max(..., 1e-9) で保護
    # 例: win_prob[i]=0.70, place_prob[j]=0.65, win_prob[j]=0.10
    #     → p_exactly_2nd[j]=0.55, 1.0-0.70-0.55 = -0.25 → ゼロ除算ガードのみでは不十分
    remaining = max(1.0 - win_prob[i] - p_exactly_2nd[j], 1e-9)
    return p_exactly_3rd[k] / remaining
```

#### 10-2-3. 馬券種ごとの的中確率と期待値（Phase 5a スコープ: 5種）

```python
# --- 単勝 (WIN) ---
hit_prob_win[i]    = win_prob[i]
ev_win[i]          = hit_prob_win[i] * odds_win[i]

# --- 複勝 (PLACE) ---
hit_prob_place[i]  = show_prob[i]          # P(top 3) を複勝的中確率として使用
ev_place[i]        = hit_prob_place[i] * odds_place[i]   # odds_place = 複勝オッズ中央値

# --- 馬連 QUINELLA (i,j どちらが 1着でも可) ---
hit_prob_quinella[i][j] = (
    win_prob[i] * p2_given_1(i, j) +
    win_prob[j] * p2_given_1(j, i)
)
ev_quinella[i][j]  = hit_prob_quinella[i][j] * odds_quinella[i][j]

# --- ワイド WIDE (i,j ともに 3着以内): Harville 式解析的計算 ---
# 【修正】Monte Carlo (n=10,000) から解析的計算に置き換え。
# 旧実装の [:3] スライス方式はインデックスバイアス（馬番小の馬に偏る）と
# 試行数不足による系統的過小評価を含んでいた。
# 解析的計算により SLA ≤ 5 秒を保ちつつバイアスを排除する。
#
# Phase 5b 以降で高精度が必要な場合は、numpy.random.choice による
# replace=False 加重サンプリング（weights=win_prob 正規化値）への移行を検討する。

def hit_prob_wide_analytical(i: int, j: int) -> float:
    """
    Harville 式を用いた解析的ワイド的中確率計算。
    P(i∈top3 ∧ j∈top3) = Σ（i,j の 1着・2着・3着への全順列配置）

    計算量: O(N) — 18 頭フルゲートで約 16 回ループ（SLA ≤ 5 秒問題なし）。
    旧 Monte Carlo 実装（≒2,750 万回 Python ループ）に比べ数百倍高速。

    Args:
        i, j: 対象馬インデックス

    Returns:
        P(i ∈ top3 and j ∈ top3) の解析的推定値
    """
    total = 0.0
    for first in (i, j):
        second = j if first == i else i
        # パターン1: first=1着, second=2着
        total += win_prob[first] * p2_given_1(first, second)
        # パターン2: first=1着, second=3着（2着は i,j 以外の馬 mid）
        for mid in range(N):
            if mid in (i, j):
                continue
            total += (win_prob[first]
                      * p2_given_1(first, mid)
                      * p3_given_12(first, mid, second))
    return min(total, 1.0)

ev_wide[i][j] = hit_prob_wide_analytical(i, j) * odds_wide[i][j]

# --- 馬単 EXACTA (i=1着 かつ j=2着) ---
hit_prob_exacta[i][j] = win_prob[i] * p2_given_1(i, j)
ev_exacta[i][j]       = hit_prob_exacta[i][j] * odds_exacta[i][j]
```

> **3連複・3連単の EV 計算（Phase 5b 以降・race_pair_odds SLA 3 収集確認後）**:
> `race_pair_odds`（SLA 5 確定後）は推論時に利用不可。代替として `race_odds_snapshot`（SLA 3）の
> 馬連・ワイドオッズから回帰推定した `implied_trifecta_odds` を使用する。
> `snapshot_at < race_start_time` を必須条件とし、回帰モデルは過去 5 年分を使ったオフライン学習とする。
> 詳細設計は Phase 5b 開始時に AREA-07 §10 に追記する。

---

### 10-3. Phase A: DL ベースライン（Betting Transformer）

#### 10-3-1. モデル定義

| 要素 | 仕様 |
|---|---|
| **アーキテクチャ** | Set Transformer（馬間の順序不変な集合処理） |
| **入力次元** | 1頭あたり: [win_prob, place_prob, show_prob, odds_win, odds_place, rel_odds_rank, ev_win, ev_place] = 8次元 |
| **レース特徴** | [distance_enc, surface_enc, grade_enc, horse_num] = 4次元（全馬共通） |
| **エンコーダ** | 2層 Multi-Head Self-Attention (heads=4, d_model=64) |
| **出力ヘッド** | 馬券種ごとに独立した MLP ヘッド（§10-3-2 参照） |
| **学習目標** | 実際の払戻に基づく ROI 最大化（回帰 + 購入バイナリの BCE 複合損失） |
| **損失関数** | `L = λ1 * MSE(ev_pred, ev_actual) + λ2 * BCE(buy_pred, buy_actual)` (λ1=0.7, λ2=0.3) |
| **訓練分割** | Train: 2020〜2023 / Val: 2024-01〜2024-09 / Test: 2024-10〜2025-12（時系列順、シャッフル禁止） |

#### 10-3-2. 対称/非対称ヘッドの分離

馬連・ワイドは対称的（i,j の順序に依存しない）、馬単・3連単は非対称（順序を保持）。これを別ヘッドに分離する。

```python
class BettingTransformer(nn.Module):
    def __init__(self, n_horse_features: int = 8, n_race_features: int = 4,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.horse_embed = nn.Linear(n_horse_features, d_model)
        self.race_embed  = nn.Linear(n_race_features, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- per-horse ヘッド ---
        self.head_win    = nn.Linear(d_model, 1)       # 単勝  (per horse)
        self.head_place  = nn.Linear(d_model, 1)       # 複勝  (per horse)

        # --- ペア対称ヘッド: 入力は常に horse_id 昇順にソートして対称性を保証 ---
        self.head_quinella = nn.Linear(d_model * 2, 1)  # 馬連  (対称)
        self.head_wide     = nn.Linear(d_model * 2, 1)  # ワイド (対称)

        # --- ペア非対称ヘッド: 入力順序を保持（1着候補を先に結合） ---
        self.head_exacta   = nn.Linear(d_model * 2, 1)  # 馬単  (非対称)

        # --- トリプルヘッド（Phase 5b 以降で有効化） ---
        self.head_trio      = nn.Linear(d_model * 3, 1)  # 3連複 (対称)
        self.head_trifecta  = nn.Linear(d_model * 3, 1)  # 3連単 (非対称)

    def forward(self, horse_feats: torch.Tensor, race_feats: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder として動作し、各馬の文脈埋め込みベクトルを返す。
        各ヘッドは forward() の外から呼び出す（下記「ヘッド呼び出し規約」参照）。

        Args:
            horse_feats: (batch, n_horse, 8)
            race_feats:  (batch, 4)
        Returns:
            x: (batch, n_horse, d_model) — 各ヘッドへの入力ベクトル
        """
        x = self.horse_embed(horse_feats)               # (B, N, d)
        r = self.race_embed(race_feats).unsqueeze(1)    # (B, 1, d)
        x = x + r                                       # レース特徴を加算
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        return x  # (B, N, d_model)

    # --- ヘッド呼び出し規約（Phase 5b 学習ループで実装すること） ---
    # x = model(horse_feats, race_feats)                # (B, N, d)
    # 単勝スコア : model.head_win(x[:, i, :])           # (B, 1)
    # 複勝スコア : model.head_place(x[:, i, :])         # (B, 1)
    # 馬連スコア : model.head_quinella(                  # (B, 1)
    #     torch.cat([x[:, min(i,j), :], x[:, max(i,j), :]], dim=-1))
    # ワイドスコア: model.head_wide(同上・昇順ソート)    # (B, 1)
    # 馬単スコア : model.head_exacta(                    # (B, 1)
    #     torch.cat([x[:, i, :], x[:, j, :]], dim=-1))  # i=1着候補を先
    # ─────────────────────────────────────────────────────
    # 上記呼び出しパターンを Phase 5b 実装 PR の train_loop.py で必ず実装し、
    # estimate_roi() も DL ヘッド出力ベースに差し替えること。

    def estimate_roi(self, state: dict, selected: list) -> float:
        """
        MCTS の value function として使用する ROI 期待値推定。
        選択済み馬券候補セットの合算期待収益率を DL スコアで近似する。

        【注意】Phase 5c 初期実装では ev_score 平均を value estimate とする暫定実装。
        Phase 5b 完了後、ヘッド出力ベースのクリティックに差し替えること。

        Args:
            state:    レース状態 dict（'horse_feats', 'race_feats' を含む）
            selected: 現在選択済み BetCandidate のリスト

        Returns:
            推定 ROI（期待値ベース）
        """
        if not selected:
            return 0.0
        horse_feats = state['horse_feats']   # (1, N, 8)
        race_feats  = state['race_feats']    # (1, 4)
        # 前向きパスは評価モードで実行（勾配不要）
        with torch.no_grad():
            _ = self.forward(horse_feats, race_feats)
        # Phase 5c 初期実装: 選択候補の ev_score 平均を value estimate とする
        return sum(c.ev_score for c in selected) / max(len(selected), 1)
```

> **⚠ 注意（DL ヘッド呼び出し — Phase 5b 実装時必須）:** `forward()` は Encoder 特徴量ベクトルを返す設計であり、`head_win` 等のヘッドは `forward()` 内で呼ばれない。Phase 5b の学習ループ実装 PR では上記「ヘッド呼び出し規約」コメントに従い各ヘッドを明示的に呼び出すこと。また `estimate_roi()` の DL ヘッド出力への差し替えも同 PR で実施すること。この対応が完了するまで「DL スコアリング」は実質 EV 平均スコアと等価であることを認識した上で Phase 5b を開始すること。

---

### 10-4. Phase B: 強化学習エージェント（PPO + MCTS）

#### 10-4-1. PPO + MCTS 連携方式の明文化

```
【推論時 MCTS / オフライン PPO 方式（Phase 5c 採用アーキテクチャ）】

本実装では AlphaZero 式のオンライン共同学習ではなく、以下の分離方式を採用する:

■ 学習時（オフライン）
  PPO は 2020〜2023 の確定払戻データを用いてオフラインで Actor-Critic を学習する。
  MCTS は推論時にのみ使用し、学習ループには参加しない（VPS 2GB 環境でのコスト制約から）。

■ 推論時（リアルタイム非同期）
  BettingTransformer（DL）の出力を初期 value estimate として MCTS ツリー探索を実行し、
  最終推薦セットを選択する。

■ PPO の advantage 推定
  1レース = 1エピソード（単一ステップ）のため GAE は使用しない。
  ベースライン差分  r − V(s)  のみで advantage を計算する。
  ※ 単一ステップ強化学習は文脈バンディット問題と等価であり PPO を適用可能。

■ 将来の移行計画
  Phase 5c 稼働後に十分なデータが蓄積した段階で AlphaZero 式への移行を検討する。
  その際には bet_recommendations テーブルの蓄積データを MCTS 自己対戦データとして活用する。

■ 採用理由
  VPS 2GB 環境でのリアルタイム MCTS + 学習同時実行は非現実的。
  分離方式により推論 SLA（≤ 120 秒/レース）を保ちながらコストを最小化する。
```

#### 10-4-2. 強化学習定式化

| 要素 | 定義 |
|---|---|
| **状態 s** | `[win_prob[N], place_prob[N], show_prob[N], odds_win[N], odds_place[N], ev_matrix[M]]`（M = 全馬券候補数） |
| **行動 a** | 馬券候補セット S ⊆ 全候補からの組み合わせ選択（最大 K=5 件まで） |
| **報酬 r** | `sum(payout[s∈S]) - sum(stake[s∈S])`（発走後確定。**推論時には使用不可**） |
| **ポリシー** | PPO (clip_eps=0.2) |
| **価値関数** | BettingTransformer の `estimate_roi()` を初期 value estimate として使用 |
| **探索** | MCTS で馬券組み合わせツリーを展開し、期待リターンが最大のパスを選択 |

#### 10-4-3. MCTS 実装

```python
from typing import List
from dataclasses import dataclass, field

@dataclass
class BetCandidate:
    bet_type:    str
    horse_ids:   List[int]
    ev_score:    float
    hit_prob:    float
    odds:        float
    kelly_frac:  float = 0.0

@dataclass
class MCTSNode:
    selected:  List[BetCandidate] = field(default_factory=list)
    remaining: List[BetCandidate] = field(default_factory=list)
    visits:    int = 0
    value_sum: float = 0.0
    parent:    'MCTSNode' = None
    children:  List['MCTSNode'] = field(default_factory=list)


class BettingMCTS:
    """
    馬券組み合わせ選択のためのモンテカルロ木探索。
    ノード : 現在選択済み馬券候補セット
    展開   : 未選択候補から 1 件追加
    評価   : DL value network による EV 推定 + UCB1
    ロールアウト: 最大深さ K=5 まで greedy 展開

    【計算量制約 — Phase 5c 参考】
    n_simulations=200 の時、18頭フルゲートでの候補ノード数は
    Phase 5a スコープ（5馬券種）で最大 ≒ 6,360。
    VPS 2GB 2vCPU での実測ベンチマークを Phase 5c 実装開始前に実施し、
    n_simulations を環境に合わせて調整すること（推奨: 50〜100 から開始）。
    """

    MAX_DEPTH = 5  # 1レースあたり最大推薦件数

    def __init__(self, candidates: List[BetCandidate],
                 value_net: 'BettingTransformer',
                 n_simulations: int = 200,
                 c_puct: float = 1.4):
        self.candidates    = candidates
        self.value_net     = value_net
        self.n_simulations = n_simulations
        self.c_puct        = c_puct

    def search(self, state: dict) -> List[BetCandidate]:
        root = MCTSNode(selected=[], remaining=list(self.candidates))
        for _ in range(self.n_simulations):
            node  = self._select(root)
            value = self._evaluate(node, state)
            self._backpropagate(node, value)
        return self._best_path(root)

    def _ucb1(self, node: MCTSNode) -> float:
        import math
        if node.visits == 0:
            return float('inf')
        exploitation = node.value_sum / node.visits
        exploration  = self.c_puct * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )
        return exploitation + exploration

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(node.children, key=self._ucb1)
        if len(node.selected) < self.MAX_DEPTH and node.remaining:
            self._expand(node)
            node = node.children[0] if node.children else node
        return node

    def _expand(self, node: MCTSNode) -> None:
        for candidate in node.remaining:
            child = MCTSNode(
                selected  = node.selected + [candidate],
                remaining = [c for c in node.remaining if c is not candidate],
                parent    = node,
            )
            node.children.append(child)

    def _evaluate(self, node: MCTSNode, state: dict) -> float:
        return self.value_net.estimate_roi(state, node.selected)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits    += 1
            node.value_sum += value
            node            = node.parent

    def _best_path(self, root: MCTSNode) -> List[BetCandidate]:
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.visits)
        return node.selected


# --- PPO 学習設定 ---
PPO_CONFIG = {
    'algorithm'   : 'PPO',
    'clip_eps'    : 0.2,
    'actor_arch'  : [256, 128, 64],   # 3層 MLP
    'critic_arch' : [256, 128, 1],    # 3層 MLP
    'input'       : 'BettingTransformer エンコード済み state vector',
    'episode_def' : '1レース = 1エピソード（単一ステップ; バンディット等価）',
    'advantage'   : 'r - V(s)（GAE 不使用）',
    'train_data'  : '2020〜2023 全レース（時系列順、シャッフル禁止）',
    'batch_size'  : 64,               # レース単位
    'gamma'       : 1.0,              # 単一ステップのため割引不要
    'gate'        : 'PHASE5_BETTING_STRATEGY=true かつ Stage 1 AUC ≥ 0.65 通過後のみ学習実行',
}
```

---

### 10-5. Hybrid アンサンブル構成

各フェーズでのスコアリング統合方針を示す。

| フェーズ | スコアリングソース | 統合方法 |
|---|---|---|
| Phase 5a | Step A: EV スコア（算術） | EV をそのままランキング指標に使用 |
| Phase 5b | Step A の EV + Step B の DL スコア | 加重平均: `0.5 * ev_score + 0.5 * dl_score`（α は Val データで調整） |
| Phase 5c | Step A + Step B + Step C（MCTS 最終推薦） | MCTS が選択したセットを最終候補、Step B スコアを tiebreaker に使用 |

> **アンサンブル重みの更新**: フェーズ移行後に Val 期間 ROI をモニタリングし、
> 重みを月次で再調整する。重み変更は `model_version` カラムで追跡する。

---

### 10-6. Kelly Criterion による bet sizing

#### 10-6-1. 分割 Kelly（Fractional Kelly）

```python
def kelly_fraction(ev: float, hit_prob: float, odds: float,
                   fraction: float = 0.25) -> float:
    """
    分割 Kelly（フル Kelly の fraction 倍）でベットサイズを算出。
    fraction=0.25 は ConoHa VPS 2GB 環境の資金管理上限として設定。

    Args:
        ev       : 期待値 (= hit_prob * odds)
        hit_prob : 的中確率
        odds     : 払戻オッズ（1.0 = 元返し）
        fraction : Kelly 係数の縮小率（デフォルト 0.25）

    Returns:
        ベット比率（0.0〜0.25、資金全体に対する割合）
    """
    if ev <= 1.0 or odds <= 1.0:
        return 0.0
    # Kelly = (EV - 1) / (odds - 1)
    full_kelly = (ev - 1.0) / (odds - 1.0)
    return min(full_kelly * fraction, 0.25)  # 上限 25%
```

---

### 10-7. 的中率バランス制約

#### 10-7-1. フィルタリング条件

| 条件 | 閾値 | 目的 |
|---|---|---|
| **EV 下限** | EV > 1.05 | 期待値マイナスの馬券を排除 |
| **的中率 下限** | hit_prob > 0.03 | 的中確率ほぼゼロの博打的馬券を排除 |
| **的中率 上限** | hit_prob < 0.75 | 過剰人気による控除率高騰馬券を排除 |
| **Kelly 下限** | kelly_frac > 0.01 | 微小ベットの排除 |
| **1レース最大推薦件数** | K ≤ 5 件 | 適度な分散投資 |
| **馬券種上限** | 同一馬券種は最大 2 件まで | 特定馬券種への過集中防止 |

```python
from collections import defaultdict
from typing import List

def apply_balance_filter(candidates: List[BetCandidate],
                         max_per_type: int = 2,
                         k: int = 5) -> List[BetCandidate]:
    """
    的中率バランス制約を適用し、最終推薦リストを返す。

    【修正】kelly_frac を事前計算して BetCandidate フィールドに代入してからフィルタする。
    旧実装では apply_balance_filter 内で kelly_fraction() を再呼び出しており、
    BetCandidate.kelly_frac が 0.0 のまま DB に書き込まれる不整合が生じていた。
    フィルタ判定と DB 格納値を同一の事前計算値で統一する。

    Args:
        candidates   : Step C（または Step A/B）の出力候補リスト
        max_per_type : 同一馬券種の最大推薦件数
        k            : 最終リストの上限件数

    Returns:
        フィルタ済み推薦リスト（EV 降順、最大 k 件）
    """
    # kelly_frac を事前計算して BetCandidate に代入（DB 格納値とフィルタ判定値を一致させる）
    for c in candidates:
        c.kelly_frac = kelly_fraction(c.ev_score, c.hit_prob, c.odds)

    filtered = [
        c for c in candidates
        if c.ev_score    > 1.05
        and 0.03         < c.hit_prob < 0.75
        and c.kelly_frac > 0.01          # 事前計算済み値で判定（DB 格納値と一致）
    ]

    # 馬券種ごとに EV 上位 max_per_type 件のみ残す
    by_type: dict[str, list] = defaultdict(list)
    for c in sorted(filtered, key=lambda x: x.ev_score, reverse=True):
        if len(by_type[c.bet_type]) < max_per_type:
            by_type[c.bet_type].append(c)

    result = [c for cs in by_type.values() for c in cs]
    # 全体で EV 上位 k 件
    return sorted(result, key=lambda x: x.ev_score, reverse=True)[:k]
```

---

### 10-8. DB スキーマ・API 設計

#### 10-8-1. `bet_recommendations` テーブル DDL

```sql
-- 配置: Layer 5.5（推薦出力層）
-- Layer 3（集計特徴量スナップショット）とは意味論が異なる推薦出力データ。
-- AREA-06 の Layer 定義を次回改訂時に Layer 6 (Recommendation Output Layer) として正式化する。
--
-- Alembic 戦略: PHASE5_BETTING_STRATEGY フラグの有無に関わらず、
-- Alembic マイグレーション適用時に常にテーブルを作成する。
-- フラグ制御はアプリケーション層でのみ行い、テーブル存在とデータ書込の有無を分離する。

CREATE TABLE bet_recommendations (
    recommendation_id    BIGSERIAL       PRIMARY KEY,
    race_id              VARCHAR(20)     NOT NULL,
    model_version        VARCHAR(50)     NOT NULL,
    recommended_at       TIMESTAMPTZ     DEFAULT NOW(),

    -- 馬券種（PLACE で統一: race_odds_snapshot.snapshot_type と一致）
    bet_type             VARCHAR(10)     NOT NULL
                         CHECK (bet_type IN (
                             'WIN', 'PLACE', 'QUINELLA', 'WIDE',
                             'EXACTA', 'TRIFECTA_BOX', 'TRIFECTA'
                         )),

    -- 対象馬（最大 3 頭: 1着/2着/3着候補）
    horse_id_1           VARCHAR(20)     NOT NULL,
    horse_id_2           VARCHAR(20),    -- 馬連/ワイド/馬単/3連複/3連単 で使用
    horse_id_3           VARCHAR(20),    -- 3連複/3連単 で使用

    -- スコア
    ev_score             NUMERIC(7,4)    NOT NULL,   -- 期待値
    hit_probability      NUMERIC(5,4)    NOT NULL,   -- 的中確率
    implied_odds         NUMERIC(7,1)    NOT NULL,   -- 使用したオッズ
    kelly_fraction       NUMERIC(5,4)    NOT NULL,   -- 推奨ベット比率（apply_balance_filter 事前計算値）

    -- 購入推薦フラグ（フィルタ通過済み）
    is_recommended       BOOLEAN         NOT NULL    DEFAULT FALSE,

    -- 生成フェーズ追跡（タイムアウト時フォールバックのトレース用）
    stage                VARCHAR(10)     NOT NULL    DEFAULT 'EV_ONLY'
                         CHECK (stage IN ('EV_ONLY', 'DL', 'RL')),

    -- オッズスナップショット紐付け（テンポラルリーク防止; NOT NULL で参照整合を保証）
    odds_snapshot_id     BIGINT          NOT NULL
                         REFERENCES race_odds_snapshot(snapshot_id)
    -- NOTE: Phase 5a スコープ（5馬券種）では対応 snapshot_type が存在するため NOT NULL 成立。
    --       3連複/3連単 追加時の設計は Phase 5b 定義時に再検討する。
);

-- UNIQUE 制約: PostgreSQL 式のかんすう表現を含むため CREATE UNIQUE INDEX で定義
CREATE UNIQUE INDEX uq_bet_rec
    ON bet_recommendations (
        race_id, model_version, bet_type, horse_id_1,
        COALESCE(horse_id_2, ''), COALESCE(horse_id_3, '')
    );

CREATE INDEX idx_bet_rec_race_model
    ON bet_recommendations (race_id, model_version, is_recommended);

CREATE INDEX idx_bet_rec_ev
    ON bet_recommendations (race_id, ev_score DESC)
    WHERE is_recommended = TRUE;

-- 月次パーティション（容量管理 — 推奨）
-- PARTITION BY RANGE (recommended_at) でパーティションテーブルに変換することを推奨。
```

#### 10-8-2. 容量見積もり・保持ポリシー

```
【Phase 5a スコープの容量見積もり（18頭フルゲートの場合）】
- 単勝  : 18 件
- 複勝  : 18 件
- 馬連  : C(18,2) = 153 件
- ワイド: C(18,2) = 153 件
- 馬単  : 18×17  = 306 件
合計     : ≒ 648 行/レース × 年間 3,000 レース = 約 194 万行/年/モデルバージョン

【保持ポリシー】
- is_recommended=FALSE のレコード: 発走確定（SLA 5 完了）後 30 日以内に削除
- is_recommended=TRUE のレコード: 永久保持（ROI 集計・モデル評価のため）
- モデルバージョン: 最新 3 バージョンのみ保持（旧バージョンは月次バッチで削除）
- 月次パーティション (PARTITION BY RANGE on recommended_at) を推奨
```

#### 10-8-3. API エンドポイント仕様

**エンドポイント**:

```
GET /api/v1/races/{race_id}/bet-recommendations
```

**クエリパラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `model_version` | string | 省略時は `recommended_at` 最新 | バージョン指定（複数バージョン比較時に使用） |

**レスポンス JSON スキーマ（Phase 5 有効時）**:

```json
{
  "race_id": "2024050101",
  "model_version": "v1.0.0",
  "phase5_enabled": true,
  "recommendations": [
    {
      "bet_type": "WIN",
      "horse_ids": ["3"],
      "ev_score": 1.23,
      "hit_probability": 0.18,
      "implied_odds": 6.8,
      "kelly_fraction": 0.056,
      "is_recommended": true,
      "stage": "EV_ONLY"
    },
    {
      "bet_type": "QUINELLA",
      "horse_ids": ["3", "7"],
      "ev_score": 1.15,
      "hit_probability": 0.09,
      "implied_odds": 12.7,
      "kelly_fraction": 0.038,
      "is_recommended": true,
      "stage": "EV_ONLY"
    }
  ]
}
```

**フォールバックレスポンス（Phase 5 無効 / AUC 未達 / 推論未完了）**:

```json
{
  "race_id": "2024050101",
  "phase5_enabled": false,
  "recommendations": []
}
```

> **model_version 選択戦略**: クエリパラメータ省略時は `recommended_at` が最新のバージョンを返す。
> 複数バージョンを比較したい場合は `?model_version=v1.0.0` で指定可能。

#### 10-8-4. Redis キャッシュ設定（AREA-03 §5-1 追記要件）

AREA-03 §5-1 に以下のキャッシュキー定義を追記すること:

```
キー    : bet_rec:{race_id}:{model_version}
TTL     : 発走時刻まで（race_start_time - now() 秒）
無効化  : bet_recommendations テーブルへの書込完了時に即時 invalidate
```

#### 10-8-5. AREA-01 / AREA-03 への同時更新要件

| 変更箇所 | 内容 |
|---|---|
| AREA-01 §2 予測ターゲット定義テーブル | T-12 行を追加: `bet_recommendation_score` — `NUMERIC(7,4) / BOOLEAN` |
| AREA-01 §3-3 `prediction_results` テーブル | `bet_recommendations` テーブルへの FK 参照コメントを追記 |
| AREA-01 §3-3 | `bet_recommendations` テーブル DDL を Layer 5.5 スキーマ一覧に追加 |
| AREA-03 §3-2 APIエンドポイント一覧 | `GET /api/v1/races/{race_id}/bet-recommendations` を追加 |
| AREA-03 §5-1 キャッシュキー定義 | `bet_rec:{race_id}:{model_version}` を追加（上記参照） |

> **注意**: AREA-01 の T-10 = ペースカテゴリ予測、T-11 = ラップ系列予測は変更しない。
> 馬券推薦は AREA-01 文脈では T-12 として採番する。
> AREA-07 と AREA-01 でターゲット番号が乖離しているため、各文書内での連番を優先し
> 相互参照は文書名（AREA-07 T-10 / AREA-01 T-12）で明示すること。

---

### 10-9. 評価指標・CI ゲート

#### 10-9-1. 評価指標

| 指標 | 定義 | 目標値 |
|---|---|---|
| **ROI** | `(Σ payout − Σ stake) / Σ stake × 100` (%) | > 0%（プラス収支） |
| **Sharpe Ratio** | `mean(r_race) / std(r_race)`（レースごとリターン系列） | > 0.3 |
| **的中率（馬券種別）** | 推薦馬券が的中した割合 | hit_prob 下限 3%〜上限 75% の範囲内 |
| **平均オッズ** | 推薦馬券の implied_odds 平均 | 単勝 ≥ 3.0 倍（過剰人気排除確認） |
| **Precision@K** | is_recommended=TRUE のうち実際に的中した割合 | > hit_prob の平均（ランダム基準超え） |
| **最大ドローダウン（MDD）** | 資金の最大ドローダウン率 | < 30% |
| **EV 推定精度（MAE）** | `|ev_pred − ev_actual|` の平均絶対誤差 | < 0.1 |
| **Kelly フィルタ後 ROI** | kelly_frac > 0.01 の馬券のみの ROI | > 全体 ROI（フィルタ有効性確認） |
| **ベースライン対 ROI 比** | `ROI(Stage4) − ROI(常に1番人気単勝購入)` | > +3% |
| **ブートストラップ 95% CI** | テスト期間 ROI のブートストラップ信頼区間 | 下限 > −10%（ノイズと区別可能） |

#### 10-9-2. テスト分割ルール

```
Train : 2020-01-01 〜 2023-12-31  （時系列順）
Val   : 2024-01-01 〜 2024-09-30  （時系列順）
Test  : 2024-10-01 〜 2025-12-31  （時系列順、テスト期間中の超過学習禁止）

ランダムシャッフル禁止（AREA-06 テンポラルリーク防止方針に準拠）。
```

> **データ収集期間との整合**: 実収集データは 2020-01-01 より開始（§5-0 参照）。
> Train 期間を 2020-01-01 以前に設定してはならない。

#### 10-9-3. CI ゲート（Stage 4 固有）

| ゲート名 | 条件 | 判定失敗時の動作 |
|---|---|---|
| **前提ゲート** | Stage 1 AUC ≥ 0.65（AREA-07 §6 定義の CI ゲートを流用） | Stage 4 学習・推論をスキップ |
| **EV 推定精度ゲート** | DL ベースラインの EV MAE < 0.1（Validation 期間） | ModelRegistry へ登録せず draft バージョンとして保持 |
| **ROI ゲート（Validation）** | Val 期間 ROI > −5% | PHASE5 フラグを自動 OFF にして Human review 依頼 |
| **Sharpe ゲート** | Val 期間 Sharpe Ratio > 0.0 | 同上 |
| **リーク検出ゲート** | 全特徴量の `snapshot_at < race_start_time` 件数 = 100%（サンプル 100 件検査） | 学習・推論パイプラインを緊急停止 |
| **ベースライン超過ゲート** | ROI ベースライン差 > +3%（Test 期間） | Phase 移行を保留し原因分析を実施 |

#### 10-9-4. 学習インフラコストと推定

| フェーズ | 学習環境 | 推定コスト/学習サイクル | 月次再学習想定コスト | 推論時モデル常駐メモリ（参考） |
|---|---|---|---|---|
| **Phase 5a（EV のみ） ** | 既存 VPS 2GB CPU | 追加コスト不要 | 0 円 | 追加なし（Python 関数のみ） |
| **Phase 5b（DL 追加）** | GCP Compute Engine n1-standard-4 + T4 GPU (preemptible) ≒ $0.12/時 | 推定学習時間 8〜15 時間 → 約 $1.0〜1.8（≒ 150〜270 円） | 月 1 回: ≒ 300 円/月 | BettingTransformer (d_model=64, 2 層): **約 2〜5 MB**（float32）— VPS 2GB 残余メモリ内で許容範囲 |
| **Phase 5c（RL 追加）** | GPU + CPU 併用（PPO 学習）/ VPS 2GB CPU のみ（推論） | 追加 $2〜5/月（PPO 学習分） | ≒ 300〜750 円/月 | PPO Actor-Critic MLP ([256,128,64]): **約 1〜2 MB** 追加 — 合計 < 10 MB |
| **implied_trifecta_odds 回帰** | CPU のみ | コスト無視可能 | — | < 1 MB |

> **メモリ見積もり根拠（Phase 5b）**: d_model=64, n_heads=4, 2 層 Transformer の総パラメータ数は約 130K、float32 換算で約 0.5 MB。推論時の中間 tensor（バッチ 1 レース分: 18 頭 × 64 次元）は約 4 KB。既存 Stage 1〜3 LightGBM モデル + PostgreSQL + Redis との競合を考慮しても **VPS 2GB 内で十分余裕あり**（Phase 5b 実装 PR で `free -m` による実測確認を必須とする）。

---

### 10-10. テンポラルリーク防止

#### 10-10-1. チェックリスト（Stage 4 固有）

| チェック項目 | 対象 | 判定 |
|---|---|---|
| `odds_win` / `odds_place` は `snapshot_at < race_start_time` のレコードのみ使用 | EV 計算 / DL 入力 | 必須 |
| 払戻金（SLA 5 確定後）は学習ラベル（報酬・損失）としてのみ使用 | RL 報酬 / DL 損失 | 必須 |
| 3連複/3連単 `implied_trifecta_odds` の回帰モデルに SLA 5 データを推論入力として使用しない | Step A（Phase 5b 以降） | 必須 |
| RL の状態特徴量にレース結果（着順・払戻）を含めない | PPO state | 必須 |
| `bet_recommendations.odds_snapshot_id` が `snapshot_at < race_start_time` を指している | DB 制約 | 必須 |
| Train/Val/Test 分割がランダムシャッフルを含まない | 全モデル学習 | 必須 |
| T-10 の出力を他ターゲット（T-1〜T-9）の学習特徴量として使用しない | 自己参照禁止 | 必須 |
| RL データローダーに Feature Store の `snapshot_at < race_start_time` フィルタを適用 | PPO 学習データ構築 | 必須 |

#### 10-10-2. フィーチャーフラグ制御

```python
# config.py

import os

# Phase 5 の有効化スイッチ
PHASE5_BETTING_STRATEGY: bool = (
    os.getenv("PHASE5_BETTING_STRATEGY", "false").lower() == "true"
)

# フェーズスコープ（EV_ONLY / DL / RL）
PHASE5_SCOPE: str = os.getenv("PHASE5_SCOPE", "EV_ONLY")


# pipeline/stage4.py

def stage1_gate_passed() -> bool:
    """
    AREA-07 §6 の CI ゲート結果を参照し、Stage 1 AUC ≥ 0.65 を確認する。
    判定方法: ModelRegistry テーブルの最新 production バージョンの
    evaluation_metrics JSONB カラムから 'auc_stage1' を取得して閾値と比較する。

    例:
        SELECT (evaluation_metrics->>'auc_stage1')::float >= 0.65
        FROM model_registry
        WHERE stage = 'stage1' AND status = 'production'
        ORDER BY registered_at DESC LIMIT 1;

    Returns:
        True  = AUC ≥ 0.65（Stage 4 実行可）
        False = 閾値未達またはレコード不在（Stage 4 スキップ）
    """
    # 実装は AREA-07 §6 の ModelRegistry スキーマ確定後に実装すること
    raise NotImplementedError("AREA-07 §6 ModelRegistry 参照実装を追加すること")


def run_stage4(race_id: str, state: dict) -> None:
    if not PHASE5_BETTING_STRATEGY:
        logger.info("PHASE5_BETTING_STRATEGY is disabled. Stage 4 skipped.")
        return

    if not stage1_gate_passed():   # AUC ≥ 0.65 チェック（AREA-07 §6 参照）
        logger.warning("Stage 1 CI gate not passed. Stage 4 aborted.")
        return

    # Step A: EV 計算（全フェーズ共通）
    candidates = run_ev_engine(state)

    # Step B: DL スコアリング（Phase 5b 以降）
    if PHASE5_SCOPE in ("DL", "RL"):
        candidates = run_dl_scoring(candidates, state)

    # Step C: PPO + MCTS（Phase 5c 以降）
    if PHASE5_SCOPE == "RL":
        try:
            candidates = run_mcts(candidates, state)
            stage_label = "RL"
        except TimeoutError:
            logger.warning("MCTS timeout. Falling back to DL results.")
            stage_label = "DL"
    elif PHASE5_SCOPE == "DL":
        stage_label = "DL"
    else:
        stage_label = "EV_ONLY"

    # Step D: Kelly + バランスフィルタ（全フェーズ共通）
    # apply_balance_filter 内で kelly_frac を事前計算・代入してから DB に書込む
    final_recs = apply_balance_filter(candidates)

    # DB 書込
    write_bet_recommendations(race_id, final_recs, stage=stage_label)
```

> **`PHASE4_LAP_PREDICTION` との対称性**: `PHASE5_SCOPE` 変数は `PHASE4_LAP_PREDICTION` と
> 同様のパターンで管理する。本番投入は Stage 1 AUC 安定後に
> `PHASE5_BETTING_STRATEGY=true` かつ `PHASE5_SCOPE=EV_ONLY` を明示設定した場合のみ有効化する。

---

### 10-11. ローカル訓練 + VPS デプロイ方針

> **結論**: 馬券最適化（Stage 4 全フェーズ）は、ローカル PC（大容量 CPU/GPU）での訓練 →
> 訓練済みモデルを VPS へ転送 → VPS は推論専用として稼働、という分離構成が**完全に可能**。
> §10-9-4 のコスト試算はこの前提に基づいて作成されている。

#### 10-11-1. フェーズ別の訓練／推論インフラ分担

| フェーズ | 訓練環境 | 推論環境 | 訓練で必要なこと | VPS でやること |
|---|---|---|---|---|
| **Phase 5a（EV only）** | 不要（純粋演算） | VPS 2GB CPU | なし | Python 関数のみ実行 |
| **Phase 5b（DL）** | ローカル GPU PC | VPS 2GB CPU | BettingTransformer を GPU で学習 | ONNX 推論（1 forward pass/レース） |
| **Phase 5c（RL + MCTS）** | ローカル GPU + CPU | VPS 2GB CPU | PPO Actor-Critic を GPU で学習 | 凍結ポリシー ONNX 推論 + MCTS（CPU） |

**訓練と推論の責務が完全に分離できる理由**:
- Stage 4 の推論パイプラインはレースごとのバッチ処理（発走 10 分前の 1 回のみ）であり、オンライン学習（推論中のパラメータ更新）を必要としない。
- PPO は過去レースデータで**オフライン**学習したポリシーを「凍結した神経回路網」として VPS に配置する。
- MCTS も訓練不要：訓練済み BettingTransformer の `estimate_roi()` forward pass を評価関数として呼ぶだけ（勾配計算なし）。

#### 10-11-2. モデルエクスポート仕様（ONNX 推奨）

```python
# ─── Phase 5b: BettingTransformer のエクスポート ───────────────────────────
import torch
import torch.onnx

model = BettingTransformer(d_model=64, n_heads=4, n_layers=2)
model.load_state_dict(torch.load("betting_transformer.pt"))
model.eval()

# ダミー入力（18 頭 × 入力特徴次元）
dummy_input = torch.randn(1, 18, INPUT_DIM)   # (batch, horses, features)

torch.onnx.export(
    model,
    dummy_input,
    "betting_transformer.onnx",
    input_names=["horse_features"],
    output_names=["ev_scores"],
    dynamic_axes={
        "horse_features": {0: "batch", 1: "horses"},  # 可変頭数に対応
        "ev_scores":      {0: "batch", 1: "horses"},
    },
    opset_version=17,
)

# ─── Phase 5c: PPO Actor ネットワークのエクスポート ─────────────────────────
actor = PPOActor(arch=[256, 128, 64])
actor.load_state_dict(torch.load("ppo_actor.pt"))
actor.eval()

dummy_state = torch.randn(1, STATE_DIM)
torch.onnx.export(
    actor,
    dummy_state,
    "ppo_actor.onnx",
    input_names=["state"],
    output_names=["action_probs"],
    opset_version=17,
)
```

```python
# ─── VPS 側の推論（onnxruntime — PyTorch フル不要）─────────────────────────
import onnxruntime as ort
import numpy as np

_sess_transformer = ort.InferenceSession(
    "models/betting_transformer.onnx",
    providers=["CPUExecutionProvider"],   # VPS は CPU のみ
)
_sess_actor = ort.InferenceSession(
    "models/ppo_actor.onnx",
    providers=["CPUExecutionProvider"],
)

def estimate_roi_onnx(state: dict, selected: list) -> float:
    feats = build_feature_vector(state, selected)   # numpy array
    result = _sess_transformer.run(None, {"horse_features": feats[np.newaxis]})
    return float(result[0].mean())

def get_action_probs_onnx(state_vec: np.ndarray) -> np.ndarray:
    result = _sess_actor.run(None, {"state": state_vec[np.newaxis]})
    return result[0][0]   # shape: (n_candidates,)
```

> **TorchScript も代替可能**: onnxruntime が VPS にインストールできない場合は
> `torch.jit.script(model)` で TorchScript 形式に変換して `torch.jit.load()` で推論する。
> ONNX は PyTorch 非依存のため、メモリフットプリントが小さく推奨。

#### 10-11-3. VPS へのモデル配置フロー（月次再学習サイクル）

```
【月次サイクル — 想定所要時間】

  ローカル PC（GPU）
  ┌─────────────────────────────────────────────────┐
  │ 1. データ取得: Feature Store から前月分取得       │ ~ 5 分
  │ 2. BettingTransformer 再学習（GPU 8〜15 時間）   │ ~ 15 時間
  │ 3. PPO 再学習（PPO_CONFIG 設定、同 GPU）         │ ~ 2〜5 時間
  │ 4. Val 期間 CI ゲート確認（ROI / Sharpe / EV MAE）│ ~ 5 分
  │ 5. ONNX エクスポート（上記コード）               │ < 1 分
  │ 6. モデルファイル転送 (rsync/scp to VPS)         │ < 1 分（< 10 MB）
  └─────────────────────────────────────────────────┘
          ↓  転送完了（scp models/*.onnx vps:/app/models/）
  VPS
  ┌─────────────────────────────────────────────────┐
  │ 7. model_version を ModelRegistry に登録         │
  │ 8. Gunicorn ワーカー restart（ゼロダウンタイム可）│
  │ 9. bet_recommendations に新バージョンで推論開始   │
  └─────────────────────────────────────────────────┘
```

**ファイルサイズ（参考）**:

| モデル | パラメータ数 | ONNX サイズ目安 |
|---|---|---|
| BettingTransformer (d_model=64, 2 層) | ~130 K | ~0.5 MB |
| PPO Actor MLP ([256, 128, 64]) | ~66 K | ~0.3 MB |
| **合計** | — | **< 1 MB**（転送はほぼ瞬時） |

#### 10-11-4. Phase 5c: MCTS 推論時の CPU 負荷管理

MCTS は訓練不要だが、推論時に CPU を消費する。VPS（2vCPU）での運用指針:

| 設定項目 | 推奨値 | 根拠 |
|---|---|---|
| `n_simulations` | **50〜100**（ベンチマーク後調整） | §10-5 の BettingMCTS コメント参照 |
| `MAX_DEPTH` | 5（固定） | 1 レースあたり最大推薦 5 件と一致 |
| タイムアウト設定 | 90 秒（フォールバックに 30 秒余裕） | §10-1-4 Phase 5c SLA 120 秒に対して |

**MCTS がローカル PC 学習不要な理由**:
- MCTS のノード評価は `value_net.estimate_roi()` = BettingTransformer の forward pass のみ。
- `_select` / `_expand` / `_backpropagate` はパラメータを持たない決定的アルゴリズム。
- 「訓練」は BettingTransformer（§10-4）と PPO ポリシー（§10-4 PPO_CONFIG）が担っており、
  MCTS はそれらを実行時に組み合わせる探索手続きにすぎない。

**推奨ベンチマーク手順（Phase 5c 実装開始前）**:

```bash
# VPS 上で実行: 18 頭フルゲートを想定した MCTS 時間計測
python - <<'EOF'
import time, numpy as np
from pipeline.stage4 import BettingMCTS, BettingTransformer

model = BettingTransformer.from_onnx("models/betting_transformer.onnx")
candidates = generate_mock_candidates(n_horses=18)

for n_sim in [50, 100, 200]:
    mcts = BettingMCTS(candidates, model, n_simulations=n_sim)
    t0 = time.time()
    mcts.search(mock_state)
    print(f"n_simulations={n_sim}: {time.time()-t0:.1f}s")
EOF
# 120 秒 SLA を下回る最大の n_simulations を採用する
```

#### 10-11-5. オンライン学習が不要な理由（設計整合確認）

Stage 4 のビジネス要件（発走前に 1 回推論し bet_recommendations に書込む）は、
以下の理由によりオンライン学習（推論中のパラメータ更新）を必要としない:

1. **報酬の遅延**: 払戻確定（SLA 5）は発走後。推論時点では報酬が確定していない。
2. **月次バッチ再学習で十分**: 競馬の配当分布は週単位で大きく変動しないため、
   月 1 回の再学習で特徴量分布の漂流（concept drift）に追従できる。
3. **リーク防止の徹底**: オンライン学習を許容するとレース結果（SLA 5）を学習に
   混入させるリスクが生じ、§10-10 テンポラルリーク防止チェックリストに抵触する。

> **結論**: ローカル PC で訓練 → ONNX/TorchScript でエクスポート → VPS へ転送という
> **オフライン学習 + 推論専用 VPS** の構成は、Stage 4 の全フェーズで設計上の制約なく採用できる。
> §10-9-4 のコスト試算（GPU はローカルまたは GCP Preemptible で学習、VPS は推論のみ）は
> この方針を前提に算出されており、追加の設計変更は不要。

---
