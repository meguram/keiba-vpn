# AREA-05 — モデリング管理要件（LightGBM バッチ推論, 学習パイプライン, SHAP, ModelRegistry, バージョニング, CI ゲート）
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001, DEC-009

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるモデリング管理全般—LightGBM バッチ推論・学習パイプライン・SHAP 解釈・ModelRegistry・バージョニング・CI ゲート—を定義する。テンポラルリーク排除を大前提とし、全工程において `as_of_race_id` によるスナップショット管理を厳守する。

---

## 2. 予測ターゲット定義

| ID | ターゲット名 | 問題設定 | 出力型 | モデル担当 Stage |
|---|---|---|---|---|
| T-1 | win_prob（勝率） | 多クラス分類（1着） | NUMERIC(5,4) | Stage 1 |
| T-2 | place_prob（連対率） | バイナリ分類（2着以内） | NUMERIC(5,4) | Stage 1 |
| T-3 | show_prob（複勝率） | バイナリ分類（3着以内） | NUMERIC(5,4) | Stage 1 |
| T-4 | predicted_win_odds | 回帰 | NUMERIC(7,1) | Stage 1 |
| T-5 | predicted_place_odds | 回帰 | NUMERIC(7,1) | Stage 1 |
| T-6 | win_roi（単回収率） | ポスト計算値（推論後算出） | NUMERIC(7,2) | ポスト処理 |
| T-7 | show_roi（複回収率） | ポスト計算値（推論後算出） | NUMERIC(7,2) | ポスト処理 |
| T-8 | predicted_position | 順位回帰 / ランキング学習 | SMALLINT | Stage 1 |
| T-9 | predicted_running_style | 4値分類（FRONT/STALKER/MID/CLOSER） | VARCHAR(10) | Stage 1 |
| T-10 | pace_category | 3値分類（HIGH/MIDDLE/SLOW） | VARCHAR(10) | Stage 2 |
| T-11 | predicted_lap_sec[] | 時系列回帰（1F単位系列出力） | NUMERIC(4,2)[] | Stage 2 |

> T-6・T-7 はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値とする。値が 100 超 = 期待値プラスのバリューベット候補。

---

## 3. モデルアーキテクチャ

### 3-1. 2ステージ構成

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
│  勝率/連対  ポジション   オッズ予測                           │
│  /複勝率    予測          (回帰)                             │
│  (分類)   (LambdaMART)                                     │
└───────────────────────────────────────────────────────────┘
                │ ポジション予測値を受け渡し
                ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 2: ラップ・ペース予測モデル                             │
│                                                             │
│  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量    │
│                                                             │
│  ┌────────────────────────────────────────┐               │
│  │  Pace & Lap Sequence Model              │               │
│  │  (LightGBM per-furlong → LSTM へ移行)  │               │
│  │                                          │               │
│  │  出力:                                  │               │
│  │   ├── ペースカテゴリ (HIGH/MIDDLE/SLOW) │               │
│  │   └── 1F毎ラップ予測値 (furlong別)     │               │
│  └────────────────────────────────────────┘               │
└───────────────────────────────────────────────────────────┘
```

### 3-2. アルゴリズム選定

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率（T-1） | LightGBM softmax | 表形式データに最強、欠損耐性が高い |
| 連対率・複勝率（T-2/3） | LightGBM binary | 同上 |
| ポジション予測（T-8） | LambdaMART（LightGBM ranker） | 相対順位を直接最適化できる |
| オッズ予測（T-4/5） | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ（T-10） | LightGBM multiclass | 3クラス・解釈性を重視 |
| 1F毎ラップ予測（T-11） | LightGBM per-furlong（初期）→ LSTM（拡張） | 解釈しやすい単独回帰から開始し、系列依存が大きければ LSTM へ移行 |

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

> **必須制約**: Layer 3 集計値は必ず `as_of_race_id = 予測対象レース ID` のスナップショットを参照すること。そのレース以後の情報を含めることを禁止する。

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

---

## 5. 学習パイプライン

### 5-1. データ分割ルール

- 時系列順に train / validation / test を分割する。**ランダムシャッフルは禁止**。
- 常に過去レースで学習し、未来レースで評価する。
- 推論時も `as_of_race_id = 対象レース ID` のスナップショットのみ使用する。

### 5-2. Stage 1 学習手順

1. 特徴量エンジニアリングパイプライン実行（脚質スコア・クロス特徴量・相対特徴量の自動生成）
2. 時系列分割により train/validation/test セット構築
3. LightGBM binary で連対率・複勝率モデルを学習
4. LightGBM softmax で勝率モデルを学習
5. LambdaMART でポジション予測モデルを学習
6. LightGBM regression でオッズ予測モデルを学習
7. 各モデルの評価指標を計算（後述）
8. モデルを ModelRegistry へ登録・バージョニング

### 5-3. Stage 2 学習手順

1. Stage 1 のポジション予測値を入力特徴量として受け渡し
2. Layer 4（ラップ・ペース・コーナー通過）の特徴量を結合
3. LightGBM multiclass でペースカテゴリモデルを学習
4. LightGBM per-furlong で 1F 毎ラップ予測モデルを学習（初期実装）
5. MAE ≤ 0.3 秒 の閾値を下回れない場合は LSTM への移行を検討（Phase 4 以降）
6. 評価指標を計算・記録
7. モデルを ModelRegistry へ登録・バージョニング

### 5-4. バッチ推論スケジュール

- 推論バッチは**発走3時間前までに完了**させる（N-9）。
- 推論時に使用するオッズ特徴量は「発走 N 分前の最終スナップショット」を固定使用する（直前確定値）。
- 推論結果は `prediction_results` および `prediction_lap_times` テーブルに保存する。

### 5-5. 回収率ポスト計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float,        # モデル予測勝率    (T-1)
    win_odds: float,        # 予測単勝オッズ    (T-4)
    show_prob: float,       # モデル予測複勝率  (T-3)
    place_odds_mid: float,  # 予測複勝オッズ中値 (T-5)
) -> dict:
    """
    単回収率 = 勝率 × 単勝オッズ × 100
    複回収率 = 複勝率 × 複勝オッズ中値 × 100
    100 超 = バリューベット候補
    """
    win_roi  = win_prob  * win_odds       * 100
    show_roi = show_prob * place_odds_mid * 100
    return {
        "win_roi":  round(win_roi,  2),   # T-6
        "show_roi": round(show_roi, 2),   # T-7
    }
```

---

## 6. SHAP による解釈性

- 全 LightGBM モデルに対して **TreeSHAP**（`shap.TreeExplainer`）を適用し、特徴量重要度を算出・記録する。
- 運用フェーズにおいて特徴量重要度の定期モニタリングを実施し、データドリフトの早期検知に活用する（Phase 4 以降の継続タスク）。
- SHAP 値は ModelRegistry のアーティファクトとしてモデルバージョンに紐付けて保存する。

---

## 7. ModelRegistry・バージョニング

### 7-1. 基本方針

- モデル管理ツールとして **MLflow**（または同等のツール）を使用する（F-16）。
- モデルアーティファクトは GCS `models/` プレフィックス配下に保存し、バージョン管理する（`gcs_paths.py` 経由）。
- `prediction_results` テーブルの `model_version` カラムによりどのモデルによる推論かを追跡可能とする。

### 7-2. ModelRegistry 実装（DEC-009 確定）

バッチワーカープロセス専用の `ModelRegistry` クラス（RWLock パターン）を実装する。Web プロセス（Flask API）からのモデルロードは **禁止**。

```python
# inference/model_registry.py
import os, threading
import lightgbm as lgb
import numpy as np
from pathlib import Path
from typing import Optional

VCPU_COUNT   = os.cpu_count() or 2
WORKER_COUNT = int(os.environ.get("GUNICORN_WORKERS", 2))
# 2vCPU / 2workers → LGBM_THREADS = 1（CPU 使用率 50% 上限を維持）
LGBM_THREADS = max(1, VCPU_COUNT // WORKER_COUNT)

class ModelRegistry:
    """バッチワーカープロセス専用。Web プロセス（Flask/Gunicorn）からの使用禁止。"""
    def __init__(self, model_path: str):
        self._lock    = threading.RLock()
        self._path    = Path(model_path)
        self._booster: Optional[lgb.Booster] = None
        self._version = 0
        self._load()

    def _load(self) -> None:
        b = lgb.Booster(model_file=str(self._path))
        b.params["num_threads"] = LGBM_THREADS  # CPU スパイク抑制
        with self._lock:
            self._booster = b
            self._version += 1

    def reload(self, new_path: Optional[str] = None) -> int:
        if new_path:
            self._path = Path(new_path)
        self._load()
        return self._version

    def predict(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._booster is None:
                raise RuntimeError("モデル未初期化")
            return self._booster.predict(features)
```

### 7-3. ETL 完了フラグガード（DEC-009 F-4）

推論バッチは ETL 完了フラグを確認してから起動する。フラグが存在しない場合は 10 分待機後に停止。

```python
# etl/pipeline.py（完了フック）
def mark_etl_complete(race_date: str) -> None:
    redis_client.set(f"etl:complete:{race_date}", "1", ex=3600)

# inference/batch.py（起動ガード）
def wait_for_etl(race_date: str, timeout_sec: int = 600) -> None:
    import time
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if redis_client.get(f"etl:complete:{race_date}"):
            return
        time.sleep(10)
    raise TimeoutError(f"ETL 未完了: {race_date}")  # Slack アラート発報
```

---

## 8. Pandera スキーマバリデーション（DEC-009 F-6）

推論前に特徴量の型・値域を検証し、異常値で即停止する。

```python
# inference/validation.py
from pandera import DataFrameSchema, Column, Check

FEATURE_SCHEMA = DataFrameSchema({
    "distance":     Column(int,   Check.in_range(800, 3600)),
    "field_size":   Column(int,   Check.in_range(2, 18)),
    "horse_weight": Column(float, Check.in_range(380, 620)),
})

def validate_and_predict(registry: ModelRegistry, raw: pd.DataFrame) -> np.ndarray:
    FEATURE_SCHEMA.validate(raw)   # 型・値域エラー時は例外で停止
    return registry.predict(raw.values)
```

---

## 9. モデルサイズ・メモリ制約（DEC-009 確定）

| 制約 | 値 | 適用場所 |
|---|---|---|
| モデルファイルサイズ | < 50 MB | CI ゲート（PR ブロック） |
| LightGBM Booster RSS 増加量 | < 250 MB（超過時は `num_leaves` 63→31 で再学習） | CI メモリ計測テスト |
| `num_threads` | `max(1, VCPU_COUNT // WORKER_COUNT)` = 1 | 推論バッチプロセス起動時 |
| バッチプロセスの nice 値 | `nice -n 10`（優先度を下げて Web API レイテンシ劣化を防止） | cron スケジュール定義 |

CI ゲート実装例:

```python
# tests/test_model_memory.py
from memory_profiler import memory_usage
import lightgbm as lgb, os

def test_model_size_within_budget():
    size_mb = os.path.getsize("models/current.lgb") / (1024 ** 2)
    assert size_mb < 50, f"モデルサイズ {size_mb:.1f}MB が 50MB 予算超過"

def test_booster_rss_within_budget():
    baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
    lgb.Booster(model_file="models/current.lgb")
    peak = max(memory_usage(-1, interval=0.1, timeout=2))
    delta = peak - baseline
    assert delta < 250, f"LightGBM RSS delta {delta:.1f}MB が 250MB 予算超過"
```