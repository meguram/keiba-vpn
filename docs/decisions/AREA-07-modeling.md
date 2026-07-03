# AREA-07: モデリング設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. モデルカタログ

`src/pipeline/mlflow/catalog.py` が単一の真実の源泉（SSoT）。

| キー | 実験名 | 登録モデル名 | 種別 | ステータス | サーブポート |
|------|--------|------------|------|-----------|-------------|
| `keiba_lgbm` | `keiba-prediction` | `keiba-lgbm` | LightGBM LambdaRank | ACTIVE | 5010 |
| `tracking_difficulty` | `tracking-difficulty` | `tracking-difficulty-lgbm` | LightGBM | ACTIVE | 5001 |
| `final_odds` | `final-odds` | `keiba-final-odds-lgbm` | LightGBM 3ヘッド | ACTIVE | 5003 |
| `pace_predictor` | `pace-prediction` | `pace-predictor-lgbm` | LightGBM | ACTIVE | 5004 |
| `finish_order` | `finish-order` | `keiba-finish-order-lgbm` | LightGBM | PLANNED | 5002 |

---

## 2. メインモデル: `keiba_lgbm`

### アーキテクチャ

```python
# src/pipeline/models/trainer.py: ModelTrainer
objective = "lambdarank"
metric = "ndcg"
ndcg_eval_at = [3, 5]
group = race_id  # グループ化（レース内ランキング）
target = "finish_position_top3"  # 複勝圏バイナリ (1 <= pos <= 3)
EXPERIMENT_NAME = "keiba-no-public-indicators"
MODEL_NAME = "keiba-lgbm-nopi"
```

### デフォルトハイパーパラメータ

```python
{
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
}
```

### 大衆指標排除ポリシー（P-05）

```python
# src/pipeline/features/feature_builder.py
PUBLIC_INDICATOR_SET = {
    "win_odds", "place_odds", "popularity_rank",  # 当日オッズ・人気
    "predicted_win_odds",                          # 予測オッズ
    # ...
}
```

- `config/settings.yaml` の `feature_policy.excluded_features` にリスト定義
- SmartRC 独自指標（`cr_value`, `smartrc_ten1f` 等）はホワイトリストで許可

---

## 3. 特化モデル

### `tracking_difficulty`（追走難度）

```python
# src/pipeline/models/tracking_difficulty.py
目的: 追走難度・脚温存度スコアを推定
特徴量例: 1F ペース差・前走比較・コース形状等
出力: stamina_index, tracking_difficulty_score
```

### `final_odds`（オッズ予測）

```python
# src/pipeline/models/final_odds_trainer.py
目的: 単勝・複勝（min/max）オッズを予測
ヘッド数: 3（win / place_min / place_max）
```

### `pace_predictor`（ペース予測）

```python
# src/pipeline/models/pace_predictor.py
目的: 1F/3F ラップタイムを予測
特徴量: 距離・脚質分布・コース・馬場状態等 11 特徴
```

### アンサンブル: `EnsembleTrainer`

```python
# src/pipeline/models/ensemble_trainer.py
Layer 1 (GroupKFold OOF):
  - LightGBM LambdaRank
  - XGBoost (binary:logistic)
  - CatBoost
  - 2-layer MLP
Layer 2:
  - Logistic Regression（スタッキング）
```

---

## 4. 特徴量エンコーダ

### `FeatureEncoder`（`src/pipeline/models/encoder.py`）

```python
# 固定マッピング
SURFACE_MAP = {"芝": 0, "ダ": 1, "障": 2}
DIRECTION_MAP = {"右": 0, "左": 1, "直": 2}
WEATHER_MAP = {"晴": 0, "曇": 1, "雨": 2, "小雨": 3, "雪": 4, "小雪": 5}
TRACK_COND_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
SEX_MAP = {"牡": 0, "牝": 1, "騸": 2}
VENUE_MAP = {各競馬場コード}

# Target Encoding（学習データから推定）
target_encode_cols = ["sire", "dam_sire", "jockey_name", "trainer_name"]
# 保存先: models/encoder.json
```

---

## 5. 学習データセット分割

```python
# src/pipeline/models/dataset_split.py
# data/local/modeling/dataset_split_manifest.json
splits = {
    "model_selection_primary": {
        "train": [2020, 2021, 2022, 2023],
        "valid": [2024],
        "test":  [2025],
    }
}
```

---

## 6. MLflow 統合

### 設定（`config/settings.yaml`）

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "keiba-prediction"
  registered_model_name: "keiba-lgbm"
  artifact_location: "mlruns"
```

### ランタイム（`src/pipeline/mlflow/runtime.py`）

```python
class MlflowServeClient:
    """MLflow serve エンドポイントへ POST /invocations"""
    def predict(self, df: pd.DataFrame) -> np.ndarray

def load_lightgbm_booster(key: str) -> lgb.Booster:
    """ローカル models/*.lgb → MLflow Registry の順でロード"""
```

### 学習フロー

```bash
# 1. 特徴量ビルド
python -m src.pipeline.register_raw_table_features
python -m src.pipeline.build_jockey_trainer_stats
python -m src.pipeline.build_rank_target
python -m src.pipeline.build_layer_a_dataset

# 2. 学習
python -m src.pipeline.run_baseline_train \
    --features-dir data/local/features \
    --manifest data/local/modeling/dataset_split_manifest.json \
    --protocol model_selection_primary
```

---

## 7. Inference 層

### `RaceDayPipeline`（`src/pipeline/inference/race_day.py`）

```python
class RaceDayPipeline:
    def run(self, date: str) -> list[RaceResult]:
        """
        1. レースID一覧取得
        2. 各レースのスクレイプ（ScraperRunner）
        3. 特徴量構築（build_race_features）
        4. FeatureEncoder でエンコード
        5. MLflow serve に POST
        6. RaceResult として返却
        7. data/predictions/ に保存
        """
class PipelineConfig:
    scrape_interval: float
    model_name: str = "keiba-lgbm-nopi"
    mlflow_tracking_uri: str
```

### `BettingOptimizer`（`src/pipeline/inference/betting.py`）

```python
class BettingConfig:
    min_ev: float = 1.05          # 期待値閾値
    kelly_fraction: float = 0.25  # Kelly 係数
    max_bet_ratio: float = 0.10   # 最大賭け金比率

bet_types = ["tansho", "fukusho", "umaren", "wide", "umatan"]
# pred_score → softmax 確率 → EV 計算 → Kelly で賭け金決定
```

### `CompositeOptimizer`（`src/pipeline/inference/composite_optimizer.py`）

```python
# グリッドサーチ最適化
optimize_params = {
    "prob_weight": [0.20, 0.40, 0.60, 0.80],
    "min_prob_honmei_ratio": [...],
    "top_n_bet": [3, 4, 5],
}
# 評価: ROI×0.40 + hit_rate×0.25 + top3_capture×0.20 + sharpe×0.15
# 出力: models/composite_params.json
```

---

## 8. 評価指標（CI ゲート）

| 指標 | 閾値 | テストファイル |
|------|------|--------------|
| Log Loss | ベースライン比 −5% 以上 | `tests/pipeline/` |
| Spearman ρ | ≥ 0.55 | `tests/pipeline/` |
| ラップ MAE | ≤ 0.3 秒 | `tests/pipeline/` |
| テンポラルリーク | 0件（`as_of_race_id` フィルタ） | `tests/ml/test_temporal_leak.py` |

---

## 9. モデル制約（VPS 2GB メモリ）

| 項目 | 制約値 |
|------|--------|
| モデルファイルサイズ | < 50 MB |
| Booster RSS 増加 | < 250 MB |
| スレッド数 | `LGBM_THREADS = max(1, vcpu // workers)` |
| バックグラウンド実行 | `nice -n 10`（低優先度） |

---

## 10. 推論タイミング

| トリガ | 条件 | 処理 |
|--------|------|------|
| T-15 自動推論 | `KEIBA_PRE_RACE_PREDICT_ENABLED=1` | スクレイプ完了後に `pre_race_predict_trigger.py` |
| 手動 | `POST /api/race/{race_id}/predict` | 即時推論 |
| バッチ | cron（未設定） | 将来対応 |
