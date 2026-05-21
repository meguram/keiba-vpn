# MLflow モデリング・推論プラットフォーム

競馬予測の学習・Registry・Model Serving・API キャッシュを **モデルキー単位** で統一する基盤。

## アーキテクチャ

```
学習 (pipeline/models/*.py)
    → MLflow Tracking + Model Registry  (catalog.ModelSpec)
    → 任意: mlflow models serve (Docker :5001–:5010)

推論 (API / バッチ)
    → Model Serving (/invocations)  … あれば優先
    → ローカル Booster (Registry / mlflow/runs)
    → ヒューリスティック / ルール

API 高速化
    → HybridStorage キャッシュ (InferenceCacheMixin)
    → 事前計算 CLI (scripts/maintenance/precompute_*.py)
```

## モデルカタログ

正の定義: `src/pipeline/mlflow/catalog.py` の `MODEL_CATALOG`

| model_key | 用途 | Registry 名 | Serving ポート | lifecycle |
|-----------|------|-------------|----------------|-----------|
| `keiba_lgbm` | 既存着順 LGBM | `keiba-lgbm` | 5010 | active |
| `tracking_difficulty` | 追走難度・位置取り | `tracking-difficulty-lgbm` | 5001 | active |
| `finish_order` | 次期着順予測 | `keiba-finish-order-lgbm` | 5002 | planned |
| `final_odds` | 最終オッズ予測 | `keiba-final-odds-lgbm` | 5003 | planned |
| `pace_predictor` | 1F/3F ペース | `pace-predictor-lgbm` | 5004 | active |

設定の上書き: `config/settings.yaml` → `mlflow.models.<key>`

環境変数: `KEIBA_MLFLOW_SERVE_<KEY>_URI`（例: `KEIBA_MLFLOW_SERVE_FINISH_ORDER_URI`）

## 新しい予測タスクの追加手順

1. **catalog** … `ModelSpec` を `MODEL_CATALOG` に追加
2. **settings** … `mlflow.models.<key>.serve_uri` / `serve_port`
3. **storage** … `CATEGORY_MAP` に `cache_category` を追加（`race` 等）
4. **学習** … `train_utils.log_lightgbm_and_register("your_key", model, ...)`
5. **推論** … `inference/<task>_service.py`（`InferenceCacheMixin` + `get_or_compute`）
6. **API** … 薄い GET + 任意 POST precompute
7. **Docker** … `mlflow/server/docker-compose.yml` に serve サービス（profile 可）

## コード例

### 学習

```python
from src.pipeline.mlflow.train_utils import start_training_run, log_lightgbm_and_register

with start_training_run("finish_order", params={"n_estimators": 500}):
    log_lightgbm_and_register(
        "finish_order",
        booster,
        metrics={"auc": 0.72},
        feature_names=cols,
    )
```

### 推論（LGBM バッチ）

```python
from src.pipeline.mlflow.runtime import get_serve_client, load_lightgbm_booster, booster_feature_names

cols = booster_feature_names("finish_order") or FEATURE_COLUMNS
client = get_serve_client("finish_order")
if client and client.is_available():
    scores = client.predict_dataframe(cols, matrix)
else:
    model = load_lightgbm_booster("finish_order")
```

### キャッシュ

```python
from src.pipeline.mlflow.inference_cache import InferenceCacheMixin

class FinishOrderCache(InferenceCacheMixin):
    model_key = "finish_order"
    cache_version = 1
```

## 運用

- ヘルス: `GET /api/inference/health`（全モデル一覧）
- CLI: `python -m src.scripts.maintenance.mlflow_platform_health`
- Tracking: `docker compose -f mlflow/server/docker-compose.yml up -d mlflow`
- Serving（追走難度のみ常時）: `... up -d mlflow-serve-tracking`
- 着順・オッズ（Registry 登録後）: `... --profile finish-order up -d`

ローカル開発で Registry artifact が `mlruns/` 参照のときは、リポジトリルートに `mlruns → mlflow/runs` リンク（`ensure_mlruns_symlink()` が自動作成）。

## 関連ファイル

- `src/pipeline/mlflow/` … catalog, runtime, inference_cache, train_utils
- `src/pipeline/inference/*_service.py` … タスク別推論
- `src/utils/mlflow_client.py` … 既存着順学習用（徐々に train_utils へ統合可）
