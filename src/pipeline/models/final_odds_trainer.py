"""最終オッズ予測モデル（LightGBM × MLflow）。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.mlflow.catalog import get_model_spec
from src.pipeline.mlflow.train_utils import log_lightgbm_and_register, start_training_run
from src.pipeline.models.final_odds_dataset import (
    EVAL_YEARS_DEFAULT,
    TRAIN_YEARS_DEFAULT,
    build_final_odds_dataset,
)
from src.pipeline.models.final_odds_features import select_model_feature_columns
from src.pipeline.models.final_odds_progress import FinalOddsProgress
from src.utils.logger import get_logger

logger = get_logger("FinalOddsTrainer")

MLFLOW_MODEL_KEY = "final_odds"
LOCAL_BUNDLE_PATH = Path("models/final_odds_bundle.json")

TARGETS = (
    ("win", "target_log_win_odds", "target_win_odds"),
    ("place_min", "target_log_place_min", "target_place_min"),
    ("place_max", "target_log_place_max", "target_place_max"),
)


@dataclass
class _TrainedHead:
    name: str
    model: Any
    feature_cols: list[str]
    metrics: dict[str, float]


class FinalOddsTrainer:
    """単勝・複勝下限・複勝上限の3ヘッドを学習し MLflow に登録。"""

    def __init__(self, mlflow_tracking_uri: str | None = None):
        spec = get_model_spec(MLFLOW_MODEL_KEY)
        self.spec = spec
        self.mlflow_uri = mlflow_tracking_uri or "http://localhost:5000"

    def train(
        self,
        storage=None,
        *,
        train_years: tuple[str, ...] = TRAIN_YEARS_DEFAULT,
        eval_years: tuple[str, ...] = EVAL_YEARS_DEFAULT,
        max_races: int | None = None,
        progress: FinalOddsProgress | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        if storage is None:
            from src.scraper.storage import HybridStorage

            storage = HybridStorage()

        prog = progress or FinalOddsProgress()
        prog.start(train_years=list(train_years), eval_years=list(eval_years))

        df = build_final_odds_dataset(
            storage,
            train_years=train_years,
            eval_years=eval_years,
            max_races=max_races,
            progress=prog,
        )
        if df.empty:
            prog.finish(error="学習データが空です")
            return {"error": "学習データが空です"}

        feature_cols = select_model_feature_columns(df)
        if len(feature_cols) < 10:
            prog.finish(error=f"特徴量不足: {len(feature_cols)}列")
            return {"error": f"特徴量不足: {len(feature_cols)}列"}

        train_df = df[df["split"] == "train"].copy()
        eval_df = df[df["split"] == "eval"].copy()
        if train_df.empty:
            prog.finish(error="train split が空です")
            return {"error": "train split が空です"}

        prog.set_phase(
            "prepare",
            total=len(TARGETS),
            message=f"train={len(train_df):,} eval={len(eval_df):,} feat={len(feature_cols)}",
        )
        prog.update(meta={"n_features": len(feature_cols)})

        try:
            import lightgbm as lgb
        except ImportError:
            prog.finish(error="lightgbm が未インストール")
            return {"error": "lightgbm が未インストール"}

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 30,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "lambda_l1": 0.05,
            "lambda_l2": 0.2,
            "verbose": -1,
        }

        heads: list[_TrainedHead] = []
        all_metrics: dict[str, Any] = {
            "n_features": len(feature_cols),
            "n_train_rows": len(train_df),
            "n_eval_rows": len(eval_df),
            "train_years": list(train_years),
            "eval_years": list(eval_years),
            "feature_names": feature_cols,
        }

        for idx, (head_name, log_col, raw_col) in enumerate(TARGETS, start=1):
            prog.set_phase(
                f"train_{head_name}",
                total=1,
                message=f"LightGBM [{head_name}] ({idx}/{len(TARGETS)})",
            )
            head = self._train_head(
                lgb,
                lgb_params,
                train_df,
                eval_df,
                feature_cols,
                head_name,
                log_col,
                raw_col,
            )
            heads.append(head)
            prog.update(
                current=1,
                message=f"MAPE={head.metrics.get(f'{head_name}_mape_pct', '—')}%",
            )

        prog.set_phase("save", message="ローカル bundle 保存")
        bundle = self._save_local_bundle(heads, feature_cols, all_metrics)
        prog.set_phase("mlflow", message="MLflow 登録")
        run_id = self._log_mlflow(heads, lgb_params, all_metrics, bundle)

        all_metrics["training_time_sec"] = round(time.time() - t0, 1)
        all_metrics["mlflow_run_id"] = run_id
        all_metrics["local_bundle"] = str(LOCAL_BUNDLE_PATH)
        prog.finish(result={
            "n_train_rows": all_metrics["n_train_rows"],
            "n_eval_rows": all_metrics["n_eval_rows"],
            "training_time_sec": all_metrics["training_time_sec"],
            "win_mape_pct": heads[0].metrics.get("win_mape_pct") if heads else None,
        })
        return all_metrics

    def _train_head(
        self,
        lgb: Any,
        params: dict,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        feature_cols: list[str],
        head_name: str,
        log_col: str,
        raw_col: str,
    ) -> _TrainedHead:
        X_train = train_df[feature_cols].fillna(0).astype(float)
        y_train = train_df[log_col].astype(float).values

        if not eval_df.empty:
            X_eval = eval_df[feature_cols].fillna(0).astype(float)
            y_eval = eval_df[log_col].astype(float).values
            valid = lgb.Dataset(X_eval, label=y_eval, reference=lgb.Dataset(X_train, label=y_train))
            valid_sets = [valid]
        else:
            n_hold = max(int(len(X_train) * 0.15), 100)
            X_eval = X_train.iloc[-n_hold:]
            y_eval = y_train[-n_hold:]
            X_tr = X_train.iloc[:-n_hold]
            y_tr = y_train[:-n_hold]
            valid = lgb.Dataset(X_eval, label=y_eval, reference=lgb.Dataset(X_tr, label=y_tr))
            valid_sets = [valid]
            X_train, y_train = X_tr, y_tr

        train_data = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1200,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

        preds_log = model.predict(X_eval)
        if not eval_df.empty:
            raw_actual = eval_df[raw_col].astype(float).values
        else:
            raw_actual = np.exp(y_eval)

        pred_raw = np.exp(preds_log)
        mape = float(np.mean(np.abs(pred_raw - raw_actual) / np.maximum(raw_actual, 1.0))) * 100
        rmse_log = float(np.sqrt(np.mean((preds_log - y_eval) ** 2)))

        importance = dict(
            zip(
                feature_cols,
                model.feature_importance(importance_type="gain").tolist(),
            )
        )
        top_imp = dict(sorted(importance.items(), key=lambda x: -x[1])[:15])

        metrics = {
            f"{head_name}_rmse_log": round(rmse_log, 4),
            f"{head_name}_mape_pct": round(mape, 2),
            f"{head_name}_best_iter": int(model.best_iteration or 0),
        }
        logger.info(
            "final_odds head [%s]: MAPE=%.1f%% RMSE_log=%.4f",
            head_name,
            mape,
            rmse_log,
        )
        return _TrainedHead(head_name, model, feature_cols, {**metrics, "top_features": top_imp})

    def _save_local_bundle(
        self,
        heads: list[_TrainedHead],
        feature_cols: list[str],
        metrics: dict,
    ) -> dict:
        bundle: dict[str, Any] = {
            "feature_cols": feature_cols,
            "heads": {},
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
        }
        for h in heads:
            bundle["heads"][h.name] = {
                "model_txt": h.model.model_to_string(),
                "metrics": h.metrics,
            }
        LOCAL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_BUNDLE_PATH, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        return bundle

    def _log_mlflow(
        self,
        heads: list[_TrainedHead],
        lgb_params: dict,
        metrics: dict,
        bundle: dict,
    ) -> str:
        flat_metrics: dict[str, float] = {}
        for h in heads:
            for k, v in h.metrics.items():
                if k == "top_features":
                    continue
                if isinstance(v, (int, float)):
                    flat_metrics[f"{h.name}_{k}" if not k.startswith(h.name) else k] = float(v)

        run_id = ""
        try:
            import mlflow
            import mlflow.lightgbm

            with start_training_run(
                self.spec,
                params={
                    **{f"lgb_{k}": v for k, v in lgb_params.items()},
                    "train_years": ",".join(metrics.get("train_years", [])),
                    "eval_years": ",".join(metrics.get("eval_years", [])),
                },
                tags={
                    "model_type": "final_odds",
                    "algorithm": "lightgbm_regression",
                    "targets": "win,place_min,place_max",
                },
                tracking_uri=self.mlflow_uri,
            ):
                mlflow.log_metrics(
                    {
                        "n_train_rows": float(metrics["n_train_rows"]),
                        "n_eval_rows": float(metrics["n_eval_rows"]),
                        "n_features": float(metrics["n_features"]),
                        **flat_metrics,
                    }
                )
                for h in heads:
                    mlflow.lightgbm.log_model(
                        h.model,
                        name=f"model_{h.name}",
                        registered_model_name=None,
                    )
                # 代表モデル（win）を Registry 名で登録
                win_head = next(x for x in heads if x.name == "win")
                run_id = log_lightgbm_and_register(
                    self.spec,
                    win_head.model,
                    metrics={k: v for k, v in flat_metrics.items() if isinstance(v, float)},
                    params={"n_heads": len(heads)},
                    feature_names=win_head.feature_cols,
                    feature_importance=win_head.metrics.get("top_features", {}),
                    artifact_name="model",
                )
                import tempfile
                import os
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                json.dump(bundle, tmp, ensure_ascii=False, indent=2)
                tmp.close()
                mlflow.log_artifact(tmp.name, "final_odds_bundle")
                os.unlink(tmp.name)
        except Exception as exc:
            logger.warning("MLflow 登録失敗（ローカル bundle は保存済み）: %s", exc)
        return run_id
