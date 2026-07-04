"""レース着順予測の storage ベース推論（スクレイプなし）。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.mlflow.inference_cache import InferenceCacheMixin

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_KEIBA_PKL = _PROJECT_ROOT / "models" / "keiba_model.pkl"
_ENCODER_PATH = _PROJECT_ROOT / "models" / "encoder.json"


class RacePredictionCache(InferenceCacheMixin):
    model_key = "keiba_lgbm"


def load_cached(storage, race_id: str) -> dict | None:
    return RacePredictionCache.load_cached(storage, race_id)


def save_cached(storage, race_id: str, payload: dict, *, source: str = "batch") -> None:
    RacePredictionCache.save_cached(storage, race_id, payload, source=source)


def build_race_data_from_storage(race_id: str, storage) -> dict[str, Any]:
    """GCS/ローカルキャッシュから scrape_race_all 相当の dict を組み立てる。"""
    race_data: dict[str, Any] = {"race_id": race_id}

    shutuba = storage.load("race_shutuba", race_id)
    if shutuba:
        race_data["race_card"] = shutuba

    for key, cat in (
        ("speed_index", "race_index"),
        ("shutuba_past", "race_shutuba_past"),
        ("paddock", "race_paddock"),
        ("barometer", "race_barometer"),
        ("oikiri", "race_oikiri"),
    ):
        blob = storage.load(cat, race_id)
        if blob:
            race_data[key] = blob

    card = race_data.get("race_card") or {}
    horse_ids = [
        e.get("horse_id", "")
        for e in (card.get("entries") or [])
        if e.get("horse_id")
    ]
    if horse_ids:
        horses: dict[str, Any] = {}
        workers = min(len(horse_ids), 8)

        def _load_hid(hid: str):
            return hid, storage.load("horse_result", hid)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for hid, hr in pool.map(_load_hid, horse_ids):
                if hr:
                    horses[hid] = hr
        race_data["horses"] = horses

    return race_data


def _load_sklearn_model():
    if not _KEIBA_PKL.is_file():
        return None
    try:
        import joblib

        return joblib.load(_KEIBA_PKL)
    except Exception as e:
        logger.warning("keiba_model.pkl ロード失敗: %s", e)
        return None


def _encode_features(features_df: pd.DataFrame) -> pd.DataFrame:
    from src.pipeline.features.feature_builder import PUBLIC_INDICATOR_SET
    from src.pipeline.models.encoder import FeatureEncoder

    enc = None
    if _ENCODER_PATH.is_file():
        try:
            enc = FeatureEncoder()
            enc.load(str(_ENCODER_PATH))
        except Exception as e:
            logger.warning("encoder ロード失敗: %s", e)

    if enc is not None:
        encoded = enc.transform(features_df)
    else:
        drop_cols = {"race_id", "horse_name", "horse_id", "race_date"}
        encoded = features_df.drop(
            columns=[c for c in drop_cols if c in features_df.columns],
            errors="ignore",
        )
        str_cols = {
            "surface", "direction", "weather", "track_condition", "sex",
            "venue", "sire", "dam_sire", "jockey_name", "trainer_name",
        }
        for i in range(1, 6):
            str_cols.add(f"prev{i}_surface")
            str_cols.add(f"prev{i}_track_cond")
        for col in str_cols:
            if col in encoded.columns:
                encoded[col] = encoded[col].astype("category").cat.codes

    if "finish_position" in encoded.columns:
        encoded = encoded.drop(columns=["finish_position"])

    numeric_cols = [
        c
        for c in encoded.select_dtypes(include=[np.number]).columns
        if c not in PUBLIC_INDICATOR_SET and c not in {"horse_number", "race_round"}
    ]
    return encoded[numeric_cols].fillna(0)


def build_race_prediction_response(
    race_id: str,
    storage,
    *,
    allow_scrape: bool = False,
) -> dict[str, Any]:
    """1レース分の race_predictions JSON を組み立てる。"""
    from src.pipeline.features.feature_builder import build_race_features
    from src.pipeline.inference.race_day import RaceDayPipeline

    t0 = time.perf_counter()
    race_data = build_race_data_from_storage(race_id, storage)

    if not (race_data.get("race_card") or {}).get("entries"):
        if allow_scrape:
            from src.scraper.run import ScraperRunner

            runner = ScraperRunner(interval=1.0, cache=True, auto_login=True)
            race_data = runner.scrape_race_all(race_id, smart_skip=True)
        else:
            return {
                "race_id": race_id,
                "status": "error",
                "error": "出馬表データがありません",
                "predictions": [],
            }

    card = race_data.get("race_card") or {}
    race_info = {
        "race_id": race_id,
        "race_name": card.get("race_name", ""),
        "venue": card.get("venue", ""),
        "round": card.get("round", 0),
        "surface": card.get("surface", ""),
        "distance": card.get("distance", 0),
        "track_condition": card.get("track_condition", ""),
    }

    features_df = build_race_features(race_data)
    if features_df.empty:
        return {
            **race_info,
            "status": "error",
            "error": "特徴量テーブルが空",
            "predictions": [],
        }

    model_type = "fallback_heuristic"
    pipe = RaceDayPipeline()
    scores = pipe._fallback_score(features_df)

    model = _load_sklearn_model()
    if model is not None:
        try:
            X = _encode_features(features_df)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
            else:
                scores = model.predict(X)
            model_type = "keiba_model_pkl"
        except Exception as e:
            logger.warning("keiba_model.pkl 予測失敗 %s: %s", race_id, e)

    meta_cols = ["race_id", "horse_number", "horse_name", "horse_id"]
    meta = features_df[[c for c in meta_cols if c in features_df.columns]].copy()
    meta["pred_score"] = scores
    meta = meta.sort_values("pred_score", ascending=False).reset_index(drop=True)
    meta["pred_rank"] = range(1, len(meta) + 1)

    raw = meta["pred_score"].values.astype(float)
    centered = raw - raw.mean()
    exp_scores = np.exp(centered / max(centered.std(), 1e-6))
    meta["softmax_prob"] = exp_scores / exp_scores.sum()

    predictions = []
    for _, row in meta.iterrows():
        predictions.append({
            "pred_rank": int(row["pred_rank"]),
            "horse_number": int(row.get("horse_number", 0)),
            "horse_name": row.get("horse_name", ""),
            "horse_id": row.get("horse_id", ""),
            "pred_score": round(float(row["pred_score"]), 4),
            "normalized_score": round(float(row["softmax_prob"]), 4),
        })

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        **race_info,
        "status": "success",
        "model_type": model_type,
        "total_horses": len(predictions),
        "elapsed_sec": elapsed,
        "predictions": predictions,
        "_compute_meta": {"source": "batch_inference", "elapsed_sec": elapsed},
    }
