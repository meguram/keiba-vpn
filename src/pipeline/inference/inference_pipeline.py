"""Stage 1 → Stage 2 推論オーケストレーション → DB / Redis / GCS 保存。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.api.cache.redis_cache import PredictionCache
from src.api.v1.services import DEFAULT_MODEL_VERSION
from src.db.batch.stats_snapshot import build_snapshots_for_race
from src.db.session import get_session, init_engine
from src.pipeline.features.cross_features import running_style_label
from src.pipeline.models.lap_predictor import predict_pace_and_laps
from src.pipeline.prediction_store import save_prediction_batch
from src.pipeline.recovery import calculate_recovery_rate, is_value_bet
from src.pipeline.inference.race_prediction_service import (
    build_race_prediction_response,
    save_cached,
)
from src.scraper.storage import HybridStorage

logger = logging.getLogger(__name__)


def _softmax_probs(scores: list[float]) -> list[float]:
    if not scores:
        return []
    import numpy as np

    arr = np.array(scores, dtype=float)
    centered = arr - arr.mean()
    exp = np.exp(centered / max(centered.std(), 1e-6))
    return (exp / exp.sum()).tolist()


def _map_stage1_to_spec(
    raw: dict[str, Any],
    *,
    model_version: str,
) -> dict[str, Any]:
    preds = raw.get("predictions") or []
    scores = [p.get("pred_score", p.get("normalized_score", 0)) for p in preds]
    win_probs = _softmax_probs(scores) if scores else [p.get("normalized_score", 0) for p in preds]

    horses = []
    for i, p in enumerate(preds):
        win_prob = float(win_probs[i]) if i < len(win_probs) else 0.0
        place_prob = min(1.0, win_prob * 2.2)
        show_prob = min(1.0, win_prob * 3.0)
        pred_win_odds = round(1.0 / max(win_prob, 0.01), 1)
        pred_place_odds = round(1.0 / max(show_prob / 3, 0.05), 1)
        roi = calculate_recovery_rate(win_prob, pred_win_odds, show_prob, pred_place_odds)
        style_score = p.get("running_style_score")
        horses.append({
            "horse_id": p.get("horse_id"),
            "post_no": p.get("horse_number"),
            "win_prob": round(win_prob, 4),
            "place_prob": round(place_prob, 4),
            "show_prob": round(show_prob, 4),
            "predicted_win_odds": pred_win_odds,
            "predicted_place_odds": pred_place_odds,
            "expected_win_roi": roi["win_roi"],
            "expected_show_roi": roi["show_roi"],
            "predicted_position": p.get("pred_rank"),
            "predicted_running_style": running_style_label(style_score),
            "is_value_bet": is_value_bet(roi["win_roi"], roi["show_roi"]),
        })

    distance = raw.get("distance") or 1600
    field_size = raw.get("total_horses") or len(horses)
    stage2 = predict_pace_and_laps(
        raw.get("race_id", ""),
        distance=int(distance),
        field_size=int(field_size),
    )

    return {
        "race_id": raw.get("race_id"),
        "model_version": model_version,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "pace_prediction": stage2,
        "horses": horses,
    }


def run_inference_for_race(
    race_id: str,
    *,
    model_version: str | None = None,
    allow_scrape: bool = False,
    persist_db: bool = True,
    persist_redis: bool = True,
    persist_gcs: bool = True,
    build_snapshots: bool = True,
) -> dict[str, Any]:
    """T-15 トリガから呼ばれる統合推論。"""
    model_version = model_version or os.environ.get("KEIBA_DEFAULT_MODEL_VERSION", DEFAULT_MODEL_VERSION)
    storage = HybridStorage()

    raw = build_race_prediction_response(race_id, storage, allow_scrape=allow_scrape)
    if raw.get("status") == "error":
        return raw

    payload = _map_stage1_to_spec(raw, model_version=model_version)

    if persist_gcs:
        save_cached(storage, race_id, raw, source="inference_pipeline")

    if persist_db:
        init_engine()
        with get_session() as session:
            if build_snapshots:
                try:
                    build_snapshots_for_race(session, race_id)
                except Exception as exc:
                    logger.warning("snapshot batch skipped: %s", exc)
            save_prediction_batch(
                session,
                race_id,
                model_version,
                payload["horses"],
                pace_prediction=payload["pace_prediction"],
            )

    if persist_redis:
        try:
            cache = PredictionCache()
            cache.set_prediction(race_id, model_version, payload)
            cache.set_lap_prediction(race_id, model_version, payload["pace_prediction"])
        except Exception as exc:
            logger.warning("redis cache skipped: %s", exc)

    return payload
