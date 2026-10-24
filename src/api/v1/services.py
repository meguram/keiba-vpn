"""予測結果の DB / Redis / GCS 統合読み取り。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.cache.redis_cache import PredictionCache
from src.db.models import Entry, PredictionLapTime, PredictionResult, Race
from src.pipeline.recovery import is_value_bet
from src.scraper.storage import HybridStorage

DEFAULT_MODEL_VERSION = os.environ.get("KEIBA_DEFAULT_MODEL_VERSION", "v1.0.0")


def _float(val) -> float | None:
    return float(val) if val is not None else None


def _int(val) -> int | None:
    return int(val) if val is not None else None


def build_predictions_response(
    session: Session,
    race_id: str,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Any] | None:
    race = session.get(Race, race_id)
    if race is None:
        return None
    if race.is_excluded:
        return {
            "race_id": race_id,
            "model_version": model_version,
            "is_excluded": True,
            "horses": [],
        }

    preds = session.scalars(
        select(PredictionResult).where(
            PredictionResult.race_id == race_id,
            PredictionResult.model_version == model_version,
        )
    ).all()
    if not preds:
        gcs_data = _predictions_from_gcs(race_id)
        if not gcs_data:
            return None
        preds = gcs_data

    entries = {
        e.horse_id: e
        for e in session.scalars(
            select(Entry).where(Entry.race_id == race_id)
        ).all()
    }

    predicted_at = None
    horses_out: list[dict[str, Any]] = []

    if preds and isinstance(preds[0], PredictionResult):
        for p in preds:
            if predicted_at is None and p.predicted_at:
                predicted_at = p.predicted_at
            entry = entries.get(p.horse_id)
            win_roi = _float(p.expected_win_roi)
            show_roi = _float(p.expected_show_roi)
            horses_out.append({
                "horse_id": p.horse_id,
                "post_no": entry.post_no if entry else None,
                "win_prob": _float(p.win_prob),
                "win_probability": _float(p.win_prob),          # alias for win_prob
                "place_prob": _float(p.place_prob),
                "show_prob": _float(p.show_prob),
                "place_probability": _float(p.show_prob),       # alias for show_prob (top-3 / 複勝)
                "predicted_win_odds": _float(p.predicted_win_odds),
                "predicted_place_odds": _float(p.predicted_place_odds),
                "expected_win_roi": win_roi,
                "expected_show_roi": show_roi,
                "predicted_position": _int(p.predicted_position),
                "predicted_running_style": p.predicted_running_style,
                "is_value_bet": is_value_bet(win_roi, show_roi),
            })
    elif isinstance(preds, dict):
        predicted_at = preds.get("predicted_at")
        for h in preds.get("horses") or preds.get("entries") or []:
            win_roi = h.get("expected_win_roi") or h.get("win_roi")
            show_roi = h.get("expected_show_roi") or h.get("show_roi")
            horses_out.append({
                "horse_id": h.get("horse_id"),
                "post_no": h.get("post_no") or h.get("horse_number"),
                "win_prob": h.get("win_prob"),
                "win_probability": h.get("win_prob"),           # alias for win_prob
                "place_prob": h.get("place_prob"),
                "show_prob": h.get("show_prob"),
                "place_probability": h.get("show_prob"),        # alias for show_prob (top-3 / 複勝)
                "predicted_win_odds": h.get("predicted_win_odds"),
                "predicted_place_odds": h.get("predicted_place_odds"),
                "expected_win_roi": win_roi,
                "expected_show_roi": show_roi,
                "predicted_position": h.get("predicted_position"),
                "predicted_running_style": h.get("predicted_running_style"),
                "is_value_bet": is_value_bet(win_roi, show_roi),
            })

    lap_rows = session.scalars(
        select(PredictionLapTime)
        .where(
            PredictionLapTime.race_id == race_id,
            PredictionLapTime.model_version == model_version,
        )
        .order_by(PredictionLapTime.furlong_index)
    ).all()

    pace_category = None
    lap_times = []
    for row in lap_rows:
        pace_category = pace_category or row.predicted_pace_cat
        lap_times.append({
            "furlong_index": row.furlong_index,
            "predicted_lap_sec": _float(row.predicted_lap_sec),
        })

    if not lap_times and isinstance(preds, dict):
        pace = preds.get("pace_prediction") or {}
        pace_category = pace.get("pace_category")
        lap_times = pace.get("lap_times") or []

    if predicted_at is None:
        predicted_at = datetime.now(timezone.utc)

    return {
        "race_id": race_id,
        "model_version": model_version,
        "predicted_at": predicted_at.isoformat() if hasattr(predicted_at, "isoformat") else predicted_at,
        "pace_prediction": {
            "pace_category": pace_category,
            "lap_times": lap_times,
        },
        "horses": horses_out,
    }


def build_laps_response(
    session: Session,
    race_id: str,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Any] | None:
    full = build_predictions_response(session, race_id, model_version)
    if full is None:
        return None
    return {
        "race_id": race_id,
        "model_version": model_version,
        "pace_category": full["pace_prediction"]["pace_category"],
        "lap_times": full["pace_prediction"]["lap_times"],
    }


def get_predictions_cached(
    session: Session,
    race_id: str,
    model_version: str = DEFAULT_MODEL_VERSION,
    cache: PredictionCache | None = None,
) -> dict[str, Any] | None:
    cache = cache or PredictionCache()
    try:
        cached = cache.get_prediction(race_id, model_version)
        if cached:
            return cached
    except Exception:
        pass

    payload = build_predictions_response(session, race_id, model_version)
    if payload is None:
        return None

    race = session.get(Race, race_id)
    post_time = None
    if race and race.race_date and race.start_time:
        post_time = datetime.combine(race.race_date, race.start_time, tzinfo=timezone.utc)
    try:
        cache.set_prediction(race_id, model_version, payload, post_time)
    except Exception:
        pass
    return payload


def _predictions_from_gcs(race_id: str) -> dict[str, Any] | None:
    storage = HybridStorage()
    try:
        return storage.load("race_predictions", race_id)
    except Exception:
        return None


def list_races(session: Session, date_str: str | None = None) -> list[dict[str, Any]]:
    q = select(Race).order_by(Race.race_date.desc(), Race.start_time)
    if date_str:
        from datetime import date
        q = q.where(Race.race_date == date.fromisoformat(date_str))
    races = session.scalars(q.limit(200)).all()
    return [
        {
            "race_id": r.race_id,
            "race_name": r.race_name,
            "venue": r.venue,
            "race_date": r.race_date.isoformat() if r.race_date else None,
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "surface": r.surface,
            "distance": r.distance,
            "field_size": r.field_size,
            "is_excluded": r.is_excluded,
        }
        for r in races
    ]


def get_race_detail(session: Session, race_id: str) -> dict[str, Any] | None:
    race = session.get(Race, race_id)
    if race is None:
        return _race_from_gcs(race_id)
    entries = session.scalars(select(Entry).where(Entry.race_id == race_id)).all()
    return {
        "race_id": race.race_id,
        "race_name": race.race_name,
        "venue": race.venue,
        "surface": race.surface,
        "distance": race.distance,
        "direction": race.direction,
        "weather": race.weather,
        "track_condition": race.track_condition,
        "start_time": race.start_time.isoformat() if race.start_time else None,
        "race_date": race.race_date.isoformat() if race.race_date else None,
        "field_size": race.field_size,
        "grade": race.grade,
        "race_class": race.race_class,
        "is_excluded": race.is_excluded,
        "entries": [
            {
                "horse_id": e.horse_id,
                "post_no": e.post_no,
                "bracket_number": e.bracket_number,
                "jockey_id": e.jockey_id,
                "trainer_id": e.trainer_id,
                "weight": e.weight,
                "weight_change": e.weight_change,
            }
            for e in entries
        ],
    }


def _race_from_gcs(race_id: str) -> dict[str, Any] | None:
    storage = HybridStorage()
    for cat in ("race_shutuba", "race_detail"):
        try:
            data = storage.load(cat, race_id)
            if data:
                return data
        except Exception:
            continue
    return None
