"""推論結果を prediction_results / prediction_lap_times に保存（F-9）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.pipeline.recovery import calculate_recovery_rate
from src.db.models import PredictionLapTime, PredictionResult


def save_prediction_batch(
    session: Session,
    race_id: str,
    model_version: str,
    horses: list[dict[str, Any]],
    *,
    pace_prediction: dict[str, Any] | None = None,
    predicted_at: datetime | None = None,
) -> None:
    ts = predicted_at or datetime.now(timezone.utc)
    for h in horses:
        win_prob = float(h["win_prob"]) if h.get("win_prob") is not None else None
        show_prob = float(h["show_prob"]) if h.get("show_prob") is not None else None
        win_odds = float(h["predicted_win_odds"]) if h.get("predicted_win_odds") is not None else None
        place_odds = float(h["predicted_place_odds"]) if h.get("predicted_place_odds") is not None else None
        roi = {}
        if all(v is not None for v in (win_prob, win_odds, show_prob, place_odds)):
            roi = calculate_recovery_rate(win_prob, win_odds, show_prob, place_odds)
        stmt = insert(PredictionResult).values(
            race_id=race_id,
            horse_id=h["horse_id"],
            model_version=model_version,
            predicted_at=ts,
            win_prob=win_prob,
            place_prob=h.get("place_prob"),
            show_prob=show_prob,
            predicted_win_odds=win_odds,
            predicted_place_odds=place_odds,
            expected_win_roi=roi.get("win_roi"),
            expected_show_roi=roi.get("show_roi"),
            predicted_position=h.get("predicted_position"),
            predicted_running_style=h.get("predicted_running_style"),
        ).on_conflict_do_update(
            constraint="uq_prediction_race_horse_model",
            set_={
                "predicted_at": ts,
                "win_prob": win_prob,
                "place_prob": h.get("place_prob"),
                "show_prob": show_prob,
                "predicted_win_odds": win_odds,
                "predicted_place_odds": place_odds,
                "expected_win_roi": roi.get("win_roi"),
                "expected_show_roi": roi.get("show_roi"),
                "predicted_position": h.get("predicted_position"),
                "predicted_running_style": h.get("predicted_running_style"),
            },
        )
        session.execute(stmt)

    if pace_prediction:
        pace_cat = pace_prediction.get("pace_category")
        for lap in pace_prediction.get("lap_times") or []:
            stmt = insert(PredictionLapTime).values(
                race_id=race_id,
                model_version=model_version,
                furlong_index=lap["furlong_index"],
                predicted_lap_sec=lap.get("predicted_lap_sec"),
                predicted_pace_cat=pace_cat,
            ).on_conflict_do_update(
                index_elements=["race_id", "model_version", "furlong_index"],
                set_={
                    "predicted_lap_sec": lap.get("predicted_lap_sec"),
                    "predicted_pace_cat": pace_cat,
                },
            )
            session.execute(stmt)
