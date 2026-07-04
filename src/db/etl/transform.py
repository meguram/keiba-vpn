"""GCS JSON → PostgreSQL Layer 1〜5 変換（AREA-06 §4-4）。"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.config.data_paths import race_path
from src.db.models import (
    Entry,
    Horse,
    Jockey,
    Race,
    RaceCornerPosition,
    RaceLapTime,
    RaceOddsSnapshot,
    RacePaceSummary,
    RaceResult,
    Trainer,
)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def _parse_time(value: str | None) -> time | None:
    if not value:
        return None
    parts = value.strip().split(":")
    if len(parts) >= 2:
        return time(int(parts[0]), int(parts[1]))
    return None


def _sex_age(sex_age: str | None) -> tuple[str | None, int | None]:
    if not sex_age or len(sex_age) < 2:
        return None, None
    return sex_age[0], int(sex_age[1:]) if sex_age[1:].isdigit() else None


def upsert_race_from_shutuba(session: Session, data: dict[str, Any]) -> Race:
    race_id = data["race_id"]
    race = session.get(Race, race_id)
    if race is None:
        race = Race(race_id=race_id)
        session.add(race)
    race.race_name = data.get("race_name")
    race.venue = data.get("venue")
    race.course = data.get("venue")
    race.surface = data.get("surface")
    race.distance = data.get("distance")
    race.direction = data.get("direction")
    race.weather = data.get("weather")
    race.track_condition = data.get("track_condition")
    race.start_time = _parse_time(data.get("start_time"))
    race.race_date = _parse_date(data.get("date"))
    race.field_size = data.get("field_size")
    race.grade = data.get("grade")
    race.race_class = data.get("race_class")
    race.weight_rule = data.get("weight_rule")
    session.flush()
    return race


def upsert_entries_from_shutuba(session: Session, data: dict[str, Any]) -> None:
    race_id = data["race_id"]
    for row in data.get("entries") or []:
        horse_id = row["horse_id"]
        horse = session.get(Horse, horse_id)
        if horse is None:
            sex, age = _sex_age(row.get("sex_age"))
            horse = Horse(
                horse_id=horse_id,
                horse_name=row.get("horse_name", ""),
                sex=sex,
                birth_year=_birth_year_from_age(age, data.get("date")),
                dam_sire=row.get("dam_sire"),
            )
            session.add(horse)
        if row.get("jockey_id"):
            if session.get(Jockey, row["jockey_id"]) is None:
                session.add(Jockey(jockey_id=row["jockey_id"], jockey_name=row.get("jockey_name", "")))
        if row.get("trainer_id"):
            if session.get(Trainer, row["trainer_id"]) is None:
                session.add(Trainer(trainer_id=row["trainer_id"], trainer_name=row.get("trainer_name", "")))
        session.flush()
        stmt = insert(Entry).values(
            race_id=race_id,
            horse_id=horse_id,
            post_no=row.get("horse_number"),
            bracket_number=row.get("bracket_number"),
            jockey_id=row.get("jockey_id"),
            trainer_id=row.get("trainer_id"),
            jockey_weight=row.get("jockey_weight"),
            weight=row.get("weight"),
            weight_change=row.get("weight_change"),
            sex_age=row.get("sex_age"),
        ).on_conflict_do_nothing(constraint="uq_entries_race_horse")
        session.execute(stmt)


def _birth_year_from_age(age: int | None, race_date: str | None) -> int | None:
    if age is None or not race_date:
        return None
    return int(race_date[:4]) - age


def upsert_results(session: Session, data: dict[str, Any]) -> None:
    race_id = data["race_id"]
    for row in data.get("results") or data.get("entries") or []:
        if row.get("finish_pos") is None and row.get("rank") is None:
            continue
        finish = row.get("finish_pos") or row.get("rank")
        stmt = insert(RaceResult).values(
            race_id=race_id,
            horse_id=row["horse_id"],
            finish_pos=finish,
            finish_time_sec=row.get("time_sec") or row.get("finish_time_sec"),
            last_3f_sec=row.get("last_3f") or row.get("last_3f_sec"),
            weight=row.get("weight"),
            jockey_id=row.get("jockey_id"),
        ).on_conflict_do_nothing(constraint="uq_race_results_race_horse")
        session.execute(stmt)


def upsert_lap_times(session: Session, data: dict[str, Any]) -> None:
    race_id = data["race_id"]
    laps = data.get("lap_times") or data.get("laps") or []
    for idx, lap in enumerate(laps, start=1):
        furlong = lap.get("furlong_index") or idx
        stmt = insert(RaceLapTime).values(
            race_id=race_id,
            furlong_index=furlong,
            lap_time_sec=lap.get("lap_time_sec") or lap.get("lap"),
            cumulative_sec=lap.get("cumulative_sec"),
        ).on_conflict_do_update(
            index_elements=["race_id", "furlong_index"],
            set_={
                "lap_time_sec": lap.get("lap_time_sec") or lap.get("lap"),
                "cumulative_sec": lap.get("cumulative_sec"),
            },
        )
        session.execute(stmt)
    pace = data.get("pace_summary") or {}
    if pace or data.get("pace_category"):
        stmt = insert(RacePaceSummary).values(
            race_id=race_id,
            first_3f_sec=pace.get("first_3f_sec"),
            last_3f_sec=pace.get("last_3f_sec"),
            pace_category=pace.get("pace_category") or data.get("pace_category"),
            front_runner_count=pace.get("front_runner_count"),
        ).on_conflict_do_update(
            index_elements=["race_id"],
            set_={
                "first_3f_sec": pace.get("first_3f_sec"),
                "last_3f_sec": pace.get("last_3f_sec"),
                "pace_category": pace.get("pace_category") or data.get("pace_category"),
                "front_runner_count": pace.get("front_runner_count"),
            },
        )
        session.execute(stmt)


def upsert_corner_positions(session: Session, data: dict[str, Any]) -> None:
    race_id = data["race_id"]
    for row in data.get("corner_positions") or data.get("results") or []:
        if not row.get("horse_id"):
            continue
        corners = row.get("corners") or {}
        stmt = insert(RaceCornerPosition).values(
            race_id=race_id,
            horse_id=row["horse_id"],
            corner_1=row.get("corner_1") or corners.get("1"),
            corner_2=row.get("corner_2") or corners.get("2"),
            corner_3=row.get("corner_3") or corners.get("3"),
            corner_4=row.get("corner_4") or corners.get("4"),
        ).on_conflict_do_update(
            index_elements=["race_id", "horse_id"],
            set_={
                "corner_1": row.get("corner_1") or corners.get("1"),
                "corner_2": row.get("corner_2") or corners.get("2"),
                "corner_3": row.get("corner_3") or corners.get("3"),
                "corner_4": row.get("corner_4") or corners.get("4"),
            },
        )
        session.execute(stmt)


def upsert_odds_snapshot(
    session: Session,
    race_id: str,
    horse_id: str,
    snapshot_type: str,
    odds_value: float,
    snapshot_at: datetime,
    *,
    odds_place_low: Optional[float] = None,
    odds_place_high: Optional[float] = None,
) -> None:
    stmt = insert(RaceOddsSnapshot).values(
        race_id=race_id,
        horse_id=horse_id,
        snapshot_type=snapshot_type,
        odds_value=odds_value,
        odds_place_low=odds_place_low,
        odds_place_high=odds_place_high,
        snapshot_at=snapshot_at,
    ).on_conflict_do_nothing(constraint="uq_odds_snapshot")
    session.execute(stmt)


def transform_shutuba(session: Session, data: dict[str, Any]) -> str:
    upsert_race_from_shutuba(session, data)
    upsert_entries_from_shutuba(session, data)
    return race_path("race_shutuba", data["race_id"])
