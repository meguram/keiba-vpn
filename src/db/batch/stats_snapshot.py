"""Layer 3 集計スナップショット生成（as_of_race_id 付き・追記のみ）。"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Iterable

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.models import (
    Entry,
    HorseStatsSnapshot,
    JockeyStatsSnapshot,
    Race,
    RaceResult,
    TrainerStatsSnapshot,
)
from src.db.session import get_session, init_engine

logger = logging.getLogger(__name__)


def _as_of_date(session: Session, as_of_race_id: str) -> date | None:
    race = session.get(Race, as_of_race_id)
    if race and race.race_date:
        return race.race_date
    if len(as_of_race_id) >= 8:
        return date(
            int(as_of_race_id[:4]),
            int(as_of_race_id[4:6]),
            int(as_of_race_id[6:8]),
        )
    return None


def _rates_from_results(session: Session, base_filter) -> dict:
    q = select(
        func.count().label("n"),
        func.avg((RaceResult.finish_pos == 1).cast(float)).label("win_rate"),
        func.avg((RaceResult.finish_pos <= 2).cast(float)).label("place_rate"),
        func.avg((RaceResult.finish_pos <= 3).cast(float)).label("show_rate"),
        func.avg(RaceResult.last_3f_sec).label("avg_last_3f"),
    ).select_from(RaceResult).join(Race, Race.race_id == RaceResult.race_id).where(base_filter)
    row = session.execute(q).one()
    return {
        "win_rate_all": row.win_rate,
        "place_rate_all": row.place_rate,
        "show_rate_all": row.show_rate,
        "avg_last_3f": row.avg_last_3f,
        "sample_count": row.n or 0,
    }


def horse_stats(session: Session, horse_id: str, as_of: date) -> dict:
    return _rates_from_results(
        session,
        and_(RaceResult.horse_id == horse_id, Race.race_date < as_of),
    )


def jockey_stats(session: Session, jockey_id: str, as_of: date) -> dict:
    return _rates_from_results(
        session,
        and_(RaceResult.jockey_id == jockey_id, Race.race_date < as_of),
    )


def trainer_stats(session: Session, trainer_id: str, as_of: date) -> dict:
    q_filter = and_(
        Entry.trainer_id == trainer_id,
        Race.race_date < as_of,
    )
    q = select(
        func.count().label("n"),
        func.avg((RaceResult.finish_pos == 1).cast(float)).label("win_rate"),
        func.avg((RaceResult.finish_pos <= 2).cast(float)).label("place_rate"),
        func.avg((RaceResult.finish_pos <= 3).cast(float)).label("show_rate"),
    ).select_from(RaceResult).join(
        Race, Race.race_id == RaceResult.race_id
    ).join(
        Entry, and_(Entry.race_id == RaceResult.race_id, Entry.horse_id == RaceResult.horse_id)
    ).where(q_filter)
    row = session.execute(q).one()
    return {
        "win_rate_all": row.win_rate,
        "place_rate_all": row.place_rate,
        "show_rate_all": row.show_rate,
        "sample_count": row.n or 0,
    }


def build_snapshots_for_race(session: Session, as_of_race_id: str) -> dict[str, int]:
    as_of = _as_of_date(session, as_of_race_id)
    if as_of is None:
        raise ValueError(f"cannot resolve as_of_date for {as_of_race_id}")

    entries = session.scalars(select(Entry).where(Entry.race_id == as_of_race_id)).all()
    horse_ids = {e.horse_id for e in entries}
    jockey_ids = {e.jockey_id for e in entries if e.jockey_id}
    trainer_ids = {e.trainer_id for e in entries if e.trainer_id}

    for hid in horse_ids:
        s = horse_stats(session, hid, as_of)
        session.execute(
            insert(HorseStatsSnapshot).values(
                horse_id=hid, as_of_race_id=as_of_race_id, as_of_date=as_of, **s
            ).on_conflict_do_nothing(constraint="uq_horse_stats_as_of")
        )
    for jid in jockey_ids:
        s = jockey_stats(session, jid, as_of)
        session.execute(
            insert(JockeyStatsSnapshot).values(
                jockey_id=jid, as_of_race_id=as_of_race_id, as_of_date=as_of,
                win_rate_all=s["win_rate_all"],
                place_rate_all=s["place_rate_all"],
                show_rate_all=s["show_rate_all"],
                sample_count=s["sample_count"],
            ).on_conflict_do_nothing(constraint="uq_jockey_stats_as_of")
        )
    for tid in trainer_ids:
        s = trainer_stats(session, tid, as_of)
        session.execute(
            insert(TrainerStatsSnapshot).values(
                trainer_id=tid, as_of_race_id=as_of_race_id, as_of_date=as_of,
                win_rate_all=s["win_rate_all"],
                place_rate_all=s["place_rate_all"],
                show_rate_all=s["show_rate_all"],
                sample_count=s["sample_count"],
            ).on_conflict_do_nothing(constraint="uq_trainer_stats_as_of")
        )

    return {"horses": len(horse_ids), "jockeys": len(jockey_ids), "trainers": len(trainer_ids)}


def verify_no_temporal_leak(session: Session, snapshot: HorseStatsSnapshot) -> bool:
    """CI 用: as_of_date 以前のデータのみで集計されていること。"""
    if snapshot.as_of_date is None:
        return False
    latest_prior = session.scalar(
        select(func.max(Race.race_date))
        .select_from(RaceResult)
        .join(Race, Race.race_id == RaceResult.race_id)
        .where(RaceResult.horse_id == snapshot.horse_id)
    )
    if latest_prior is None:
        return True
    return latest_prior < snapshot.as_of_date


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Layer 3 stats snapshot batch")
    parser.add_argument("as_of_race_id")
    args = parser.parse_args(list(argv) if argv is not None else None)
    init_engine()
    with get_session() as session:
        counts = build_snapshots_for_race(session, args.as_of_race_id)
    logger.info("snapshots built for %s: %s", args.as_of_race_id, counts)


if __name__ == "__main__":
    main()
