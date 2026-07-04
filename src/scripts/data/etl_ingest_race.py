"""GCS JSON を PostgreSQL に取り込む CLI。"""

from __future__ import annotations

import argparse
import json
import logging

from src.db.etl.transform import transform_shutuba, upsert_lap_times, upsert_results
from src.db.scrape_runs import log_scrape_run
from src.db.session import get_session
from src.scraper.storage import HybridStorage

logger = logging.getLogger(__name__)


def ingest_race(category: str, race_id: str) -> None:
    storage = HybridStorage()
    data = storage.load(category, race_id)
    if not data:
        raise FileNotFoundError(f"{category}/{race_id} not found in storage")

    started = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    with get_session() as session:
        gcs_path = None
        if category == "race_shutuba":
            gcs_path = transform_shutuba(session, data)
        elif category in ("race_result", "race_result_on_time"):
            upsert_results(session, data)
            gcs_path = storage._gcs_blob_path(category, race_id)
        elif category == "race_result_lap":
            upsert_lap_times(session, data)
            gcs_path = storage._gcs_blob_path(category, race_id)
        else:
            raise ValueError(f"unsupported category: {category}")

        log_scrape_run(
            session,
            target_type=category,
            target_id=race_id,
            status="SUCCESS",
            started_at=started,
            gcs_path=gcs_path,
        )
    logger.info("ingested %s/%s", category, race_id)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="ETL: GCS JSON → PostgreSQL")
    parser.add_argument("category", help="race_shutuba / race_result / race_result_lap")
    parser.add_argument("race_id", help="12-digit race_id")
    args = parser.parse_args(argv)
    ingest_race(args.category, args.race_id)


if __name__ == "__main__":
    main()
