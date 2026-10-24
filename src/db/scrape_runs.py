"""scrape_runs テーブル操作（F-5）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import ScrapeRun


def log_scrape_run(
    session: Session,
    *,
    target_type: str,
    target_id: str,
    status: str,
    started_at: datetime,
    finished_at: Optional[datetime] = None,
    retry_count: int = 0,
    gcs_path: Optional[str] = None,
    error_message: Optional[str] = None,
) -> ScrapeRun:
    run = ScrapeRun(
        target_type=target_type,
        target_id=target_id,
        status=status,
        retry_count=retry_count,
        started_at=started_at,
        finished_at=finished_at or datetime.now(timezone.utc),
        gcs_path=gcs_path,
        error_message=error_message,
    )
    session.add(run)
    session.flush()
    return run
