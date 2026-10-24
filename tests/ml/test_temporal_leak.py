"""テンポラルリーク検知（AREA-08 §2-4-1）。"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.db.batch.stats_snapshot import verify_no_temporal_leak
from src.db.models import HorseStatsSnapshot


def test_no_temporal_leak_in_snapshot_logic():
    """as_of_date 以前の成績のみ集計されていることを検証。"""
    snapshot = HorseStatsSnapshot(
        horse_id="2019105678",
        as_of_race_id="202506010811",
        as_of_date=date(2025, 6, 1),
    )
    session = MagicMock()
    session.scalar.return_value = date(2025, 5, 20)
    assert verify_no_temporal_leak(session, snapshot) is True

    session.scalar.return_value = date(2025, 6, 2)
    assert verify_no_temporal_leak(session, snapshot) is False
