"""金曜週次: レース完了後の馬ID収集（キャッシュ無効化・shutuba フォールバック）。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_weekly_collect_prefers_race_result_entries() -> None:
    from src.scraper.auto_scrape_queue import _weekly_collect_horse_ids_from_specs

    storage = MagicMock()

    def _load(cat: str, rid: str) -> dict:
        if cat == "race_result" and rid == "202501010101":
            return {"entries": [{"horse_id": " 2015100001 "}, {"horse_id": "2015100002"}]}
        return {}

    storage.load.side_effect = _load
    s = _weekly_collect_horse_ids_from_specs(
        storage,
        [{"target_id": "202501010101"}],
    )
    assert s == {"2015100001", "2015100002"}
    storage.invalidate_load_cache.assert_called()


def test_weekly_collect_fallback_shutuba_when_result_empty() -> None:
    from src.scraper.auto_scrape_queue import _weekly_collect_horse_ids_from_specs

    storage = MagicMock()

    def _load(cat: str, rid: str) -> dict:
        if cat == "race_result":
            return {}
        if cat == "race_shutuba" and rid == "202501010102":
            return {"entries": [{"horse_id": "2015100999"}]}
        return {}

    storage.load.side_effect = _load
    s = _weekly_collect_horse_ids_from_specs(
        storage,
        [{"target_id": "202501010102"}],
    )
    assert s == {"2015100999"}
