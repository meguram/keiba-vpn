"""Redis 4 層キャッシュ（AREA-03 §5 / AREA-06 §6）。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None  # type: ignore

POST_RACE_TTL_SEC = 60
ODDS_TTL_SEC = 300


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_redis_client():
    if redis is None:
        raise RuntimeError("redis package is not installed")
    return redis.from_url(_redis_url(), decode_responses=True)


def prediction_key(race_id: str, model_version: str) -> str:
    return f"prediction:{race_id}:{model_version}"


def lap_prediction_key(race_id: str, model_version: str) -> str:
    return f"lap:prediction:{race_id}:{model_version}"


def odds_latest_key(race_id: str) -> str:
    return f"odds:latest:{race_id}"


def race_entries_key(race_id: str) -> str:
    return f"race:entries:{race_id}"


def race_results_key(race_id: str) -> str:
    return f"race:results:{race_id}"


def track_speed_key(date_str: str, venue: str) -> str:
    return f"track:speed:{date_str}:{venue}"


def ttl_until_post_time(post_time: datetime | None) -> int:
    """発走前: 発走時刻まで / 発走後: 60 秒。"""
    if post_time is None:
        return POST_RACE_TTL_SEC
    now = datetime.now(timezone.utc)
    if post_time.tzinfo is None:
        post_time = post_time.replace(tzinfo=timezone.utc)
    delta = (post_time - now).total_seconds()
    if delta > 0:
        return int(delta)
    return POST_RACE_TTL_SEC


class PredictionCache:
    """L2/L3 Redis — 予測・ラップ予測キャッシュ。"""

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = get_redis_client()
        return self._client

    def get_prediction(self, race_id: str, model_version: str) -> dict[str, Any] | None:
        raw = self.client.get(prediction_key(race_id, model_version))
        return json.loads(raw) if raw else None

    def set_prediction(
        self,
        race_id: str,
        model_version: str,
        payload: dict[str, Any],
        post_time: datetime | None = None,
    ) -> None:
        ttl = ttl_until_post_time(post_time)
        self.client.setex(
            prediction_key(race_id, model_version),
            ttl,
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def get_lap_prediction(self, race_id: str, model_version: str) -> dict[str, Any] | None:
        raw = self.client.get(lap_prediction_key(race_id, model_version))
        return json.loads(raw) if raw else None

    def set_lap_prediction(
        self,
        race_id: str,
        model_version: str,
        payload: dict[str, Any],
        post_time: datetime | None = None,
    ) -> None:
        ttl = ttl_until_post_time(post_time)
        self.client.setex(
            lap_prediction_key(race_id, model_version),
            ttl,
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def get_odds_snapshot(self, race_id: str) -> dict[str, Any] | None:
        raw = self.client.get(odds_latest_key(race_id))
        return json.loads(raw) if raw else None

    def set_odds_snapshot(self, race_id: str, payload: dict[str, Any]) -> None:
        self.client.setex(
            odds_latest_key(race_id),
            ODDS_TTL_SEC,
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def invalidate_entries(self, race_id: str) -> None:
        self.client.delete(race_entries_key(race_id))
