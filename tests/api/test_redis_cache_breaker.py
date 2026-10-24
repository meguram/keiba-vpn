"""Redis 障害時に待たずにキャッシュミス扱いへ切り替わること（T-055）。外部接続なし。"""
import json
import unittest

from src.api.cache import redis_cache
from src.api.cache.redis_cache import PredictionCache, get_redis_client
from src.utils.circuit_breaker import CircuitBreaker


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class DownClient:
    def __init__(self):
        self.calls = 0

    def get(self, key):
        self.calls += 1
        raise ConnectionError("redis down")

    setex = delete = get


class MemClient:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def setex(self, key, ttl, value):
        self.d[key] = value

    def delete(self, key):
        self.d.pop(key, None)


def _breaker(clock):
    return CircuitBreaker("redis-test", failure_threshold=3, recovery_timeout=10.0, trip_on=(OSError,), clock=clock)


class PredictionCacheBreakerTest(unittest.TestCase):
    def test_round_trip_when_healthy(self):
        c = PredictionCache(client=MemClient(), breaker=_breaker(Clock()))
        c.set_prediction("r1", "v1", {"horses": [1]})
        self.assertEqual(c.get_prediction("r1", "v1"), {"horses": [1]})

    def test_outage_stops_touching_redis_after_threshold(self):
        client, clock = DownClient(), Clock()
        c = PredictionCache(client=client, breaker=_breaker(clock))
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                c.get_prediction("r1", "v1")
        before = client.calls
        # 開いた後は Redis に触れず、読みはキャッシュミス・書きは何もしない
        self.assertIsNone(c.get_prediction("r1", "v1"))
        self.assertIsNone(c.get_lap_prediction("r1", "v1"))
        self.assertIsNone(c.get_odds_snapshot("r1"))
        c.set_prediction("r1", "v1", {"x": 1})
        c.invalidate_entries("r1")
        self.assertEqual(client.calls, before)

    def test_recovers_via_probe(self):
        clock = Clock()
        breaker = _breaker(clock)
        down = DownClient()
        c = PredictionCache(client=down, breaker=breaker)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                c.get_prediction("r1", "v1")
        clock.t = 10.0
        c._client = MemClient()  # Redis が復旧
        c.set_prediction("r1", "v1", {"ok": True})
        self.assertEqual(breaker.state, "closed")
        self.assertEqual(json.loads(c._client.d["prediction:r1:v1"]), {"ok": True})

    def test_bad_json_does_not_trip_breaker(self):
        client, clock = MemClient(), Clock()
        client.d["prediction:r1:v1"] = "{not json"
        breaker = _breaker(clock)
        c = PredictionCache(client=client, breaker=breaker)
        for _ in range(5):
            with self.assertRaises(ValueError):
                c.get_prediction("r1", "v1")
        self.assertEqual(breaker.state, "closed")

    @unittest.skipIf(redis_cache.redis is None, "redis パッケージ未導入の環境ではスキップ")
    def test_client_has_short_timeouts(self):
        client = get_redis_client()
        kw = client.connection_pool.connection_kwargs
        self.assertLessEqual(kw["socket_connect_timeout"], 1.0)
        self.assertLessEqual(kw["socket_timeout"], 2.0)


if __name__ == "__main__":
    unittest.main()
