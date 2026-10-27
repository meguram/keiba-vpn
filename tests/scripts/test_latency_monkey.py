"""latency_monkey: 副作用のある GET を測らない／パラメータの割当／集計／外部通信の遮断。"""

import random
import socket

import pytest

from src.scripts.ops import latency_monkey as M

POOLS = {"items": [
    {"date": "20261017", "hot": True, "race_id": "202605040105", "horse_ids": ["2024108375"], "jockey_ids": ["01091"], "trainer_ids": ["01194"]},
    {"date": "20250101", "hot": False, "race_id": "202501010101", "horse_ids": ["2020100001"], "jockey_ids": [], "trainer_ids": []},
]}


def test_exclude_pattern_covers_side_effect_gets():
    for p in ("/api/scrape-queue/enqueue-missing", "/logout", "/api/admin/cron-jobs", "/docs", "/openapi.json", "/api/stallion-sire-tree/rebuild/status"):
        assert M.EXCLUDE.search(p), p
    for p in ("/api/race/{race_id}", "/api/race-list/{date}", "/api/v1/betting/race-values", "/api/health"):
        assert not M.EXCLUDE.search(p), p


def test_catalog_marks_dangerous_endpoints_excluded_and_keeps_the_main_ones():
    eps = {e.eid: e for e in M.build_catalog()}
    assert eps["F /api/scrape-queue/enqueue-missing"].excluded
    assert not eps["F /api/race/{race_id}/predictions"].excluded
    assert not eps["K /api/v1/betting/race-values"].excluded
    assert len([e for e in eps.values() if not e.excluded]) > 150


def test_param_sets_fill_path_and_query_from_pools():
    ep = M.Endpoint("F", "/api/race/{race_id}/entry/{horse_id}/card")
    (path, q), = M.param_sets(ep, POOLS, random.Random(1), 1)
    assert "{" not in path and path.startswith("/api/race/2")
    ep2 = M.Endpoint("F", "/api/race-quality/day", required=["date"])
    (p2, q2), = M.param_sets(ep2, POOLS, random.Random(1), 1)
    assert p2 == "/api/race-quality/day" and q2["date"] in ("20261017", "20250101")
    assert M.param_sets(M.Endpoint("F", "/api/health"), POOLS, random.Random(1), 3) == [("/api/health", {})]
    page = M.Endpoint("N", "/horse/{id}")
    assert M.param_sets(page, POOLS, random.Random(1), 1)[0][0].startswith("/horse/20")


def test_summarize_splits_process_cold_new_key_and_warm():
    rows = [
        {"eid": "F /a", "kind": "F", "path": "/a", "query": {}, "set": 0, "call": 0, "status": 200, "ms": 900.0, "bytes": 10, "net": 0},
        {"eid": "F /a", "kind": "F", "path": "/a", "query": {}, "set": 0, "call": 1, "status": 200, "ms": 10.0, "bytes": 10, "net": 0},
        {"eid": "F /a", "kind": "F", "path": "/a", "query": {}, "set": 1, "call": 0, "status": 200, "ms": 300.0, "bytes": 10, "net": 1},
        {"eid": "F /a", "kind": "F", "path": "/a", "query": {}, "set": 1, "call": 1, "status": 200, "ms": 20.0, "bytes": 10, "net": 0},
    ]
    s, = M.summarize(rows, [M.Endpoint("F", "/a")])
    assert (s["first_process_cold_ms"], s["first_new_key_ms"], s["warm_ms"], s["net_attempts"]) == (900.0, 300.0, 15.0, 1)
    assert M.bucket_counts([50, 200, 5000, 20000])["<0.1秒"] == 1 and M.bucket_counts([20000])["10秒以上"] == 1


def test_net_guard_blocks_and_counts_external_hosts_only():
    with M.NetGuard() as g:
        with pytest.raises(OSError):
            socket.getaddrinfo("db.netkeiba.com", 443)
        assert g.attempts == ["db.netkeiba.com"]
        socket.getaddrinfo("localhost", 80)          # 外部以外は通す
    socket.getaddrinfo("localhost", 80)
