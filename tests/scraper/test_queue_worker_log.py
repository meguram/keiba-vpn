"""
queue_worker_log: ファイルリングとメモリのマージ、API 応答形の検証。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# app 読み込みを軽くする（他の API テストと同様）
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")


@pytest.fixture
def ring_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """実データの .worker_log_ring.jsonl を触らない。"""
    p = tmp_path / "worker_log_ring.jsonl"

    import src.scraper.queue_worker_log as qwl

    monkeypatch.setattr(qwl, "_get_log_ring_file", lambda: p)
    qwl.clear_worker_logs()
    return p


def test_merge_file_and_memory_prefers_union(ring_path: Path) -> None:
    import src.scraper.queue_worker_log as qwl

    e_file = {
        "id": 999,
        "ts": 1000.0,
        "ts_iso_jst": "x",
        "tz": "Asia/Tokyo",
        "level": "INFO",
        "logger": "scraper.merge_test",
        "message": "only-in-file",
    }
    ring_path.write_text(json.dumps(e_file, ensure_ascii=False) + "\n", encoding="utf-8")

    with qwl._lock:
        qwl._buffer.append(
            {
                "id": 1,
                "ts": 1001.0,
                "ts_iso_jst": "x",
                "tz": "Asia/Tokyo",
                "level": "INFO",
                "logger": "scraper.merge_test",
                "message": "only-in-mem",
            }
        )

    snap = qwl._build_merged_log_snapshot()
    assert len(snap) == 2
    assert [e["message"] for e in snap] == ["only-in-file", "only-in-mem"]
    assert snap[0]["id"] == 0 and snap[1]["id"] == 1

    tail = qwl.get_worker_logs(after=-1, limit=10)
    assert tail["max_id"] == 1
    assert len(tail["entries"]) == 2
    assert tail["total_buffered"] == 2

    inc = qwl.get_worker_logs(after=0, limit=10)
    assert len(inc["entries"]) == 1
    assert inc["entries"][0]["message"] == "only-in-mem"


def test_memory_nonempty_still_includes_file(ring_path: Path) -> None:
    """旧バグ: メモリが1件でもあるとファイルを無視していた。"""
    import src.scraper.queue_worker_log as qwl

    ring_path.write_text(
        json.dumps(
            {
                "id": 1,
                "ts": 10.0,
                "ts_iso_jst": "x",
                "tz": "Asia/Tokyo",
                "level": "INFO",
                "logger": "scraper.file_body",
                "message": "from-disk",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with qwl._lock:
        qwl._buffer.append(
            {
                "id": 2,
                "ts": 11.0,
                "ts_iso_jst": "x",
                "tz": "Asia/Tokyo",
                "level": "INFO",
                "logger": "scraper.mem_noise",
                "message": "from-ram",
            }
        )

    g = qwl.get_worker_logs(after=-1, limit=50)
    msgs = {e["message"] for e in g["entries"]}
    assert msgs == {"from-disk", "from-ram"}


def test_api_worker_logs_json_shape() -> None:
    from fastapi.testclient import TestClient

    from src.api import auth
    from src.api.app import app

    c = TestClient(app, raise_server_exceptions=True)
    c.cookies.set(auth.COOKIE_NAME, auth._make_token())
    r = c.get("/api/scrape-queue/worker-logs?after=-1&limit=20")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "error" not in data or data.get("entries") is not None
    assert "entries" in data
    assert isinstance(data["entries"], list)
    assert "max_id" in data
    assert data.get("display_timezone") == "Asia/Tokyo"
