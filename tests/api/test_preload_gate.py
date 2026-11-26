"""起動時の先読みの印: 予測の計算は先読みの完了を上限つきで待つ。印が下りない道でも待ち過ぎない。"""

from __future__ import annotations

import threading
import time

import pytest

from src.api import preload_gate as G


@pytest.fixture(autouse=True)
def _clean():
    G.reset()
    yield
    G.reset()


def test_no_wait_without_preload():
    assert not G.pending() and G.wait(5) == 0.0              # ホスト・テスト（begin が呼ばれない）では待たない


def test_waits_until_finish_then_never_again():
    G.begin()
    assert G.pending()
    threading.Timer(0.2, G.finish).start()
    waited = G.wait(5)
    assert 0.1 < waited < 2 and not G.pending() and G.wait(5) == 0.0


def test_wait_is_bounded_and_expires(monkeypatch):
    G.begin()
    t0 = time.monotonic()
    assert G.wait(0.2) < 1                                   # 上限で戻る（先読みが止まっていても要求は自分で読む）
    assert time.monotonic() - t0 < 1
    monkeypatch.setattr(G, "EXPIRE_SEC", 0.0)
    assert not G.pending() and G.wait(5) == 0.0              # finish の呼び忘れでも、期限を過ぎれば待たない


def test_predictions_compute_waits_for_preload(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app as A

    class St:
        def load(self, cat, key):
            return {"entries": [{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1"}]} if (cat, key) == ("race_shutuba", "R1") else None

    monkeypatch.setattr(A, "_get_storage", lambda: St())
    monkeypatch.setattr(A, "_keiba_env", lambda: "stg")
    for k in ("KEIBA_STG_MOCK_PREDICTIONS", "KEIBA_PRERENDER", "KEIBA_PREWARM"):
        monkeypatch.delenv(k, raising=False)
    c = TestClient(A.app)
    assert "preload_wait" not in c.get("/api/race/R1/predictions").headers["Server-Timing"]
    G.begin()
    threading.Timer(0.3, G.finish).start()
    r = c.get("/api/race/R1/predictions")
    assert r.status_code == 200 and r.json()["predictions"][0]["horse_name"] == "イチ"
    st = dict(p.strip().split(";dur=") for p in r.headers["Server-Timing"].split(","))
    assert float(st["preload_wait"]) >= 200                  # 先読みの完了まで待ってから読む
