"""予測の応答の段ごとの秒数（Server-Timing）。中身は段の名前と時間だけで、予測の値は入れない。"""

from __future__ import annotations

import re

from src.api.stage_timing import StageTimer, merge_server_timing


def test_stage_timer_header_and_token():
    tm = StageTimer()
    with tm.stage("primary"):
        pass
    try:
        with tm.stage("位置取り"):           # token に無い文字は _ に替える
            raise ValueError
    except ValueError:
        pass                                    # 例外でも段は残る
    names = [n for n, _ in tm.items()]
    assert names == ["primary", "____"]
    assert re.fullmatch(r"primary;dur=\d+\.\d, ____;dur=\d+\.\d", tm.header())
    assert merge_server_timing(None, "", "a;dur=1.0", "b;dur=2.0") == "a;dur=1.0, b;dur=2.0"


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


def test_predictions_endpoint_sends_stage_timing(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app as A

    st = FakeStorage({("race_shutuba", "R1"): {"entries": [{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1"}]}})
    monkeypatch.setattr(A, "_get_storage", lambda: st)
    monkeypatch.setattr(A, "_keiba_env", lambda: "stg")
    for k in ("KEIBA_STG_MOCK_PREDICTIONS", "KEIBA_PRERENDER", "KEIBA_PREWARM"):    # 事前生成の置き場に当たらないよう、毎回計算させる
        monkeypatch.delenv(k, raising=False)
    r = TestClient(A.app).get("/api/race/R1/predictions")
    assert r.status_code == 200 and r.json()["predictions"][0]["horse_name"] == "イチ"
    names = [part.split(";")[0].strip() for part in r.headers["Server-Timing"].split(",")]
    assert "entries_only" in names and names[-2:] == ["cache", "tier"]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+;dur=\d+\.\d", p.strip()) for p in r.headers["Server-Timing"].split(","))
