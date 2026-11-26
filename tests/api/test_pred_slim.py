"""予測の API を「AI の予測値・印・サンプル買い目」だけにする版（KEIBA_PRED_SLIM。既定 1）と、確定したレースの印の固定。"""

from __future__ import annotations

import pytest

from datetime import datetime, timedelta, timezone

from src.pipeline.inference.odds_source import pre_race_snapshot, race_start_at

JST = timezone(timedelta(hours=9))


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


ATTACH = [
    ("src.pipeline.inference.position_forecast", "attach_position_forecast"),
    ("src.pipeline.inference.pace_coherence", "attach_pace_coherence"),
    ("src.pipeline.inference.tracking_v2_service", "attach_tracking_v2_cached"),
    ("src.pipeline.inference.lap_forecast_service", "attach_lap_forecast"),
    ("src.pipeline.inference.pace_type", "attach_pace_type"),
    ("src.pipeline.inference.training_coverage", "attach_training_coverage"),
    ("src.pipeline.inference.expected_megu", "attach_expected_megu"),
]


@pytest.mark.parametrize("slim", ["", "1", "0"])
def test_slim_skips_every_add_on_but_keeps_marks_and_status(monkeypatch, slim):
    """既定（空）と 1 は付け足しを 1 つも呼ばない。0 は従来どおり全部呼ぶ。出馬表の段階・工程の表示は残す。"""
    import importlib

    from fastapi.testclient import TestClient

    from src.api import app as A

    called = []
    for mod, name in ATTACH:
        m = importlib.import_module(mod)
        monkeypatch.setattr(m, name, lambda resp, *a, _n=name, **k: called.append(_n) or resp)
    st = FakeStorage({("race_shutuba", "R1"): {"entries": [{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1"}]}})
    monkeypatch.setattr(A, "_get_storage", lambda: st)
    monkeypatch.setattr(A, "_keiba_env", lambda: "stg")
    for k in ("KEIBA_STG_MOCK_PREDICTIONS", "KEIBA_PRERENDER", "KEIBA_PREWARM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KEIBA_PRED_SLIM", slim)
    r = TestClient(A.app).get("/api/race/R1/predictions")
    assert r.status_code == 200 and r.json()["predictions"][0]["horse_name"] == "イチ"
    assert "entries_status" in r.json() and "pipeline_status" in r.json()
    assert called == ([n for _, n in ATTACH] if slim == "0" else [])


def test_member_only_groups_follow_the_switch(monkeypatch):
    from src.api import tier

    monkeypatch.setenv("KEIBA_PRED_SLIM", "1")
    names = tier.locked_group_names()
    assert "展開・ラップ・追走難度" not in names and "めぐ指数・成長度・クラス基準" not in names and "勝率・複勝率・順位" in names
    monkeypatch.setenv("KEIBA_PRED_SLIM", "0")
    assert tier.locked_group_names() == sorted(tier.LOCKED_GROUPS)


def _odds(phase, snaps, at="2026-11-22T10:05:00+09:00"):
    return {"race_id": "R1", "phase": phase, "snapshot_at": at,
            "entries": [{"horse_number": 1, "win_odds": 9.9}], "snapshots": snaps}


START = datetime(2026, 11, 22, 9, 50, tzinfo=JST)
T60 = {"phase": "t60", "snapshot_at": "2026-11-22T08:50:00+09:00", "entries": [{"horse_number": 1, "win_odds": 3.0}]}
T15 = {"phase": "t15", "snapshot_at": "2026-11-22T09:35:00+09:00", "entries": [{"horse_number": 1, "win_odds": 4.0}]}
T02_LATE = {"phase": "t02", "snapshot_at": "2026-11-22T09:50:01+09:00", "entries": [{"horse_number": 1, "win_odds": 5.0}]}


def test_pre_race_snapshot_picks_the_last_odds_taken_before_the_start():
    after = START + timedelta(hours=1)
    got = pre_race_snapshot(_odds("final", [T60, T02_LATE, T15]), START, now=after)
    assert got["phase"] == "t15" and got["entries"][0]["win_odds"] == 4.0 and "snapshots" not in got   # 発走の後に取った t02 は使わない
    assert pre_race_snapshot(_odds("final", [T60]), START, now=after)["phase"] == "t60"
    live = _odds("final", [T60, T15])
    assert pre_race_snapshot(live, START, now=START - timedelta(minutes=5)) is live   # 発走前は最新のまま
    assert pre_race_snapshot(live, None, now=after) is live                    # 発走の時刻が分からなければそのまま
    none_pre = _odds("final", [T02_LATE])
    assert pre_race_snapshot(none_pre, START, now=after) is none_pre           # 発走前の時点が無ければ確定のまま
    assert pre_race_snapshot(None, START) is None


def test_marks_do_not_change_when_the_final_odds_arrive():
    """本番の積み方（merge_snapshot）: 時点なしの取り直し（final）は発走前でも final で、確定オッズの final に上書きされる。
    発走の後に選ぶ時点は、確定オッズを取る前と後で同じ（印が 2 回変わらない）。"""
    from src.scraper.odds_snapshot import merge_snapshot

    def put(doc, phase, at, odds):
        return merge_snapshot(doc, {"race_id": "R1", "entries": [{"horse_number": 1, "win_odds": odds}]}, phase, ("entries",), snapshot_at=at)

    after = START + timedelta(minutes=30)
    d = put(None, "t60", "2026-11-22T08:50:00+09:00", 3.0)
    d = put(d, "t15", "2026-11-22T09:35:00+09:00", 4.0)
    d = put(d, "t05", "2026-11-22T09:45:00+09:00", 4.5)
    d = put(d, "final", "2026-11-22T09:48:00+09:00", 5.0)                       # 発走前の取り直し（時点なし）
    before_final = pre_race_snapshot(d, START, now=after)
    d = put(d, "final", "2026-11-22T16:30:00+09:00", 6.0)                       # 確定オッズ
    after_final = pre_race_snapshot(d, START, now=after)
    assert before_final["phase"] == after_final["phase"] == "t05" and after_final["entries"][0]["win_odds"] == 4.5
    top_phased = put(put(None, "t60", "2026-11-22T08:50:00+09:00", 3.0), "t05", "2026-11-22T09:45:00+09:00", 4.5)
    assert pre_race_snapshot(top_phased, START, now=after)["phase"] == "t05"   # 確定オッズの前で先頭が時点つきでも同じ時点


def test_race_start_at_reads_the_card():
    assert race_start_at({"date": "2026-11-22", "start_time": "09:50"}) == START
    assert race_start_at({"date": "20261122", "start_time": ""}) is None and race_start_at(None) is None


def test_local_payload_uses_pre_race_odds_only_when_asked(monkeypatch, tmp_path):
    """予測の画面（pre_race_odds=True）だけ発走前のオッズで印を付ける。サンプル買い目などほかの呼び出しは今のまま。"""
    import pandas as pd

    from src.api import local_predictions as L

    p = tmp_path / "R1.parquet"
    pd.DataFrame({"horse_id": ["h1"], "p_win": [0.3], "p_top2": [0.5], "p_top3": [0.6]}).to_parquet(p)
    monkeypatch.setattr(L, "find_local_prediction", lambda *a, **k: (p, "shadow", "v6e"))
    seen = []
    monkeypatch.setattr(L, "build_payload", lambda rid, card, odds_doc, *a, **k: seen.append(odds_doc) or {"ok": True})
    st = FakeStorage({("race_shutuba", "R1"): {"date": "2026-11-22", "start_time": "09:50"}, ("race_odds", "R1"): _odds("final", [T15])})
    L.load_local_prediction_payload("R1", st)
    L.load_local_prediction_payload("R1", st, pre_race_odds=True)
    assert seen[0]["phase"] == "final" and seen[1]["phase"] == "t15"
