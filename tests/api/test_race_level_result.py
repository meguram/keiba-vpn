"""レース結果の API の race_level（会員・開発者だけ・確定だけ・値が無ければ算出なし）と、言葉の段・一つ抜きの開催日レベル。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.race_time import levels as L


def test_words_and_pace_edges():
    assert L.word(-0.6) == "速い" and L.word(-0.59) == "やや速い" and L.word(0.0) == "標準" and L.word(0.2) == "標準"
    assert L.word(0.21) == "やや時計が掛かる" and L.word(0.7) == "時計が掛かる" and L.word(None) is None
    assert L.word(-0.4, L.MEMBER_WORDS) == "高い" and L.word(0.1, L.MEMBER_WORDS) == "標準" and L.word(0.5, L.MEMBER_WORDS) == "低い"
    assert L.pace_label(-0.7) == "ハイ" and L.pace_label(0.0) == "平均" and L.pace_label(0.67) == "スロー"


def _write(tmp_path, member_visible=True):
    d = tmp_path / L.LEVEL_REL
    d.mkdir(parents=True)
    races = pd.DataFrame([{"race_id": "202605050501", "date": "20261122", "venue": "東京", "surf": "ダ", "race_level": -0.83, "race_level_raw": -1.21,
                           "race_word": "やや速い", "pace_label": "ハイ", "day_level": -0.52, "day_word": "やや速い", "member_level": -0.31,
                           "member_word": "やや高い", "n_other_races": 5}])
    races.to_parquet(d / "races_2026.parquet")
    pd.DataFrame([{"date": "20261122", "venue": "東京", "surf": "ダ", "n_races": 6, "day_level_per1k": -0.4, "day_word": "やや速い"}]).to_parquet(d / "days_2026.parquet")
    (d / "meta.json").write_text(json.dumps({"version": "race_time_v2", "built": "2026-11-26", "member_visible": member_visible}), encoding="utf-8")


def test_result_block_reads_precomputed_values(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_REPO", tmp_path)
    L._cache.clear()
    _write(tmp_path)
    b = L.result_block("202605050501", "2026-11-22")
    assert b["available"] and b["race"] == {"sec": -0.8, "raw_sec": -1.2, "word": "やや速い", "pace": "ハイ"}
    assert b["day"]["sec"] == -0.5 and b["day"]["n_races"] == 6 and b["member"]["sec"] == -0.3 and "T̂" in b["formula"]
    assert L.result_block("202605050502", "2026-11-22") == {"available": False, "note": "算出なし", "version": "race_time_v2"}


def test_member_level_hidden_when_the_check_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_REPO", tmp_path)
    L._cache.clear()
    _write(tmp_path, member_visible=False)
    assert L.result_block("202605050501", "20261122")["member"] == {"sec": None, "word": None}


def test_member_level_needs_two_other_races():
    row = {"race_level": -0.5, "race_word": "標準", "member_level": -0.3, "member_word": "やや高い", "n_other_races": 1}
    assert L.view_of(row, None, {})["member"] == {"sec": None, "word": None}


def test_loo_day_level_excludes_own_race():
    """一つ抜きの開催日レベル: 自分のレースが飛び抜けて速くても、自分の値で馬場を測らない。"""
    from src.pipeline.race_time.levels import _shrunk_median

    v = np.array([-3.0, 0.1, 0.0, -0.1])
    assert _shrunk_median(np.delete(v, 0), 2.0) == pytest.approx(0.0)


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


@pytest.mark.parametrize("member,expect", [(False, False), (True, True)])
def test_api_adds_race_level_only_for_members(monkeypatch, tmp_path, member, expect):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(L, "_REPO", tmp_path)
    L._cache.clear()
    _write(tmp_path)
    doc = {"race_name": "2歳未勝利", "date": "2026-11-22", "venue": "東京", "surface": "ダート", "distance": 1600,
           "entries": [{"finish_position": 1, "horse_number": 3, "horse_name": "ア", "finish_time": "1:37.5"}],
           "payoff": [{"bet_type": "win", "bet_label": "単勝", "rows": [{"numbers": "3", "payout": 300, "popularity": 1}]}]}
    monkeypatch.setattr(A, "_get_storage", lambda: FakeStorage({("race_result", "202605050501"): doc}))
    monkeypatch.setattr(A, "is_member", lambda req: member)
    monkeypatch.setattr(A, "is_developer", lambda req: False)
    r = TestClient(A.app).get("/api/race/202605050501/result")
    if not member:
        assert r.status_code == 401                                                                 # 結果の API はもともと会員だけ（匿名は 401）
        return
    assert r.status_code == 200 and ("race_level" in r.json()) is expect
    if expect:
        assert r.json()["race_level"]["race"]["word"] == "やや速い"


def test_provisional_result_has_no_race_level(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(L, "_REPO", tmp_path)
    L._cache.clear()
    _write(tmp_path)
    doc = {"date": "2026-11-22", "entries": [{"finish_position": 1, "horse_number": 3, "horse_name": "ア"}]}
    monkeypatch.setattr(A, "_get_storage", lambda: FakeStorage({("race_result_on_time", "202605050501"): doc}))
    monkeypatch.setattr(A, "is_member", lambda req: True)
    j = TestClient(A.app).get("/api/race/202605050501/result").json()
    assert j["status"] == "provisional" and "race_level" not in j                                    # 速報には付けない


def test_compute_leave_one_out_member_level():
    """compute(): 日・場・芝ダごとの一つ抜き。自分のレースが飛び抜けて速くても、メンバーレベルは自分の値で消されない。"""
    from src.pipeline.race_time.model import RaceTimeModel

    class Flat(RaceTimeModel):
        def predict(self, df, parts=False):
            z = np.full(len(df), 100.0)
            return {"full": z, "no_pace": z} if parts else z

    m = Flat()
    m.pace_ref, m.pace_sd = {"x|ダ|1000": 34.0}, {"x|ダ|1000": 0.5}
    races = pd.DataFrame({"race_id": [f"R{i}" for i in range(5)], "date": "20261122", "venue": "x", "surf": "ダ", "distance": 1000,
                          "course": "x|ダ|1000", "class_ord": 1, "age_ord": 2, "newcomer": 0, "field_size": 12, "front600": 34.0,
                          "y_top5": [97.0, 100.0, 100.2, 99.8, 100.0], "y_win": [97.0, 100.0, 100.2, 99.8, 100.0]})
    out, days = L.compute(races, m, k=2.0, win_model=m)
    r0 = out[out.race_id == "R0"].iloc[0]
    assert r0["race_level"] == pytest.approx(-3.0) and r0["n_other_races"] == 4
    assert r0["day_level_loo_per1k"] == pytest.approx(0.0 * 4 / 6)                                  # ほかの 4 レースの中央値 0 を 4/(4+2) で縮める
    assert r0["member_level"] == pytest.approx(-3.0) and r0["member_word"] == "高い"
    assert days.iloc[0]["n_races"] == 5 and days.iloc[0]["day_level_per1k"] == pytest.approx(0.0)
