"""トラックバイアスの画面の値（ホストで作る serving・Cloud Run で読む view・結果の API）。合成のレースだけで、手元のデータは読まない。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.race_time import levels as L
from src.pipeline.track_bias import serving as S
from src.pipeline.track_bias import view as V
from src.research.track_bias import baseline as B
from src.research.track_bias import data as D
from src.research.track_bias import day as Y

TIE = "202601100190"


@pytest.fixture(autouse=True)
def _no_mirror(monkeypatch):
    monkeypatch.delenv("KEIBA_MIRROR_PULL", raising=False)        # 本物の GCS を読まない（fetch_one は何もしない）
    L._cache.clear()
    yield
    L._cache.clear()


def _race(rid, date, n=10, venue="東京", fins=None, pos=None, rnd=1):
    fins = fins if fins is not None else list(range(1, n + 1))
    pos = pos if pos is not None else list(range(1, n + 1))
    return pd.DataFrame({
        "race_id": rid, "date": date, "venue": venue, "surface": "芝", "distance": 1600, "grade": "1勝", "track_condition": "良",
        "start_time": "10:00", "round": rnd, "finish_position": fins, "bracket_number": [(i // 2) + 1 for i in range(n)],
        "horse_number": list(range(1, n + 1)), "horse_name": [f"馬{i}" for i in range(n)], "horse_id": [f"{rid}_{i}" for i in range(n)],
        "passing_order": [f"{p}-{p}" for p in pos], "last_3f": 34.0, "odds": [2.0 + i for i in range(n)],
        "popularity": list(range(1, n + 1)), "corner_passing": None})


def _built(main=None):
    rng = np.random.default_rng(0)
    parts = []
    for k in range(6):
        ts = pd.Timestamp("2025-12-06") + pd.Timedelta(days=7 * k)               # 年をまたぐ（2025・2026）
        for j in range(4):
            parts.append(_race(f"{ts:%Y%m%d}{j + 1:02d}{k:02d}"[:12], ts.strftime("%Y-%m-%d"), fins=list(rng.permutation(10) + 1), rnd=j + 1))
    parts.append(_race(TIE, "2026-01-10", fins=[1, 2, 2, 4, 5, 6, 7, 8, 9, 10], rnd=5))
    h = D.horse_values(D.clean(pd.concat(parts, ignore_index=True)))
    R = B.add_baseline(D.race_values(h), D.METRICS + D.AUX, {c: 50 for c in D.METRICS + D.AUX})
    cal = {"metrics": {c: {"芝": {"mu": 0.0, "sd": 1.0}} for c in D.METRICS + D.AUX}}
    if main:
        cal["main"] = main
    days = Y.apply(Y.day_raw(R), cal)                 # 主の値は cal の main に合う（無ければ既定）
    contrib = Y.race_contrib(R, days, cal)
    return h, days, contrib


def test_export_columns_and_top3(tmp_path):
    h, days, contrib = _built()
    d, r = S.export(h, days, contrib, Y.FRONT_MAIN, Y.INNER_MAIN)
    assert list(d.columns) == S.DAY_COLS and list(r.columns) == S.RACE_COLS
    t = json.loads(r.set_index("race_id").loc[TIE, "top3"])
    assert [x["pos"] for x in t] == [1, 2, 2]                                     # 同着はそのまま並べる
    assert t[0] == {"pos": 1, "horse_number": 1, "popularity": 1, "passing_order": "1-1", "early": 1, "corner4": 1, "field": 10}
    out = tmp_path / "page_reference" / "track_bias"
    files = S.write(d, r, out, {"version": "x"})
    assert set(files) == {"days_2025.parquet", "days_2026.parquet", "races_2025.parquet", "races_2026.parquet", "meta.json"}
    assert not list(tmp_path.rglob("*.tmp"))                                      # 一時ファイルは残らない（木の外に作って置き換える）


def test_export_follows_the_main_of_calibration():
    """校正の JSON の main を替えると、画面の内外の列もその値になる（名前の食い違いを見張る）。"""
    h, days, contrib = _built(main={"front": "front_4c", "inner": "inner_a"})
    try:
        assert (Y.FRONT_MAIN, Y.INNER_MAIN) == ("front_4c", "inner_a")
        d, r = S.export(h, days, contrib, Y.FRONT_MAIN, Y.INNER_MAIN)
        assert np.allclose(r["inner_se"], contrib["SE_inner_a"].round(3), equal_nan=True)
        assert np.allclose(r["front_se"], contrib["SE_front_4c"].round(3), equal_nan=True)
        assert np.allclose(d["z_inner"], days["z_inner_a"].round(3), equal_nan=True)
    finally:
        Y.set_main({})                                                             # 既定の主に戻す（ほかのテストに残さない）


def test_top3_without_passing_order_has_no_rank():
    h, _, _ = _built()
    h = h.copy()
    h.loc[(h["race_id"] == TIE) & (h["fin"] == 1), "passing_order"] = ""
    t = json.loads(S.top3_rows(h)[TIE])
    assert t[0]["passing_order"] is None and t[0]["early"] is None and t[0]["corner4"] is None


def _place(tmp_path):
    h, days, contrib = _built()
    d, r = S.export(h, days, contrib, Y.FRONT_MAIN, Y.INNER_MAIN)
    S.write(d, r, tmp_path / S.REL, {"version": "v1", "built": "2026-11-27 12:00", "last_date": str(d["date"].max())})
    return r


def test_view_reads_precomputed_values(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_REPO", tmp_path)
    assert V.result_block(TIE, "2026-01-10") is None                               # 置き場がまだ無い → 塊を付けない
    r = _place(tmp_path)
    b = V.result_block(TIE, "20260110")                                            # 日付は 8 桁でもよい
    assert b["available"] and b["day"]["surf"] == "芝" and b["day"]["label"] and "偶然" in b["note"]
    assert set(b["race"]) == {"front_se", "front_zc", "inner_se", "inner_zc", "front4c_se"} and len(b["top3"]) == 3
    assert V.result_block("209912310101", "2099-12-31") == {"available": False, "note": V.NOT_PLACED, "version": "v1"}   # その年の表が無い
    assert V.result_block("202601310101", "2026-01-31")["note"] == V.NOT_YET                                          # 作った日より後
    assert V.result_block("202601100199", "2026-01-10")["note"] == V.NOT_COVERED                                      # 作った日まで・対象外
    assert r["race_id"].str.len().eq(12).all()


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


def _doc():
    return {"race_name": "1勝", "date": "2026-01-10", "venue": "東京", "surface": "芝", "distance": 1600,
            "entries": [{"finish_position": 1, "horse_number": 1, "horse_name": "ア", "finish_time": "1:35.0"}],
            "payoff": [{"bet_type": "win", "bet_label": "単勝", "rows": [{"numbers": "1", "payout": 300, "popularity": 1}]}]}


@pytest.mark.parametrize("status_key,expect", [("race_result", True), ("race_result_on_time", False)])
def test_api_adds_track_bias_only_to_final_for_members(monkeypatch, tmp_path, status_key, expect):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(L, "_REPO", tmp_path)
    _place(tmp_path)
    monkeypatch.setattr(A, "_get_storage", lambda: FakeStorage({(status_key, TIE): _doc()}))
    monkeypatch.setattr(A, "is_member", lambda req: True)
    monkeypatch.setattr(A, "is_developer", lambda req: False)
    j = TestClient(A.app).get(f"/api/race/{TIE}/result").json()
    assert ("track_bias" in j) is expect                                          # 確定だけに付け、速報には付けない
    if expect:
        tb = j["track_bias"]
        assert tb["available"] and tb["day"]["venue"] == "東京" and [t["pos"] for t in tb["top3"]] == [1, 2, 2]


def test_api_without_files_has_no_track_bias(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(L, "_REPO", tmp_path)                                     # 置き場が無い（meta.json が無い）
    monkeypatch.setattr(A, "_get_storage", lambda: FakeStorage({("race_result", TIE): _doc()}))
    monkeypatch.setattr(A, "is_member", lambda req: True)
    monkeypatch.setattr(A, "is_developer", lambda req: False)
    j = TestClient(A.app).get(f"/api/race/{TIE}/result").json()
    assert j["status"] == "final" and "track_bias" not in j


def test_api_anonymous_gets_401(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(A, "is_member", lambda req: False)
    monkeypatch.setattr(A, "is_developer", lambda req: False)
    assert TestClient(A.app).get(f"/api/race/{TIE}/result").status_code == 401    # 結果の API は会員だけ（匿名に track_bias は出ない）


def test_serving_does_not_touch_the_ledger():
    t = (S.REPO / "src/pipeline/track_bias/serving.py").read_text(encoding="utf-8") + (S.REPO / "src/pipeline/track_bias/view.py").read_text(encoding="utf-8")
    assert "data_ledger" not in t and "tb_ledger" not in t                       # 本番の表示なので台帳は通さない（研究の build-all は通す）
