"""開催日パイプラインの判定（取得 → 予測 → 結果 → 成績集計）。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.ops import pipeline_health as H
from src.ops.gcs_io import MemoryIO

JST = timezone(timedelta(hours=9))
RIDS = [f"2026050409{n:02d}" for n in range(1, 4)]
CATS = ("race_shutuba", "race_odds", "race_result_on_time", "race_index", "race_result")


class Fake:
    def __init__(self, have=CATS, events=None, files=None, results=None, days=("20261024", "20261025"), mx_none=False, rids=RIDS):
        self.have, self._events, self._files, self._results, self._days = set(have), events or [], files or {}, results, list(days)
        self.mx_none, self.rids = mx_none, rids

    def race_days(self, now):
        return self._days

    def matrix(self, date):
        if self.mx_none:
            return None
        return {"race_ids": self.rids, "race_meta": {r: {"venue": "阪神", "race_num": i + 1} for i, r in enumerate(self.rids)},
                "matrix": {r: {c: c in self.have for c in CATS} for r in self.rids}}

    def events(self, date):
        return self._events

    def pred_files(self, engine, date):
        return set(self._files.get(engine, ()))

    def results_summary(self, date):
        return self._results


def ev(rid, **kw):
    base = {"race_id": rid, "at": "2026-10-25T08:40:00+09:00", "status": "ok", "v3_status": "ok", "v5_status": "ok", "v6_status": "ok", "v6_sec": 30.0}
    base.update(kw)
    return base


def all_ok_src(**kw):
    return Fake(events=[ev(r) for r in RIDS], files={"v6": RIDS, "v5": RIDS, "v3": RIDS}, results={"status": "final", "built_at": "2026-10-26T20:00:45+09:00"}, **kw)


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=JST)


def test_complete_day_is_ok():
    d = H.day_health("20261025", at("2026-10-28T12:00:00"), all_ok_src())
    assert d["level"] == "ok" and all(s["state"] == "ok" for s in d["stages"].values())
    assert len(d["race_rows"]) == 3 and all(r["predict"] and r["final"] for r in d["race_rows"])


def test_engine_error_in_one_race_is_warned_even_if_primary_files_exist():
    src = all_ok_src()
    src._events[1] = ev(RIDS[1], v6_status="error", v6_error="boom")
    d = H.day_health("20261025", at("2026-10-28T12:00:00"), src)
    assert d["stages"]["predict"]["state"] == "warn" and d["level"] == "warn"
    assert any("boom" in n for n in d["notes"])


def test_missing_predictions_after_deadline_is_ng_but_before_is_wait():
    src = Fake(events=[], files={}, results=None)
    after = H.day_health("20261025", at("2026-10-25T19:00:00"), src)
    assert after["stages"]["predict"]["state"] == "ng" and after["level"] == "ng"
    before = H.day_health("20261025", at("2026-10-25T09:00:00"), Fake(have=("race_shutuba",), results=None))
    assert before["stages"]["predict"]["state"] == "wait" and before["stages"]["odds"]["state"] == "wait"
    assert before["stages"]["eve"]["state"] == "ok" and before["level"] == "wait"


def test_final_results_not_yet_published_is_wait_with_hint():
    src = Fake(have=("race_shutuba", "race_odds", "race_result_on_time", "race_index"), events=[ev(r) for r in RIDS], files={"v6": RIDS, "v5": RIDS, "v3": RIDS},
               results={"status": "provisional", "built_at": "2026-10-26T20:00:00+09:00"})
    d = H.day_health("20261025", at("2026-10-26T10:00:00"), src)
    assert d["stages"]["final"]["state"] == "wait" and "公開待ち" in d["stages"]["final"]["detail"]
    assert d["stages"]["results"]["state"] == "wait"
    later = H.day_health("20261025", at("2026-10-29T00:30:00"), src)
    assert later["stages"]["final"]["state"] == "ng" and later["stages"]["results"]["state"] == "ng"


def test_partial_coverage_is_warn_not_ng():
    rids = [f"2026050409{n:02d}" for n in range(1, 21)]
    src = Fake(rids=rids)
    src.matrix = lambda date: {"race_ids": rids, "race_meta": {}, "matrix": {r: {"race_odds": i < 19} for i, r in enumerate(rids)}}
    d = H.day_health("20261025", at("2026-10-25T19:00:00"), src)
    assert d["stages"]["odds"]["state"] == "warn" and d["stages"]["odds"]["done"] == 19


def test_unreadable_matrix_and_empty_day():
    d = H.day_health("20261025", at("2026-10-28T12:00:00"), Fake(mx_none=True))
    assert d["level"] == "na" and d["race_rows"] == []
    empty = H.day_health("20261031", at("2026-10-28T12:00:00"), Fake(rids=[]))
    assert empty["level"] == "wait" and "レース一覧" in empty["notes"][0]


def test_build_picks_recent_days_and_next_and_summary():
    src = all_ok_src(days=("20261017", "20261018", "20261024", "20261025", "20261031"))
    res = H.build(src, at("2026-10-28T12:00:00"), n_days=2)
    assert [d["date"] for d in res["days"]] == ["20261025", "20261024"] and res["next"]["date"] == "20261031"
    s = H.summary(res)
    assert s["status"] == "ok" and s["date"] == "20261025" and s["level"] == "ok" and s["next"]["date"] == "20261031"
    assert H.summary(None)["status"] == "not_collected"


def test_weekend_gaps_warns_only_for_weekends_without_shutuba_after_the_deadline():
    # 2026-10-28(水)。直後の 10/31(土)・11/1(日) は前日 20:00 前なので、まだ警告しない
    assert H.weekend_gaps(["20261025"], at("2026-10-28T12:00:00")) == []
    gaps = H.weekend_gaps(["20261025"], at("2026-10-30T21:00:00"))
    assert [g["date"] for g in gaps] == ["20261031"]            # 10/31 だけ期限（10/30 20:00）を過ぎた。11/1 はまだ
    assert H.weekend_gaps(["20261031"], at("2026-10-30T21:00:00")) == []


# ── GCS から読む実体 ──────────────────────────────────────────────

def _put(io, key, obj):
    io.put_bytes(key, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def gcs_fixture(full=True):
    io = MemoryIO()
    pc = H.PC
    groups = ("2026040404", "2026050409")                                                  # 10/25 の新潟・東京（各 12 レース）
    rids = [f"{g}{n:02d}" for g in groups for n in range(1, 13)]
    for rid in rids if full else rids[:5]:
        _put(io, f"{pc}/race_shutuba/2026/{rid}.json", {"date": "2026-10-25"})
    for rid in rids:
        for cat in ("race_odds", "race_result_on_time", "race_index", "race_result"):
            _put(io, f"{pc}/{cat}/2026/{rid}.json", {})
        io.put_bytes(f"{H.INF}/predictions_v6/20261025/{rid}.parquet", b"x")
    _put(io, f"{pc}/race_shutuba/2026/202610010101.json", {"date": "2026-10-04"})          # 古い開催日（窓の外）
    io.put_bytes(f"{H.INF}/events/20261025.jsonl", ("\n".join(json.dumps(ev(r)) for r in rids)).encode())
    _put(io, "ops/prediction_results/20261025.json", {"status": "final", "built_at": "2026-10-26T20:00:45+09:00"})
    return io


def test_gcs_sources_finds_race_days_and_judges_a_complete_day():
    io = gcs_fixture()
    res = H.build(H.GcsSources(io), at("2026-10-28T12:00:00"))
    assert [d["date"] for d in res["days"]] == ["20261025"]                            # 窓の外の開催日は出ない
    d = res["days"][0]
    assert d["races"] == 24 and d["level"] == "ok", d["stages"]
    assert {r["venue"] for r in d["race_rows"]} == {"新潟", "東京"}
    assert d["stages"]["predict"]["detail"].startswith("v6 24/24")


def test_gcs_sources_flags_missing_predictions_and_reads_dates_once():
    io = gcs_fixture()
    for k in list(io.list(f"{H.INF}/predictions_v6/")):
        io.delete(k)
    src = H.GcsSources(io)
    d = H.build(src, at("2026-10-28T12:00:00"))["days"][0]
    assert d["stages"]["predict"]["state"] == "ng" and d["level"] == "ng"
    n = len(src._group_date)
    src.race_days(at("2026-10-28T12:00:00"))
    assert len(src._group_date) == n                                                    # 日付の読み取りは覚えている


def test_gcs_sources_tolerates_broken_objects():
    io = gcs_fixture()
    io.put_bytes(f"{H.INF}/events/20261025.jsonl", b"{broken\n")
    io.put_bytes("ops/prediction_results/20261025.json", b"not json")
    d = H.build(H.GcsSources(io), at("2026-10-28T12:00:00"))["days"][0]
    assert d["stages"]["results"]["state"] == "ng"


def test_partial_entry_publication_is_not_reported_as_complete():
    d = H.build(H.GcsSources(gcs_fixture(full=False)), at("2026-10-24T21:00:00"))["next"]       # 前日 20:00 を過ぎても出馬表が 5/24 本だけ
    assert d["stages"]["eve"]["state"] == "ng" and d["stages"]["eve"]["done"] == 5 and d["stages"]["eve"]["total"] == 12   # 出馬表が 1 件も無い場は見つけられない（限界）


def test_group_date_is_found_even_when_race_one_is_missing():
    io = MemoryIO()
    _put(io, f"{H.PC}/race_shutuba/2026/202604041011.json", {"date": "2026-11-01"})       # 特別登録のレースだけ先に出ている
    src = H.GcsSources(io)
    assert src.race_days(at("2026-10-28T12:00:00")) == ["20261101"]
