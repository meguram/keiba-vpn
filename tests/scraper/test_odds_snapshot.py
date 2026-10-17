"""T-60 / T-15 オッズスナップショットと 3連複取得（合成データのみ。外部アクセスなし）。"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.scraper import queue_tasks, schemas
from src.scraper.auto_scrape_queue import (
    ODDS_T60_TASKS,
    T15_RACE_TASKS,
    build_odds_events,
    select_due_events,
)
from src.scraper.odds_snapshot import ODDS_PAYLOAD_KEYS, PAIR_ODDS_PAYLOAD_KEYS, merge_snapshot
from src.scraper.parsers import OddsParser
from src.scraper.run import ScraperRunner
from src.scraper.verify_scrape_completeness import default_task_ids_for_entity

JST = ZoneInfo("Asia/Tokyo")


# ── 3連複パース ──────────────────────────────────────────────

class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _FakeSession:
    def __init__(self, by_type, status="result"):
        self.by_type = by_type
        self.status = status
        self.urls = []

    def get(self, url, timeout=0):
        self.urls.append(url)
        t = url.rsplit("type=", 1)[1].split("&")[0]
        if t not in self.by_type:
            return _Resp({}, 404)
        return _Resp({"status": self.status, "data": {"odds": {t: self.by_type[t]}}})


def test_parse_sanrenpuku_sorts_combo_and_ignores_garbage():
    sess = _FakeSession({
        "4": {"0102": ["10.5", "", "3"]},
        "7": {
            "010203": ["45.6", "", "2"],
            "030102": ["12.0", "", "1"],       # 順不同でも昇順 combo に正規化
            "010101": ["1.0", "", "9"],        # 重複馬番は捨てる
            "0102": ["9.9", "", "9"],          # 桁数違い
            "000102": ["9.9", "", "9"],        # 馬番 0
            "ab0102": ["9.9", "", "9"],
        },
    })
    out = OddsParser().parse_pair_odds_from_api(sess, "202605010101")
    assert [r["combo"] for r in out["sanrenpuku"]] == [[1, 2, 3], [1, 2, 3]]
    assert {r["odds"] for r in out["sanrenpuku"]} == {45.6, 12.0}
    assert out["umaren"] == [{"pair": [1, 2], "odds": 10.5, "popularity": 3}]
    assert out["wide"] == [] and out["umatan"] == []


def test_odds_api_needs_action_init_and_accepts_selling_status():
    # 発売中は status="middle"、action=init が無いと data が空（2026-10-17 の開催日に実測）
    win = {"01": ["3.3", "0", "1"], "02": ["8.3", "0", "6"]}
    plc = {"01": ["1.5", "2.0", "1"], "02": ["2.4", "3.1", "6"]}
    sess = _FakeSession({"1": win, "2": plc, "7": {"010203": ["115.8", "0", "39"]}}, status="middle")
    out = OddsParser().parse_from_api(sess, "202604040101")
    assert [e["win_odds"] for e in out["entries"]] == [3.3, 8.3]
    pair = OddsParser().parse_pair_odds_from_api(sess, "202604040101")
    assert pair["sanrenpuku"] == [{"combo": [1, 2, 3], "odds": 115.8, "popularity": 39}]
    assert sess.urls and all("action=init" in u for u in sess.urls)


def test_odds_api_ignores_unknown_status():
    sess = _FakeSession({"1": {"01": ["3.3", "0", "1"]}}, status="yoso")
    assert OddsParser().parse_from_api(sess, "202604040101")["entries"] == []


def test_sanrenpuku_failure_does_not_break_other_bet_types():
    class Boom(_FakeSession):
        def get(self, url, timeout=0):
            if "type=7" in url:
                raise RuntimeError("boom")
            return super().get(url, timeout)

    out = OddsParser().parse_pair_odds_from_api(Boom({"4": {"0102": ["10.5", "", "3"]}}), "202605010101")
    assert len(out["umaren"]) == 1 and out["sanrenpuku"] == []


# ── スナップショットのマージ ─────────────────────────────────

def _odds(v):
    return {"race_id": "r", "entries": [{"horse_number": 1, "win_odds": v}]}


def test_merge_first_snapshot_has_no_history():
    d = merge_snapshot(None, _odds(5.0), "t60", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:05:00+09:00")
    assert d["phase"] == "t60" and d["snapshot_at"] == "2026-05-02T09:05:00+09:00"
    assert "snapshots" not in d


def test_merge_different_phase_goes_to_history():
    t60 = merge_snapshot(None, _odds(5.0), "t60", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:05:00+09:00")
    t15 = merge_snapshot(t60, _odds(4.5), "t15", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:50:00+09:00")
    assert t15["phase"] == "t15" and t15["entries"][0]["win_odds"] == 4.5
    assert t15["snapshots"] == [
        {"phase": "t60", "snapshot_at": "2026-05-02T09:05:00+09:00", "entries": t60["entries"]}
    ]
    assert "snapshots" not in t15["snapshots"][0]


def test_merge_same_phase_overwrites_and_keeps_one_per_phase():
    t60 = merge_snapshot(None, _odds(5.0), "t60", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:05:00+09:00")
    t15 = merge_snapshot(t60, _odds(4.5), "t15", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:50:00+09:00")
    t15b = merge_snapshot(t15, _odds(4.2), "t15", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:52:00+09:00")
    assert t15b["entries"][0]["win_odds"] == 4.2
    assert [s["phase"] for s in t15b["snapshots"]] == ["t60"]      # t15 は履歴に残らない
    t60b = merge_snapshot(t15b, _odds(5.5), "t60", ODDS_PAYLOAD_KEYS, snapshot_at="2026-05-02T09:53:00+09:00")
    assert t60b["phase"] == "t60"
    assert [s["phase"] for s in t60b["snapshots"]] == ["t15"]       # 旧 t60 は消え、t15 が履歴へ


def test_merge_legacy_doc_without_phase_is_not_archived():
    legacy = _odds(6.0)
    d = merge_snapshot(legacy, _odds(5.0), "t60", ODDS_PAYLOAD_KEYS)
    assert "snapshots" not in d
    assert legacy == _odds(6.0)                                     # 引数は破壊しない


def test_merge_pair_odds_keeps_all_bet_types():
    base = {"race_id": "r", "umaren": [{"pair": [1, 2], "odds": 9.0}], "wide": [], "umatan": [],
            "sanrenpuku": [{"combo": [1, 2, 3], "odds": 30.0}]}
    t60 = merge_snapshot(None, base, "t60", PAIR_ODDS_PAYLOAD_KEYS, snapshot_at="a")
    t15 = merge_snapshot(t60, {**base, "sanrenpuku": [{"combo": [1, 2, 3], "odds": 28.0}]}, "t15",
                         PAIR_ODDS_PAYLOAD_KEYS, snapshot_at="b")
    snap = t15["snapshots"][0]
    assert snap["sanrenpuku"][0]["odds"] == 30.0 and snap["umaren"] and "race_id" not in snap
    assert schemas.validate("race_pair_odds", t15)["passed"]


# ── ScraperRunner の phase 引数 ──────────────────────────────

class _MemStorage:
    def __init__(self):
        self.d = {}

    def exists(self, cat, key):
        return (cat, key) in self.d

    def load(self, cat, key):
        return self.d.get((cat, key))

    def save(self, cat, key, data):
        self.d[(cat, key)] = data
        return True


class _StubOddsParser:
    def __init__(self, win, sanren):
        self.win, self.sanren = win, sanren

    def parse_from_api(self, session, rid):
        return {"race_id": rid, "entries": [{"horse_number": 1, "win_odds": self.win}]}

    def parse_pair_odds_from_api(self, session, rid):
        return {"race_id": rid, "umaren": [], "wide": [], "umatan": [],
                "sanrenpuku": [{"combo": [1, 2, 3], "odds": self.sanren}]}


def _runner(win=5.0, sanren=30.0):
    r = ScraperRunner.__new__(ScraperRunner)
    r.storage = _MemStorage()
    r.odds_parser = _StubOddsParser(win, sanren)
    r.client = type("C", (), {"_session": object()})()
    return r


def test_runner_without_phase_is_plain_overwrite():
    r = _runner()
    r.scrape_odds("202605010101", skip_existing=False)
    doc = r.storage.load("race_odds", "202605010101")
    assert set(doc) == {"race_id", "entries"}


def test_runner_phase_chain_t60_then_t15():
    r = _runner(win=5.0)
    r.scrape_odds("202605010101", skip_existing=False, phase="t60")
    r.odds_parser.win = 4.5
    r.scrape_odds("202605010101", skip_existing=False, phase="t15")
    doc = r.storage.load("race_odds", "202605010101")
    assert doc["phase"] == "t15" and doc["entries"][0]["win_odds"] == 4.5
    assert doc["snapshots"][0]["phase"] == "t60" and doc["snapshots"][0]["entries"][0]["win_odds"] == 5.0
    assert schemas.validate("race_odds", doc)["passed"]


def test_runner_pair_odds_saved_when_only_sanrenpuku_present():
    r = _runner()
    out = r.scrape_pair_odds("202605010101", skip_existing=False, phase="t60")
    assert out and out["phase"] == "t60" and out["sanrenpuku"][0]["odds"] == 30.0


# ── 開催日ランナーのイベント列 ───────────────────────────────

def _race(rid, hh, mm):
    return {"race_id": rid, "post_time": datetime(2026, 5, 2, hh, mm, tzinfo=JST)}


def test_build_odds_events_orders_by_time_and_kind():
    sched = [_race("A", 10, 5), _race("B", 10, 40), _race("C", 11, 5)]
    ev = build_odds_events(sched)
    got = [(e["race"]["race_id"], e["kind"], e["at"].strftime("%H:%M")) for e in ev]
    assert got == [
        ("A", "t60", "09:05"), ("B", "t60", "09:40"), ("A", "t15", "09:50"),
        ("C", "t60", "10:05"), ("B", "t15", "10:25"), ("C", "t15", "10:50"),
    ]
    # 同時刻なら t60 が先
    tie = build_odds_events([_race("X", 10, 15), _race("Y", 11, 0)])
    # X t15 = 10:00, Y t60 = 10:00
    at_10 = [(e["race"]["race_id"], e["kind"]) for e in tie if e["at"].strftime("%H:%M") == "10:00"]
    assert at_10 == [("Y", "t60"), ("X", "t15")]


def test_select_due_events_skips_t60_already_past_t15_and_is_idempotent():
    sched = [_race("A", 10, 5), _race("B", 10, 40)]
    ev = build_odds_events(sched)
    done: set = set()

    # 09:20: A の T-60 のみ到達
    now = datetime(2026, 5, 2, 9, 20, tzinfo=JST)
    t60, t15, skipped = select_due_events(ev, done, now)
    assert [e["race"]["race_id"] for e in t60] == ["A"] and not t15 and not skipped

    # 起動が遅れて 10:00（A の T-15 後・B の T-15 前）: A は t60 スキップ+t15 実行、B は t60 実行
    now = datetime(2026, 5, 2, 10, 0, tzinfo=JST)
    t60, t15, skipped = select_due_events(ev, done, now)
    assert [e["race"]["race_id"] for e in skipped] == ["A"]
    assert [e["race"]["race_id"] for e in t60] == ["B"]
    assert [e["race"]["race_id"] for e in t15] == ["A"]

    for e in t60 + t15 + skipped:
        done.add((e["race"]["race_id"], e["kind"]))
    assert select_due_events(ev, done, now) == ([], [], [])


def test_t15_tasks_use_phase_variants_and_t60_is_odds_only():
    assert "race_odds_t15" in T15_RACE_TASKS and "race_pair_odds_t15" in T15_RACE_TASKS
    assert "race_odds" not in T15_RACE_TASKS and "race_pair_odds" not in T15_RACE_TASKS
    assert ODDS_T60_TASKS == ["race_odds_t60", "race_pair_odds_t60"]


# ── タスクカタログ ───────────────────────────────────────────

def test_scheduled_odds_tasks_are_registered_but_hidden_from_default_verification():
    ids = {t["id"]: t for t in queue_tasks.TASK_CATALOG}
    for tid in ("race_odds_t60", "race_pair_odds_t60", "race_odds_t15", "race_pair_odds_t15"):
        assert ids[tid]["entity"] == "race" and ids[tid]["scheduled"] is True
        assert queue_tasks.validate_tasks_for_kind("race", [tid]) is None
    defaults = default_task_ids_for_entity("race")
    assert "race_odds" in defaults and "race_odds_t60" not in defaults


def test_queue_task_passes_phase_to_runner():
    calls = []

    class R:
        def scrape_odds(self, rid, skip_existing=True, phase=None):
            calls.append(("odds", rid, skip_existing, phase))
            return {"entries": [1]}

        def scrape_pair_odds(self, rid, skip_existing=True, phase=None):
            calls.append(("pair", rid, skip_existing, phase))
            return {"umaren": [1]}

    for task in ("race_odds_t60", "race_pair_odds_t15"):
        queue_tasks._race_task(R(), "202605010101", task, "20260502", smart_skip=False)
    assert calls == [("odds", "202605010101", False, "t60"), ("pair", "202605010101", False, "t15")]


# ── phase 指定なしの再取得（夕方の確定オッズ・backfill）が履歴を壊さない ──

def _runner_with(existing_odds=None, existing_pair=None):
    from types import SimpleNamespace

    from src.scraper import run as R

    r = R.ScraperRunner.__new__(R.ScraperRunner)
    saved: dict = {}

    class S:
        def exists(self, cat, key):
            return False

        def load(self, cat, key):
            return {"race_odds": existing_odds, "race_pair_odds": existing_pair}.get(cat)

        def save(self, cat, key, data):
            saved[cat] = data
            return True

    class P:
        def parse_from_api(self, session, rid):
            return {"race_id": rid, "entries": [{"horse_number": 1, "win_odds": 3.0}]}

        def parse_pair_odds_from_api(self, session, rid):
            return {"race_id": rid, "umaren": [{"pair": [1, 2], "odds": 9.0}], "wide": [], "umatan": [], "sanrenpuku": []}

    r.storage, r.odds_parser, r.client = S(), P(), SimpleNamespace(_session=object())
    return r, saved


T15_DOC = {"race_id": "r", "entries": [{"horse_number": 1, "win_odds": 5.0}], "phase": "t15", "snapshot_at": "2026-10-11T14:15:00+09:00",
           "snapshots": [{"phase": "t60", "snapshot_at": "2026-10-11T13:00:00+09:00", "entries": [{"horse_number": 1, "win_odds": 6.0}]}]}


def test_phaseless_refetch_keeps_history_as_final():
    r, saved = _runner_with(existing_odds=T15_DOC)
    r.scrape_odds("r", skip_existing=False)  # 夕方 raceday-evening / backfill 相当
    doc = saved["race_odds"]
    assert doc["phase"] == "final" and doc["entries"][0]["win_odds"] == 3.0
    assert sorted(s["phase"] for s in doc["snapshots"]) == ["t15", "t60"]  # 発走前の履歴が残る


def test_phaseless_pair_refetch_keeps_history_as_final():
    existing = {"race_id": "r", "umaren": [{"pair": [1, 2], "odds": 12.0}], "wide": [], "umatan": [], "sanrenpuku": [],
                "phase": "t15", "snapshot_at": "x", "snapshots": [{"phase": "t60", "snapshot_at": "y", "umaren": [], "wide": [], "umatan": [], "sanrenpuku": []}]}
    r, saved = _runner_with(existing_pair=existing)
    r.scrape_pair_odds("r", skip_existing=False)
    doc = saved["race_pair_odds"]
    assert doc["phase"] == "final" and sorted(s["phase"] for s in doc["snapshots"]) == ["t15", "t60"]


def test_phaseless_refetch_of_legacy_doc_is_plain_overwrite():
    r, saved = _runner_with(existing_odds={"race_id": "r", "entries": [{"horse_number": 1, "win_odds": 9.0}]})
    r.scrape_odds("r", skip_existing=False)
    assert "phase" not in saved["race_odds"] and "snapshots" not in saved["race_odds"]
    r2, saved2 = _runner_with(existing_odds=None)
    r2.scrape_odds("r", skip_existing=False)
    assert "phase" not in saved2["race_odds"]


def test_evening_job_uses_final_phase_tasks():
    from src.scraper import auto_scrape_queue as Q, queue_tasks as QT

    src = open(Q.__file__, encoding="utf-8").read()
    assert '"race_odds_final", "race_pair_odds_final"' in src
    assert QT._ODDS_PHASE_TASKS["race_odds_final"] == "final" == QT._ODDS_PHASE_TASKS["race_pair_odds_final"]
