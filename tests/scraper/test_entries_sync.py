"""entries_sync の統合テスト（ストレージとキューは偽物。ネットワークに触れない）。"""

import json
from datetime import datetime

import pytest

from src.scraper import entries_phase as EP
from src.scraper import entries_sync as ES

JST = EP.JST


class FakeStorage:
    def __init__(self, lists, cards):
        self.lists, self.cards = lists, cards

    def list_keys(self, category, year=None):
        return sorted(self.lists) if category == "race_lists" else sorted(self.cards)

    def load(self, category, key, bypass_cache=False):
        if category == "race_lists":
            return self.lists.get(key)
        if category == "race_shutuba":
            return self.cards.get(key)
        return None

    def invalidate_load_cache(self, *a, **k):
        pass


class FakeQueue:
    def __init__(self):
        self.added = []
        self.on_add = None          # ジョブの完了（= 保存済みの出馬表が更新される）を模す

    def bulk_add_jobs(self, specs):
        self.added.extend(specs)
        if self.on_add:
            self.on_add()
        return {"created": len(specs), "requeued": 0, "duplicate": 0, "skipped": 0, "total_specs": len(specs)}


def _ent(n, **kw):
    base = {"horse_id": f"h{n}", "horse_name": f"馬{n}", "horse_number": 0, "bracket_number": 0, "jockey_id": "", "jockey_weight": 0, "weight": 0}
    base.update(kw)
    return base


def _rl(date, ids, cnt=3):
    return {"date": date, "races": [{"race_id": r, "round": i + 1, "venue": "京都", "race_name": f"R{i}", "entries_count": cnt} for i, r in enumerate(ids)],
            "_meta": {"race_list_complete": True}}


SCHEDULE = {
    "window_minutes": 25, "lookahead_days": 14, "entries_decided_at": {"weekday": "thu", "time": "12:00"},
    "checkpoints": {"wed": ["20:00"], "fri": ["16:05"]}, "critical": {"fri": ["16:05"]},
}


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ENTRIES_STATE_DIR", str(tmp_path / "meta" / "entries_state"))


def _run(now, storage, queue, **kw):
    return ES.run_entries_sync(now, storage=storage, queue=queue, schedule=SCHEDULE, wait=False, refresh_lists=False, **kw)


def test_not_due_does_nothing():
    st, q = FakeStorage({"20261024": _rl("20261024", ["202608040801"])}, {}), FakeQueue()
    r = _run(datetime(2026, 10, 21, 12, 0, tzinfo=JST), st, q)
    assert r == {"status": "skipped", "reason": "not-due"} and q.added == []


def test_registered_then_final_flow_and_history():
    ids = ["202608040801", "202608040802"]
    cards = {i: {"race_id": i, "entries": [_ent(1), _ent(2), _ent(3)]} for i in ids}
    st, q = FakeStorage({"20261024": _rl("20261024", ids)}, cards), FakeQueue()
    # 水曜 20:00 → 登録馬（暫定）として状態を記録
    r1 = _run(datetime(2026, 10, 21, 20, 5, tzinfo=JST), st, q)
    assert r1["status"] == "ok" and r1["fetched"] == 2 and r1["phases"] == {"registered": 2}
    assert [s["tasks"] for s in q.added] == [["race_shutuba"]] * 2 and all(s["smart_skip"] is False for s in q.added)
    assert all(s["priority"] < 0 for s in q.added)               # 開催日ランナーのジョブより後回し
    s1 = ES.load_state(ids[0])
    assert s1["phase"] == "registered" and s1["first_seen_phase"] == "registered" and len(s1["history"]) == 1
    # 同じ checkpoint は 1 回だけ
    assert _run(datetime(2026, 10, 21, 20, 10, tzinfo=JST), st, q) == {"status": "skipped", "reason": "not-due"}
    # 金曜 16:05 に確定（全頭に馬番）→ final。critical なので確定前のレースも取り直される
    for i in ids:
        cards[i] = {"race_id": i, "entries": [_ent(n, horse_number=n, bracket_number=n, jockey_id=f"j{n}") for n in (1, 2, 3)]}
    r2 = _run(datetime(2026, 10, 23, 16, 6, tzinfo=JST), st, q)
    assert r2["critical"] is True and r2["phases"] == {"final": 2}
    s2 = ES.load_state(ids[0])
    assert s2["phase"] == "final" and s2["first_seen_phase"] == "registered" and len(s2["history"]) == 2
    assert s2["updated_at"] >= s1["updated_at"]


def test_scratch_is_recorded_as_notable_change():
    rid = "202608040801"
    ents = [_ent(n, horse_number=n, bracket_number=n, jockey_id=f"j{n}") for n in (1, 2, 3)]
    cards = {rid: {"race_id": rid, "entries": ents}}
    st, q = FakeStorage({"20261024": _rl("20261024", [rid])}, cards), FakeQueue()
    _run(datetime(2026, 10, 21, 20, 5, tzinfo=JST), st, q)
    q.on_add = lambda: cards.update({rid: {"race_id": rid, "entries": ents[:2]}})        # 取り直しで 1 頭取消が反映される
    r = _run(datetime(2026, 10, 23, 16, 6, tzinfo=JST), st, q)
    assert any(c["type"] == "removed" and c["horse_id"] == "h3" for c in r["notable_changes"])
    assert any(c["type"] == "removed" for c in ES.load_state(rid)["changes"])


def test_final_races_are_skipped_outside_critical_and_today_excluded():
    rid_fin, rid_new = "202608040801", "202608040802"
    cards = {rid_fin: {"race_id": rid_fin, "entries": [_ent(n, horse_number=n, bracket_number=n, jockey_id="j") for n in (1, 2)]},
             rid_new: {"race_id": rid_new, "entries": [_ent(1), _ent(2)]}}
    st, q = FakeStorage({"20261024": _rl("20261024", [rid_fin, rid_new], cnt=2), "20261021": _rl("20261021", ["202608040701"])}, cards), FakeQueue()
    ES._atomic_write(ES.state_path(rid_fin), {"race_id": rid_fin, "phase": "final", "list_entries_count": 2})
    # 水曜 20:00（critical でない）: 確定済みは取らず、未確定だけ取る。今日(10/21)が対象日でも今日のレースは除外される
    r = _run(datetime(2026, 10, 21, 20, 5, tzinfo=JST), st, q)
    assert [s["target_id"] for s in q.added] == [rid_new] and r["to_fetch"] == 1


def test_dry_run_and_lock(tmp_path):
    rid = "202608040801"
    st, q = FakeStorage({"20261024": _rl("20261024", [rid])}, {}), FakeQueue()
    r = _run(datetime(2026, 10, 21, 20, 5, tzinfo=JST), st, q, dry_run=True)
    assert r["plan"][0]["refetch"] is True and q.added == []
    with ES._lock() as got1:
        assert got1 is True
        with ES._lock() as got2:
            assert got2 is False          # 二重起動は見送り


def test_update_state_roundtrip_json():
    card = {"entries": [_ent(1), _ent(2)]}
    st, ch = ES.update_state(None, None, card, "r1", "20261024", datetime(2026, 10, 21, 20, 0, tzinfo=JST), 2)
    json.dumps(st)                              # JSON にできる
    assert st["phase"] == "registered" and ch == [] and st["list_entries_count"] == 2
