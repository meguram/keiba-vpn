import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.ops import fetch_task as ft

T0 = datetime(2026, 10, 24, 10, 0, tzinfo=ft.JST)


def _t(task="race-odds-t60", nb=T0, **kw):
    args = kw.pop("args", {"race_id": "202605040101", "date": "20261024"})
    return ft.make_task(task, nb, args=args, **kw)


def test_allowlist_modes_exist_in_auto_scrape():
    src = Path("src/scraper/auto_scrape.py").read_text(encoding="utf-8")
    block = src[src.index("\nTASKS = {"):]
    block = block[: block.index("\n}")]
    modes = set(re.findall(r'^\s+"([a-z0-9-]+)":', block, re.M))
    assert ft.AUTO_SCRAPE_MODES <= modes


def test_validation_rejects_bad_tasks():
    with pytest.raises(ft.TaskError):
        ft.make_task("shell", T0)
    with pytest.raises(ft.TaskError):
        _t(args={"race_id": "123", "date": "20261024"})
    with pytest.raises(ft.TaskError):
        ft.make_task("entries-sync", T0, args={"netkeiba_password": "x"})
    with pytest.raises(ft.TaskError):
        ft.make_task("entries-sync", T0, args={"x": 1})
    ok = ft.make_task("entries-sync", T0)
    assert ok.task_id.startswith("entries-sync-20261024T1000")


def test_json_roundtrip_and_lead():
    t = _t("race-odds-t05")
    assert t.lead_seconds == 60 and t.start_at() == T0 - timedelta(seconds=60)
    assert ft.FetchTask.from_json(t.to_json()) == t
    for k in ("task_id", "task", "args", "not_before", "lead_seconds", "deadline", "attempts", "max_attempts", "claimed_by", "outcome"):
        assert k in t.to_json()


def test_enqueue_idempotent_and_due():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t()
    assert q.enqueue(t) and not q.enqueue(t)
    assert q.due_pending(T0 - timedelta(seconds=1)) == []
    assert [x.task_id for x in q.due_pending(T0)] == [t.task_id]


def test_lead_seconds_makes_it_due_early():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t("race-odds-t02")
    q.enqueue(t)
    assert q.due_pending(T0 - timedelta(seconds=61)) == []
    assert len(q.due_pending(T0 - timedelta(seconds=60))) == 1


def test_claim_is_exclusive():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t(); q.enqueue(t)
    a = q.claim(t.task_id, "vps-a", T0)
    b = q.claim(t.task_id, "host", T0)
    assert a is not None and a.attempts == 1 and a.claimed_by == "vps-a"
    assert b is None
    assert q.list_ids("pending") == [] and q.list_ids("running") == [t.task_id]


def test_claim_race_when_pending_still_visible():
    """両方が pending を読んだ後に claim しても、running の作成に成功するのは片方だけ。"""
    store = ft.MemoryTaskStore()
    q = ft.TaskQueue(store)
    t = _t(); q.enqueue(t)
    assert store.create(ft.key_of("running", t.task_id), "x")   # 他者が先に claim 済み
    assert q.claim(t.task_id, "me", T0) is None
    assert q.due_pending(T0) == []                               # running がある pending は起動候補にしない


def test_complete_and_fail_retry_then_failed():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t(max_attempts=2); q.enqueue(t)
    c = q.claim(t.task_id, "a", T0)
    assert q.fail(c, {"rc": 1}, T0) == "requeued"
    assert q.list_ids("pending") == [t.task_id] and q.list_ids("running") == []
    c = q.claim(t.task_id, "a", T0)
    assert c.attempts == 2
    assert q.fail(c, {"rc": 1}, T0) == "failed"
    assert q.list_ids("failed") == [t.task_id]
    t2 = _t("race-odds-t15"); q.enqueue(t2)
    q.complete(q.claim(t2.task_id, "a", T0), {"rc": 0}, T0)
    assert q.load("done", t2.task_id).outcome == {"rc": 0}


def test_no_retry_after_deadline():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t("race-odds-t10"); q.enqueue(t)
    c = q.claim(t.task_id, "a", T0)
    assert q.fail(c, {"rc": 1}, T0 + timedelta(minutes=10)) == "failed"


def test_expire_overdue_and_reap_stale():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    t = _t("race-odds-t10"); q.enqueue(t)
    assert q.expire_overdue(T0 + timedelta(seconds=300))[0].endswith("deadline_exceeded")
    assert q.list_ids("failed") == [t.task_id]
    t2 = _t("race-odds-t15", nb=T0 + timedelta(hours=1)); q.enqueue(t2)
    q.claim(t2.task_id, "a", T0 + timedelta(hours=1))
    assert q.reap_stale(900, T0 + timedelta(hours=1, minutes=5)) == []
    assert q.reap_stale(900, T0 + timedelta(hours=1, minutes=20)) == [f"{t2.task_id}:requeued"]
    assert q.list_ids("pending") == [t2.task_id]
