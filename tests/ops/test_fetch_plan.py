from datetime import datetime, timedelta

from src.ops import fetch_agent as fa
from src.ops import fetch_plan as fp
from src.ops import fetch_task as ft

D = datetime(2026, 10, 24, tzinfo=ft.JST)


def sched(n=2):
    return [{"race_id": f"2026050401{i:02d}", "post_time": D.replace(hour=10) + timedelta(minutes=30 * i)} for i in range(1, n + 1)]


def names(tasks):
    return [t.task for t in tasks]


def test_plan_day_basic_counts():
    now = D.replace(hour=6)
    ts = fp.plan_day(sched(2), "20261024", now=now, weight_poll_sec=300)
    assert names(ts).count("race-odds-t60") == 2 and names(ts).count("race-odds-t15") == 2
    assert names(ts).count("race-odds-t10") == 0
    assert names(ts).count("race-weight-check") == 2 * 8          # T-65..T-30 を 5 分おき
    assert all(t.args["date"] == "20261024" for t in ts)
    assert [t.not_before for t in ts] == sorted(t.not_before for t in ts)


def test_plan_day_late_adds_three_kinds_with_lead():
    ts = fp.plan_day(sched(1), "20261024", late=True, weight=False, now=D.replace(hour=6))
    late = [t for t in ts if t.task in ("race-odds-t10", "race-odds-t05", "race-odds-t02")]
    assert len(late) == 3 and all(t.lead_seconds == 60 for t in late)
    post = sched(1)[0]["post_time"]
    assert sorted(ft.parse_ts(t.not_before) for t in late) == [post - timedelta(minutes=m) for m in (10, 5, 2)]


def test_plan_skips_past_events_and_redundant_t60():
    # 10:50: 1R(10:30 発走)の予定は全て過去。2R(11:00)の T-60 は T-15(10:45) を過ぎているので積まず、T-15 だけを今すぐ実行できる形で積む
    ts = fp.plan_day(sched(2), "20261024", now=D.replace(hour=10, minute=50), weight=False, result=False)
    assert [(t.task, t.args["race_id"]) for t in ts] == [("race-odds-t15", "202605040102")]
    # 10:20: 2R の T-60(10:00) は過去で積まない／T-15 は積む。1R の T-15(10:15) は猶予内なので積む
    ts = fp.plan_day(sched(2), "20261024", now=D.replace(hour=10, minute=20), weight=False, result=False)
    assert [t.task for t in ts if t.args["race_id"] == "202605040102"] == ["race-odds-t15"]


def test_plan_day_adds_result_task_15min_after_post():
    ts = fp.plan_day(sched(2), "20261024", now=D.replace(hour=6), weight=False)
    res = [t for t in ts if t.task == "race-result-t15"]
    assert [t.args["race_id"] for t in res] == ["202605040101", "202605040102"]
    assert sorted(ft.parse_ts(t.not_before) for t in res) == [s["post_time"] + timedelta(minutes=15) for s in sched(2)]
    assert "race-result-t15" in ft.ALLOWED_TASKS
    assert not [t for t in fp.plan_day(sched(2), "20261024", now=D.replace(hour=6), weight=False, result=False) if t.task == "race-result-t15"]


def test_enqueue_is_idempotent():
    q = ft.TaskQueue(ft.MemoryTaskStore())
    ts = fp.plan_day(sched(2), "20261024", now=D.replace(hour=6))
    assert sum(q.enqueue(t) for t in ts) == len(ts)
    assert sum(q.enqueue(t) for t in ts) == 0


def test_recurring_slot_is_stable():
    a = fp.recurring_task("entries-sync", D.replace(hour=12, minute=34, second=5))
    b = fp.recurring_task("entries-sync", D.replace(hour=12, minute=39, second=59))
    c = fp.recurring_task("entries-sync", D.replace(hour=12, minute=40))
    assert a.task_id == b.task_id != c.task_id


def test_weight_check_skipped_once_announced():
    clk = lambda: D.replace(hour=9, minute=10)
    q = ft.TaskQueue(ft.MemoryTaskStore())
    cfg = fa.AgentConfig(agent_id="vps-t", enabled=True, tasks=frozenset({ft.RACE_WEIGHT_TASK}))
    calls = []
    ag = fa.FetchAgent(q, cfg, lambda t, to: calls.append(t) or fa.RunResult(rc=0, result={"errors": []}), clock=clk, sleep=lambda s: None)
    args = {"race_id": "202605040101", "date": "20261024"}
    q.enqueue(ft.make_task(ft.RACE_WEIGHT_TASK, D.replace(hour=9, minute=0), args=args, task_id="w1", grace_seconds=900))
    q.store.create("ops/events/weights_announced/20261024/202605040101.json", "{}")
    assert ag.tick().endswith(":done") and calls == []
    assert q.load("done", "w1").outcome["skipped"] == "already_announced"
