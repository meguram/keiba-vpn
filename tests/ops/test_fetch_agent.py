import json
from datetime import datetime, timedelta

from src.ops import fetch_agent as fa
from src.ops import fetch_owner as fo
from src.ops import fetch_task as ft

T0 = datetime(2026, 10, 24, 10, 0, tzinfo=ft.JST)
RACE = {"race_id": "202605040101", "date": "20261024"}


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t
    def sleep(self, s): self.t += timedelta(seconds=s)


def make(tasks=None, runner=None, t=T0):
    clk = Clock(t)
    q = ft.TaskQueue(ft.MemoryTaskStore())
    cfg = fa.AgentConfig(agent_id="vps-test", enabled=True,
                         tasks=frozenset(tasks or ft.ALLOWED_TASKS), access_pause_seconds=1800)
    calls = []

    def default_runner(task, timeout):
        calls.append((task.task_id, timeout, clk()))
        return fa.RunResult(rc=0, result={"errors": []})

    ag = fa.FetchAgent(q, cfg, runner or default_runner, clock=clk, sleep=clk.sleep)
    return ag, q, clk, calls


def test_idle_when_nothing_due():
    ag, q, clk, calls = make()
    q.enqueue(ft.make_task("race-odds-t60", T0 + timedelta(minutes=5), args=RACE))
    assert ag.tick() == "idle" and calls == []
    assert ag.next_sleep(clk()) == 60            # 5 分先でも poll 上限 60 秒


def test_runs_due_task_and_records_done():
    ag, q, clk, calls = make()
    t = ft.make_task("race-odds-t60", T0, args=RACE); q.enqueue(t)
    assert ag.tick() == f"ran:{t.task_id}:done"
    assert q.load("done", t.task_id).claimed_by == "vps-test" and calls[0][1] == 240


def test_waits_until_not_before_for_lead_tasks():
    ag, q, clk, calls = make(t=T0 - timedelta(seconds=45))
    t = ft.make_task("race-odds-t05", T0, args=RACE); q.enqueue(t)
    ag.tick()
    assert calls[0][2] == T0                      # 45 秒待ってから実行


def test_next_sleep_shortens_before_a_late_task():
    ag, q, clk, calls = make(t=T0 - timedelta(seconds=100))
    q.enqueue(ft.make_task("race-odds-t05", T0, args=RACE))
    assert ag.tick() == "idle"
    assert 38 <= ag.next_sleep(clk()) <= 41      # start_at = T0-60 まで約 40 秒


def test_only_owned_tasks_are_claimed():
    ag, q, clk, calls = make(tasks={"entries-sync"})
    t = ft.make_task("race-odds-t60", T0, args=RACE); q.enqueue(t)
    assert ag.tick() == "idle" and q.list_ids("pending") == [t.task_id]


def test_failure_requeues_then_fails():
    ag, q, clk, calls = make(runner=lambda t, to: fa.RunResult(rc=3, result={"errors": ["x"]}, tail="boom"))
    t = ft.make_task("race-odds-t15", T0, args=RACE, max_attempts=2); q.enqueue(t)
    assert ag.tick().endswith(":requeued")
    assert ag.tick().endswith(":failed")
    assert q.load("failed", t.task_id).outcome["tail"] == "boom"


def test_deadline_exceeded_is_not_run():
    ag, q, clk, calls = make(t=T0 + timedelta(seconds=300))
    t = ft.make_task("race-odds-t10", T0, args=RACE); q.enqueue(t)
    ag.tick()
    assert calls == [] and q.load("failed", t.task_id).outcome["reason"] == "deadline_exceeded"


def test_access_restriction_holds_until_a_human_clears_it():
    from src.ops import access_hold

    ag, q, clk, calls = make(runner=lambda t, to: fa.RunResult(rc=3, result={"errors": ["a"], "access_restriction": True}))
    t1 = ft.make_task("race-odds-t15", T0, args=RACE); q.enqueue(t1)
    t2 = ft.make_task("race-odds-t60", T0, args={"race_id": "202605040102", "date": "20261024"}); q.enqueue(t2)
    assert ag.tick().endswith(":failed")
    assert q.load("failed", t1.task_id)                                       # 再試行せず failed
    assert access_hold.read_hold() is not None
    assert ag.tick() == "held"
    clk.t += timedelta(days=1)                                                # 時間が経っても自動では再開しない
    assert ag.tick() == "held"
    access_hold.clear_hold()                                                  # 人が解除したあとだけ再開する
    assert ag.tick() != "held"


def test_weight_event_published_once():
    ag, q, clk, calls = make(runner=lambda t, to: fa.RunResult(rc=0, result={"errors": [], "announced": True, "ratio": 1.0}))
    t = ft.make_task("race-weight-check", T0, args=RACE); q.enqueue(t)
    ag.tick()
    key = "ops/events/weights_announced/20261024/202605040101.json"
    body = json.loads(q.store.get(key)[0])
    assert body["race_id"] == RACE["race_id"]
    assert q.store.create(key, "x") is False


def test_one_task_per_tick_in_time_order():
    ag, q, clk, calls = make(t=T0 + timedelta(minutes=1))
    a = ft.make_task("race-odds-t15", T0, args=RACE)
    b = ft.make_task("race-odds-t60", T0 - timedelta(minutes=1), args=RACE)
    q.enqueue(a); q.enqueue(b)
    ag.tick(); ag.tick()
    assert [c[0] for c in calls] == [b.task_id, a.task_id]


def test_build_argv_and_redact():
    t = ft.make_task("race-odds-t60", T0, args=RACE)
    assert fa.build_argv(t, "py")[-4:] == ["race", "race-odds-t60", RACE["race_id"], RACE["date"]]
    assert fa.build_argv(ft.make_task("entries-sync", T0), "py")[-2:] == ["auto", "entries-sync"]
    env = {"NETKEIBA_PASSWORD": "hunter2secret", "HOME": "/x"}
    out = fa.redact("login hunter2secret ok eyJabcdefghijklmnop.qrs", env)
    assert "hunter2secret" not in out and "eyJ" not in out


def test_config_from_env_defaults_off():
    cfg = fa.AgentConfig.from_env({})
    assert cfg.enabled is False
    cfg = fa.AgentConfig.from_env({"KEIBA_FETCH_AGENT_ENABLED": "1", "KEIBA_FETCH_AGENT_TASKS": "entries-sync,evil"})
    assert cfg.enabled and cfg.tasks == frozenset({"entries-sync"})


def test_fetch_owner(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"agent_tasks": ["entries-sync"]}))
    assert fo.host_must_skip("entries-sync", env={}, path=p)
    assert not fo.host_must_skip("entries-sync", env={"KEIBA_RUN_LOCATION": "vps"}, path=p)
    assert not fo.host_must_skip("recent-retry", env={}, path=p)
    assert fo.agent_tasks(tmp_path / "missing.json") == set()


def test_subprocess_runner_parses_result_and_times_out(monkeypatch, tmp_path):
    import sys
    t = ft.make_task("entries-sync", T0)
    monkeypatch.setattr(fa, "build_argv", lambda task, py: [sys.executable, "-c",
                        "print('x'); print('FETCH_RESULT {\"errors\": [], \"n\": 1}')"])
    r = fa.subprocess_runner(cwd=tmp_path)(t, 20)
    assert r.rc == 0 and r.result == {"errors": [], "n": 1} and not r.timed_out
    monkeypatch.setattr(fa, "build_argv", lambda task, py: [sys.executable, "-c", "import time; time.sleep(30)"])
    r = fa.subprocess_runner(cwd=tmp_path)(t, 1)
    assert r.timed_out and r.rc == -9


def test_child_task_lists(monkeypatch):
    from src.ops import fetch_child as fc
    monkeypatch.delenv("KEIBA_ODDS_TRIFECTA", raising=False)
    assert fc.race_task_list("race-odds-t60") == ["race_odds_t60", "race_pair_odds_t60"]
    assert fc.race_task_list("race-odds-t05") == ["race_odds_t05"]
    assert "race_shutuba" in fc.race_task_list("race-odds-t15")
    assert fc.race_task_list("race-weight-check") == ["race_shutuba"]
    assert fc.race_task_list("race-result-t15") == ["race_result_on_time"]


def test_host_runner_is_skipped_when_replacement_tasks_move(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"agent_tasks": ["race-odds-t15", "race-weight-check"]}))
    assert fo.host_must_skip("raceday-runner", env={}, path=p)
    assert fo.host_must_skip("raceday-weight-runner", env={}, path=p)
    assert not fo.host_must_skip("raceday-result-runner", env={}, path=p)
    p.write_text(json.dumps({"agent_tasks": ["race-result-t15"]}))
    assert fo.host_must_skip("raceday-result-runner", env={}, path=p)
    p.write_text(json.dumps({"agent_tasks": ["entries-sync"]}))
    assert not fo.host_must_skip("raceday-runner", env={}, path=p)
