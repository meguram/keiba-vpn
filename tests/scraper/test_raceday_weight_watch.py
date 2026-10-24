from datetime import datetime, timedelta, timezone

from src.scraper import raceday_weight_watch as ww

JST = timezone(timedelta(hours=9))


def _sched():
    base = datetime(2026, 10, 17, 10, 0, tzinfo=JST)
    return [{"race_id": f"20260503{r:02d}", "post_time": base + timedelta(minutes=30 * (r - 1))} for r in range(1, 4)]


def _card(weights):
    return {"entries": [{"horse_number": i + 1, "weight": w} for i, w in enumerate(weights)]}


def test_plan_windows_uses_t65_to_t30_and_sorts():
    w = ww.plan_windows(list(reversed(_sched())))
    assert [x.race_id for x in w] == ["2026050301", "2026050302", "2026050303"]
    assert w[0].start == datetime(2026, 10, 17, 8, 55, tzinfo=JST)
    assert w[0].deadline == datetime(2026, 10, 17, 9, 30, tzinfo=JST)


def test_weights_ratio_and_threshold():
    assert ww.weights_ratio(_card([480, 500, None, 0, 470, 450, 460, 455, 470, 480])) == 0.8
    assert not ww.weights_announced(_card([480] * 8 + [None, None]))  # 0.8 < 0.9
    assert ww.weights_announced(_card([480] * 9 + [None]))  # 0.9
    assert ww.weights_ratio(None) == 0.0 and ww.weights_ratio({"entries": []}) == 0.0


def test_decide_announced_wait_and_deadline():
    w = ww.plan_windows(_sched())[0]
    mid = w.start + timedelta(minutes=10)
    assert ww.decide(_card([480] * 10), w, mid) == "announced"
    assert ww.decide(_card([None] * 10), w, mid) == "wait"
    assert ww.decide(_card([None] * 10), w, w.deadline) == "deadline"
    assert ww.decide(None, w, w.deadline + timedelta(minutes=1)) == "deadline"
    assert ww.decide(_card([480] * 10), w, w.deadline + timedelta(minutes=5)) == "announced"  # 期限後でも揃っていれば announced


def test_open_windows_and_next_wake():
    wins = ww.plan_windows(_sched())
    t0 = wins[0].start - timedelta(minutes=1)
    assert ww.open_windows(wins, set(), t0) == []
    assert ww.next_wake(wins, set(), t0, 90) == wins[0].start
    t1 = wins[0].start + timedelta(seconds=1)
    assert [w.race_id for w in ww.open_windows(wins, set(), t1)] == ["2026050301"]
    assert ww.next_wake(wins, set(), t1, 90) == t1 + timedelta(seconds=90)
    done = {w.race_id for w in wins}
    assert ww.open_windows(wins, done, t1) == [] and ww.next_wake(wins, done, t1, 90) is None
    # 開いている窓が全部済んだら、次の窓の開始まで眠る
    assert ww.next_wake(wins, {"2026050301"}, t1, 90) == wins[1].start


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("KEIBA_WEIGHT_WATCH_START_MIN", "70")
    monkeypatch.setenv("KEIBA_WEIGHT_WATCH_DEADLINE_MIN", "20")
    monkeypatch.setenv("KEIBA_WEIGHT_WATCH_POLL_SEC", "10")
    monkeypatch.setenv("KEIBA_WEIGHT_WATCH_MIN_RATIO", "0.5")
    w = ww.plan_windows(_sched())[0]
    assert (w.race["post_time"] - w.start).total_seconds() == 70 * 60
    assert (w.race["post_time"] - w.deadline).total_seconds() == 20 * 60
    assert ww.poll_sec() == 30.0  # 下限
    assert ww.weights_announced(_card([480, None]))
