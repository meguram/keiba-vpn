"""定期実行ジョブの判定ルール（src/ops/schedule_rules.py）の境界。"""

from datetime import datetime, timedelta, timezone

import pytest

from src.ops import schedule_rules as R
from src.ops import scheduled_jobs as SJ

JST = timezone(timedelta(hours=9))


def ts(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=JST).timestamp()


def job(**kw):
    j = {"id": "j", "where": "gcp", "kind": "scheduler", "status": "active", "schedule": {"cron": "30 7 * * *"}, "applies": "daily",
         "grace_min": 30, "max_runtime_min": 10, "weight": "med", "what": "x"}
    j.update(kw)
    return j


def run(start, minutes=5, ok=True, code=0):
    s = ts(start)
    return {"start": s, "end": None if minutes is None else s + minutes * 60, "ok": ok if minutes is not None else None, "code": code}


NOW = "2026-11-13 09:00:00"   # 金曜


def test_cron_slots_follow_weekday_and_ranges():
    s = SJ.slots_before("*/20 12-21 * * 0,1,2,6", ts(NOW), n=2)
    assert [datetime.fromtimestamp(t, JST).strftime("%m/%d %H:%M") for t in s] == ["11/10 21:40", "11/10 21:20"]   # 火曜の最後
    assert SJ.slots_before("30 7 * * *", ts("2026-11-13 07:29:00"), n=1)[0] == ts("2026-11-12 07:30:00")
    with pytest.raises(ValueError):
        SJ.parse_cron("61 * * * *")


def test_on_time_is_green_and_grace_boundary_is_exact():
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 07:30:05")]}, ts(NOW))
    assert r["level"] == "green" and r["reasons"] == []
    # 予定 07:30 + 猶予 30 分ちょうどで、まだ開始なし → 未実行（前の日は動いた）
    r = R.evaluate_job(job(), {"runs": [run("2026-11-12 07:30:05")]}, ts("2026-11-13 08:00:00"))
    assert r["level"] == "yellow" and r["reasons"][0].startswith("未実行")
    # 猶予の 1 秒前なら、前の回（昨日）で判定して正常
    r = R.evaluate_job(job(), {"runs": [run("2026-11-12 07:30:05")]}, ts("2026-11-13 07:59:59"))
    assert r["level"] == "green"


def test_missed_twice_or_high_weight_is_red():
    r = R.evaluate_job(job(), {"runs": [run("2026-11-11 07:30:00")]}, ts(NOW))
    assert r["level"] == "red" and "2 回続けて" in r["reasons"][0]
    r = R.evaluate_job(job(weight="high"), {"runs": [run("2026-11-12 07:30:00")]}, ts(NOW))
    assert r["level"] == "red"


def test_late_start_and_double_start_are_caution():
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 07:50:00")]}, ts(NOW))   # 猶予の半分（15 分）より後
    assert r["level"] == "yellow" and r["reasons"][0].startswith("遅延")
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 07:30:00"), run("2026-11-13 07:31:00")]}, ts(NOW))
    assert r["level"] == "yellow" and "二重起動" in r["reasons"][0]


def test_long_running_yellow_then_red_at_twice():
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 08:45:00", minutes=None)]}, ts(NOW))   # 15 分（上限 10 分）
    assert r["level"] == "yellow" and r["reasons"][-1].startswith("長時間実行")
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 08:35:00", minutes=None)]}, ts(NOW))   # 25 分（2 倍超）
    assert r["level"] == "red"


def test_failures_count_consecutive():
    r = R.evaluate_job(job(), {"runs": [run("2026-11-13 07:30:00", ok=False, code=1), run("2026-11-12 07:30:00", ok=False, code=1),
                                        run("2026-11-11 07:30:00")]}, ts(NOW))
    assert r["level"] == "red" and "直近 2 回続けて失敗" in r["reasons"][0] and "終了コード 1" in r["reasons"][0]


def test_runtime_anomaly_against_median():
    runs = [run("2026-11-13 07:30:00", minutes=20)] + [run(f"2026-11-{d:02d} 07:30:00", minutes=5) for d in (12, 11, 10, 9)]
    r = R.evaluate_job(job(max_runtime_min=60), {"runs": runs}, ts(NOW))
    assert r["level"] == "yellow" and r["reasons"][0].startswith("実行時間の異常")


def test_interval_jobs_use_age_of_last_start_and_skip_fine_grained_rules():
    j = job(where="vps", kind="timer", schedule={"every_min": 5}, grace_min=15, weight="high")
    assert R.evaluate_job(j, {"runs": [run("2026-11-13 08:50:00", minutes=1)]}, ts(NOW))["level"] == "green"
    assert R.evaluate_job(j, {"runs": [run("2026-11-13 08:40:00", minutes=1)]}, ts(NOW))["level"] == "red"   # 20 分（high）
    runs = [run("2026-11-13 08:58:00", minutes=4)] + [run(f"2026-11-13 08:{m:02d}:00", minutes=1) for m in (53, 48, 43, 38)]
    assert R.evaluate_job(j, {"runs": runs}, ts(NOW))["level"] == "green"   # 短い間隔は実行時間の異常を見ない


def test_days_not_applicable_are_skipped_and_race_days_can_be_estimated():
    j = job(applies="race_days", schedule={"cron": "0 8 * * *"})
    # 金曜の 09:00: 開催日（土日）でないので、前の予定は 11/08（日）
    r = R.evaluate_job(j, {"runs": [run("2026-11-08 08:00:00")]}, ts(NOW))
    assert r["level"] == "green" and r["estimated"] is True
    r = R.evaluate_job(j, {"runs": [run("2026-11-08 08:00:00")]}, ts(NOW), race_days={"2026-11-13"})   # 金曜が開催日なら未実行
    assert r["level"] == "yellow" and r["estimated"] is False


def test_paused_or_retired_are_gray_but_unexpected_runs_are_caution():
    assert R.evaluate_job(job(status="planned"), {"runs": []}, ts(NOW))["level"] == "gray"
    r = R.evaluate_job(job(status="retired"), {"runs": [run("2026-11-13 07:30:00")]}, ts(NOW))
    assert r["level"] == "yellow" and "想定外の実行" in r["reasons"][0]


def test_no_record_is_unknown_and_daemon_state():
    assert R.evaluate_job(job(), {"runs": []}, ts(NOW))["level"] == "unknown"
    d = job(kind="daemon", schedule=None)
    assert R.evaluate_job(d, {"active": "failed"}, ts(NOW))["level"] == "red"
    assert R.evaluate_job(d, {"active": "active"}, ts(NOW))["level"] == "green"


def test_source_silence_is_one_alert_not_a_flood():
    jobs = [job(id=f"v{i}", where="vps", kind="timer", schedule={"every_min": 5}, grace_min=15) for i in range(5)]
    res = R.evaluate(jobs, {}, ts(NOW), sources={"vps": {"seen_at": ts("2026-11-13 08:14:00")}})   # 46 分前
    assert [a["key"] for a in res["alerts"]] == ["source:vps"] and res["alerts"][0]["level"] == "red"
    assert all(r["level"] == "unknown" for r in res["rows"]) and res["overall"] == "red"
    ok = R.evaluate(jobs, {f"v{i}": {"runs": [run("2026-11-13 08:58:00", 1)]} for i in range(5)}, ts(NOW),
                    sources={"vps": {"seen_at": ts("2026-11-13 08:16:00")}})   # 44 分前はまだ途絶でない
    assert ok["alerts"] == [] and ok["overall"] == "green"


def test_unreadable_gcp_and_unregistered_scheduler():
    res = R.evaluate([job()], {}, ts(NOW), sources={"gcp": {"error": "権限がありません"}},
                     unregistered=[{"name": "keiba-g-0530", "schedule": "30 5 * * *", "state": "ENABLED"}])
    keys = {a["key"]: a["level"] for a in res["alerts"]}
    assert keys == {"source:gcp": "unknown", "unregistered:keiba-g-0530": "yellow"}
    assert res["rows"][0]["level"] == "unknown" and res["overall"] == "yellow"   # 読めないことは全体を赤くしない


def test_transitions_only_report_changes():
    res = R.evaluate([job(weight="high")], {"j": {"runs": [run("2026-11-12 07:30:00")]}}, ts(NOW))
    new, gone = R.transitions({}, res)
    assert list(new) == ["job:j:未実行"] and gone == []
    new, gone = R.transitions(R.red_keys(res), res)
    assert new == {} and gone == []
    ok = R.evaluate([job(weight="high")], {"j": {"runs": [run("2026-11-13 07:30:00")]}}, ts(NOW))
    assert R.transitions(R.red_keys(res), ok) == ({}, ["job:j:未実行"])


def test_vps_rows_are_judged_at_the_time_the_record_was_written():
    j = job(id="v", where="vps", kind="timer", schedule={"every_min": 5}, grace_min=15, weight="high")
    o = {"v": {"runs": [run("2026-11-13 08:28:00", minutes=1)]}}
    # 記録は 08:30 に書かれ、今は 09:00（30 分前の記録。途絶の 45 分より前）。今の時刻で見ると 32 分空きで赤になってしまう
    res = R.evaluate([j], o, ts(NOW), sources={"vps": {"seen_at": ts("2026-11-13 08:30:00"), "as_of": ts("2026-11-13 08:30:00")}})
    assert res["rows"][0]["level"] == "green" and res["alerts"] == []


def test_no_silence_alarm_before_the_reporter_is_installed():
    jobs = [job(id="v", where="vps", kind="timer", schedule={"every_min": 5}, grace_min=15),
            job(id="rep", where="vps", kind="timer", schedule={"every_min": 15}, grace_min=20, status="planned", reporter=True)]
    res = R.evaluate(jobs, {}, ts(NOW), sources={"vps": {"seen_at": None}})
    assert res["alerts"] == [] and res["rows"][0]["level"] == "unknown"   # 見張りを入れる前は「記録がまだ無い」だけ
    jobs[1]["status"] = "active"
    assert R.evaluate(jobs, {}, ts(NOW), sources={"vps": {"seen_at": None}})["alerts"][0]["level"] == "red"


def test_short_interval_cron_still_detects_a_miss():
    j = job(schedule={"cron": "*/5 * * * *"}, grace_min=30)
    r = R.evaluate_job(j, {"runs": [run("2026-11-13 08:00:00", minutes=1)]}, ts(NOW))
    assert r["level"] in ("yellow", "red") and r["reasons"][0].startswith("未実行")
    assert SJ.parse_cron("5/20 * * * *")[0] == {5, 25, 45}


def test_red_key_changes_with_the_kind_and_unknown_is_not_recovery():
    hi = job(weight="high")
    missed = R.evaluate([hi], {"j": {"runs": [run("2026-11-12 07:30:00")]}}, ts(NOW))
    failed = R.evaluate([hi], {"j": {"runs": [run("2026-11-13 07:30:00", ok=False, code=1)]}}, ts(NOW))
    new, gone = R.transitions(R.red_keys(missed), failed)
    assert list(new)[0].startswith("job:j:失敗") and gone == [list(R.red_keys(missed))[0]]   # 理由が変われば知らせ直す
    blind = R.evaluate([hi], {}, ts(NOW))   # 記録が無くなった（判定不能）だけでは解消としない
    assert R.transitions(R.red_keys(missed), blind) == ({}, [])
