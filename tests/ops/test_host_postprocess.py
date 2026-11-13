"""取得を VPS に渡したあとのホストの「後処理だけ」のジョブ（src/ops/host_postprocess.py）。"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone

from src.ops import host_postprocess as HP

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 14, 19, 10, tzinfo=JST)


def test_does_nothing_until_the_fetch_moves_to_vps():
    calls = []
    assert HP.run("eve-precompute", owned=set(), now=NOW, runner=lambda a, e: calls.append(a) or 0) == 0
    assert calls == []


def test_runs_existing_entrypoints_with_dates_when_moved(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_HOST_POSTPROCESS_LOCK_DIR", str(tmp_path))
    calls = []
    run = lambda a, e: calls.append((a, e.get("KEIBA_TRACKING_V2"))) or 0  # noqa: E731
    assert HP.run("eve-precompute", owned={"raceday-eve-fetch"}, now=NOW, runner=run) == 0
    assert calls == [(["-m", "src.ops.jobs.eve_precompute", "--date", "20261115"], "1")]
    calls.clear()
    HP.run("prediction-results-refresh", owned={"recent-retry"}, now=NOW, runner=run)
    assert [c[0][-1] for c in calls] == ["--skip-final", "--add-target-bets"] and calls[0][0][3] == "20261104"
    calls.clear()
    HP.run("track-speed-daily", owned={"raceday-evening"}, now=NOW, runner=run)
    assert calls[0][0] == ["-m", "src.scripts.data.track_speed_daily", "--from", "20261104", "--to", "20261114"]


def test_failure_rc_is_returned_and_stops_the_next_step(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_HOST_POSTPROCESS_LOCK_DIR", str(tmp_path))
    calls = []
    assert HP.run("prediction-results-refresh", owned={"recent-retry"}, now=NOW, runner=lambda a, e: calls.append(a) or 3) == 3
    assert len(calls) == 1                                                     # 前の段が失敗したら次の段は動かさない


def test_overlapping_run_is_skipped(monkeypatch, tmp_path):
    import fcntl

    monkeypatch.setenv("KEIBA_HOST_POSTPROCESS_LOCK_DIR", str(tmp_path))
    with open(tmp_path / "keiba_host_postprocess_track-speed-daily.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        calls = []
        assert HP.run("track-speed-daily", owned={"recent-retry"}, now=NOW, runner=lambda a, e: calls.append(a) or 0) == 0
        assert calls == []


def test_cron_script_parses():
    assert subprocess.run(["bash", "-n", "scripts/cron/install_host_postprocess_cron.sh"]).returncode == 0


def test_page_quality_check_skips_on_host_when_moved(monkeypatch, tmp_path, caplog):
    import json
    import sys

    from src.monitor import page_quality_check as PQ

    p = tmp_path / "o.json"
    p.write_text(json.dumps({"agent_tasks": ["page-quality-check"]}))
    monkeypatch.setenv("KEIBA_FETCH_OWNER_FILE", str(p))
    monkeypatch.delenv("KEIBA_RUN_LOCATION", raising=False)
    monkeypatch.setattr(sys, "argv", ["page_quality_check"])
    called = []
    for name in dir(PQ):
        if name.startswith("run") and callable(getattr(PQ, name)):
            monkeypatch.setattr(PQ, name, lambda *a, **k: called.append(1))
    PQ.main()
    assert called == []


def test_raceday_evening_skips_track_speed_on_vps(monkeypatch):
    import src.scraper.auto_scrape as AS
    import src.scraper.auto_scrape_queue as Q
    import src.scraper.period_runners as PR

    monkeypatch.setattr(AS, "_load_race_calendar", lambda: {})
    monkeypatch.setattr(AS, "_get_race_day_venues", lambda c, d: ["東京"])
    monkeypatch.setattr(Q, "_storage", lambda: object())

    class Qu:
        def bulk_add_jobs(self, specs):
            return {"created": len(specs)}

    monkeypatch.setattr(Q, "_queue", lambda: Qu())
    monkeypatch.setattr(Q, "_ensure_race_list_date", lambda s, q, d: True)
    monkeypatch.setattr(Q, "_kick_and_wait", lambda q, timeout_sec=0: (True, "", None))
    monkeypatch.setattr(PR, "collect_jra_race_job_specs_for_period", lambda *a, **k: ([{"target_id": "202605050101"}], {}))
    called = []
    monkeypatch.setattr(AS, "_trigger_track_speed_for_date", lambda d: called.append(d))
    monkeypatch.setenv("KEIBA_RUN_LOCATION", "vps")
    r = Q.run_raceday_evening_for_date("20261114")
    assert r["post_processing"] == "host" and called == []
    monkeypatch.delenv("KEIBA_RUN_LOCATION")
    Q.run_raceday_evening_for_date("20261114")
    assert called == ["2026-11-14"]                                          # ホストでは今どおり
