"""VPS の定期実行の見張り（src/ops/sched_status.py）: systemctl の読み取り・履歴・状態が変わったときだけの通知。"""

from datetime import datetime, timedelta, timezone

import pytest

from src.ops import sched_status as S

JST = timezone(timedelta(hours=9))


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=JST).timestamp()


def show(units):
    """systemctl show --timestamp=unix の出力を作る。"""
    blocks = []
    for uid, props in units.items():
        blocks.append("\n".join([f"Id={uid}", *[f"{k}={v}" for k, v in props.items()]]))
    return "\n\n".join(blocks) + "\n"


def svc(start, end, result="success", status="0", active="inactive"):
    return {"ActiveState": active, "Result": result, "ExecMainStatus": status,
            "ExecMainStartTimestamp": f"@{int(ts(start))}" if start else "", "ExecMainExitTimestamp": f"@{int(ts(end))}" if end else ""}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE", tmp_path / "state.json")
    return tmp_path


def _reader(units):
    return lambda jobs: S.parse_show(show(units))


def test_parse_show_and_timestamps():
    d = S.parse_show(show({"a.service": {"ExecMainStartTimestamp": "@1794500000"}, "b.timer": {"LastTriggerUSec": "n/a"}}))
    assert S._ts(d["a.service"]["ExecMainStartTimestamp"]) == 1794500000.0 and S._ts(d["b.timer"]["LastTriggerUSec"]) is None


def test_healthy_run_writes_once_and_sends_nothing(env):
    units = {"keiba-ui-health.service": svc("2026-11-13 08:55:00", "2026-11-13 08:55:02"),
             "keiba-ui-health.timer": {"UnitFileState": "enabled", "LastTriggerUSec": f"@{int(ts('2026-11-13 08:55:00'))}"}}
    written, sent = [], []
    out = S.run_once(now=ts("2026-11-13 09:00:00"), read=_reader(units), write=lambda d: written.append(d), notify=lambda t: sent.append(t) or True)
    assert len(written) == 1 and written[0]["jobs"]["vps-ui-health"]["runs"][0]["ok"] is True
    assert sent == [] and out["gcs_error"] is None


def test_red_is_sent_once_then_recovery_once(env):
    bad = {"keiba-ui-health.service": svc("2026-11-13 08:55:00", "2026-11-13 08:55:02", result="exit-code", status="1")}
    sent = []
    note = lambda t: sent.append(t) or True   # noqa: E731
    S.run_once(now=ts("2026-11-13 09:00:00"), read=_reader(bad), write=lambda d: None, notify=note)
    S.run_once(now=ts("2026-11-13 09:05:00"), read=_reader(bad), write=lambda d: None, notify=note)   # 同じ赤は送らない
    assert len(sent) == 1 and "異常" in sent[0] and "vps-ui-health" in sent[0]
    good = {"keiba-ui-health.service": svc("2026-11-13 09:25:00", "2026-11-13 09:25:01")}
    S.run_once(now=ts("2026-11-13 09:30:00"), read=_reader(good), write=lambda d: None, notify=note)
    assert len(sent) == 2 and "解消" in sent[1]


def test_slack_failure_retries_next_time_and_gcs_failure_does_not_stop_it(env):
    bad = {"keiba-ui-health.service": svc("2026-11-13 08:55:00", "2026-11-13 08:55:02", result="exit-code", status="1")}
    out = S.run_once(now=ts("2026-11-13 09:00:00"), read=_reader(bad), write=lambda d: "Forbidden", notify=lambda t: False)
    assert out["gcs_error"] == "Forbidden" and out["sent"] == []
    sent = []
    out = S.run_once(now=ts("2026-11-13 09:15:00"), read=_reader(bad), write=lambda d: "Forbidden", notify=lambda t: sent.append(t) or True)
    assert len(sent) == 1   # 前回送れなかった赤は、次の回に送る


def test_history_keeps_distinct_runs_and_updates_the_same_run(env):
    hist = {}
    jobs = [j for j in S.vps_jobs(S.SJ.load()) if j["id"] == "vps-release-pull-fetch"]
    running = S.parse_show(show({"keiba-release-pull@fetch.service": svc("2026-11-13 03:20:00", None, active="activating", result="success")}))
    o = S.observe(jobs, running, hist, ts("2026-11-13 03:21:00"))
    assert o["vps-release-pull-fetch"]["runs"][0]["end"] is None   # 実行中
    done = S.parse_show(show({"keiba-release-pull@fetch.service": svc("2026-11-13 03:20:00", "2026-11-13 03:22:00")}))
    o = S.observe(jobs, done, hist, ts("2026-11-13 03:30:00"))
    assert len(o["vps-release-pull-fetch"]["runs"]) == 1 and o["vps-release-pull-fetch"]["runs"][0]["ok"] is True
    nxt = S.parse_show(show({"keiba-release-pull@fetch.service": svc("2026-11-14 03:20:00", "2026-11-14 03:21:00")}))
    o = S.observe(jobs, nxt, hist, ts("2026-11-14 03:30:00"))
    assert len(o["vps-release-pull-fetch"]["runs"]) == 2


def test_dry_run_touches_nothing(env):
    out = S.run_once(now=ts("2026-11-13 09:00:00"), read=_reader({}), write=lambda d: pytest.fail("書いた"), notify=lambda t: pytest.fail("送った"), dry_run=True)
    assert "rows" in out and not (env / "state.json").exists()


def test_systemctl_failure_is_written_as_an_error_not_hidden(env):
    class Res:
        returncode, stdout, stderr = 1, "", "Unknown option --timestamp"
    written = []
    out = S.run_once(now=ts("2026-11-13 09:00:00"), read=lambda jobs: S.read_systemd(jobs, run=lambda *a, **k: Res()),
                     write=lambda d: written.append(d), notify=lambda t: True)
    assert "systemctl show が失敗" in out["error"] and written[0]["error"] and written[0]["jobs"] == {}
