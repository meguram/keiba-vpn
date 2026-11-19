"""定期実行の毎朝の点検（src/ops/jobs/schedule_check.py）: 赤があれば 1 通、無ければ送らない。"""

import json
import time

from src.ops.gcs_io import MemoryIO
from src.ops import scheduled_jobs as SJ
from src.ops.jobs import schedule_check as C


class Api:
    def __init__(self, ex):
        self.ex = ex

    def scheduler_jobs(self):
        return [{"name": "keiba-g-0730-shadow", "schedule": "30 7 * * *", "state": "ENABLED"}]

    def executions(self, run_job):
        return self.ex.get(run_job, [])


def test_red_sends_one_message_and_green_sends_none():
    now = time.time()
    io = MemoryIO()
    sent = []
    note = lambda title, lines, severity="warning": sent.append((title, lines, severity))   # noqa: E731
    old = SJ.slots_before("30 7 * * *", now - 2 * 86400, n=1)[0] + 10   # 2 日前の 07:30 の回だけ（その後の 2 回は未実行）
    two_days_ago = [{"start": old, "end": old + 60, "ok": True, "code": None}]
    out = C.run(io, now=now, api=Api({"keiba-dispatch-s": two_days_ago}), notify=note)   # 毎朝のグループが 2 回続けて未実行 → 赤
    assert out["sent"] and len(sent) == 1 and sent[0][2] == "critical" and any("gcp-g-0730-shadow" in x for x in sent[0][1])
    io.put_bytes("ops/heartbeat/vps/status.json", json.dumps({"written_at": now - 60, "jobs": {}}).encode())
    sent.clear()
    out = C.run(io, now=now, api=Api({"keiba-dispatch-s": [{"start": now - 600, "end": now - 500, "ok": True, "code": None}]}), notify=note, dry_run=False)
    assert all("VPS の記録の途絶" not in x for x in out["red"])


def test_dry_run_never_sends():
    out = C.run(MemoryIO(), now=time.time(), api=Api({"keiba-dispatch-s": [{"start": SJ.slots_before("30 7 * * *", time.time() - 2 * 86400, n=1)[0] + 10, "end": None, "ok": None, "code": None}]}),
                notify=lambda *a, **k: (_ for _ in ()).throw(AssertionError("送った")), dry_run=True)
    assert out["sent"] is False and out["red"]


def test_unreadable_gcp_is_reported_as_a_warning_not_silence():
    sent = []
    out = C.run(MemoryIO(), now=time.time(), api=None, notify=lambda title, lines, severity="warning": sent.append((title, lines, severity)) or {"ok": True})
    assert out["blind"] and sent and sent[0][2] == "warning" and any("GCP の状態を読めない" in x for x in sent[0][1])


def test_login_rotate_alerts_only_when_status_exists_and_not_switched():
    import json
    from datetime import datetime, timedelta, timezone

    from src.ops.gcs_io import MemoryIO
    from src.ops.jobs.schedule_check import login_rotate_alerts

    jst = timezone(timedelta(hours=9))
    io = MemoryIO()
    day1 = datetime(2027, 1, 1, 7, 30, tzinfo=jst).timestamp()
    assert login_rotate_alerts(io, day1) == ([], [])                        # 新方式がまだ動いていない
    put = lambda d: io.put_bytes("ops/login_rotate/status.json", json.dumps(d).encode())
    put({"step": "draft", "month": "2027-01", "result": "drafted", "ts": day1 - 4 * 86400})
    assert len(login_rotate_alerts(io, day1)[0]) == 1                       # 1 日なのに切り替えの記録が無い
    put({"step": "switch", "month": "2027-01", "result": "switched", "ts": day1 - 5400})
    assert login_rotate_alerts(io, day1) == ([], [])
    put({"step": "switch", "month": "2027-01", "result": "failed", "error": "x", "ts": day1 - 5400})
    assert len(login_rotate_alerts(io, day1 + 5 * 86400)[0]) == 1           # 今月の切り替えの失敗は、直るまで言い続ける
    put({"step": "prepare", "month": "2027-02", "result": "failed", "error": "x", "ts": day1 + 25 * 86400})
    assert len(login_rotate_alerts(io, day1 + 25 * 86400 + 5400)[0]) == 1   # 失敗が残っている
    assert login_rotate_alerts(io, day1 + 28 * 86400) == ([], [])           # 古い準備の失敗は言わない
    io.put_bytes("ops/login_rotate/status.json", b"[1, 2]")
    assert login_rotate_alerts(io, day1) == ([], ["共有ログインの切り替えの結果のファイルが壊れています"])   # 点検全体は止めない
    io.put_bytes("ops/login_rotate/status.json", json.dumps({"ts": "x"}).encode())
    assert login_rotate_alerts(io, day1)[1]


def test_login_rotate_failure_is_sent_with_other_reds(monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone

    from src.monitor import schedule_sources
    from src.ops.gcs_io import MemoryIO
    from src.ops.jobs import schedule_check as C

    io = MemoryIO()
    now = datetime(2027, 1, 1, 7, 30, tzinfo=timezone(timedelta(hours=9))).timestamp()
    io.put_bytes("ops/login_rotate/status.json", json.dumps({"step": "switch", "month": "2027-01", "result": "failed", "error": "次の版が無い", "ts": now - 5400}).encode())
    monkeypatch.setattr(schedule_sources, "build_view", lambda io_, now=None, api=None: {"result": {"alerts": [], "rows": []}, "alerts": [], "notes": [], "overall": "green", "counts": {}})
    sent = []
    out = C.run(io, now=now, notify=lambda title, lines, severity=None: sent.append((title, lines, severity)) or {"ok": True})
    assert out["sent"] and len(out["red"]) == 1 and "切り替え" in out["red"][0]
    assert sent[0][2] == "critical" and "次の版が無い" in sent[0][1][0]
