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
