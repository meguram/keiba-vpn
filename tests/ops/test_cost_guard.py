"""費用の歯止め（src.ops.cost_guard）。課金・Scheduler・GCS・Slack はすべて差し替えで、実際には問い合わせない。"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from src.ops import cost_guard as CG
from src.ops.gcs_io import MemoryIO


def _daily(per_day: float, start: date, end: date) -> dict[str, float]:
    out, d = {}, start
    while d <= end:
        out[d.isoformat()] = per_day
        d += timedelta(days=1)
    return out


NOV = lambda per_day, end_day=19: _daily(per_day, date(2026, 10, 25), date(2026, 11, end_day))   # noqa: E731


def test_assess_uses_completed_days_for_projection():
    today = date(2026, 11, 20)
    a = CG.assess(NOV(40.0), today, 2000.0)
    assert a["projection"] == 1200.0 and a["recent_avg"] == 40.0 and a["missing_days"] == []      # 40 × 30 日
    assert a["month_to_date"] == 40.0 * 19 and not a["over"]
    assert CG.assess(NOV(60.0), today, 2000.0)["over"]                                               # 60 × 30 = 1,800 ≥ 1,600


def test_assess_threshold_is_inclusive():
    a = CG.assess(_daily(1600 / 30, date(2026, 10, 25), date(2026, 11, 19)), date(2026, 11, 20), 2000.0)
    assert a["projection"] == pytest.approx(1600.0, abs=0.2) and a["over"]


def test_assess_over_by_month_to_date_even_if_recent_is_low():
    d = {**NOV(0.0, 11), "2026-11-01": 1700.0}
    a = CG.assess(d, date(2026, 11, 12), 2000.0)
    assert a["over"] and not a["over_budget"]


def test_assess_early_month_uses_previous_month_days():
    a = CG.assess(_daily(100.0, date(2026, 11, 20), date(2026, 12, 1)), date(2026, 12, 2), 2000.0)
    # 12/02: 平均は 11/28〜11/30（前の月）。今月の残りは 12/01〜12/31 の 31 日
    assert a["recent_avg"] == 100.0 and a["projection"] == 3100.0 and a["over"]
    a = CG.assess(_daily(100.0, date(2027, 2, 1), date(2027, 2, 27)), date(2027, 2, 28), 2000.0)       # 平年の 2 月末
    assert a["projection"] == pytest.approx(100.0 * 28, abs=0.1)
    a = CG.assess(_daily(100.0, date(2028, 2, 1), date(2028, 2, 28)), date(2028, 2, 29), 2000.0)       # うるう年
    assert a["projection"] == pytest.approx(100.0 * 29, abs=0.1)


def test_assess_reports_missing_days_instead_of_treating_as_zero():
    d = NOV(100.0, 15)                                   # 11/16 以降の課金データが無い（エクスポートの遅れ・停止）
    a = CG.assess(d, date(2026, 11, 20), 2000.0)
    assert a["missing_days"] == ["2026-11-16", "2026-11-17", "2026-11-18"]


def test_pause_targets_keeps_cost_report_and_paused():
    jobs = [{"name": "p/l/jobs/keiba-g-hourly", "state": "ENABLED"}, {"name": "p/l/jobs/keiba-g-0730-shadow", "state": "ENABLED"},
            {"name": "p/l/jobs/keiba-g-results", "state": "PAUSED"}, {"name": "p/l/jobs/smoke-auto-x", "state": "ENABLED"},
            {"name": "p/l/jobs/keiba-g-0730-shadow2", "state": "PAUSED"}]
    assert CG.pause_targets(jobs) == ["keiba-g-hourly"]
    assert CG.paused_groups(jobs + [{"name": "p/l/jobs/keiba-g-0730-shadow", "state": "PAUSED"}]) == ["keiba-g-0730-shadow2", "keiba-g-results"]


class _Resp:
    def __init__(self, data=None, code=200):
        self._d, self.status_code, self.text = data or {}, code, "denied"

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, jobs, pages=None, post_code=200, get_code=200):
        self.jobs, self.pages, self.posts, self.post_code, self.get_code = jobs, pages, [], post_code, get_code

    def get(self, url, params=None, timeout=None):
        if self.pages:
            tok = (params or {}).get("pageToken") or "0"
            jobs, nxt = self.pages[tok]
            return _Resp({"jobs": jobs, **({"nextPageToken": nxt} if nxt else {})}, self.get_code)
        return _Resp({"jobs": self.jobs}, self.get_code)

    def post(self, url, timeout=None):
        self.posts.append(url.rsplit("/", 1)[-1])
        return _Resp({"state": "PAUSED"}, self.post_code)


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(CG, "budget_jpy", lambda: 2000.0)
    monkeypatch.setattr(CG, "project_id", lambda: "proj")


def test_run_pauses_when_over_and_records():
    sess = _Session([{"name": "p/l/jobs/keiba-g-hourly", "state": "ENABLED"}, {"name": "p/l/jobs/keiba-g-0730-shadow", "state": "ENABLED"}])
    io, sent = MemoryIO(), []
    rec = CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(100.0), session=sess, io=io, notify=sent.append, env={})
    assert rec["over"] and rec["applied"] and sess.posts == ["keiba-g-hourly:pause"]
    assert len(sent) == 1 and "1 件止めました" in sent[0]
    assert json.loads(io.get_bytes("ops/cost_guard/20261120.json"))["paused"] == {"keiba-g-hourly": "PAUSED"}


def test_run_dry_run_and_under_budget_do_nothing():
    sess, io, sent = _Session([{"name": "p/l/jobs/keiba-g-hourly", "state": "ENABLED"}]), MemoryIO(), []
    rec = CG.run(apply=False, today=date(2026, 11, 20), daily=NOV(100.0), session=sess, io=io, notify=sent.append, env={})
    assert rec["over"] and not rec["applied"] and sess.posts == [] and sent == [] and io.list("ops/") == []
    rec = CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(20.0), session=sess, io=io, notify=sent.append, env={})
    assert not rec["over"] and sess.posts == [] and sent == []


def test_run_notifies_once_a_day_even_when_already_paused():
    """全部止まっていても、上限に近いなら 1 日 1 回は知らせる（止められない課金源で増えていても気づけるように）。"""
    sess, io, sent = _Session([{"name": "p/l/jobs/keiba-g-hourly", "state": "PAUSED"}]), MemoryIO(), []
    for _ in range(2):
        CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(100.0), session=sess, io=io, notify=sent.append, env={})
    assert sess.posts == [] and len(sent) == 1 and "止まっています" in sent[0]
    CG.run(apply=True, today=date(2026, 11, 21), daily=NOV(100.0, 20), session=sess, io=io, notify=sent.append, env={})
    assert len(sent) == 2


def test_run_reports_failed_pause_and_missing_targets():
    sess, sent = _Session([{"name": "p/l/jobs/keiba-g-hourly", "state": "ENABLED"}], post_code=403), []
    rec = CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(100.0), session=sess, io=MemoryIO(), notify=sent.append, env={})
    assert rec["paused"]["keiba-g-hourly"].startswith("error 403") and "止められませんでした" in sent[0]
    sent.clear()
    CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(100.0), session=_Session([]), io=MemoryIO(), notify=sent.append, env={})
    assert "見つかりません" in sent[0]


def test_run_pause_can_be_disabled_by_env():
    sess, sent = _Session([{"name": "p/l/jobs/keiba-g-hourly", "state": "ENABLED"}]), []
    CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(100.0), session=sess, io=MemoryIO(), notify=sent.append,
           env={"KEIBA_COST_GUARD_PAUSE": "0"})
    assert sess.posts == [] and "KEIBA_COST_GUARD_PAUSE=0" in sent[0]


def test_run_alerts_on_missing_billing_days_even_under_budget():
    sent = []
    CG.run(apply=True, today=date(2026, 11, 20), daily=NOV(10.0, 15), session=_Session([]), io=MemoryIO(), notify=sent.append, env={})
    assert len(sent) == 1 and "課金データが無い日" in sent[0]


def test_scheduler_jobs_follows_pages():
    sess = _Session(None, pages={"0": ([{"name": "a"}], "t1"), "t1": ([{"name": "b"}], "")})
    assert [j["name"] for j in CG.scheduler_jobs(sess, "p", "l")] == ["a", "b"]


def test_main_notifies_and_fails_when_guard_cannot_run(monkeypatch):
    sent = []
    monkeypatch.setattr(CG, "_notify", sent.append)
    monkeypatch.setattr("src.utils.project_env.load_project_dotenv", lambda *a, **k: None)

    def boom(**k):
        raise RuntimeError("BigQuery が応答しない")

    monkeypatch.setattr(CG, "run", boom)
    assert CG.main(["--apply"]) == 1 and "判定できませんでした" in sent[0]
