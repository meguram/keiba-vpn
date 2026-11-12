"""Slack 通知（slack_notify / daily_results_notify / scrape_status_notify）。送信はすべてモックで、実際には送らない。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.pipeline.inference import prediction_results as PR
from src.scripts.data import daily_results_notify as DN
from src.scripts.data import scrape_status_notify as SS
from src.utils import slack_notify as SN

JST = timezone(timedelta(hours=9))
SECRET = "https://hooks.slack.com/services/TTTT/BBBB/SECRETSECRET"


class FakeResp:
    def __init__(self, status: int, headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}


def poster(*statuses: int, headers: list[dict] | None = None):
    calls: list[dict] = []
    seq = list(statuses)

    def post(url, json=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "json": json})
        i = len(calls) - 1
        st = seq[min(i, len(seq) - 1)]
        if st == -1:
            raise ConnectionError(f"cannot reach {url}")
        return FakeResp(st, (headers or [{}])[min(i, len(headers or [{}]) - 1)])

    post.calls = calls  # type: ignore[attr-defined]
    return post


class FormatTest(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(SN.fmt_yen(12300), "¥12,300")
        self.assertEqual(SN.fmt_yen(-500, signed=True), "-¥500")
        self.assertEqual(SN.fmt_yen(1200, signed=True), "+¥1,200")
        self.assertEqual(SN.fmt_yen(None), "—")
        self.assertEqual(SN.fmt_pct(0.1234), "12.3%")
        self.assertEqual(SN.fmt_pct(None), "—")
        self.assertEqual(SN.fmt_ratio(3, 8), "3/8 (37.5%)")
        self.assertEqual(SN.fmt_ratio(0, 0), "0/0")
        self.assertEqual(SN.fmt_delta_pt(0.052), "▲5.2pt")
        self.assertEqual(SN.fmt_delta_pt(-0.01), "▼1.0pt")
        self.assertEqual(SN.fmt_delta_pt(None), "")
        self.assertEqual(SN.jp_weekday("20261018"), "日")

    def test_escape_and_clip(self):
        self.assertEqual(SN.esc("a<b>&c"), "a&lt;b&gt;&amp;c")
        self.assertEqual(len(SN.clip("あ" * 50, 10)), 10)

    def test_code_table_aligns_wide_chars(self):
        t = SN.code_table(["券種", "購入"], [["単勝", "1,200"], ["3連複", "4,100"]])
        lines = t.strip("`").splitlines()
        self.assertTrue(t.startswith("```") and t.endswith("```"))
        self.assertEqual(len(lines), 3)

    def test_block_limit_truncates(self):
        blocks = [SN.section(f"b{i}") for i in range(80)]
        p = SN.build_message("t", blocks)
        self.assertLessEqual(len(p["attachments"][0]["blocks"]), SN.MAX_BLOCKS)
        self.assertIn("省略", json.dumps(p, ensure_ascii=False))

    def test_fields_limit(self):
        f = SN.fields([(f"k{i}", "v") for i in range(15)])
        self.assertEqual(len(f["fields"]), SN.MAX_FIELDS)


class SendTest(unittest.TestCase):
    ENV = {"SLACK_WEBHOOK_URL": SECRET}

    def test_routing_and_fallback(self):
        env = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/a", "SLACK_WEBHOOK_URL_ALERT": "https://hooks.slack.com/b"}
        self.assertEqual(SN.webhook_url("alert", env), "https://hooks.slack.com/b")
        self.assertEqual(SN.webhook_url("results", env), "https://hooks.slack.com/a")
        self.assertEqual(SN.webhook_url("default", {}), "")

    def test_not_configured(self):
        r = SN.send_payload({"text": "x"}, env={}, post=poster(200))
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "webhook_not_configured")
        self.assertEqual(r["attempts"], 0)

    def test_success_first_try(self):
        p = poster(200)
        r = SN.send_payload({"text": "x"}, env=self.ENV, post=p)
        self.assertTrue(r["ok"])
        self.assertEqual(len(p.calls), 1)

    def test_429_follows_retry_after_then_succeeds(self):
        waits: list[float] = []
        p = poster(429, 200, headers=[{"Retry-After": "2"}, {}])
        r = SN.send_payload({"text": "x"}, env=self.ENV, post=p, sleep=waits.append)
        self.assertTrue(r["ok"])
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(waits, [2.0])

    def test_5xx_retries_then_gives_up(self):
        waits: list[float] = []
        p = poster(500)
        r = SN.send_payload({"text": "x"}, env=self.ENV, post=p, sleep=waits.append, retries=3)
        self.assertFalse(r["ok"])
        self.assertEqual(len(p.calls), 3)
        self.assertEqual(len(waits), 2)

    def test_4xx_not_retried(self):
        p = poster(404)
        r = SN.send_payload({"text": "x"}, env=self.ENV, post=p, sleep=lambda s: None)
        self.assertFalse(r["ok"])
        self.assertEqual(len(p.calls), 1)

    def test_exception_does_not_leak_secret(self):
        p = poster(-1)
        r = SN.send_payload({"text": "x"}, env=self.ENV, post=p, sleep=lambda s: None, retries=2)
        self.assertFalse(r["ok"])
        self.assertNotIn("SECRETSECRET", json.dumps(r))
        self.assertIn("<webhook>", r["error"])

    def test_redact(self):
        self.assertNotIn("SECRETSECRET", SN.redact(f"error at {SECRET} here", self.ENV))

    def test_send_test_is_minimal(self):
        p = poster(200)
        r = SN.send_test(env=self.ENV, post=p)
        self.assertTrue(r["ok"])
        self.assertIn("接続テスト", json.dumps(p.calls[0]["json"], ensure_ascii=False))

    def test_preview_html_renders(self):
        payload = SN.build_message("t", [SN.section("*x* a<b"), SN.fields([("k", "v")]), SN.context(["c"])])
        h = SN.render_preview_html(payload)
        self.assertIn("<b>x</b>", h)
        self.assertIn("grid", h)


def make_doc(date="20261018", status="final", invested=10000, returned=13000, no_pred=0, no_result=0, kesi=None) -> dict:
    s = PR.summarize([])
    s.update(races=12, races_settled=12 - no_pred, races_no_prediction=no_pred, races_no_result=no_result)
    s["top1"].update(n=12, f1=3, f2=3, f3=2, f4p=4)
    s["marks"]["honmei"].update(n=12, f1=4, f2=3, f3=2, f4p=3)
    s["marks"]["pair"].update(n=12, f1=2, f2=3, f3=1, f4p=6)
    if kesi:
        s["marks"]["kesi"].update(kesi)
    s["bets"].update(invested=invested, returned=returned, races_bought=12 if invested else 0, races_hit=5 if invested else 0)
    if invested:
        s["bets"]["by_type"] = {"tansho": {"invested": 2000, "returned": 3000, "n": 6, "hit": 2},
                                "sanrenpuku": {"invested": 8000, "returned": 10000, "n": 30, "hit": 3}}
    s["logloss"] = {"n": 12, "sum": 28.8}
    return {"date": date, "status": status, "summary": s, "models": ["v4"], "prediction_sources": {"gcs": 12},
            "highlights": [{"kind": "big_miss", "race_id": "x", "value": 0.5, "text": "東京1R ◎テスト（勝率50%）は8着"}], "races": [], "notes": []}


class DailyResultsTest(unittest.TestCase):
    def test_check_ready(self):
        self.assertEqual(DN.check_ready(None), (False, "no_doc"))
        self.assertEqual(DN.check_ready(make_doc(status="provisional")), (False, "not_final"))
        self.assertEqual(DN.check_ready(make_doc(status="provisional"), allow_provisional=True), (True, "ok"))
        self.assertEqual(DN.check_ready(make_doc(no_result=2)), (False, "has_no_result"))
        d = make_doc()
        d["summary"]["races_settled"] = 0
        self.assertEqual(DN.check_ready(d), (False, "no_races"))

    def test_payload_numbers(self):
        p = DN.build_payload(make_doc(), None, kesi_target=0.05)
        txt = json.dumps(p, ensure_ascii=False)
        self.assertIn("130.0%", txt)            # 回収率
        self.assertIn("+¥3,000", txt)           # 損益
        self.assertIn("5/12R", txt)             # 的中レース
        self.assertIn("確定", txt)
        self.assertEqual(p["attachments"][0]["color"], SN.SEVERITY_COLOR["good"])   # 回収率100%以上は緑

    def test_payload_loss_day_and_provisional(self):
        p = DN.build_payload(make_doc(status="provisional", returned=4000, kesi={"n": 4, "f1": 0, "f2": 0, "f3": 1, "f4p": 3}), None, kesi_target=0.05)
        txt = json.dumps(p, ensure_ascii=False)
        self.assertIn("40.0%", txt)
        self.assertIn("暫定", txt)
        self.assertIn("消印", txt)
        self.assertEqual(p["attachments"][0]["color"], SN.SEVERITY_COLOR["normal"])

    def test_payload_no_bets_no_prediction_day(self):
        d = make_doc(invested=0, returned=0, no_pred=3)
        d["summary"]["marks"]["honmei"] = {"n": 0, "f1": 0, "f2": 0, "f3": 0, "f4p": 0}
        d["highlights"] = []
        p = DN.build_payload(d, None)
        txt = json.dumps(p, ensure_ascii=False)
        self.assertIn("購入対象なし", txt)
        self.assertIn("予測なし 3R", txt)

    def test_escape_in_highlights(self):
        d = make_doc()
        d["highlights"][0]["text"] = "<script>&"
        self.assertIn("&lt;script&gt;&amp;", json.dumps(DN.build_payload(d, None), ensure_ascii=False))

    def test_run_dry_run_writes_preview_and_does_not_send(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "res").mkdir()
            PR.write_day(make_doc(), t / "res")
            p = poster(200)
            r = DN.run("20261018", results_dir=t / "res", out_dir=t / "out", sent_dir=t / "sent", poster=p, env={"SLACK_WEBHOOK_URL": SECRET})
            self.assertEqual(r["code"], DN.EXIT_OK)
            self.assertEqual(p.calls, [])
            self.assertTrue((t / "out" / "results_20261018.json").exists())
            self.assertTrue((t / "out" / "results_20261018.html").exists())
            self.assertNotIn("SECRETSECRET", (t / "out" / "results_20261018.json").read_text(encoding="utf-8"))

    def test_run_send_dedupes_and_requires_flag(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "res").mkdir()
            PR.write_day(make_doc(), t / "res")
            env = {"SLACK_WEBHOOK_URL": SECRET}
            kw = dict(results_dir=t / "res", sent_dir=t / "sent", out_dir=None)
            self.assertEqual(DN.run("20261018", send=True, poster=poster(200), env=env, **kw)["code"], DN.EXIT_DISABLED)
            env["KEIBA_SLACK_NOTIFY"] = "1"
            p = poster(200)
            self.assertEqual(DN.run("20261018", send=True, poster=p, env=env, **kw)["code"], DN.EXIT_OK)
            self.assertEqual(len(p.calls), 1)
            r2 = DN.run("20261018", send=True, poster=p, env=env, **kw)
            self.assertEqual(r2["code"], DN.EXIT_SENT)
            self.assertEqual(len(p.calls), 1)
            self.assertEqual(DN.run("20261018", send=True, force=True, poster=p, env=env, **kw)["code"], DN.EXIT_OK)
            self.assertEqual(len(p.calls), 2)
            self.assertNotIn("SECRETSECRET", DN.sent_path("20261018", t / "sent").read_text(encoding="utf-8"))

    def test_failed_send_is_not_marked_sent(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "res").mkdir()
            PR.write_day(make_doc(), t / "res")
            env = {"SLACK_WEBHOOK_URL": SECRET, "KEIBA_SLACK_NOTIFY": "1"}
            import src.utils.slack_notify as mod
            orig = mod.time.sleep
            mod.time.sleep = lambda s: None
            try:
                r = DN.run("20261018", send=True, poster=poster(500), env=env, results_dir=t / "res", sent_dir=t / "sent")
            finally:
                mod.time.sleep = orig
            self.assertEqual(r["code"], DN.EXIT_FAILED)
            self.assertIsNone(DN.already_sent("20261018", t / "sent"))

    def test_not_ready_codes(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "res").mkdir()
            PR.write_day(make_doc(status="provisional"), t / "res")
            self.assertEqual(DN.run("20261018", results_dir=t / "res")["code"], DN.EXIT_NOT_READY)
            self.assertEqual(DN.run("20991231", results_dir=t / "res")["code"], DN.EXIT_NONE)


def job(i, name="job", group="g", result="success", cf=0, od=None, **kw):
    d = dict(id=i, name=name, group=group, state="waiting", result=result, started_at="2026-10-18T18:00:00+09:00", ended_at="2026-10-18T18:05:00+09:00",
             duration_sec=300, exit_code=0 if result == "success" else 1, next_run="2026-10-19T07:00:00+09:00", consecutive_failures=cf,
             overdue_min=od, running_hours=None, yields=False, cause=None, cause_id=None)
    d.update(kw)
    return d


class ScrapeStatusTest(unittest.TestCase):
    NOW = datetime(2026, 10, 18, 21, 30, tzinfo=JST)

    def test_prev_fire(self):
        n = datetime(2026, 10, 18, 12, 40, tzinfo=timezone.utc)
        self.assertEqual(SS.prev_fire("30 12 * * *", n), datetime(2026, 10, 18, 12, 30, tzinfo=timezone.utc))
        self.assertEqual(SS.prev_fire("*/20 6-11 * * 0,6", n), datetime(2026, 10, 18, 11, 40, tzinfo=timezone.utc))   # 日曜 11:40 UTC
        self.assertEqual(SS.prev_fire("0 1 * * 1", n), datetime(2026, 10, 12, 1, 0, tzinfo=timezone.utc))              # 直前の月曜

    def test_consecutive_failures_and_runs(self):
        self.assertEqual(SS.consecutive_failures([(1, 2, 0), (3, 4, 1), (5, 6, 2)]), 2)
        self.assertEqual(SS.consecutive_failures([(1, 2, 1), (3, 4, 0)]), 0)
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "j.tsv").write_text("1\t2\t0\t10\n3\t4\t1\t11\nbad\n", encoding="utf-8")
            self.assertEqual(SS.read_runs("j", Path(t)), [(1.0, 2.0, 0), (3.0, 4.0, 1)])
            self.assertEqual(SS.read_runs("none", Path(t)), [])

    def test_expectation(self):
        self.assertEqual(SS.expectation_of("raceday-eve"), "eve")
        self.assertEqual(SS.expectation_of("raceday-runner-2"), "raceday")
        self.assertEqual(SS.expectation_of("daily-race-lists"), "always")

    def test_detect_anomalies(self):
        jobs = [job("a", "A", cf=2), job("b", "B", cf=3), job("c", "C", od=30), job("d", "D", od=240), job("e", "E", running_hours=7, state="running")]
        an = SS.detect_anomalies(jobs, None, None, {})
        self.assertEqual(an["cron:a:fail"]["severity"], "warning")
        self.assertEqual(an["cron:b:fail"]["severity"], "critical")
        self.assertEqual(an["cron:c:overdue"]["severity"], "warning")
        self.assertEqual(an["cron:d:overdue"]["severity"], "critical")
        self.assertIn("cron:e:stuck", an)

    def test_detect_network_schema_coverage(self):
        q = {"stats": {}, "failed_reasons": {"http_403": 4, "http_429": 2}}
        schema = {"last24": 40, "by_day": {"a": 2, "b": 3, "c": 4, "d": 40}, "categories": {"x": 40}}
        misc = {"training_coverage": {"block": 0.2, "month": "2026-10", "floor": 0.9}, "health": {"overall": "fail", "findings": ["x"]}}
        an = SS.detect_anomalies([], q, schema, misc)
        self.assertIn("net:block", an)
        self.assertIn("schema:spike", an)
        self.assertEqual(an["coverage:training"]["severity"], "critical")
        self.assertIn("health:fail", an)
        self.assertEqual(SS.detect_anomalies([], {"stats": {}, "failed_reasons": {"http_403": 2}}, None, {}), {})
        small = {"last24": 5, "by_day": {"a": 1, "b": 1, "c": 1, "d": 5}, "categories": {}}
        self.assertNotIn("schema:spike", SS.detect_anomalies([], None, small, {}))

    def test_queue_waiting_is_not_anomaly(self):
        an = SS.detect_anomalies([job("q", yields=True, od=None)], {"stats": {"running": 1, "queued": 30}, "failed_reasons": {}}, None, {})
        self.assertEqual(an, {})

    def test_step_state_suppression(self):
        cur = {"k": {"severity": "warning", "title": "t", "detail": ""}}
        t0 = self.NOW
        n, r, st = SS.step_state({}, cur, t0)
        self.assertEqual([x["kind"] for x in n], ["new"])
        n, r, st = SS.step_state(st, cur, t0 + timedelta(minutes=15))
        self.assertEqual(n, [])
        n, r, st = SS.step_state(st, cur, t0 + timedelta(minutes=61))
        self.assertEqual([x["kind"] for x in n], ["repeat"])
        for i in range(2, 6):
            n, r, st = SS.step_state(st, cur, t0 + timedelta(minutes=61 * i))
        self.assertEqual(st["k"]["repeats"], SS.MAX_REPEATS)
        n, r, st = SS.step_state(st, cur, t0 + timedelta(minutes=61 * 8))
        self.assertEqual(n, [])
        n, r, st = SS.step_state(st, {}, t0 + timedelta(minutes=61 * 9))
        self.assertEqual([x["key"] for x in r], ["k"])
        self.assertEqual(st, {})
        n, r, st = SS.step_state(st, {}, t0 + timedelta(minutes=61 * 10))
        self.assertEqual((n, r), ([], []))

    def test_host_down_windows(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            self.assertEqual(SS.host_down_windows(t, self.NOW), [])                       # 記録が無ければ判定しない
            base = self.NOW.timestamp() - 10 * 3600
            starts = [base + i * 900 for i in range(4)] + [base + 8 * 3600 + i * 900 for i in range(8)]   # 0:45〜8:00 が空白
            (t / f"{SS.HEARTBEAT_JOB}.tsv").write_text("".join(f"{s}\t{s + 5}\t0\t1\n" for s in starts), encoding="utf-8")
            w = SS.host_down_windows(t, self.NOW)
            self.assertEqual(w, [(base + 3 * 900, base + 8 * 3600)])
            later = self.NOW + timedelta(hours=2)                                         # 最後の記録から今までの空白（再起動直後）
            self.assertEqual(SS.host_down_windows(t, later), w)                           # alerts 以外は数えない
            self.assertEqual(SS.host_down_windows(t, later, include_tail=True)[-1], (starts[-1], later.timestamp()))
            much_later = self.NOW + timedelta(hours=SS.HOST_TAIL_MAX_H + 1)               # 記録が止まっただけの長い空白は数えない
            self.assertEqual(SS.host_down_windows(t, much_later, include_tail=True), w)
            past = datetime.fromtimestamp(base + 3 * 3600, JST)                           # --now で過去を再現: それより後の記録は使わない
            self.assertEqual(SS.host_down_windows(t, past, include_tail=True), [(base + 3 * 900, past.timestamp())])

    def test_merge_down_windows(self):
        now = self.NOW.timestamp()
        m = SS.merge_down_windows([[now - 3600, now - 1800], ["bad"], [now - 20 * 86400, now - 19 * 86400]],
                                  [(now - 3600, now - 900), (now - 600, now - 60)], self.NOW)
        self.assertEqual(m, [(now - 3600, now - 900), (now - 600, now - 60)])

    @mock.patch.dict(os.environ, {"KEIBA_CRON_TZ": "UTC"})
    def test_collect_cron_marks_missed_during_downtime(self):
        now = datetime(2026, 11, 12, 0, 30, tzinfo=timezone.utc)                         # JST 9:30
        st = {"groups": [{"name": "g", "jobs": [
            {"id": "morning", "cron_exprs": ["0 22 * * *"], "state": "waiting", "last_run": {"started_at": "2026-11-10T22:00:00+00:00"}},
            {"id": "after", "cron_exprs": ["0 0 * * *"], "state": "waiting", "last_run": {"started_at": "2026-11-10T00:00:00+00:00"}},
        ]}]}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            up = [datetime(2026, 11, 11, 17, m, tzinfo=timezone.utc).timestamp() for m in (0, 15)] \
                + [datetime(2026, 11, 11, 23, 45, tzinfo=timezone.utc).timestamp(), datetime(2026, 11, 12, 0, 15, tzinfo=timezone.utc).timestamp()]
            (t / f"{SS.HEARTBEAT_JOB}.tsv").write_text("".join(f"{s}\t{s + 5}\t0\t1\n" for s in up), encoding="utf-8")
            jobs = {j["id"]: j for j in SS.collect_cron(now, status_fn=lambda: st, runs_dir=t)}
        self.assertIsNone(jobs["morning"]["overdue_min"])                                # 22:00 UTC はホスト停止中
        self.assertIsNotNone(jobs["morning"]["missed_down"])
        self.assertEqual(jobs["after"]["overdue_min"], 30)                               # 再起動後の 0:00 UTC は通常どおり未開始
        self.assertIsNone(jobs["after"]["missed_down"])
        an = SS.detect_anomalies(list(jobs.values()), None, None, {})
        self.assertNotIn("cron:morning:overdue", an)
        down = [k for k in an if k.startswith("host:down:")]
        self.assertEqual(len(down), 1)
        self.assertIs(an[down[0]]["repeat"], False)
        self.assertEqual(an[down[0]]["severity"], "warning")

    @mock.patch.dict(os.environ, {"KEIBA_CRON_TZ": "UTC"})
    def test_run_alerts_downtime_flow_and_saved_windows(self):
        """alerts: 再起動直後は末尾の空白で判定（保存しない）→ 次の回は閉じた停止期間を状態に保存 → summary も状態から判定。"""
        u = lambda d, h, m: datetime(2026, 11, d, h, m, tzinfo=timezone.utc)
        st = {"groups": [{"name": "g", "jobs": [
            {"id": "morning", "name": "朝のジョブ", "cron_exprs": ["0 22 * * *"], "state": "waiting",
             "last_run": {"started_at": "2026-11-10T22:00:00+00:00"}}]}]}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            hb = t / f"{SS.HEARTBEAT_JOB}.tsv"
            hb.write_text("".join(f"{x}\t{x + 5}\t0\t1\n" for x in (u(11, 17, 0).timestamp(), u(11, 17, 15).timestamp())), encoding="utf-8")
            env = {"SLACK_WEBHOOK_URL": SECRET, "KEIBA_SLACK_NOTIFY": "1"}
            inp = {"runs_dir": str(t), "status_fn": lambda: st, "queue": None, "schema": None, "misc": {}, "meta": str(t),
                   "raceday": lambda d: True, "maintenance": False}
            sp = t / "state.json"
            p = poster(200)
            r = SS.run("alerts", u(12, 0, 0), send=True, state_path=sp, inputs=inp, poster=p, env=env)
            self.assertEqual([k for k in r["to_notify"] if k.startswith("host:down:")], r["to_notify"])
            self.assertEqual(len(r["to_notify"]), 1)
            saved = json.loads(sp.read_text(encoding="utf-8"))
            self.assertNotIn("down", saved.get("_meta") or {})                    # 末尾の空白は保存しない
            with hb.open("a", encoding="utf-8") as f:                               # 再起動後の最初の回が記録された
                f.write(f"{u(12, 0, 0).timestamp()}\t{u(12, 0, 0).timestamp() + 5}\t0\t1\n")
            r = SS.run("alerts", u(12, 0, 15), send=True, state_path=sp, inputs=inp, poster=p, env=env)
            self.assertEqual((r["to_notify"], r["resolved"]), ([], []))
            self.assertEqual(len(p.calls), 1)                                       # 1 通だけ
            saved = json.loads(sp.read_text(encoding="utf-8"))
            self.assertEqual(saved["_meta"]["down"], [[u(11, 17, 15).timestamp(), u(12, 0, 0).timestamp()]])
            hb.write_text(f"{u(12, 0, 0).timestamp()}\t{u(12, 0, 0).timestamp() + 5}\t0\t1\n", encoding="utf-8")   # 切り詰めで停止前が消えた
            r = SS.run("summary", u(12, 0, 20), send=False, state_path=sp, inputs=inp, env=env)
            self.assertEqual([k for k in r["anomalies"] if k.startswith("host:down:")], [k for k in saved if k.startswith("host:down:")])
            self.assertNotIn("cron:morning:overdue", r["anomalies"])               # 状態に残した期間で判定できる

    def test_step_state_no_repeat_for_downtime_and_batched_repeats(self):
        t0 = self.NOW
        cur = {"host:down:x": {"severity": "warning", "title": "停止", "repeat": False}, "a": {"severity": "warning", "title": "A"}}
        n, _, st = SS.step_state({}, cur, t0)
        self.assertEqual(sorted(x["key"] for x in n), ["a", "host:down:x"])
        self.assertNotIn("repeat", n[0])
        n, _, st = SS.step_state(st, cur, t0 + timedelta(minutes=61))
        self.assertEqual([x["key"] for x in n], ["a"])                                  # repeat=False は再通知しない
        # 30 分後に新しい異常 b。再通知は全体で 1 時間に 1 回なので、b の期限が来ても次の窓まで待って a とまとめる
        cur2 = {**cur, "b": {"severity": "warning", "title": "B"}}
        n, _, st = SS.step_state(st, cur2, t0 + timedelta(minutes=91))
        self.assertEqual([x["kind"] for x in n], ["new"])
        n, _, st = SS.step_state(st, cur2, t0 + timedelta(minutes=130))
        self.assertEqual([x["key"] for x in n], ["a"])                                  # b はまだ 39 分
        n, _, st = SS.step_state(st, cur2, t0 + timedelta(minutes=160))
        self.assertEqual(n, [])                                                         # b は期限が来たが全体の窓は閉じている
        n, _, st = SS.step_state(st, cur2, t0 + timedelta(minutes=191))
        self.assertEqual(sorted(x["key"] for x in n), ["a", "b"])                        # 1 通にまとめて再通知
        n, r, st = SS.step_state(st, {}, t0 + timedelta(minutes=200))
        self.assertEqual(sorted(x["title"] for x in r), ["A", "B"])                      # 復旧は題名で出す。停止のまとめは復旧も黙る
        self.assertEqual(st, {})

    def test_step_state_escalation_notifies_immediately(self):
        t0 = self.NOW
        _, _, st = SS.step_state({}, {"k": {"severity": "warning", "title": "t"}}, t0)
        n, _, st = SS.step_state(st, {"k": {"severity": "critical", "title": "t"}}, t0 + timedelta(minutes=5))
        self.assertEqual([x["kind"] for x in n], ["escalated"])

    def test_signal(self):
        self.assertEqual(SS.signal_of([job("a")], {}, None, None), "green")
        self.assertEqual(SS.signal_of([job("a", result="failed", cf=1)], {}, None, None), "yellow")
        self.assertEqual(SS.signal_of([job("a", od=60)], {}, None, None), "red")
        self.assertEqual(SS.signal_of([job("a", cf=2)], {}, None, None), "red")
        self.assertEqual(SS.signal_of([job("a")], {"x": {"severity": "critical"}}, None, None), "red")
        self.assertEqual(SS.signal_of([job("a")], {}, {"last24": 3}, None), "yellow")

    def test_summary_and_coverage_payloads(self):
        jobs = [job("a", "前日取得", "開催日"), job("b", "調教の取り直し", "バックフィル", result="failed", cf=3, cause="HTTP 403")]
        misc = {"auto": {"raceday-eve": {"status": "ok", "races": 8, "error_count": 0}}, "training_coverage": {"block": 0.2, "month": "2026-10", "floor": 0.9},
                "health": {"overall": "warn", "counts": {"fail": 0, "warn": 1}, "complete": False, "reasons": ["範囲が短い"], "missing_lists": {"by_race": 12}, "findings": []}}
        an = SS.detect_anomalies(jobs, None, None, misc)
        p = SS.build_summary(self.NOW, jobs, {"stats": {"running": 0, "queued": 2}, "failed_reasons": {}}, {"last24": 3, "by_day": {}, "categories": {"x": 3}}, misc, an, race_day=True)
        txt = json.dumps(p, ensure_ascii=False)
        self.assertIn("開催日", txt)
        self.assertIn("要対応", txt)
        self.assertEqual(p["attachments"][0]["color"], SS.SIGNAL_COLOR["red"])
        self.assertIn("調教の被覆", txt)
        c = SS.build_coverage(self.NOW, misc, None)
        self.assertEqual(c["attachments"][0]["color"], SN.SEVERITY_COLOR["critical"])

    def test_alert_message_new_and_resolved(self):
        self.assertIsNone(SS.build_alert([], [], self.NOW))
        items = [{"key": "k", "severity": "critical", "title": "X が失敗", "detail": "d", "kind": "new"}]
        p = SS.build_alert(items, [{"key": "old", "since": 0}], self.NOW)
        txt = json.dumps(p, ensure_ascii=False)
        self.assertIn("緊急", txt)
        self.assertIn("復旧しました", txt)
        only_resolved = SS.build_alert([], [{"key": "old", "since": 0}], self.NOW)
        self.assertEqual(only_resolved["attachments"][0]["color"], SN.SEVERITY_COLOR["good"])

    def test_run_alerts_state_roundtrip_and_send(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            env = {"SLACK_WEBHOOK_URL": SECRET, "KEIBA_SLACK_NOTIFY": "1"}
            inp = {"jobs": [job("a", "A", cf=3)], "queue": None, "schema": None, "misc": {}, "meta": str(t), "raceday": lambda d: True}
            sp = t / "state.json"
            p = poster(200)
            r = SS.run("alerts", self.NOW, send=True, state_path=sp, inputs=inp, poster=p, env=env)
            self.assertEqual((r["code"], r["to_notify"]), (0, ["cron:a:fail"]))
            self.assertEqual(len(p.calls), 1)
            r = SS.run("alerts", self.NOW + timedelta(minutes=15), send=True, state_path=sp, inputs=inp, poster=p, env=env)
            self.assertEqual(r["to_notify"], [])
            self.assertEqual(len(p.calls), 1)                      # 同じ異常は黙る
            inp2 = dict(inp, jobs=[job("a", "A")])
            r = SS.run("alerts", self.NOW + timedelta(minutes=30), send=True, state_path=sp, inputs=inp2, poster=p, env=env)
            self.assertEqual(r["resolved"], ["cron:a:fail"])
            self.assertEqual(len(p.calls), 2)                      # 復旧の 1 通
            self.assertEqual(json.loads(sp.read_text(encoding="utf-8")), {})

    def test_failed_send_keeps_state_for_retry(self):
        import src.utils.slack_notify as mod
        orig = mod.time.sleep
        mod.time.sleep = lambda s: None
        try:
            with tempfile.TemporaryDirectory() as t:
                t = Path(t)
                env = {"SLACK_WEBHOOK_URL": SECRET, "KEIBA_SLACK_NOTIFY": "1"}
                inp = {"jobs": [job("a", "A", cf=3)], "queue": None, "schema": None, "misc": {}, "meta": str(t), "raceday": lambda d: True}
                sp = t / "state.json"
                r = SS.run("alerts", self.NOW, send=True, state_path=sp, inputs=inp, poster=poster(500), env=env)
                self.assertEqual(r["code"], 14)
                self.assertFalse(sp.exists())
        finally:
            mod.time.sleep = orig

    def test_dry_run_does_not_touch_state_or_send(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            inp = {"jobs": [job("a", "A", cf=3)], "queue": None, "schema": None, "misc": {}, "meta": str(t), "raceday": lambda d: True}
            p = poster(200)
            r = SS.run("alerts", self.NOW, send=False, state_path=t / "s.json", inputs=inp, poster=p, out_dir=t / "out", env={})
            self.assertEqual(r["code"], 0)
            self.assertEqual(p.calls, [])
            self.assertFalse((t / "s.json").exists())
            self.assertTrue(list((t / "out").glob("scrape_alerts_*.html")))

    def test_send_requires_flag(self):
        r = SS.run("summary", self.NOW, send=True, inputs={"jobs": [], "queue": None, "schema": None, "misc": {}, "raceday": lambda d: True}, env={})
        self.assertEqual(r["code"], 13)


if __name__ == "__main__":
    unittest.main()


def test_env_label_added_to_sent_payload():
    from src.utils import slack_notify as SN

    sent = []

    class R:
        status_code = 200
        headers = {}

    def post(url, json=None, timeout=None):
        sent.append(json)
        return R()

    base = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/X"}
    payload = SN.build_message("テスト", [SN.section("本文")], fallback="テスト")
    SN.send_payload(payload, post=post, env={**base, "KEIBA_ENV": "stg", "KEIBA_RUN_LOCATION": "host"})
    SN.send_payload(payload, post=post, env={**base, "KEIBA_ENV": "prod", "KEIBA_RUN_LOCATION": "gcp"})
    SN.send_payload(payload, post=post, env=dict(base))
    assert sent[0]["text"].startswith("【STG】") and "ホストPC" in str(sent[0]["attachments"][0]["blocks"][-1])
    assert sent[1]["text"].startswith("【PROD】") and "GCP" in str(sent[1]["attachments"][0]["blocks"][-1])
    assert sent[2]["text"] == "テスト" and len(sent[2]["attachments"][0]["blocks"]) == 1
    assert payload["text"] == "テスト" and len(payload["attachments"][0]["blocks"]) == 1  # 元のペイロードは変えない
    assert not any("_keiba_env_label" in p for p in sent)


def test_webhook_url_prefers_environment_specific():
    from src.utils import slack_notify as SN

    env = {
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/DEFAULT",
        "SLACK_WEBHOOK_URL_ALERT": "https://hooks.slack.com/services/T/B/ALERT",
        "SLACK_WEBHOOK_URL_PROD": "https://hooks.slack.com/services/T/B/PRODDEF",
        "SLACK_WEBHOOK_URL_ALERT_PROD": "https://hooks.slack.com/services/T/B/ALERTPROD",
    }
    assert SN.webhook_url("alert", {**env, "KEIBA_ENV": "prod"}).endswith("ALERTPROD")
    assert SN.webhook_url("results", {**env, "KEIBA_ENV": "prod"}).endswith("PRODDEF")
    assert SN.webhook_url("alert", {**env, "KEIBA_ENV": "stg"}).endswith("ALERT")      # stg 専用が無ければ従来どおり
    assert SN.webhook_url("results", {**env, "KEIBA_ENV": "stg"}).endswith("DEFAULT")
    assert SN.webhook_url("alert", env).endswith("ALERT")                                # 環境未設定は従来どおり
    assert "ALERTPROD" not in SN.redact("err https://hooks.slack.com/services/T/B/ALERTPROD", env)
