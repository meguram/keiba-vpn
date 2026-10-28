"""API 課金ガード: 課金される起動時間 → 月額の見込み・夜間の起動の判定、取得失敗の扱い、通知の重複抑止。外部接続なし。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ops import api_cost_guard as G
from src.ops.gcs_io import MemoryIO

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=JST).timestamp()
PR = {"fx": 150.0, "vcpu_s": 1.8e-05, "gib_s": 2e-06, "free_vcpu": 240000.0, "free_gib": 450000.0}


def hours(per_hour, n=48):
    """now から遡って n 時間。per_hour(jst_hour) = その時間の秒数。"""
    out = []
    for i in range(n, 0, -1):
        end = NOW - (i - 1) * 3600
        out.append({"end": end, "sec": per_hour(datetime.fromtimestamp(end - 1800, JST).hour)})
    return out


def test_always_on_one_instance_is_far_over_budget_and_flagged_overnight():
    a = G.analyze(hours(lambda h: 3600), NOW, PR)
    assert a["level"] == "red" and a["avg_instances_24h"] == pytest.approx(1.0)
    assert 7000 < a["projected_month_yen"] < 9000          # 試算（1 台で月 約 7,800 円）と一致
    assert {x["key"] for x in a["alerts"]} == {"api_alive_overnight", "api_cost_projection"}


def test_two_instances_always_on_doubles_it():
    a = G.analyze(hours(lambda h: 7200), NOW, PR)
    assert a["avg_instances_24h"] == pytest.approx(2.0) and a["projected_month_yen"] > 15000


def test_daytime_only_use_is_green():
    a = G.analyze(hours(lambda h: 1200 if 9 <= h < 18 else 0), NOW, PR)
    assert a["level"] == "green" and a["alerts"] == [] and a["quiet_alive_hours"] == 0 and a["projected_month_yen"] < G.API_BUDGET_YEN


def test_one_night_blip_does_not_alert_but_two_do():
    one = G.analyze(hours(lambda h: 3600 if h == 2 else 0), NOW, PR)
    assert one["quiet_alive_hours"] == 1 and not any(x["key"] == "api_alive_overnight" for x in one["alerts"])
    two = G.analyze(hours(lambda h: 3600 if h in (2, 3) else 0), NOW, PR)
    assert any(x["key"] == "api_alive_overnight" and x["severity"] == "warning" for x in two["alerts"])


def test_free_tier_matters():
    day = hours(lambda h: 3600)
    assert G.analyze(day, NOW, PR)["projected_month_yen_no_free"] > G.analyze(day, NOW, PR)["projected_month_yen"]


def test_evaluate_handles_missing_project_permission_failure_and_empty():
    assert G.evaluate(NOW, project="", fetch=lambda *a: [])["status"] == "unavailable"
    def deny(*a): raise PermissionError("403")
    r = G.evaluate(NOW, project="p", fetch=deny)
    assert r["status"] == "unavailable" and "roles/monitoring.viewer" in r["reason"]
    def boom(*a): raise RuntimeError("x")
    assert "RuntimeError" in G.evaluate(NOW, project="p", fetch=boom)["reason"]
    assert G.evaluate(NOW, project="p", fetch=lambda *a: [])["reason"] == "指標のデータがありません"
    ok = G.evaluate(NOW, project="p", fetch=lambda *a: hours(lambda h: 0))
    assert ok["status"] == "ok" and ok["level"] == "green"


def test_run_check_notifies_once_then_recovers_and_stores_latest():
    io, sent = MemoryIO(), []
    notify = lambda title, lines, sev: sent.append((title, sev))  # noqa: E731
    bad = lambda *a: hours(lambda h: 3600)                      # noqa: E731
    r1 = G.run_check(io, notify, now=NOW, project="p", fetch=bad)
    assert r1["ok"] and set(r1["sent"]) == {"api_alive_overnight", "api_cost_projection"} and len(sent) == 2
    assert json.loads(io.get_bytes(G.LATEST))["status"] == "ok"
    assert G.run_check(io, notify, now=NOW + 1800, project="p", fetch=bad)["sent"] == [] and len(sent) == 2     # 同じ異常は再通知しない
    good = lambda *a: hours(lambda h: 0)                        # noqa: E731
    r3 = G.run_check(io, notify, now=NOW + 3600, project="p", fetch=good)
    assert set(r3["recovered"]) == {"api_alive_overnight", "api_cost_projection"} and sent[-1][1] == "normal"


def test_run_check_unavailable_does_not_notify_or_touch_state():
    io, sent = MemoryIO(), []
    def deny(*a): raise PermissionError("403")
    r = G.run_check(io, lambda *a: sent.append(a), now=NOW, project="p", fetch=deny)
    assert not r["ok"] and sent == [] and io.get_bytes(G.STATE) is None


def test_fetch_hourly_parses_response_and_maps_403():
    class R:
        def __init__(self, code, body): self.status_code, self._b = code, body
        def json(self): return self._b
    class Sess:
        def __init__(self, r): self.r, self.calls = r, []
        def get(self, url, params=None, timeout=None):
            self.calls.append((url, params)); return self.r
    body = {"timeSeries": [{"points": [{"interval": {"endTime": "2026-10-28T03:00:00Z"}, "value": {"doubleValue": 3600.5}},
                                       {"interval": {"endTime": "2026-10-28T02:00:00Z"}, "value": {"int64Value": "7200"}}]}]}
    s = Sess(R(200, body))
    out = G.fetch_hourly("p", "keiba-api", NOW, session=s)
    assert [h["sec"] for h in out] == [7200.0, 3600.5] and "billable_instance_time" in s.calls[0][1]["filter"]
    assert 'service_name="keiba-api"' in s.calls[0][1]["filter"]
    with pytest.raises(PermissionError):
        G.fetch_hourly("p", "keiba-api", NOW, session=Sess(R(403, {})))
