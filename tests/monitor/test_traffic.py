"""アクセス解析（VPS のログ集計 → 送信 → GCS の束 → 画面向けの読み出し）。VPS 側のスクリプトはリポジトリ外で動くので、パスで読み込む。"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitor import traffic as T
from src.ops.gcs_io import MemoryIO

ROOT = Path(__file__).resolve().parents[2]
VPS = ROOT / "scripts" / "ops" / "vps"
JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=JST)
CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile/15E148 Safari/604.1"
GOOGLEBOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"


def _load(name: str):
    if str(VPS) not in sys.path:
        sys.path.insert(0, str(VPS))
    spec = importlib.util.spec_from_file_location(name, VPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("vps_traffic_probe")
C = _load("traffic_client")


def line(ip="1.1.1.1", ts="28/Oct/2026:10:05:00 +0900", req="GET /race/202604040401 HTTP/1.1", status=200, ref="-", ua=CHROME):
    return f'{ip} - - [{ts}] "{req}" {status} 1234 "{ref}" "{ua}"'


def run_lines(lines):
    days, windows = {}, {}
    P.analyze(lines, days, windows, NOW.strftime("%Y%m%d"), NOW)
    return {k: v.out() for k, v in days.items()}, {k: len(v) for k, v in windows.items()}


# ───────────── VPS のログ集計 ─────────────

def test_page_views_exclude_assets_api_bots_and_prefetch():
    days, _ = run_lines([
        line(),
        line(req="GET /_next/static/chunks/a.js HTTP/1.1"),
        line(req="GET /api/v1/races HTTP/1.1"),
        line(req="GET /race/202604040402?_rsc=abc HTTP/1.1"),
        line(ua=GOOGLEBOT, req="GET /race/202604040403 HTTP/1.1"),
        line(req="GET /missing HTTP/1.1", status=404),
        line(req="POST /login HTTP/1.1", status=200),
        line(ip="127.0.0.1", req="GET /api/health HTTP/1.1"),
    ])
    d = days["20261028"]
    assert d["requests"] == 8
    assert d["page_views"] == 1 and d["assets"] == 1 and d["api"] == 1 and d["rsc"] == 1
    assert d["bots"] == [["Googlebot", 1]] and d["bot_kind"] == {"search": 1}
    assert d["err_pages"] == [["/missing", 1]]
    assert d["internal"] == 1


def test_visitors_are_distinct_ip_ua_and_windows_span_days():
    days, win = run_lines([
        line(ip="1.1.1.1"), line(ip="1.1.1.1", req="GET /races HTTP/1.1"),
        line(ip="2.2.2.2", ua=IPHONE),
        line(ip="1.1.1.1", ts="27/Oct/2026:09:00:00 +0900"),
    ])
    assert days["20261028"]["visitors"] == 2 and days["20261028"]["page_views"] == 3
    assert days["20261027"]["visitors"] == 1
    assert win["today"] == 2 and win["7d"] == 2  # 1.1.1.1 は 2 日にまたがっても 1 人
    assert days["20261028"]["devices"] == {"desktop": 2, "mobile": 1}


def test_timezone_is_converted_to_jst():
    days, _ = run_lines([line(ts="27/Oct/2026:20:30:00 +0000")])  # = 28 日 05:30 JST
    assert "20261028" in days and days["20261028"]["hours"][5] == 1


def test_patterns_group_ids_and_referrers_are_classified():
    days, _ = run_lines([
        line(req="GET /race/202604040401 HTTP/1.1", ref="https://www.google.com/"),
        line(ip="3.3.3.3", req="GET /race/202604040402 HTTP/1.1", ref="https://t.co/xyz"),
        line(ip="4.4.4.4", req="GET /horse/2019104251 HTTP/1.1", ref="https://app.megu-keiba.com/races"),
    ])
    d = days["20261028"]
    assert dict(d["patterns"]) == {"/race/:id": 2, "/horse/:id": 1}
    assert dict(d["referrers"]) == {"search:Google": 1, "social:X": 1}
    assert d["referrer_kind"] == {"search": 1, "social": 1, "direct": 1}


def test_seo_files_and_5xx_are_tracked():
    days, _ = run_lines([line(req="GET /robots.txt HTTP/1.1"), line(req="GET /sitemap.xml HTTP/1.1", status=404, ua=GOOGLEBOT),
                         line(req="GET /boom HTTP/1.1", status=502)])
    d = days["20261028"]
    assert d["seo_files"] == {"/robots.txt 2xx": 1, "/sitemap.xml 4xx": 1}
    assert d["page_views"] == 0 and d["err5xx"] == [["/boom", 1]]


def test_output_has_no_raw_ip_or_ua():
    days, _ = run_lines([line(ip="203.0.113.77")])
    assert "203.0.113.77" not in json.dumps(days) and "Chrome" not in json.dumps(days)


def test_run_reads_plain_and_gzip_and_reports_unreadable(tmp_path):
    (tmp_path / "access.log").write_text(line() + "\n", encoding="utf-8")
    with gzip.open(tmp_path / "access.log.2.gz", "wt", encoding="utf-8") as f:
        f.write(line(ts="26/Oct/2026:10:00:00 +0900") + "\n")
    res = P.run(str(tmp_path / "access.log*"), NOW)
    assert res["readable"] and set(res["days"]) == {"20261028", "20261026"} and res["lines"] == 2
    empty = P.run(str(tmp_path / "nothing*"), NOW)
    assert not empty["readable"] and "見つからない" in empty["reason"]


# ───────────── VPS の保存と送信の組み立て ─────────────

def test_client_merge_keeps_larger_day_and_builds_bundle(tmp_path):
    days, win = run_lines([line(ip=f"9.9.9.{i}") for i in range(5)] + [line(ts="27/Oct/2026:10:00:00 +0900", ip="8.8.8.8", req="GET /races HTTP/1.1")])
    store = str(tmp_path)
    assert C.merge(days, store) == 2
    assert C.merge({"20261028": {"requests": 1, "page_views": 1}}, store) == 0  # ログの端が欠けた小さい結果では上書きしない
    res = {"readable": True, "files": ["access.log"], "lines": 6, "windows": win}
    bundle = C.build_bundle(res, C.load_store(store), now=NOW.timestamp())
    assert bundle["readable"] and bundle["days"]["20261028"]["page_views"] == 5
    assert "pages" in bundle["days"]["20261028"]
    assert len(C.hb.encode(bundle)) < C.MAX_BODY


def test_client_slims_old_days(tmp_path):
    stored = {f"202610{d:02d}": {"requests": 9, "page_views": 3, "pages": [["/x", 3]], "visitors": 2} for d in range(1, 21)}
    out = C.build_bundle({"readable": True}, stored)["days"]
    assert "pages" not in out["20261001"] and "pages" in out["20261020"] and out["20261001"]["page_views"] == 3


def test_client_endpoint_falls_back_to_heartbeat_url():
    assert C.endpoint({"TRAFFIC_URL": "https://m/ops/traffic"}) == "https://m/ops/traffic"
    assert C.endpoint({"HEARTBEAT_URL": "https://m/ops/heartbeat"}) == "https://m/ops/traffic"
    assert C.endpoint({}) == ""


# ───────────── 監視ポータル側（GCS の束 → 画面） ─────────────

def _bundle(received_at=None, readable=True, **kw):
    days, win = run_lines([line(ip=f"9.9.9.{i}", ref="https://www.google.com/") for i in range(5)]
                          + [line(ts="27/Oct/2026:10:00:00 +0900", ip="8.8.8.8", req="GET /races HTTP/1.1"),
                             line(ua=GOOGLEBOT, req="GET /races HTTP/1.1", status=404)])
    return {"v": 1, "readable": readable, "reason": kw.get("reason"), "windows": win, "days": days if readable else {},
            "received_at": NOW.timestamp() - 120 if received_at is None else received_at}


def test_view_builds_kpi_ranges_and_series():
    d = T.view(_bundle(), NOW)
    assert d["status"] == "ok" and not d["stale"]
    assert d["kpi"]["today"]["page_views"] == 5 and d["kpi"]["yesterday"]["page_views"] == 1
    assert d["kpi"]["d7"]["page_views"] == 6 and d["kpi"]["d7"]["visitors"] == 6
    assert d["top_pages_7d"][0] == ["/race/202604040401", 5]
    assert len(d["series"]) == 90 and d["series"][-1]["date"] == "20261028" and d["series"][0]["page_views"] is None
    assert d["referrer_kind_7d"]["search"] == 5
    assert d["crawlers"][0]["name"] == "Googlebot" and d["crawlers"][0]["status"] == {"4xx": 1}
    s = T.summary(d)
    assert s["d7_pv"] == 6 and s["top"][0] == "/race/202604040401"


def test_view_states():
    assert T.view(None, NOW)["status"] == "not_collected"
    d = T.view(_bundle(readable=False, reason="ログを読む権限がありません"), NOW)
    assert d["status"] == "unreadable" and "権限" in d["reason"] and T.summary(d)["setup"].endswith(".sh")
    old = T.view(_bundle(received_at=NOW.timestamp() - 3 * 3600), NOW)
    assert old["status"] == "ok" and old["stale"]


def test_read_bundle_tolerates_missing_and_broken():
    io = MemoryIO()
    assert T.read_bundle(io) is None
    io.put_bytes(T.OBJECT, b"{broken")
    assert T.read_bundle(io) is None
    io.put_bytes(T.OBJECT, json.dumps({"v": 1}).encode())
    assert T.read_bundle(io) == {"v": 1}
