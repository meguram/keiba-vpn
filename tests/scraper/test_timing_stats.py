"""実測時間の記録・集計（src/scraper/timing_stats.py）。ネット無し。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.scraper import timing_stats as T


@pytest.fixture()
def tdir(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_SCRAPE_TIMING_DIR", str(tmp_path))
    monkeypatch.delenv("KEIBA_SCRAPE_TIMING", raising=False)
    return tmp_path


class _Sess:
    def get(self, url, **kw):
        time.sleep(0.003)
        return SimpleNamespace(status_code=200, content=b"abc")


class _Runner:
    def __init__(self):
        self.s = _Sess()
        self.s.get = T.wrap_session_get(self.s.get)

    @T.timed_scrape
    def scrape_odds(self, race_id, skip_existing=True, phase=None):
        self.s.get("https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=1&type=7")
        return {"entries": [1]}

    @T.timed_scrape
    def scrape_cached(self, race_id):
        return {"x": 1}  # HTTP 無し = キャッシュヒット相当

    @T.timed_scrape
    def scrape_all(self, race_id):
        self.scrape_odds(race_id, phase="t60")
        return True


def _events(tdir):
    out = []
    for p in tdir.glob("*.jsonl"):
        out += [json.loads(x) for x in p.read_text().splitlines()]
    return out


def test_step_and_http_events_recorded(tdir):
    _Runner().scrape_odds("202605010101", phase="t15")
    ev = _events(tdir)
    step = next(e for e in ev if e["kind"] == "step")
    http = next(e for e in ev if e["kind"] == "http")
    assert step["method"] == "scrape_odds" and step["key"] == "202605010101" and step["phase"] == "t15"
    assert step["http_n"] == 1 and step["ok"] is True and step["nested"] is False
    assert step["http_sec"] > 0 and step["wait_sec"] >= 0
    assert http["label"] == "オッズAPI/3連複" and http["bytes"] == 3


def test_nested_calls_are_flagged(tdir):
    _Runner().scrape_all("202605010101")
    steps = [e for e in _events(tdir) if e["kind"] == "step"]
    by = {e["method"]: e for e in steps}
    assert by["scrape_all"]["nested"] is False and by["scrape_odds"]["nested"] is True
    assert by["scrape_all"]["http_n"] == 1  # 内側の HTTP も外側の累積に入る


def test_disable_and_steps_only(tdir, monkeypatch):
    monkeypatch.setenv("KEIBA_SCRAPE_TIMING", "steps")
    _Runner().scrape_odds("1")
    assert {e["kind"] for e in _events(tdir)} == {"step"}
    monkeypatch.setenv("KEIBA_SCRAPE_TIMING", "0")
    n = len(_events(tdir))
    _Runner().scrape_odds("1")
    assert len(_events(tdir)) == n


def test_recording_failure_never_breaks_scraping(monkeypatch):
    monkeypatch.setenv("KEIBA_SCRAPE_TIMING_DIR", "/proc/definitely/not/writable")
    assert _Runner().scrape_odds("1") == {"entries": [1]}


def test_summarize_excludes_cache_and_nested_by_default(tdir):
    r = _Runner()
    for _ in range(3):
        r.scrape_odds("1", phase="t60")
    r.scrape_cached("1")
    r.scrape_all("2")
    s = T.summarize(T.load_events(datetime.now() - timedelta(days=1)))
    assert set(s["steps"]) == {"scrape_odds[t60]", "scrape_all"}
    assert s["steps"]["scrape_odds[t60]"]["n"] == 3  # 入れ子の 1 件は除外
    assert "scrape_cached" not in s["steps"]
    s2 = T.summarize(T.load_events(datetime.now() - timedelta(days=1)), include_cache=True, include_nested=True)
    assert "scrape_cached" in s2["steps"] and s2["steps"]["scrape_odds[t60]"]["n"] == 4
    assert s["http"]["オッズAPI/3連複"]["n"] == 4  # HTTP は入れ子も含め全件（3 + scrape_all 内の 1）


def test_by_hour_groups_and_cli(tdir, capsys):
    _Runner().scrape_odds("1")
    s = T.summarize(T.load_events(datetime.now() - timedelta(days=1)), by="hour")
    assert any(k.startswith("scrape_odds ") and k.endswith("時台") for k in s["steps"])
    assert T.main(["report", "--days", "1"]) == 0
    assert "scrape_odds" in capsys.readouterr().out
    assert T.main(["report", "--days", "1", "--json"]) == 0
