"""tests/scraper/manual/live_odds_scenarios.py の計測・見積りロジックを、ネット無し（偽 session）で検証する。"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parent / "manual" / "live_odds_scenarios.py"
spec = importlib.util.spec_from_file_location("live_odds_scenarios", SCRIPT)
L = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = L  # dataclass が参照する
spec.loader.exec_module(L)  # type: ignore[union-attr]


class FakeSession:
    def get(self, url, **kw):
        time.sleep(0.01 if "type=7" in url else 0.002)
        return SimpleNamespace(status_code=200, content=b"x" * 100)


class FakeRunner:
    def __init__(self):
        self.client = SimpleNamespace(_session=FakeSession())

    def scrape_race_card(self, rid, skip_existing=True):
        self.client._session.get("https://race.netkeiba.com/race/shutuba.html?race_id=" + rid)
        return {"entries": []}

    def scrape_odds(self, rid, skip_existing=True, phase=None):
        for t in (1, 2):
            self.client._session.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type={t}")
        time.sleep(0.005)  # 間隔制御の待機を模擬
        return {"entries": [1]}

    def scrape_pair_odds(self, rid, skip_existing=True, phase=None):
        for t in (4, 5, 6, 7):
            self.client._session.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type={t}")
        return {"umaren": [1]}


def test_label_maps_odds_types_and_html():
    assert L.Timing.label("https://x/api/api_get_jra_odds.html?race_id=1&type=7") == "オッズAPI/3連複"
    assert L.Timing.label("https://x/api/api_get_jra_odds.html?race_id=1&type=4") == "オッズAPI/馬連"
    assert L.Timing.label("https://race.netkeiba.com/race/shutuba.html?race_id=1") == "HTML/shutuba"


def test_timing_collects_per_request_and_step_and_wait_split():
    r = FakeRunner()
    tm = L.Timing()
    tm.install(r)
    rep = L.RaceReport("202605010101")
    summary = L.scenario_timing(rep, r, tm, repeat=3, gap=0)
    assert rep.results[-1].status == "PASS"
    assert summary["http"]["オッズAPI/3連複"]["n"] == 3
    assert summary["http"]["オッズAPI/3連複"]["med"] > summary["http"]["オッズAPI/馬連"]["med"]
    s = summary["steps"]
    assert s["2連系+3連複オッズ"]["requests"] == 4 and s["単複オッズ"]["requests"] == 2
    # 待機（意図的な sleep）は HTTP と分けて集計される
    assert s["単複オッズ"]["wait_med"] >= 0.004
    assert s["出馬表"]["wait_med"] < 0.004


def test_estimate_lead_scales_with_same_time_races_and_rounds_up():
    summary = {"steps": {"単複オッズ": {"p95": 20.0}, "2連系+3連複オッズ": {"p95": 40.0}}}
    one = L.estimate_lead(summary, same_time_races=1, predict_sec=30, margin_sec=60)
    three = L.estimate_lead(summary, same_time_races=3, predict_sec=30, margin_sec=60)
    assert one["min_lead_sec"] == 150.0 and one["min_lead_min_rounded_up_to_5"] == 5
    assert three["batch_sec"] == 180.0 and three["min_lead_sec"] == 270.0 and three["min_lead_min_rounded_up_to_5"] == 5
    assert L.estimate_lead({"steps": {}}, same_time_races=1, predict_sec=0, margin_sec=0) == {}


def test_save_timing_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    r = FakeRunner()
    tm = L.Timing()
    tm.install(r)
    summary = L.scenario_timing(L.RaceReport("202605010101"), r, tm, repeat=2, gap=0)
    path = L.save_timing("202605010101", summary, {}, tm)
    assert path.parent == tmp_path / "data" / "local" / "meta" / "scrape_timing" and path.read_text().startswith("{")
