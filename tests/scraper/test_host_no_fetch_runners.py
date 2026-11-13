"""取得を VPS に移したあとのホストの常駐ランナー（取得しない・後処理だけ）。キューに触らず、予測・後処理だけを呼ぶ。"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.scraper import auto_scrape as AS
from src.scraper import auto_scrape_queue as Q

JST = ZoneInfo("Asia/Tokyo")


def _no_queue(*a, **k):
    raise AssertionError("取得しないモードでキュー（取得）に触った")


@pytest.fixture
def host(monkeypatch):
    monkeypatch.delenv("KEIBA_FRESH_EXTRA_CATEGORIES", raising=False)   # ランナーが setdefault するので、テストの後に戻るよう先に消す
    monkeypatch.delenv("LOAD_CACHE_TTL_FRESH_SEC", raising=False)
    saved = {}
    monkeypatch.setattr(AS, "_load_race_calendar", lambda: {})
    monkeypatch.setattr(AS, "_is_race_day", lambda c, d: True)
    monkeypatch.setattr(AS, "_get_race_day_venues", lambda c, d: ["東京"])
    monkeypatch.setattr(AS, "_save_status", lambda k, v: saved.__setitem__(k, v))
    class St:
        saved_at = datetime.now(JST).timestamp() + 86400                     # VPS の T-15 の保存がある（どのレースの T-15 よりも後の時刻）

        def get_scraped_at_from_blob(self, cat, key):
            return self.saved_at

    monkeypatch.setattr(Q, "_storage", lambda: St())
    monkeypatch.setattr(Q, "_queue", _no_queue)
    monkeypatch.setattr(Q, "_ensure_race_list_date", _no_queue)
    monkeypatch.setattr(Q, "_kick_and_wait", _no_queue)
    monkeypatch.setattr(Q, "_host_no_fetch", lambda runner: True)
    monkeypatch.setattr(Q, "_sleep_until", lambda at: None)
    future = datetime.now(JST) + timedelta(hours=3)
    sched = [{"race_id": "202605050101", "post_time": future}, {"race_id": "202605050102", "post_time": future + timedelta(minutes=30)}]
    monkeypatch.setattr(Q, "_fetch_race_schedule_storage", lambda s, d: sched)
    return saved


def test_raceday_runner_only_triggers_predictions(host, monkeypatch):
    import src.scraper.raceday_pre_race_pipeline as P

    called = []
    monkeypatch.setattr(P, "refresh_jra_baba_live", lambda **k: True)
    monkeypatch.setattr(P, "trigger_pre_race_predict",
                        lambda r, bundle_result=None, allow_scrape=True: called.append((r["race_id"], bundle_result.jra_baba_refreshed, allow_scrape)))
    Q.task_raceday_runner()
    assert called == [("202605050101", True, False), ("202605050102", True, False)]      # 予測の中でも取り直さない
    assert host["raceday-runner"]["mode"] == "postprocess-only" and host["raceday-runner"]["predicted"] == 2


def test_raceday_runner_can_leave_jra_baba_to_vps(host, monkeypatch):
    import src.scraper.raceday_pre_race_pipeline as P

    monkeypatch.setenv("KEIBA_HOST_JRA_BABA_LIVE", "0")
    monkeypatch.setattr(P, "refresh_jra_baba_live", _no_queue)
    got = []
    monkeypatch.setattr(P, "trigger_pre_race_predict", lambda r, bundle_result=None, allow_scrape=True: got.append(bundle_result.jra_baba_refreshed))
    Q.task_raceday_runner()
    assert got == [False, False]


def test_result_runner_postprocesses_once_without_fetching(host, monkeypatch):
    import src.pipeline.inference.prediction_results as PR

    calls = []
    monkeypatch.setattr(AS, "_trigger_track_speed_for_date", lambda d: calls.append(("speed", d)))
    monkeypatch.setattr(PR, "refresh_for_dates", lambda ds, reason=None: calls.append(("results", tuple(ds))))
    Q.task_raceday_result_runner()
    assert [c[0] for c in calls] == ["speed", "results"]
    assert host["raceday-result-runner"]["mode"] == "postprocess-only"


def test_weight_runner_does_not_fetch(host, monkeypatch):
    import src.pipeline.inference.weight_triggered_predict as WTP
    import src.scraper.raceday_weight_watch as WW

    class Ctx:
        def warm(self):
            return {}

    monkeypatch.delenv("KEIBA_WEIGHT_WATCH_DATE", raising=False)
    monkeypatch.setattr(WTP, "InferenceContext", Ctx)
    monkeypatch.setattr(WW, "plan_windows", lambda s: [type("W", (), {"deadline": datetime.now(JST)})()])
    monkeypatch.setattr(WW, "next_wake", lambda *a: None)             # 1 周もしない（準備の段階で取得に触らないことを見る）
    Q.task_raceday_weight_runner()
    assert host["raceday-weight-runner"]["no_fetch"] is True


def test_runners_still_fetch_when_nothing_is_moved(monkeypatch, tmp_path):
    """担当表が空（今の本番）なら、ランナーは今までどおり取得する（_host_no_fetch は False）。"""
    p = tmp_path / "o.json"
    p.write_text('{"agent_tasks": []}')
    monkeypatch.setenv("KEIBA_FETCH_OWNER_FILE", str(p))
    monkeypatch.delenv("KEIBA_HOST_NO_FETCH", raising=False)
    monkeypatch.delenv("KEIBA_RUN_LOCATION", raising=False)
    assert not any(Q._host_no_fetch(r) for r in ("raceday-runner", "raceday-weight-runner", "raceday-result-runner"))


def test_grace_env_is_parsed_safely(monkeypatch):
    monkeypatch.setenv("KEIBA_HOST_NO_FETCH_GRACE_SEC", "abc")
    assert Q._no_fetch_grace_sec() == 180
    monkeypatch.setenv("KEIBA_HOST_NO_FETCH_GRACE_SEC", "60")
    assert Q._no_fetch_grace_sec() == 60


def test_raceday_runner_waits_for_vps_t15_save_and_records_missing(host, monkeypatch):
    import src.scraper.raceday_pre_race_pipeline as P

    class Never:
        def get_scraped_at_from_blob(self, cat, key):
            return 0.0

    monkeypatch.setattr(Q, "_storage", lambda: Never())
    monkeypatch.setattr(Q.time, "sleep", lambda s: None)
    monkeypatch.setattr(Q, "_wait_saved_after", lambda st, c, k, since, until: st.get_scraped_at_from_blob(c, k) >= since.timestamp())
    monkeypatch.setattr(P, "refresh_jra_baba_live", lambda **k: True)
    monkeypatch.setattr(P, "trigger_pre_race_predict", lambda r, bundle_result=None, allow_scrape=True: None)
    Q.task_raceday_runner()
    assert host["raceday-runner"]["t15_missing"] == 2 and host["raceday-runner"]["t15_saved"] == 0


def test_wait_saved_after_polls_until_saved_or_deadline(monkeypatch):
    seq = iter([0.0, 0.0, 2e9])
    monkeypatch.setattr(Q.time, "sleep", lambda s: None)

    class St:
        def get_scraped_at_from_blob(self, c, k):
            return next(seq)

    now = datetime.now(JST)
    assert Q._wait_saved_after(St(), "race_odds", "r", now, now + timedelta(hours=1)) is True
    assert Q._wait_saved_after(type("S", (), {"get_scraped_at_from_blob": lambda self, c, k: 0.0})(), "race_odds", "r", now,
                               now - timedelta(seconds=1)) is False


def test_fresh_extra_categories_only_in_this_process(monkeypatch):
    from src.scraper.storage import HybridStorage

    st = HybridStorage.__new__(HybridStorage)
    monkeypatch.delenv("KEIBA_FRESH_EXTRA_CATEGORIES", raising=False)
    monkeypatch.delenv("LOAD_CACHE_TTL_FRESH_SEC", raising=False)
    assert st._fresh_cap("race_odds") is None and st._fresh_cap("race_shutuba") is not None
    monkeypatch.setenv("KEIBA_FRESH_EXTRA_CATEGORIES", Q.FRESH_ODDS)
    assert st._fresh_cap("race_odds") == 300.0 and st._fresh_cap("race_result") is None
