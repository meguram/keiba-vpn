"""availability_watch: シグナル判定と first_seen_at の記録ロジック（ネット無し）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.scraper import availability_watch as W

JST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 14, 9, 0, tzinfo=JST)


def _card(drawn=True, weights=True):
    return {"entries": [{"horse_number": (i + 1) if drawn else 0, "weight": 480 if weights else 0} for i in range(3)]}


def test_signals_pre_race_stages():
    assert W.evaluate_signals({})["card.entries"] is False
    s = W.evaluate_signals({"card": _card(drawn=False, weights=False)})
    assert s["card.entries"] and not s["card.draw_fixed"] and not s["card.weights"]
    s = W.evaluate_signals({"card": _card(drawn=True, weights=False), "odds": {"entries": [{"win_odds": 0}]}})
    assert s["card.draw_fixed"] and not s["card.weights"] and not s["odds.win"]
    s = W.evaluate_signals({"card": _card(), "odds": {"entries": [{"win_odds": 3.5}]}, "pair_odds": {"umaren": [1], "sanrenpuku": []}})
    assert s["card.weights"] and s["odds.win"] and s["pair_odds.umaren"] and not s["pair_odds.sanrenpuku"]


def test_signals_barometer_and_comment():
    s = W.evaluate_signals({"barometer": {"entries": [], "not_published": True}})
    assert s["barometer.not_published"] and not s["barometer"]
    assert W.evaluate_signals({"barometer": {"entries": [{"horse_number": 1}]}})["barometer"]
    assert not W.evaluate_signals({"trainer_comment": {"entries": [{"comment": "  "}]}})["trainer_comment"]
    assert W.evaluate_signals({"trainer_comment": {"entries": [{"comment": "好調"}]}})["trainer_comment"]
    assert W.evaluate_signals({"result_lap": {"lap_times": [12.0]}})["result_lap"]


def test_first_seen_is_recorded_once_and_disappearance_is_reported():
    state: dict = {}
    ch = W.update_state(state, "r1", {"odds.win": False, "oikiri": False}, T0)
    assert ch == [] and state["races"]["r1"]["odds.win"]["first_seen_at"] is None
    t1 = T0 + timedelta(minutes=30)
    ch = W.update_state(state, "r1", {"odds.win": True, "oikiri": False}, t1)
    assert [(c["signal"], c["event"]) for c in ch] == [("odds.win", "first_seen")]
    first = state["races"]["r1"]["odds.win"]["first_seen_at"]
    ch = W.update_state(state, "r1", {"odds.win": True, "oikiri": True}, t1 + timedelta(minutes=30))
    assert [c["signal"] for c in ch] == ["oikiri"] and state["races"]["r1"]["odds.win"]["first_seen_at"] == first  # 上書きしない
    ch = W.update_state(state, "r1", {"odds.win": False, "oikiri": True}, t1 + timedelta(hours=1))
    assert [(c["signal"], c["event"]) for c in ch] == [("odds.win", "disappeared")]


def test_report_lists_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WATCH_DIR", tmp_path)
    st: dict = {}
    W.update_state(st, "r1", {"odds.win": True}, T0)
    W.save_state(st, [{"race_id": "r1", "signal": "odds.win", "event": "first_seen", "at": "x"}])
    assert (tmp_path / "observations.jsonl").read_text().count("first_seen") == 1
    assert "odds.win" in W.report(W.load_state()) and "2026-10-14T09:00:00+09:00" in W.report(W.load_state())


def test_refuses_without_dev_env_or_consent(monkeypatch, capsys):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    assert W.main(["--races", "x", "--yes"]) == 2
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert W.main(["--races", "x"]) == 2
