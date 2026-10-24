from datetime import datetime, timedelta, timezone

from src.api import pipeline_status as ps

JST = timezone(timedelta(hours=9))
RID = "202604040301"


class St:
    def __init__(self, odds=True):
        self.odds = odds

    def load(self, cat, rid):
        if cat == "race_shutuba":
            return {"date": "2026-10-24", "start_time": "10:05"}
        if cat == "race_odds" and self.odds:
            return {"entries": [{"horse_number": 1}], "phase": "t60", "snapshot_at": "2026-10-24T09:05:00+09:00"}
        return None


def _resp(pred=True, phase="weighed"):
    return {
        "has_prediction": pred, "prediction_source": "shadow" if pred else None,
        "entries_status": {"phase": phase, "label": "馬体重発表済み", "n_entries": 16, "n_weight": 16, "race_date": "2026-10-24", "updated_at": "2026-10-24T09:00:00+09:00"},
    }


def _bets(monkeypatch, ok=True, n=3):
    ps._BETS_CACHE.clear()
    monkeypatch.setattr(ps, "_bets_state", lambda rid: {"ok": True, "n": n} if ok else {"ok": False, "reason": "オッズがありません"})
    monkeypatch.setattr(ps, "_prediction_time", lambda *a, **k: "2026-10-24T09:10:00+09:00")


def test_complete_when_all_five_stages_done(monkeypatch):
    _bets(monkeypatch)
    out = ps.pipeline_status(RID, _resp(), St(), now=datetime(2026, 10, 24, 9, 30, tzinfo=JST))
    assert out["complete"] is True and out["done"] == 5 and out["next"] is None
    assert [s["key"] for s in out["stages"]] == list(ps.STAGES)
    assert out["completed_at"] == "2026-10-24T09:10:00+09:00"


def test_no_recommendation_still_counts_as_complete(monkeypatch):
    _bets(monkeypatch, n=0)
    out = ps.pipeline_status(RID, _resp(), St(), now=datetime(2026, 10, 24, 9, 30, tzinfo=JST))
    assert out["complete"] is True
    assert "見送り" in out["stages"][4]["detail"]


def test_waiting_before_weights_points_to_window(monkeypatch):
    _bets(monkeypatch)
    resp = _resp(pred=False, phase="final")
    resp["entries_status"].update(n_weight=0)
    out = ps.pipeline_status(RID, resp, St(odds=False), now=datetime(2026, 10, 24, 8, 0, tzinfo=JST))
    assert out["complete"] is False and out["next"] == "weights"
    assert out["stages"][0]["state"] == "done"
    assert "09:00〜09:35" in out["stages"][1]["detail"]
    assert out["stages"][4]["state"] == "waiting"


def test_bets_failure_keeps_pipeline_incomplete(monkeypatch):
    _bets(monkeypatch, ok=False)
    out = ps.pipeline_status(RID, _resp(), St(odds=False), now=datetime(2026, 10, 24, 9, 30, tzinfo=JST))
    assert out["complete"] is False and out["next"] == "odds"
    assert out["stages"][4]["detail"] == "オッズがありません"


def test_attach_never_raises_and_skips_mock():
    assert ps.attach_pipeline_status({"_mock": True}, RID, None) == {"_mock": True}
    out = ps.attach_pipeline_status({"has_prediction": False}, RID, None)
    assert out["pipeline_status"]["complete"] is False and out["pipeline_status"]["next"] == "entries"
