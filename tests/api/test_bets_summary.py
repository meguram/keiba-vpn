import asyncio
import json

from src.api import prediction_results_api as api
from src.api import tier
from src.pipeline.inference import prediction_results as pr


def _doc(date, races):
    recs = [{"race_id": rid, "included": True, "bets": {"total_bet": tb, "total_return": ret, "estimated": False}} for rid, tb, ret in races]
    s = pr.summarize(recs)
    return {"date": date, "status": "final", "summary": s, "highlights": [], "races": recs}


def test_bets_summary_flags_hits_and_gives_day_and_week_roi(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "store_dir", lambda: tmp_path)
    monkeypatch.setattr(api, "_is_dev", lambda: False)
    api._CACHE.clear()
    (tmp_path / "20261010.json").write_text(json.dumps(_doc("20261010", [("R1", 1000, 0), ("R2", 1000, 3000)])), encoding="utf-8")      # 土曜
    (tmp_path / "20261011.json").write_text(json.dumps(_doc("20261011", [("R3", 2000, 1000)])), encoding="utf-8")                    # 日曜
    out = json.loads(asyncio.run(api.prediction_results_bets_summary("20261010")).body)
    assert out["races"]["R2"]["hit"] is True and out["races"]["R1"]["hit"] is False
    assert out["day"]["roi"] == 1.5 and out["day"]["races_hit"] == 1
    assert (out["week"]["from"], out["week"]["to"]) == ("20261005", "20261011")
    assert out["week"]["invested"] == 4000 and out["week"]["returned"] == 4000 and out["week"]["roi"] == 1.0


def test_bets_summary_is_member_only():
    assert tier.member_only_path("/api/prediction-results/bets-summary")
    assert not tier.member_only_path("/api/prediction-results/mark-card")
