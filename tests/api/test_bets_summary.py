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
    monkeypatch.setattr(api, "is_member", lambda _request: True)
    monkeypatch.setattr(api, "is_developer", lambda _request: True)
    api._CACHE.clear()
    (tmp_path / "20261010.json").write_text(json.dumps(_doc("20261010", [("R1", 1000, 0), ("R2", 1000, 3000)])), encoding="utf-8")      # 土曜
    (tmp_path / "20261011.json").write_text(json.dumps(_doc("20261011", [("R3", 2000, 1000)])), encoding="utf-8")                    # 日曜
    out = json.loads(asyncio.run(api.prediction_results_bets_summary(object(), "20261010")).body)
    assert out["races"]["R2"]["hit"] is True and out["races"]["R1"]["hit"] is False
    assert out["day"]["roi"] == 1.5 and out["day"]["races_hit"] == 1
    assert (out["week"]["from"], out["week"]["to"]) == ("20261005", "20261011")
    assert out["week"]["invested"] == 4000 and out["week"]["returned"] == 4000 and out["week"]["roi"] == 1.0


def test_bets_summary_is_member_only():
    assert tier.member_only_path("/api/prediction-results/bets-summary")
    assert not tier.member_only_path("/api/prediction-results/mark-card")


def test_attach_bet_results_paid_and_full_views():
    from src.api.v1.delegates import attach_bet_results

    class St:
        def load(self, cat, rid):
            if cat != "race_result":
                return None
            return {"entries": [{"horse_number": n, "finish_position": p} for n, p in ((5, 1), (1, 2), (2, 3), (3, 4))],
                    "payoff": {"単勝": {"numbers": "5", "payout": "450"}, "馬連": {"numbers": "1-5", "payout": "1,230"}}}

    paid = {"view": "paid", "bets": [{"bet_type": "tansho", "combo": [5], "yen": 200}, {"bet_type": "umaren", "combo": [1, 2], "yen": 100}]}
    out = attach_bet_results(paid, "R1", St())
    assert out["bets"][0]["result"] == {"hit": True, "payout": 900, "estimated": False}
    assert out["bets"][1]["result"]["hit"] is False
    r = out["result"]
    assert (r["invested"], r["returned"], r["hits"], r["n_bets"], r["roi"]) == (300, 900, 1, 2, 3.0)
    assert [(f["pos"], f["no"]) for f in r["finish"]] == [(1, 5), (2, 1), (3, 2)]
    full = {"candidates": [{"bet_type": "umaren", "pair": [5, 1], "bet_amount": 100, "odds": 12.0}]}
    out2 = attach_bet_results(full, "R1", St())
    assert out2["candidates"][0]["result"]["payout"] == 1230
    none = attach_bet_results({"view": "paid", "bets": []}, "R2", type("S", (), {"load": lambda self, c, r: None})())
    assert "result" not in none


def test_bets_by_level_is_member_only():
    assert tier.member_only_path("/api/prediction-results/bets-by-level")
