import json
from datetime import date

from src.pipeline.inference import prediction_results as pr


def _rec(rid, level, tb, ret):
    return {"race_id": rid, "included": True, "bets": {"total_bet": tb, "total_return": ret, "estimated": False, "level": level}}


def _write(tmp_path, d, recs):
    doc = {"date": d, "status": "final", "summary": pr.summarize(recs), "highlights": [], "races": recs}
    (tmp_path / f"{d}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_summarize_and_merge_split_bets_by_level():
    a = pr.summarize([_rec("R1", 4, 1000, 3000), _rec("R2", 4, 1000, 0), _rec("R3", 0, 500, 0)])
    assert a["bets"]["by_level"]["4"] == {"races": 2, "hit": 1, "invested": 2000, "returned": 3000}
    assert a["bets"]["by_level"]["0"]["races"] == 1
    b = pr.summarize([_rec("R4", 4, 1000, 1000), _rec("R5", 2, 2000, 1000)])
    m = pr.merge_summaries([a, b])
    assert m["bets"]["by_level"]["4"] == {"races": 3, "hit": 2, "invested": 3000, "returned": 4000}
    assert m["bets"]["by_level"]["2"]["invested"] == 2000


def test_bets_level_card_excludes_skipped_and_reports_unrated(tmp_path):
    _write(tmp_path, "20261017", [_rec("R1", 3, 1000, 1500), _rec("R2", 3, 1000, 0), _rec("R3", 0, 500, 0), _rec("R4", None, 800, 0)])
    # 旧データ（推奨度の記録が無い日）: by_level を持たない
    old = [{"race_id": "R9", "included": True, "bets": {"total_bet": 700, "total_return": 0, "estimated": False}}]
    doc = {"date": "20261010", "status": "final", "summary": pr.summarize(old), "highlights": [], "races": old}
    doc["summary"]["bets"].pop("by_level")
    (tmp_path / "20261010.json").write_text(json.dumps(doc), encoding="utf-8")
    out = pr.bets_level_card(date(2026, 10, 24), tmp_path)
    p = next(x for x in out["periods"] if x["key"] == "recent_5w")
    assert [x["level"] for x in p["levels"]] == [4, 3, 2, 1]               # 見送り(0)は行に出さない
    l3 = next(x for x in p["levels"] if x["level"] == 3)
    assert (l3["races"], l3["hit_races"], l3["invested"], l3["returned"], l3["roi"]) == (2, 1, 2000, 1500, 0.75)
    assert p["total"]["invested"] == 2000 and p["skipped_races"] == 1
    assert p["unrated_races"] == 2                                           # 旧データ 1R + 推奨度なし 1R


def test_portfolio_level_matches_ui_thresholds(monkeypatch):
    class C:
        def __init__(self, ev): self.bet_type, self.ev, self.bet_amount = "tansho", ev, 100
    monkeypatch.setattr("src.utils.ev_calibration.calibrate_portfolio", lambda cs: None)
    assert pr.portfolio_level([C(1.2)], 1.2) == (4, 1.2)
    assert pr.portfolio_level([C(0.9)], 0.9)[0] == 0
    assert pr.portfolio_level([], None) == (0, None)


def test_add_levels_only_adds_level_and_keeps_numbers(tmp_path, monkeypatch):
    recs = [_rec("R1", None, 1000, 3000), _rec("R2", None, 1000, 0), _rec("R3", 2, 500, 0)]
    for r in recs:
        r["bets"].pop("level")
    recs[2]["bets"]["level"] = 2
    recs[0]["marks"] = {"honmei": []}
    _write(tmp_path, "20261017", recs)
    before = json.loads((tmp_path / "20261017.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(pr, "load_predictions", lambda st, rid: {"predictions": [{"win_prob": 0.2}]})
    monkeypatch.setattr(pr, "load_race_result", lambda st, rid: {"finish": {}})
    new = {"R1": {"status": "ok", "total_bet": 1000, "level": 4, "exp_roi": 1.2}, "R2": {"status": "ok", "total_bet": 1200, "level": 1, "exp_roi": 0.98}}
    monkeypatch.setattr(pr, "simulate_bets", lambda entries, st, rid, res, cap: new[rid])
    out = pr.add_levels("20261017", None, tmp_path)
    assert out == {"races": 3, "added": 1, "already": 1, "mismatch": 1, "failed": 0}      # R2 は買い目が変わっている → 付けない
    after = json.loads((tmp_path / "20261017.json").read_text(encoding="utf-8"))
    assert after["races"][0]["bets"]["level"] == 4 and "level" not in after["races"][1]["bets"]
    assert after["summary"]["bets"]["by_level"]["4"]["returned"] == 3000
    assert after["summary"]["bets"]["invested"] == before["summary"]["bets"]["invested"]
    assert after["summary"]["marks"] == before["summary"]["marks"] and after["status"] == before["status"]
