import json
from datetime import date

from src.pipeline.inference import prediction_results as pr


def _doc(d: str, marks: dict) -> dict:
    s = pr.summarize([])
    s["races_settled"] = 12
    for m, (f1, f2, f3, f4) in marks.items():
        s["marks"][m] = {"n": f1 + f2 + f3 + f4, "f1": f1, "f2": f2, "f3": f3, "f4p": f4}
    return {"date": d, "status": "final", "summary": s, "highlights": [], "races": []}


def _store(tmp_path, days: dict[str, dict]):
    for d, marks in days.items():
        (tmp_path / f"{d}.json").write_text(json.dumps(_doc(d, marks)), encoding="utf-8")
    return tmp_path


def test_marks_include_show_val_between_pair_and_star():
    assert pr.MARKS == ("honmei", "pair", "show_val", "star", "kesi")
    assert [s for _, s in pr.MARK_ORDER] == ["◎", "○", "▲", "★", "消"]


def test_periods_last_week_recent_5w_and_year(tmp_path):
    base = _store(tmp_path, {
        "20260110": {"honmei": (1, 0, 0, 0)},                      # 年にだけ入る
        "20260905": {"honmei": (2, 1, 0, 0)},                      # 直近5週間にだけ（今日 10/13 から 34 日前=9/9 より前なので入らない）
        "20260919": {"honmei": (1, 1, 1, 1), "show_val": (0, 1, 0, 3)},     # 直近5週間
        "20261004": {"honmei": (3, 0, 1, 2)},                      # 先週（10/5〜10/11 の前の週ではない。下で確認）
        "20261010": {"honmei": (4, 2, 1, 1), "kesi": (0, 0, 0, 9)},        # 先週（10/5〜10/11）
    })
    r = pr.mark_card(date(2026, 10, 13), base)                    # 火曜。先週=10/5(月)〜10/11(日)
    by = {p["key"]: p for p in r["periods"]}
    assert (by["last_week"]["from"], by["last_week"]["to"]) == ("20261005", "20261011")
    hon = lambda k: next(m for m in by[k]["marks"] if m["key"] == "honmei")      # noqa: E731
    assert hon("last_week")["text"] == "4-2-1-1"                      # 10/10 だけ
    assert (by["recent_5w"]["from"], by["recent_5w"]["to"]) == ("20260909", "20261013")
    assert hon("recent_5w")["text"] == "8-3-3-4"                      # 9/19 + 10/4 + 10/10 = (1+3+4, 1+0+2, 1+1+1, 1+2+1)
    assert r["year"] == 2026 and hon("year")["text"] == "11-4-3-4"    # 年は全日（1+2+1+3+4, ...）
    sv = next(m for m in by["recent_5w"]["marks"] if m["key"] == "show_val")
    assert sv["symbol"] == "▲" and sv["text"] == "0-1-0-3" and sv["top3_rate"] == 0.25
    assert [m["symbol"] for m in by["year"]["marks"]] == ["◎", "○", "▲", "★", "消"]


def test_year_switches_to_last_year_when_less_than_two_months(tmp_path):
    days = ["20250105", "20251228", "20260104", "20260125"]
    assert pr.pick_card_year(days, date(2026, 1, 28)) == 2025        # 今年は 3 週間分だけ → 去年
    assert pr.pick_card_year(days + ["20260310"], date(2026, 3, 11)) == 2026     # 約 2 か月以上 → 今年
    assert pr.pick_card_year(["20260104", "20260125"], date(2026, 1, 28)) == 2026     # 去年のデータが無ければ今年


def test_backfill_adds_marks_without_touching_bets(monkeypatch):
    from src.scripts.data import backfill_prediction_marks as bf

    doc = {"summary": {"marks": {"honmei": {"n": 1, "f1": 1, "f2": 0, "f3": 0, "f4p": 0}}, "bets": {"invested": 100}}, "races": [
        {"race_id": "R1", "included": True, "marks": {"honmei": [{"finish": 1}], "pair": [], "star": []}, "bets": {"total_bet": 100}}]}
    preds = {"predictions": [
        {"horse_number": 1, "win_prob": 0.4, "mark_type": "honmei"}, {"horse_number": 2, "win_prob": 0.3, "mark_type": "show_val"},
        {"horse_number": 3, "win_prob": 0.1, "mark_type": "kesi"}, {"horse_number": 4, "win_prob": 0.1, "mark_type": "kesi"}]}
    monkeypatch.setattr(pr, "load_predictions", lambda st, rid: preds)
    monkeypatch.setattr(pr, "load_race_result", lambda st, rid: {"finish": {1: 1, 2: 3, 3: 2, 4: 8}, "popularity": {1: 2, 2: 1, 3: 6, 4: 9}, "odds": {1: 3.4, 2: 2.1, 3: 15.0, 4: 40.0}})
    assert bf.backfill_day(doc, None) == 1
    m = doc["summary"]["marks"]
    assert (m["show_val"]["n"], m["show_val"]["f3"], m["show_val"]["pop_sum"], m["show_val"]["odds_sum"]) == (1, 1, 1, 2.1)       # 2番の馬: 1人気・2.1倍
    assert (m["kesi"]["n"], m["kesi"]["f2"], m["kesi"]["f4p"], m["kesi"]["pop_sum"]) == (2, 1, 1, 15)                              # 6人気+9人気
    assert m["honmei"]["f1"] == 1 and m["honmei"]["pop_sum"] == 0 and m["honmei"]["pop_n"] == 0       # 既存の馬に人気が無いまま（finish だけの旧記録）は人気を足せない
    assert doc["races"][0]["marks"]["honmei"][0]["final_odds"] is None
    assert doc["summary"]["bets"] == {"invested": 100} and doc["races"][0]["bets"] == {"total_bet": 100}       # 仮想購入は触らない
    assert bf.backfill_day(doc, None) == 0                           # 2 回目は何も足さない（冪等）


def test_average_popularity_and_odds_in_card(tmp_path):
    doc = _doc("20261010", {})
    doc["summary"]["marks"]["honmei"] = {"n": 3, "f1": 1, "f2": 1, "f3": 0, "f4p": 1, "pop_sum": 6, "pop_n": 3, "odds_sum": 12.0, "odds_n": 3}
    doc["summary"]["marks"]["pair"] = {"n": 2, "f1": 0, "f2": 1, "f3": 0, "f4p": 1, "pop_sum": 0, "pop_n": 0, "odds_sum": 0.0, "odds_n": 0}    # 人気が取れていない
    (tmp_path / "20261010.json").write_text(json.dumps(doc), encoding="utf-8")
    r = pr.mark_card(date(2026, 10, 13), tmp_path)
    marks = {m["key"]: m for m in r["periods"][0]["marks"]}
    assert marks["honmei"]["avg_popularity"] == 2.0 and marks["honmei"]["avg_odds"] == 4.0
    assert marks["pair"]["avg_popularity"] is None and marks["pair"]["avg_odds"] is None
    merged = pr.merge_summaries([doc["summary"], doc["summary"]])
    assert merged["marks"]["honmei"]["pop_sum"] == 12 and merged["marks"]["honmei"]["odds_n"] == 6


def test_win_only_average_popularity_and_odds():
    c = pr._blank_mark()
    for fin, pop, odds in ((1, 2, 4.0), (1, 1, 2.0), (3, 5, 12.0), (8, 9, 40.0)):
        pr.add_pick_to_mark(c, {"finish": fin, "popularity": pop, "final_odds": odds})
    assert (c["n"], c["pop_n"], c["pop_sum"]) == (4, 4, 17)                      # 全体: 平均 4.25 人気
    assert (c["w_pop_n"], c["w_pop_sum"], c["w_odds_n"], c["w_odds_sum"]) == (2, 3, 2, 6.0)      # 1着だけ: 平均 1.5 人気・3.0 倍
    doc = _doc("20261010", {})
    doc["summary"]["marks"]["honmei"] = {**c, "f1": 2, "f2": 0, "f3": 1, "f4p": 1}
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()); (d / "20261010.json").write_text(json.dumps(doc), encoding="utf-8")
    m = {x["key"]: x for x in pr.mark_card(date(2026, 10, 13), d)["periods"][0]["marks"]}["honmei"]
    assert m["win_avg_popularity"] == 1.5 and m["win_avg_odds"] == 3.0 and m["avg_popularity"] == 4.25
    assert {x["key"]: x for x in pr.mark_card(date(2026, 10, 13), d)["periods"][0]["marks"]}["pair"]["win_avg_popularity"] is None      # 1着の馬がいなければ None


def test_backfill_recounts_win_averages_without_reading_results(monkeypatch):
    from src.scripts.data import backfill_prediction_marks as bf

    pick = lambda fin, pop, od: {"finish": fin, "popularity": pop, "final_odds": od}          # noqa: E731
    doc = {"summary": {"marks": {m: {"n": 0, "f1": 0, "f2": 0, "f3": 0, "f4p": 0} for m in pr.MARKS}}, "races": [
        {"race_id": "R1", "included": True, "marks": {m: ([pick(1, 3, 6.0)] if m == "honmei" else []) for m in pr.MARKS}}]}
    monkeypatch.setattr(pr, "load_predictions", lambda *a: (_ for _ in ()).throw(AssertionError("予測は読まない")))
    assert bf.backfill_day(doc, None) == 1
    h = doc["summary"]["marks"]["honmei"]
    assert (h["f1"], h["w_pop_sum"], h["w_odds_sum"]) == (1, 3, 6.0)
    assert bf.backfill_day(doc, None) == 0                                                    # 2 回目は何もしない
