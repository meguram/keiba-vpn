import pytest

from src.pipeline.inference import prediction_results as pr


class FakeStorage:
    def __init__(self, docs):
        self.docs = docs

    def load(self, cat, rid):
        return self.docs.get((cat, rid))


def _pred_entries():
    # 6 頭立て。馬番1が本命、2が対抗、5が穴（★）
    rows = [(1, 0.40, 0.62, 0.80, "honmei", 2.5), (2, 0.25, 0.45, 0.65, "pair", 4.0), (3, 0.15, 0.30, 0.50, "none", 8.0),
            (4, 0.10, 0.20, 0.35, "none", 12.0), (5, 0.06, 0.15, 0.30, "star", 18.0), (6, 0.04, 0.08, 0.15, "none", 30.0)]
    return [{"horse_number": hn, "horse_name": f"馬{hn}", "win_prob": w, "top2_prob": t2, "top3_prob": t3, "mark_type": m,
             "win_odds": o, "ev_win": round(w * o, 3), "pred_score": w} for hn, w, t2, t3, m, o in rows]


def _odds_doc():
    return {"phase": "final", "entries": [
        {"horse_number": hn, "win_odds": o, "place_odds_min": max(1.1, o / 3), "place_odds_max": max(1.2, o / 2)}
        for hn, o in [(1, 2.5), (2, 4.0), (3, 8.0), (4, 12.0), (5, 18.0), (6, 30.0)]]}


def _result_entries(order):
    return [{"horse_number": hn, "finish_position": pos} for pos, hn in enumerate(order, 1)]


PAYOFF = {"単勝": {"numbers": "5", "payout": "1,800", "popularity": "5"},
          "複勝": [{"numbers": "5", "payout": "480", "popularity": "5"}, {"numbers": "1", "payout": "130", "popularity": "1"},
                  {"numbers": "2", "payout": "160", "popularity": "2"}],
          "馬連": {"numbers": "1-5", "payout": "5,400", "popularity": "8"},
          "ワイド": [{"numbers": "1-5", "payout": "1,200", "popularity": "7"}, {"numbers": "2-5", "payout": "1,500", "popularity": "9"},
                    {"numbers": "1-2", "payout": "300", "popularity": "1"}],
          "馬単": {"numbers": "5-1", "payout": "11,000", "popularity": "20"},
          "三連複": {"numbers": "1-2-5", "payout": "9,000", "popularity": "12"}}


def _storage(result_cat="race_result", with_payoff=True, order=(5, 1, 2, 3, 4, 6)):
    doc = {"entries": _result_entries(order)}
    if with_payoff:
        doc["payoff"] = PAYOFF
    rid = "202610180101"
    return FakeStorage({
        ("race_lists", "20261018"): {"races": [{"race_id": rid, "venue": "東京", "round": 1, "race_name": "テスト"}]},
        ("race_predictions", rid): {"predictions": _pred_entries(), "prediction_source": "gcs", "model_version": "v4"},
        ("race_odds", rid): _odds_doc(), (result_cat, rid): doc,
    })


def test_parse_payoff_handles_commas_lists_and_order():
    p = pr.parse_payoff(PAYOFF)
    assert p["tansho"] == {(5,): 1800}
    assert p["fukusho"][(1,)] == 130 and len(p["fukusho"]) == 3
    assert p["umaren"] == {(1, 5): 5400}                    # 組合せは昇順に正規化
    assert p["umatan"] == {(5, 1): 11000}                   # 馬単は着順どおり
    assert p["wide"][(2, 5)] == 1500 and p["sanrenpuku"] == {(1, 2, 5): 9000}
    assert pr.parse_payoff(None) == {} and pr.parse_payoff({"枠連": {"numbers": "1-2", "payout": "100"}}) == {}


def test_settle_uses_official_payoff_and_falls_back_to_odds():
    res = {"finish": {5: 1, 1: 2, 2: 3, 3: 4, 4: 5, 6: 6}, "n_runners": 6, "payoff": pr.parse_payoff(PAYOFF)}
    assert pr.settle_candidate("tansho", (5,), 200, 18.0, res) == (True, 3600.0, False)
    assert pr.settle_candidate("tansho", (1,), 200, 2.5, res)[0] is False
    assert pr.settle_candidate("fukusho", (1,), 300, 1.6, res) == (True, 390.0, False)   # 6 頭立ての複勝は 2 着まで
    assert pr.settle_candidate("fukusho", (2,), 100, 2.0, res)[0] is False               # 3 着の馬は払い戻されない
    assert pr.settle_candidate("umaren", (1, 5), 100, 50.0, res) == (True, 5400.0, False)
    assert pr.settle_candidate("wide", (2, 5), 100, 14.0, res) == (True, 1500.0, False)
    assert pr.settle_candidate("sanrenpuku", (2, 1, 5), 100, 90.0, res)[0] is True
    est = dict(res, payoff={})                                                         # 暫定: 払戻なし → オッズからの概算
    assert pr.settle_candidate("tansho", (5,), 200, 18.0, est) == (True, 3600.0, True)


def test_fukusho_pays_top2_when_fewer_than_8_runners_but_top3_is_hit_on_8():
    fm8 = {i: i for i in range(1, 9)}
    r8 = {"finish": fm8, "n_runners": 8, "payoff": {}}
    assert pr.settle_candidate("fukusho", (3,), 100, 2.0, r8)[0] is True
    r7 = {"finish": {i: i for i in range(1, 8)}, "n_runners": 7, "payoff": {}}
    assert pr.settle_candidate("fukusho", (3,), 100, 2.0, r7)[0] is False


def test_build_day_final_counts_mark_finish_distribution_and_roi():
    doc = pr.build_day("20261018", _storage())
    s = doc["summary"]
    assert doc["status"] == "final" and s["races_settled"] == 1
    assert {k: s["marks"]["honmei"][k] for k in ("n", "f1", "f2", "f3", "f4p")} == {"n": 1, "f1": 0, "f2": 1, "f3": 0, "f4p": 0}          # ◎(1番)は 2 着
    assert s["marks"]["star"]["f1"] == 1 and s["marks"]["pair"]["f3"] == 1               # ★(5番)は 1 着、○(2番)は 3 着
    assert s["top1"]["f2"] == 1
    b = s["bets"]
    assert b["invested"] > 0 and b["invested"] <= 2000 and b["returned"] >= 0
    assert s["logloss"]["n"] == 1 and doc["races"][0]["result_top3"][0] == {"horse_number": 5, "finish": 1}


def test_provisional_when_only_quick_result_and_no_payoff():
    doc = pr.build_day("20261018", _storage("race_result_on_time", with_payoff=False))
    assert doc["status"] == "provisional"
    assert doc["races"][0]["result_status"] == "provisional" and doc["races"][0]["payoff_official"] is False
    assert doc["summary"]["marks"]["honmei"]["f2"] == 1


def test_missing_prediction_or_result_is_excluded_and_counted():
    st = _storage()
    st.docs[("race_lists", "20261018")]["races"].append({"race_id": "202610180102", "venue": "東京", "round": 2})
    st.docs[("race_lists", "20261018")]["races"].append({"race_id": "202610180103", "venue": "東京", "round": 3})
    st.docs[("race_predictions", "202610180102")] = {"predictions": _pred_entries()}          # 予測だけ（結果なし）
    st.docs[("race_result", "202610180103")] = {"entries": _result_entries([1, 2, 3]), "payoff": PAYOFF}  # 結果だけ（予測なし）
    doc = pr.build_day("20261018", st)
    s = doc["summary"]
    assert s["races"] == 3 and s["races_settled"] == 1 and s["races_no_result"] == 1 and s["races_no_prediction"] == 1
    assert doc["status"] == "provisional"            # 結果の揃っていないレースがあれば確定とは言わない


def test_idempotent_and_final_is_not_overwritten_by_provisional(tmp_path):
    st = _storage()
    a = pr.build_day("20261018", st)
    b = pr.build_day("20261018", st)
    assert a["summary"] == b["summary"] and a["races"] == b["races"]
    assert pr.write_day(a, tmp_path) is not None and pr.read_day("20261018", tmp_path)["status"] == "final"
    prov = pr.build_day("20261018", _storage("race_result_on_time", with_payoff=False))
    assert pr.write_day(prov, tmp_path) is None                                      # 確定済みを暫定で上書きしない
    assert pr.read_day("20261018", tmp_path)["status"] == "final"
    assert pr.write_day(prov, tmp_path, force=True) is not None


def test_merge_period_summary_sums_counts_and_series(tmp_path):
    for d in ("20261017", "20261018"):
        doc = pr.build_day("20261018", _storage())
        doc["date"] = d
        pr.write_day(doc, tmp_path)
    ps = pr.period_summary("week", "20261018", tmp_path)
    assert ps["days"] == ["20261017", "20261018"] and ps["summary"]["marks"]["honmei"]["n"] == 2
    assert ps["summary"]["marks"]["honmei"]["f2"] == 2 and len(ps["series"]) == 2 and ps["status"] == "final"
    assert pr.period_range("week", "20261018") == ("20261012", "20261018")           # 月曜始まり（土日の開催が同じ週）
    assert pr.period_range("month", "20261018") == ("20261001", "20261031")
    assert pr.period_range("season", "20261018") == ("20260101", "20261231")
    assert pr.rate(1, 4) == 0.25 and pr.rate(1, 0) is None


def test_highlights_pick_high_odds_hit_and_big_miss():
    recs = [{"included": True, "race_id": "r1", "venue": "京都", "round": 11,
             "marks": {"honmei": [{"horse_name": "A", "win_prob": 0.35, "finish": 7, "win_odds": 2.0}],
                       "pair": [], "star": [{"horse_name": "B", "win_prob": 0.05, "finish": 2, "win_odds": 22.0}]},
             "bets": {"total_bet": 1000, "total_return": 6000}}]
    kinds = {h["kind"] for h in pr.highlights(recs)}
    assert kinds == {"ana_hit", "big_return", "big_miss"}


@pytest.mark.parametrize("raw,expected", [("1,940", 1940), ("3", 3), (None, None), ("x", None)])
def test_to_int(raw, expected):
    assert pr._to_int(raw) == expected


def test_non_positive_finish_positions_are_ignored():
    assert pr._finish_map([{"horse_number": 1, "finish_position": 1}, {"horse_number": 2, "finish_position": -1},
                           {"horse_number": 3, "finish_position": "中止"}, {"horse_number": 4, "finish_position": 0}]) == {1: 1}


def test_old_saved_marks_with_many_pairs_and_old_anchor_still_aggregate():
    """旧ルールで保存された予測（○・✓・▲が複数）でも、予測成績の集計が壊れない。保存済みの印は書き換えない。"""
    st = _storage()
    preds = st.docs[("race_predictions", "202610180101")]["predictions"]
    for e, m in zip(preds, ["honmei", "pair", "pair", "anchor", "show_val", "show_val"]):
        e["mark_type"] = m
    rec = pr.build_race_record("202610180101", {"venue": "東京", "round": 1}, st)
    assert rec["included"]
    assert [x["horse_number"] for x in rec["marks"]["pair"]] == [2, 3]
    assert [e["mark_type"] for e in preds] == ["honmei", "pair", "pair", "anchor", "show_val", "show_val"]
    assert pr.MARK_SYMBOLS["anchor"] == "△"


def _rec(level, f1_finish, bet=100, ret=0, status="ok"):
    pick = {"horse_number": 1, "horse_name": "a", "finish": f1_finish, "win_prob": 0.3}
    bt = {"status": status, "strategy": "target", "level": level, "total_bet": bet if level else 0, "total_return": ret if level else 0,
          "by_type": {"sanrenpuku": {"invested": bet, "returned": ret, "n": 1, "hit": int(ret > 0)}} if level else {}, "estimated": False}
    rec = {"race_id": f"r{level}{f1_finish}", "included": True, "venue": "東京", "round": 1, "top1": pick, "marks": {"honmei": [pick]}, "winner_logloss": 1.0,
           "bets": {"total_bet": 999, "total_return": 0, "by_type": {}, "level": 3}}      # 従来の最適化の仮想購入（集計には使わない）
    if level is not None:
        rec["bets_t"] = bt
    return rec


def test_rated_records_use_target_level_and_target_bets_only():
    recs = [_rec(0, 1), _rec(1, 1, ret=200), _rec(3, 5), _rec(None, 1), _rec(2, 2, status="error"), {"race_id": "x", "included": False}]
    rated = pr.rated_records(recs)
    assert [r["race_id"] for r in rated] == ["r11", "r35"]
    assert rated[0]["bets"]["total_bet"] == 100 and rated[0]["bets"]["total_return"] == 200       # 従来の bets（999円）ではなく target の購入
    assert pr.rated_counts(recs) == {"settled": 5, "rated": 2, "skipped": 1, "unrated": 2}          # 計算できなかった(error)・未記録は unrated
    s = pr.summarize(rated)
    assert s["marks"]["honmei"]["n"] == 2 and s["races_settled"] == 2 and s["bets"]["invested"] == 200 and s["bets"]["returned"] == 200


def test_level_summary_counts_skipped_and_unrated_separately():
    recs = [_rec(0, 1), _rec(1, 1, ret=300), _rec(1, 4), _rec(None, 1)]
    by, unrated = pr.level_summary(recs)
    assert unrated == 1 and by["0"]["races"] == 1 and by["0"]["invested"] == 0
    assert by["1"] == {"races": 2, "hit": 1, "invested": 200, "returned": 300}


def test_period_summary_has_rated_summary_series_and_counts(tmp_path):
    recs = [_rec(0, 1), _rec(2, 1, bet=200, ret=300), _rec(1, 4)]
    pr.write_day({"date": "20261018", "status": "final", "summary": pr.summarize(recs), "highlights": [], "races": recs, "notes": []}, tmp_path)
    p = pr.period_summary("day", "20261018", tmp_path)
    assert p["summary"]["marks"]["honmei"]["n"] == 3 and p["rated_summary"]["marks"]["honmei"]["n"] == 2
    assert p["rated_counts"] == {"settled": 3, "rated": 2, "skipped": 1, "unrated": 0}
    pt = p["series"][0]
    assert pt["honmei_win"] == pytest.approx(2 / 3) and pt["honmei_win_rated"] == pytest.approx(1 / 2) and pt["roi_rated"] == pytest.approx(300 / 300)


def test_simulate_target_bets_settles_displayed_plan_and_skip_has_no_purchase(monkeypatch):
    from src.api.v1 import delegates as D

    result = {"finish": {1: 1, 2: 2, 3: 3, 4: 4}, "n_runners": 4, "payoff": {"sanrenpuku": {(1, 2, 3): 5000}}}
    bets = [{"bet_type": "sanrenpuku", "combo": [1, 2, 3], "yen": 200, "est_multiple": 40.0}, {"bet_type": "sanrenpuku", "combo": [1, 2, 4], "yen": 100, "est_multiple": 30.0}]
    monkeypatch.setattr(D, "_optimize_target", lambda body, view, storage=None: ({"status": "bet", "level": 3, "bets": bets}, 200))
    out = pr.simulate_target_bets(object(), "r", result)
    assert out["status"] == "ok" and out["level"] == 3 and out["total_bet"] == 300 and out["total_return"] == 10000 and out["by_type"]["sanrenpuku"]["hit"] == 1
    monkeypatch.setattr(D, "_optimize_target", lambda body, view, storage=None: ({"status": "skip", "level": 0, "bets": []}, 200))
    skip = pr.simulate_target_bets(object(), "r", result)
    assert skip["level"] == 0 and skip["total_bet"] == 0 and skip["total_return"] == 0
    monkeypatch.setattr(D, "_optimize_target", lambda body, view, storage=None: ({"error": "予測結果がありません"}, 404))
    assert pr.simulate_target_bets(object(), "r", result)["status"] == "error"


def test_add_target_bets_fills_only_missing_races_and_keeps_existing(tmp_path, monkeypatch):
    recs = [_rec(None, 1), _rec(2, 1)]
    recs[0]["race_id"], recs[1]["race_id"] = "a", "b"
    pr.write_day({"date": "20261018", "status": "final", "summary": pr.summarize(recs), "highlights": [], "races": recs, "notes": []}, tmp_path)
    monkeypatch.setattr(pr, "load_race_result", lambda storage, rid: {"finish": {1: 1}})
    monkeypatch.setattr(pr, "simulate_target_bets", lambda storage, rid, res, cap: {"status": "ok", "level": 1, "total_bet": 100, "total_return": 0, "by_type": {}, "estimated": False})
    out = pr.add_target_bets("20261018", object(), tmp_path)
    assert out == {"races": 2, "added": 1, "already": 1, "failed": 0}
    doc = pr.read_day("20261018", tmp_path)
    assert doc["races"][0]["bets_t"]["level"] == 1 and doc["races"][1]["bets_t"]["level"] == 2


def test_cli_skip_final_does_not_reaggregate_final_days(monkeypatch, capsys):
    from src.scraper import storage as st
    from src.scripts.data import prediction_results as cli

    docs = {"20261017": {"status": "final"}, "20261018": {"status": "provisional"}}
    called: list[str] = []
    monkeypatch.setattr(st, "HybridStorage", lambda *a, **k: object())
    monkeypatch.setattr(cli.pr, "read_day", lambda d, base=None: docs.get(d))
    monkeypatch.setattr(cli.pr, "aggregate_day",
                        lambda d, storage, **k: called.append(d) or {"saved": True, "status": "final", "summary": {"races_settled": 0}})
    assert cli.main(["--from", "20261017", "--to", "20261018", "--skip-final"]) == 0
    assert called == ["20261018"] and "確定済みのため再集計しない" in capsys.readouterr().out
