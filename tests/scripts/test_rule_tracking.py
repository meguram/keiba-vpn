"""rule_tracking: 払戻の読み取り・買い目・決まりの判定・事前登録の検査・集計。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import rule_tracking as rt

PAY = json.dumps({"単勝": [{"numbers": "3", "payout": "250", "popularity": "1"}],
                  "複勝": [{"numbers": "3", "payout": "120", "popularity": "1"}, {"numbers": "1", "payout": "150", "popularity": "2"},
                          {"numbers": "2", "payout": "300", "popularity": "5"}],
                  "馬連": [{"numbers": "1-3", "payout": "900", "popularity": "2"}],
                  "ワイド": [{"numbers": "1-3", "payout": "300", "popularity": "1"}, {"numbers": "2-3", "payout": "400", "popularity": "3"},
                           {"numbers": "1-2", "payout": "1,200", "popularity": "9"}]}, ensure_ascii=False)


def _cfg():
    return rt.load_cfg()


def test_parse_payoffs_and_returns():
    pay = rt.parse_payoffs(PAY)
    assert pay["wide"][(1, 2)] == 1200.0 and pay["fukusho"][1] == 150.0
    r = rt.bet_returns([3, 1, 2], pay)
    assert r == {"tansho": 250.0, "fukusho": 120.0, "umaren": 900.0, "wide": 700.0}     # ワイドは 2 点とも当たり
    assert rt.bet_returns([4, 1, 2], pay) == {"tansho": 0.0, "fukusho": 0.0, "umaren": 0.0, "wide": 0.0}
    assert rt.parse_payoffs(json.dumps({"単勝": []})) is None                               # 4 券種がそろわなければ読めない扱い
    bad = json.loads(PAY)
    bad["ワイド"] = {"numbers": "13231", "payout": "300400"}                                 # 連結された行
    assert rt.parse_payoffs(json.dumps(bad, ensure_ascii=False)) is None
    bad["ワイド"] = [{"numbers": "1-2", "payout": "300"}]                                     # 人気の無い行でも、一覧の形なら普通に読む（古い保存の形）
    assert rt.parse_payoffs(json.dumps(bad, ensure_ascii=False))["wide"] == {(1, 2): 300.0}
    bad["複勝"] = {"numbers": "12", "payout": "120150"}                                         # 複勝が 1 つの辞書（人気なし）= 連結
    assert rt.parse_payoffs(json.dumps(bad, ensure_ascii=False)) is None
    dh = json.loads(PAY)
    dh["単勝"] = [{"numbers": "3", "payout": "150", "popularity": "1"}, {"numbers": "3", "payout": "160", "popularity": "1"}]
    assert rt.parse_payoffs(json.dumps(dh, ensure_ascii=False))["tansho"][3] == 310.0       # 同じ組の複数行は足す


def test_confidence():
    c = rt.confidence(np.array([0.5, 0.3, 0.2]))
    assert c["p1"] == pytest.approx(0.5) and c["gap"] == pytest.approx(0.2) and 0 < c["flat"] < 1
    assert rt.confidence(np.array([1.0, 1.0, 1.0, 1.0]))["flat"] == pytest.approx(1.0)


def _runners():
    return pd.DataFrame({"horse_id": ["a", "b", "c", "d", "e"], "horse_number": [1, 2, 3, 4, 5],
                         "odds": [3.0, 5.0, 1.8, 20.0, np.nan], "popularity": [2, 3, 1, 4, np.nan], "going": "良"})


def test_race_record_drops_scratched_and_market():
    pred = pd.DataFrame({"horse_id": ["a", "b", "c", "d", "e"], "p_win": [0.2, 0.1, 0.3, 0.1, 0.3]})   # e は取消（オッズ無し）
    rec, why = rt.race_record("r1", "2026-11-15", pred, _runners(), rt.parse_payoffs(PAY), None)
    assert why == "" and rec["top"] == [3, 1, 2] and rec["field"] == 4
    assert rec["p1"] == pytest.approx(0.3 / 0.7)                                          # 取消を外して正規化し直す
    assert rec["fav_final"] == {"pop": 1, "odds": 1.8}
    assert rec["mkt_final"]["top"] == [3, 1, 2] and rec["fav_pre"] is None


def test_race_record_skips_when_runner_missing_from_pred():
    pred = pd.DataFrame({"horse_id": ["a", "b", "c"], "p_win": [0.4, 0.3, 0.3]})               # d は走ったのに予測が無い
    rec, why = rt.race_record("r1", "2026-11-15", pred, _runners(), rt.parse_payoffs(PAY), None)
    assert rec is None and why == "予測に無い出走馬がいる"


def test_race_record_reranks_pre_popularity_without_scratched():
    pred = pd.DataFrame({"horse_id": ["a", "b", "c", "d"], "p_win": [0.4, 0.1, 0.3, 0.2]})
    pre = pd.DataFrame({"race_id": "r1", "phase": "t02", "horse_number": [1, 2, 3, 4, 5], "win_odds": [2.5, 6.0, 3.0, 15.0, 1.5],
                        "popularity": [2, 4, 3, 5, 1]})                                       # 5 番（取消）が発走前は 1 番人気
    rec, _ = rt.race_record("r1", "2026-11-15", pred, _runners(), rt.parse_payoffs(PAY), pre)
    assert rec["fav_pre"] == {"pop": 1, "odds": 2.5} and rec["mkt_pre"]["top"] == [1, 3, 2]   # 走った馬だけで数え直す


def test_race_record_uses_pre_snapshot():
    pred = pd.DataFrame({"horse_id": ["a", "b", "c", "d"], "p_win": [0.4, 0.1, 0.3, 0.2]})
    pre = pd.DataFrame({"race_id": "r1", "phase": "t02", "horse_number": [1, 2, 3, 4], "win_odds": [2.5, 6.0, 2.2, 15.0], "popularity": [2, 3, 1, 4]})
    rec, _ = rt.race_record("r1", "2026-11-15", pred, _runners(), rt.parse_payoffs(PAY), pre)
    assert rec["fav_pre"] == {"pop": 2, "odds": 2.5} and rec["pre_phase"] == "t02" and rec["mkt_pre"]["top"] == [3, 1, 2]


def test_pick_pre_respects_post_time():
    from datetime import datetime

    pre = pd.DataFrame({"race_id": "r1", "phase": ["t02", "t60"], "snapshot_at": ["2026-11-15T10:01:00+09:00", "2026-11-15T09:00:00+09:00"],
                        "horse_number": [1, 1], "win_odds": [2.0, 3.0], "popularity": [1, 1]})
    post = datetime.fromisoformat("2026-11-15T10:00:00+09:00")
    assert rt.pick_pre(pre, "r1", post)["phase"].iloc[0] == "t60"                             # 発走後のスナップショットは使わない
    assert rt.pick_pre(pre, "r2", post) is None


def test_decide_rules():
    fr = _cfg()["frozen"]
    by = {r["id"]: r for r in fr["rules"]}
    rec = {"p1": 0.5, "gap": 0.2, "flat": 0.6, "field": 16, "going": "不良", "fav_final": {"pop": 1, "odds": 1.5}, "fav_pre": None}
    c = rt.conditions(rec, fr, "final")
    assert rt.decide(by["T00"], c) is True and rt.decide(by["T01"], c) is False and rt.decide(by["T04"], c) is False
    assert rt.decide(by["F01"], c) is True and rt.decide(by["F02"], c) is True
    assert rt.decide(by["U01"], c) is True and rt.decide(by["U02"], c) is False and rt.decide(by["U04"], c) is True
    cp = rt.conditions(rec, fr, "pre")                                                       # 発走前のオッズが無い → 単勝の決まりは判定できない
    assert rt.decide(by["T01"], cp) is None and rt.decide(by["F01"], cp) is True


def test_registration_rejects_changed_rule_and_allows_added_with_since(tmp_path):
    cfg = _cfg()
    assert rt.check_registration(cfg, tmp_path) == (True, [])
    changed = json.loads(json.dumps(cfg))
    changed["frozen"]["rules"][1]["skip_if"] = ["odds_lt"]
    ok, bad = rt.check_registration(changed, tmp_path, write=False)
    assert not ok and "T01" in bad
    thr = json.loads(json.dumps(cfg))
    thr["frozen"]["p1_low"] = 0.3
    assert rt.check_registration(thr, tmp_path, write=False)[1] == ["thresholds"]
    added = json.loads(json.dumps(cfg))
    added["frozen"]["rules"].append({"id": "T99", "bet": "tansho", "label": "x", "buy_if": [], "skip_if": ["fav"]})
    assert not rt.check_registration(added, tmp_path, write=False)[0]                      # since の無い追加は拒否
    added["frozen"]["rules"][-1]["since"] = "20261201"
    assert not rt.check_registration(added, tmp_path, write=False)[0]                      # 区切りの無い日付は拒否
    added["frozen"]["rules"][-1]["since"] = "2026-11-20"
    assert not rt.check_registration(added, tmp_path, write=False, today="2026-11-22")[0]   # 検査した当日より前（さかのぼる）は拒否
    added["frozen"]["rules"][-1]["since"] = "2026-12-01"
    assert rt.check_registration(added, tmp_path, today="2026-11-22") == (True, [])
    assert rt.check_registration(added, tmp_path, write=False, today="2026-12-10") == (True, [])   # 記録した後は、日が進んでも一致
    added["frozen"]["rules"][-1]["skip_if"] = ["odds_lt"]
    assert "T99" in rt.check_registration(added, tmp_path, write=False)[1]                   # 記録した追加の決まりも、変えたら不一致


def test_summarize():
    s = rt.summarize(np.array([0.0, 300.0, 0.0]), np.array([100.0, 0.0, 0.0]), np.array(["d1", "d1", "d2"]), 100.0, n_boot=200)
    assert s["n"] == 3 and s["roi"] == pytest.approx(1.0) and s["mkt_roi"] == pytest.approx(1 / 3)
    assert s["hit_rate"] == pytest.approx(1 / 3) and s["days"] == 2
    assert s["max_drawdown_yen"] == pytest.approx(100.0)                                   # d1 +100、d2 −100
    assert rt.summarize(np.array([]), None, np.array([]), 100.0) == {"n": 0}


def test_era_of():
    cfg = {"eras": {"v6": [{"from": "2026-10-25", "bundle": "A"}, {"from": "2026-12-01", "bundle": "B"}]}}
    assert rt.era_of(cfg, "v6", "2026-10-24") == "" and rt.era_of(cfg, "v6", "2026-11-15") == "A" and rt.era_of(cfg, "v6", "2026-12-02") == "B"


def test_scorecard_excludes_by_date_and_separates_inactive(monkeypatch, tmp_path):
    cfgft = tmp_path / "ft.json"
    cfgft.write_text(json.dumps({"excluded_dates": {"2026-11-16": "テスト"}}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(rt.ft, "CFG_FT", cfgft)
    cfg = rt.load_cfg()
    base = {"model": "v6", "source": "forward", "field": 12, "going": "良", "p1": 0.3, "gap": 0.1, "flat": 0.7,
            "ret": {"tansho": 0.0, "fukusho": 0.0, "umaren": 0.0, "wide": 0.0}, "mkt_final": {"top": [1, 2, 3], "ret": {"tansho": 0.0, "fukusho": 0.0, "umaren": 0.0, "wide": 0.0}},
            "mkt_pre": None}
    rows = [{**base, "race_id": "r1", "date": "2026-11-15", "buy": {"final": {"T00": True, "T01": None}, "pre": {}}},
            {**base, "race_id": "r2", "date": "2026-11-16", "buy": {"final": {"T00": True, "T01": True}, "pre": {}}},      # 除外日
            {**base, "race_id": "r3", "date": "2026-11-22", "buy": {"final": {"T00": True}, "pre": {}}}]                   # T01 はまだ対象外（キー無し）
    card = rt.scorecard(pd.DataFrame(rows), cfg)
    c = {(x["variant"], x["rule"]): x for x in card["cards"] if x["period"] == "forward"}
    assert card["excluded_rows"] == 1 and "2026-11-16" in card["excluded_dates"]
    assert c[("final", "T00")]["n"] == 2 and c[("final", "T00")]["races"] == 2
    assert c[("final", "T01")]["undecided"] == 1 and c[("final", "T01")]["races"] == 1 and c[("final", "T01")]["n"] == 0
    assert c[("final", "T00")]["era"].startswith("final_v6e")                               # 区分は集計のときに日付で付く


def test_run_track_waits_for_pre_odds_and_skips_done_days(tmp_path):
    tables = tmp_path / "tables"
    (tables / "2026").mkdir(parents=True)
    pd.DataFrame({"race_id": "202605010101", "date": "2026-11-15", "start_time": "10:00", "horse_number": [1, 2, 3, 4],
                  "horse_id": ["a", "b", "c", "d"], "odds": [3.0, 5.0, 1.8, 20.0], "popularity": [2, 3, 1, 4], "track_condition": "良",
                  "payoff": PAY}).to_parquet(tables / "2026" / "race_result_flat.parquet")
    pred_dir = tmp_path / "pred" / "20261115"
    pred_dir.mkdir(parents=True)
    pd.DataFrame({"race_id": "202605010101", "horse_id": ["a", "b", "c", "d"], "p_win": [0.2, 0.1, 0.5, 0.2], "p_top2": 0.3, "p_top3": 0.4}).to_parquet(pred_dir / "202605010101.parquet")
    cfg = rt.load_cfg()
    cfg["sources"] = [{"model": "v6", "kind": "forward", "dir": str(tmp_path / "pred")}]
    out = tmp_path / "out"
    s = rt.run_track(["20261115"], cfg, out, tables=tables, pre_path=tmp_path / "none.parquet", verbose=False, today="2026-11-20")
    assert s["added"] == 0 and s["skipped"] == {"発走前オッズ待ち": 1}                        # 14 日までは書かずに待つ
    s = rt.run_track(["20261115"], cfg, out, tables=tables, pre_path=tmp_path / "none.parquet", verbose=False, today="2026-12-10")
    assert s["added"] == 1                                                                   # 待ちを過ぎたら pre 無しで書く
    led = rt.ft.load_ledger(out / "ledger.jsonl")
    assert led["buy"].iloc[0]["final"]["T01"] is False and led["buy"].iloc[0]["pre"]["T01"] is None
    s = rt.run_track(["20261115"], cfg, out, tables=tables, pre_path=tmp_path / "none.parquet", verbose=False, today="2026-12-11")
    assert s == {"added": 0, "skipped": {}, "days": 1}                                       # 処理済みの日は読み直さない


def test_log_excluded_records_changes(tmp_path):
    assert rt.log_excluded(tmp_path, ["2026-11-07"])[-1]["added"] == []
    assert len(rt.log_excluded(tmp_path, ["2026-11-07"])) == 1                               # 同じなら書かない
    log = rt.log_excluded(tmp_path, ["2026-11-07", "2026-12-06"])
    assert log[-1]["added"] == ["2026-12-06"] and log[-1]["removed"] == []


def test_track_stops_with_code_3_and_reports_on_mismatch(tmp_path, monkeypatch):
    import argparse

    cfg = rt.load_cfg()
    cfg["frozen"]["p1_low"] = 0.3
    monkeypatch.setattr(rt, "load_cfg", lambda path=None: cfg)
    with pytest.raises(SystemExit) as e:
        rt.cmd_track(argparse.Namespace(out_dir=str(tmp_path), to="", days="20261115", **{"from": "20261115"}))
    assert e.value.code == 3
    card = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert card["registration_changed"]["changed"] and "停止中" in (tmp_path / "latest.md").read_text(encoding="utf-8")
